"""
Metrics for multilabel geological classification.

Key design decisions:
  - Per-class accuracy and F1 computed on GPU where possible
  - Only move to CPU/numpy for sklearn F1 (unavoidable)
  - Accumulate predictions on CPU during training to avoid GPU OOM
  - Evaluation function works both inline (end of training) and standalone
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    hamming_loss,
)


# ============================================================================
# GPU-Friendly Metrics
# ============================================================================

class MetricsAccumulator:
    """
    Accumulate predictions and labels across batches on CPU.
    
    Usage:
        acc = MetricsAccumulator()
        for batch in loader:
            logits = model(...)
            acc.update(logits, labels)
        metrics = acc.compute()
    """

    def __init__(self):
        self.all_logits: List[torch.Tensor] = []
        self.all_labels: List[torch.Tensor] = []
        self.running_loss = 0.0
        self.n_samples = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: Optional[float] = None, batch_size: int = 0):
        """
        Store batch predictions. Moves to CPU immediately to free GPU.
        """
        self.all_logits.append(logits.detach().cpu())
        self.all_labels.append(labels.detach().cpu())
        if loss is not None:
            self.running_loss += loss * (batch_size or logits.size(0))
            self.n_samples += batch_size or logits.size(0)

    def compute(self, class_names: Optional[List[str]] = None) -> Dict:
        """Compute all metrics from accumulated predictions."""
        logits = torch.cat(self.all_logits, dim=0)
        labels = torch.cat(self.all_labels, dim=0)

        # Predictions via sigmoid threshold
        preds = (torch.sigmoid(logits) > 0.5).float()

        # --- GPU-friendly metrics (stay in torch) ---
        correct = (preds == labels).float()
        per_class_acc = correct.mean(dim=0).numpy()
        avg_acc = correct.mean().item()
        exact_match = (preds == labels).all(dim=1).float().mean().item()

        # --- sklearn metrics (need numpy) ---
        preds_np = preds.numpy().astype(int)
        labels_np = labels.numpy().astype(int)

        per_class_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)
        macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
        micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
        per_class_precision = precision_score(labels_np, preds_np, average=None, zero_division=0)
        per_class_recall = recall_score(labels_np, preds_np, average=None, zero_division=0)

        # Hamming loss
        hamming = hamming_loss(labels_np, preds_np)

        # AUC-ROC (requires probabilities)
        probs_np = torch.sigmoid(logits).numpy()
        try:
            per_class_auc = roc_auc_score(labels_np, probs_np, average=None)
            macro_auc = roc_auc_score(labels_np, probs_np, average="macro")
        except ValueError:
            # Can fail if a class has no positive samples
            per_class_auc = np.zeros(labels_np.shape[1])
            macro_auc = 0.0

        avg_loss = self.running_loss / max(self.n_samples, 1)

        metrics = {
            "loss": avg_loss,
            "avg_accuracy": avg_acc,
            "exact_match_accuracy": exact_match,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hamming_loss": hamming,
            "macro_auc": macro_auc,
            "per_class_accuracy": per_class_acc,
            "per_class_f1": per_class_f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_auc": per_class_auc,
        }

        # Per-class confusion matrices
        if class_names:
            per_class_cm = {}
            for i, name in enumerate(class_names):
                cm = confusion_matrix(labels_np[:, i], preds_np[:, i], labels=[0, 1])
                per_class_cm[name] = cm
            metrics["per_class_confusion"] = per_class_cm

        return metrics

    def reset(self):
        self.all_logits.clear()
        self.all_labels.clear()
        self.running_loss = 0.0
        self.n_samples = 0


# ============================================================================
# Pretty Printing
# ============================================================================

def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict,
    val_metrics: Dict,
    lr: float,
    class_names: List[str],
):
    """Print formatted epoch summary."""
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{total_epochs} | LR: {lr:.2e}")
    print(f"{'='*60}")
    print(f"  Train — Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['avg_accuracy']:.4f} | Macro-F1: {train_metrics['macro_f1']:.4f}")
    print(f"  Val   — Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['avg_accuracy']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Val Exact Match: {val_metrics['exact_match_accuracy']:.4f}")

    print(f"\n  {'Class':<8} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*34}")
    for i, name in enumerate(class_names):
        print(
            f"  {name:<8} "
            f"{val_metrics['per_class_accuracy'][i]:>6.3f} "
            f"{val_metrics['per_class_f1'][i]:>6.3f} "
            f"{val_metrics['per_class_precision'][i]:>6.3f} "
            f"{val_metrics['per_class_recall'][i]:>6.3f}"
        )


# ============================================================================
# Evaluation (Test Set)
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    use_amp: bool = True,
    mode: str = "full",
) -> Dict:
    """
    Run full evaluation on a dataset split.
    
    Used both at end of training (test set) and as standalone via evaluate.py.
    
    Args:
        mode: "full" (spectral+topo dict) or "rgb" (single tensor).
    
    Returns comprehensive metrics dict.
    """
    from models import prepare_inputs, forward_batch

    model.eval()
    acc = MetricsAccumulator()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs_dev = prepare_inputs(inputs, device, mode)
        labels_dev = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = forward_batch(model, inputs_dev, mode)
            loss = criterion(logits, labels_dev)

        acc.update(logits, labels_dev, loss=loss.item(), batch_size=labels.size(0))

        if batch_idx % 20 == 0:
            print(f"  Eval batch {batch_idx}/{len(dataloader)}")

    metrics = acc.compute(class_names=class_names)

    # Print detailed report
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Avg Accuracy: {metrics['avg_accuracy']:.4f}")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")

    print(f"\n  {'Class':<8} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*34}")
    for i, name in enumerate(class_names):
        print(
            f"  {name:<8} "
            f"{metrics['per_class_accuracy'][i]:>6.3f} "
            f"{metrics['per_class_f1'][i]:>6.3f} "
            f"{metrics['per_class_precision'][i]:>6.3f} "
            f"{metrics['per_class_recall'][i]:>6.3f}"
        )

    # Print confusion matrices
    if "per_class_confusion" in metrics:
        print(f"\n  Per-Class Confusion Matrices (TN, FP, FN, TP):")
        for name, cm in metrics["per_class_confusion"].items():
            tn, fp, fn, tp = cm.ravel()
            print(f"  {name:<8} TN={tn:>5} FP={fp:>5} FN={fn:>5} TP={tp:>5}")

    return metrics

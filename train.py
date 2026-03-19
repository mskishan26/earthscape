"""
EarthScape — Training Entry Point

Usage:
    # Local (small run on RTX 4060)
    python train.py --experiment experiments/midfusion_all.yaml --data.max_sets 1 --training.num_epochs 3

    # SageMaker (full run)
    python train.py --experiment experiments/midfusion_all.yaml

    # RGB model experiment
    python train.py --experiment experiments/resnet50_rgb.yaml

    # Override anything via CLI
    python train.py --experiment experiments/midfusion_all.yaml --training.lr 3e-4
"""

import os
import gc
import glob
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from utils import load_config, set_seed, get_device, compute_data_version
from dataset import (
    CachedEarthScapeDataset,
    EarthscapePatchAdapter,
    SimpleCache,
    list_sets,
    compute_channel_stats,
    compute_pos_weights,
)
from torch.utils.data import DataLoader
from models import build_model, forward_batch, prepare_inputs, get_model_mode
from metrics import MetricsAccumulator, print_epoch_summary, evaluate


def setup_wandb(cfg: dict) -> bool:
    """Initialize W&B if enabled. Returns True if active."""
    if not cfg["logging"]["use_wandb"]:
        print("[W&B] Disabled")
        return False
    try:
        import wandb

        # Build run name from experiment and hyperparams
        exp_name = cfg.get("_experiment", {}).get("name", "")
        arch = cfg["model"].get("architecture", "midfusion")
        run_name = f"{exp_name or arch}-ep{cfg['training']['num_epochs']}-lr{cfg['training']['lr']}"

        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"].get("wandb_entity"),
            config=cfg,
            name=run_name,
            group=cfg["logging"].get("wandb_group"),  # Group related experiments
            tags=cfg["logging"].get("wandb_tags", []),  # Tags for filtering
            job_type="train",
            mode="online",
            reinit="finish_previous",  # Allow multiple runs in same script
        )
        print(f"[W&B] Initialized: {run_name} (group={cfg['logging'].get('wandb_group')})")
        return True
    except ImportError:
        print("[W&B] wandb not installed, skipping")
        return False
    except Exception as e:
        print(f"[W&B] Failed to initialize: {e}")
        return False


def cleanup_old_checkpoints(checkpoint_dir: str, keep_top_k: int = 3):
    """Keep only top K periodic checkpoints by modification time."""
    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth")),
        key=os.path.getmtime,
        reverse=True
    )
    for old_ckpt in ckpts[keep_top_k:]:
        os.remove(old_ckpt)
        print(f"  [Cleanup] Removed old checkpoint: {os.path.basename(old_ckpt)}")


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def log_embedding_visualization(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    mode: str = "full",
    n_samples: int = 1000,
):
    """
    Extract embeddings and log t-SNE/UMAP visualizations to W&B.
    Call at end of training for output analysis.
    """
    try:
        import wandb
        from sklearn.manifold import TSNE
    except ImportError:
        print("[Viz] sklearn or wandb not available, skipping embeddings")
        return

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("[Viz] umap not installed, using t-SNE only")

    print(f"[Viz] Extracting embeddings from {n_samples} samples...")
    model.eval()
    embeddings = []
    labels_list = []
    probs_list = []

    # Hook to capture embeddings before classifier
    # Works for both MidFusion (has .gap) and RGBBackbone (has .backbone)
    activation = {}
    def hook_fn(module, input, output):
        activation['embedding'] = output.detach()

    # Find a suitable hook point (GAP layer or classifier input)
    if hasattr(model, 'gap'):
        hook = model.gap.register_forward_hook(hook_fn)
    elif hasattr(model, 'classifier'):
        hook = model.classifier.register_forward_hook(
            lambda m, inp, out: activation.update({'embedding': inp[0].detach()})
        )
    else:
        print("[Viz] Cannot find hook point for embeddings, skipping")
        return

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if len(embeddings) * dataloader.batch_size >= n_samples:
                break
            inputs_dev = prepare_inputs(inputs, device, mode)
            logits = forward_batch(model, inputs_dev, mode)
            probs = torch.sigmoid(logits)

            embeddings.append(activation['embedding'].flatten(1).cpu())
            labels_list.append(labels)
            probs_list.append(probs.cpu())

    hook.remove()

    embeddings = torch.cat(embeddings, dim=0)[:n_samples].numpy()
    labels_all = torch.cat(labels_list, dim=0)[:n_samples].numpy()
    probs_all = torch.cat(probs_list, dim=0)[:n_samples].numpy()

    print(f"[Viz] Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_emb = tsne.fit_transform(embeddings)

    # Log t-SNE scatter plots per class
    for i, name in enumerate(class_names):
        data = [[x, y, int(l), float(p)] for x, y, l, p in 
                zip(tsne_emb[:, 0], tsne_emb[:, 1], labels_all[:, i], probs_all[:, i])]
        table = wandb.Table(data=data, columns=["x", "y", "label", "prob"])
        wandb.log({f"embeddings/tsne_{name}": wandb.plot.scatter(
            table, "x", "y", title=f"t-SNE: {name}"
        )})

    # UMAP if available
    if HAS_UMAP:
        print("[Viz] Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_emb = reducer.fit_transform(embeddings)

        for i, name in enumerate(class_names):
            data = [[x, y, int(l), float(p)] for x, y, l, p in 
                    zip(umap_emb[:, 0], umap_emb[:, 1], labels_all[:, i], probs_all[:, i])]
            table = wandb.Table(data=data, columns=["x", "y", "label", "prob"])
            wandb.log({f"embeddings/umap_{name}": wandb.plot.scatter(
                table, "x", "y", title=f"UMAP: {name}"
            )})

    print("[Viz] Embedding visualizations logged to W&B")


def build_datasets(cfg: dict):
    """
    Build train/val/test datasets with caching and prefetching.
    
    Feature sets come from cfg["_features"] (set by experiment config).
    Stats are always computed fresh (never loaded from cache).
    
    Returns loaders, channel stats, metadata.
    """
    data_cfg = cfg["data"]
    cache_cfg = cfg["cache"]
    train_cfg = cfg["training"]
    features = cfg["_features"]
    mode = features["mode"]

    bucket = data_cfg["bucket"]
    base_prefixes = data_cfg["base_prefixes"]
    if isinstance(base_prefixes, str):
        base_prefixes = [base_prefixes]  # Backward compatibility

    # Discover sets from all base prefixes
    all_sets = []
    for base_prefix in base_prefixes:
        sets_in_prefix = list_sets(bucket, base_prefix)
        print(f"[Data] Found {len(sets_in_prefix)} sets in {base_prefix}")
        all_sets.extend(sets_in_prefix)
    
    max_sets = data_cfg.get("max_sets")
    if max_sets is not None:
        all_sets = all_sets[:max_sets]
        print(f"[Data] Using {max_sets} of {len(all_sets)} sets (debug mode)")
    print(f"[Data] Total sets: {len(all_sets)}")

    # Build modality list from experiment features
    if mode == "rgb":
        rgb_mods = list(features["rgb_modalities"])
        spectral_mods = rgb_mods  # Used for stats computation
        topo_mods = []
        all_modalities = rgb_mods
        print(f"[Data] RGB mode: {len(rgb_mods)} channels {rgb_mods}")
    else:
        spectral_mods = list(features["spectral_modalities"])
        topo_mods = list(features["topo_modalities"])
        all_modalities = spectral_mods + topo_mods
        print(f"[Data] Full mode: {len(spectral_mods)} spectral + {len(topo_mods)} topo channels")

    label_cols = data_cfg["label_cols"]

    # Simple persistent cache (no eviction, no prefetcher)
    cache = SimpleCache(cache_cfg["local_cache_dir"])

    # Base datasets (stream from S3, cached to disk)
    train_base = CachedEarthScapeDataset(
        bucket, all_sets, all_modalities, "train", label_cols, cache
    )
    val_base = CachedEarthScapeDataset(
        bucket, all_sets, all_modalities, "val", label_cols, cache
    )
    test_base = CachedEarthScapeDataset(
        bucket, all_sets, all_modalities, "test", label_cols, cache
    )

    # Always compute normalization stats fresh (features vary per experiment)
    print("[Stats] Computing fresh normalization stats for this experiment...")
    stats = compute_channel_stats(
        train_base, spectral_mods, topo_mods,
        n_batches=cfg["stats"]["num_batches"],
        batch_size=train_cfg["batch_size"],
    )
    # Save for reference (but never loaded automatically)
    stats_path = cfg["paths"]["stats_path"]
    np.save(stats_path, stats)
    print(f"[Stats] Saved to {stats_path}")

    # Adapted datasets (grouped + normalized)
    train_adapted = EarthscapePatchAdapter(train_base, spectral_mods, topo_mods, stats, mode=mode)
    val_adapted = EarthscapePatchAdapter(val_base, spectral_mods, topo_mods, stats, mode=mode)
    test_adapted = EarthscapePatchAdapter(test_base, spectral_mods, topo_mods, stats, mode=mode)

    # Standard PyTorch DataLoaders (handles prefetching via num_workers)
    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "pin_memory": train_cfg["pin_memory"],
        "prefetch_factor": train_cfg["prefetch_factor"],
        "persistent_workers": train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0,
    }

    train_loader = DataLoader(train_adapted, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_adapted, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_adapted, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, stats, all_sets, topo_mods, spectral_mods, cache


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    val_metrics: Dict,
    train_metrics: Dict,
    cfg: dict,
    stats: dict,
    data_version: dict,
):
    """Save a training checkpoint with full reproducibility info."""
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "val_metrics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_metrics.items() if k != "per_class_confusion"},
        "train_loss": train_metrics["loss"],
        "config": cfg,
        "normalization_stats": stats,
        "data_version": data_version,
    }
    torch.save(ckpt, path)


def train(cfg: dict):
    """Main training function."""
    device = get_device()
    set_seed(cfg["seed"])

    # W&B
    wandb_active = setup_wandb(cfg)

    # Data
    print("\n[Data] Building datasets...")
    (train_loader, val_loader, test_loader, stats,
     all_sets, topo_mods, spectral_mods, cache) = build_datasets(cfg)

    print(f"[Data] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Data version
    data_version = compute_data_version(all_sets, config=cfg)
    print(f"[Version] {data_version}")

    # Model (built from experiment config via registry)
    arch = cfg["model"]["architecture"]
    mode = get_model_mode(arch)
    model = build_model(cfg).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] {arch} ({mode} mode): {param_count:.1f}M parameters")

    # Loss
    pos_weight = compute_pos_weights(train_loader.dataset, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Scheduler
    sched_cfg = cfg["scheduler"]
    if sched_cfg["type"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_cfg["factor"], patience=sched_cfg["patience"]
        )
    elif sched_cfg["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["num_epochs"]
        )

    # AMP
    use_amp = cfg["amp"]["enabled"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"[AMP] {'Enabled' if use_amp else 'Disabled (CPU or config)'}")

    # Checkpoint resume
    paths = cfg["paths"]
    best_ckpt_path = os.path.join(paths["checkpoint_dir"], "best.pth")
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if os.path.exists(best_ckpt_path):
        print(f"[Resume] Loading checkpoint from {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_metrics"]["loss"]
        print(f"[Resume] Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Metric history
    history = {
        "train_loss": [], "val_loss": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_acc": [], "val_acc": [],
        "lr": [],
    }

    class_names = cfg["data"]["label_cols"]
    num_epochs = cfg["training"]["num_epochs"]
    log_every = cfg["logging"]["log_every_n_batches"]
    es_patience = cfg["early_stopping"]["patience"]

    # ========================
    # Training Loop
    # ========================
    print(f"\n{'='*60}")
    print(f"Starting training: {num_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_acc = MetricsAccumulator()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs_dev = prepare_inputs(inputs, device, mode)
            labels_dev = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = forward_batch(model, inputs_dev, mode)
                loss = criterion(logits, labels_dev)

            scaler.scale(loss).backward()
            
            # Compute gradient norm before optimizer step
            grad_norm = compute_gradient_norm(model)
            
            scaler.step(optimizer)
            scaler.update()

            train_acc.update(logits, labels_dev, loss=loss.item(), batch_size=labels.size(0))

            if batch_idx % log_every == 0:
                print(f"  [Train] Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | GradNorm: {grad_norm:.2f}")

        train_metrics = train_acc.compute(class_names=class_names)

        # --- Validate ---
        model.eval()
        val_acc = MetricsAccumulator()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs_dev = prepare_inputs(inputs, device, mode)
                labels_dev = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=use_amp):
                    logits = forward_batch(model, inputs_dev, mode)
                    loss = criterion(logits, labels_dev)

                val_acc.update(logits, labels_dev, loss=loss.item(), batch_size=labels.size(0))

        val_metrics = val_acc.compute(class_names=class_names)

        # --- Scheduler step ---
        current_lr = optimizer.param_groups[0]["lr"]
        if sched_cfg["type"] == "reduce_on_plateau":
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step()

        # --- Logging ---
        epoch_time = time.time() - epoch_start
        print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, current_lr, class_names)
        print(f"  Epoch time: {epoch_time:.1f}s | Cache: {cache.size_gb:.1f} GB")

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["train_acc"].append(train_metrics["avg_accuracy"])
        history["val_acc"].append(val_metrics["avg_accuracy"])
        history["lr"].append(current_lr)

        # Compute throughput
        samples_per_sec = len(train_loader.dataset) / epoch_time
        
        if wandb_active:
            import wandb
            
            wandb.log({
                "epoch": epoch,
                # Train metrics
                "train/loss": train_metrics["loss"],
                "train/macro_f1": train_metrics["macro_f1"],
                "train/accuracy": train_metrics["avg_accuracy"],
                "train/hamming_loss": train_metrics.get("hamming_loss", 0),
                "train/macro_auc": train_metrics.get("macro_auc", 0),
                "train/grad_norm": grad_norm,
                # Val metrics
                "val/loss": val_metrics["loss"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/accuracy": val_metrics["avg_accuracy"],
                "val/exact_match": val_metrics["exact_match_accuracy"],
                "val/hamming_loss": val_metrics.get("hamming_loss", 0),
                "val/macro_auc": val_metrics.get("macro_auc", 0),
                # Training info
                "lr": current_lr,
                "epoch_time_s": epoch_time,
                "throughput/samples_per_sec": samples_per_sec,
                # Per-class metrics
                **{f"val/f1_{name}": val_metrics["per_class_f1"][i] for i, name in enumerate(class_names)},
                **{f"val/acc_{name}": val_metrics["per_class_accuracy"][i] for i, name in enumerate(class_names)},
                **{f"val/auc_{name}": val_metrics["per_class_auc"][i] for i, name in enumerate(class_names)},
            }, step=epoch)

        # --- Checkpointing ---
        if val_metrics["loss"] < best_val_loss - cfg["early_stopping"]["min_delta"]:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0

            save_checkpoint(
                best_ckpt_path, model, optimizer, scaler,
                epoch, val_metrics, train_metrics, cfg, stats, data_version,
            )
            print(f"  ✔ Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{es_patience} epochs")

        # Periodic checkpoint
        if epoch % cfg["checkpoint"]["save_every_n_epochs"] == 0:
            periodic_path = os.path.join(paths["checkpoint_dir"], f"epoch_{epoch}.pth")
            save_checkpoint(
                periodic_path, model, optimizer, scaler,
                epoch, val_metrics, train_metrics, cfg, stats, data_version,
            )
            # Cleanup old checkpoints (keep top K)
            cleanup_old_checkpoints(paths["checkpoint_dir"], keep_top_k=cfg["checkpoint"].get("keep_top_k", 3))

        # Early stopping
        if epochs_no_improve >= es_patience:
            print(f"\n[Early Stop] No improvement for {es_patience} epochs. Stopping.")
            break

        # Memory cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================
    # Final Model Save
    # ========================
    final_path = os.path.join(paths["model_dir"], "final_model.pth")
    save_checkpoint(
        final_path, model, optimizer, scaler,
        epoch, val_metrics, train_metrics, cfg, stats, data_version,
    )
    print(f"\n[Save] Final model saved to {final_path}")

    # Save history
    history_path = os.path.join(paths["output_dir"], "training_history.npy")
    np.save(history_path, history)

    # ========================
    # Test Set Evaluation
    # ========================
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")

    # Load best checkpoint for test eval
    if os.path.exists(best_ckpt_path):
        print("[Test] Loading best checkpoint for evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(
        model, test_loader, criterion, device, class_names, use_amp=use_amp, mode=mode
    )

    # Save test metrics
    test_metrics_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in test_metrics.items()
        if k != "per_class_confusion"
    }
    test_path = os.path.join(paths["output_dir"], "test_metrics.npy")
    np.save(test_path, test_metrics_serializable)
    print(f"[Save] Test metrics saved to {test_path}")

    if wandb_active:
        import wandb
        
        # Log test metrics
        wandb.log({
            "test/loss": test_metrics["loss"],
            "test/macro_f1": test_metrics["macro_f1"],
            "test/micro_f1": test_metrics["micro_f1"],
            "test/accuracy": test_metrics["avg_accuracy"],
            "test/exact_match": test_metrics["exact_match_accuracy"],
            "test/hamming_loss": test_metrics.get("hamming_loss", 0),
            "test/macro_auc": test_metrics.get("macro_auc", 0),
            **{f"test/f1_{name}": test_metrics["per_class_f1"][i] for i, name in enumerate(class_names)},
            **{f"test/auc_{name}": test_metrics["per_class_auc"][i] for i, name in enumerate(class_names)},
        })
        
        # Log confusion matrices
        if "per_class_confusion" in test_metrics:
            for name, cm in test_metrics["per_class_confusion"].items():
                tn, fp, fn, tp = cm.ravel()
                wandb.log({
                    f"confusion/{name}_tn": tn,
                    f"confusion/{name}_fp": fp,
                    f"confusion/{name}_fn": fn,
                    f"confusion/{name}_tp": tp,
                })
        
        # Log model artifact
        artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
        artifact.add_file(final_path)
        wandb.log_artifact(artifact)
        print("[W&B] Model artifact logged")
        
        # Log embedding visualizations if enabled
        if cfg["logging"].get("log_embeddings", True):
            log_embedding_visualization(model, test_loader, device, class_names, mode=mode)
        
        wandb.finish()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Artifacts: {paths['model_dir']}")
    print(f"{'='*60}")

    return model, history, test_metrics


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    cfg = load_config()

    # Validate that experiment config is present
    if "_features" not in cfg:
        print("[ERROR] No experiment config found. Use --experiment <path> to specify one.")
        print("  Example: python train.py --experiment experiments/midfusion_all.yaml")
        print(f"  Available: experiments/*.yaml")
        import sys; sys.exit(1)

    exp = cfg.get("_experiment", {})
    print(f"\n[Config] Resolved config:")
    print(f"  Experiment: {exp.get('name', 'N/A')}")
    print(f"  Architecture: {cfg['model']['architecture']}")
    print(f"  Mode: {cfg['_features']['mode']}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {cfg['training']['num_epochs']}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  LR: {cfg['training']['lr']}")
    print(f"  AMP: {cfg['amp']['enabled']}")
    print(f"  W&B: {cfg['logging']['use_wandb']}")
    print(f"  SageMaker: {cfg['_is_sagemaker']}")

    model, history, test_metrics = train(cfg)

"""
Standalone evaluation script.

Usage:
    # Evaluate with experiment config (recommended)
    python evaluate.py --checkpoint outputs/checkpoints/best.pth --experiment experiments/midfusion_all.yaml

    # Evaluate using config saved in checkpoint (if experiment was saved)
    python evaluate.py --checkpoint outputs/checkpoints/best.pth

    # Override split
    python evaluate.py --checkpoint outputs/checkpoints/best.pth --experiment experiments/resnet50_rgb.yaml --split val
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    CachedEarthScapeDataset,
    EarthscapePatchAdapter,
    SimpleCache,
    compute_channel_stats,
    list_sets,
)
from metrics import evaluate
from models import build_model
from utils import get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained EarthScape model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pth file"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config YAML path"
    )
    parser.add_argument(
        "--experiment", type=str, default=None, help="Experiment YAML path"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for metrics .npy"
    )
    args = parser.parse_args()

    # Build CLI overrides to pass experiment through to load_config
    cli_overrides = ["--config", args.config]
    if args.experiment:
        cli_overrides += ["--experiment", args.experiment]

    cfg = load_config(args.config, cli_overrides=cli_overrides)
    device = get_device()
    set_seed(cfg["seed"])

    # Load checkpoint
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Use config from checkpoint if available and no experiment specified
    # Note: cfg["data"] always comes from current config.yaml (never overwritten
    # by checkpoint) so dataset changes (e.g. new base_prefixes) are picked up.
    if "config" in ckpt:
        saved_cfg = ckpt["config"]
        # Only use checkpoint model/features if no experiment was provided
        if "_features" not in cfg:
            cfg["model"] = saved_cfg.get("model", cfg["model"])
            cfg["_features"] = saved_cfg.get("_features", {})
            cfg["_experiment"] = saved_cfg.get("_experiment", {})

    # Validate we have feature config
    if "_features" not in cfg or not cfg["_features"]:
        print(
            "[Eval] ERROR: No feature config found. Provide --experiment or use a checkpoint that has it."
        )
        sys.exit(1)

    features = cfg["_features"]
    mode = features["mode"]
    arch = cfg["model"]["architecture"]

    # Print data version info if available
    if "data_version" in ckpt:
        dv = ckpt["data_version"]
        print(f"[Eval] Model trained on {dv.get('num_sets', '?')} sets")
        print(f"[Eval] Git commit: {dv.get('git_commit', 'unknown')}")
        print(f"[Eval] Trained at: {dv.get('timestamp', 'unknown')}")

    print(f"[Eval] Architecture: {arch} ({mode} mode)")

    # Build dataset for requested split
    data_cfg = cfg["data"]
    cache_cfg = cfg["cache"]

    bucket = data_cfg["bucket"]
    base_prefixes = data_cfg.get("base_prefixes", [data_cfg.get("base_prefix", "")])
    if isinstance(base_prefixes, str):
        base_prefixes = [base_prefixes]

    all_sets = []
    for base_prefix in base_prefixes:
        all_sets.extend(list_sets(bucket, base_prefix))

    max_sets = data_cfg.get("max_sets")
    if max_sets:
        all_sets = all_sets[:max_sets]

    # Build modality list from experiment features
    if mode == "rgb":
        rgb_mods = list(features["rgb_modalities"])
        spectral_mods = rgb_mods
        topo_mods = []
        all_modalities = rgb_mods
    else:
        spectral_mods = list(features["spectral_modalities"])
        topo_mods = list(features["topo_modalities"])
        all_modalities = spectral_mods + topo_mods

    label_cols = data_cfg["label_cols"]
    cache = SimpleCache(cache_cfg["local_cache_dir"])

    base_dataset = CachedEarthScapeDataset(
        bucket, all_sets, all_modalities, args.split, label_cols, cache
    )

    # Load stats from checkpoint or compute fresh
    stats = ckpt.get("normalization_stats")
    if stats is None:
        print("[Stats] No stats in checkpoint, computing fresh...")
        stats = compute_channel_stats(
            base_dataset,
            spectral_mods,
            topo_mods,
            n_batches=cfg["stats"]["num_batches"],
            batch_size=cfg["training"]["batch_size"],
        )

    adapted = EarthscapePatchAdapter(
        base_dataset, spectral_mods, topo_mods, stats, mode=mode
    )

    loader = DataLoader(
        adapted,
        shuffle=False,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        prefetch_factor=cfg["training"]["prefetch_factor"],
        persistent_workers=cfg["training"]["persistent_workers"]
        and cfg["training"]["num_workers"] > 0,
    )

    # Build model from registry
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Eval] Model loaded (epoch {ckpt.get('epoch', '?')})")

    # Loss (no pos_weight for evaluation)
    criterion = nn.BCEWithLogitsLoss()

    # Evaluate
    use_amp = cfg["amp"]["enabled"] and device.type == "cuda"
    metrics = evaluate(
        model, loader, criterion, device, label_cols, use_amp=use_amp, mode=mode
    )

    # Save
    output_path = args.output or os.path.join(
        cfg["paths"]["output_dir"], f"{args.split}_metrics.npy"
    )
    metrics_save = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
        if k != "per_class_confusion"
    }
    np.save(output_path, metrics_save)
    print(f"\n[Eval] Metrics saved to {output_path}")


if __name__ == "__main__":
    main()

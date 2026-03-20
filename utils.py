"""
Utilities: config loading, seed setting, path resolution, data versioning.
"""

import argparse
import hashlib
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import yaml

# ============================================================================
# Config Loading
# ============================================================================


def load_config(
    config_path: str = "config.yaml", cli_overrides: list | None = None
) -> dict[str, Any]:
    """
    Load YAML config, merge experiment config if provided, and apply CLI overrides.

    CLI overrides use dot notation: --training.lr 3e-4 --data.use_nhd false
    Experiment config: --experiment experiments/midfusion_all.yaml
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Parse CLI overrides
    if cli_overrides is None:
        cli_overrides = sys.argv[1:]

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument("--experiment", type=str, default=None)
    # Capture all unknown args as overrides
    known, unknown = parser.parse_known_args(cli_overrides)

    # Merge experiment config if provided
    if known.experiment:
        experiment = load_experiment(known.experiment)
        cfg = merge_experiment_into_config(cfg, experiment)

    # Apply overrides: --training.lr 3e-4 -> cfg["training"]["lr"] = 3e-4
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key_path = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = _parse_value(unknown[i + 1])
                _set_nested(cfg, key_path, value)
                i += 2
            else:
                _set_nested(cfg, key_path, True)
                i += 1
        else:
            i += 1

    # Resolve auto-detected paths
    cfg = _resolve_paths(cfg)

    return cfg


def load_experiment(experiment_path: str) -> dict[str, Any]:
    """
    Load an experiment YAML file.

    Experiment files define model architecture, feature sets, and optional
    training overrides for quick experimentation.
    """
    with open(experiment_path) as f:
        experiment = yaml.safe_load(f)
    print(f"[Experiment] Loaded: {experiment.get('name', experiment_path)}")
    if experiment.get("description"):
        print(f"  {experiment['description']}")
    return experiment


def merge_experiment_into_config(
    cfg: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge experiment config into base config.

    Experiment can override:
      - model.* (architecture, dropout, etc.)
      - training.* (lr, epochs, etc.)
      - features.* (spectral_modalities, topo_modalities, rgb_modalities, mode)
      - Any other top-level section

    Features are stored in cfg["_features"] for use by train/evaluate.
    Experiment metadata stored in cfg["_experiment"].
    """
    # Store experiment metadata
    cfg["_experiment"] = {
        "name": experiment.get("name", "unnamed"),
        "description": experiment.get("description", ""),
    }

    # Merge model config
    if "model" in experiment:
        cfg["model"] = {**cfg.get("model", {}), **experiment["model"]}

    # Store features config (this is the key per-experiment setting)
    features = experiment.get("features", {})
    cfg["_features"] = {
        "spectral_modalities": features.get("spectral_modalities", []),
        "topo_modalities": features.get("topo_modalities", []),
        "rgb_modalities": features.get("rgb_modalities", []),
        "mode": features.get("mode", "full"),
    }

    # Merge any training overrides from experiment
    if "training" in experiment:
        cfg["training"] = {**cfg.get("training", {}), **experiment["training"]}

    # Merge any other top-level sections
    for key in experiment:
        if key not in ("name", "description", "model", "features", "training"):
            if isinstance(experiment[key], dict) and key in cfg:
                cfg[key] = {**cfg[key], **experiment[key]}
            else:
                cfg[key] = experiment[key]

    return cfg


def _parse_value(v: str) -> Any:
    """Parse string value to appropriate Python type."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "null" or v.lower() == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _set_nested(d: dict, key_path: str, value: Any):
    """Set a nested dict value using dot notation: 'training.lr' -> d['training']['lr']."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _resolve_paths(cfg: dict) -> dict:
    """Auto-detect SageMaker vs local environment and resolve paths.

    When an experiment is specified, outputs are namespaced under the
    experiment name so different runs don't clobber each other:
        ./outputs/midfusion_all/checkpoints/best.pth
        ./outputs/resnet50_rgb/checkpoints/best.pth
    """
    is_sagemaker = os.path.exists("/opt/ml")

    paths = cfg.get("paths", {})
    cache = cfg.get("cache", {})

    # Experiment name for subdirectory namespacing
    exp_name = cfg.get("_experiment", {}).get("name", "")

    if is_sagemaker:
        default_model_dir = "/opt/ml/model"
        default_output_dir = "/opt/ml/output"
        default_cache_dir = "/opt/ml/input/data_cache"
    else:
        base_output = "./outputs"
        if exp_name:
            default_model_dir = os.path.join(base_output, exp_name)
            default_output_dir = os.path.join(base_output, exp_name)
        else:
            default_model_dir = base_output
            default_output_dir = base_output
        default_cache_dir = "./data_cache"

    paths["model_dir"] = paths.get("model_dir") or default_model_dir
    paths["output_dir"] = paths.get("output_dir") or default_output_dir
    paths["checkpoint_dir"] = paths.get("checkpoint_dir") or os.path.join(
        paths["model_dir"], "checkpoints"
    )
    paths["stats_path"] = paths.get("stats_path") or os.path.join(
        paths["model_dir"], "normalization_stats.npy"
    )

    cache["local_cache_dir"] = cache.get("local_cache_dir") or default_cache_dir

    cfg["paths"] = paths
    cfg["cache"] = cache
    cfg["_is_sagemaker"] = is_sagemaker

    # Create directories
    for d in [paths["model_dir"], paths["output_dir"], paths["checkpoint_dir"]]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(cache["local_cache_dir"], exist_ok=True)

    return cfg


# ============================================================================
# Seed
# ============================================================================


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds set to {seed}")


# ============================================================================
# Data Versioning
# ============================================================================


def compute_data_version(
    sets_used: list,
    split_csv_content: str | None = None,
    config: dict | None = None,
) -> dict:
    """
    Create a data version fingerprint for reproducibility.

    Store this in checkpoints and log to W&B.
    """
    version = {
        "sets_used": sorted(sets_used),
        "num_sets": len(sets_used),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Hash the split CSV if provided
    if split_csv_content:
        version["split_hash"] = hashlib.md5(split_csv_content.encode()).hexdigest()

    # Try to get git commit
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        version["git_commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        version["git_commit"] = "unknown"

    # Include config snapshot
    if config:
        version["config_hash"] = hashlib.md5(
            yaml.dump(config, sort_keys=True).encode()
        ).hexdigest()

    return version


# ============================================================================
# Device
# ============================================================================


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        print(
            f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (AMP autocast will fall back to float32)")
    return device

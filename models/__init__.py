"""
Model registry and unified interface for EarthScape architectures.

Two model families:
  - "full" mode: dual-backbone models (spectral + topo inputs)
  - "rgb" mode: single-backbone models (RGB image input)

Usage:
    from models import build_model, forward_batch, prepare_inputs, get_model_mode

    model = build_model(cfg)
    mode = get_model_mode(cfg["model"]["architecture"])

    inputs_dev = prepare_inputs(inputs, device, mode)
    logits = forward_batch(model, inputs_dev, mode)
"""

from typing import Dict, Union

import torch
import torch.nn as nn

from models.midfusion import MidFusionResNet
from models.midfusion_v3 import MidFusionResNet_V3
from models.midfusion_v4 import MidFusionResNet_V4
from models.deit import DeiTBackbone, DeiTEarlyFusion, DeiTLateFusion, DeiTLateFusion_Gated
from models.modern_fusion import ModernMidFusion, ModernMidFusion_Gated
from models.rgb_backbone import RGBBackbone

# ============================================================================
# Registry: architecture_name -> (class, mode, default_kwargs)
# ============================================================================

MODEL_REGISTRY = {
    # Full mode (spectral + topo)
    "midfusion": {
        "cls": MidFusionResNet,
        "mode": "full",
    },
    "midfusion_v3": {
        "cls": MidFusionResNet_V3,
        "mode": "full",
    },
    "midfusion_v4": {
        "cls": MidFusionResNet_V4,
        "mode": "full",
    },
    # RGB mode (single image backbone)
    "resnet50": {
        "cls": RGBBackbone,
        "mode": "rgb",
        "kwargs": {"backbone": "resnet50"},
    },
    "efficientnet_b0": {
        "cls": RGBBackbone,
        "mode": "rgb",
        "kwargs": {"backbone": "efficientnet_b0"},
    },
    "convnext_tiny": {
        "cls": RGBBackbone,
        "mode": "rgb",
        "kwargs": {"backbone": "convnext_tiny"},
    },
    "swin_t": {
        "cls": RGBBackbone,
        "mode": "rgb",
        "kwargs": {"backbone": "swin_t"},
    },
    # Full mode — modern backbones (spectral + topo)
    "convnext_fusion": {
        "cls": ModernMidFusion,
        "mode": "full",
        "kwargs": {"backbone": "convnext_tiny"},
    },
    "swin_fusion": {
        "cls": ModernMidFusion,
        "mode": "full",
        "kwargs": {"backbone": "swin_t"},
    },
    "convnext_fusion_gated": {
        "cls": ModernMidFusion_Gated,
        "mode": "full",
        "kwargs": {"backbone": "convnext_tiny"},
    },
    "swin_fusion_gated": {
        "cls": ModernMidFusion_Gated,
        "mode": "full",
        "kwargs": {"backbone": "swin_t"},
    },
    # DeiT (Data-efficient Image Transformer)
    "deit_small": {
        "cls": DeiTBackbone,
        "mode": "rgb",
    },
    "deit_fusion": {
        "cls": DeiTLateFusion,
        "mode": "full",
    },
    "deit_early_fusion": {
        "cls": DeiTEarlyFusion,
        "mode": "full",
    },
    "deit_fusion_gated": {
        "cls": DeiTLateFusion_Gated,
        "mode": "full",
    },
    # ConvNeXt-Small variants
    "convnext_small_fusion": {
        "cls": ModernMidFusion,
        "mode": "full",
        "kwargs": {"backbone": "convnext_small"},
    },
    "convnext_small_fusion_gated": {
        "cls": ModernMidFusion_Gated,
        "mode": "full",
        "kwargs": {"backbone": "convnext_small"},
    },
}


def get_model_mode(architecture: str) -> str:
    """Return 'full' or 'rgb' for a given architecture name."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[architecture]["mode"]


def build_model(cfg: dict) -> nn.Module:
    """
    Build a model from config.

    Expects cfg to have:
      cfg["model"]["architecture"]  -> registry key
      cfg["model"]["num_classes"]
      cfg["model"]["dropout"]
      cfg["_features"]["spectral_modalities"]  (for full mode)
      cfg["_features"]["topo_modalities"]      (for full mode)
      cfg["_features"]["rgb_modalities"]       (for rgb mode)
    """
    model_cfg = cfg["model"]
    arch = model_cfg["architecture"]
    entry = MODEL_REGISTRY[arch]

    mode = entry["mode"]
    extra_kwargs = entry.get("kwargs", {})

    # Optional model params — only forwarded when present in config
    if "drop_path_rate" in model_cfg:
        extra_kwargs = {**extra_kwargs, "drop_path_rate": model_cfg["drop_path_rate"]}

    if mode == "full":
        spectral_ch = len(cfg["_features"]["spectral_modalities"])
        topo_ch = len(cfg["_features"]["topo_modalities"])
        model = entry["cls"](
            num_classes=model_cfg["num_classes"],
            spectral_in_ch=spectral_ch,
            topo_in_ch=topo_ch,
            dropout=model_cfg.get("dropout", 0.3),
            **extra_kwargs,
        )
    elif mode == "rgb":
        in_ch = len(
            cfg["_features"].get("rgb_modalities", ["aerialr", "aerialg", "aerialb"])
        )
        model = entry["cls"](
            num_classes=model_cfg["num_classes"],
            in_channels=in_ch,
            dropout=model_cfg.get("dropout", 0.3),
            **extra_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}' for architecture '{arch}'")

    return model


def prepare_inputs(
    inputs: torch.Tensor | dict[str, torch.Tensor],
    device: torch.device,
    mode: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Move batch inputs to device based on model mode."""
    if mode == "rgb":
        return inputs.to(device, non_blocking=True)
    else:
        return {
            "spectral": inputs["spectral"].to(device, non_blocking=True),
            "topo": inputs["topo"].to(device, non_blocking=True),
        }


def forward_batch(
    model: nn.Module,
    inputs_dev: torch.Tensor | dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Run model forward pass based on mode."""
    if mode == "rgb":
        return model(inputs_dev)
    else:
        return model(inputs_dev["spectral"], inputs_dev["topo"])

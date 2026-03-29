"""
DeiT3-Small dual-backbone late fusion for geological classification.

Architecture:
  - Spectral encoder (DeiT3-Small) → 384-dim CLS token
  - Topo encoder (DeiT3-Small) → 384-dim CLS token
  - Concatenate → 2-layer MLP classifier (768→512→num_classes)

Late fusion is natural for ViTs since they output a global CLS token
rather than spatial feature maps — no spatial alignment needed.
"""

import torch
import torch.nn as nn
import timm


class DeiTLateFusion(nn.Module):
    """
    Dual-backbone DeiT3-Small with late fusion.

    Args:
        num_classes: number of output classes
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 or 8)
        dropout: classifier dropout rate
        img_size: input image size (default 256)
        drop_path_rate: stochastic depth rate (0.0 = disabled)
    """

    def __init__(
        self,
        num_classes: int = 7,
        spectral_in_ch: int = 4,
        topo_in_ch: int = 7,
        dropout: float = 0.3,
        img_size: int = 256,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.spec_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=spectral_in_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        self.topo_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=topo_in_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        feat_dim = self.spec_encoder.embed_dim  # 384

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        spec_feat = self.spec_encoder(spectral)  # [B, 384]
        topo_feat = self.topo_encoder(topo)  # [B, 384]
        combined = torch.cat([spec_feat, topo_feat], dim=1)  # [B, 768]
        return self.classifier(combined)


# ============================================================================
# Layer-wise Learning Rate Decay (LLRD)
# ============================================================================


def _get_deit_layer_id(name: str, num_layers: int = 12) -> int:
    """Assign layer id: 0 = patch embed, 1..12 = blocks, 13 = head."""
    if "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
        return 0
    elif "blocks." in name:
        return int(name.split("blocks.")[1].split(".")[0]) + 1
    else:
        return num_layers + 1


def build_deit_llrd_param_groups(
    model: DeiTLateFusion,
    base_lr: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
    num_layers: int = 12,
) -> list[dict]:
    """
    Build optimizer param groups with layer-wise LR decay.

    Both encoders get the same LLRD schedule independently.
    Classifier gets base_lr (no decay).

    LR schedule:
      - patch_embed / cls_token / pos_embed: base_lr * decay^13
      - block i: base_lr * decay^(12 - i)
      - classifier: base_lr

    Args:
        model: DeiTLateFusion instance
        base_lr: base learning rate (for classifier)
        weight_decay: weight decay value
        llrd_decay: decay factor per layer (default 0.75)
        num_layers: number of transformer blocks (12 for DeiT-Small)
    """
    param_groups: dict = {}
    no_decay_keywords = {"bias", "norm", "cls_token", "pos_embed"}
    encoder_attrs = ["spec_encoder", "topo_encoder"]

    # Encoder params with LLRD
    for enc_attr in encoder_attrs:
        encoder = getattr(model, enc_attr)
        for param_name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue

            layer_id = _get_deit_layer_id(param_name, num_layers)
            lr = base_lr * llrd_decay ** (num_layers + 1 - layer_id)
            wd = 0.0 if any(kw in param_name for kw in no_decay_keywords) else weight_decay

            key = (round(lr, 10), wd)
            if key not in param_groups:
                param_groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
            param_groups[key]["params"].append(param)

    # Classifier params at base_lr
    head_decay, head_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(enc) for enc in encoder_attrs):
            continue
        if any(kw in name for kw in no_decay_keywords):
            head_no_decay.append(param)
        else:
            head_decay.append(param)

    if head_decay:
        param_groups[("head_decay",)] = {
            "params": head_decay, "lr": base_lr, "weight_decay": weight_decay,
        }
    if head_no_decay:
        param_groups[("head_no_decay",)] = {
            "params": head_no_decay, "lr": base_lr, "weight_decay": 0.0,
        }

    groups = list(param_groups.values())
    total_params = sum(p.numel() for g in groups for p in g["params"])
    lrs = sorted(set(g["lr"] for g in groups))
    print(f"[LLRD] {len(groups)} param groups, {total_params/1e6:.1f}M params")
    print(f"[LLRD] LR range: {min(lrs):.2e} → {max(lrs):.2e} (decay={llrd_decay})")

    return groups

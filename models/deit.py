"""
DeiT (Data-efficient Image Transformer) backbones for geological classification.

Uses DeiT3-Small from timm (patch16, 384-dim embeddings, ~22M params).
DeiT is designed to train well on smaller datasets via distillation and
strong augmentations — a good fit for our ~thousands-of-patches regime.

Two variants:
  - DeiTBackbone: single-input RGB/spectral classifier
  - DeiTLateFusion: dual-backbone late fusion (spectral + topo)

Since ViTs produce a global CLS token (not spatial feature maps), we use
late fusion (concat feature vectors) rather than mid-level spatial fusion.
"""

import torch
import torch.nn as nn
import timm


class DeiTBackbone(nn.Module):
    """
    Single-backbone DeiT classifier for geological formations.

    Uses DeiT3-Small-patch16 from timm with ImageNet pretraining.
    Supports arbitrary input channels via timm's in_chans parameter.

    Args:
        num_classes: number of output classes (7 geological formations)
        in_channels: input channels (3 for RGB, 4 for RGBNIR, etc.)
        dropout: classifier dropout rate
        img_size: input image size (default 256)
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        dropout: float = 0.3,
        img_size: int = 256,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,  # remove classification head, get features only
            in_chans=in_channels,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.embed_dim  # 384 for DeiT-Small

        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        with torch.no_grad():
            self.classifier.weight.mul_(0.001)
            self.classifier.bias.mul_(0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            logits: [B, num_classes]
        """
        features = self.backbone(x)  # [B, 384]
        return self.classifier(self.head_drop(features))


class DeiTEarlyFusion(nn.Module):
    """
    Early-fusion DeiT classifier for multimodal geological classification.

    Concatenates spectral and topo channels into a single (spectral_in_ch + topo_in_ch)
    channel image, then processes through a single DeiT3-Small backbone.

    This is the cleanest multimodal ViT approach: the patch embedding projects
    all channels jointly, and self-attention learns cross-modal relationships
    implicitly across patches. No architecture surgery required.

    Note: The patch embedding is reinitialized for the non-standard channel count,
    so ImageNet pretraining is lost at the input layer (but preserved everywhere else).

    Args:
        num_classes: number of output classes (7 geological formations)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        dropout: classifier dropout rate
        img_size: input image size (default 256)
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
        total_ch = spectral_in_ch + topo_in_ch

        self.backbone = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=total_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.embed_dim  # 384 for DeiT-Small

        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        with torch.no_grad():
            self.classifier.weight.mul_(0.001)
            self.classifier.bias.mul_(0.001)

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, spectral_in_ch, H, W]
            topo: [B, topo_in_ch, H, W]
        Returns:
            logits: [B, num_classes]
        """
        x = torch.cat([spectral, topo], dim=1)  # [B, total_ch, H, W]
        features = self.backbone(x)  # [B, 384]
        return self.classifier(self.head_drop(features))


class DeiTLateFusion(nn.Module):
    """
    Dual-backbone DeiT with late fusion for multimodal geological classification.

    Each modality (spectral, topo) gets its own DeiT3-Small encoder.
    Feature vectors are concatenated and passed through a classifier.

    Late fusion is natural for ViTs since they output a global CLS token
    rather than spatial feature maps — no need for spatial alignment.

    Args:
        num_classes: number of output classes (7 geological formations)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        dropout: classifier dropout rate
        img_size: input image size (default 256)
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

        # Spectral encoder
        self.spec_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=spectral_in_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        # Topo encoder
        self.topo_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=topo_in_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        feat_dim = self.spec_encoder.embed_dim  # 384

        # Classifier on concatenated features (384 + 384 = 768)
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim * 2, num_classes)
        with torch.no_grad():
            self.classifier.weight.mul_(0.001)
            self.classifier.bias.mul_(0.001)

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, spectral_in_ch, H, W]
            topo: [B, topo_in_ch, H, W]
        Returns:
            logits: [B, num_classes]
        """
        spec_feat = self.spec_encoder(spectral)  # [B, 384]
        topo_feat = self.topo_encoder(topo)  # [B, 384]
        combined = torch.cat([spec_feat, topo_feat], dim=1)  # [B, 768]
        return self.classifier(self.head_drop(combined))


class DeiTLateFusion_Gated(nn.Module):
    """
    Dual-backbone DeiT with gated late fusion for multimodal geological classification.

    Same as DeiTLateFusion but adds a lightweight gate MLP that produces
    per-sample scalar weights for each branch's CLS token before concatenation.
    The gate sees both feature vectors and learns when to up/down-weight
    each modality — analogous to ModernMidFusion_Gated but operating on
    global vectors instead of spatial feature maps.

    Gate is zero-initialized so it starts at sigmoid(0) = 0.5 (equal weighting),
    identical to unweighted DeiTLateFusion at init.

    Args:
        num_classes: number of output classes (7 geological formations)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        dropout: classifier dropout rate
        img_size: input image size (default 256)
        drop_path_rate: stochastic depth rate for transformer blocks
        gate_bottleneck: hidden dim for the gate MLP
    """

    def __init__(
        self,
        num_classes: int = 7,
        spectral_in_ch: int = 4,
        topo_in_ch: int = 7,
        dropout: float = 0.3,
        img_size: int = 256,
        drop_path_rate: float = 0.0,
        gate_bottleneck: int = 64,
    ):
        super().__init__()

        # Spectral encoder
        self.spec_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=spectral_in_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        # Topo encoder
        self.topo_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=topo_in_ch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        feat_dim = self.spec_encoder.embed_dim  # 384

        # Gate MLP: sees both CLS tokens (768-dim) → produces 2 scalars
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, gate_bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(gate_bottleneck, 2),
        )
        # Zero-init so gates start at sigmoid(0) = 0.5
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        # Classifier on concatenated features (384 + 384 = 768)
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim * 2, num_classes)
        with torch.no_grad():
            self.classifier.weight.mul_(0.001)
            self.classifier.bias.mul_(0.001)

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, spectral_in_ch, H, W]
            topo: [B, topo_in_ch, H, W]
        Returns:
            logits: [B, num_classes]
        """
        spec_feat = self.spec_encoder(spectral)  # [B, 384]
        topo_feat = self.topo_encoder(topo)  # [B, 384]

        combined = torch.cat([spec_feat, topo_feat], dim=1)  # [B, 768]

        # Per-sample gating
        gates = torch.sigmoid(self.gate_mlp(combined))  # [B, 2]
        gate_spec = gates[:, 0:1]  # [B, 1]
        gate_topo = gates[:, 1:2]  # [B, 1]

        gated = torch.cat([gate_spec * spec_feat, gate_topo * topo_feat], dim=1)
        return self.classifier(self.head_drop(gated))

    def get_gate_values(
        self, spectral: torch.Tensor, topo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return gate values for analysis."""
        with torch.no_grad():
            spec_feat = self.spec_encoder(spectral)
            topo_feat = self.topo_encoder(topo)
            combined = torch.cat([spec_feat, topo_feat], dim=1)
            gates = torch.sigmoid(self.gate_mlp(combined))
            return gates[:, 0], gates[:, 1]


# ============================================================================
# Layer-wise Learning Rate Decay (LLRD) for DeiT
# ============================================================================


def _get_deit_layer_id(name: str, num_layers: int = 12) -> int:
    """
    Assign a layer id (0 = patch embed, 1..num_layers = blocks, num_layers+1 = head).

    Used to build per-layer LR groups for LLRD.
    """
    if "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
        return 0
    elif "blocks." in name:
        block_id = int(name.split("blocks.")[1].split(".")[0])
        return block_id + 1
    else:
        # norm, classifier, head_drop, gate_mlp, etc.
        return num_layers + 1


def build_deit_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
    num_layers: int = 12,
) -> list[dict]:
    """
    Build optimizer parameter groups with layer-wise learning rate decay.

    For dual-backbone DeiT models (DeiTLateFusion, DeiTLateFusion_Gated):
      - Both encoders get the same LLRD schedule independently
      - Classifier / gate_mlp / head_drop get base_lr (no decay)

    For single-backbone DeiT models (DeiTBackbone, DeiTEarlyFusion):
      - Single backbone gets LLRD schedule
      - Classifier gets base_lr

    Layer LR schedule:
      - patch_embed / cls_token / pos_embed: base_lr * decay^(num_layers + 1)
      - block i (0-indexed): base_lr * decay^(num_layers - i)
      - head / classifier: base_lr * 1.0

    Args:
        model: DeiT model instance
        base_lr: base learning rate (for head)
        weight_decay: weight decay value
        llrd_decay: decay factor per layer (default 0.75)
        num_layers: number of transformer blocks (12 for DeiT-Small)

    Returns:
        List of param group dicts for torch.optim
    """
    param_groups: dict[int, dict] = {}
    no_decay_keywords = {"bias", "norm", "cls_token", "pos_embed"}

    # Determine encoder attribute names
    if hasattr(model, "spec_encoder"):
        encoder_attrs = ["spec_encoder", "topo_encoder"]
    elif hasattr(model, "backbone"):
        encoder_attrs = ["backbone"]
    else:
        raise ValueError("Cannot determine encoder attributes for LLRD")

    # Assign encoder params to layer groups
    for enc_attr in encoder_attrs:
        encoder = getattr(model, enc_attr)
        for param_name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue

            layer_id = _get_deit_layer_id(param_name, num_layers)
            lr_scale = llrd_decay ** (num_layers + 1 - layer_id)
            lr = base_lr * lr_scale

            # No weight decay for biases and norms
            wd = 0.0 if any(kw in param_name for kw in no_decay_keywords) else weight_decay

            group_key = (round(lr, 10), wd)
            if group_key not in param_groups:
                param_groups[group_key] = {
                    "params": [],
                    "lr": lr,
                    "weight_decay": wd,
                }
            param_groups[group_key]["params"].append(param)

    # Head params (classifier, head_drop, gate_mlp) get base_lr
    head_params_decay = []
    head_params_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip encoder params (already handled above)
        is_encoder = any(name.startswith(enc) for enc in encoder_attrs)
        if is_encoder:
            continue
        if any(kw in name for kw in no_decay_keywords):
            head_params_no_decay.append(param)
        else:
            head_params_decay.append(param)

    if head_params_decay:
        param_groups[("head_decay",)] = {
            "params": head_params_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
        }
    if head_params_no_decay:
        param_groups[("head_no_decay",)] = {
            "params": head_params_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
        }

    groups = list(param_groups.values())

    # Print summary
    total_params = sum(p.numel() for g in groups for p in g["params"])
    lrs = sorted(set(g["lr"] for g in groups))
    print(f"[LLRD] {len(groups)} param groups, {total_params/1e6:.1f}M params")
    print(f"[LLRD] LR range: {min(lrs):.2e} → {max(lrs):.2e} (decay={llrd_decay})")

    return groups

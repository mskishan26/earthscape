"""
Dual-head mid-fusion with modern backbones (ConvNeXt-Tiny, Swin-T).

Architecture:
  - Spectral backbone (ConvNeXt/Swin, stages 0-2) → 384-ch feature maps
  - Topo backbone (ConvNeXt/Swin, stages 0-2) → 384-ch feature maps
  - 1x1 conv fusion at stage-2 output (768→384)
  - Shared stage-3 → GAP → classifier (768-dim features)

Both ConvNeXt-Tiny and Swin-T share the same stage dimensions:
  Stage 0: 96ch, Stage 1: 192ch, Stage 2: 384ch, Stage 3: 768ch

This module also includes a conditioned (SE-style) gating variant,
following the same pattern as MidFusionResNet_V4.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    Swin_T_Weights,
    convnext_small,
    convnext_tiny,
    swin_t,
)
from torchvision.ops.stochastic_depth import StochasticDepth



class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels-first tensors [B, C, H, W]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ============================================================================
# Helpers
# ============================================================================


def _adapt_first_conv(model, in_channels: int):
    """Replace the first conv/patch-embed layer for non-3-channel input."""
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        if in_channels >= 3:
            new_conv.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c : c + 1] = mean_rgb * 0.1
        else:
            new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
    model.features[0][0] = new_conv
    return model


def _build_backbone(backbone_name: str):
    """Build a pretrained backbone and return (model, stage_dims)."""
    if backbone_name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    elif backbone_name == "convnext_small":
        model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    elif backbone_name == "swin_t":
        model = swin_t(weights=Swin_T_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    # Both have stage dims [96, 192, 384, 768]
    return model


def _forward_stages(model, x: torch.Tensor, end_stage: int) -> torch.Tensor:
    """
    Run model.features[0..end_stage*2+1] (inclusive).

    Both ConvNeXt-Tiny and Swin-T structure their features as pairs:
      features[0], features[1] → stage 0  (output: 96ch)
      features[2], features[3] → stage 1  (output: 192ch)
      features[4], features[5] → stage 2  (output: 384ch)
      features[6], features[7] → stage 3  (output: 768ch)

    end_stage=2 runs through features[0..5], producing 384ch output.
    """
    end_idx = end_stage * 2 + 2  # exclusive
    for i in range(end_idx):
        x = model.features[i](x)
    return x


def _forward_final_stage(model, x: torch.Tensor) -> torch.Tensor:
    """Run the final stage (features[6], features[7])."""
    x = model.features[6](x)
    x = model.features[7](x)
    return x


def _set_drop_path_rate(backbone: nn.Module, rate: float):
    """Rescale stochastic depth to a linear schedule [0 → rate] across blocks."""
    sd_layers = [m for m in backbone.modules() if isinstance(m, StochasticDepth)]
    n = len(sd_layers)
    for i, layer in enumerate(sd_layers):
        layer.p = rate * (i + 1) / n


# ============================================================================
# ModernMidFusion: ConvNeXt/Swin dual-backbone with mid-level fusion
# ============================================================================


class ModernMidFusion(nn.Module):
    """
    Dual-backbone mid-fusion using ConvNeXt-Tiny or Swin-T.

    Fuses at stage-2 output (384ch each), runs shared stage-3 (→768ch),
    then GAP + classifier.

    Args:
        backbone: "convnext_tiny" or "swin_t"
        num_classes: number of output classes (7 geological formations)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        dropout: classifier dropout rate
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 7,
        spectral_in_ch: int = 4,
        topo_in_ch: int = 7,
        dropout: float = 0.3,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone
        mid_ch = 384  # stage-2 output for both ConvNeXt-T and Swin-T
        final_ch = 768  # stage-3 output

        # Spectral backbone
        self.spec_backbone = _build_backbone(backbone)
        if spectral_in_ch != 3:
            self.spec_backbone = _adapt_first_conv(self.spec_backbone, spectral_in_ch)

        # Topo backbone
        self.topo_backbone = _build_backbone(backbone)
        if topo_in_ch != 3:
            self.topo_backbone = _adapt_first_conv(self.topo_backbone, topo_in_ch)

        # Fusion: concat stage-2 outputs (384+384) → 1x1 conv → 384
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1, bias=False),
            LayerNorm2d(mid_ch),
            nn.GELU(),
        )

        # Shared final stage (from spectral backbone)
        # We'll run features[6] and features[7] on the fused output
        self.shared_final = nn.ModuleList(
            [self.spec_backbone.features[6], self.spec_backbone.features[7]]
        )

        # Stochastic depth for finetuning
        if drop_path_rate > 0:
            _set_drop_path_rate(self.spec_backbone, drop_path_rate)
            _set_drop_path_rate(self.topo_backbone, drop_path_rate)

        # Classifier (ConvNeXt finetuning recipe: single linear + head_init_scale)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_ch, num_classes)
        with torch.no_grad():
            self.classifier.weight.mul_(0.001)
            self.classifier.bias.mul_(0.001)

    def _forward_branches(
        self, spectral: torch.Tensor, topo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract stage-2 features from both branches."""
        spec_feat = _forward_stages(self.spec_backbone, spectral, end_stage=2)
        topo_feat = _forward_stages(self.topo_backbone, topo, end_stage=2)

        # Swin-T outputs [B, H, W, C] (channels-last) — permute to [B, C, H, W]
        if self.backbone_name == "swin_t":
            spec_feat = spec_feat.permute(0, 3, 1, 2).contiguous()
            topo_feat = topo_feat.permute(0, 3, 1, 2).contiguous()

        return spec_feat, topo_feat

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, spectral_in_ch, H, W]
            topo: [B, topo_in_ch, H, W]
        Returns:
            logits: [B, num_classes]
        """
        spec_feat, topo_feat = self._forward_branches(spectral, topo)

        # Fuse
        fused = self.fusion(
            torch.cat([spec_feat, topo_feat], dim=1)
        )  # [B, 384, H', W']

        # For Swin, need to permute back to [B, H, W, C] for stage-3
        if self.backbone_name == "swin_t":
            fused = fused.permute(0, 2, 3, 1).contiguous()  # [B, H', W', C]

        # Shared final stage
        for layer in self.shared_final:
            fused = layer(fused)

        # Pool + classify
        if self.backbone_name == "swin_t":
            # Swin output: [B, H', W', 768] → permute to [B, 768, H', W'] for GAP
            fused = fused.permute(0, 3, 1, 2).contiguous()

        out = self.gap(fused).flatten(1)  # [B, 768]
        return self.classifier(self.head_drop(out))


# ============================================================================
# ModernMidFusion_Gated: adds SE-style conditioned gating (like V4)
# ============================================================================


class ModernMidFusion_Gated(nn.Module):
    """
    Dual-backbone mid-fusion with conditioned (SE-style) gating.

    Same as ModernMidFusion but adds per-sample gate values for each branch,
    following the MidFusionResNet_V4 pattern.

    Args:
        backbone: "convnext_tiny" or "swin_t"
        num_classes: number of output classes (7 geological formations)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        dropout: classifier dropout rate
        gate_bottleneck: hidden dim for the gate MLP
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 7,
        spectral_in_ch: int = 4,
        topo_in_ch: int = 7,
        dropout: float = 0.3,
        gate_bottleneck: int = 64,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone
        mid_ch = 384
        final_ch = 768

        # Spectral backbone
        self.spec_backbone = _build_backbone(backbone)
        if spectral_in_ch != 3:
            self.spec_backbone = _adapt_first_conv(self.spec_backbone, spectral_in_ch)

        # Topo backbone
        self.topo_backbone = _build_backbone(backbone)
        if topo_in_ch != 3:
            self.topo_backbone = _adapt_first_conv(self.topo_backbone, topo_in_ch)

        # Fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1, bias=False),
            LayerNorm2d(mid_ch),
            nn.GELU(),
        )

        # Conditioned gate MLP
        self.gate_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_mlp = nn.Sequential(
            nn.Linear(mid_ch * 2, gate_bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(gate_bottleneck, 2),
        )
        # Zero-init so gates start at sigmoid(0) = 0.5
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        # Shared final stage
        self.shared_final = nn.ModuleList(
            [self.spec_backbone.features[6], self.spec_backbone.features[7]]
        )

        # Stochastic depth for finetuning
        if drop_path_rate > 0:
            _set_drop_path_rate(self.spec_backbone, drop_path_rate)
            _set_drop_path_rate(self.topo_backbone, drop_path_rate)

        # Classifier (ConvNeXt finetuning recipe: single linear + head_init_scale)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_ch, num_classes)
        with torch.no_grad():
            self.classifier.weight.mul_(0.001)
            self.classifier.bias.mul_(0.001)

    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Swin [B, H, W, C] → [B, C, H, W]."""
        return x.permute(0, 3, 1, 2).contiguous()

    def _to_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → Swin [B, H, W, C]."""
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, spectral_in_ch, H, W]
            topo: [B, topo_in_ch, H, W]
        Returns:
            logits: [B, num_classes]
        """
        spec_feat = _forward_stages(self.spec_backbone, spectral, end_stage=2)
        topo_feat = _forward_stages(self.topo_backbone, topo, end_stage=2)

        # Ensure spatial format for fusion conv and gating
        is_swin = self.backbone_name == "swin_t"
        if is_swin:
            spec_feat = self._to_channels_first(spec_feat)
            topo_feat = self._to_channels_first(topo_feat)

        concat = torch.cat([spec_feat, topo_feat], dim=1)  # [B, 768, H', W']

        # Fusion conv
        fused = self.fusion(concat)  # [B, 384, H', W']

        # Compute per-sample gates
        pooled = self.gate_pool(concat).view(concat.size(0), -1)  # [B, 768]
        gates = torch.sigmoid(self.gate_mlp(pooled))  # [B, 2]
        gate_spec = gates[:, 0].view(-1, 1, 1, 1)
        gate_topo = gates[:, 1].view(-1, 1, 1, 1)

        # Apply conditioned residual
        fused = fused + gate_spec * spec_feat + gate_topo * topo_feat

        # Back to sequence format for Swin stage-3
        if is_swin:
            fused = self._to_channels_last(fused)

        # Shared final stage
        for layer in self.shared_final:
            fused = layer(fused)

        # Pool + classify
        if is_swin:
            fused = self._to_channels_first(fused)

        out = self.gap(fused).flatten(1)  # [B, 768]
        return self.classifier(self.head_drop(out))

    def get_gate_values(
        self, spectral: torch.Tensor, topo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return gate values for analysis."""
        with torch.no_grad():
            spec_feat = _forward_stages(self.spec_backbone, spectral, end_stage=2)
            topo_feat = _forward_stages(self.topo_backbone, topo, end_stage=2)

            if self.backbone_name == "swin_t":
                spec_feat = self._to_channels_first(spec_feat)
                topo_feat = self._to_channels_first(topo_feat)

            concat = torch.cat([spec_feat, topo_feat], dim=1)
            pooled = self.gate_pool(concat).view(concat.size(0), -1)
            gates = torch.sigmoid(self.gate_mlp(pooled))
            return gates[:, 0], gates[:, 1]

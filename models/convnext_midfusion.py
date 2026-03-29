"""
ConvNeXt-Tiny dual-backbone mid-fusion for geological classification.

Architecture:
  - Spectral backbone (ConvNeXt-Tiny, stages 0-2) → 384-ch feature maps
  - Topo backbone (ConvNeXt-Tiny, stages 0-2) → 384-ch feature maps
  - 1x1 conv fusion (768→384) with LayerNorm + GELU
  - Shared stage-3 → GAP → 2-layer MLP classifier (768→512→num_classes)

Stage dimensions: 96 → 192 → 384 → 768
"""

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.ops.stochastic_depth import StochasticDepth


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels-first tensors [B, C, H, W]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _adapt_first_conv(model, in_channels: int):
    """Replace the first conv layer for non-3-channel input."""
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
        new_conv.weight[:, :3] = old_conv.weight
        if in_channels > 3:
            mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
            for c in range(3, in_channels):
                new_conv.weight[:, c : c + 1] = mean_rgb * 0.1
    model.features[0][0] = new_conv
    return model


def _forward_stages(model, x: torch.Tensor, end_stage: int) -> torch.Tensor:
    """
    Run model.features through the given stage (inclusive).

    features[0],features[1] → stage 0 (96ch)
    features[2],features[3] → stage 1 (192ch)
    features[4],features[5] → stage 2 (384ch)
    features[6],features[7] → stage 3 (768ch)
    """
    for i in range(end_stage * 2 + 2):
        x = model.features[i](x)
    return x


def _set_drop_path_rate(backbone: nn.Module, rate: float):
    """Rescale stochastic depth to a linear schedule [0 → rate] across blocks."""
    sd_layers = [m for m in backbone.modules() if isinstance(m, StochasticDepth)]
    n = len(sd_layers)
    for i, layer in enumerate(sd_layers):
        layer.p = rate * (i + 1) / n


class ConvNeXtMidFusion(nn.Module):
    """
    Dual-backbone ConvNeXt-Tiny with mid-level fusion.

    Args:
        num_classes: number of output classes
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        topo_in_ch: topo input channels (7 or 8)
        dropout: classifier dropout rate
        drop_path_rate: stochastic depth rate (0.0 = disabled)
    """

    def __init__(
        self,
        num_classes: int = 7,
        spectral_in_ch: int = 4,
        topo_in_ch: int = 7,
        dropout: float = 0.3,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Backbones
        self.spec_backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.topo_backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        if spectral_in_ch != 3:
            _adapt_first_conv(self.spec_backbone, spectral_in_ch)
        if topo_in_ch != 3:
            _adapt_first_conv(self.topo_backbone, topo_in_ch)

        # Fusion: concat (384+384) → 1x1 conv → LN → GELU → 384
        self.fusion = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1, bias=False),
            LayerNorm2d(384),
            nn.GELU(),
        )

        # Shared final stage (stage-3 from spectral backbone)
        self.shared_final = nn.ModuleList(
            [self.spec_backbone.features[6], self.spec_backbone.features[7]]
        )

        # Stochastic depth
        if drop_path_rate > 0:
            _set_drop_path_rate(self.spec_backbone, drop_path_rate)
            _set_drop_path_rate(self.topo_backbone, drop_path_rate)

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        spec_feat = _forward_stages(self.spec_backbone, spectral, end_stage=2)
        topo_feat = _forward_stages(self.topo_backbone, topo, end_stage=2)

        fused = self.fusion(torch.cat([spec_feat, topo_feat], dim=1))

        for layer in self.shared_final:
            fused = layer(fused)

        out = self.gap(fused).flatten(1)
        return self.classifier(out)

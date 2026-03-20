"""
Dual-head ResNet50 with mid-fusion and static gating (V3).

Architecture:
  - Same dual-backbone structure as MidFusionResNet
  - Adds two learnable scalars (alpha, beta) that gate weighted branch
    residuals into the fused signal
  - Zero-initialized so the model starts near-identical to base MidFusionResNet
  - After training, inspect sigmoid(alpha) and sigmoid(beta) to see per-branch
    contribution — fully interpretable

Total added parameters over base: 2 (literally two floats)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from models.midfusion import adapt_conv1, _forward_until_layer3


class MidFusionResNet_V3(nn.Module):
    """
    Dual-backbone ResNet50 with mid-level fusion and static gating.

    fusion_output = conv1x1(concat) + sigmoid(alpha) * spec_feat
                                    + sigmoid(beta)  * topo_feat

    Args:
        num_classes: number of output classes (7 geological formations)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        dropout: classifier dropout rate
    """

    def __init__(
        self,
        num_classes: int = 7,
        topo_in_ch: int = 7,
        spectral_in_ch: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Spectral backbone
        self.spec_resnet = adapt_conv1(
            resnet50(weights=ResNet50_Weights.DEFAULT), spectral_in_ch
        )

        # Topo backbone
        self.topo_resnet = adapt_conv1(
            resnet50(weights=ResNet50_Weights.DEFAULT), topo_in_ch
        )

        # Fusion: concat layer3 outputs (1024+1024) → 1x1 conv → 1024
        self.fusion = nn.Sequential(
            nn.Conv2d(1024 * 2, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Static gate parameters
        # Raw (pre-sigmoid) values, zero-init so sigmoid gives 0.5 at start
        self.alpha = nn.Parameter(torch.zeros(1))  # spectral gate
        self.beta = nn.Parameter(torch.zeros(1))   # topo gate

        # Shared layer4 (from spectral backbone)
        self.shared_layer4 = self.spec_resnet.layer4

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, spectral: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, 4, H, W]
            topo: [B, 7-8, H, W]
        Returns:
            logits: [B, num_classes]
        """
        spec_feat = _forward_until_layer3(
            self.spec_resnet, spectral
        )  # [B, 1024, H', W']
        topo_feat = _forward_until_layer3(self.topo_resnet, topo)  # [B, 1024, H', W']

        fused = self.fusion(
            torch.cat([spec_feat, topo_feat], dim=1)
        )  # [B, 1024, H', W']

        # Add weighted residuals from both branches
        gate_spec = torch.sigmoid(self.alpha)  # scalar, same for all patches
        gate_topo = torch.sigmoid(self.beta)   # scalar, same for all patches
        fused = fused + gate_spec * spec_feat + gate_topo * topo_feat

        out = self.shared_layer4(fused)  # [B, 2048, H'', W'']
        out = self.gap(out).flatten(1)  # [B, 2048]
        return self.classifier(out)  # [B, num_classes]

    def get_gate_values(self) -> dict[str, float]:
        """Call after training to inspect learned gate weights."""
        return {
            "spectral_gate": torch.sigmoid(self.alpha).item(),
            "topo_gate": torch.sigmoid(self.beta).item(),
        }

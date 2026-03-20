"""
Dual-head ResNet50 with mid-fusion and conditioned gating — SE-style (V4).

Architecture:
  - Same dual-backbone structure as MidFusionResNet
  - Adds a small MLP that looks at pooled concatenated features and produces
    per-sample gate values for each branch
  - The model learns *when* each modality matters, not just *how much* on average
  - Gate MLP zero-initialized so both branches start contributing equally

Total added parameters: ~2048 * bottleneck + bottleneck * 2
With bottleneck=64: ~133K params (tiny vs ResNet50's ~23M per branch)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from models.midfusion import adapt_conv1, _forward_until_layer3


class MidFusionResNet_V4(nn.Module):
    """
    Dual-backbone ResNet50 with mid-level fusion and conditioned (SE-style) gating.

    1. Concat spec_feat + topo_feat → [B, 2048, H', W']
    2. Global avg pool → [B, 2048]
    3. MLP: 2048 → bottleneck → 2 → sigmoid → (gate_spec, gate_topo) per sample
    4. fusion = conv1x1(concat) + gate_spec * spec_feat + gate_topo * topo_feat

    Args:
        num_classes: number of output classes (7 geological formations)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        dropout: classifier dropout rate
        gate_bottleneck: hidden dim for the gate MLP
    """

    def __init__(
        self,
        num_classes: int = 7,
        topo_in_ch: int = 7,
        spectral_in_ch: int = 4,
        dropout: float = 0.3,
        gate_bottleneck: int = 64,
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

        # Conditioned gate MLP
        # Squeeze: global avg pool reduces [B, 2048, H', W'] → [B, 2048]
        # Excitation: MLP maps 2048 → bottleneck → 2 gate values
        self.gate_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_mlp = nn.Sequential(
            nn.Linear(2048, gate_bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(gate_bottleneck, 2),
        )

        # Initialize gate MLP so outputs are near 0 → sigmoid(0) = 0.5
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

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

        # Concat for both fusion conv and gate input
        concat = torch.cat([spec_feat, topo_feat], dim=1)  # [B, 2048, H', W']

        # Fusion conv
        fused = self.fusion(concat)  # [B, 1024, H', W']

        # Compute per-sample gate values
        pooled = self.gate_pool(concat).view(concat.size(0), -1)  # [B, 2048]
        gates = torch.sigmoid(self.gate_mlp(pooled))              # [B, 2]
        gate_spec = gates[:, 0].view(-1, 1, 1, 1)                # [B, 1, 1, 1]
        gate_topo = gates[:, 1].view(-1, 1, 1, 1)                # [B, 1, 1, 1]

        # Apply conditioned residual
        fused = fused + gate_spec * spec_feat + gate_topo * topo_feat

        out = self.shared_layer4(fused)  # [B, 2048, H'', W'']
        out = self.gap(out).flatten(1)  # [B, 2048]
        return self.classifier(out)  # [B, num_classes]

    def get_gate_values(
        self, spectral: torch.Tensor, topo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass and return gate values for analysis.
        Returns: (gate_spec, gate_topo) each of shape [B]
        """
        with torch.no_grad():
            spec_feat = _forward_until_layer3(self.spec_resnet, spectral)
            topo_feat = _forward_until_layer3(self.topo_resnet, topo)
            concat = torch.cat([spec_feat, topo_feat], dim=1)
            pooled = self.gate_pool(concat).view(concat.size(0), -1)
            gates = torch.sigmoid(self.gate_mlp(pooled))
            return gates[:, 0], gates[:, 1]

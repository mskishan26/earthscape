"""
Dual-head ResNet50 with mid-fusion for multimodal geological classification.

Architecture:
  - Spectral backbone (ResNet50, 4ch: R/G/B/NIR) → conv1..layer3
  - Topo backbone (ResNet50, 7-8ch: DEM/slopes/etc) → conv1..layer3
  - 1x1 conv fusion at layer3 output (2048→1024)
  - Shared layer4 → GAP → classifier
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, resnet50


def adapt_conv1(
    resnet: torchvision.models.ResNet, in_channels: int
) -> torchvision.models.ResNet:
    """
    Replace resnet.conv1 to accept in_channels instead of 3.
    Copies pretrained weights for first 3 channels; initializes extras
    as scaled mean of RGB weights.
    """
    old = resnet.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        if in_channels >= 3:
            new_conv.weight[:, :3] = old.weight
            if in_channels > 3:
                mean_rgb = old.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c : c + 1] = mean_rgb * 0.1
        else:
            new_conv.weight[:, :in_channels] = old.weight[:, :in_channels]
    resnet.conv1 = new_conv
    return resnet


def _forward_until_layer3(
    resnet: torchvision.models.ResNet, x: torch.Tensor
) -> torch.Tensor:
    """Run ResNet forward pass up to and including layer3."""
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    return x  # [B, 1024, H', W']


class MidFusionResNet(nn.Module):
    """
    Dual-backbone ResNet50 with mid-level feature fusion.

    Args:
        num_classes: number of output classes (7 geological formations)
        topo_in_ch: topo input channels (7 without NHD, 8 with)
        spectral_in_ch: spectral input channels (4: R,G,B,NIR)
        pretrained: use ImageNet pretrained weights
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

        # Shared layer4 (from spectral backbone)
        # layer4 expects 1024 input channels → outputs 2048
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

        out = self.shared_layer4(fused)  # [B, 2048, H'', W'']
        out = self.gap(out).flatten(1)  # [B, 2048]
        return self.classifier(out)  # [B, num_classes]

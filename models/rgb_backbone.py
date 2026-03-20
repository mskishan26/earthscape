"""
RGB-input backbone models for geological classification.

Wraps standard torchvision models (ResNet50, EfficientNet-B0) with a
classification head matching the EarthScape output format.

These models take a single RGB (or N-channel) image as input.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


class RGBBackbone(nn.Module):
    """
    Single-backbone image classifier for geological formations.

    Supports:
      - resnet50: ImageNet-pretrained ResNet-50
      - efficientnet_b0: ImageNet-pretrained EfficientNet-B0

    Args:
        backbone: architecture name ("resnet50" or "efficientnet_b0")
        num_classes: number of output classes (7 geological formations)
        in_channels: input channels (3 for RGB, can be adapted)
        dropout: classifier dropout rate
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 7,
        in_channels: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Adapt conv1 if not 3 channels
            if in_channels != 3:
                base = self._adapt_resnet_conv1(base, in_channels)
            feat_dim = base.fc.in_features  # 2048
            base.fc = nn.Identity()
            self.backbone = base
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self._use_gap = False  # ResNet already has avgpool

        elif backbone == "efficientnet_b0":
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Adapt first conv if not 3 channels
            if in_channels != 3:
                base = self._adapt_effnet_conv1(base, in_channels)
            feat_dim = base.classifier[1].in_features  # 1280
            base.classifier = nn.Identity()
            self.backbone = base
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self._use_gap = False  # EfficientNet already has avgpool

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classifier head (matches MidFusion style)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _adapt_resnet_conv1(resnet, in_channels: int):
        """Replace ResNet conv1 for non-3-channel input."""
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

    @staticmethod
    def _adapt_effnet_conv1(effnet, in_channels: int):
        """Replace EfficientNet first conv for non-3-channel input."""
        old_conv = effnet.features[0][0]
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
        effnet.features[0][0] = new_conv
        return effnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            logits: [B, num_classes]
        """
        features = self.backbone(x)  # [B, feat_dim]
        if features.dim() == 4:
            features = self.gap(features).flatten(1)
        return self.classifier(features)

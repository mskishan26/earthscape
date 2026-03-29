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
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,  # remove classification head, get features only
            in_chans=in_channels,
            img_size=img_size,
        )
        feat_dim = self.backbone.embed_dim  # 384 for DeiT-Small

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            logits: [B, num_classes]
        """
        features = self.backbone(x)  # [B, 384]
        return self.classifier(features)


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
    ):
        super().__init__()
        total_ch = spectral_in_ch + topo_in_ch

        self.backbone = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=total_ch,
            img_size=img_size,
        )
        feat_dim = self.backbone.embed_dim  # 384 for DeiT-Small

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

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
        return self.classifier(features)


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
    ):
        super().__init__()

        # Spectral encoder
        self.spec_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=spectral_in_ch,
            img_size=img_size,
        )

        # Topo encoder
        self.topo_encoder = timm.create_model(
            "deit3_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=topo_in_ch,
            img_size=img_size,
        )

        feat_dim = self.spec_encoder.embed_dim  # 384

        # Classifier on concatenated features (384 + 384 = 768)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

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
        return self.classifier(combined)

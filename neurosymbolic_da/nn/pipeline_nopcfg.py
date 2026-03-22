"""Ablation: Backbone + Concept Bottleneck + Linear classifier (no PCFG).

Replaces the PCFG scoring head with a linear layer on pooled
bottleneck heatmap features. Used in Experiment 4 (Section 6.6)
to isolate the contribution of structural programs.
"""

import torch
import torch.nn as nn
from torch import Tensor

from neurosymbolic_da.nn.backbone import LeNetBackbone, ResNetBackbone
from neurosymbolic_da.nn.bottleneck import ConceptBottleneck


class NoPCFGPipeline(nn.Module):
    """Backbone + Concept Bottleneck + Linear classifier (no grammar).

    The bottleneck still produces k heatmaps, but instead of feeding
    them to a PCFG, we pool the heatmap features and classify with
    a linear layer.

    Args:
        n_primitives: number of primitive types (k)
        n_classes: number of output classes
        backbone_variant: "resnet18" or "resnet50"
        pretrained_backbone: use ImageNet pretrained weights
    """

    def __init__(
        self,
        n_primitives: int,
        n_classes: int,
        backbone_variant: str = "resnet18",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes

        if backbone_variant == "lenet":
            self.backbone = LeNetBackbone()
        else:
            self.backbone = ResNetBackbone(backbone_variant, pretrained_backbone)
        self.bottleneck = ConceptBottleneck(self.backbone.out_channels, n_primitives)

        # Pool heatmaps → feature vector → linear classifier
        # Features: k coords (x,y) + k confidences = 3k
        self.classifier = nn.Linear(3 * n_primitives, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class scores for a batch of images.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            log_probs: log-softmax class scores [B, n_classes]
        """
        features = self.backbone(x)  # [B, C, H, W]
        heatmaps, coords, _ = self.bottleneck(features)

        # Build feature vector from bottleneck outputs
        B, k, H, W = heatmaps.shape
        conf = torch.sigmoid(heatmaps.view(B, k, -1).max(dim=-1).values)  # [B, k]
        # coords is [B, k, 2], flatten to [B, 2k]
        coords_flat = coords.view(B, -1)  # [B, 2k]
        # Concatenate: [B, 3k]
        bottleneck_features = torch.cat([coords_flat, conf], dim=-1)

        logits = self.classifier(bottleneck_features)  # [B, n_classes]
        return torch.log_softmax(logits, dim=-1)

    def get_heatmaps(self, x: Tensor) -> Tensor:
        """Extract heatmaps only (for MMD alignment during adaptation).

        Args:
            x: input images [B, 3, H, W]

        Returns:
            heatmaps: [B, k, H, W]
        """
        features = self.backbone(x)
        heatmaps, _, _ = self.bottleneck(features)
        return heatmaps

    def get_backbone_features(self, x: Tensor) -> Tensor:
        """Extract GAP-pooled backbone features (for CDAN alignment).

        Args:
            x: input images [B, 3, H, W]

        Returns:
            features: [B, C] — global average pooled backbone features
        """
        features = self.backbone(x)  # [B, C, H, W]
        return features.mean(dim=[2, 3])  # [B, C]

    def get_bottleneck_features(self, x: Tensor) -> Tensor:
        """Extract compact bottleneck features (for MMD alignment).

        Args:
            x: input images [B, 3, H, W]

        Returns:
            features: [B, k*3] — (cx, cy, conf) per primitive
        """
        features = self.backbone(x)
        heatmaps, coords, _ = self.bottleneck(features)
        B = x.shape[0]
        conf = torch.sigmoid(
            heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values
        )  # [B, k]
        flat_coords = coords.view(B, -1)  # [B, k*2]
        return torch.cat([flat_coords, conf], dim=-1)  # [B, k*3]

"""Backbone feature extractors.

Wraps torchvision models to return spatial feature tensors [B, C, H, W]
from the last convolutional layer (before global pooling).
"""

import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class ResNetBackbone(nn.Module):
    """ResNet backbone that returns spatial features.

    Args:
        variant: "resnet18" or "resnet50"
        pretrained: use ImageNet pretrained weights
        multiscale: if True, return dict of {layer_name: features} for FPN
    """

    # Output channels for each variant's layer4
    _out_channels = {"resnet18": 512, "resnet50": 2048}
    # Channels per layer (for multiscale)
    _layer_channels = {
        "resnet18": {"layer2": 128, "layer3": 256, "layer4": 512},
        "resnet50": {"layer2": 512, "layer3": 1024, "layer4": 2048},
    }

    def __init__(self, variant: str = "resnet18", pretrained: bool = True,
                 multiscale: bool = False):
        super().__init__()
        self.multiscale = multiscale
        if variant == "resnet18":
            weights = "IMAGENET1K_V1" if pretrained else None
            base = resnet18(weights=weights)
        elif variant == "resnet50":
            weights = "IMAGENET1K_V2" if pretrained else None
            base = resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        if multiscale:
            self.feat_extractor = create_feature_extractor(
                base, return_nodes={
                    "layer2": "layer2",
                    "layer3": "layer3",
                    "layer4": "layer4",
                }
            )
            self.layer_channels = self._layer_channels[variant]
        else:
            self.feat_extractor = create_feature_extractor(
                base, return_nodes={"layer4": "features"}
            )
        self.out_channels = self._out_channels[variant]
        self.variant = variant

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        """Extract spatial features.

        If multiscale=False: [B, 3, H, W] -> [B, C, 7, 7]
        If multiscale=True: [B, 3, H, W] -> {"layer2": [B,C2,28,28], "layer3": [B,C3,14,14], "layer4": [B,C4,7,7]}
        """
        out = self.feat_extractor(x)
        if self.multiscale:
            return out
        return out["features"]


class LeNetBackbone(nn.Module):
    """LeNet-style backbone returning spatial features [B, C, H, W].

    Standard architecture used in DA benchmarks (DANN, ADDA, etc.).
    Three conv layers with max pooling, outputting spatial feature maps.

    Input: [B, 3, 28, 28] or [B, 3, 32, 32] (NOT 224x224).
    For fair comparison with published DA results.
    """

    def __init__(self):
        super().__init__()
        self.out_channels = 48
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 48, kernel_size=5),
            nn.ReLU(),
            # No final pool — keep spatial dims for bottleneck
        )

    def forward(self, x: Tensor) -> Tensor:
        """Extract spatial features. Input: [B, 3, H, W] → Output: [B, 48, H', W']."""
        return self.features(x)

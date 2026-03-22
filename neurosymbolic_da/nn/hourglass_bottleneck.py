"""Hourglass/Multiscale Concept Bottleneck.

Fuses multi-scale backbone features (layer2/3/4) via a lightweight FPN-like
neck, then produces k heatmaps at the finest resolution. This gives each
primitive's heatmap access to both fine-grained texture (layer2, 28x28) and
high-level semantics (layer4, 7x7).

The grammar and pipeline are unchanged — this is a drop-in bottleneck
replacement that improves primitive detection quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import spatial_soft_argmax2d
from torch import Tensor

from neurosymbolic_da.dsl.primitives import Env, Primitive


class HourglassBottleneck(nn.Module):
    """Multi-scale FPN bottleneck for primitive detection.

    Takes multi-scale features from ResNet (layer2/3/4) and fuses them
    via top-down pathway with lateral connections (FPN-style). Produces
    k primitive heatmaps at 14x14 resolution (layer3 scale).

    Args:
        layer_channels: dict mapping layer name to channel count
            e.g. {"layer2": 512, "layer3": 1024, "layer4": 2048} for ResNet-50
        n_primitives: number of primitive types (k)
        fpn_channels: internal FPN channel dimension
        output_scale: which scale to produce heatmaps at ("layer3" = 14x14)
        temperature: softmax temperature for soft-argmax
    """

    def __init__(
        self,
        layer_channels: dict[str, int],
        n_primitives: int,
        fpn_channels: int = 256,
        output_scale: str = "layer3",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.output_scale = output_scale

        # Lateral connections: reduce each layer to fpn_channels
        self.lateral_layer4 = nn.Conv2d(layer_channels["layer4"], fpn_channels, 1)
        self.lateral_layer3 = nn.Conv2d(layer_channels["layer3"], fpn_channels, 1)
        self.lateral_layer2 = nn.Conv2d(layer_channels["layer2"], fpn_channels, 1)

        # Top-down smoothing after merge
        self.smooth_layer3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_layer2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

        # Merge all scales into output scale (concat + reduce)
        self.merge_conv = nn.Conv2d(fpn_channels * 3, fpn_channels, 1)

        # Final heatmap head on fused features
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, n_primitives, 1),
        )

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, multiscale_features: dict[str, Tensor]) -> tuple[Tensor, Tensor, Env]:
        """Extract primitives from multi-scale features.

        Args:
            multiscale_features: dict with keys "layer2", "layer3", "layer4"

        Returns:
            heatmaps: [B, k, H, W] at output_scale resolution
            coords: [B, k, 2] normalized coordinates in [-1, 1]
            env: Env dict for single image
        """
        f2 = multiscale_features["layer2"]  # [B, C2, 28, 28]
        f3 = multiscale_features["layer3"]  # [B, C3, 14, 14]
        f4 = multiscale_features["layer4"]  # [B, C4, 7, 7]

        # Lateral projections
        p4 = self.lateral_layer4(f4)  # [B, fpn, 7, 7]
        p3 = self.lateral_layer3(f3)  # [B, fpn, 14, 14]
        p2 = self.lateral_layer2(f2)  # [B, fpn, 28, 28]

        # Top-down pathway: upsample + add + smooth
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode="nearest")
        p3 = self.smooth_layer3(p3)

        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode="nearest")
        p2 = self.smooth_layer2(p2)

        # Determine output spatial size based on output_scale
        if self.output_scale == "layer2":
            out_size = p2.shape[2:]
        elif self.output_scale == "layer3":
            out_size = p3.shape[2:]
        else:
            out_size = p4.shape[2:]

        # Resize all scales to output resolution and concatenate
        p4_up = F.interpolate(p4, size=out_size, mode="nearest")
        p3_rs = F.interpolate(p3, size=out_size, mode="nearest") if p3.shape[2:] != out_size else p3
        p2_rs = F.interpolate(p2, size=out_size, mode="bilinear", align_corners=False) if p2.shape[2:] != out_size else p2

        fused = self.merge_conv(torch.cat([p2_rs, p3_rs, p4_up], dim=1))

        # Generate heatmaps
        heatmaps = self.heatmap_head(fused)  # [B, k, H, W]

        # Coordinates via soft-argmax
        coords = spatial_soft_argmax2d(
            heatmaps,
            temperature=self.temperature,
            normalized_coordinates=True,
        )  # [B, k, 2]

        # Confidence
        B, k, H, W = heatmaps.shape
        conf = torch.sigmoid(
            heatmaps.view(B, k, -1).max(dim=-1).values
        )  # [B, k]

        env = self.extract_env(coords[0], conf[0], heatmaps[0])
        return heatmaps, coords, env

    def extract_env(self, coords: Tensor, conf: Tensor, heatmaps: Tensor) -> Env:
        """Build Env dict from extracted primitives for a single image."""
        env: Env = {}
        k = coords.shape[0]
        bbox_half = self._estimate_bbox_half_sizes(heatmaps)

        for j in range(k):
            cx, cy = coords[j, 0], coords[j, 1]
            hx, hy = bbox_half[j, 0], bbox_half[j, 1]
            env[j] = Primitive(
                cx=cx, cy=cy,
                x1=cx - hx, y1=cy - hy,
                x2=cx + hx, y2=cy + hy,
                conf=conf[j],
                type_idx=j,
            )
        return env

    def extract_batched_env(
        self, coords: Tensor, conf: Tensor, heatmaps: Tensor
    ) -> Env:
        """Build a batched Env dict — Primitive fields have shape [B]."""
        env: Env = {}
        k = coords.shape[1]
        bbox_half = self._estimate_batched_bbox_half_sizes(heatmaps)

        for j in range(k):
            cx = coords[:, j, 0]
            cy = coords[:, j, 1]
            hx = bbox_half[:, j, 0]
            hy = bbox_half[:, j, 1]
            env[j] = Primitive(
                cx=cx, cy=cy,
                x1=cx - hx, y1=cy - hy,
                x2=cx + hx, y2=cy + hy,
                conf=conf[:, j],
                type_idx=j,
            )
        return env

    def _estimate_bbox_half_sizes(self, heatmaps: Tensor) -> Tensor:
        """Estimate bbox half-sizes from heatmap spread. [k, H, W] -> [k, 2]"""
        k, H, W = heatmaps.shape
        flat = heatmaps.view(k, -1)
        weights = torch.softmax(flat / self.temperature, dim=-1).view(k, H, W)

        gy = torch.linspace(-1, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        gx = torch.linspace(-1, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        mean_x = (weights * grid_x).sum(dim=(-2, -1))
        mean_y = (weights * grid_y).sum(dim=(-2, -1))
        var_x = (weights * (grid_x - mean_x[:, None, None]) ** 2).sum(dim=(-2, -1))
        var_y = (weights * (grid_y - mean_y[:, None, None]) ** 2).sum(dim=(-2, -1))

        half_w = 2.0 * torch.sqrt(var_x + 1e-6)
        half_h = 2.0 * torch.sqrt(var_y + 1e-6)
        return torch.stack([half_w, half_h], dim=-1)

    def _estimate_batched_bbox_half_sizes(self, heatmaps: Tensor) -> Tensor:
        """Estimate bbox half-sizes for a batch. [B, k, H, W] -> [B, k, 2]"""
        B, k, H, W = heatmaps.shape
        flat = heatmaps.view(B, k, -1)
        weights = torch.softmax(flat / self.temperature, dim=-1).view(B, k, H, W)

        gy = torch.linspace(-1, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        gx = torch.linspace(-1, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        mean_x = (weights * grid_x).sum(dim=(-2, -1))
        mean_y = (weights * grid_y).sum(dim=(-2, -1))
        var_x = (weights * (grid_x - mean_x[:, :, None, None]) ** 2).sum(dim=(-2, -1))
        var_y = (weights * (grid_y - mean_y[:, :, None, None]) ** 2).sum(dim=(-2, -1))

        half_w = 2.0 * torch.sqrt(var_x + 1e-6)
        half_h = 2.0 * torch.sqrt(var_y + 1e-6)
        return torch.stack([half_w, half_h], dim=-1)

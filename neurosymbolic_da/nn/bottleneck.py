"""Concept Bottleneck: maps backbone features to k spatial heatmaps,
then extracts primitives via differentiable soft-argmax (Section 3.2).

Uses kornia's spatial_soft_argmax2d (DSNT) for coordinate extraction.
"""

import torch
import torch.nn as nn
from kornia.geometry.subpix import spatial_soft_argmax2d
from torch import Tensor

from neurosymbolic_da.dsl.primitives import Env, Primitive


class MoEBottleneck(nn.Module):
    """MoE-style concept bottleneck with spatial routing.

    Instead of a single 1x1 conv producing k heatmaps (which can collapse
    to all-same patterns), uses a router + value decomposition inspired by
    GMoE (Li et al., ICLR 2023). Each spatial location is softly assigned
    to one primitive via softmax over k (competition). Load balancing loss
    ensures all primitives get equal spatial coverage.

    Args:
        in_channels: backbone output channels
        n_primitives: number of primitive types (k)
        temperature: softmax temperature for soft-argmax
    """

    def __init__(self, in_channels: int, n_primitives: int, temperature: float = 1.0):
        super().__init__()
        self.n_primitives = n_primitives

        # Router: assigns each spatial location to a primitive (softmax over k)
        self.router_conv = nn.Conv2d(in_channels, n_primitives, kernel_size=1)
        # Value: each primitive's activation strength (independent of routing)
        self.value_conv = nn.Conv2d(in_channels, n_primitives, kernel_size=1)

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.load_balance_loss = torch.tensor(0.0)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, "Env"]:
        """Extract primitives with MoE-style spatial routing.

        Args:
            features: backbone output [B, C, H, W]

        Returns:
            heatmaps: routed heatmaps [B, k, H, W]
            coords: normalized coordinates [B, k, 2] in [-1, 1]
            env: Env dict for single image
        """
        B, C, H, W = features.shape
        k = self.n_primitives

        # Router: competitive assignment — each location goes to ~1 primitive
        routing_logits = self.router_conv(features)  # [B, k, H, W]
        routing_probs = torch.softmax(routing_logits, dim=1)  # [B, k, H, W]

        # Value: primitive activation strength
        values = self.value_conv(features)  # [B, k, H, W]

        # Gate values by routing — each primitive only "sees" its assigned locations
        heatmaps = values * routing_probs  # [B, k, H, W]

        # Load balance loss: penalize deviation from uniform 1/k assignment
        f = routing_probs.mean(dim=[0, 2, 3])  # [k]
        self.load_balance_loss = k * ((f - 1.0 / k) ** 2).sum()

        # Coordinates via soft-argmax
        coords = spatial_soft_argmax2d(
            heatmaps,
            temperature=self.temperature,
            normalized_coordinates=True,
        )  # [B, k, 2]

        # Confidence
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


class ConceptBottleneck(nn.Module):
    """Maps spatial features to k primitive heatmaps and extracts primitives.

    Args:
        in_channels: backbone output channels (e.g. 512 for ResNet18)
        n_primitives: number of primitive types (k)
        temperature: softmax temperature for soft-argmax (learnable)
    """

    def __init__(self, in_channels: int, n_primitives: int, temperature: float = 1.0):
        super().__init__()
        self.n_primitives = n_primitives

        # 1x1 conv projects backbone features to k heatmaps
        self.heatmap_conv = nn.Conv2d(in_channels, n_primitives, kernel_size=1)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Env]:
        """Extract primitives from spatial features.

        Args:
            features: backbone output [B, C, H, W]

        Returns:
            heatmaps: raw heatmaps [B, k, H, W] (for MMD alignment in adaptation)
            coords: normalized coordinates [B, k, 2] in [-1, 1]
            env: Env dict for a single image (type_idx -> Primitive)
                 Only returned for B=1; for batched use, call extract_env() per image.
        """
        heatmaps = self.heatmap_conv(features)  # [B, k, H, W]

        # Differentiable soft-argmax: heatmap -> (x, y) coordinates
        # kornia returns [B, k, 2] with coords in [-1, 1] (normalized)
        coords = spatial_soft_argmax2d(
            heatmaps,
            temperature=self.temperature,
            normalized_coordinates=True,
        )  # [B, k, 2]

        # Confidence: max activation per heatmap channel
        B, k, H, W = heatmaps.shape
        conf = heatmaps.view(B, k, -1).max(dim=-1).values  # [B, k]
        # Clamp to non-negative (sigmoid would also work)
        conf = torch.sigmoid(conf)

        # Build Env for single-image case
        env = self.extract_env(coords[0], conf[0], heatmaps[0])

        return heatmaps, coords, env

    def extract_env(self, coords: Tensor, conf: Tensor, heatmaps: Tensor) -> Env:
        """Build an Env dict from extracted primitives for a single image.

        Args:
            coords: [k, 2] normalized coordinates (x, y) in [-1, 1]
            conf: [k] confidence scores
            heatmaps: [k, H, W] heatmaps (for bbox estimation)
        """
        env: Env = {}
        k = coords.shape[0]
        bbox_half = self._estimate_bbox_half_sizes(heatmaps)  # [k, 2]

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
        """Build a batched Env dict — Primitive fields have shape [B].

        Args:
            coords: [B, k, 2] normalized coordinates
            conf: [B, k] confidence scores
            heatmaps: [B, k, H, W] heatmaps
        """
        env: Env = {}
        k = coords.shape[1]
        bbox_half = self._estimate_batched_bbox_half_sizes(heatmaps)  # [B, k, 2]

        for j in range(k):
            cx = coords[:, j, 0]  # [B]
            cy = coords[:, j, 1]  # [B]
            hx = bbox_half[:, j, 0]  # [B]
            hy = bbox_half[:, j, 1]  # [B]
            env[j] = Primitive(
                cx=cx, cy=cy,
                x1=cx - hx, y1=cy - hy,
                x2=cx + hx, y2=cy + hy,
                conf=conf[:, j],  # [B]
                type_idx=j,
            )
        return env

    def _estimate_bbox_half_sizes(self, heatmaps: Tensor) -> Tensor:
        """Estimate bounding box half-sizes from heatmap spread.

        Uses the spatial variance of the softmax-normalized heatmap
        as a differentiable proxy for bounding box extent.

        Args:
            heatmaps: [k, H, W]

        Returns:
            half_sizes: [k, 2] (half-width, half-height) in normalized coords
        """
        k, H, W = heatmaps.shape

        # Softmax over spatial dims
        flat = heatmaps.view(k, -1)  # [k, H*W]
        weights = torch.softmax(flat / self.temperature, dim=-1)  # [k, H*W]
        weights = weights.view(k, H, W)

        # Create normalized coordinate grids in [-1, 1]
        gy = torch.linspace(-1, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        gx = torch.linspace(-1, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # [H, W] each

        # Weighted mean (should be close to soft-argmax coords)
        mean_x = (weights * grid_x).sum(dim=(-2, -1))  # [k]
        mean_y = (weights * grid_y).sum(dim=(-2, -1))  # [k]

        # Weighted variance → standard deviation as half-size proxy
        var_x = (weights * (grid_x - mean_x[:, None, None]) ** 2).sum(dim=(-2, -1))
        var_y = (weights * (grid_y - mean_y[:, None, None]) ** 2).sum(dim=(-2, -1))

        # 2 * std as bbox half-size (covers ~95% of mass)
        half_w = 2.0 * torch.sqrt(var_x + 1e-6)
        half_h = 2.0 * torch.sqrt(var_y + 1e-6)

        return torch.stack([half_w, half_h], dim=-1)  # [k, 2]

    def _estimate_batched_bbox_half_sizes(self, heatmaps: Tensor) -> Tensor:
        """Estimate bbox half-sizes for a batch of heatmaps.

        Args:
            heatmaps: [B, k, H, W]

        Returns:
            half_sizes: [B, k, 2]
        """
        B, k, H, W = heatmaps.shape

        flat = heatmaps.view(B, k, -1)  # [B, k, H*W]
        weights = torch.softmax(flat / self.temperature, dim=-1)  # [B, k, H*W]
        weights = weights.view(B, k, H, W)

        gy = torch.linspace(-1, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        gx = torch.linspace(-1, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # [H, W]

        mean_x = (weights * grid_x).sum(dim=(-2, -1))  # [B, k]
        mean_y = (weights * grid_y).sum(dim=(-2, -1))  # [B, k]

        var_x = (weights * (grid_x - mean_x[:, :, None, None]) ** 2).sum(dim=(-2, -1))
        var_y = (weights * (grid_y - mean_y[:, :, None, None]) ** 2).sum(dim=(-2, -1))

        half_w = 2.0 * torch.sqrt(var_x + 1e-6)
        half_h = 2.0 * torch.sqrt(var_y + 1e-6)

        return torch.stack([half_w, half_h], dim=-1)  # [B, k, 2]

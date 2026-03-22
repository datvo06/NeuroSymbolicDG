"""Slot Attention Bottleneck: discovers object parts via iterative
competitive attention (Locatello et al., 2020).

Unlike ConceptBottleneck (1x1 conv → k fixed heatmaps), slots compete
for spatial feature patches, naturally discovering distinct parts.
Each slot outputs spatial attention map (heatmap), coordinates, and
confidence — same interface as ConceptBottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import spatial_soft_argmax2d
from torch import Tensor

from neurosymbolic_da.dsl.primitives import Env, Primitive


class SlotAttention(nn.Module):
    """Slot Attention module (Locatello et al., 2020).

    Iteratively refines K slots via cross-attention over spatial features.
    Slots compete for input patches via softmax over slots (not over inputs),
    ensuring each patch is explained by exactly one slot.

    Args:
        n_slots: number of slots (= number of primitives k)
        input_dim: dimension of input features (backbone channels)
        slot_dim: dimension of slot representations
        n_iters: number of attention iterations
        hidden_dim: MLP hidden dimension for slot update
    """

    def __init__(
        self,
        n_slots: int,
        input_dim: int,
        slot_dim: int = 64,
        n_iters: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters

        # Learnable slot initialization (shared mean + learned per-slot offset)
        self.slot_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, slot_dim))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.input_norm = nn.LayerNorm(slot_dim)

        # Attention: Q from slots, K/V from inputs
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # Slot update via GRU + residual MLP
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.slot_norm = nn.LayerNorm(slot_dim)
        self.mlp_norm = nn.LayerNorm(slot_dim)

        self._scale = slot_dim ** -0.5

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Run slot attention.

        Args:
            features: [B, N, D_in] spatial features (N = H*W)

        Returns:
            slots: [B, K, D_slot] refined slot representations
            attn_maps: [B, K, N] attention weights (sum to 1 over slots per position)
        """
        B, N, _ = features.shape

        # Project inputs
        inputs = self.input_norm(self.input_proj(features))  # [B, N, D_slot]

        # Initialize slots (learnable mean + noise for diversity)
        slots = self.slot_mu.expand(B, -1, -1)
        if self.training:
            slots = slots + torch.exp(self.slot_log_sigma) * torch.randn_like(slots)

        # Pre-compute keys and values (don't change across iterations)
        k = self.to_k(inputs)  # [B, N, D_slot]
        v = self.to_v(inputs)  # [B, N, D_slot]

        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.slot_norm(slots)

            # Attention: queries from slots
            q = self.to_q(slots)  # [B, K, D_slot]

            # Dot-product attention: [B, K, N]
            dots = torch.einsum('bkd,bnd->bkn', q, k) * self._scale

            # Normalize over SLOTS (not over inputs) — competition between slots
            attn = F.softmax(dots, dim=1)  # [B, K, N] — sums to 1 over K

            # Weighted mean of values per slot
            # Normalize attention per slot to get proper weighted average
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K, N]
            updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)  # [B, K, D_slot]

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.n_slots, -1),
                slots_prev.reshape(B * self.n_slots, -1),
            ).reshape(B, self.n_slots, -1)

            # Residual MLP
            slots = slots + self.mlp(self.mlp_norm(slots))

        # Final attention maps for output
        q = self.to_q(self.slot_norm(slots))
        dots = torch.einsum('bkd,bnd->bkn', q, k) * self._scale
        attn_maps = F.softmax(dots, dim=1)  # [B, K, N]

        return slots, attn_maps


class SlotAttentionBottleneck(nn.Module):
    """Slot Attention bottleneck — drop-in replacement for ConceptBottleneck.

    Uses Slot Attention to discover parts via competitive attention.
    Produces the same interface: heatmaps [B,k,H,W], coords [B,k,2], env.

    Args:
        in_channels: backbone output channels
        n_primitives: number of slots/primitives (k)
        slot_dim: slot representation dimension
        n_iters: number of slot attention iterations
    """

    def __init__(
        self,
        in_channels: int,
        n_primitives: int,
        slot_dim: int = 64,
        n_iters: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_primitives = n_primitives

        self.slot_attention = SlotAttention(
            n_slots=n_primitives,
            input_dim=in_channels,
            slot_dim=slot_dim,
            n_iters=n_iters,
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Env]:
        """Extract primitives via slot attention.

        Args:
            features: backbone output [B, C, H, W]

        Returns:
            heatmaps: attention maps [B, k, H, W]
            coords: normalized coordinates [B, k, 2] in [-1, 1]
            env: Env dict (for single image, B=1)
        """
        B, C, H, W = features.shape

        # Flatten spatial: [B, H*W, C]
        flat_features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Run slot attention
        slots, attn_maps = self.slot_attention(flat_features)
        # attn_maps: [B, K, H*W]

        # Reshape attention maps to spatial heatmaps
        heatmaps = attn_maps.view(B, self.n_primitives, H, W)  # [B, k, H, W]

        # Extract coordinates via soft-argmax on attention maps
        # Scale heatmaps for soft-argmax (attention values are small)
        scaled_heatmaps = heatmaps * 10.0  # amplify for sharper soft-argmax
        coords = spatial_soft_argmax2d(
            scaled_heatmaps,
            temperature=self.temperature,
            normalized_coordinates=True,
        )  # [B, k, 2]

        # Confidence: max attention per slot
        conf = attn_maps.max(dim=-1).values  # [B, k]
        conf = torch.sigmoid(conf * 10.0)  # scale and normalize

        # Build Env for single-image case
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
        """Estimate bbox half-sizes from attention map spread.

        Args:
            heatmaps: [k, H, W]

        Returns:
            half_sizes: [k, 2]
        """
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
        """Estimate bbox half-sizes for a batch.

        Args:
            heatmaps: [B, k, H, W]

        Returns:
            half_sizes: [B, k, 2]
        """
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

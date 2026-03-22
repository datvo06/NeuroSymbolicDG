"""Spatial relation functions (pure PyTorch, no effects).

Each relation takes two Primitives and a RelationParams module,
returning a soft score in [0, 1].

Supports optional coordinate transforms for invariance:
- normalize_coords: scale invariance (coords normalized by primitive spread)
- canonicalize_coords: rotation invariance (PCA aligns principal axis to x-axis)
"""

import torch
import torch.nn as nn
from torch import Tensor

from neurosymbolic_da.dsl.primitives import Primitive

RELATION_NAMES = ("above", "left_of", "aligned_h", "aligned_v", "near", "contains")

# Extended set including rotation-invariant relations
RELATION_NAMES_INVARIANT = RELATION_NAMES + ("dist_ratio",)


# ---------- Coordinate transforms for invariance ----------

def normalize_coords(
    cx: Tensor, cy: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize primitive coordinates to unit spread (scale invariance).

    Centers coords at their mean and divides by max distance from center.
    All subsequent relation computations become scale-independent.

    Args:
        cx: [B, k] x-coordinates
        cy: [B, k] y-coordinates

    Returns:
        norm_cx: [B, k] normalized x
        norm_cy: [B, k] normalized y
        spread: [B] the normalization factor (for inverse transform)
    """
    mean_x = cx.mean(dim=1, keepdim=True)  # [B, 1]
    mean_y = cy.mean(dim=1, keepdim=True)  # [B, 1]
    centered_x = cx - mean_x
    centered_y = cy - mean_y

    # Max distance from centroid
    dists = torch.sqrt(centered_x ** 2 + centered_y ** 2 + 1e-8)  # [B, k]
    spread = dists.max(dim=1, keepdim=True).values.clamp(min=1e-6)  # [B, 1]

    return centered_x / spread, centered_y / spread, spread.squeeze(1)


def canonicalize_coords(
    cx: Tensor, cy: Tensor
) -> tuple[Tensor, Tensor]:
    """Rotate primitive coordinates to canonical frame via PCA (rotation invariance).

    Aligns the principal axis of the primitive layout with the x-axis.
    After canonicalization, 'above'/'left_of' become relative to the
    object's own structure rather than absolute image orientation.

    Args:
        cx: [B, k] x-coordinates (should be centered, e.g. after normalize_coords)
        cy: [B, k] y-coordinates

    Returns:
        canon_cx: [B, k] x in canonical frame
        canon_cy: [B, k] y in canonical frame
    """
    # Stack into [B, k, 2]
    coords = torch.stack([cx, cy], dim=-1)  # [B, k, 2]

    # Center (in case not already centered)
    centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, 2]
    centered = coords - centroid  # [B, k, 2]

    # Covariance matrix: [B, 2, 2]
    cov = torch.bmm(centered.transpose(1, 2), centered) / coords.size(1)

    # Eigenvectors: eigh returns sorted ascending eigenvalues
    # eigenvalues [B, 2], eigenvectors [B, 2, 2]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Principal axis = largest eigenvalue = last column
    # Flip so principal axis is first: [B, 2, 2]
    rotation = eigenvectors.flip(-1)

    # Resolve sign ambiguity: ensure principal direction has positive sum
    # This makes canonicalization deterministic
    projected = torch.bmm(centered, rotation)  # [B, k, 2]
    sign = projected.sum(dim=1).sign()  # [B, 2]
    sign = sign.where(sign != 0, torch.ones_like(sign))  # avoid 0
    rotation = rotation * sign.unsqueeze(1)  # [B, 2, 2]

    # Rotate
    canonical = torch.bmm(centered, rotation)  # [B, k, 2]
    return canonical[:, :, 0], canonical[:, :, 1]


def transform_bbox(
    x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor,
    cx: Tensor, cy: Tensor,
    norm_cx: Tensor, norm_cy: Tensor, spread: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Transform bbox coordinates consistently with center transforms.

    Shifts and scales bboxes to match the normalized/canonicalized centers.

    Args:
        x1, y1, x2, y2: [B, k] original bbox coords
        cx, cy: [B, k] original center coords
        norm_cx, norm_cy: [B, k] transformed center coords
        spread: [B] normalization spread factor

    Returns:
        norm_x1, norm_y1, norm_x2, norm_y2: [B, k] transformed bbox
    """
    spread_2d = spread.unsqueeze(1)  # [B, 1]
    # Half-sizes don't change with rotation in this approximation,
    # just scale by spread
    half_w = (x2 - x1) / 2.0
    half_h = (y2 - y1) / 2.0
    norm_half_w = half_w / spread_2d
    norm_half_h = half_h / spread_2d
    return (
        norm_cx - norm_half_w,
        norm_cy - norm_half_h,
        norm_cx + norm_half_w,
        norm_cy + norm_half_h,
    )


class RelationParams(nn.Module):
    """Learnable parameters for all spatial relations."""

    def __init__(self):
        super().__init__()
        # Sigmoid sharpness and margin for directional relations
        self.lambda_above = nn.Parameter(torch.tensor(5.0))
        self.margin_above = nn.Parameter(torch.tensor(0.0))
        self.lambda_left = nn.Parameter(torch.tensor(5.0))
        self.margin_left = nn.Parameter(torch.tensor(0.0))
        # Gaussian bandwidth for alignment relations
        self.tau_h = nn.Parameter(torch.tensor(0.1))
        self.tau_v = nn.Parameter(torch.tensor(0.1))
        # Gaussian bandwidth for proximity
        self.rho = nn.Parameter(torch.tensor(0.2))
        # Sigmoid sharpness for containment
        self.lambda_contains = nn.Parameter(torch.tensor(5.0))
        # Gaussian bandwidth for distance ratio (rotation-invariant)
        self.sigma_dist = nn.Parameter(torch.tensor(0.3))


def compute_relation(name: str, a: Primitive, b: Primitive, params: RelationParams) -> torch.Tensor:
    """Dispatch to the appropriate relation function."""
    match name:
        case "above":
            return _above(a, b, params)
        case "left_of":
            return _left_of(a, b, params)
        case "aligned_h":
            return _aligned_h(a, b, params)
        case "aligned_v":
            return _aligned_v(a, b, params)
        case "near":
            return _near(a, b, params)
        case "contains":
            return _contains(a, b, params)
        case "dist_ratio":
            return _dist_ratio(a, b, params)
        case _:
            raise ValueError(f"Unknown relation: {name}")


def _above(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """a is above b: sigmoid(lambda * (cy_b - cy_a - margin))"""
    return torch.sigmoid(p.lambda_above * (b.cy - a.cy - p.margin_above))


def _left_of(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """a is left of b: sigmoid(lambda * (cx_b - cx_a - margin))"""
    return torch.sigmoid(p.lambda_left * (b.cx - a.cx - p.margin_left))


def _aligned_h(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """Horizontal alignment: exp(-|cy_a - cy_b|^2 / (2 * tau^2))"""
    return torch.exp(-(a.cy - b.cy) ** 2 / (2 * p.tau_h ** 2))


def _aligned_v(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """Vertical alignment: exp(-|cx_a - cx_b|^2 / (2 * tau^2))"""
    return torch.exp(-(a.cx - b.cx) ** 2 / (2 * p.tau_v ** 2))


def _near(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """Proximity: exp(-||c_a - c_b||^2 / (2 * rho^2))"""
    dist_sq = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2
    return torch.exp(-dist_sq / (2 * p.rho ** 2))


def _contains(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """a contains b: sigmoid(lambda * min(all four margin checks))

    Works with both scalar and batched [B] primitive fields.
    """
    # For a to contain b: a.x1 <= b.x1, a.y1 <= b.y1, a.x2 >= b.x2, a.y2 >= b.y2
    margins = torch.stack([
        b.x1 - a.x1,
        b.y1 - a.y1,
        a.x2 - b.x2,
        a.y2 - b.y2,
    ], dim=-1)  # [..., 4] — works for scalar (shape [4]) and batched ([B, 4])
    return torch.sigmoid(p.lambda_contains * margins.min(dim=-1).values)


def _dist_ratio(a: Primitive, b: Primitive, p: RelationParams) -> torch.Tensor:
    """Rotation-invariant distance ratio: exp(-d(a,b)^2 / (2 * sigma^2)).

    Like 'near' but intended for use with normalized coordinates,
    making it both scale- and rotation-invariant.
    """
    dist_sq = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2
    return torch.exp(-dist_sq / (2 * p.sigma_dist ** 2))


class LearnedRelationParams(nn.Module):
    """Fully learned relation network replacing hand-coded relations.

    Instead of 6 hand-coded functions (sigmoid/Gaussian), uses a small MLP
    that maps pairwise spatial features → n_relations scores in [0, 1].

    Inspired by Relation Networks (Santoro et al., 2017, arXiv 1706.01427):
    relation score = f_theta(features(a, b)).

    The MLP takes 6 pairwise features: dx, dy, |dx|, |dy|, dist, log(area_ratio).
    Output is per-relation scores passed through sigmoid to stay in [0, 1].

    Args:
        n_relations: number of output relation channels (default 6, matching RELATION_NAMES)
        hidden_dim: hidden dimension of relation MLP (default 32)
    """

    def __init__(self, n_relations: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.n_relations = n_relations

        # Keep dummy params so that grammar code referencing params.X doesn't crash
        # during _extract_primitives — these won't be used for scoring
        self.lambda_above = nn.Parameter(torch.tensor(5.0))
        self.margin_above = nn.Parameter(torch.tensor(0.0))
        self.lambda_left = nn.Parameter(torch.tensor(5.0))
        self.margin_left = nn.Parameter(torch.tensor(0.0))
        self.tau_h = nn.Parameter(torch.tensor(0.1))
        self.tau_v = nn.Parameter(torch.tensor(0.1))
        self.rho = nn.Parameter(torch.tensor(0.2))
        self.lambda_contains = nn.Parameter(torch.tensor(5.0))
        self.sigma_dist = nn.Parameter(torch.tensor(0.3))

        # Input: 6 pairwise spatial features
        input_dim = 6
        self.relation_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_relations),
        )
        # Initialize final layer near the hand-coded defaults via small init
        nn.init.xavier_uniform_(self.relation_mlp[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.relation_mlp[2].weight, gain=0.5)
        nn.init.xavier_uniform_(self.relation_mlp[4].weight, gain=0.1)
        nn.init.zeros_(self.relation_mlp[4].bias)

    def compute_relations(self, pairwise_features: Tensor) -> Tensor:
        """Compute all relation scores from pairwise features.

        Args:
            pairwise_features: [B, n_pairs, 6]

        Returns:
            relation_scores: [B, n_pairs, n_relations] in [0, 1]
        """
        return torch.sigmoid(self.relation_mlp(pairwise_features))

    def compute_pairwise_features(
        self, cx_a: Tensor, cy_a: Tensor, cx_b: Tensor, cy_b: Tensor,
        x1_a: Tensor, y1_a: Tensor, x2_a: Tensor, y2_a: Tensor,
        x1_b: Tensor, y1_b: Tensor, x2_b: Tensor, y2_b: Tensor,
    ) -> Tensor:
        """Compute pairwise spatial features.

        Args:
            All inputs: [B, n_pairs]

        Returns:
            features: [B, n_pairs, 6]
        """
        dx = cx_b - cx_a
        dy = cy_b - cy_a
        abs_dx = dx.abs()
        abs_dy = dy.abs()
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        area_a = ((x2_a - x1_a) * (y2_a - y1_a)).clamp(min=1e-8)
        area_b = ((x2_b - x1_b) * (y2_b - y1_b)).clamp(min=1e-8)
        area_ratio = torch.log(area_b / area_a)

        return torch.stack([dx, dy, abs_dx, abs_dy, dist, area_ratio], dim=-1)


def _cayley_transform(A: Tensor) -> Tensor:
    """Cayley transform: maps skew-symmetric A to orthogonal matrix.

    Q = (I - A)(I + A)^{-1}, where A is skew-symmetric (A = -A^T).
    Guarantees Q is orthogonal (Q^T Q = I).

    Args:
        A: [..., n, n] skew-symmetric matrix

    Returns:
        Q: [..., n, n] orthogonal matrix
    """
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


class OrthogonalLearnedRelationParams(LearnedRelationParams):
    """Learned relations with orthogonal output channels and sparse activations.

    Extends LearnedRelationParams with:
    1. Cayley-parameterized final layer: output channels are provably orthogonal,
       so each relation captures a distinct, non-redundant spatial pattern.
    2. Sparse relation loss: L1 penalty on relation activations encourages each
       grammar production to use only 1-2 relation types.

    The hidden→output projection is W = Cayley(A) where A is skew-symmetric.
    Since n_relations (6) < hidden_dim (32), we use a two-step projection:
    hidden → n_relations (learned) → orthogonal mixing (Cayley).

    Args:
        n_relations: number of output relation channels (default 6)
        hidden_dim: hidden dimension of relation MLP (default 32)
        relation_sparsity: L1 penalty weight on relation activations (default 0.01)
    """

    def __init__(self, n_relations: int = 6, hidden_dim: int = 32,
                 relation_sparsity: float = 0.01):
        super().__init__(n_relations=n_relations, hidden_dim=hidden_dim)
        self.relation_sparsity = relation_sparsity

        # Replace the final linear layer with orthogonal projection
        # Step 1: project hidden_dim → n_relations (standard linear)
        self.pre_orth = nn.Linear(hidden_dim, n_relations)
        nn.init.xavier_uniform_(self.pre_orth.weight, gain=0.5)
        nn.init.zeros_(self.pre_orth.bias)

        # Step 2: Cayley-parameterized orthogonal mixing [n_relations, n_relations]
        # A is skew-symmetric: only upper triangle is free params
        # n_relations=6 → 15 free params
        self._skew_params = nn.Parameter(torch.zeros(n_relations * (n_relations - 1) // 2))

        # Override the MLP to only go up to hidden (remove final linear)
        self.relation_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        nn.init.xavier_uniform_(self.relation_mlp[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.relation_mlp[2].weight, gain=0.5)

        # Cache for last relation activations (for sparsity loss)
        self._last_activations: Tensor | None = None

    def _get_orthogonal_matrix(self) -> Tensor:
        """Build orthogonal matrix from skew-symmetric parameters via Cayley transform."""
        n = self.n_relations
        # Build skew-symmetric matrix from upper triangle params
        A = torch.zeros(n, n, device=self._skew_params.device, dtype=self._skew_params.dtype)
        idx = torch.triu_indices(n, n, offset=1)
        A[idx[0], idx[1]] = self._skew_params
        A = A - A.T  # make skew-symmetric
        return _cayley_transform(A)

    def compute_relations(self, pairwise_features: Tensor) -> Tensor:
        """Compute relation scores with orthogonal output channels.

        Args:
            pairwise_features: [B, n_pairs, 6]

        Returns:
            relation_scores: [B, n_pairs, n_relations] in [0, 1]
        """
        hidden = self.relation_mlp(pairwise_features)  # [B, n_pairs, hidden_dim]
        projected = self.pre_orth(hidden)  # [B, n_pairs, n_relations]
        Q = self._get_orthogonal_matrix()  # [n_relations, n_relations]
        orthogonal_out = projected @ Q.T  # [B, n_pairs, n_relations]
        activations = torch.sigmoid(orthogonal_out)
        self._last_activations = activations
        return activations

    def get_relation_sparsity_loss(self) -> Tensor:
        """L1 sparsity loss on relation activations.

        Encourages each primitive pair to activate only 1-2 relations.
        Call after forward pass.

        Returns:
            scalar loss (multiply by self.relation_sparsity externally)
        """
        if self._last_activations is None:
            return torch.tensor(0.0)
        # L1 on activations: mean over batch and pairs, sum over relations
        return self.relation_sparsity * self._last_activations.mean()


class ResidualRelationParams(RelationParams):
    """Hand-coded relations + learned residual correction.

    score_r(a,b) = base_r(a,b) + epsilon * g_theta(features(a,b))

    The correction MLP takes pairwise spatial features (dx, dy, dw, dh, dist, overlap)
    and outputs a residual for each relation type. An orthogonal-inspired constraint
    (spectral normalization) prevents the correction from dominating the base.

    Args:
        n_relations: number of relation types (default 6)
        hidden_dim: hidden dimension of correction MLP (default 8)
        epsilon: scaling factor for residual correction (default 0.1)
    """

    def __init__(self, n_relations: int = 6, hidden_dim: int = 8,
                 epsilon: float = 0.1):
        super().__init__()
        self.n_relations = n_relations
        self.epsilon = nn.Parameter(torch.tensor(epsilon))

        # Input features per pair: dx, dy, |dx|, |dy|, dist, area_ratio (6 dims)
        input_dim = 6
        self.correction_mlp = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(input_dim, hidden_dim)
            ),
            nn.Tanh(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(hidden_dim, n_relations)
            ),
            nn.Tanh(),  # bound output to [-1, 1]
        )
        # Initialize near zero so initial behavior = pure hand-coded
        for m in self.correction_mlp:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_pairwise_features(
        self, cx_a: Tensor, cy_a: Tensor, cx_b: Tensor, cy_b: Tensor,
        x1_a: Tensor, y1_a: Tensor, x2_a: Tensor, y2_a: Tensor,
        x1_b: Tensor, y1_b: Tensor, x2_b: Tensor, y2_b: Tensor,
    ) -> Tensor:
        """Compute pairwise spatial features for correction MLP.

        Args:
            All inputs: [B, n_pairs]

        Returns:
            features: [B, n_pairs, 6]
        """
        dx = cx_b - cx_a
        dy = cy_b - cy_a
        abs_dx = dx.abs()
        abs_dy = dy.abs()
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # Area ratio: area_b / area_a (log scale for stability)
        area_a = ((x2_a - x1_a) * (y2_a - y1_a)).clamp(min=1e-8)
        area_b = ((x2_b - x1_b) * (y2_b - y1_b)).clamp(min=1e-8)
        area_ratio = torch.log(area_b / area_a)

        return torch.stack([dx, dy, abs_dx, abs_dy, dist, area_ratio], dim=-1)

    def compute_residual(self, pairwise_features: Tensor) -> Tensor:
        """Compute residual corrections for all relation types.

        Args:
            pairwise_features: [B, n_pairs, 6]

        Returns:
            corrections: [B, n_pairs, n_relations] in [-epsilon, epsilon]
        """
        return self.epsilon * self.correction_mlp(pairwise_features)

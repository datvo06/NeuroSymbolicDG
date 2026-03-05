"""Spatial relation functions (pure PyTorch, no effects).

Each relation takes two Primitives and a RelationParams module,
returning a soft score in [0, 1].
"""

import torch
import torch.nn as nn

from neurosymbolic_da.dsl.primitives import Primitive

RELATION_NAMES = ("above", "left_of", "aligned_h", "aligned_v", "near", "contains")


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
    """a contains b: sigmoid(lambda * min(all four margin checks))"""
    margins = torch.stack([
        a.x1 - b.x1,  # a's left edge is further left
        a.y1 - b.y1,  # a's top edge is further up
        b.x2 - a.x2,  # b's right edge is further right... wait, reversed
        b.y2 - a.y2,  # b's bottom edge is further down
    ])
    # Actually: for a to contain b, we need:
    # a.x1 <= b.x1, a.y1 <= b.y1, a.x2 >= b.x2, a.y2 >= b.y2
    # So margins should be: b.x1 - a.x1, b.y1 - a.y1, a.x2 - b.x2, a.y2 - b.y2
    margins = torch.stack([
        b.x1 - a.x1,
        b.y1 - a.y1,
        a.x2 - b.x2,
        a.y2 - b.y2,
    ])
    return torch.sigmoid(p.lambda_contains * torch.min(margins))

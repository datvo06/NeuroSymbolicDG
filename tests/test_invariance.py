"""Tests for scale and rotation invariance features."""

import torch

from neurosymbolic_da.dsl.primitives import Primitive
from neurosymbolic_da.dsl.relations import (
    RelationParams,
    canonicalize_coords,
    compute_relation,
    normalize_coords,
    transform_bbox,
)
from neurosymbolic_da.dsl.grammar import LayoutGrammar


def _make_prim(cx, cy, x1=0.0, y1=0.0, x2=1.0, y2=1.0, conf=1.0, type_idx=0):
    return Primitive(
        cx=torch.tensor(cx, dtype=torch.float32),
        cy=torch.tensor(cy, dtype=torch.float32),
        x1=torch.tensor(x1, dtype=torch.float32),
        y1=torch.tensor(y1, dtype=torch.float32),
        x2=torch.tensor(x2, dtype=torch.float32),
        y2=torch.tensor(y2, dtype=torch.float32),
        conf=torch.tensor(conf, dtype=torch.float32),
        type_idx=type_idx,
    )


# ---------- normalize_coords ----------

def test_normalize_coords_unit_spread():
    """After normalization, max distance from centroid should be ~1."""
    cx = torch.tensor([[0.0, 1.0, 2.0]])  # [1, 3]
    cy = torch.tensor([[0.0, 0.0, 0.0]])
    norm_cx, norm_cy, spread = normalize_coords(cx, cy)
    dists = torch.sqrt(norm_cx ** 2 + norm_cy ** 2)
    assert torch.allclose(dists.max(), torch.tensor(1.0), atol=0.01)


def test_normalize_coords_scale_invariance():
    """Scaling original coords should give same normalized result."""
    cx1 = torch.tensor([[0.0, 1.0, 0.5]])
    cy1 = torch.tensor([[0.0, 0.0, 1.0]])
    cx2 = cx1 * 10.0
    cy2 = cy1 * 10.0

    norm_cx1, norm_cy1, _ = normalize_coords(cx1, cy1)
    norm_cx2, norm_cy2, _ = normalize_coords(cx2, cy2)
    assert torch.allclose(norm_cx1, norm_cx2, atol=1e-4)
    assert torch.allclose(norm_cy1, norm_cy2, atol=1e-4)


def test_normalize_coords_batched():
    """Works with batch dimension."""
    cx = torch.tensor([[0.0, 1.0], [0.0, 2.0]])  # [2, 2]
    cy = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    norm_cx, norm_cy, spread = normalize_coords(cx, cy)
    assert norm_cx.shape == (2, 2)
    assert spread.shape == (2,)


# ---------- canonicalize_coords ----------

def test_canonicalize_deterministic():
    """Same input should give same output."""
    cx = torch.tensor([[0.0, 1.0, 0.5]])
    cy = torch.tensor([[0.0, 0.0, 1.0]])
    c1x, c1y = canonicalize_coords(cx, cy)
    c2x, c2y = canonicalize_coords(cx, cy)
    assert torch.allclose(c1x, c2x)
    assert torch.allclose(c1y, c2y)


def test_canonicalize_rotation_invariance():
    """Rotating input coords should give same canonical output."""
    # Original: points along x-axis
    cx1 = torch.tensor([[-1.0, 0.0, 1.0]])
    cy1 = torch.tensor([[0.0, 0.0, 0.0]])

    # Rotated 45 degrees
    angle = torch.tensor(torch.pi / 4)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    cx2 = cx1 * cos_a - cy1 * sin_a
    cy2 = cx1 * sin_a + cy1 * cos_a

    c1x, c1y = canonicalize_coords(cx1, cy1)
    c2x, c2y = canonicalize_coords(cx2, cy2)

    # After canonicalization, both should be aligned to x-axis
    # y-coordinates should both be ~0
    assert torch.allclose(c1y.abs(), torch.zeros_like(c1y), atol=1e-4)
    assert torch.allclose(c2y.abs(), torch.zeros_like(c2y), atol=1e-4)


def test_canonicalize_gradient_flow():
    """Gradients should flow through canonicalization."""
    cx = torch.tensor([[0.0, 1.0, 0.5]], requires_grad=True)
    cy = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    canon_cx, canon_cy = canonicalize_coords(cx, cy)
    loss = (canon_cx ** 2 + canon_cy ** 2).sum()
    loss.backward()
    assert cx.grad is not None
    assert cy.grad is not None


# ---------- transform_bbox ----------

def test_transform_bbox_consistent():
    """Bbox centers should match transformed centers."""
    cx = torch.tensor([[0.5, 1.5]])
    cy = torch.tensor([[0.5, 1.5]])
    x1 = torch.tensor([[0.0, 1.0]])
    y1 = torch.tensor([[0.0, 1.0]])
    x2 = torch.tensor([[1.0, 2.0]])
    y2 = torch.tensor([[1.0, 2.0]])

    norm_cx, norm_cy, spread = normalize_coords(cx, cy)
    nx1, ny1, nx2, ny2 = transform_bbox(x1, y1, x2, y2, cx, cy, norm_cx, norm_cy, spread)

    # Transformed bbox centers should equal norm_cx, norm_cy
    bbox_cx = (nx1 + nx2) / 2
    bbox_cy = (ny1 + ny2) / 2
    assert torch.allclose(bbox_cx, norm_cx, atol=1e-5)
    assert torch.allclose(bbox_cy, norm_cy, atol=1e-5)


# ---------- dist_ratio relation ----------

def test_dist_ratio_same_point():
    """Same point should give score ~1."""
    params = RelationParams()
    a = _make_prim(cx=0.5, cy=0.5)
    b = _make_prim(cx=0.5, cy=0.5)
    val = compute_relation("dist_ratio", a, b, params)
    assert val.item() > 0.99


def test_dist_ratio_far_points():
    """Far points should give low score."""
    params = RelationParams()
    a = _make_prim(cx=0.0, cy=0.0)
    b = _make_prim(cx=2.0, cy=2.0)
    val = compute_relation("dist_ratio", a, b, params)
    assert val.item() < 0.1


def test_dist_ratio_gradient():
    """Gradients flow through dist_ratio."""
    params = RelationParams()
    a = Primitive(
        cx=torch.tensor(0.3, requires_grad=True),
        cy=torch.tensor(0.2, requires_grad=True),
        x1=torch.tensor(0.0), y1=torch.tensor(0.0),
        x2=torch.tensor(0.6), y2=torch.tensor(0.4),
        conf=torch.tensor(0.9), type_idx=0,
    )
    b = _make_prim(cx=0.7, cy=0.8)
    val = compute_relation("dist_ratio", a, b, params)
    val.backward()
    assert a.cx.grad is not None


# ---------- Grammar with invariant_coords ----------

def test_grammar_invariant_productions():
    """Grammar with invariant_coords should have more productions (dist_ratio)."""
    g_normal = LayoutGrammar(4, 3, max_depth=1)
    g_inv = LayoutGrammar(4, 3, max_depth=1, invariant_coords=True)
    # With k=4: has=4, pairs=12
    # Normal: 4 + 6*12 = 76
    # Invariant: 4 + 7*12 = 88
    assert g_inv.n_productions > g_normal.n_productions
    assert g_inv.n_productions == g_normal.n_productions + 12  # +1 rel * 12 pairs


def test_grammar_invariant_forward():
    """Grammar with invariant_coords should produce valid outputs."""
    k, n_classes = 4, 3
    B = 2
    grammar = LayoutGrammar(k, n_classes, max_depth=1, invariant_coords=True)
    params = RelationParams()

    # Build batched env
    env = {}
    for j in range(k):
        env[j] = Primitive(
            cx=torch.randn(B), cy=torch.randn(B),
            x1=torch.randn(B) - 0.5, y1=torch.randn(B) - 0.5,
            x2=torch.randn(B) + 0.5, y2=torch.randn(B) + 0.5,
            conf=torch.sigmoid(torch.randn(B)), type_idx=j,
        )

    scores = grammar.forward_vectorized(env, params)
    assert scores.shape == (B, n_classes)
    assert torch.isfinite(scores).all()


def test_grammar_invariant_scale_equivariance():
    """Scores should be similar for scaled versions of the same layout."""
    k, n_classes = 4, 3
    B = 1
    grammar = LayoutGrammar(k, n_classes, max_depth=1, invariant_coords=True)
    params = RelationParams()

    torch.manual_seed(42)
    env1, env2 = {}, {}
    for j in range(k):
        cx, cy = torch.randn(B), torch.randn(B)
        conf = torch.sigmoid(torch.randn(B))
        hw = 0.2
        env1[j] = Primitive(
            cx=cx, cy=cy, x1=cx - hw, y1=cy - hw, x2=cx + hw, y2=cy + hw,
            conf=conf, type_idx=j,
        )
        # Scale by 5x
        env2[j] = Primitive(
            cx=cx * 5, cy=cy * 5, x1=(cx - hw) * 5, y1=(cy - hw) * 5,
            x2=(cx + hw) * 5, y2=(cy + hw) * 5,
            conf=conf, type_idx=j,
        )

    s1 = grammar.forward_vectorized(env1, params)
    s2 = grammar.forward_vectorized(env2, params)
    # After normalization, scores should be identical
    assert torch.allclose(s1, s2, atol=1e-3), f"Scale invariance broken: {s1} vs {s2}"


def test_grammar_invariant_gradient_flow():
    """Gradients flow through invariant grammar forward."""
    k, n_classes = 4, 3
    B = 2
    grammar = LayoutGrammar(k, n_classes, max_depth=1, invariant_coords=True)
    params = RelationParams()

    env = {}
    cx_list = []
    for j in range(k):
        cx = torch.randn(B, requires_grad=True)
        cx_list.append(cx)
        env[j] = Primitive(
            cx=cx, cy=torch.randn(B),
            x1=torch.randn(B) - 0.5, y1=torch.randn(B) - 0.5,
            x2=torch.randn(B) + 0.5, y2=torch.randn(B) + 0.5,
            conf=torch.sigmoid(torch.randn(B)), type_idx=j,
        )

    scores = grammar.forward_vectorized(env, params)
    scores.sum().backward()
    assert cx_list[0].grad is not None


def test_grammar_invariant_hierarchical():
    """Invariant coords work with hierarchical grammar (max_depth=2)."""
    k, n_classes = 4, 3
    B = 2
    grammar = LayoutGrammar(k, n_classes, max_depth=2, invariant_coords=True)
    params = RelationParams()

    env = {}
    for j in range(k):
        env[j] = Primitive(
            cx=torch.randn(B), cy=torch.randn(B),
            x1=torch.randn(B) - 0.5, y1=torch.randn(B) - 0.5,
            x2=torch.randn(B) + 0.5, y2=torch.randn(B) + 0.5,
            conf=torch.sigmoid(torch.randn(B)), type_idx=j,
        )

    scores = grammar.forward_vectorized(env, params)
    assert scores.shape == (B, n_classes)
    assert torch.isfinite(scores).all()

"""Test spatial relation functions and gradient flow."""

import torch

from neurosymbolic_da.dsl.primitives import Primitive
from neurosymbolic_da.dsl.relations import RelationParams, compute_relation


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


def test_above_high_score():
    """a clearly above b should give score near 1."""
    params = RelationParams()
    a = _make_prim(cx=0.5, cy=0.2)  # higher (smaller cy)
    b = _make_prim(cx=0.5, cy=0.8)  # lower
    val = compute_relation("above", a, b, params)
    assert val.item() > 0.9


def test_above_low_score():
    """a below b should give score near 0."""
    params = RelationParams()
    a = _make_prim(cx=0.5, cy=0.8)
    b = _make_prim(cx=0.5, cy=0.2)
    val = compute_relation("above", a, b, params)
    assert val.item() < 0.1


def test_left_of_high_score():
    params = RelationParams()
    a = _make_prim(cx=0.2, cy=0.5)
    b = _make_prim(cx=0.8, cy=0.5)
    val = compute_relation("left_of", a, b, params)
    assert val.item() > 0.9


def test_aligned_h():
    """Same cy = perfect horizontal alignment."""
    params = RelationParams()
    a = _make_prim(cx=0.2, cy=0.5)
    b = _make_prim(cx=0.8, cy=0.5)
    val = compute_relation("aligned_h", a, b, params)
    assert val.item() > 0.99


def test_aligned_h_misaligned():
    params = RelationParams()
    a = _make_prim(cx=0.2, cy=0.1)
    b = _make_prim(cx=0.8, cy=0.9)
    val = compute_relation("aligned_h", a, b, params)
    assert val.item() < 0.01


def test_near_same_point():
    params = RelationParams()
    a = _make_prim(cx=0.5, cy=0.5)
    b = _make_prim(cx=0.5, cy=0.5)
    val = compute_relation("near", a, b, params)
    assert val.item() > 0.99


def test_contains():
    params = RelationParams()
    outer = _make_prim(cx=0.5, cy=0.5, x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    inner = _make_prim(cx=0.5, cy=0.5, x1=0.2, y1=0.2, x2=0.8, y2=0.8)
    val = compute_relation("contains", outer, inner, params)
    assert val.item() > 0.7


def test_gradient_flow():
    """Verify gradients flow through relation computation."""
    params = RelationParams()
    a = Primitive(
        cx=torch.tensor(0.3, requires_grad=True),
        cy=torch.tensor(0.2, requires_grad=True),
        x1=torch.tensor(0.0), y1=torch.tensor(0.0),
        x2=torch.tensor(0.6), y2=torch.tensor(0.4),
        conf=torch.tensor(0.9),
        type_idx=0,
    )
    b = _make_prim(cx=0.7, cy=0.8)
    val = compute_relation("above", a, b, params)
    val.backward()
    assert a.cy.grad is not None
    assert a.cy.grad.item() != 0.0

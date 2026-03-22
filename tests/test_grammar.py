"""Test grammar construction and forward pass under different handlers."""

import torch

from effectful.ops.semantics import handler
from effectful.ops.types import Term

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from neurosymbolic_da.dsl.handlers.eval import make_eval_handler
from neurosymbolic_da.dsl.handlers.inside import make_inside_handler
from neurosymbolic_da.dsl.handlers.symbolic import DerivNode, make_symbolic_handler
from neurosymbolic_da.dsl.primitives import Primitive
from neurosymbolic_da.dsl.relations import RELATION_NAMES, RelationParams


def _make_env(n):
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    confs = [0.9, 0.8, 0.7]
    env = {}
    for i in range(n):
        cx, cy = positions[i % len(positions)]
        env[i] = Primitive(
            cx=torch.tensor(cx), cy=torch.tensor(cy),
            x1=torch.tensor(cx - 0.1), y1=torch.tensor(cy - 0.1),
            x2=torch.tensor(cx + 0.1), y2=torch.tensor(cy + 0.1),
            conf=torch.tensor(confs[i % len(confs)]),
            type_idx=i,
        )
    return env


def test_grammar_construction():
    g = LayoutGrammar(n_primitives=3, n_classes=2, max_depth=1)
    # Level 0: 3 has productions
    # Level 1: 6 relations * 3*2 ordered pairs = 36
    expected = 3 + len(RELATION_NAMES) * 3 * 2
    assert g.n_productions == expected


def test_grammar_weights_shape():
    g = LayoutGrammar(n_primitives=3, n_classes=5)
    assert g.log_weights.shape == (5, g.n_productions)


def test_grammar_unhandled_produces_term():
    """Without a handler, grammar forward should produce a Term."""
    g = LayoutGrammar(n_primitives=2, n_classes=2, max_depth=1)
    result = g(class_idx=0)
    assert isinstance(result, Term)


def test_grammar_eval_handler():
    """Grammar forward with eval handler produces a scalar Tensor."""
    n = 3
    env = _make_env(n)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=1)

    with handler(make_eval_handler(env, params)):
        result = g(class_idx=0)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # scalar
    assert result.item() > 0


def test_grammar_inside_handler():
    """Grammar forward with inside handler produces an InsideTable."""
    n = 2
    env = _make_env(n)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=1)

    with handler(make_inside_handler(env, params)):
        result = g(class_idx=0)

    assert isinstance(result, dict)
    # Should have entries for various subsets
    assert len(result) > 0


def test_grammar_symbolic_handler():
    """Grammar forward with symbolic handler produces a DerivNode."""
    g = LayoutGrammar(n_primitives=2, n_classes=2, max_depth=1)
    with handler(make_symbolic_handler()):
        result = g(class_idx=0)

    assert isinstance(result, DerivNode)
    assert result.op == "choice"
    tree_str = str(result)
    assert "has" in tree_str or "rel" in tree_str


def test_grammar_gradient_flow():
    """Verify gradients flow from grammar output to relation params."""
    n = 2
    env = _make_env(n)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=1)

    with handler(make_eval_handler(env, params)):
        result = g(class_idx=0)

    result.backward()
    assert g.log_weights.grad is not None
    assert params.lambda_above.grad is not None


# --- Hierarchical grammar tests (max_depth >= 2) ---


def test_hierarchical_grammar_construction():
    """Hierarchical grammar has Level 0-3 productions."""
    g = LayoutGrammar(n_primitives=3, n_classes=2, max_depth=2)
    # Level 0: 3 has, Level 1: 6 rels * 6 pairs = 36 → 39 base
    n_base = 3 + 6 * 6  # = 39
    # Level 2: C(3,2) = 3 sublayouts (pairs of primitives)
    n_sublayout = 3
    # Level 3: k=3, all sublayouts share primitives, no disjoint pairs → 0
    n_group_rel = 0
    expected = n_base + n_sublayout + n_group_rel
    assert g.n_productions == expected

    # k=4 has disjoint sublayouts
    g4 = LayoutGrammar(n_primitives=4, n_classes=2, max_depth=2)
    assert g4.n_productions > g4._n_base_productions + g4._n_sublayout_productions


def test_hierarchical_grammar_backward_compat():
    """max_depth=1 gives same productions as before."""
    g1 = LayoutGrammar(n_primitives=3, n_classes=2, max_depth=1)
    # Should only have Level 0+1
    expected = 3 + 6 * 6
    assert g1.n_productions == expected


def test_hierarchical_grammar_inside_handler():
    """Hierarchical grammar produces valid inside tables."""
    n = 4  # need k>=4 for disjoint sublayouts / group_rel
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.5)]
    confs = [0.9, 0.8, 0.7, 0.6]
    env = {}
    for i in range(n):
        cx, cy = positions[i]
        env[i] = Primitive(
            cx=torch.tensor(cx), cy=torch.tensor(cy),
            x1=torch.tensor(cx - 0.1), y1=torch.tensor(cy - 0.1),
            x2=torch.tensor(cx + 0.1), y2=torch.tensor(cy + 0.1),
            conf=torch.tensor(confs[i]),
            type_idx=i,
        )
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=2)

    with handler(make_inside_handler(env, params)):
        result = g(class_idx=0)

    assert isinstance(result, dict)
    assert len(result) > 0
    for key, val in result.items():
        assert isinstance(key, frozenset)
        assert val.item() >= 0


def test_hierarchical_grammar_gradient_flow():
    """Verify gradients flow through hierarchical grammar."""
    n = 4
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.5)]
    confs = [0.9, 0.8, 0.7, 0.6]
    env = {}
    for i in range(n):
        cx, cy = positions[i]
        env[i] = Primitive(
            cx=torch.tensor(cx), cy=torch.tensor(cy),
            x1=torch.tensor(cx - 0.1), y1=torch.tensor(cy - 0.1),
            x2=torch.tensor(cx + 0.1), y2=torch.tensor(cy + 0.1),
            conf=torch.tensor(confs[i]),
            type_idx=i,
        )
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=2)

    with handler(make_eval_handler(env, params)):
        result = g(class_idx=0)

    result.backward()
    assert g.log_weights.grad is not None
    assert params.lambda_above.grad is not None


# --- Vectorized forward tests ---


def _make_batched_env(n, batch_size=4):
    """Create a batched env with shape [B] fields."""
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.5)]
    confs = [0.9, 0.8, 0.7, 0.6]
    env = {}
    for i in range(n):
        cx, cy = positions[i % len(positions)]
        env[i] = Primitive(
            cx=torch.full((batch_size,), cx),
            cy=torch.full((batch_size,), cy),
            x1=torch.full((batch_size,), cx - 0.1),
            y1=torch.full((batch_size,), cy - 0.1),
            x2=torch.full((batch_size,), cx + 0.1),
            y2=torch.full((batch_size,), cy + 0.1),
            conf=torch.full((batch_size,), confs[i % len(confs)]),
            type_idx=i,
        )
    return env


def test_vectorized_flat_shape():
    """Vectorized flat forward produces [B, n_classes]."""
    n, B = 3, 4
    env = _make_batched_env(n, B)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=1)
    result = g.forward_vectorized(env, params)
    assert result.shape == (B, 2)


def test_vectorized_hierarchical_shape():
    """Vectorized hierarchical forward produces [B, n_classes]."""
    n, B = 4, 4
    env = _make_batched_env(n, B)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=2)
    result = g.forward_vectorized(env, params)
    assert result.shape == (B, 2)
    assert result.sum().item() > 0


def test_vectorized_hierarchical_gradient_flow():
    """Gradients flow through vectorized hierarchical forward."""
    n, B = 4, 2
    env = _make_batched_env(n, B)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=2)
    result = g.forward_vectorized(env, params)
    result.sum().backward()
    assert g.log_weights.grad is not None
    assert params.lambda_above.grad is not None


def test_vectorized_hierarchical_no_nan():
    """Vectorized hierarchical produces no NaN or Inf."""
    n = 4
    env = _make_batched_env(n, batch_size=2)
    params = RelationParams()
    g = LayoutGrammar(n_primitives=n, n_classes=2, max_depth=2)
    result = g.forward_vectorized(env, params)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

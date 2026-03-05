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

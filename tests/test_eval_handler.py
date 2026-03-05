"""Test the direct evaluation handler."""

import torch

from effectful.ops.semantics import handler

from neurosymbolic_da.dsl.handlers.eval import make_eval_handler
from neurosymbolic_da.dsl.ops import choice, conj, has, rel, score
from neurosymbolic_da.dsl.primitives import Primitive
from neurosymbolic_da.dsl.relations import RelationParams


def _make_env():
    """Create a simple 3-primitive environment."""
    env = {}
    for i, (cx, cy, conf) in enumerate([
        (0.2, 0.2, 0.9),  # prim 0: top-left
        (0.8, 0.2, 0.8),  # prim 1: top-right
        (0.5, 0.8, 0.7),  # prim 2: bottom-center
    ]):
        env[i] = Primitive(
            cx=torch.tensor(cx), cy=torch.tensor(cy),
            x1=torch.tensor(cx - 0.1), y1=torch.tensor(cy - 0.1),
            x2=torch.tensor(cx + 0.1), y2=torch.tensor(cy + 0.1),
            conf=torch.tensor(conf),
            type_idx=i,
        )
    return env


def test_has():
    env = _make_env()
    params = RelationParams()
    with handler(make_eval_handler(env, params)):
        result = has(0)
    assert torch.isclose(result, torch.tensor(0.9))


def test_rel():
    env = _make_env()
    params = RelationParams()
    with handler(make_eval_handler(env, params)):
        # prim 0 is above prim 2 (cy 0.2 vs 0.8)
        result = rel("above", 0, 2)
    assert result.item() > 0.9


def test_conj():
    env = _make_env()
    params = RelationParams()
    with handler(make_eval_handler(env, params)):
        result = conj(has(0), has(1))
    expected = torch.tensor(0.9 * 0.8)
    assert torch.isclose(result, expected)


def test_choice():
    env = _make_env()
    params = RelationParams()
    with handler(make_eval_handler(env, params)):
        result = choice(has(0), has(1), has(2))
    expected = torch.tensor(0.9 + 0.8 + 0.7)
    assert torch.isclose(result, expected)


def test_score():
    env = _make_env()
    params = RelationParams()
    w = torch.tensor(0.5)
    with handler(make_eval_handler(env, params)):
        result = score(w, has(0))
    expected = torch.tensor(0.5 * 0.9)
    assert torch.isclose(result, expected)


def test_compound_expression():
    """Test a realistic compound DSL expression."""
    env = _make_env()
    params = RelationParams()
    with handler(make_eval_handler(env, params)):
        # "prim 0 exists AND prim 0 is above prim 2"
        result = conj(has(0), rel("above", 0, 2))
    # conf_0 * above(0, 2) — both should be high
    assert result.item() > 0.5

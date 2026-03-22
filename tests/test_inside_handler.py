"""Test the inside algorithm handler."""

import torch

from effectful.ops.semantics import handler

from neurosymbolic_da.dsl.handlers.inside import (
    InsideTable,
    get_class_score,
    make_inside_handler,
)
from neurosymbolic_da.dsl.ops import choice, conj, group_rel, has, rel, score
from neurosymbolic_da.dsl.primitives import Primitive, SubLayout, aggregate
from neurosymbolic_da.dsl.relations import RelationParams


def _make_env(n=3):
    """Create a simple n-primitive environment."""
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    confs = [0.9, 0.8, 0.7]
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
    return env


def test_has_returns_singleton_table():
    env = _make_env()
    params = RelationParams()
    with handler(make_inside_handler(env, params)):
        table = has(0)
    assert isinstance(table, dict)
    assert frozenset({0}) in table
    assert torch.isclose(table[frozenset({0})], torch.tensor(0.9))


def test_rel_returns_pair_table():
    env = _make_env()
    params = RelationParams()
    with handler(make_inside_handler(env, params)):
        table = rel("above", 0, 2)
    assert frozenset({0, 2}) in table
    assert table[frozenset({0, 2})].item() > 0.9


def test_conj_disjoint_sets():
    env = _make_env()
    params = RelationParams()
    with handler(make_inside_handler(env, params)):
        t1 = has(0)   # {0} -> 0.9
        t2 = has(1)   # {1} -> 0.8
        table = conj(t1, t2)
    # Should produce {0, 1} -> 0.9 * 0.8
    assert frozenset({0, 1}) in table
    expected = torch.tensor(0.9 * 0.8)
    assert torch.isclose(table[frozenset({0, 1})], expected)


def test_conj_overlapping_sets_empty():
    """Conj of overlapping sets produces no entries."""
    env = _make_env()
    params = RelationParams()
    with handler(make_inside_handler(env, params)):
        t1 = has(0)
        t2 = has(0)  # same primitive
        table = conj(t1, t2)
    # {0} is not disjoint with {0}, so no entries
    assert len(table) == 0


def test_choice_merges_tables():
    env = _make_env()
    params = RelationParams()
    with handler(make_inside_handler(env, params)):
        t1 = has(0)  # {0} -> 0.9
        t2 = has(1)  # {1} -> 0.8
        table = choice(t1, t2)
    assert frozenset({0}) in table
    assert frozenset({1}) in table


def test_choice_sums_overlapping():
    env = _make_env()
    params = RelationParams()
    w1 = torch.tensor(0.3)
    w2 = torch.tensor(0.7)
    with handler(make_inside_handler(env, params)):
        t1 = score(w1, has(0))  # {0} -> 0.3 * 0.9
        t2 = score(w2, has(0))  # {0} -> 0.7 * 0.9
        table = choice(t1, t2)
    # Should sum: {0} -> (0.3 + 0.7) * 0.9 = 0.9
    expected = torch.tensor((0.3 + 0.7) * 0.9)
    assert torch.isclose(table[frozenset({0})], expected)


def test_full_inside_2_primitives():
    """Verify a small grammar over 2 primitives by hand."""
    env = _make_env(2)
    params = RelationParams()
    w_has0 = torch.tensor(0.4)
    w_has1 = torch.tensor(0.6)
    w_rel = torch.tensor(1.0)

    with handler(make_inside_handler(env, params)):
        # Grammar: choice(score(w, conj(has(0), has(1))), score(w_rel, rel("above", 0, 1)))
        t1 = score(w_has0, conj(has(0), has(1)))  # {0,1} -> 0.4 * 0.9 * 0.8
        above_score = rel("above", 0, 1)  # {0,1} -> above(0,1)
        t2 = score(w_rel, above_score)
        table = choice(t1, t2)

    # Both contribute to {0, 1}
    full_set = frozenset({0, 1})
    assert full_set in table
    class_score = get_class_score(table, 2)
    assert class_score.item() > 0


def test_gradient_flow_inside():
    """Verify gradients flow through the inside algorithm."""
    env = _make_env(2)
    # Make positions require grad
    for p in env.values():
        p.cx = p.cx.clone().requires_grad_(True)
        p.cy = p.cy.clone().requires_grad_(True)

    params = RelationParams()
    w = torch.tensor(1.0)

    with handler(make_inside_handler(env, params)):
        table = score(w, conj(has(0), rel("above", 0, 1)))

    result = get_class_score(table, 2)
    if result.item() > 0:
        result.backward()
        assert env[0].cy.grad is not None


# --- Tests for SubLayout / aggregate / group_rel ---


def test_aggregate_centroid():
    """Aggregate centroid is confidence-weighted mean."""
    env = _make_env(3)
    sub = aggregate(env, frozenset({0, 1}))
    assert isinstance(sub, SubLayout)
    assert sub.members == frozenset({0, 1})
    # Confidence-weighted mean of (0.2, 0.2) w=0.9 and (0.8, 0.2) w=0.8
    expected_cx = (0.9 * 0.2 + 0.8 * 0.8) / (0.9 + 0.8)
    assert torch.isclose(sub.cx, torch.tensor(expected_cx), atol=1e-5)


def test_aggregate_bbox():
    """Aggregate bbox is enclosing box."""
    env = _make_env(3)
    sub = aggregate(env, frozenset({0, 2}))
    # prim 0: x1=0.1, prim 2: x1=0.4 → min=0.1
    # prim 0: x2=0.3, prim 2: x2=0.6 → max=0.6
    assert sub.x1.item() < sub.x2.item()
    assert sub.y1.item() < sub.y2.item()


def test_aggregate_conf():
    """Aggregate confidence is sum of member confidences."""
    env = _make_env(3)
    sub = aggregate(env, frozenset({0, 1}))
    assert torch.isclose(sub.conf, torch.tensor(0.9 + 0.8))


def test_group_rel_inside_handler():
    """group_rel produces inside table entries with group-level relation scores."""
    env = _make_env(3)
    params = RelationParams()

    with handler(make_inside_handler(env, params)):
        # Build two sublayout tables
        t1 = conj(has(0), has(1))  # {0,1} -> 0.9 * 0.8
        t2 = has(2)               # {2} -> 0.7
        # Group-level relation: above(group{0,1}, group{2})
        table = group_rel("above", t1, t2)

    # Should produce {0,1,2} entry
    full = frozenset({0, 1, 2})
    assert full in table
    assert table[full].item() > 0


def test_group_rel_gradient_flow():
    """Verify gradients flow through group_rel."""
    env = _make_env(3)
    for p in env.values():
        p.cx = p.cx.clone().requires_grad_(True)
        p.cy = p.cy.clone().requires_grad_(True)

    params = RelationParams()

    with handler(make_inside_handler(env, params)):
        t1 = conj(has(0), has(1))
        t2 = has(2)
        table = group_rel("above", t1, t2)

    result = get_class_score(table, 3)
    result.backward()
    # Gradient should flow to primitive positions (used in aggregate centroid)
    assert env[0].cy.grad is not None
    assert env[2].cy.grad is not None
    # And to relation params
    assert params.lambda_above.grad is not None

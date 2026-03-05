"""Test that unhandled DSL ops produce symbolic Terms."""

from effectful.ops.types import Term

from neurosymbolic_da.dsl.ops import choice, conj, has, rel, score


def test_has_produces_term():
    result = has(0)
    assert isinstance(result, Term)


def test_rel_produces_term():
    result = rel("above", 0, 1)
    assert isinstance(result, Term)


def test_conj_produces_term():
    t1 = has(0)
    t2 = has(1)
    result = conj(t1, t2)
    assert isinstance(result, Term)


def test_choice_produces_term():
    t1 = has(0)
    t2 = has(1)
    result = choice(t1, t2)
    assert isinstance(result, Term)


def test_score_produces_term():
    t = has(0)
    result = score(0.5, t)
    assert isinstance(result, Term)


def test_nested_term():
    """A compound DSL expression produces a nested Term tree."""
    expr = choice(
        conj(has(0), rel("above", 0, 1)),
        conj(has(1), rel("left_of", 1, 2)),
    )
    assert isinstance(expr, Term)
    assert str(expr)  # should be printable

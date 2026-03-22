"""Test extract_derivations script utilities."""

import torch

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from scripts.extract_derivations import (
    extract_symbolic_tree,
    format_production,
    get_top_productions,
)


def _make_grammar(n_primitives=3, n_classes=2):
    return LayoutGrammar(n_primitives, n_classes, max_depth=1)


def test_get_top_productions():
    grammar = _make_grammar()
    # Set one production to have high weight
    grammar.log_weights.data[0, 0] = 10.0
    top = get_top_productions(grammar, class_idx=0, top_k=3)
    assert len(top) == 3
    # First should be the boosted production
    assert top[0][0] > top[1][0]
    assert top[0][1] == grammar.productions[0]


def test_format_production_has():
    prod = {"type": "has", "prim": 2}
    assert format_production(prod) == "has(p2)"


def test_format_production_rel():
    prod = {"type": "rel", "name": "above", "a": 0, "b": 1}
    assert format_production(prod) == "above(p0, p1)"


def test_extract_symbolic_tree():
    grammar = _make_grammar(n_primitives=2, n_classes=2)
    tree = extract_symbolic_tree(grammar, class_idx=0)
    assert tree is not None
    assert tree.op == "choice"
    assert len(tree.children) > 0


def test_extract_symbolic_tree_string():
    grammar = _make_grammar(n_primitives=2, n_classes=2)
    tree = extract_symbolic_tree(grammar, class_idx=0)
    tree_str = str(tree)
    assert "choice" in tree_str
    assert "has" in tree_str or "rel" in tree_str

"""Symbolic tree-building handler.

Reinterprets DSL ops to build DerivNode trees for interpretability
(paper Section 7.1 / 4.1). Returns tree structures instead of scores.
"""

from dataclasses import dataclass, field

from effectful.ops.types import Interpretation

from neurosymbolic_da.dsl.ops import choice, conj, has, rel, score


@dataclass
class DerivNode:
    """A node in a derivation tree."""

    op: str
    args: list = field(default_factory=list)
    children: list["DerivNode"] = field(default_factory=list)

    def __str__(self, indent: int = 0) -> str:
        prefix = "  " * indent
        args_str = ", ".join(str(a) for a in self.args)
        result = f"{prefix}{self.op}({args_str})"
        for child in self.children:
            result += "\n" + child.__str__(indent + 1)
        return result


def make_symbolic_handler() -> Interpretation:
    """Create a handler that builds derivation trees."""

    def _has(type_idx: int) -> DerivNode:
        return DerivNode(op="has", args=[type_idx])

    def _rel(name: str, a: int, b: int) -> DerivNode:
        return DerivNode(op="rel", args=[name, a, b])

    def _conj(c1: DerivNode, c2: DerivNode) -> DerivNode:
        return DerivNode(op="conj", children=[c1, c2])

    def _choice(*alternatives: DerivNode) -> DerivNode:
        return DerivNode(op="choice", children=list(alternatives))

    def _score(weight, body: DerivNode) -> DerivNode:
        return DerivNode(op="score", args=[weight], children=[body])

    return {
        has: _has,
        rel: _rel,
        conj: _conj,
        choice: _choice,
        score: _score,
    }

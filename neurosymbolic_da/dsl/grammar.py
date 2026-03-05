"""PCFG grammar over layout programs.

Builds a universal grammar (all productions up to depth D_max) and
provides a forward() method that calls DSL ops. The handler installed
at call time determines the semantics (eval, inside, symbolic).
"""

import torch
import torch.nn as nn
from torch import Tensor

from neurosymbolic_da.dsl.ops import choice, conj, has, rel, score
from neurosymbolic_da.dsl.relations import RELATION_NAMES


class LayoutGrammar(nn.Module):
    """A PCFG over the layout DSL.

    The grammar enumerates all productions up to a bounded depth and stores
    log-weights as parameters. forward() calls DSL ops — the active handler
    determines semantics.

    Args:
        n_primitives: number of primitive types (k)
        n_classes: number of output classes
        max_depth: maximum derivation depth (D_max)
    """

    def __init__(self, n_primitives: int, n_classes: int, max_depth: int = 2):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes
        self.max_depth = max_depth

        # Build production list
        productions = self._enumerate_productions()
        self.n_productions = len(productions)
        self.productions = productions

        # Log-weights: one set per class [n_classes, n_productions]
        self.log_weights = nn.Parameter(torch.zeros(n_classes, self.n_productions))

    def _enumerate_productions(self) -> list[dict]:
        """Enumerate all productions in the universal grammar."""
        prods = []
        k = self.n_primitives

        # Level 0: has(j) for each primitive type j
        for j in range(k):
            prods.append({"type": "has", "prim": j})

        # Level 1: rel(name, i, j) for each relation and ordered pair i != j
        for name in RELATION_NAMES:
            for i in range(k):
                for j in range(k):
                    if i != j:
                        prods.append({"type": "rel", "name": name, "a": i, "b": j})

        # Level 2+: conjunctions are handled dynamically in forward()
        # via the conj op over level 0/1 productions

        return prods

    def _get_weights(self, class_idx: int) -> Tensor:
        """Get softmax-normalized weights for a class."""
        return torch.softmax(self.log_weights[class_idx], dim=0)

    def forward(self, class_idx: int):
        """Run the grammar for a given class, producing a DSL expression.

        The result type depends on the active handler:
        - eval handler: Tensor (scalar score)
        - inside handler: InsideTable
        - symbolic handler: DerivNode
        """
        weights = self._get_weights(class_idx)

        # Build all base constraint terms (level 0 + level 1)
        terms = []
        for idx, prod in enumerate(self.productions):
            w = weights[idx]
            if prod["type"] == "has":
                term = score(w, has(prod["prim"]))
            else:  # rel
                term = score(w, rel(prod["name"], prod["a"], prod["b"]))
            terms.append(term)

        # Level 2+: conjunctions of base terms
        # For depth > 1, build pairwise conjunctions
        if self.max_depth >= 2 and len(terms) > 1:
            conj_terms = []
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    conj_terms.append(conj(terms[i], terms[j]))
            terms = terms + conj_terms

        # Marginalize: choice over all terms
        return choice(*terms)

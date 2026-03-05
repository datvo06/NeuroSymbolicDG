"""Inside algorithm handler.

Reinterprets DSL ops to compute inside tables. Each operation returns
a dict[frozenset[int], Tensor] mapping primitive subsets to scores,
implementing the set-based inside recurrence from the paper (Section 3.7).
"""

import torch
from torch import Tensor

from effectful.ops.types import Interpretation

from neurosymbolic_da.dsl.ops import choice, conj, has, rel, score
from neurosymbolic_da.dsl.primitives import Env
from neurosymbolic_da.dsl.relations import RelationParams, compute_relation

# Inside table: maps subsets of primitive indices to scores
InsideTable = dict[frozenset[int], Tensor]


def make_inside_handler(env: Env, params: RelationParams) -> Interpretation:
    """Create a handler that computes inside tables.

    Each DSL op returns an InsideTable instead of a scalar Tensor.
    The final result for the full set of primitives gives the
    marginalized class score.
    """

    def _has(type_idx: int) -> InsideTable:
        return {frozenset({type_idx}): env[type_idx].conf}

    def _rel(name: str, a: int, b: int) -> InsideTable:
        val = compute_relation(name, env[a], env[b], params)
        return {frozenset({a, b}): val}

    def _conj(t1: InsideTable, t2: InsideTable) -> InsideTable:
        result: InsideTable = {}
        for s1, v1 in t1.items():
            for s2, v2 in t2.items():
                if s1.isdisjoint(s2):
                    key = s1 | s2
                    val = v1 * v2
                    if key in result:
                        result[key] = result[key] + val
                    else:
                        result[key] = val
        return result

    def _choice(*alternatives: InsideTable) -> InsideTable:
        result: InsideTable = {}
        for table in alternatives:
            for key, val in table.items():
                if key in result:
                    result[key] = result[key] + val
                else:
                    result[key] = val
        return result

    def _score(weight: Tensor, body: InsideTable) -> InsideTable:
        return {key: weight * val for key, val in body.items()}

    return {
        has: _has,
        rel: _rel,
        conj: _conj,
        choice: _choice,
        score: _score,
    }


def get_class_score(table: InsideTable, n_primitives: int) -> Tensor:
    """Extract the class score from an inside table.

    The class score is the entry for the full set of primitives.
    Returns 0 if the full set is not in the table.
    """
    full_set = frozenset(range(n_primitives))
    return table.get(full_set, torch.tensor(0.0))

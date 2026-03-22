"""Inside algorithm handler.

Reinterprets DSL ops to compute inside tables. Each operation returns
a dict[frozenset[int], Tensor] mapping primitive subsets to scores,
implementing the set-based inside recurrence from the paper (Section 3.7).

For hierarchical grammars (Level 3), the group_rel op computes spatial
relations between SubLayouts — groups of primitives with aggregate
spatial features (confidence-weighted centroid, enclosing bbox).
"""

import torch
from torch import Tensor

from effectful.ops.types import Interpretation

from neurosymbolic_da.dsl.ops import choice, conj, group_rel, has, rel, score
from neurosymbolic_da.dsl.primitives import Env, aggregate
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

    def _group_rel(name: str, t1: InsideTable, t2: InsideTable) -> InsideTable:
        """Level 3: spatial relation between two SubLayouts.

        I(A, S) = sum_{S=S1+S2} I(B,S1) * I(C,S2) * rel(agg(S1), agg(S2))

        For each disjoint partition (S1, S2), computes aggregate spatial
        features for each group and evaluates the relation on them.
        """
        result: InsideTable = {}
        for s1, v1 in t1.items():
            for s2, v2 in t2.items():
                if s1.isdisjoint(s2):
                    key = s1 | s2
                    # Compute aggregate features for each group
                    agg1 = aggregate(env, s1)
                    agg2 = aggregate(env, s2)
                    # Evaluate spatial relation on aggregates
                    rel_score = compute_relation(name, agg1, agg2, params)
                    val = v1 * v2 * rel_score
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
        group_rel: _group_rel,
        choice: _choice,
        score: _score,
    }


def get_class_score(table: InsideTable, n_primitives: int) -> Tensor:
    """Extract the class score from an inside table.

    Prefers the full-set entry (exact marginalization over all primitives).
    If the full set is unreachable (e.g. max_depth too shallow), falls back
    to summing all table entries — this gives a partial marginalization that
    still maintains gradients.
    """
    full_set = frozenset(range(n_primitives))
    if full_set in table:
        return table[full_set]
    # Full set unreachable: sum all entries as partial marginalization
    if table:
        return sum(table.values())  # type: ignore[return-value]
    return torch.tensor(0.0)

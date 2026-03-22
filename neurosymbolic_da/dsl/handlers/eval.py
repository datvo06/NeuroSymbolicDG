"""Direct evaluation handler.

Maps DSL ops to their concrete implementations given an environment
of detected primitives and learnable relation parameters.
"""

from torch import Tensor

from effectful.ops.types import Interpretation

from neurosymbolic_da.dsl.ops import choice, conj, group_rel, has, rel, score
from neurosymbolic_da.dsl.primitives import Env
from neurosymbolic_da.dsl.relations import RelationParams, compute_relation


def make_eval_handler(env: Env, params: RelationParams) -> Interpretation:
    """Create a handler that directly evaluates DSL ops.

    Args:
        env: maps type_idx -> Primitive for all detected primitives
        params: learnable spatial relation parameters
    """

    def _has(type_idx: int) -> Tensor:
        return env[type_idx].conf

    def _rel(name: str, a: int, b: int) -> Tensor:
        return compute_relation(name, env[a], env[b], params)

    def _conj(c1: Tensor, c2: Tensor) -> Tensor:
        return c1 * c2

    def _group_rel(name: str, g1: Tensor, g2: Tensor) -> Tensor:
        # In eval mode, group_rel is just conj — the relation is already
        # folded into the group scores. This is a simplification; the
        # proper hierarchical evaluation uses the inside handler.
        return g1 * g2

    def _choice(*alternatives: Tensor) -> Tensor:
        return sum(alternatives)  # type: ignore[return-value]

    def _score(weight: Tensor, body: Tensor) -> Tensor:
        return weight * body

    return {
        has: _has,
        rel: _rel,
        conj: _conj,
        group_rel: _group_rel,
        choice: _choice,
        score: _score,
    }

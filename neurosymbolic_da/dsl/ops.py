"""Core DSL operations as effectful ops.

All ops take primitive type indices (int), not Primitive objects.
When unhandled, they produce symbolic Terms; handlers give them concrete semantics.

Note on relations: only `above` and `left_of` are defined (not `below`/`right_of`)
because the grammar enumerates all ordered pairs (i,j) with i!=j, so
above(b,a) = below(a,b) and left_of(b,a) = right_of(a,b).
"""

from torch import Tensor

from effectful.ops.types import NotHandled, Operation

defop = Operation.define


@defop
def has(type_idx: int) -> Tensor:
    """Primitive existence check — returns confidence score."""
    raise NotHandled


@defop
def rel(name: str, a: int, b: int) -> Tensor:
    """Spatial relation between primitives a and b.

    name: one of 'above', 'left_of', 'aligned_h', 'aligned_v', 'near', 'contains'
    a, b: primitive type indices
    """
    raise NotHandled


@defop
def conj(c1: Tensor, c2: Tensor) -> Tensor:
    """Conjunction — semiring multiplication."""
    raise NotHandled


@defop
def choice(*alternatives: Tensor) -> Tensor:
    """Marginalization — semiring addition over alternatives."""
    raise NotHandled


@defop
def score(weight: Tensor, body: Tensor) -> Tensor:
    """Weighted scoring — semiring multiplication by weight."""
    raise NotHandled


@defop
def group_rel(name: str, g1: Tensor, g2: Tensor) -> Tensor:
    """Spatial relation between two SubLayouts (groups of primitives).

    Under the inside handler, g1 and g2 are InsideTables.
    The handler computes aggregate spatial features for each group
    and evaluates the relation on the aggregates.
    """
    raise NotHandled

from dataclasses import dataclass

from torch import Tensor


@dataclass
class Primitive:
    """A detected spatial primitive with location, bounding box, and confidence."""

    cx: Tensor  # center x
    cy: Tensor  # center y
    x1: Tensor  # bbox top-left x
    y1: Tensor  # bbox top-left y
    x2: Tensor  # bbox bottom-right x
    y2: Tensor  # bbox bottom-right y
    conf: Tensor  # detection confidence
    type_idx: int  # primitive type index


# Maps type_idx -> Primitive for all detected primitives in an image.
Env = dict[int, Primitive]

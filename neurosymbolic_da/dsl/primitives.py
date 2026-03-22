from dataclasses import dataclass

import torch
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


@dataclass
class SubLayout:
    """Aggregate spatial features of a group of primitives.

    Centroid is confidence-weighted mean of members.
    Bounding box is the enclosing box of all members.
    """

    cx: Tensor
    cy: Tensor
    x1: Tensor
    y1: Tensor
    x2: Tensor
    y2: Tensor
    conf: Tensor  # sum of member confidences
    members: frozenset[int]


# Spatial — anything with cx, cy, x1, y1, x2, y2, conf fields.
# Both Primitive and SubLayout satisfy this.
Spatial = Primitive | SubLayout

# Maps type_idx -> Primitive for all detected primitives in an image.
Env = dict[int, Primitive]

# Batched env: Primitive fields have shape [B] instead of scalar.
BatchedEnv = dict[int, Primitive]


def aggregate(env: Env, members: frozenset[int]) -> SubLayout:
    """Compute aggregate spatial features for a group of primitives.

    Centroid: confidence-weighted mean of member centroids.
    Bbox: enclosing box of all member bboxes.
    Conf: sum of member confidences.
    """
    confs = torch.stack([env[j].conf for j in members])
    cxs = torch.stack([env[j].cx for j in members])
    cys = torch.stack([env[j].cy for j in members])

    total_conf = confs.sum(dim=0)
    # Avoid division by zero
    w = confs / (total_conf + 1e-8)

    return SubLayout(
        cx=(w * cxs).sum(dim=0),
        cy=(w * cys).sum(dim=0),
        x1=torch.stack([env[j].x1 for j in members]).min(dim=0).values,
        y1=torch.stack([env[j].y1 for j in members]).min(dim=0).values,
        x2=torch.stack([env[j].x2 for j in members]).max(dim=0).values,
        y2=torch.stack([env[j].y2 for j in members]).max(dim=0).values,
        conf=total_conf,
        members=members,
    )

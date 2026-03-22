"""Synthetic Compositional Benchmark (SCB) — Section 6.4.

Generates images composed of k geometric parts with specific spatial
relations. Three domain-shift conditions:

  (A) Structure-preserving: same layout, different part appearance
  (B) Appearance-preserving: same parts, different layout
  (C) Both shift: different parts AND different layout

This benchmark isolates the compositional transfer advantage.
"""

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Part shapes
SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star"]

# Style palettes — each domain uses a different palette
PALETTES = {
    "domain_a": [
        (0.9, 0.2, 0.2),  # red
        (0.2, 0.7, 0.2),  # green
        (0.2, 0.2, 0.9),  # blue
        (0.9, 0.9, 0.2),  # yellow
        (0.9, 0.2, 0.9),  # magenta
        (0.2, 0.9, 0.9),  # cyan
    ],
    "domain_b": [
        (0.6, 0.3, 0.1),  # brown
        (0.1, 0.4, 0.3),  # teal
        (0.4, 0.1, 0.5),  # purple
        (0.5, 0.5, 0.1),  # olive
        (0.7, 0.4, 0.3),  # salmon
        (0.3, 0.3, 0.6),  # slate blue
    ],
}


@dataclass
class PartSpec:
    """Specification for a single part."""

    shape_idx: int  # which shape
    cx: float  # center x (0-1)
    cy: float  # center y (0-1)
    size: float  # radius (0-1)
    color: tuple[float, float, float]  # RGB (0-1)


@dataclass
class ClassLayout:
    """Defines a class's spatial composition."""

    name: str
    part_positions: list[tuple[float, float]]  # relative positions (0-1)
    part_shapes: list[int]  # shape indices
    relations: list[str]  # human-readable description


@dataclass
class HierarchicalClassLayout:
    """Defines a class with explicit group structure.

    Each class has two groups of parts with intra-group and inter-group
    spatial relations. This makes the hierarchical grammar necessary.
    """

    name: str
    group_a_positions: list[tuple[float, float]]
    group_b_positions: list[tuple[float, float]]
    group_a_shapes: list[int]
    group_b_shapes: list[int]
    inter_group_relation: str  # e.g., "above", "left_of"


def _make_hierarchical_layouts() -> list[HierarchicalClassLayout]:
    """Create class layouts with explicit 2-group hierarchical structure.

    Each class has 4 parts: group A (parts 0,1) and group B (parts 2,3).
    Inter-group relation describes how group A relates to group B spatially.
    """
    return [
        HierarchicalClassLayout(
            name="top_pair_above_bottom_pair",
            group_a_positions=[(0.3, 0.25), (0.7, 0.25)],
            group_b_positions=[(0.3, 0.75), (0.7, 0.75)],
            group_a_shapes=[0, 1],
            group_b_shapes=[2, 3],
            inter_group_relation="above",
        ),
        HierarchicalClassLayout(
            name="left_pair_leftof_right_pair",
            group_a_positions=[(0.2, 0.3), (0.2, 0.7)],
            group_b_positions=[(0.8, 0.3), (0.8, 0.7)],
            group_a_shapes=[0, 2],
            group_b_shapes=[1, 3],
            inter_group_relation="left_of",
        ),
        HierarchicalClassLayout(
            name="inner_pair_contained_by_outer_pair",
            group_a_positions=[(0.4, 0.4), (0.6, 0.6)],
            group_b_positions=[(0.2, 0.2), (0.8, 0.8)],
            group_a_shapes=[0, 1],
            group_b_shapes=[2, 3],
            inter_group_relation="near",
        ),
        HierarchicalClassLayout(
            name="diagonal_groups",
            group_a_positions=[(0.2, 0.2), (0.4, 0.4)],
            group_b_positions=[(0.6, 0.6), (0.8, 0.8)],
            group_a_shapes=[0, 3],
            group_b_shapes=[1, 2],
            inter_group_relation="above",
        ),
        HierarchicalClassLayout(
            name="horizontal_aligned_groups",
            group_a_positions=[(0.3, 0.3), (0.3, 0.7)],
            group_b_positions=[(0.7, 0.3), (0.7, 0.7)],
            group_a_shapes=[0, 1],
            group_b_shapes=[3, 2],
            inter_group_relation="aligned_h",
        ),
        HierarchicalClassLayout(
            name="vertical_aligned_groups",
            group_a_positions=[(0.3, 0.3), (0.7, 0.3)],
            group_b_positions=[(0.3, 0.7), (0.7, 0.7)],
            group_a_shapes=[1, 0],
            group_b_shapes=[2, 3],
            inter_group_relation="aligned_v",
        ),
        HierarchicalClassLayout(
            name="compact_near_groups",
            group_a_positions=[(0.25, 0.4), (0.35, 0.6)],
            group_b_positions=[(0.65, 0.4), (0.75, 0.6)],
            group_a_shapes=[0, 2],
            group_b_shapes=[3, 1],
            inter_group_relation="left_of",
        ),
        HierarchicalClassLayout(
            name="scattered_groups",
            group_a_positions=[(0.2, 0.5), (0.5, 0.2)],
            group_b_positions=[(0.8, 0.5), (0.5, 0.8)],
            group_a_shapes=[1, 3],
            group_b_shapes=[0, 2],
            inter_group_relation="near",
        ),
    ]


def _make_default_layouts(n_parts: int = 4) -> list[ClassLayout]:
    """Create default class layouts with distinct spatial compositions."""
    layouts = [
        ClassLayout(
            name="vertical_stack",
            part_positions=[(0.5, 0.2), (0.5, 0.4), (0.5, 0.6), (0.5, 0.8)],
            part_shapes=[0, 1, 2, 3],
            relations=["above(0,1)", "above(1,2)", "above(2,3)", "aligned_v"],
        ),
        ClassLayout(
            name="horizontal_row",
            part_positions=[(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)],
            part_shapes=[0, 1, 2, 3],
            relations=["left_of(0,1)", "left_of(1,2)", "left_of(2,3)", "aligned_h"],
        ),
        ClassLayout(
            name="grid_2x2",
            part_positions=[(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)],
            part_shapes=[0, 1, 2, 3],
            relations=["above(0,2)", "above(1,3)", "left_of(0,1)", "left_of(2,3)"],
        ),
        ClassLayout(
            name="diagonal",
            part_positions=[(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8)],
            part_shapes=[0, 1, 2, 3],
            relations=["above(0,1)", "left_of(0,1)", "near(1,2)", "near(2,3)"],
        ),
        ClassLayout(
            name="T_shape",
            part_positions=[(0.3, 0.3), (0.5, 0.3), (0.7, 0.3), (0.5, 0.7)],
            part_shapes=[0, 1, 2, 3],
            relations=["left_of(0,1)", "left_of(1,2)", "aligned_h(0,1,2)", "above(1,3)"],
        ),
        ClassLayout(
            name="L_shape",
            part_positions=[(0.3, 0.3), (0.3, 0.5), (0.3, 0.7), (0.6, 0.7)],
            part_shapes=[0, 1, 2, 3],
            relations=["above(0,1)", "above(1,2)", "aligned_v(0,1,2)", "left_of(2,3)"],
        ),
        ClassLayout(
            name="cross_shape",
            part_positions=[(0.5, 0.3), (0.3, 0.5), (0.7, 0.5), (0.5, 0.7)],
            part_shapes=[0, 1, 2, 3],
            relations=["above(0,3)", "left_of(1,2)", "near(0,1)", "near(0,2)"],
        ),
        ClassLayout(
            name="triangle_formation",
            part_positions=[(0.5, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.55)],
            part_shapes=[0, 1, 2, 3],
            relations=["above(0,1)", "above(0,2)", "left_of(1,2)", "contains(0,3)"],
        ),
    ]
    return layouts[:min(len(layouts), 8)]


def _draw_shape(
    canvas: Tensor,
    shape_idx: int,
    cx: float,
    cy: float,
    size: float,
    color: tuple[float, float, float],
) -> None:
    """Draw a filled shape onto a canvas tensor [3, H, W] in-place."""
    _, H, W = canvas.shape
    px, py = int(cx * W), int(cy * H)
    r = max(int(size * min(H, W)), 2)

    # Compute bounding box
    y_min, y_max = max(0, py - r), min(H, py + r + 1)
    x_min, x_max = max(0, px - r), min(W, px + r + 1)
    if y_min >= y_max or x_min >= x_max:
        return

    # Create coordinate grids relative to center
    dy = torch.arange(y_min - py, y_max - py, dtype=torch.float32)
    dx = torch.arange(x_min - px, x_max - px, dtype=torch.float32)
    gy, gx = torch.meshgrid(dy, dx, indexing="ij")

    # Compute mask based on shape
    match shape_idx % 6:
        case 0:  # circle
            mask = gx * gx + gy * gy <= r * r
        case 1:  # square
            mask = (gx.abs() <= r * 0.7) & (gy.abs() <= r * 0.7)
        case 2:  # triangle
            mask = (gy >= 0) & (gx.abs() <= r * (1.0 - gy / r))
        case 3:  # diamond
            mask = gx.abs() + gy.abs() <= r
        case 4:  # cross
            mask = (gx.abs() <= r * 0.25) | (gy.abs() <= r * 0.25)
        case 5:  # star
            mask = (
                (gx.abs() + gy.abs() <= r)
                | ((gx.abs() <= r * 0.25) & (gy.abs() <= r))
                | ((gx.abs() <= r) & (gy.abs() <= r * 0.25))
            )

    # Apply color where mask is True
    for c in range(3):
        canvas[c, y_min:y_max, x_min:x_max][mask] = color[c]


def _add_noise(canvas: Tensor, noise_std: float = 0.05) -> Tensor:
    """Add Gaussian noise and clamp to [0, 1]."""
    return (canvas + torch.randn_like(canvas) * noise_std).clamp(0, 1)


def _perturb_positions(
    positions: list[tuple[float, float]], jitter: float = 0.05
) -> list[tuple[float, float]]:
    """Add small random jitter to part positions."""
    return [
        (
            max(0.1, min(0.9, x + random.uniform(-jitter, jitter))),
            max(0.1, min(0.9, y + random.uniform(-jitter, jitter))),
        )
        for x, y in positions
    ]


def _shuffle_layout_positions(
    positions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Generate a random permutation of positions (breaks spatial structure)."""
    new_positions = []
    for _ in positions:
        new_positions.append(
            (random.uniform(0.15, 0.85), random.uniform(0.15, 0.85))
        )
    return new_positions


class SCBDataset(Dataset):
    """Synthetic Compositional Benchmark dataset.

    Args:
        n_classes: number of classes (up to 8)
        n_samples_per_class: samples per class
        n_parts: parts per image
        image_size: output image size
        domain: domain name for palette selection
        condition: shift condition for target domain
            - "source": original layout + original palette
            - "A": structure-preserving (same layout, different palette)
            - "B": appearance-preserving (same palette, shuffled layout)
            - "C": both (different palette + shuffled layout)
            - "D_source": hierarchical source (grouped layout + source palette)
            - "D": hierarchical structure-preserving (same grouped layout, different palette)
            - "E": hierarchical appearance-preserving (same palette, shuffled groups)
            - "F": hierarchical both (different palette + shuffled groups)
        part_size: size of each part
        jitter: position jitter
        noise_std: background noise
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        n_classes: int = 8,
        n_samples_per_class: int = 200,
        n_parts: int = 4,
        image_size: int = 224,
        domain: str = "source",
        condition: str = "source",
        part_size: float = 0.06,
        jitter: float = 0.04,
        noise_std: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class
        self.n_parts = n_parts
        self.image_size = image_size
        self.condition = condition
        self.part_size = part_size
        self.jitter = jitter
        self.noise_std = noise_std

        self.hierarchical = condition.startswith("D") or condition in ("E", "F")

        if self.hierarchical:
            self.hier_layouts = _make_hierarchical_layouts()[:n_classes]
            self.layouts = None  # not used
        else:
            self.layouts = _make_default_layouts(n_parts)[:n_classes]
            self.hier_layouts = None

        # Select palette based on condition
        if condition in ("source", "B", "D_source", "E"):
            self.palette = PALETTES["domain_a"]
        else:  # "A", "C", "D", "F" — different appearance
            self.palette = PALETTES["domain_b"]

        self.shuffle_layout = condition in ("B", "C", "E", "F")

        # Pre-generate all data (seed both random and torch for reproducibility)
        rng = random.Random(seed)
        torch_gen = torch.Generator().manual_seed(seed)
        self.images: list[Tensor] = []
        self.labels: list[int] = []

        for class_idx in range(n_classes):
            for _ in range(n_samples_per_class):
                if self.hierarchical:
                    img = self._generate_hierarchical_image(
                        self.hier_layouts[class_idx], class_idx, rng, torch_gen
                    )
                else:
                    img = self._generate_image(
                        self.layouts[class_idx], class_idx, rng, torch_gen
                    )
                self.images.append(img)
                self.labels.append(class_idx)

    def _generate_image(
        self, layout: ClassLayout, class_idx: int, rng: random.Random,
        torch_gen: torch.Generator | None = None,
    ) -> Tensor:
        """Generate a single image from a layout spec."""
        canvas = torch.full(
            (3, self.image_size, self.image_size), 0.1
        )  # dark background

        positions = list(layout.part_positions)

        if self.shuffle_layout:
            # Condition B or C: randomize positions (breaks structure)
            positions = [
                (
                    rng.uniform(0.15, 0.85),
                    rng.uniform(0.15, 0.85),
                )
                for _ in positions
            ]
        else:
            # Add jitter but preserve structure
            positions = [
                (
                    max(0.1, min(0.9, x + rng.uniform(-self.jitter, self.jitter))),
                    max(0.1, min(0.9, y + rng.uniform(-self.jitter, self.jitter))),
                )
                for x, y in positions
            ]

        # Draw parts
        for i, (px, py) in enumerate(positions):
            shape_idx = layout.part_shapes[i % len(layout.part_shapes)]
            color = self.palette[shape_idx % len(self.palette)]
            _draw_shape(
                canvas, shape_idx, px, py, self.part_size, color
            )

        # Add noise
        noise = torch.randn(canvas.shape, generator=torch_gen)
        canvas = canvas + noise * self.noise_std
        canvas = canvas.clamp(0, 1)

        return canvas

    def _generate_hierarchical_image(
        self, layout: HierarchicalClassLayout, class_idx: int,
        rng: random.Random, torch_gen: torch.Generator | None = None,
    ) -> Tensor:
        """Generate image from hierarchical layout with group structure."""
        canvas = torch.full(
            (3, self.image_size, self.image_size), 0.1
        )

        if self.shuffle_layout:
            # Shuffle: randomize group positions (breaks inter-group structure)
            # But preserve intra-group relative positions (offset the group)
            offset_a = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
            offset_b = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
            pos_a = [
                (max(0.1, min(0.9, x + offset_a[0] + rng.uniform(-self.jitter, self.jitter))),
                 max(0.1, min(0.9, y + offset_a[1] + rng.uniform(-self.jitter, self.jitter))))
                for x, y in layout.group_a_positions
            ]
            pos_b = [
                (max(0.1, min(0.9, x + offset_b[0] + rng.uniform(-self.jitter, self.jitter))),
                 max(0.1, min(0.9, y + offset_b[1] + rng.uniform(-self.jitter, self.jitter))))
                for x, y in layout.group_b_positions
            ]
        else:
            pos_a = [
                (max(0.1, min(0.9, x + rng.uniform(-self.jitter, self.jitter))),
                 max(0.1, min(0.9, y + rng.uniform(-self.jitter, self.jitter))))
                for x, y in layout.group_a_positions
            ]
            pos_b = [
                (max(0.1, min(0.9, x + rng.uniform(-self.jitter, self.jitter))),
                 max(0.1, min(0.9, y + rng.uniform(-self.jitter, self.jitter))))
                for x, y in layout.group_b_positions
            ]

        # Draw group A parts
        for i, (px, py) in enumerate(pos_a):
            shape_idx = layout.group_a_shapes[i]
            color = self.palette[shape_idx % len(self.palette)]
            _draw_shape(canvas, shape_idx, px, py, self.part_size, color)

        # Draw group B parts
        for i, (px, py) in enumerate(pos_b):
            shape_idx = layout.group_b_shapes[i]
            color = self.palette[shape_idx % len(self.palette)]
            _draw_shape(canvas, shape_idx, px, py, self.part_size, color)

        noise = torch.randn(canvas.shape, generator=torch_gen)
        canvas = canvas + noise * self.noise_std
        return canvas.clamp(0, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self.images[idx], self.labels[idx]


def get_scb_loaders(
    n_classes: int = 8,
    n_samples_per_class: int = 200,
    n_parts: int = 4,
    image_size: int = 224,
    condition: str = "A",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get SCB train/test loaders for source and target domains.

    Args:
        condition: "A"/"B"/"C" (flat) or "D"/"E"/"F" (hierarchical)
        Other args control dataset generation.

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    n_train = int(n_samples_per_class * 0.8)
    n_test = n_samples_per_class - n_train

    # For hierarchical conditions, source uses D_source
    source_cond = "D_source" if condition in ("D", "E", "F") else "source"

    src_train = SCBDataset(
        n_classes=n_classes,
        n_samples_per_class=n_train,
        n_parts=n_parts,
        image_size=image_size,
        condition=source_cond,
        seed=seed,
    )
    src_test = SCBDataset(
        n_classes=n_classes,
        n_samples_per_class=n_test,
        n_parts=n_parts,
        image_size=image_size,
        condition=source_cond,
        seed=seed + 1,
    )
    tgt_train = SCBDataset(
        n_classes=n_classes,
        n_samples_per_class=n_train,
        n_parts=n_parts,
        image_size=image_size,
        condition=condition,
        seed=seed + 2,
    )
    tgt_test = SCBDataset(
        n_classes=n_classes,
        n_samples_per_class=n_test,
        n_parts=n_parts,
        image_size=image_size,
        condition=condition,
        seed=seed + 3,
    )

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(src_train, shuffle=True, **kwargs),
        DataLoader(src_test, shuffle=False, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )

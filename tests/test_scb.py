"""Test Synthetic Compositional Benchmark (SCB)."""

import torch

from neurosymbolic_da.data.scb import (
    SCBDataset,
    _draw_shape,
    _make_default_layouts,
    get_scb_loaders,
)


def test_default_layouts():
    layouts = _make_default_layouts(n_parts=4)
    assert len(layouts) == 8
    for layout in layouts:
        assert len(layout.part_positions) == 4
        assert len(layout.part_shapes) == 4
        assert len(layout.relations) > 0


def test_draw_shape_circle():
    canvas = torch.zeros(3, 64, 64)
    _draw_shape(canvas, 0, 0.5, 0.5, 0.1, (1.0, 0.0, 0.0))
    # Should have drawn some red pixels
    assert canvas[0].sum() > 0
    assert canvas[1].sum() == 0  # no green
    assert canvas[2].sum() == 0  # no blue


def test_draw_shape_all_types():
    """All 6 shape types should draw without error."""
    for shape_idx in range(6):
        canvas = torch.zeros(3, 64, 64)
        _draw_shape(canvas, shape_idx, 0.5, 0.5, 0.1, (1.0, 1.0, 1.0))
        assert canvas.sum() > 0, f"Shape {shape_idx} drew nothing"


def test_scb_dataset_source():
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="source", seed=42,
    )
    assert len(ds) == 20  # 4 classes * 5 samples
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= label < 4
    assert img.min() >= 0 and img.max() <= 1


def test_scb_dataset_condition_A():
    """Condition A: same layout, different palette."""
    src = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="source", seed=42,
    )
    tgt = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="A", seed=42,
    )
    assert len(tgt) == 20
    # Images should differ (different palette) but same number
    img_src, _ = src[0]
    img_tgt, _ = tgt[0]
    # They use different palettes, so pixel values should differ
    assert not torch.allclose(img_src, img_tgt, atol=0.2)


def test_scb_dataset_condition_B():
    """Condition B: same palette, shuffled layout."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="B", seed=42,
    )
    assert len(ds) == 20
    img, label = ds[0]
    assert img.shape == (3, 64, 64)


def test_scb_dataset_condition_C():
    """Condition C: different palette + shuffled layout."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="C", seed=42,
    )
    assert len(ds) == 20


def test_scb_label_distribution():
    """Labels should be evenly distributed across classes."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=10, n_parts=4,
        image_size=64, condition="source", seed=42,
    )
    labels = [ds[i][1] for i in range(len(ds))]
    for c in range(4):
        assert labels.count(c) == 10


def test_scb_reproducibility():
    """Same seed should produce same images."""
    ds1 = SCBDataset(
        n_classes=2, n_samples_per_class=3, image_size=64,
        condition="source", seed=123,
    )
    ds2 = SCBDataset(
        n_classes=2, n_samples_per_class=3, image_size=64,
        condition="source", seed=123,
    )
    for i in range(len(ds1)):
        img1, l1 = ds1[i]
        img2, l2 = ds2[i]
        assert torch.equal(img1, img2)
        assert l1 == l2


def test_scb_loaders():
    src_train, src_test, tgt_train, tgt_test = get_scb_loaders(
        n_classes=4, n_samples_per_class=10, n_parts=4,
        image_size=64, condition="A", batch_size=4,
        num_workers=0, seed=42,
    )
    # Train: 80% of 10 = 8 per class, 4 classes = 32
    assert len(src_train.dataset) == 32
    # Test: 20% of 10 = 2 per class, 4 classes = 8
    assert len(src_test.dataset) == 8

    batch_x, batch_y = next(iter(src_train))
    assert batch_x.shape == (4, 3, 64, 64)
    assert batch_y.shape == (4,)


def test_scb_compatible_with_pipeline():
    """SCB images should work with the NeuroSymbolic pipeline."""
    from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline

    ds = SCBDataset(
        n_classes=4, n_samples_per_class=2, n_parts=4,
        image_size=224, condition="source", seed=42,
    )
    model = NeuroSymbolicPipeline(
        n_primitives=4, n_classes=4,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=False,
    )
    img, label = ds[0]
    log_probs = model(img.unsqueeze(0))
    assert log_probs.shape == (1, 4)


# --- Hierarchical SCB tests ---


def test_scb_hierarchical_source():
    """D_source condition produces valid hierarchical images."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="D_source", seed=42,
    )
    assert len(ds) == 20
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= label < 4


def test_scb_hierarchical_condition_D():
    """Condition D: same grouped layout, different palette."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="D", seed=42,
    )
    assert len(ds) == 20


def test_scb_hierarchical_condition_E():
    """Condition E: same palette, shuffled group positions."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="E", seed=42,
    )
    assert len(ds) == 20


def test_scb_hierarchical_condition_F():
    """Condition F: different palette + shuffled groups."""
    ds = SCBDataset(
        n_classes=4, n_samples_per_class=5, n_parts=4,
        image_size=64, condition="F", seed=42,
    )
    assert len(ds) == 20


def test_scb_hierarchical_loaders():
    """Hierarchical condition D uses D_source for source domain."""
    src_train, src_test, tgt_train, tgt_test = get_scb_loaders(
        n_classes=4, n_samples_per_class=10, n_parts=4,
        image_size=64, condition="D", batch_size=4,
        num_workers=0, seed=42,
    )
    assert len(src_train.dataset) == 32
    assert src_train.dataset.hierarchical is True
    assert tgt_train.dataset.hierarchical is True

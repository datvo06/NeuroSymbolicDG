"""Test shared dataset loading utilities."""

import pytest

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes


def test_get_n_classes():
    assert get_n_classes("digits") == 10
    assert get_n_classes("office31") == 31
    assert get_n_classes("officehome") == 65
    assert get_n_classes("scb") == 8


def test_get_n_classes_unknown():
    with pytest.raises(ValueError):
        get_n_classes("unknown")


def test_get_loaders_scb():
    """SCB loaders should work via get_loaders."""
    src_train, src_test, tgt_train, tgt_test = get_loaders(
        "scb", "source", "A",
        scb_n_classes=4, scb_n_samples=10, scb_image_size=64,
        num_workers=0,
    )
    assert len(src_train.dataset) == 32  # 4 classes * 8 train
    assert len(tgt_test.dataset) == 8    # 4 classes * 2 test

    batch_x, batch_y = next(iter(src_train))
    assert batch_x.shape[1:] == (3, 64, 64)

"""Test data loading modules."""

from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from neurosymbolic_da.data.digits import (
    _GRAYSCALE_TRANSFORM,
    _SVHN_TRANSFORM,
    get_digit_loaders,
)
from neurosymbolic_da.data.office import (
    OFFICE31_DOMAINS,
    OFFICEHOME_DOMAINS,
    _default_transform,
)


def test_grayscale_transform_shape():
    """Grayscale transform should produce 3-channel 224x224 images."""
    from PIL import Image
    img = Image.new("L", (28, 28), 128)  # MNIST-like
    tensor = _GRAYSCALE_TRANSFORM(img)
    assert tensor.shape == (3, 224, 224)


def test_svhn_transform_shape():
    """SVHN transform should produce 3-channel 224x224 images."""
    from PIL import Image
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    tensor = _SVHN_TRANSFORM(img)
    assert tensor.shape == (3, 224, 224)


def test_office_train_transform():
    """Train transform should include random augmentations."""
    t = _default_transform(224, train=True)
    from PIL import Image
    img = Image.new("RGB", (300, 300), (128, 128, 128))
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


def test_office_test_transform():
    """Test transform should use center crop."""
    t = _default_transform(224, train=False)
    from PIL import Image
    img = Image.new("RGB", (300, 300), (128, 128, 128))
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


def test_office31_domains():
    assert "amazon" in OFFICE31_DOMAINS
    assert "dslr" in OFFICE31_DOMAINS
    assert "webcam" in OFFICE31_DOMAINS


def test_officehome_domains():
    assert "Art" in OFFICEHOME_DOMAINS
    assert "Clipart" in OFFICEHOME_DOMAINS
    assert "Product" in OFFICEHOME_DOMAINS
    assert "Real_World" in OFFICEHOME_DOMAINS


def test_digit_loaders_with_mock():
    """Test get_digit_loaders with mocked datasets."""
    fake_ds = TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,)))

    with patch("neurosymbolic_da.data.digits.get_mnist", return_value=fake_ds), \
         patch("neurosymbolic_da.data.digits.get_usps", return_value=fake_ds):
        src_train, src_test, tgt_train, tgt_test = get_digit_loaders(
            "mnist", "usps", batch_size=16, num_workers=0,
        )
        batch_x, batch_y = next(iter(src_train))
        assert batch_x.shape[0] <= 16
        assert batch_x.shape[1:] == (3, 224, 224)

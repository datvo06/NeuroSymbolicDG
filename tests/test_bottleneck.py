"""Test concept bottleneck: heatmap generation and primitive extraction."""

import torch

from neurosymbolic_da.nn.bottleneck import ConceptBottleneck
from neurosymbolic_da.nn.hourglass_bottleneck import HourglassBottleneck


def test_bottleneck_output_shapes():
    bn = ConceptBottleneck(in_channels=512, n_primitives=5)
    features = torch.randn(2, 512, 7, 7)
    heatmaps, coords, env = bn(features)

    assert heatmaps.shape == (2, 5, 7, 7)
    assert coords.shape == (2, 5, 2)
    # env is built from first image in batch
    assert len(env) == 5


def test_coords_in_range():
    """Soft-argmax coords should be in [-1, 1]."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=3)
    features = torch.randn(1, 512, 7, 7)
    _, coords, _ = bn(features)
    assert coords.min() >= -1.0
    assert coords.max() <= 1.0


def test_confidence_in_range():
    """Confidence (sigmoid of max activation) should be in (0, 1)."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=3)
    features = torch.randn(1, 512, 7, 7)
    _, _, env = bn(features)
    for prim in env.values():
        assert 0.0 < prim.conf.item() < 1.0


def test_env_has_all_primitives():
    """Env should contain one Primitive per type_idx."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=4)
    features = torch.randn(1, 512, 7, 7)
    _, _, env = bn(features)
    assert set(env.keys()) == {0, 1, 2, 3}
    for j, prim in env.items():
        assert prim.type_idx == j


def test_bbox_contains_center():
    """Bounding box should contain the center point."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=3)
    features = torch.randn(1, 512, 7, 7)
    _, _, env = bn(features)
    for prim in env.values():
        assert prim.x1.item() <= prim.cx.item() <= prim.x2.item()
        assert prim.y1.item() <= prim.cy.item() <= prim.y2.item()


def test_gradient_flow_through_bottleneck():
    """Gradients should flow from primitives back through the bottleneck."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=3)
    features = torch.randn(1, 512, 7, 7, requires_grad=True)
    heatmaps, coords, env = bn(features)

    # Loss on coordinates
    loss = coords.sum()
    loss.backward()
    assert features.grad is not None
    assert features.grad.abs().sum() > 0


def test_gradient_flow_through_confidence():
    """Gradients should flow from confidence back through the bottleneck."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=3)
    features = torch.randn(1, 512, 7, 7, requires_grad=True)
    _, _, env = bn(features)

    loss = sum(p.conf for p in env.values())
    loss.backward()
    assert features.grad is not None


def test_gradient_flow_through_bbox():
    """Gradients should flow from bbox back through the bottleneck."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=3)
    features = torch.randn(1, 512, 7, 7, requires_grad=True)
    _, _, env = bn(features)

    loss = sum(p.x2 - p.x1 + p.y2 - p.y1 for p in env.values())
    loss.backward()
    assert features.grad is not None


def test_extract_env_detached():
    """extract_env should work with manually provided tensors."""
    bn = ConceptBottleneck(in_channels=512, n_primitives=2)
    coords = torch.tensor([[0.3, -0.2], [0.5, 0.7]])
    conf = torch.tensor([0.9, 0.6])
    heatmaps = torch.randn(2, 7, 7)

    env = bn.extract_env(coords, conf, heatmaps)
    assert len(env) == 2
    assert torch.isclose(env[0].cx, torch.tensor(0.3))
    assert torch.isclose(env[1].conf, torch.tensor(0.6))


# --- Hourglass bottleneck tests ---

_R50_CHANNELS = {"layer2": 512, "layer3": 1024, "layer4": 2048}


def test_hourglass_output_shapes():
    bn = HourglassBottleneck(_R50_CHANNELS, n_primitives=5)
    features = {
        "layer2": torch.randn(2, 512, 28, 28),
        "layer3": torch.randn(2, 1024, 14, 14),
        "layer4": torch.randn(2, 2048, 7, 7),
    }
    heatmaps, coords, env = bn(features)

    assert heatmaps.shape == (2, 5, 14, 14)  # output at layer3 scale
    assert coords.shape == (2, 5, 2)
    assert len(env) == 5


def test_hourglass_gradient_flow():
    bn = HourglassBottleneck(_R50_CHANNELS, n_primitives=3)
    features = {
        "layer2": torch.randn(2, 512, 28, 28, requires_grad=True),
        "layer3": torch.randn(2, 1024, 14, 14, requires_grad=True),
        "layer4": torch.randn(2, 2048, 7, 7, requires_grad=True),
    }
    heatmaps, coords, _ = bn(features)
    loss = coords.sum() + heatmaps.sum()
    loss.backward()
    # Gradients should flow to all scales
    for name, feat in features.items():
        assert feat.grad is not None, f"No gradient for {name}"
        assert feat.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_hourglass_batched_env():
    bn = HourglassBottleneck(_R50_CHANNELS, n_primitives=4)
    features = {
        "layer2": torch.randn(3, 512, 28, 28),
        "layer3": torch.randn(3, 1024, 14, 14),
        "layer4": torch.randn(3, 2048, 7, 7),
    }
    heatmaps, coords, _ = bn(features)
    B, k = 3, 4
    conf = torch.sigmoid(heatmaps.view(B, k, -1).max(dim=-1).values)
    env = bn.extract_batched_env(coords, conf, heatmaps)
    assert len(env) == 4
    for j in range(4):
        assert env[j].cx.shape == (3,)
        assert env[j].conf.shape == (3,)

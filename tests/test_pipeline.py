"""Test the full neuro-symbolic pipeline."""

import torch

from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline


def test_pipeline_output_shape():
    model = NeuroSymbolicPipeline(
        n_primitives=3, n_classes=4,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=False,
    )
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    assert log_probs.shape == (2, 4)


def test_pipeline_log_probs_sum_to_one():
    model = NeuroSymbolicPipeline(
        n_primitives=2, n_classes=3,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=False,
    )
    x = torch.randn(1, 3, 224, 224)
    log_probs = model(x)
    probs = log_probs.exp()
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_pipeline_gradient_flow():
    """Gradients should flow from loss to all learnable parameters."""
    model = NeuroSymbolicPipeline(
        n_primitives=2, n_classes=3,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=False,
    )
    x = torch.randn(1, 3, 224, 224)
    log_probs = model(x)

    # Cross-entropy loss
    target = torch.tensor([1])
    loss = torch.nn.functional.nll_loss(log_probs, target)
    loss.backward()

    # Check key parameter groups got gradients
    assert model.grammar.log_weights.grad is not None
    assert model.bottleneck.heatmap_conv.weight.grad is not None
    assert model.relation_params.lambda_above.grad is not None


def test_pipeline_get_heatmaps():
    model = NeuroSymbolicPipeline(
        n_primitives=4, n_classes=2,
        backbone_variant="resnet18", pretrained_backbone=False,
    )
    x = torch.randn(2, 3, 224, 224)
    heatmaps = model.get_heatmaps(x)
    assert heatmaps.shape == (2, 4, 7, 7)


def test_pipeline_inside_mode():
    """Smoke test for inside algorithm mode."""
    model = NeuroSymbolicPipeline(
        n_primitives=2, n_classes=2,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=True,
    )
    x = torch.randn(1, 3, 224, 224)
    log_probs = model(x)
    assert log_probs.shape == (1, 2)

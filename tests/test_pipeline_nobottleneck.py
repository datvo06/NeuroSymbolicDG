"""Test NoBottleneck ablation pipeline."""

import torch

from neurosymbolic_da.nn.pipeline_nobottleneck import NoBottleneckPipeline


def _make_model(n_classes=3, n_primitives=2):
    return NoBottleneckPipeline(
        n_primitives=n_primitives,
        n_classes=n_classes,
        backbone_variant="resnet18",
        pretrained_backbone=False,
        max_depth=1,
        use_inside=False,
    )


def test_nobottleneck_output_shape():
    model = _make_model(n_classes=5, n_primitives=4)
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    assert log_probs.shape == (2, 5)


def test_nobottleneck_log_probs_sum_to_one():
    model = _make_model()
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_nobottleneck_gradient_flow():
    model = _make_model()
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    loss = -log_probs[:, 0].mean()
    loss.backward()

    # Backbone and projection should get gradients
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    assert backbone_has_grad
    assert model.proj.weight.grad is not None


def test_nobottleneck_has_grammar():
    """NoBottleneck should have grammar and relation_params."""
    model = _make_model()
    param_names = [name for name, _ in model.named_parameters()]
    assert any("grammar" in n for n in param_names)
    assert any("relation_params" in n for n in param_names)


def test_nobottleneck_no_bottleneck():
    """NoBottleneck should NOT have a bottleneck module."""
    model = _make_model()
    assert not hasattr(model, "bottleneck")


def test_nobottleneck_get_heatmaps():
    model = _make_model(n_primitives=4)
    x = torch.randn(2, 3, 224, 224)
    hm = model.get_heatmaps(x)
    assert hm.shape[0] == 2
    assert hm.shape[1] == 4

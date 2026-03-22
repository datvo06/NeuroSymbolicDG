"""Test NoPCFG ablation pipeline."""

import torch

from neurosymbolic_da.nn.pipeline_nopcfg import NoPCFGPipeline


def _make_model(n_classes=3, n_primitives=2):
    return NoPCFGPipeline(
        n_primitives=n_primitives,
        n_classes=n_classes,
        backbone_variant="resnet18",
        pretrained_backbone=False,
    )


def test_nopcfg_output_shape():
    model = _make_model(n_classes=5, n_primitives=4)
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    assert log_probs.shape == (2, 5)


def test_nopcfg_log_probs_sum_to_one():
    model = _make_model()
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_nopcfg_gradient_flow():
    model = _make_model()
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    loss = -log_probs[:, 0].mean()
    loss.backward()

    # Backbone, bottleneck, and classifier should all get gradients
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    assert backbone_has_grad
    assert model.bottleneck.heatmap_conv.weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_nopcfg_get_heatmaps():
    model = _make_model(n_primitives=4)
    x = torch.randn(2, 3, 224, 224)
    heatmaps = model.get_heatmaps(x)
    assert heatmaps.shape[0] == 2
    assert heatmaps.shape[1] == 4


def test_nopcfg_no_grammar_params():
    """NoPCFG should have no grammar or relation_params modules."""
    model = _make_model()
    param_names = [name for name, _ in model.named_parameters()]
    assert not any("grammar" in n for n in param_names)
    assert not any("relation_params" in n for n in param_names)


def test_nopcfg_fewer_params_than_pcfg():
    """NoPCFG should have fewer parameters than the full pipeline."""
    from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline

    nopcfg = _make_model(n_classes=3, n_primitives=2)
    full = NeuroSymbolicPipeline(
        n_primitives=2, n_classes=3,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=False,
    )
    n_nopcfg = sum(p.numel() for p in nopcfg.parameters())
    n_full = sum(p.numel() for p in full.parameters())
    # NoPCFG should be smaller (no grammar weights, no relation params)
    assert n_nopcfg < n_full

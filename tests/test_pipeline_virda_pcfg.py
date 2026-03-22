"""Test VirDA + PCFG ablation pipeline."""

import torch

from neurosymbolic_da.nn.pipeline_virda_pcfg import (
    VirDAPCFGPipeline,
    VisualReprogrammingLayer,
)


def _make_model(n_classes=3, n_primitives=2):
    return VirDAPCFGPipeline(
        n_primitives=n_primitives,
        n_classes=n_classes,
        backbone_variant="resnet18",
        pretrained_backbone=False,
        max_depth=1,
        use_inside=False,
        pad_size=10,
    )


def test_reprogramming_layer():
    layer = VisualReprogrammingLayer(image_size=64, pad_size=10)
    x = torch.randn(2, 3, 64, 64)
    out = layer(x)
    assert out.shape == x.shape
    # Center should be unchanged (mask is 0 there)
    assert torch.allclose(x[:, :, 10:-10, 10:-10], out[:, :, 10:-10, 10:-10])
    # Border should be different (perturbation applied)
    assert not torch.allclose(x[:, :, :10, :], out[:, :, :10, :])


def test_virda_pcfg_output_shape():
    model = _make_model(n_classes=5, n_primitives=4)
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    assert log_probs.shape == (2, 5)


def test_virda_pcfg_log_probs_sum_to_one():
    model = _make_model()
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_virda_pcfg_backbone_frozen():
    """Backbone should be frozen in VirDA + PCFG."""
    model = _make_model()
    for p in model.backbone.parameters():
        assert not p.requires_grad


def test_virda_pcfg_trainable_components():
    """Reprogramming, bottleneck, grammar should be trainable."""
    model = _make_model()
    assert model.reprogramming.perturbation.requires_grad
    assert model.bottleneck.heatmap_conv.weight.requires_grad
    assert model.grammar.log_weights.requires_grad


def test_virda_pcfg_gradient_flow():
    model = _make_model()
    x = torch.randn(2, 3, 224, 224)
    log_probs = model(x)
    loss = -log_probs[:, 0].mean()
    loss.backward()

    # Reprogramming and bottleneck should get gradients
    assert model.reprogramming.perturbation.grad is not None
    assert model.bottleneck.heatmap_conv.weight.grad is not None
    # Backbone should NOT get gradients (frozen)
    backbone_has_grad = any(
        p.grad is not None for p in model.backbone.parameters()
    )
    assert not backbone_has_grad


def test_virda_pcfg_get_heatmaps():
    model = _make_model(n_primitives=4)
    x = torch.randn(2, 3, 224, 224)
    hm = model.get_heatmaps(x)
    assert hm.shape[0] == 2
    assert hm.shape[1] == 4


def test_virda_pcfg_param_count():
    """VirDA+PCFG should have fewer trainable params than full pipeline."""
    from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline

    virda = _make_model(n_classes=3, n_primitives=2)
    full = NeuroSymbolicPipeline(
        n_primitives=2, n_classes=3,
        backbone_variant="resnet18", pretrained_backbone=False,
        max_depth=1, use_inside=False,
    )
    n_virda = sum(p.numel() for p in virda.parameters() if p.requires_grad)
    n_full = sum(p.numel() for p in full.parameters() if p.requires_grad)
    # VirDA has frozen backbone, so fewer trainable params
    assert n_virda < n_full

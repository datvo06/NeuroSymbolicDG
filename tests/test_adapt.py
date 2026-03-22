"""Test domain adaptation protocol."""

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import (
    adapt,
    adapt_epoch,
    freeze_structure,
    get_adaptable_params,
)


def _make_fake_loader(n_samples=16, n_classes=3, batch_size=8):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _make_model(n_classes=3):
    return NeuroSymbolicPipeline(
        n_primitives=2,
        n_classes=n_classes,
        backbone_variant="resnet18",
        pretrained_backbone=False,
        max_depth=1,
        use_inside=False,
    )


def test_freeze_structure():
    """freeze_structure should freeze grammar and relation params."""
    model = _make_model()
    freeze_structure(model)

    for p in model.grammar.parameters():
        assert not p.requires_grad
    for p in model.relation_params.parameters():
        assert not p.requires_grad
    # Backbone and bottleneck should still be trainable
    for p in model.backbone.parameters():
        assert p.requires_grad
    for p in model.bottleneck.parameters():
        assert p.requires_grad


def test_freeze_structure_unfreeze_grammar():
    """freeze_grammar=False should keep grammar trainable."""
    model = _make_model()
    freeze_structure(model, freeze_grammar=False)

    # Grammar should still be trainable
    for p in model.grammar.parameters():
        assert p.requires_grad
    # Relation params should still be frozen
    for p in model.relation_params.parameters():
        assert not p.requires_grad
    # Backbone and bottleneck should still be trainable
    for p in model.backbone.parameters():
        assert p.requires_grad


def test_get_adaptable_params_includes_grammar_when_unfrozen():
    """When grammar is unfrozen, get_adaptable_params should include it."""
    model = _make_model()
    freeze_structure(model, freeze_grammar=False)
    params = get_adaptable_params(model)

    grammar_params = list(model.grammar.parameters())
    # Grammar params should be in adaptable params
    for gp in grammar_params:
        assert any(p is gp for p in params)


def test_get_adaptable_params():
    model = _make_model()
    freeze_structure(model)
    params = get_adaptable_params(model)

    # Should only include backbone + bottleneck params
    assert len(params) > 0
    # All should require grad
    assert all(p.requires_grad for p in params)


def test_frozen_params_unchanged():
    """Frozen params should not change during adaptation."""
    model = _make_model()
    freeze_structure(model)

    grammar_before = model.grammar.log_weights.clone()
    rel_before = model.relation_params.lambda_above.clone()

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    adapt_epoch(model, src_loader, tgt_loader, optimizer, torch.device("cpu"))

    assert torch.equal(model.grammar.log_weights, grammar_before)
    assert torch.equal(model.relation_params.lambda_above, rel_before)


def test_adaptable_params_change():
    """Backbone and bottleneck params should change during adaptation."""
    model = _make_model()
    freeze_structure(model)

    bn_before = model.bottleneck.heatmap_conv.weight.data.clone()

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.1)

    adapt_epoch(model, src_loader, tgt_loader, optimizer, torch.device("cpu"))

    assert not torch.equal(model.bottleneck.heatmap_conv.weight.data, bn_before)


def test_adapt_epoch_returns_losses():
    model = _make_model()
    freeze_structure(model)

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    avg_mmd, avg_ent, avg_loss = adapt_epoch(
        model, src_loader, tgt_loader, optimizer, torch.device("cpu")
    )

    assert isinstance(avg_mmd, float)
    assert isinstance(avg_ent, float)
    assert isinstance(avg_loss, float)
    assert avg_mmd >= 0
    assert avg_ent >= 0


def test_adapt_loop():
    model = _make_model()
    freeze_structure(model)

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    tgt_test_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    metrics = adapt(
        model=model,
        source_loader=src_loader,
        target_loader=tgt_loader,
        target_test_loader=tgt_test_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        n_epochs=2,
        log_interval=100,
    )

    assert len(metrics.history) == 2
    assert metrics.epoch == 2
    assert all(h["target_acc"] >= 0 for h in metrics.history)


def test_adapt_with_save(tmp_path):
    model = _make_model()
    freeze_structure(model)

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    tgt_test_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)
    save_path = str(tmp_path / "adapted.pt")

    adapt(
        model=model,
        source_loader=src_loader,
        target_loader=tgt_loader,
        target_test_loader=tgt_test_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        n_epochs=2,
        log_interval=100,
        save_path=save_path,
    )

    checkpoint = torch.load(save_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "target_acc" in checkpoint


def test_target_loader_cycling():
    """When target loader is shorter than source, it should cycle."""
    model = _make_model()
    freeze_structure(model)

    # Source: 16 samples, Target: 4 samples (will need to cycle)
    src_loader = _make_fake_loader(n_samples=16, batch_size=8)
    tgt_loader = _make_fake_loader(n_samples=4, batch_size=4)
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    # Should not crash
    avg_mmd, avg_ent, avg_loss = adapt_epoch(
        model, src_loader, tgt_loader, optimizer, torch.device("cpu")
    )
    assert isinstance(avg_loss, float)


def test_adapt_epoch_im_loss():
    """IM loss variant should run without error."""
    model = _make_model()
    freeze_structure(model)

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    avg_mmd, avg_ent, avg_loss = adapt_epoch(
        model, src_loader, tgt_loader, optimizer, torch.device("cpu"),
        use_im_loss=True,
    )
    assert isinstance(avg_loss, float)


def test_adapt_epoch_bottleneck_mmd():
    """Bottleneck MMD variant should run without error."""
    model = _make_model()
    freeze_structure(model)

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    avg_mmd, avg_ent, avg_loss = adapt_epoch(
        model, src_loader, tgt_loader, optimizer, torch.device("cpu"),
        use_bottleneck_mmd=True,
    )
    assert isinstance(avg_loss, float)


def test_adapt_epoch_im_plus_bottleneck():
    """IM loss + bottleneck MMD combined should run without error."""
    model = _make_model()
    freeze_structure(model)

    src_loader = _make_fake_loader()
    tgt_loader = _make_fake_loader()
    optimizer = SGD(get_adaptable_params(model), lr=0.01)

    avg_mmd, avg_ent, avg_loss = adapt_epoch(
        model, src_loader, tgt_loader, optimizer, torch.device("cpu"),
        use_im_loss=True, use_bottleneck_mmd=True,
    )
    assert isinstance(avg_loss, float)

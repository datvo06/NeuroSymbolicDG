"""Test the training loop."""

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.trainer import evaluate, train, train_epoch


def _make_fake_loader(n_samples=16, n_classes=3, image_size=224, batch_size=8):
    """Create a small fake dataset for testing."""
    x = torch.randn(n_samples, 3, image_size, image_size)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _make_small_model(n_classes=3):
    return NeuroSymbolicPipeline(
        n_primitives=2,
        n_classes=n_classes,
        backbone_variant="resnet18",
        pretrained_backbone=False,
        max_depth=1,
        use_inside=False,
    )


def test_train_epoch():
    model = _make_small_model()
    loader = _make_fake_loader()
    optimizer = SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    loss, acc = train_epoch(model, loader, optimizer, device)

    assert isinstance(loss, float)
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_evaluate():
    model = _make_small_model()
    loader = _make_fake_loader()
    device = torch.device("cpu")

    loss, acc = evaluate(model, loader, device)

    assert isinstance(loss, float)
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_train_loop():
    model = _make_small_model()
    train_loader = _make_fake_loader()
    val_loader = _make_fake_loader()
    optimizer = SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        n_epochs=2,
        log_interval=1,
    )

    assert len(metrics.history) == 2
    assert metrics.epoch == 2
    assert all(h["train_loss"] > 0 for h in metrics.history)
    assert all(0.0 <= h["train_acc"] <= 1.0 for h in metrics.history)


def test_train_loss_decreases():
    """After a few epochs, training loss should decrease (on a tiny dataset)."""
    torch.manual_seed(42)
    model = _make_small_model()
    loader = _make_fake_loader(n_samples=8, batch_size=8)
    optimizer = SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    metrics = train(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        device=device,
        n_epochs=10,
        log_interval=100,  # suppress output
    )

    first_loss = metrics.history[0]["train_loss"]
    last_loss = metrics.history[-1]["train_loss"]
    assert last_loss < first_loss, f"Loss did not decrease: {first_loss} -> {last_loss}"


def test_train_with_save(tmp_path):
    model = _make_small_model()
    loader = _make_fake_loader()
    optimizer = SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    save_path = str(tmp_path / "best.pt")

    train(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        device=device,
        n_epochs=2,
        log_interval=100,
        save_path=save_path,
    )

    checkpoint = torch.load(save_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "val_acc" in checkpoint
    assert "epoch" in checkpoint


def test_train_with_scheduler():
    model = _make_small_model()
    loader = _make_fake_loader()
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    device = torch.device("cpu")

    metrics = train(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        device=device,
        n_epochs=3,
        scheduler=scheduler,
        log_interval=100,
    )

    assert len(metrics.history) == 3
    # LR should have decayed
    assert optimizer.param_groups[0]["lr"] < 0.01


def test_gradient_updates_all_components():
    """Verify that one training step updates backbone, bottleneck, and grammar."""
    model = _make_small_model()
    loader = _make_fake_loader(n_samples=4, batch_size=4)
    optimizer = SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Record params before
    grammar_before = model.grammar.log_weights.clone()
    bn_before = model.bottleneck.heatmap_conv.weight.data.clone()

    train_epoch(model, loader, optimizer, device)

    # Check params changed
    assert not torch.equal(model.grammar.log_weights, grammar_before)
    assert not torch.equal(model.bottleneck.heatmap_conv.weight.data, bn_before)

    # Relation params get gradients (may be small, just check they exist)
    assert model.relation_params.lambda_above.grad is not None

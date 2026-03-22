"""Test Particle MCMC for grammar structure search."""

import random

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.pmcmc import (
    MoveType,
    PMCMCStats,
    Particle,
    apply_particle_weights,
    get_best_particle,
    hybrid_train_epoch,
    init_particles,
    mcmc_sweep,
    propose_move,
    score_particle,
    sparse_prior,
)


def _make_model(n_classes=3, n_primitives=2):
    return NeuroSymbolicPipeline(
        n_primitives=n_primitives,
        n_classes=n_classes,
        backbone_variant="resnet18",
        pretrained_backbone=False,
        max_depth=1,
        use_inside=False,
    )


def _make_fake_loader(n_samples=8, n_classes=3, batch_size=4):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _make_fake_batch(n_samples=8, n_classes=3):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, n_classes, (n_samples,))
    return x, y


def test_init_particles():
    particles = init_particles(
        n_particles=5, n_classes=3, n_productions=20, n_active_init=8
    )
    assert len(particles) == 5
    for p in particles:
        assert p.log_weights.shape == (3, 20)
        # Each class should have ~8 active productions
        for c in range(3):
            n_active = (p.log_weights[c] > -10).sum().item()
            assert n_active == 8


def test_init_particles_capped():
    """n_active_init > n_productions should be capped."""
    particles = init_particles(
        n_particles=2, n_classes=2, n_productions=5, n_active_init=100
    )
    for p in particles:
        for c in range(2):
            n_active = (p.log_weights[c] > -10).sum().item()
            assert n_active == 5


def test_sparse_prior():
    lw = torch.zeros(2, 10)  # all active
    prior_all = sparse_prior(lw, sparsity_lambda=0.1)

    # Make some inactive
    lw_sparse = lw.clone()
    lw_sparse[:, 5:] = -20.0  # 5 active per class
    prior_sparse = sparse_prior(lw_sparse, sparsity_lambda=0.1)

    # Sparse should have higher (less negative) prior
    assert prior_sparse > prior_all


def test_sparse_prior_zero_lambda():
    lw = torch.zeros(2, 10)
    prior = sparse_prior(lw, sparsity_lambda=0.0)
    assert prior == 0.0


def test_propose_birth():
    random.seed(42)
    lw = torch.full((2, 10), -20.0)
    lw[0, :3] = 0.0  # 3 active, 7 inactive
    particle = Particle(log_weights=lw)

    # Force birth by trying many times
    got_birth = False
    for _ in range(50):
        proposed, move, log_ratio = propose_move(particle, class_idx=0)
        if move == MoveType.BIRTH:
            got_birth = True
            # Should have one more active production
            n_active_before = (particle.log_weights[0] > -10).sum().item()
            n_active_after = (proposed.log_weights[0] > -10).sum().item()
            assert n_active_after == n_active_before + 1
            break
    assert got_birth


def test_propose_death():
    random.seed(42)
    lw = torch.zeros(2, 10)  # all active
    particle = Particle(log_weights=lw)

    got_death = False
    for _ in range(50):
        proposed, move, log_ratio = propose_move(particle, class_idx=0)
        if move == MoveType.DEATH:
            got_death = True
            n_active_before = (particle.log_weights[0] > -10).sum().item()
            n_active_after = (proposed.log_weights[0] > -10).sum().item()
            assert n_active_after == n_active_before - 1
            break
    assert got_death


def test_propose_swap():
    random.seed(42)
    lw = torch.full((2, 10), -20.0)
    lw[0, :5] = 0.0  # 5 active, 5 inactive
    particle = Particle(log_weights=lw)

    got_swap = False
    for _ in range(50):
        proposed, move, log_ratio = propose_move(particle, class_idx=0)
        if move == MoveType.SWAP:
            got_swap = True
            # Same number of active productions
            n_active_before = (particle.log_weights[0] > -10).sum().item()
            n_active_after = (proposed.log_weights[0] > -10).sum().item()
            assert n_active_after == n_active_before
            # But different indices
            active_before = set(
                (particle.log_weights[0] > -10).nonzero().squeeze(-1).tolist()
            )
            active_after = set(
                (proposed.log_weights[0] > -10).nonzero().squeeze(-1).tolist()
            )
            assert active_before != active_after
            break
    assert got_swap


def test_propose_perturb():
    random.seed(42)
    lw = torch.zeros(2, 10)
    particle = Particle(log_weights=lw)

    got_perturb = False
    for _ in range(50):
        proposed, move, log_ratio = propose_move(particle, class_idx=0)
        if move == MoveType.PERTURB:
            got_perturb = True
            # Weights should differ slightly
            assert not torch.equal(
                proposed.log_weights[0], particle.log_weights[0]
            )
            # Only one weight should change
            diff = (proposed.log_weights[0] - particle.log_weights[0]).abs()
            assert (diff > 1e-6).sum().item() == 1
            break
    assert got_perturb


def test_propose_does_not_mutate_original():
    lw = torch.zeros(2, 10)
    particle = Particle(log_weights=lw)
    original_weights = particle.log_weights.clone()

    for _ in range(20):
        propose_move(particle, class_idx=0)

    assert torch.equal(particle.log_weights, original_weights)


def test_propose_keeps_at_least_one_active():
    """Death should not deactivate the last production."""
    lw = torch.full((2, 10), -20.0)
    lw[0, 0] = 0.0  # only 1 active
    particle = Particle(log_weights=lw)

    for _ in range(50):
        proposed, move, _ = propose_move(particle, class_idx=0)
        if move == MoveType.DEATH:
            # Should never kill the last one (death requires n_active > 1)
            assert False, "Death should not be proposed with only 1 active"


def test_score_particle():
    torch.manual_seed(42)
    model = _make_model()
    batch = _make_fake_batch(n_samples=4, n_classes=3)

    particle = Particle(log_weights=model.grammar.log_weights.data.clone())
    ll = score_particle(particle, model, batch, torch.device("cpu"))

    assert isinstance(ll, float)
    assert ll < 0  # log-likelihood should be negative
    assert not math.isnan(ll)


def test_score_particle_restores_weights():
    """score_particle should not permanently change model weights."""
    model = _make_model()
    orig_weights = model.grammar.log_weights.data.clone()

    particle = Particle(log_weights=torch.randn_like(orig_weights))
    batch = _make_fake_batch(n_samples=4, n_classes=3)
    score_particle(particle, model, batch, torch.device("cpu"))

    assert torch.equal(model.grammar.log_weights.data, orig_weights)


def test_mcmc_sweep():
    torch.manual_seed(42)
    random.seed(42)
    model = _make_model()
    batch = _make_fake_batch(n_samples=4, n_classes=3)

    particles = init_particles(
        n_particles=3,
        n_classes=3,
        n_productions=model.grammar.n_productions,
        n_active_init=5,
    )

    stats = PMCMCStats()
    result = mcmc_sweep(
        particles, model, batch, torch.device("cpu"),
        n_proposals_per_particle=3, stats=stats,
    )

    assert len(result) == 3
    assert stats.n_proposed == 9  # 3 particles * 3 proposals
    assert stats.n_proposed >= stats.n_accepted >= 0
    # All particles should now be scored
    for p in result:
        assert p.log_likelihood != float("-inf")


def test_get_best_particle():
    p1 = Particle(log_weights=torch.zeros(2, 5), log_likelihood=-10.0, log_prior=-1.0)
    p2 = Particle(log_weights=torch.zeros(2, 5), log_likelihood=-5.0, log_prior=-1.0)
    p3 = Particle(log_weights=torch.zeros(2, 5), log_likelihood=-8.0, log_prior=-0.5)

    best = get_best_particle([p1, p2, p3])
    assert best is p2  # -5 + -1 = -6 > -8.5 > -11


def test_apply_particle_weights():
    model = _make_model()
    particle = Particle(log_weights=torch.ones(3, model.grammar.n_productions) * 42.0)

    apply_particle_weights(model, particle)
    assert torch.allclose(model.grammar.log_weights.data, particle.log_weights)


def test_hybrid_train_epoch():
    torch.manual_seed(42)
    random.seed(42)
    model = _make_model()
    loader = _make_fake_loader(n_samples=8, n_classes=3, batch_size=4)

    particles = init_particles(
        n_particles=2,
        n_classes=3,
        n_productions=model.grammar.n_productions,
        n_active_init=5,
    )

    # Only optimize continuous params (backbone + bottleneck + relation_params)
    continuous_params = [
        p for name, p in model.named_parameters()
        if "grammar" not in name
    ]
    optimizer = SGD(continuous_params, lr=0.01)

    stats = PMCMCStats()
    avg_loss, accuracy = hybrid_train_epoch(
        model, particles, loader, optimizer,
        torch.device("cpu"),
        n_mcmc_proposals=2,
        stats=stats,
    )

    assert isinstance(avg_loss, float)
    assert isinstance(accuracy, float)
    assert avg_loss > 0
    assert 0.0 <= accuracy <= 1.0
    assert stats.n_proposed > 0


def test_hybrid_train_epoch_updates_continuous_params():
    """Gradient steps should update backbone/bottleneck but not grammar."""
    torch.manual_seed(42)
    random.seed(42)
    model = _make_model()
    loader = _make_fake_loader(n_samples=8, n_classes=3, batch_size=4)

    particles = init_particles(
        n_particles=2,
        n_classes=3,
        n_productions=model.grammar.n_productions,
        n_active_init=5,
    )

    bn_before = model.bottleneck.heatmap_conv.weight.data.clone()

    continuous_params = [
        p for name, p in model.named_parameters()
        if "grammar" not in name
    ]
    optimizer = SGD(continuous_params, lr=0.1)

    hybrid_train_epoch(
        model, particles, loader, optimizer,
        torch.device("cpu"),
        n_mcmc_proposals=2,
    )

    # Bottleneck should have changed
    assert not torch.equal(model.bottleneck.heatmap_conv.weight.data, bn_before)


def test_pmcmc_stats():
    stats = PMCMCStats()
    stats.n_proposed = 10
    stats.n_accepted = 3
    assert stats.acceptance_rate == 0.3

    stats.record_sweep()
    assert len(stats.acceptance_rates) == 1
    assert stats.acceptance_rates[0] == 0.3
    # Counters reset
    assert stats.n_proposed == 0
    assert stats.n_accepted == 0


def test_particle_clone():
    p = Particle(
        log_weights=torch.randn(2, 5),
        log_likelihood=-5.0,
        log_prior=-1.0,
    )
    p2 = p.clone()

    assert torch.equal(p.log_weights, p2.log_weights)
    assert p.log_likelihood == p2.log_likelihood

    # Mutation should not affect original
    p2.log_weights[0, 0] = 999.0
    assert p.log_weights[0, 0] != 999.0


import math

"""Particle MCMC for grammar structure search (Section 3.6).

Explores the combinatorial space of grammar weight configurations using
Metropolis-Hastings proposals scored by the inside algorithm.

The hybrid training alternates:
  - Outer level: MCMC sweeps over grammar structure (which productions active)
  - Inner level: gradient descent on continuous params (backbone, bottleneck, relations)
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn as nn
from effectful.ops.semantics import handler
from torch import Tensor
from torch.utils.data import DataLoader

from neurosymbolic_da.dsl.handlers.inside import get_class_score, make_inside_handler
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline


class MoveType(Enum):
    BIRTH = auto()
    DEATH = auto()
    SWAP = auto()
    PERTURB = auto()


@dataclass
class Particle:
    """A grammar weight configuration.

    Each particle holds its own log_weights tensor and tracks
    which productions are "active" (not masked to -inf).
    """

    log_weights: Tensor  # [n_classes, n_productions]
    log_likelihood: float = float("-inf")
    log_prior: float = float("-inf")

    def clone(self) -> "Particle":
        return Particle(
            log_weights=self.log_weights.clone(),
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
        )


@dataclass
class PMCMCStats:
    """Statistics from PMCMC sweeps."""

    n_accepted: int = 0
    n_proposed: int = 0
    acceptance_rates: list[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def record_sweep(self):
        self.acceptance_rates.append(self.acceptance_rate)
        self.n_accepted = 0
        self.n_proposed = 0


# Threshold below which a production is considered inactive
_INACTIVE_THRESHOLD = -10.0


def _is_active(log_weight: float) -> bool:
    return log_weight > _INACTIVE_THRESHOLD


def _get_active_mask(log_weights: Tensor, class_idx: int) -> list[bool]:
    """Return which productions are active for a given class."""
    return [_is_active(w.item()) for w in log_weights[class_idx]]


def _count_active(log_weights: Tensor, class_idx: int) -> int:
    return sum(_get_active_mask(log_weights, class_idx))


def init_particles(
    n_particles: int,
    n_classes: int,
    n_productions: int,
    n_active_init: int = 15,
    init_scale: float = 0.5,
) -> list[Particle]:
    """Initialize particles with random sparse subsets of active productions.

    Args:
        n_particles: number of particles K
        n_classes: number of classes
        n_productions: total productions in universal grammar
        n_active_init: initial number of active productions per class
        init_scale: scale for initial weight sampling
    """
    particles = []
    n_active = min(n_active_init, n_productions)

    for _ in range(n_particles):
        # Start with all inactive
        log_weights = torch.full((n_classes, n_productions), -20.0)

        for c in range(n_classes):
            # Randomly activate a subset
            active_idx = random.sample(range(n_productions), n_active)
            for idx in active_idx:
                log_weights[c, idx] = torch.randn(1).item() * init_scale

        particles.append(Particle(log_weights=log_weights))

    return particles


def sparse_prior(log_weights: Tensor, sparsity_lambda: float = 0.1) -> float:
    """Log prior encouraging sparse grammars.

    Penalizes the number of active productions per class.
    """
    log_p = 0.0
    for c in range(log_weights.shape[0]):
        n_active = _count_active(log_weights, c)
        log_p -= sparsity_lambda * n_active
    return log_p


def propose_move(
    particle: Particle,
    class_idx: int,
    perturb_std: float = 0.3,
    birth_scale: float = 0.5,
) -> tuple[Particle, MoveType, float]:
    """Propose a structural modification to a particle's grammar.

    Returns:
        (proposed_particle, move_type, log_proposal_ratio)
        where log_proposal_ratio = log q(theta | theta') - log q(theta' | theta)
    """
    proposed = particle.clone()
    lw = proposed.log_weights
    n_productions = lw.shape[1]

    active_mask = _get_active_mask(lw, class_idx)
    active_idx = [i for i, a in enumerate(active_mask) if a]
    inactive_idx = [i for i, a in enumerate(active_mask) if not a]

    n_active = len(active_idx)
    n_inactive = len(inactive_idx)

    # Choose move type based on what's possible
    possible_moves = [MoveType.PERTURB]
    if n_inactive > 0:
        possible_moves.append(MoveType.BIRTH)
    if n_active > 1:  # keep at least 1 active
        possible_moves.append(MoveType.DEATH)
    if n_active > 0 and n_inactive > 0:
        possible_moves.append(MoveType.SWAP)

    move = random.choice(possible_moves)
    log_proposal_ratio = 0.0

    if move == MoveType.BIRTH:
        # Activate a random inactive production
        idx = random.choice(inactive_idx)
        new_weight = torch.randn(1).item() * birth_scale
        lw[class_idx, idx] = new_weight

        # Proposal ratio: birth proposes from n_inactive choices,
        # reverse (death) would choose from (n_active + 1) choices
        # Also account for weight proposal density
        n_active_after = n_active + 1
        n_inactive_after = n_inactive - 1
        # P(choose birth) * P(choose this prod) vs P(choose death) * P(choose this prod)
        log_proposal_ratio = (
            math.log(n_active_after) - math.log(n_inactive)
        )

    elif move == MoveType.DEATH:
        # Deactivate a random active production
        idx = random.choice(active_idx)
        old_weight = lw[class_idx, idx].item()
        lw[class_idx, idx] = -20.0

        n_active_after = n_active - 1
        n_inactive_after = n_inactive + 1
        log_proposal_ratio = (
            math.log(n_inactive_after) - math.log(n_active)
        )

    elif move == MoveType.SWAP:
        # Replace one active production with an inactive one
        a_idx = random.choice(active_idx)
        i_idx = random.choice(inactive_idx)
        lw[class_idx, i_idx] = lw[class_idx, a_idx].clone()
        lw[class_idx, a_idx] = -20.0
        # Swap is symmetric: log_proposal_ratio = 0

    elif move == MoveType.PERTURB:
        if n_active > 0:
            idx = random.choice(active_idx)
            lw[class_idx, idx] += torch.randn(1).item() * perturb_std
        # Symmetric Gaussian perturbation: log_proposal_ratio = 0

    return proposed, move, log_proposal_ratio


def score_particle(
    particle: Particle,
    model: NeuroSymbolicPipeline,
    data_batch: tuple[Tensor, Tensor],
    device: torch.device,
    class_indices: list[int] | None = None,
) -> float:
    """Score a particle using the inside algorithm on a data batch.

    Computes the log-likelihood: sum over (x, y) of log W_y(x) - log sum_y' W_y'(x).

    Args:
        particle: the particle to score
        model: the pipeline (used for backbone, bottleneck, relation_params)
        data_batch: (images, labels) tensors
        device: torch device
        class_indices: if set, only score these classes (optimization)

    Returns:
        log-likelihood (float)
    """
    images, labels = data_batch
    images = images.to(device)
    labels = labels.to(device)

    model.eval()

    # Temporarily install particle's weights
    orig_weights = model.grammar.log_weights.data.clone()
    model.grammar.log_weights.data.copy_(particle.log_weights.to(device))

    try:
        with torch.no_grad():
            log_probs = model(images)  # [B, n_classes]
            # Log-likelihood = sum of log p(y|x) for correct classes
            log_lik = 0.0
            for i in range(images.shape[0]):
                log_lik += log_probs[i, labels[i]].item()
    finally:
        # Restore original weights
        model.grammar.log_weights.data.copy_(orig_weights)

    return log_lik


def mcmc_sweep(
    particles: list[Particle],
    model: NeuroSymbolicPipeline,
    data_batch: tuple[Tensor, Tensor],
    device: torch.device,
    n_proposals_per_particle: int = 10,
    sparsity_lambda: float = 0.1,
    perturb_std: float = 0.3,
    birth_scale: float = 0.5,
    stats: PMCMCStats | None = None,
) -> list[Particle]:
    """Run one MCMC sweep over all particles.

    For each particle, propose n_proposals_per_particle modifications
    and accept/reject via Metropolis-Hastings.

    Args:
        particles: list of current particles
        model: the pipeline
        data_batch: (images, labels) for scoring
        device: torch device
        n_proposals_per_particle: number of proposals M per particle per sweep
        sparsity_lambda: sparsity prior strength
        perturb_std: std for weight perturbation moves
        birth_scale: scale for birth move weight initialization
        stats: optional stats tracker
    """
    n_classes = particles[0].log_weights.shape[0]

    for p_idx, particle in enumerate(particles):
        # Score current particle if not yet scored
        if particle.log_likelihood == float("-inf"):
            particle.log_likelihood = score_particle(
                particle, model, data_batch, device
            )
            particle.log_prior = sparse_prior(
                particle.log_weights, sparsity_lambda
            )

        for _ in range(n_proposals_per_particle):
            # Pick a random class to modify
            class_idx = random.randint(0, n_classes - 1)

            # Propose
            proposed, move, log_proposal_ratio = propose_move(
                particle, class_idx, perturb_std, birth_scale
            )

            # Score proposal
            proposed.log_likelihood = score_particle(
                proposed, model, data_batch, device
            )
            proposed.log_prior = sparse_prior(
                proposed.log_weights, sparsity_lambda
            )

            # Metropolis-Hastings acceptance
            log_alpha = (
                (proposed.log_likelihood + proposed.log_prior)
                - (particle.log_likelihood + particle.log_prior)
                + log_proposal_ratio
            )

            if stats is not None:
                stats.n_proposed += 1

            if math.log(random.random() + 1e-30) < log_alpha:
                # Accept
                particles[p_idx] = proposed
                particle = proposed
                if stats is not None:
                    stats.n_accepted += 1

    return particles


def get_best_particle(particles: list[Particle]) -> Particle:
    """Return the particle with highest posterior (likelihood + prior)."""
    return max(particles, key=lambda p: p.log_likelihood + p.log_prior)


def apply_particle_weights(
    model: NeuroSymbolicPipeline, particle: Particle
) -> None:
    """Copy a particle's grammar weights into the model."""
    model.grammar.log_weights.data.copy_(particle.log_weights.to(
        model.grammar.log_weights.device
    ))


def hybrid_train_epoch(
    model: NeuroSymbolicPipeline,
    particles: list[Particle],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_mcmc_proposals: int = 10,
    sparsity_lambda: float = 0.1,
    mcmc_batch_size: int = 32,
    stats: PMCMCStats | None = None,
) -> tuple[float, float]:
    """One epoch of hybrid training: MCMC sweep then gradient steps.

    1. Sample a batch for MCMC scoring
    2. Run MCMC sweep to update grammar structure
    3. Apply best particle's weights
    4. Run gradient descent epoch on continuous params

    Returns:
        (avg_loss, accuracy)
    """
    # Step 1: Get a batch for MCMC scoring
    mcmc_iter = iter(train_loader)
    mcmc_batch = next(mcmc_iter)

    # Step 2: MCMC sweep
    particles = mcmc_sweep(
        particles,
        model,
        mcmc_batch,
        device,
        n_proposals_per_particle=n_mcmc_proposals,
        sparsity_lambda=sparsity_lambda,
        stats=stats,
    )

    # Step 3: Apply best particle's weights to model
    best = get_best_particle(particles)
    apply_particle_weights(model, best)

    # Step 4: Gradient descent on continuous params (backbone + bottleneck + relations)
    # Grammar weights are set from particle, freeze them for gradient step
    model.grammar.log_weights.requires_grad_(False)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        log_probs = model(batch_x)
        loss = nn.functional.nll_loss(log_probs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = log_probs.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    # Re-enable grammar weights grad (particles manage them)
    model.grammar.log_weights.requires_grad_(True)

    # Update particles with current model state (re-score after gradient step)
    for p in particles:
        p.log_likelihood = float("-inf")  # mark for re-scoring next sweep

    return total_loss / total, correct / total

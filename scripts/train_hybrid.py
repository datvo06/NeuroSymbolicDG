#!/usr/bin/env python3
"""Hybrid training script: Particle MCMC + gradient descent (Section 3.6).

Alternates between MCMC sweeps over grammar structure and gradient
optimization of continuous parameters (backbone, bottleneck, relations).

Usage:
    uv run python scripts/train_hybrid.py \
        --dataset digits --source mnist --target usps \
        --n-primitives 8 --n-particles 10 --mcmc-proposals 20
"""

import argparse
import random
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.pmcmc import (
    PMCMCStats,
    apply_particle_weights,
    get_best_particle,
    hybrid_train_epoch,
    init_particles,
)
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid PMCMC + gradient training (Phase 1)"
    )
    # Dataset
    parser.add_argument(
        "--dataset", required=True, choices=["digits", "office31", "officehome", "scb", "cubdg"]
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    # Model
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true",
                        help="Use sparsemax instead of softmax for grammar weights")

    # PMCMC
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--n-active-init", type=int, default=15)
    parser.add_argument("--mcmc-proposals", type=int, default=20)
    parser.add_argument("--sparsity-lambda", type=float, default=0.1)
    parser.add_argument("--perturb-std", type=float, default=0.3)
    parser.add_argument("--birth-scale", type=float, default=0.5)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strong-aug", action="store_true",
                        help="Use strong data augmentation")
    parser.add_argument("--residual-relations", action="store_true",
                        help="Use learned residual corrections on top of hand-coded relations")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    image_size = 32 if args.backbone == "lenet" else 224
    n_classes = get_n_classes(args.dataset)
    src_train, src_test, tgt_train, tgt_test = get_loaders(
        args.dataset, args.source, args.target,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        strong_aug=args.strong_aug,
    )

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target})")
    print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")

    # Build model — vectorized forward computes exact marginal scores
    # (same as inside algorithm but 26x faster via tensor ops)
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=args.pretrained,
        max_depth=args.max_depth,
        use_inside=False,
        use_sparsemax=args.use_sparsemax,
        residual_relations=getattr(args, 'residual_relations', False),
    )
    model.to(device)

    # Initialize particles
    particles = init_particles(
        n_particles=args.n_particles,
        n_classes=n_classes,
        n_productions=model.grammar.n_productions,
        n_active_init=args.n_active_init,
        init_scale=args.birth_scale,
    )
    print(
        f"Particles: {args.n_particles}, "
        f"Productions: {model.grammar.n_productions}, "
        f"Initial active: {args.n_active_init}/class"
    )

    # Optimizer for continuous params only (grammar weights managed by PMCMC)
    continuous_params = [
        p for name, p in model.named_parameters() if "grammar" not in name
    ]
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.bottleneck.parameters()) + list(model.relation_params.parameters())
    )
    optimizer = Adam(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Training loop
    save_path = (
        args.save_path
        or f"checkpoint_hybrid_{args.dataset}_{args.source}_{args.target}.pt"
    )
    best_val_acc = 0.0
    stats = PMCMCStats()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        avg_loss, train_acc = hybrid_train_epoch(
            model=model,
            particles=particles,
            train_loader=src_train,
            optimizer=optimizer,
            device=device,
            n_mcmc_proposals=args.mcmc_proposals,
            sparsity_lambda=args.sparsity_lambda,
            stats=stats,
        )

        val_loss, val_acc = evaluate(model, src_test, device)
        scheduler.step()

        epoch_time = time.time() - t0

        if epoch % args.log_interval == 0:
            best = get_best_particle(particles)
            n_active = sum(
                (best.log_weights[c] > -10).sum().item() for c in range(n_classes)
            )
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={avg_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"accept={stats.acceptance_rate:.2f} "
                f"active={n_active:.0f} | "
                f"time={epoch_time:.1f}s"
            )

        stats.record_sweep()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_particle = get_best_particle(particles)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "particle_log_weights": best_particle.log_weights,
                    "val_acc": val_acc,
                },
                save_path,
            )

    # Final target evaluation
    _, tgt_acc = evaluate(model, tgt_test, device)
    print(f"\n--- Final Results ---")
    print(f"Source test acc: {val_acc:.4f}")
    print(f"Target test acc: {tgt_acc:.4f} (no adaptation)")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {save_path}")

    # Report grammar sparsity
    best = get_best_particle(particles)
    for c in range(min(n_classes, 5)):
        n_active = (best.log_weights[c] > -10).sum().item()
        print(f"  Class {c}: {n_active}/{model.grammar.n_productions} active productions")


if __name__ == "__main__":
    main()

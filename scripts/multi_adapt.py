#!/usr/bin/env python3
"""Multi-domain adaptation (Experiment 5, Section 6.7).

Train on one source domain, then adapt to k target domains sequentially.
Reports total parameter count vs number of target domains to test
whether our approach shows sublinear parameter growth (shared grammar,
per-domain detectors).

Usage:
    uv run python scripts/multi_adapt.py \
        --checkpoint checkpoint_office31_amazon.pt \
        --dataset office31 --source amazon \
        --targets webcam dslr \
        --epochs 20 --lr 1e-4
"""

import argparse
import copy

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import adapt, freeze_structure, get_adaptable_params
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> dict[str, int]:
    """Count parameters by component."""
    counts = {}
    for name, param in model.named_parameters():
        component = name.split(".")[0]
        counts[component] = counts.get(component, 0) + param.numel()
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Multi-domain adaptation (Experiment 5)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["digits", "office31", "officehome"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True,
                        help="List of target domains to adapt to")
    parser.add_argument("--data-root", default="./data")

    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-inside", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-entropy", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--use-im-loss", action="store_true",
                        help="Use Information Maximization loss")
    parser.add_argument("--use-bottleneck-mmd", action="store_true",
                        help="Align bottleneck features instead of raw heatmaps")

    args = parser.parse_args()

    device = get_device()
    n_classes = get_n_classes(args.dataset)

    # Load source model
    source_model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
        max_depth=args.max_depth,
        use_inside=args.use_inside,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    source_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded source checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Parameter counts
    base_counts = count_params(source_model)
    grammar_params = base_counts.get("grammar", 0) + base_counts.get("relation_params", 0)
    detector_params = base_counts.get("backbone", 0) + base_counts.get("bottleneck", 0)

    print(f"\nParameter breakdown:")
    print(f"  Grammar (shared):     {grammar_params:,}")
    print(f"  Detectors (per-domain): {detector_params:,}")
    print(f"  Total:                {sum(base_counts.values()):,}")

    # Adapt to each target domain
    results = []
    print(f"\n{'='*70}")
    print(f"Adapting {args.source} -> {args.targets}")
    print(f"{'='*70}")

    for k, target in enumerate(args.targets, 1):
        print(f"\n--- Target {k}/{len(args.targets)}: {target} ---")

        # Fresh copy of source model for each target
        model = copy.deepcopy(source_model)
        model.to(device)

        # Load target data
        src_train, _, tgt_train, tgt_test = get_loaders(
            args.dataset, args.source, target,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Evaluate before adaptation
        _, pre_acc = evaluate(model, tgt_test, device)

        # Freeze and adapt
        freeze_structure(model)
        adaptable_params = get_adaptable_params(model)
        optimizer = Adam(adaptable_params, lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        save_path = f"{args.save_dir}/adapted_{args.dataset}_{args.source}_{target}.pt"
        metrics = adapt(
            model=model,
            source_loader=src_train,
            target_loader=tgt_train,
            target_test_loader=tgt_test,
            optimizer=optimizer,
            device=device,
            n_epochs=args.epochs,
            lambda_entropy=args.lambda_entropy,
            scheduler=scheduler,
            log_interval=5,
            save_path=save_path,
            use_im_loss=args.use_im_loss,
            use_bottleneck_mmd=args.use_bottleneck_mmd,
        )

        results.append({
            "target": target,
            "pre_adapt_acc": pre_acc,
            "post_adapt_acc": metrics.target_acc,
            "improvement": metrics.target_acc - pre_acc,
        })

        # Cumulative parameter count:
        # grammar is shared, detectors are per-domain
        total_params = grammar_params + detector_params * k
        print(f"  Pre-adapt: {pre_acc:.4f}, Post-adapt: {metrics.target_acc:.4f}")
        print(f"  Cumulative params ({k} targets): {total_params:,}")

    # Summary
    print(f"\n{'='*70}")
    print(f"MULTI-DOMAIN SUMMARY")
    print(f"{'='*70}")
    print(f"{'k':>3} {'Target':>12} {'Pre-adapt':>10} {'Post-adapt':>11} {'Improve':>8} {'Total params':>14}")

    for k, r in enumerate(results, 1):
        total_params = grammar_params + detector_params * k
        print(
            f"{k:3d} {r['target']:>12} "
            f"{r['pre_adapt_acc']:10.4f} {r['post_adapt_acc']:11.4f} "
            f"{r['improvement']:+8.4f} {total_params:14,}"
        )

    print(f"\nParameter scaling:")
    print(f"  Shared (grammar):    {grammar_params:,} (constant)")
    print(f"  Per-domain (detectors): {detector_params:,}")
    print(f"  k=1: {grammar_params + detector_params:,}")
    print(f"  k={len(args.targets)}: {grammar_params + detector_params * len(args.targets):,}")
    print(f"  Growth factor: {(grammar_params + detector_params * len(args.targets)) / (grammar_params + detector_params):.2f}x")


if __name__ == "__main__":
    main()

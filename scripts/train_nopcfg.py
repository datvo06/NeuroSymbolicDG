#!/usr/bin/env python3
"""NoPCFG ablation training script (Experiment 4a, Section 6.6).

Trains backbone + concept bottleneck + linear classifier (no PCFG grammar).
Isolates the contribution of structural programs.

Usage:
    uv run python scripts/train_nopcfg.py \
        --dataset office31 --source amazon --target webcam \
        --data-root ./data/office31 --backbone resnet50 --pretrained

    uv run python scripts/train_nopcfg.py \
        --dataset scb --source source --target A \
        --n-primitives 4 --backbone resnet18
"""

import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline_nopcfg import NoPCFGPipeline
from neurosymbolic_da.training.trainer import train


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="NoPCFG ablation training (Phase 1)"
    )
    # Dataset
    parser.add_argument(
        "--dataset", required=True,
        choices=["digits", "office31", "officehome", "scb", "cubdg"],
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    # Model
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()

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
    )

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target})")
    print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
    print("Pipeline: NoPCFG (backbone + bottleneck + linear)")

    # Build model
    model = NoPCFGPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=args.pretrained,
    )

    # Separate LRs: lower for pretrained backbone, higher for heads
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.bottleneck.parameters()) + list(model.classifier.parameters())
    )
    backbone_lr_mult = 0.1 if (args.pretrained and args.backbone != "lenet") else 1.0
    optimizer = Adam(
        [
            {"params": backbone_params, "lr": args.lr * backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Train
    save_path = (
        args.save_path
        or f"checkpoint_nopcfg_{args.dataset}_{args.source}_{args.target}.pt"
    )
    metrics = train(
        model=model,
        train_loader=src_train,
        val_loader=src_test,
        optimizer=optimizer,
        device=device,
        n_epochs=args.epochs,
        scheduler=scheduler,
        log_interval=args.log_interval,
        save_path=save_path,
    )

    # Final evaluation on target
    from neurosymbolic_da.training.trainer import evaluate as eval_fn

    model.to(device)
    tgt_loss, tgt_acc = eval_fn(model, tgt_test, device)
    print(f"\n--- Final Results ---")
    print(f"Source test acc: {metrics.val_acc:.4f}")
    print(f"Target test acc: {tgt_acc:.4f} (no adaptation)")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VirDA + PCFG ablation training script (Experiment 4f, Section 6.6).

Combines VirDA's visual reprogramming (frozen backbone + learnable input
perturbation) with our concept bottleneck + PCFG scoring head.

Usage:
    uv run python scripts/train_virda_pcfg.py \
        --dataset office31 --source amazon --target webcam \
        --data-root ./data/office31 --backbone resnet50 --pretrained
"""

import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline_virda_pcfg import VirDAPCFGPipeline
from neurosymbolic_da.training.trainer import train


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="VirDA + PCFG ablation training (Phase 1)"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["digits", "office31", "officehome", "scb", "cubdg"],
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-inside", action="store_true")
    parser.add_argument("--pad-size", type=int, default=30,
                        help="VirDA reprogramming border width")

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

    n_classes = get_n_classes(args.dataset)
    src_train, src_test, tgt_train, tgt_test = get_loaders(
        args.dataset, args.source, args.target,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target})")
    print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
    print("Pipeline: VirDA + PCFG (frozen backbone + reprogramming + bottleneck + grammar)")

    model = VirDAPCFGPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=args.pretrained,
        max_depth=args.max_depth,
        use_inside=args.use_inside,
        pad_size=args.pad_size,
    )

    # All trainable params (backbone is frozen inside the model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Frozen backbone parameters: {n_frozen:,}")

    save_path = (
        args.save_path
        or f"checkpoint_virda_pcfg_{args.dataset}_{args.source}_{args.target}.pt"
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

    from neurosymbolic_da.training.trainer import evaluate as eval_fn

    model.to(device)
    _, tgt_acc = eval_fn(model, tgt_test, device)
    print(f"\n--- Final Results ---")
    print(f"Source test acc: {metrics.val_acc:.4f}")
    print(f"Target test acc: {tgt_acc:.4f} (no adaptation)")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

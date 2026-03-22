#!/usr/bin/env python3
"""Source training script (Phase 1, Section 4.3).

Usage:
    # Digits (MNIST → USPS)
    uv run python scripts/train_source.py --dataset digits --source mnist --target usps

    # Office-31 (Amazon → Webcam)
    uv run python scripts/train_source.py --dataset office31 --source amazon --target webcam \
        --data-root ./data/office31

    # With options
    uv run python scripts/train_source.py --dataset digits --source mnist --target usps \
        --n-primitives 8 --backbone resnet18 --lr 1e-3 --epochs 30 --batch-size 64
"""

import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.trainer import train


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Source training (Phase 1)")
    # Dataset
    parser.add_argument("--dataset", required=True, choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True, help="Source domain name")
    parser.add_argument("--target", required=True, help="Target domain name (for eval only)")
    parser.add_argument("--data-root", default="./data", help="Data root directory")

    # Model
    parser.add_argument("--n-primitives", type=int, default=8, help="Number of primitive types (k)")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--max-depth", type=int, default=1, help="Grammar max depth")
    parser.add_argument("--use-inside", action="store_true", help="Use inside algorithm")
    parser.add_argument("--use-sparsemax", action="store_true",
                        help="Use sparsemax instead of softmax for grammar weights (hard sparsity)")
    parser.add_argument("--input-conditional", action="store_true",
                        help="Use input-conditional grammar weights (attention over productions)")
    parser.add_argument("--bottleneck-type", default="conv", choices=["conv", "slot", "moe", "hourglass"],
                        help="Bottleneck type: conv (1x1 heatmaps), slot (Slot Attention), moe (MoE routing), or hourglass (multi-scale FPN)")
    parser.add_argument("--slot-iters", type=int, default=3,
                        help="Number of Slot Attention iterations")
    parser.add_argument("--invariant-coords", action="store_true",
                        help="Enable scale+rotation invariant coordinate transforms")
    parser.add_argument("--residual-relations", action="store_true",
                        help="Use learned residual corrections on top of hand-coded relations")
    parser.add_argument("--learned-relations", action="store_true",
                        help="Use fully learned relation network (MLP) instead of hand-coded relations")
    parser.add_argument("--strong-aug", action="store_true",
                        help="Use strong data augmentation (ColorJitter + RandomErasing)")
    parser.add_argument("--randaugment", action="store_true",
                        help="Use RandAugment (num_ops=2, magnitude=9)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.1 = typical)")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", default=None, help="Path to save best checkpoint")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--grammar-lr-mult", type=float, default=1.0,
                        help="LR multiplier for grammar log_weights (e.g., 10.0)")
    parser.add_argument("--grammar-l1", type=float, default=0.0,
                        help="L1 sparsity penalty on grammar log_weights")
    parser.add_argument("--bottleneck-reg", type=float, default=0.0,
                        help="Bottleneck diversity+concentration regularization weight")
    parser.add_argument("--bottleneck-reg-v2", action="store_true",
                        help="Use v2 bottleneck reg (orthogonality + stronger concentration)")

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
        strong_aug=args.strong_aug,
        randaugment=args.randaugment,
    )

    print(f"Dataset: {args.dataset} ({args.source} → {args.target})")
    print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")

    # Build model
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=args.pretrained,
        max_depth=args.max_depth,
        use_inside=args.use_inside,
        use_sparsemax=args.use_sparsemax,
        input_conditional=args.input_conditional,
        bottleneck_type=args.bottleneck_type,
        slot_iters=args.slot_iters,
        invariant_coords=args.invariant_coords,
        residual_relations=getattr(args, 'residual_relations', False),
        learned_relations=getattr(args, 'learned_relations', False),
    )

    # Separate LRs: lower for pretrained backbone, higher for new heads
    backbone_params = list(model.backbone.parameters())
    other_head_params = (
        list(model.bottleneck.parameters())
        + list(model.relation_params.parameters())
    )
    grammar_params = list(model.grammar.parameters())
    backbone_lr_mult = 0.1 if (args.pretrained and args.backbone != "lenet") else 1.0
    optimizer = Adam([
        {"params": backbone_params, "lr": args.lr * backbone_lr_mult},
        {"params": other_head_params, "lr": args.lr},
        {"params": grammar_params, "lr": args.lr * args.grammar_lr_mult},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Train
    save_path = args.save_path or f"checkpoint_{args.dataset}_{args.source}_{args.target}.pt"
    if args.grammar_l1 > 0:
        print(f"Grammar L1 sparsity: {args.grammar_l1}")
    if args.bottleneck_reg > 0:
        reg_ver = "v2 (orthogonality)" if args.bottleneck_reg_v2 else "v1 (cosine diversity)"
        print(f"Bottleneck regularization: {args.bottleneck_reg} ({reg_ver})")
    if args.grammar_lr_mult != 1.0:
        print(f"Grammar LR multiplier: {args.grammar_lr_mult}x")
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")
    if args.randaugment:
        print(f"RandAugment: num_ops=2, magnitude=9")

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
        grammar_l1=args.grammar_l1,
        bottleneck_reg=args.bottleneck_reg,
        bottleneck_reg_version=2 if args.bottleneck_reg_v2 else 1,
        label_smoothing=args.label_smoothing,
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

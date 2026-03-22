#!/usr/bin/env python3
"""Target adaptation script (Phase 2, Section 4.3).

Loads a source-trained checkpoint, freezes grammar + relation params,
and adapts backbone + bottleneck using MMD + entropy minimization.

Usage:
    # After source training:
    uv run python scripts/adapt_target.py \
        --checkpoint checkpoint_digits_mnist_usps.pt \
        --dataset digits --source mnist --target usps \
        --epochs 20 --lr 1e-4 --lambda-entropy 0.1
"""

import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Target adaptation (Phase 2)")
    # Checkpoint
    parser.add_argument("--checkpoint", required=True, help="Path to source-trained checkpoint")

    # Dataset
    parser.add_argument("--dataset", required=True, choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True, help="Source domain name")
    parser.add_argument("--target", required=True, help="Target domain name")
    parser.add_argument("--data-root", default="./data", help="Data root directory")

    # Model (must match checkpoint)
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-inside", action="store_true")

    # Adaptation
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-entropy", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--unfreeze-grammar", action="store_true",
                        help="Ablation: adapt grammar weights alongside detectors")
    parser.add_argument("--use-im-loss", action="store_true",
                        help="Use Information Maximization loss (prevents collapse)")
    parser.add_argument("--use-bottleneck-mmd", action="store_true",
                        help="Align bottleneck features instead of raw heatmaps")
    parser.add_argument("--lambda-l2sp", type=float, default=0.0,
                        help="L2-SP regularization weight (penalizes deviation from source weights)")
    parser.add_argument("--production-mmd", action="store_true",
                        help="Also align production scores via MMD (cross-domain grammar alignment)")
    parser.add_argument("--lambda-production-mmd", type=float, default=0.1,
                        help="Weight for production score MMD alignment")
    parser.add_argument("--adapt-mode", default="full",
                        choices=["full", "bn-only", "bottleneck-only"],
                        help="What to adapt: full (backbone+bottleneck), bn-only (batch norm only), "
                             "bottleneck-only (freeze backbone, adapt bottleneck)")
    parser.add_argument("--use-sparsemax", action="store_true",
                        help="Use sparsemax for grammar weights (must match source checkpoint)")
    parser.add_argument("--input-conditional", action="store_true",
                        help="Use input-conditional grammar weights (must match source checkpoint)")
    parser.add_argument("--invariant-coords", action="store_true",
                        help="Enable scale+rotation invariant coordinate transforms (must match source)")

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

    print(f"Dataset: {args.dataset} ({args.source} → {args.target})")

    # Build model and load checkpoint
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,  # weights come from checkpoint
        max_depth=args.max_depth,
        use_inside=args.use_inside,
        use_sparsemax=args.use_sparsemax,
        input_conditional=args.input_conditional,
        invariant_coords=args.invariant_coords,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: allows loading old checkpoints that lack new buffer keys
    # (e.g., grammar._pair_a/_pair_b added for vectorized eval)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(source val acc={checkpoint.get('val_acc', 'N/A')})")

    # Evaluate before adaptation
    model.to(device)
    _, pre_adapt_acc = evaluate(model, tgt_test, device)
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")

    # Save source params for L2-SP before any adaptation
    source_params = None
    if args.lambda_l2sp > 0:
        source_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        print(f"L2-SP regularization: lambda={args.lambda_l2sp}")

    # Freeze grammar + relation params (unless --unfreeze-grammar ablation)
    freeze_structure(model, freeze_grammar=not args.unfreeze_grammar)
    if args.unfreeze_grammar:
        print("ABLATION: grammar weights are NOT frozen (--unfreeze-grammar)")

    # Apply adapt mode: restrict what parameters are trainable
    if args.adapt_mode == "bn-only":
        # TENT-style: only adapt batch norm parameters
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False
        # Re-enable only BN params
        for module in model.backbone.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
        print("ADAPT MODE: bn-only (only batch norm parameters)")
    elif args.adapt_mode == "bottleneck-only":
        # Freeze backbone, adapt only bottleneck
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("ADAPT MODE: bottleneck-only (backbone frozen)")

    adaptable_params = get_adaptable_params(model)
    # Also include any BN params that were re-enabled
    if args.adapt_mode == "bn-only":
        adaptable_params = [p for p in model.parameters() if p.requires_grad]
    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable params: {n_adapt:,} | Frozen params: {n_frozen:,}")

    # Optimizer over adaptable params only
    optimizer = Adam(adaptable_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Adapt
    save_path = args.save_path or f"adapted_{args.dataset}_{args.source}_{args.target}.pt"
    if args.use_im_loss:
        print("Using Information Maximization loss (entropy + diversity)")
    if args.use_bottleneck_mmd:
        print("Using bottleneck feature MMD (compact k*3 features)")
    if getattr(args, 'production_mmd', False):
        print(f"Using production score MMD (lambda={args.lambda_production_mmd})")

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
        log_interval=args.log_interval,
        save_path=save_path,
        use_im_loss=args.use_im_loss,
        use_bottleneck_mmd=args.use_bottleneck_mmd,
        source_params=source_params,
        lambda_l2sp=args.lambda_l2sp,
        use_production_mmd=getattr(args, 'production_mmd', False),
        lambda_production_mmd=getattr(args, 'lambda_production_mmd', 0.1),
    )

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")
    print(f"Target acc AFTER  adaptation: {metrics.target_acc:.4f}")
    print(f"Improvement: {metrics.target_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

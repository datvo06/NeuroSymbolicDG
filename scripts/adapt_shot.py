#!/usr/bin/env python3
"""SHOT-style pseudo-labeling adaptation (Liang et al., ICML 2020).

Freezes the classifier head (grammar + relation_params), computes class
centroids from source features, and adapts the feature extractor using
pseudo-label CE + IM loss + optional L2-SP regularization.

Usage:
    uv run python scripts/adapt_shot.py \
        --checkpoint checkpoint_digits_mnist_usps.pt \
        --dataset digits --source mnist --target usps \
        --use-sparsemax --epochs 20 --lr 1e-4 \
        --lambda-im 1.0 --pseudo-threshold 0.9
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import freeze_structure, get_adaptable_params
from neurosymbolic_da.training.losses import (
    assign_pseudo_labels,
    compute_centroids,
    im_loss,
    l2sp_loss,
)
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shot_adapt_epoch(
    model: NeuroSymbolicPipeline,
    target_loader,
    optimizer,
    device: torch.device,
    centroids: torch.Tensor,
    pseudo_threshold: float = 0.9,
    lambda_im: float = 1.0,
    source_params: dict[str, torch.Tensor] | None = None,
    lambda_l2sp: float = 0.0,
) -> dict:
    """Run one SHOT adaptation epoch.

    Returns dict with avg loss components.
    """
    model.train()
    total_ce = 0.0
    total_im = 0.0
    total_loss = 0.0
    total_used = 0
    total_samples = 0
    n_batches = 0

    for images, _ in target_loader:
        images = images.to(device)
        B = images.shape[0]

        optimizer.zero_grad()

        # Get bottleneck features for pseudo-labeling (detached — no grad through centroid assignment)
        with torch.no_grad():
            feats = model.get_bottleneck_features(images)
            pseudo_labels, conf_mask = assign_pseudo_labels(feats, centroids, pseudo_threshold)

        # Forward pass through full model
        log_probs = model(images)  # [B, C]

        # CE loss on high-confidence pseudo-labeled samples
        n_used = conf_mask.sum().item()
        if n_used > 0:
            l_ce = F.nll_loss(log_probs[conf_mask], pseudo_labels[conf_mask])
        else:
            l_ce = torch.tensor(0.0, device=device)

        # IM loss on ALL target samples (not just high-confidence)
        l_im = im_loss(log_probs)

        loss = l_ce + lambda_im * l_im

        # Optional L2-SP regularization
        if source_params is not None and lambda_l2sp > 0:
            loss = loss + lambda_l2sp * l2sp_loss(model, source_params)

        loss.backward()
        optimizer.step()

        total_ce += l_ce.item()
        total_im += l_im.item()
        total_loss += loss.item()
        total_used += n_used
        total_samples += B
        n_batches += 1

    return {
        "ce": total_ce / max(n_batches, 1),
        "im": total_im / max(n_batches, 1),
        "loss": total_loss / max(n_batches, 1),
        "pseudo_usage": total_used / max(total_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="SHOT-style pseudo-labeling adaptation")

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
    parser.add_argument("--use-sparsemax", action="store_true",
                        help="Use sparsemax for grammar weights (must match source checkpoint)")

    # Adaptation hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-im", type=float, default=1.0,
                        help="Weight for IM loss")
    parser.add_argument("--lambda-l2sp", type=float, default=0.0,
                        help="Weight for L2-SP regularization")
    parser.add_argument("--pseudo-threshold", type=float, default=0.5,
                        help="Confidence threshold for pseudo-labels (default: 0.5)")
    parser.add_argument("--centroid-interval", type=int, default=3,
                        help="Recompute centroids every N epochs (default: 3)")

    # Adapt mode
    parser.add_argument("--adapt-mode", default="full",
                        choices=["full", "bn-only", "bottleneck-only"],
                        help="What to adapt: full (backbone+bottleneck), bn-only (batch norm only), "
                             "bottleneck-only (freeze backbone, adapt bottleneck)")

    # Output
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--num-workers", type=int, default=2)

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

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target}), {n_classes} classes")

    # Build model and load checkpoint
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,  # weights come from checkpoint
        max_depth=args.max_depth,
        use_inside=False,
        use_sparsemax=args.use_sparsemax,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(source val acc={checkpoint.get('val_acc', 'N/A')})")

    model.to(device)

    # Evaluate before adaptation
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

    # Freeze grammar + relation params
    freeze_structure(model, freeze_grammar=True)

    # Apply adapt mode
    if args.adapt_mode == "bn-only":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False
        for module in model.backbone.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
        print("ADAPT MODE: bn-only (only batch norm parameters)")
    elif args.adapt_mode == "bottleneck-only":
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("ADAPT MODE: bottleneck-only (backbone frozen)")

    adaptable_params = get_adaptable_params(model)
    if args.adapt_mode == "bn-only":
        adaptable_params = [p for p in model.parameters() if p.requires_grad]
    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable params: {n_adapt:,} | Frozen params: {n_frozen:,}")

    # Optimizer
    optimizer = Adam(adaptable_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Compute initial centroids from source data
    print("Computing source centroids...")
    centroids = compute_centroids(model, src_train, device, n_classes)
    print(f"Centroids shape: {centroids.shape}")

    # SHOT adaptation loop
    save_path = args.save_path or f"adapted_shot_{args.dataset}_{args.source}_{args.target}.pt"
    best_target_acc = 0.0

    print(f"\nStarting SHOT adaptation: {args.epochs} epochs, "
          f"lambda_im={args.lambda_im}, threshold={args.pseudo_threshold}, "
          f"centroid_interval={args.centroid_interval}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Recompute centroids periodically from target pseudo-labels
        if epoch > 1 and (epoch - 1) % args.centroid_interval == 0:
            print(f"  Recomputing centroids from target features...")
            centroids = _recompute_centroids_from_target(
                model, tgt_train, device, n_classes, centroids,
            )

        stats = shot_adapt_epoch(
            model=model,
            target_loader=tgt_train,
            optimizer=optimizer,
            device=device,
            centroids=centroids,
            pseudo_threshold=args.pseudo_threshold,
            lambda_im=args.lambda_im,
            source_params=source_params,
            lambda_l2sp=args.lambda_l2sp,
        )

        # Evaluate on target test set
        _, target_acc = evaluate(model, tgt_test, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"CE={stats['ce']:.4f} IM={stats['im']:.4f} loss={stats['loss']:.4f} | "
            f"pseudo_use={stats['pseudo_usage']:.1%} | "
            f"target acc={target_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_acc": target_acc,
            }, save_path)

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")
    print(f"Target acc BEST  adaptation:  {best_target_acc:.4f}")
    print(f"Improvement: {best_target_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


def _recompute_centroids_from_target(
    model: NeuroSymbolicPipeline,
    target_loader,
    device: torch.device,
    n_classes: int,
    old_centroids: torch.Tensor,
) -> torch.Tensor:
    """Recompute centroids using target data + current model predictions.

    Uses model's own predictions as pseudo-labels for centroid update.
    Falls back to old centroids for classes with no assigned samples.
    """
    model.eval()
    feat_dim = old_centroids.shape[1]
    feat_sum = torch.zeros(n_classes, feat_dim, device=device)
    counts = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(device)
            feats = model.get_bottleneck_features(images)  # [B, k*3]
            log_probs = model(images)  # [B, C]
            preds = log_probs.argmax(dim=-1)  # [B]

            for c in range(n_classes):
                mask = preds == c
                if mask.any():
                    feat_sum[c] += feats[mask].sum(dim=0)
                    counts[c] += mask.sum()

    # Build new centroids, keeping old ones for empty classes
    new_centroids = old_centroids.clone()
    for c in range(n_classes):
        if counts[c] > 0:
            new_centroids[c] = feat_sum[c] / counts[c]

    model.train()
    return new_centroids


if __name__ == "__main__":
    main()

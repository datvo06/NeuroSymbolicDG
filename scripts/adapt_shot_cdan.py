#!/usr/bin/env python3
"""SHOT+CDAN combined adaptation.

Combines CDAN adversarial domain alignment with SHOT pseudo-labeling:
- CDAN: discriminator conditioned on [features | softmax(preds)] + GRL
- SHOT: centroid-based pseudo-labels + CE loss on high-confidence targets
- IM loss on all target samples
- L2-SP regularization

Loss = l_task + lambda_im * l_im + lambda_adv * l_adv + l_pseudo_ce + lambda_l2sp * l2sp

Usage:
    uv run python scripts/adapt_shot_cdan.py \
        --checkpoint checkpoints/pcfg_sparse_office31_r50_dslr_amazon.pt \
        --dataset office31 --source dslr --target amazon \
        --backbone resnet50 --n-primitives 8 --use-sparsemax \
        --epochs 30 --lr 1e-4 --lr-disc 1e-3 \
        --lambda-adv 1.0 --lambda-im 1.0 --lambda-l2sp 0.01 \
        --pseudo-threshold 0.9
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import freeze_structure, get_adaptable_params
from neurosymbolic_da.training.adversarial import (
    DomainDiscriminator,
    GradientReversalLayer,
    cdan_condition,
)
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


def shot_cdan_adapt_epoch(
    model: NeuroSymbolicPipeline,
    discriminator: DomainDiscriminator,
    grl: GradientReversalLayer,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    optimizer_feat: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device,
    centroids: torch.Tensor,
    pseudo_threshold: float = 0.9,
    lambda_adv: float = 1.0,
    lambda_im: float = 1.0,
    lambda_pseudo: float = 1.0,
    source_params: dict[str, torch.Tensor] | None = None,
    lambda_l2sp: float = 0.0,
) -> dict[str, float]:
    """Run one SHOT+CDAN adaptation epoch.

    Iterates over source_loader, cycles target_loader. Each step:
    1. Update discriminator on detached features
    2. Update feature extractor with:
       - Source NLL task loss
       - Target pseudo-label CE (high-confidence, from SHOT centroids)
       - Target IM loss
       - CDAN adversarial loss (via GRL)
       - L2-SP regularization
    """
    model.train()
    discriminator.train()

    bce = nn.BCEWithLogitsLoss()

    totals = {
        "disc": 0.0, "adv": 0.0, "im": 0.0, "pseudo_ce": 0.0,
        "l2sp": 0.0, "total": 0.0,
    }
    total_pseudo_used = 0
    total_target_samples = 0
    n_batches = 0

    target_iter = iter(target_loader)

    for source_batch in source_loader:
        source_x, source_y = source_batch[0].to(device), source_batch[1].to(device)

        # Get target batch (cycle if shorter)
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        target_x = target_batch[0].to(device)

        # ---- Step 1: Update discriminator ----
        optimizer_disc.zero_grad()

        with torch.no_grad():
            src_feats = model.get_bottleneck_features(source_x)
            src_log_probs = model(source_x)
            tgt_feats = model.get_bottleneck_features(target_x)
            tgt_log_probs = model(target_x)

        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)

        src_domain = torch.zeros(src_cond.size(0), 1, device=device)
        tgt_domain = torch.ones(tgt_cond.size(0), 1, device=device)

        disc_input = torch.cat([src_cond, tgt_cond], dim=0)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)

        disc_logits = discriminator(disc_input)
        l_disc = bce(disc_logits, disc_labels)

        l_disc.backward()
        optimizer_disc.step()

        # ---- Step 2: Update feature extractor ----
        optimizer_feat.zero_grad()

        # Forward through model (need gradients)
        src_feats = model.get_bottleneck_features(source_x)
        src_log_probs = model(source_x)
        tgt_feats = model.get_bottleneck_features(target_x)
        tgt_log_probs = model(target_x)

        # Source task loss (NLL)
        l_task = F.nll_loss(src_log_probs, source_y)

        # Target pseudo-label CE (SHOT component)
        with torch.no_grad():
            pseudo_labels, conf_mask = assign_pseudo_labels(
                tgt_feats, centroids, pseudo_threshold
            )
        n_used = conf_mask.sum().item()
        if n_used > 0:
            l_pseudo = F.nll_loss(tgt_log_probs[conf_mask], pseudo_labels[conf_mask])
        else:
            l_pseudo = torch.tensor(0.0, device=device)

        # IM loss on ALL target samples
        l_im = im_loss(tgt_log_probs)

        # Adversarial loss via GRL (CDAN component)
        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)
        cond_all = torch.cat([src_cond, tgt_cond], dim=0)
        cond_reversed = grl(cond_all)
        disc_logits = discriminator(cond_reversed)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)
        l_adv = bce(disc_logits, disc_labels)

        loss = (l_task
                + lambda_im * l_im
                + lambda_adv * l_adv
                + lambda_pseudo * l_pseudo)

        # L2-SP regularization
        l_l2sp_val = 0.0
        if source_params is not None and lambda_l2sp > 0:
            l_l2sp = l2sp_loss(model, source_params)
            loss = loss + lambda_l2sp * l_l2sp
            l_l2sp_val = l_l2sp.item()

        loss.backward()
        optimizer_feat.step()

        totals["disc"] += l_disc.item()
        totals["adv"] += l_adv.item()
        totals["im"] += l_im.item()
        totals["pseudo_ce"] += l_pseudo.item()
        totals["l2sp"] += l_l2sp_val
        totals["total"] += loss.item()
        total_pseudo_used += n_used
        total_target_samples += target_x.size(0)
        n_batches += 1

    avgs = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avgs["pseudo_usage"] = total_pseudo_used / max(total_target_samples, 1)
    return avgs


def _recompute_centroids_from_target(
    model: NeuroSymbolicPipeline,
    target_loader,
    device: torch.device,
    n_classes: int,
    old_centroids: torch.Tensor,
) -> torch.Tensor:
    """Recompute centroids using target data + current model predictions."""
    model.eval()
    feat_dim = old_centroids.shape[1]
    feat_sum = torch.zeros(n_classes, feat_dim, device=device)
    counts = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(device)
            feats = model.get_bottleneck_features(images)
            log_probs = model(images)
            preds = log_probs.argmax(dim=-1)

            for c in range(n_classes):
                mask = preds == c
                if mask.any():
                    feat_sum[c] += feats[mask].sum(dim=0)
                    counts[c] += mask.sum()

    new_centroids = old_centroids.clone()
    for c in range(n_classes):
        if counts[c] > 0:
            new_centroids[c] = feat_sum[c] / counts[c]

    model.train()
    return new_centroids


def main():
    parser = argparse.ArgumentParser(description="SHOT+CDAN combined adaptation")

    # Checkpoint
    parser.add_argument("--checkpoint", required=True)

    # Dataset
    parser.add_argument("--dataset", required=True,
                        choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    # Model (must match checkpoint)
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true")

    # Adaptation hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Feature extractor LR")
    parser.add_argument("--lr-disc", type=float, default=1e-3, help="Discriminator LR")
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-im", type=float, default=1.0)
    parser.add_argument("--lambda-pseudo", type=float, default=1.0,
                        help="Weight for pseudo-label CE loss")
    parser.add_argument("--lambda-l2sp", type=float, default=0.01)
    parser.add_argument("--pseudo-threshold", type=float, default=0.8)
    parser.add_argument("--centroid-interval", type=int, default=3)

    # Adapt mode
    parser.add_argument("--adapt-mode", default="full",
                        choices=["full", "bn-only"])

    # Output
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
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

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target}), {n_classes} classes")

    # Build model and load checkpoint
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
        max_depth=args.max_depth,
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

    # Save source params for L2-SP
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
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
        print("ADAPT MODE: bn-only (only batch norm parameters)")

    # Get adaptable params
    adaptable_params = get_adaptable_params(model)
    if args.adapt_mode == "bn-only":
        adaptable_params = [p for p in model.parameters() if p.requires_grad]

    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable params: {n_adapt:,} | Frozen params: {n_frozen:,}")

    # Build discriminator and GRL
    feat_dim = args.n_primitives * 3
    cond_dim = feat_dim + n_classes
    discriminator = DomainDiscriminator(cond_dim, hidden_dim=1024, dropout=0.5).to(device)
    grl = GradientReversalLayer(lambda_=args.lambda_adv)

    n_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator params: {n_disc:,} (input dim={cond_dim})")

    # Optimizers
    optimizer_feat = Adam(adaptable_params, lr=args.lr)
    optimizer_disc = Adam(discriminator.parameters(), lr=args.lr_disc)
    scheduler_feat = CosineAnnealingLR(optimizer_feat, T_max=args.epochs)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs)

    # Compute initial centroids from source data
    print("Computing source centroids...")
    centroids = compute_centroids(model, src_train, device, n_classes)
    print(f"Centroids shape: {centroids.shape}")

    # Adaptation loop
    save_path = args.save_path or f"adapted_shot_cdan_{args.dataset}_{args.source}_{args.target}.pt"
    best_target_acc = 0.0

    print(f"\nStarting SHOT+CDAN adaptation for {args.epochs} epochs")
    print(f"  lambda_adv={args.lambda_adv}, lambda_im={args.lambda_im}, "
          f"lambda_pseudo={args.lambda_pseudo}, lambda_l2sp={args.lambda_l2sp}")
    print(f"  pseudo_threshold={args.pseudo_threshold}, "
          f"centroid_interval={args.centroid_interval}")
    print(f"  lr_feat={args.lr}, lr_disc={args.lr_disc}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Recompute centroids periodically
        if epoch > 1 and (epoch - 1) % args.centroid_interval == 0:
            print(f"  Recomputing centroids from target features...")
            centroids = _recompute_centroids_from_target(
                model, tgt_train, device, n_classes, centroids,
            )

        # Progressive GRL lambda (DANN schedule)
        p = epoch / args.epochs
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        grl.set_lambda(grl_lambda * args.lambda_adv)

        losses = shot_cdan_adapt_epoch(
            model=model,
            discriminator=discriminator,
            grl=grl,
            source_loader=src_train,
            target_loader=tgt_train,
            optimizer_feat=optimizer_feat,
            optimizer_disc=optimizer_disc,
            device=device,
            centroids=centroids,
            pseudo_threshold=args.pseudo_threshold,
            lambda_adv=args.lambda_adv,
            lambda_im=args.lambda_im,
            lambda_pseudo=args.lambda_pseudo,
            source_params=source_params,
            lambda_l2sp=args.lambda_l2sp,
        )

        # Evaluate on target test set
        _, target_acc = evaluate(model, tgt_test, device)

        scheduler_feat.step()
        scheduler_disc.step()

        epoch_time = time.time() - t0

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"disc={losses['disc']:.4f} adv={losses['adv']:.4f} "
                f"im={losses['im']:.4f} pce={losses['pseudo_ce']:.4f} "
                f"total={losses['total']:.4f} | "
                f"pseudo_use={losses['pseudo_usage']:.1%} "
                f"target_acc={target_acc:.4f} grl_λ={grl_lambda:.3f} | "
                f"{epoch_time:.1f}s"
            )

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "target_acc": target_acc,
                "args": vars(args),
            }, save_path)

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")
    print(f"Target acc AFTER  adaptation: {target_acc:.4f}")
    print(f"Best target acc:              {best_target_acc:.4f}")
    print(f"Improvement (final):  {target_acc - pre_adapt_acc:+.4f}")
    print(f"Improvement (best):   {best_target_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

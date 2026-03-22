#!/usr/bin/env python3
"""CDAN adaptation for NoPCFG model (backbone + bottleneck + linear head).

Same CDAN adversarial alignment as adapt_cdan.py but using NoPCFGPipeline.
Tests whether simpler linear classifier features align better than PCFG.

Usage:
    python scripts/adapt_cdan_nopcfg.py \
        --checkpoint checkpoints/nopcfg_office31_r50_dslr_amazon.pt \
        --dataset office31 --source dslr --target amazon \
        --backbone resnet50 --n-primitives 8 \
        --epochs 30 --lr 1e-4 --lr-disc 1e-3 \
        --lambda-adv 1.0 --lambda-im 1.0 --lambda-l2sp 0.01 \
        --align-level backbone
"""

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline_nopcfg import NoPCFGPipeline
from neurosymbolic_da.training.adversarial import (
    DomainDiscriminator,
    GradientReversalLayer,
    cdan_condition,
)
from neurosymbolic_da.training.losses import im_loss, l2sp_loss
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cdan_adapt_epoch(
    model, discriminator, grl,
    source_loader, target_loader,
    optimizer_feat, optimizer_disc,
    device, lambda_adv=1.0, lambda_im=1.0,
    source_params=None, lambda_l2sp=0.0,
    align_level="backbone",
):
    model.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()

    totals = {"disc": 0.0, "adv": 0.0, "im": 0.0, "l2sp": 0.0, "total": 0.0}
    n_batches = 0
    target_iter = iter(target_loader)

    for source_batch in source_loader:
        source_x, source_y = source_batch[0].to(device), source_batch[1].to(device)
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        target_x = target_batch[0].to(device)

        get_feats = (model.get_backbone_features if align_level == "backbone"
                     else model.get_bottleneck_features)

        # Step 1: Update discriminator
        optimizer_disc.zero_grad()
        with torch.no_grad():
            src_feats = get_feats(source_x)
            src_log_probs = model(source_x)
            tgt_feats = get_feats(target_x)
            tgt_log_probs = model(target_x)

        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)
        src_domain = torch.zeros(src_cond.size(0), 1, device=device)
        tgt_domain = torch.ones(tgt_cond.size(0), 1, device=device)

        disc_input = torch.cat([src_cond, tgt_cond], dim=0)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)
        l_disc = bce(discriminator(disc_input), disc_labels)
        l_disc.backward()
        optimizer_disc.step()

        # Step 2: Update feature extractor
        optimizer_feat.zero_grad()
        src_feats = get_feats(source_x)
        src_log_probs = model(source_x)
        tgt_feats = get_feats(target_x)
        tgt_log_probs = model(target_x)

        l_task = nn.functional.nll_loss(src_log_probs, source_y)
        l_im = im_loss(tgt_log_probs)

        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)
        cond_all = torch.cat([src_cond, tgt_cond], dim=0)
        cond_reversed = grl(cond_all)
        disc_logits = discriminator(cond_reversed)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)
        l_adv = bce(disc_logits, disc_labels)

        loss = l_task + lambda_im * l_im + lambda_adv * l_adv

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
        totals["l2sp"] += l_l2sp_val
        totals["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="CDAN adaptation for NoPCFG")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-disc", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-im", type=float, default=1.0)
    parser.add_argument("--lambda-l2sp", type=float, default=0.01)
    parser.add_argument("--align-level", default="backbone",
                        choices=["bottleneck", "backbone"])
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    image_size = 32 if args.backbone == "lenet" else 224
    n_classes = get_n_classes(args.dataset)
    src_train, src_test, tgt_train, tgt_test = get_loaders(
        args.dataset, args.source, args.target,
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=image_size,
    )
    print(f"Dataset: {args.dataset} ({args.source} -> {args.target}), {n_classes} classes")

    model = NoPCFGPipeline(
        n_primitives=args.n_primitives, n_classes=n_classes,
        backbone_variant=args.backbone, pretrained_backbone=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(source val acc={checkpoint.get('val_acc', 'N/A')})")

    model.to(device)
    _, pre_adapt_acc = evaluate(model, tgt_test, device)
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")

    source_params = None
    if args.lambda_l2sp > 0:
        source_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        print(f"L2-SP: lambda={args.lambda_l2sp}")

    # Freeze nothing extra for NoPCFG — adapt all params
    adaptable_params = list(model.parameters())
    n_adapt = sum(p.numel() for p in adaptable_params)
    print(f"Adaptable params: {n_adapt:,}")

    if args.align_level == "backbone":
        feat_dim = model.backbone.out_channels
    else:
        feat_dim = args.n_primitives * 3
    cond_dim = feat_dim + n_classes
    print(f"ALIGN: {args.align_level} ({feat_dim}-dim + {n_classes} classes = {cond_dim}-dim)")

    discriminator = DomainDiscriminator(cond_dim, hidden_dim=1024, dropout=0.5).to(device)
    grl = GradientReversalLayer(lambda_=args.lambda_adv)

    optimizer_feat = Adam(adaptable_params, lr=args.lr)
    optimizer_disc = Adam(discriminator.parameters(), lr=args.lr_disc)
    scheduler_feat = CosineAnnealingLR(optimizer_feat, T_max=args.epochs)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs)

    save_path = args.save_path or f"adapted_cdan_nopcfg_{args.dataset}_{args.source}_{args.target}.pt"
    best_target_acc = 0.0

    print(f"\nStarting NoPCFG CDAN adaptation for {args.epochs} epochs")
    print(f"  adv={args.lambda_adv}, im={args.lambda_im}, l2sp={args.lambda_l2sp}")
    print(f"  lr={args.lr}, lr_disc={args.lr_disc}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        p = epoch / args.epochs
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        grl.set_lambda(grl_lambda * args.lambda_adv)

        losses = cdan_adapt_epoch(
            model, discriminator, grl, src_train, tgt_train,
            optimizer_feat, optimizer_disc, device,
            lambda_adv=args.lambda_adv, lambda_im=args.lambda_im,
            source_params=source_params, lambda_l2sp=args.lambda_l2sp,
            align_level=args.align_level,
        )

        _, target_acc = evaluate(model, tgt_test, device)
        scheduler_feat.step()
        scheduler_disc.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"disc={losses['disc']:.4f} adv={losses['adv']:.4f} "
            f"im={losses['im']:.4f} total={losses['total']:.4f} | "
            f"target_acc={target_acc:.4f} grl_l={grl_lambda:.3f} | "
            f"{elapsed:.1f}s"
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
    print(f"Improvement (best):   {best_target_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

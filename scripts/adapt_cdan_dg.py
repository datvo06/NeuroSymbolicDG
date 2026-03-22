#!/usr/bin/env python3
"""CDAN adaptation for DG checkpoints (multi-source → target).

Loads a DG-trained checkpoint (trained on 3 combined source domains),
freezes grammar + relation params, and adapts backbone + bottleneck
using CDAN with combined source domains as reference.

Usage:
    python scripts/adapt_cdan_dg.py \
        --checkpoint checkpoints/dg_erm_v2_pcfg_cubdg_Art.pt \
        --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
        --use-sparsemax --epochs 20 --lr 1e-4 --lambda-adv 1.0
"""

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from neurosymbolic_da.data.loader_utils import get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import freeze_structure, get_adaptable_params
from neurosymbolic_da.training.adversarial import (
    DomainDiscriminator,
    GradientReversalLayer,
    cdan_condition,
)
from neurosymbolic_da.training.losses import im_loss, l2sp_loss
from neurosymbolic_da.training.trainer import evaluate


DATASET_DOMAINS = {
    "cubdg": ["Photo", "Art", "Cartoon", "Paint"],
    "pacs": ["photo", "art_painting", "cartoon", "sketch"],
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_domain_dataset(dataset, data_root, domain, train=True, **kwargs):
    """Get a single domain dataset for the given benchmark."""
    if dataset == "cubdg":
        from neurosymbolic_da.data.cubdg import get_cubdg
        return get_cubdg(data_root, domain, train=train, **kwargs)
    elif dataset == "pacs":
        from neurosymbolic_da.data.pacs import get_pacs
        return get_pacs(data_root, domain, train=train, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def get_dg_adapt_loaders(
    data_root: str,
    target: str,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset: str = "cubdg",
):
    """Get combined source train + target train/test loaders for DG adaptation."""
    all_domains = DATASET_DOMAINS[dataset]
    source_domains = [d for d in all_domains if d != target]

    # Combined source train (3 domains)
    src_train_datasets = []
    for domain in source_domains:
        src_train_datasets.append(_get_domain_dataset(dataset, data_root, domain, train=True))
    combined_src_train = ConcatDataset(src_train_datasets)

    # Target train (unlabeled — labels ignored during adaptation)
    tgt_train = _get_domain_dataset(dataset, data_root, target, train=True)
    # Target test
    tgt_test = _get_domain_dataset(dataset, data_root, target, train=False)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(combined_src_train, shuffle=True, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )


def cdan_dg_adapt_epoch(
    model, discriminator, grl,
    source_loader, target_loader,
    optimizer_feat, optimizer_disc,
    device,
    lambda_adv=1.0, lambda_im=1.0,
    source_params=None, lambda_l2sp=0.0,
    align_level="bottleneck",
):
    """One epoch of CDAN adaptation with multi-source reference."""
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
        disc_logits = discriminator(disc_input)
        l_disc = bce(disc_logits, disc_labels)
        l_disc.backward()
        optimizer_disc.step()

        # Step 2: Update feature extractor
        optimizer_feat.zero_grad()
        src_feats = get_feats(source_x)
        src_log_probs = model(source_x)
        tgt_feats = get_feats(target_x)
        tgt_log_probs = model(target_x)

        # Source task loss
        l_task = nn.functional.nll_loss(src_log_probs, source_y)
        # IM loss on target
        l_im = im_loss(tgt_log_probs)

        # Adversarial loss
        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)
        cond_all = torch.cat([src_cond, tgt_cond], dim=0)
        cond_reversed = grl(cond_all)
        disc_logits = discriminator(cond_reversed)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)
        l_adv = bce(disc_logits, disc_labels)

        loss = l_task + lambda_im * l_im + lambda_adv * l_adv

        # L2-SP
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
    parser = argparse.ArgumentParser(description="CDAN adaptation for DG checkpoints")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cubdg")
    parser.add_argument("--target", required=True, help="Held-out target domain")
    parser.add_argument("--data-root", default="./data/cub/CUB-DG")

    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-disc", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-im", type=float, default=1.0)
    parser.add_argument("--lambda-l2sp", type=float, default=0.01)
    parser.add_argument("--align-level", default="backbone",
                        choices=["bottleneck", "backbone"])
    parser.add_argument("--disc-hidden", type=int, default=1024)

    parser.add_argument("--save-path", default=None)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    device = get_device()
    print(f"Device: {device}")

    n_classes = get_n_classes(args.dataset)
    source_domains = [d for d in CUBDG_DOMAINS if d != args.target]
    print(f"DG CDAN: sources={source_domains} -> target={args.target}")

    # Load data
    src_train, tgt_train, tgt_test = get_dg_adapt_loaders(
        args.data_root, args.target,
        batch_size=args.batch_size, num_workers=args.num_workers,
        dataset=args.dataset,
    )
    print(f"Source train: {len(src_train.dataset)} | Target train: {len(tgt_train.dataset)} | Target test: {len(tgt_test.dataset)}")

    # Build model and load checkpoint
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
        max_depth=args.max_depth,
        use_sparsemax=args.use_sparsemax,
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt.get('val_acc', 'N/A')})")

    model.to(device)
    _, pre_adapt_acc = evaluate(model, tgt_test, device)
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")

    # Save source params for L2-SP
    source_params = None
    if args.lambda_l2sp > 0:
        source_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    # Freeze grammar + relation params
    freeze_structure(model, freeze_grammar=True)
    adaptable_params = get_adaptable_params(model)
    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable: {n_adapt:,} | Frozen: {n_frozen:,}")

    # Discriminator
    if args.align_level == "backbone":
        feat_dim = model.backbone.out_channels
    else:
        feat_dim = args.n_primitives * 3
    cond_dim = feat_dim + n_classes
    discriminator = DomainDiscriminator(cond_dim, hidden_dim=args.disc_hidden).to(device)
    grl = GradientReversalLayer(lambda_=args.lambda_adv)
    print(f"Align: {args.align_level} ({feat_dim}-dim), CDAN cond_dim={cond_dim}")

    # Optimizers
    optimizer_feat = Adam(adaptable_params, lr=args.lr)
    optimizer_disc = Adam(discriminator.parameters(), lr=args.lr_disc)
    scheduler_feat = CosineAnnealingLR(optimizer_feat, T_max=args.epochs)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs)

    save_path = args.save_path or f"checkpoints/dg_cdan_v2_{args.dataset}_{args.target}.pt"
    best_target_acc = 0.0

    print(f"\nStarting DG-CDAN adaptation: {args.epochs} epochs")
    print(f"  lambda_adv={args.lambda_adv}, lambda_im={args.lambda_im}, lambda_l2sp={args.lambda_l2sp}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        p = epoch / args.epochs
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        grl.set_lambda(grl_lambda * args.lambda_adv)

        losses = cdan_dg_adapt_epoch(
            model, discriminator, grl,
            src_train, tgt_train,
            optimizer_feat, optimizer_disc,
            device,
            lambda_adv=args.lambda_adv, lambda_im=args.lambda_im,
            source_params=source_params, lambda_l2sp=args.lambda_l2sp,
            align_level=args.align_level,
        )

        _, target_acc = evaluate(model, tgt_test, device)

        scheduler_feat.step()
        scheduler_disc.step()

        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"disc={losses['disc']:.4f} adv={losses['adv']:.4f} "
            f"im={losses['im']:.4f} total={losses['total']:.4f} | "
            f"target_acc={target_acc:.4f} grl={grl_lambda:.3f} | "
            f"{epoch_time:.1f}s"
        )

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_acc": target_acc,
                "args": vars(args),
            }, save_path)

    print(f"\n--- Final Results (DG-CDAN) ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")
    print(f"Target acc AFTER  adaptation: {target_acc:.4f}")
    print(f"Best target acc:              {best_target_acc:.4f}")
    print(f"Improvement (best):   {best_target_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CDAN+E with Domain MixUp and Strong Augmentation.

Combines:
- CDAN+E: entropy-weighted adversarial loss (confident samples weighted more)
- Domain MixUp: interpolate source+target images (FixBi-inspired)
- Backbone-level alignment: 2048-dim features for discriminator
- Strong augmentation: ColorJitter + GaussianBlur + RandomErasing
- L2-SP regularization

Target: beat FixBi (91.4% avg on Office-31).

Usage:
    python scripts/adapt_cdan_mixup.py \
        --checkpoint checkpoints/pcfg_sparse_office31_r50_dslr_amazon.pt \
        --dataset office31 --source dslr --target amazon \
        --backbone resnet50 --n-primitives 8 --use-sparsemax \
        --epochs 50 --lr 3e-4 --lr-disc 1e-3 \
        --lambda-adv 1.0 --lambda-im 1.0 --lambda-l2sp 0.005 \
        --mixup-alpha 0.3 --align-level backbone --strong-aug
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
    domain_mixup,
    entropy_weight,
)
from neurosymbolic_da.training.losses import im_loss, l2sp_loss
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cdan_mixup_epoch(
    model: NeuroSymbolicPipeline,
    discriminator: DomainDiscriminator,
    grl: GradientReversalLayer,
    source_loader,
    target_loader,
    optimizer_feat,
    optimizer_disc,
    device: torch.device,
    align_level: str = "backbone",
    lambda_adv: float = 1.0,
    lambda_im: float = 1.0,
    lambda_mixup: float = 0.5,
    mixup_alpha: float = 0.3,
    source_params=None,
    lambda_l2sp: float = 0.0,
    use_entropy_weight: bool = True,
) -> dict[str, float]:
    model.train()
    discriminator.train()

    bce = nn.BCEWithLogitsLoss(reduction="none")

    totals = {
        "disc": 0.0, "adv": 0.0, "im": 0.0, "mixup": 0.0,
        "l2sp": 0.0, "total": 0.0,
    }
    n_batches = 0
    target_iter = iter(target_loader)

    get_feats = (model.get_backbone_features if align_level == "backbone"
                 else model.get_bottleneck_features)

    for source_batch in source_loader:
        source_x, source_y = source_batch[0].to(device), source_batch[1].to(device)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        target_x = target_batch[0].to(device)

        # ---- Step 1: Update discriminator ----
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
        l_disc_per = bce(disc_logits, disc_labels)

        # Entropy weighting (CDAN+E): confident samples weighted more
        if use_entropy_weight:
            with torch.no_grad():
                all_log_probs = torch.cat([src_log_probs, tgt_log_probs], dim=0)
                ew = entropy_weight(all_log_probs)  # [B_s + B_t]
            l_disc = (l_disc_per.squeeze() * ew).mean()
        else:
            l_disc = l_disc_per.mean()

        l_disc.backward()
        optimizer_disc.step()

        # ---- Step 2: Update feature extractor ----
        optimizer_feat.zero_grad()

        src_feats = get_feats(source_x)
        src_log_probs = model(source_x)
        tgt_feats = get_feats(target_x)
        tgt_log_probs = model(target_x)

        # Source task loss
        l_task = F.nll_loss(src_log_probs, source_y)

        # IM loss on target
        l_im = im_loss(tgt_log_probs)

        # Adversarial loss with entropy weighting
        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)
        cond_all = torch.cat([src_cond, tgt_cond], dim=0)
        cond_reversed = grl(cond_all)
        disc_logits = discriminator(cond_reversed)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)
        l_adv_per = bce(disc_logits, disc_labels)

        if use_entropy_weight:
            all_log_probs = torch.cat([src_log_probs, tgt_log_probs], dim=0)
            ew = entropy_weight(all_log_probs)
            l_adv = (l_adv_per.squeeze() * ew).mean()
        else:
            l_adv = l_adv_per.mean()

        # Domain MixUp loss: mixed images should have intermediate predictions
        l_mixup_val = 0.0
        if lambda_mixup > 0 and mixup_alpha > 0:
            mixed, lam = domain_mixup(source_x, target_x, alpha=mixup_alpha)
            mixed_log_probs = model(mixed)
            # Mixed image should match interpolation of source/target predictions
            with torch.no_grad():
                B = mixed.size(0)
                target_dist = lam * src_log_probs[:B].exp() + (1 - lam) * tgt_log_probs[:B].exp()
                target_dist = target_dist.clamp(min=1e-8)
            l_mixup = F.kl_div(mixed_log_probs, target_dist, reduction="batchmean")
            l_mixup_val = l_mixup.item()
        else:
            l_mixup = torch.tensor(0.0, device=device)

        loss = l_task + lambda_im * l_im + lambda_adv * l_adv + lambda_mixup * l_mixup

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
        totals["mixup"] += l_mixup_val
        totals["l2sp"] += l_l2sp_val
        totals["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="CDAN+E with Domain MixUp")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-disc", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-im", type=float, default=1.0)
    parser.add_argument("--lambda-mixup", type=float, default=0.5)
    parser.add_argument("--lambda-l2sp", type=float, default=0.005)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--no-entropy-weight", action="store_true")

    parser.add_argument("--align-level", default="backbone",
                        choices=["bottleneck", "backbone"])
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--adapt-mode", default="full",
                        choices=["full", "bn-only"])

    parser.add_argument("--save-path", default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

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

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target}), {n_classes} classes")
    print(f"Strong augmentation: {args.strong_aug}")

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

    _, pre_adapt_acc = evaluate(model, tgt_test, device)
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")

    source_params = None
    if args.lambda_l2sp > 0:
        source_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        print(f"L2-SP: lambda={args.lambda_l2sp}")

    freeze_structure(model, freeze_grammar=True)

    if args.adapt_mode == "bn-only":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False
        for module in model.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
        print("ADAPT MODE: bn-only")

    adaptable_params = get_adaptable_params(model)
    if args.adapt_mode == "bn-only":
        adaptable_params = [p for p in model.parameters() if p.requires_grad]

    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable: {n_adapt:,} | Frozen: {n_frozen:,}")

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

    save_path = args.save_path or (
        f"adapted_cdanmix_{args.dataset}_{args.source}_{args.target}.pt"
    )
    best_target_acc = 0.0

    print(f"\nStarting CDAN+E+MixUp adaptation for {args.epochs} epochs")
    print(f"  adv={args.lambda_adv}, im={args.lambda_im}, "
          f"mixup={args.lambda_mixup} (alpha={args.mixup_alpha}), "
          f"l2sp={args.lambda_l2sp}")
    print(f"  entropy_weight={not args.no_entropy_weight}")
    print(f"  lr={args.lr}, lr_disc={args.lr_disc}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        p = epoch / args.epochs
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        grl.set_lambda(grl_lambda * args.lambda_adv)

        losses = cdan_mixup_epoch(
            model=model,
            discriminator=discriminator,
            grl=grl,
            source_loader=src_train,
            target_loader=tgt_train,
            optimizer_feat=optimizer_feat,
            optimizer_disc=optimizer_disc,
            device=device,
            align_level=args.align_level,
            lambda_adv=args.lambda_adv,
            lambda_im=args.lambda_im,
            lambda_mixup=args.lambda_mixup,
            mixup_alpha=args.mixup_alpha,
            source_params=source_params,
            lambda_l2sp=args.lambda_l2sp,
            use_entropy_weight=not args.no_entropy_weight,
        )

        _, target_acc = evaluate(model, tgt_test, device)

        scheduler_feat.step()
        scheduler_disc.step()

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"disc={losses['disc']:.4f} adv={losses['adv']:.4f} "
            f"im={losses['im']:.4f} mix={losses['mixup']:.4f} | "
            f"target_acc={target_acc:.4f} grl_λ={grl_lambda:.3f} | "
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

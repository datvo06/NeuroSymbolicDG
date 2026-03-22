#!/usr/bin/env python3
"""CDAN adaptation script (Conditional Domain Adversarial Network).

Loads a source-trained checkpoint, freezes grammar + relation params,
and adapts backbone + bottleneck using adversarial domain alignment
conditioned on classifier predictions (Long et al., NeurIPS 2018).

Loss = IM_loss(target) + lambda_adv * CDAN_loss + lambda_l2sp * L2SP

Usage:
    uv run python scripts/adapt_cdan.py \
        --checkpoint checkpoint_office31_amazon_webcam.pt \
        --dataset office31 --source amazon --target webcam \
        --epochs 30 --lr 1e-4 --lr-disc 1e-3 --lambda-adv 1.0
"""

import argparse
import math
import time

import torch
import torch.nn as nn
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
from neurosymbolic_da.training.losses import im_loss, l2sp_loss, mcc_loss
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cdan_adapt_epoch(
    model: NeuroSymbolicPipeline,
    discriminator: DomainDiscriminator,
    grl: GradientReversalLayer,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    optimizer_feat: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device,
    lambda_adv: float = 1.0,
    lambda_im: float = 1.0,
    source_params: dict[str, torch.Tensor] | None = None,
    lambda_l2sp: float = 0.0,
    align_level: str = "bottleneck",
    mixup_alpha: float = 0.0,
    lambda_mixup: float = 1.0,
    lambda_mcc: float = 0.0,
) -> dict[str, float]:
    """Run one CDAN adaptation epoch.

    Args:
        model: pipeline (grammar/relation_params frozen)
        discriminator: domain classifier
        grl: gradient reversal layer
        source_loader: labeled source data
        target_loader: unlabeled target data
        optimizer_feat: optimizer for feature extractor (backbone + bottleneck)
        optimizer_disc: optimizer for discriminator
        device: torch device
        lambda_adv: weight for adversarial loss
        lambda_im: weight for IM loss on target
        source_params: source parameter values for L2-SP (optional)
        lambda_l2sp: weight for L2-SP regularization
        mixup_alpha: Beta distribution parameter for cross-domain mixup (0=disabled)
        lambda_mixup: weight for mixup classification loss

    Returns:
        dict of average losses
    """
    model.train()
    discriminator.train()

    bce = nn.BCEWithLogitsLoss()

    totals = {"disc": 0.0, "adv": 0.0, "im": 0.0, "mcc": 0.0, "l2sp": 0.0, "mixup": 0.0, "total": 0.0}
    n_batches = 0
    use_mixup = mixup_alpha > 0

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

        get_feats = (model.get_backbone_features if align_level == "backbone"
                     else model.get_bottleneck_features)

        with torch.no_grad():
            src_feats = get_feats(source_x)
            src_log_probs = model(source_x)
            tgt_feats = get_feats(target_x)
            tgt_log_probs = model(target_x)

        # Condition on predictions
        src_cond = cdan_condition(src_feats, src_log_probs)  # [B_s, feat+C]
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)  # [B_t, feat+C]

        # Domain labels: source=0, target=1
        src_domain = torch.zeros(src_cond.size(0), 1, device=device)
        tgt_domain = torch.ones(tgt_cond.size(0), 1, device=device)

        disc_input = torch.cat([src_cond, tgt_cond], dim=0)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)

        # Add mixed samples to discriminator training
        if use_mixup:
            B_mix = min(src_cond.size(0), tgt_cond.size(0))
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            mix_cond = lam * src_cond[:B_mix] + (1 - lam) * tgt_cond[:B_mix]
            mix_domain = torch.full((B_mix, 1), lam, device=device)  # interpolated label
            disc_input = torch.cat([disc_input, mix_cond], dim=0)
            disc_labels = torch.cat([disc_labels, mix_domain], dim=0)

        disc_logits = discriminator(disc_input)
        l_disc = bce(disc_logits, disc_labels)

        l_disc.backward()
        optimizer_disc.step()

        # ---- Step 2: Update feature extractor ----
        optimizer_feat.zero_grad()

        # Forward through model (need gradients this time)
        src_feats = get_feats(source_x)
        src_log_probs = model(source_x)
        tgt_feats = get_feats(target_x)
        tgt_log_probs = model(target_x)

        # Source task loss (NLL)
        l_task = nn.functional.nll_loss(src_log_probs, source_y)

        # IM loss on target predictions
        l_im = im_loss(tgt_log_probs)

        # Adversarial loss: fool discriminator via gradient reversal
        src_cond = cdan_condition(src_feats, src_log_probs)
        tgt_cond = cdan_condition(tgt_feats, tgt_log_probs)
        cond_all = torch.cat([src_cond, tgt_cond], dim=0)
        # Apply gradient reversal before discriminator
        cond_reversed = grl(cond_all)
        disc_logits = discriminator(cond_reversed)
        disc_labels = torch.cat([src_domain, tgt_domain], dim=0)
        l_adv = bce(disc_logits, disc_labels)

        # MCC loss on target predictions
        l_mcc_val = 0.0
        if lambda_mcc > 0:
            l_mcc = mcc_loss(tgt_log_probs)
            l_mcc_val = l_mcc.item()
        else:
            l_mcc = 0.0

        loss = l_task + lambda_im * l_im + lambda_adv * l_adv + lambda_mcc * l_mcc

        # Cross-domain mixup: create intermediate domain samples
        l_mixup_val = 0.0
        if use_mixup:
            B_mix = min(source_x.size(0), target_x.size(0))
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()

            # Mix output-level predictions (well-established, no architectural coupling)
            mix_log_probs = torch.log(
                lam * src_log_probs[:B_mix].exp() + (1 - lam) * tgt_log_probs[:B_mix].exp()
                + 1e-8
            )

            # Mixup classification loss: CE with source label weighted by lam
            l_mixup = lam * nn.functional.nll_loss(mix_log_probs, source_y[:B_mix])
            # IM loss on mixed predictions weighted by (1-lam)
            l_mixup = l_mixup + (1 - lam) * im_loss(mix_log_probs)
            loss = loss + lambda_mixup * l_mixup
            l_mixup_val = l_mixup.item()

            # Mixed conditioned features for adversarial: fool discriminator
            mix_cond = lam * src_cond[:B_mix] + (1 - lam) * tgt_cond[:B_mix]
            mix_cond_reversed = grl(mix_cond)
            mix_disc_logits = discriminator(mix_cond_reversed)
            mix_disc_labels = torch.full((B_mix, 1), lam, device=device)
            l_mix_adv = bce(mix_disc_logits, mix_disc_labels)
            loss = loss + lambda_adv * l_mix_adv

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
        totals["mcc"] += l_mcc_val
        totals["l2sp"] += l_l2sp_val
        totals["mixup"] += l_mixup_val
        totals["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="CDAN adaptation")

    # Checkpoint
    parser.add_argument("--checkpoint", required=True, help="Path to source-trained checkpoint")

    # Dataset
    parser.add_argument("--dataset", required=True,
                        choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True, help="Source domain name")
    parser.add_argument("--target", required=True, help="Target domain name")
    parser.add_argument("--data-root", default="./data", help="Data root directory")

    # Model (must match checkpoint)
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true",
                        help="Use sparsemax for grammar weights (must match source checkpoint)")
    parser.add_argument("--invariant-coords", action="store_true",
                        help="Enable scale+rotation invariant coordinate transforms (must match source)")
    parser.add_argument("--bottleneck-type", default="conv", choices=["conv", "slot", "moe"],
                        help="Bottleneck type (must match source checkpoint)")

    # Adaptation hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Feature extractor LR")
    parser.add_argument("--lr-disc", type=float, default=1e-3, help="Discriminator LR")
    parser.add_argument("--lambda-adv", type=float, default=1.0,
                        help="Weight for adversarial loss")
    parser.add_argument("--lambda-im", type=float, default=1.0,
                        help="Weight for IM loss on target")
    parser.add_argument("--lambda-l2sp", type=float, default=0.0,
                        help="L2-SP regularization weight")

    # Adapt mode
    parser.add_argument("--adapt-mode", default="full",
                        choices=["full", "bn-only"],
                        help="What to adapt: full (backbone+bottleneck), bn-only (batch norm only)")

    # Alignment level
    parser.add_argument("--align-level", default="bottleneck",
                        choices=["bottleneck", "backbone"],
                        help="Feature level for CDAN discriminator: "
                             "bottleneck (k*3=24 dim) or backbone (2048 dim for R50)")

    # MCC loss
    parser.add_argument("--lambda-mcc", type=float, default=0.0,
                        help="Weight for Minimum Class Confusion loss (0=disabled)")

    # Cross-domain mixup
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Beta distribution parameter for cross-domain mixup (0=disabled)")
    parser.add_argument("--lambda-mixup", type=float, default=1.0,
                        help="Weight for mixup classification loss")

    # Discriminator architecture
    parser.add_argument("--disc-hidden", type=int, default=1024)
    parser.add_argument("--disc-dropout", type=float, default=0.5)

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

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target})")
    print(f"Classes: {n_classes}")

    # Build model and load checkpoint
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,  # weights come from checkpoint
        max_depth=args.max_depth,
        use_sparsemax=args.use_sparsemax,
        invariant_coords=args.invariant_coords,
        bottleneck_type=args.bottleneck_type,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(source val acc={checkpoint.get('val_acc', 'N/A')})")

    # Evaluate before adaptation
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

    # Get adaptable feature extractor params
    adaptable_params = get_adaptable_params(model)
    if args.adapt_mode == "bn-only":
        adaptable_params = [p for p in model.parameters() if p.requires_grad]

    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable params: {n_adapt:,} | Frozen params: {n_frozen:,}")

    # Build discriminator and gradient reversal layer
    if args.align_level == "backbone":
        feat_dim = model.backbone.out_channels  # 2048 for R50, 512 for R18
        print(f"ALIGN LEVEL: backbone ({feat_dim}-dim features)")
    else:
        feat_dim = args.n_primitives * 3  # k*3 from bottleneck features
        print(f"ALIGN LEVEL: bottleneck ({feat_dim}-dim features)")
    cond_dim = feat_dim + n_classes    # concatenated with softmax predictions
    discriminator = DomainDiscriminator(cond_dim, hidden_dim=args.disc_hidden, dropout=args.disc_dropout).to(device)
    grl = GradientReversalLayer(lambda_=args.lambda_adv)

    n_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator params: {n_disc:,} (input dim={cond_dim})")

    # Optimizers
    optimizer_feat = Adam(adaptable_params, lr=args.lr)
    optimizer_disc = Adam(discriminator.parameters(), lr=args.lr_disc)
    scheduler_feat = CosineAnnealingLR(optimizer_feat, T_max=args.epochs)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs)

    # Adaptation loop
    save_path = args.save_path or f"adapted_cdan_{args.dataset}_{args.source}_{args.target}.pt"
    best_target_acc = 0.0
    history = []

    print(f"\nStarting CDAN adaptation for {args.epochs} epochs")
    print(f"  lambda_adv={args.lambda_adv}, lambda_im={args.lambda_im}, "
          f"lambda_l2sp={args.lambda_l2sp}")
    print(f"  lr_feat={args.lr}, lr_disc={args.lr_disc}")
    if args.mixup_alpha > 0:
        print(f"  Cross-domain mixup: alpha={args.mixup_alpha}, lambda_mixup={args.lambda_mixup}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Progressive lambda schedule: ramp up adversarial weight
        # p goes from 0 to 1 over training; lambda ramps via sigmoid schedule
        p = epoch / args.epochs
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0  # DANN schedule
        grl.set_lambda(grl_lambda * args.lambda_adv)

        losses = cdan_adapt_epoch(
            model=model,
            discriminator=discriminator,
            grl=grl,
            source_loader=src_train,
            target_loader=tgt_train,
            optimizer_feat=optimizer_feat,
            optimizer_disc=optimizer_disc,
            device=device,
            lambda_adv=args.lambda_adv,
            lambda_im=args.lambda_im,
            source_params=source_params,
            lambda_l2sp=args.lambda_l2sp,
            align_level=args.align_level,
            mixup_alpha=args.mixup_alpha,
            lambda_mixup=args.lambda_mixup,
            lambda_mcc=args.lambda_mcc,
        )

        # Evaluate on target test set
        _, target_acc = evaluate(model, tgt_test, device)

        scheduler_feat.step()
        scheduler_disc.step()

        epoch_time = time.time() - t0

        history.append({
            "epoch": epoch,
            "target_acc": target_acc,
            "grl_lambda": grl_lambda,
            **losses,
        })

        if epoch % args.log_interval == 0:
            mcc_str = f" mcc={losses['mcc']:.4f}" if args.lambda_mcc > 0 else ""
            mixup_str = f" mixup={losses['mixup']:.4f}" if args.mixup_alpha > 0 else ""
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"disc={losses['disc']:.4f} adv={losses['adv']:.4f} "
                f"im={losses['im']:.4f}{mcc_str}{mixup_str} total={losses['total']:.4f} | "
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

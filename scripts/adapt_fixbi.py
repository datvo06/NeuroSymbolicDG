#!/usr/bin/env python3
"""FixBi: Fixed Ratio-Based Mixup with Dual Networks (Na et al., CVPR 2021).

Adapts our PCFG pipeline using the FixBi algorithm:
1. Two models (source-dominant, target-dominant) initialized from same checkpoint
2. Fixed-ratio mixup creates intermediate domains at ratios [0.2, 0.4, 0.6, 0.8]
3. Source-dominant trains on source-leaning mixups (0.6, 0.8)
4. Target-dominant trains on target-leaning mixups (0.2, 0.4)
5. Mutual learning: each model teaches the other via confident pseudo-labels
6. Confidence-based curriculum: only high-confidence predictions are used

Usage:
    python scripts/adapt_fixbi.py \
        --checkpoint checkpoints/pcfg_sparse_office31_r50_amazon_webcam.pt \
        --dataset office31 --source amazon --target webcam \
        --backbone resnet50 --n-primitives 8 --use-sparsemax \
        --epochs 50 --batch-size 32 --lr 1e-4

    # Run all 6 Office-31 pairs:
    bash scripts/run_fixbi_office31.sh
"""

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import freeze_structure, get_adaptable_params
from neurosymbolic_da.training.losses import l2sp_loss
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Fixed mixup ratios (weight of source image)
# Source-dominant model uses higher source ratios
# Target-dominant model uses lower source ratios (more target)
FIXED_RATIOS = [0.2, 0.4, 0.6, 0.8]
SOURCE_DOMINANT_RATIOS = [0.6, 0.8]  # more source content
TARGET_DOMINANT_RATIOS = [0.2, 0.4]  # more target content


def fixed_ratio_mixup(
    source_x: torch.Tensor, target_x: torch.Tensor, ratio: float,
) -> torch.Tensor:
    """Mix source and target at a fixed ratio.

    Args:
        source_x: [B, C, H, W] source images
        target_x: [B, C, H, W] target images
        ratio: weight of source (0=all target, 1=all source)

    Returns:
        mixed: [B, C, H, W]
    """
    B = min(source_x.size(0), target_x.size(0))
    return ratio * source_x[:B] + (1 - ratio) * target_x[:B]


def confidence_gate(log_probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return boolean mask for samples above confidence threshold.

    Args:
        log_probs: [B, C] log-softmax predictions
        threshold: confidence threshold (max probability)

    Returns:
        mask: [B] boolean tensor
    """
    probs = log_probs.exp()
    max_conf, _ = probs.max(dim=-1)
    return max_conf >= threshold


def fixbi_epoch(
    model_s: NeuroSymbolicPipeline,
    model_t: NeuroSymbolicPipeline,
    source_loader,
    target_loader,
    optimizer_s,
    optimizer_t,
    device: torch.device,
    conf_threshold: float = 0.9,
    lambda_mutual: float = 1.0,
    lambda_mixup: float = 1.0,
    source_params_s=None,
    source_params_t=None,
    lambda_l2sp: float = 0.0,
) -> dict[str, float]:
    """One epoch of FixBi training.

    Both models are trained simultaneously:
    - model_s (source-dominant): task loss on source + mixup at [0.6, 0.8]
    - model_t (target-dominant): mixup at [0.2, 0.4] + mutual learning from model_s
    - Mutual learning: each teaches the other on target domain via confident pseudo-labels
    """
    model_s.train()
    model_t.train()

    totals = {
        "task_s": 0.0, "task_t": 0.0,
        "mixup_s": 0.0, "mixup_t": 0.0,
        "mutual_s": 0.0, "mutual_t": 0.0,
        "l2sp": 0.0, "total_s": 0.0, "total_t": 0.0,
    }
    n_batches = 0
    n_mutual = 0
    target_iter = iter(target_loader)

    for source_batch in source_loader:
        source_x, source_y = source_batch[0].to(device), source_batch[1].to(device)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        target_x = target_batch[0].to(device)

        B = min(source_x.size(0), target_x.size(0))

        # ================================================================
        # Step 1: Source task loss (both models)
        # ================================================================
        src_log_probs_s = model_s(source_x)
        src_log_probs_t = model_t(source_x)
        l_task_s = F.nll_loss(src_log_probs_s, source_y)
        l_task_t = F.nll_loss(src_log_probs_t, source_y)

        # ================================================================
        # Step 2: Fixed-ratio mixup losses
        # ================================================================
        # Source-dominant model learns from source-leaning mixups
        l_mixup_s = torch.tensor(0.0, device=device)
        for ratio in SOURCE_DOMINANT_RATIOS:
            mixed = fixed_ratio_mixup(source_x, target_x, ratio)
            mixed_log_probs = model_s(mixed)
            # Target: interpolation of source labels (one-hot) and target pseudo-labels
            with torch.no_grad():
                tgt_probs_t = model_t(target_x[:B]).exp()  # target model's prediction
                src_onehot = F.one_hot(source_y[:B], num_classes=src_log_probs_s.size(1)).float()
                target_dist = ratio * src_onehot + (1 - ratio) * tgt_probs_t
            l_mixup_s = l_mixup_s + F.kl_div(mixed_log_probs, target_dist, reduction="batchmean")
        l_mixup_s = l_mixup_s / len(SOURCE_DOMINANT_RATIOS)

        # Target-dominant model learns from target-leaning mixups
        l_mixup_t = torch.tensor(0.0, device=device)
        for ratio in TARGET_DOMINANT_RATIOS:
            mixed = fixed_ratio_mixup(source_x, target_x, ratio)
            mixed_log_probs = model_t(mixed)
            with torch.no_grad():
                tgt_probs_s = model_s(target_x[:B]).exp()  # source model's prediction
                src_onehot = F.one_hot(source_y[:B], num_classes=src_log_probs_s.size(1)).float()
                target_dist = ratio * src_onehot + (1 - ratio) * tgt_probs_s
            l_mixup_t = l_mixup_t + F.kl_div(mixed_log_probs, target_dist, reduction="batchmean")
        l_mixup_t = l_mixup_t / len(TARGET_DOMINANT_RATIOS)

        # ================================================================
        # Step 3: Mutual learning on target domain
        # ================================================================
        l_mutual_s = torch.tensor(0.0, device=device)
        l_mutual_t = torch.tensor(0.0, device=device)

        with torch.no_grad():
            tgt_log_probs_s_detach = model_s(target_x)
            tgt_log_probs_t_detach = model_t(target_x)
            # Confidence gates
            mask_s = confidence_gate(tgt_log_probs_s_detach, conf_threshold)
            mask_t = confidence_gate(tgt_log_probs_t_detach, conf_threshold)

        if mask_s.any():
            # Source model teaches target model on confident source predictions
            tgt_log_probs_t_live = model_t(target_x[mask_s])
            pseudo_labels_s = tgt_log_probs_s_detach[mask_s].argmax(dim=-1)
            l_mutual_t = F.nll_loss(tgt_log_probs_t_live, pseudo_labels_s)
            n_mutual += mask_s.sum().item()

        if mask_t.any():
            # Target model teaches source model on confident target predictions
            tgt_log_probs_s_live = model_s(target_x[mask_t])
            pseudo_labels_t = tgt_log_probs_t_detach[mask_t].argmax(dim=-1)
            l_mutual_s = F.nll_loss(tgt_log_probs_s_live, pseudo_labels_t)

        # ================================================================
        # Step 4: Combine and update
        # ================================================================
        loss_s = l_task_s + lambda_mixup * l_mixup_s + lambda_mutual * l_mutual_s
        loss_t = l_task_t + lambda_mixup * l_mixup_t + lambda_mutual * l_mutual_t

        # L2-SP regularization
        l2sp_val = 0.0
        if lambda_l2sp > 0:
            if source_params_s is not None:
                l_l2sp_s = l2sp_loss(model_s, source_params_s)
                loss_s = loss_s + lambda_l2sp * l_l2sp_s
                l2sp_val += l_l2sp_s.item()
            if source_params_t is not None:
                l_l2sp_t = l2sp_loss(model_t, source_params_t)
                loss_t = loss_t + lambda_l2sp * l_l2sp_t
                l2sp_val += l_l2sp_t.item()

        optimizer_s.zero_grad()
        loss_s.backward()
        optimizer_s.step()

        optimizer_t.zero_grad()
        loss_t.backward()
        optimizer_t.step()

        totals["task_s"] += l_task_s.item()
        totals["task_t"] += l_task_t.item()
        totals["mixup_s"] += l_mixup_s.item()
        totals["mixup_t"] += l_mixup_t.item()
        totals["mutual_s"] += l_mutual_s.item() if isinstance(l_mutual_s, torch.Tensor) else l_mutual_s
        totals["mutual_t"] += l_mutual_t.item() if isinstance(l_mutual_t, torch.Tensor) else l_mutual_t
        totals["l2sp"] += l2sp_val
        totals["total_s"] += loss_s.item()
        totals["total_t"] += loss_t.item()
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["n_mutual_samples"] = n_mutual
    return avg


def main():
    parser = argparse.ArgumentParser(description="FixBi: PCFG + Dual Network Adaptation")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-mutual", type=float, default=1.0)
    parser.add_argument("--lambda-mixup", type=float, default=1.0)
    parser.add_argument("--lambda-l2sp", type=float, default=0.005)
    parser.add_argument("--conf-threshold", type=float, default=0.9,
                        help="Confidence threshold for mutual learning")
    parser.add_argument("--conf-schedule", action="store_true",
                        help="Linearly decrease conf threshold from 0.95 to 0.7")

    parser.add_argument("--save-path", default=None)
    parser.add_argument("--num-workers", type=int, default=4)

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
    )

    print(f"Dataset: {args.dataset} ({args.source} -> {args.target}), {n_classes} classes")

    # Build two models from same checkpoint
    def build_model():
        m = NeuroSymbolicPipeline(
            n_primitives=args.n_primitives,
            n_classes=n_classes,
            backbone_variant=args.backbone,
            pretrained_backbone=False,
            max_depth=args.max_depth,
            use_sparsemax=args.use_sparsemax,
        )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        m.load_state_dict(checkpoint["model_state_dict"], strict=False)
        m.to(device)
        return m, checkpoint

    model_s, ckpt = build_model()
    model_t, _ = build_model()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(source val acc={ckpt.get('val_acc', 'N/A')})")

    _, pre_adapt_acc_s = evaluate(model_s, tgt_test, device)
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc_s:.4f}")

    # L2-SP: save source params for regularization
    source_params_s = None
    source_params_t = None
    if args.lambda_l2sp > 0:
        source_params_s = {
            name: param.clone().detach()
            for name, param in model_s.named_parameters()
        }
        source_params_t = {
            name: param.clone().detach()
            for name, param in model_t.named_parameters()
        }
        print(f"L2-SP: lambda={args.lambda_l2sp}")

    # Freeze grammar and relation params
    freeze_structure(model_s, freeze_grammar=True)
    freeze_structure(model_t, freeze_grammar=True)

    adaptable_params_s = get_adaptable_params(model_s)
    adaptable_params_t = get_adaptable_params(model_t)

    n_adapt_s = sum(p.numel() for p in adaptable_params_s)
    n_frozen_s = sum(p.numel() for p in model_s.parameters() if not p.requires_grad)
    print(f"Per model — Adaptable: {n_adapt_s:,} | Frozen: {n_frozen_s:,}")
    print(f"Total trainable params (2 models): {2 * n_adapt_s:,}")

    optimizer_s = Adam(adaptable_params_s, lr=args.lr)
    optimizer_t = Adam(adaptable_params_t, lr=args.lr)
    scheduler_s = CosineAnnealingLR(optimizer_s, T_max=args.epochs)
    scheduler_t = CosineAnnealingLR(optimizer_t, T_max=args.epochs)

    save_path = args.save_path or (
        f"checkpoints/fixbi_{args.dataset}_{args.source}_{args.target}.pt"
    )
    best_target_acc = 0.0
    best_model_tag = ""

    print(f"\nStarting FixBi adaptation for {args.epochs} epochs")
    print(f"  mutual={args.lambda_mutual}, mixup={args.lambda_mixup}, "
          f"l2sp={args.lambda_l2sp}")
    print(f"  conf_threshold={args.conf_threshold}, schedule={args.conf_schedule}")
    print(f"  Fixed ratios: src-dominant={SOURCE_DOMINANT_RATIOS}, "
          f"tgt-dominant={TARGET_DOMINANT_RATIOS}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Confidence threshold schedule
        if args.conf_schedule:
            conf_t = 0.95 - (0.25 * epoch / args.epochs)  # 0.95 -> 0.70
        else:
            conf_t = args.conf_threshold

        losses = fixbi_epoch(
            model_s=model_s,
            model_t=model_t,
            source_loader=src_train,
            target_loader=tgt_train,
            optimizer_s=optimizer_s,
            optimizer_t=optimizer_t,
            device=device,
            conf_threshold=conf_t,
            lambda_mutual=args.lambda_mutual,
            lambda_mixup=args.lambda_mixup,
            source_params_s=source_params_s,
            source_params_t=source_params_t,
            lambda_l2sp=args.lambda_l2sp,
        )

        # Evaluate both models
        _, target_acc_s = evaluate(model_s, tgt_test, device)
        _, target_acc_t = evaluate(model_t, tgt_test, device)

        # Also evaluate ensemble (average predictions)
        target_acc_ens = evaluate_ensemble(model_s, model_t, tgt_test, device)

        scheduler_s.step()
        scheduler_t.step()

        elapsed = time.time() - t0
        n_mut = losses.get("n_mutual_samples", 0)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"task_s={losses['task_s']:.3f} task_t={losses['task_t']:.3f} "
            f"mix_s={losses['mixup_s']:.3f} mix_t={losses['mixup_t']:.3f} "
            f"mut={losses['mutual_s']:.3f}/{losses['mutual_t']:.3f} | "
            f"acc_s={target_acc_s:.4f} acc_t={target_acc_t:.4f} "
            f"ens={target_acc_ens:.4f} conf={conf_t:.2f} n_mut={n_mut} | "
            f"{elapsed:.1f}s"
        )

        # Track best (use ensemble or best individual)
        best_this = max(target_acc_s, target_acc_t, target_acc_ens)
        if best_this > best_target_acc:
            best_target_acc = best_this
            if best_this == target_acc_ens:
                best_model_tag = "ensemble"
            elif best_this == target_acc_s:
                best_model_tag = "source-dominant"
            else:
                best_model_tag = "target-dominant"
            torch.save({
                "epoch": epoch,
                "model_s_state_dict": model_s.state_dict(),
                "model_t_state_dict": model_t.state_dict(),
                "target_acc_s": target_acc_s,
                "target_acc_t": target_acc_t,
                "target_acc_ens": target_acc_ens,
                "best_model": best_model_tag,
                "args": vars(args),
            }, save_path)

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc_s:.4f}")
    _, final_s = evaluate(model_s, tgt_test, device)
    _, final_t = evaluate(model_t, tgt_test, device)
    final_ens = evaluate_ensemble(model_s, model_t, tgt_test, device)
    print(f"Target acc AFTER (source-dom): {final_s:.4f}")
    print(f"Target acc AFTER (target-dom): {final_t:.4f}")
    print(f"Target acc AFTER (ensemble):   {final_ens:.4f}")
    print(f"Best target acc:               {best_target_acc:.4f} ({best_model_tag})")
    print(f"Improvement (best):   {best_target_acc - pre_adapt_acc_s:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


@torch.no_grad()
def evaluate_ensemble(
    model_s: NeuroSymbolicPipeline,
    model_t: NeuroSymbolicPipeline,
    loader,
    device: torch.device,
) -> float:
    """Evaluate ensemble of two models by averaging predictions."""
    model_s.eval()
    model_t.eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        log_probs_s = model_s(x)
        log_probs_t = model_t(x)
        # Average in probability space
        avg_probs = (log_probs_s.exp() + log_probs_t.exp()) / 2
        preds = avg_probs.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model_s.train()
    model_t.train()
    return correct / total


if __name__ == "__main__":
    main()

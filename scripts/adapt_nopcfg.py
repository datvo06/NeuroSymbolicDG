#!/usr/bin/env python3
"""NoPCFG target adaptation script.

Loads a NoPCFG source-trained checkpoint and adapts backbone + bottleneck
using MMD + entropy minimization (same protocol as PCFG adaptation,
but no grammar/relation params to freeze).

Usage:
    uv run python scripts/adapt_nopcfg.py \
        --checkpoint checkpoints/nopcfg_office31_r50_amazon_webcam.pt \
        --dataset office31 --source amazon --target webcam \
        --epochs 20 --lr 1e-4 --lambda-entropy 0.1
"""

import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline_nopcfg import NoPCFGPipeline
from neurosymbolic_da.training.losses import entropy_loss, im_loss, l2sp_loss, mmd_loss
from neurosymbolic_da.training.trainer import evaluate

import time


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def adapt_epoch_nopcfg(model, source_loader, target_loader, optimizer, device,
                       lambda_entropy=0.1, use_im_loss=False, use_bottleneck_mmd=False,
                       source_params=None, lambda_l2sp=0.0):
    model.train()
    total_mmd = 0.0
    total_ent = 0.0
    total_loss = 0.0
    n_batches = 0
    target_iter = iter(target_loader)

    for source_batch in source_loader:
        source_x = source_batch[0].to(device)
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        target_x = target_batch[0].to(device)

        optimizer.zero_grad()

        if use_bottleneck_mmd:
            source_feats = model.get_bottleneck_features(source_x)
            target_feats = model.get_bottleneck_features(target_x)
            l_mmd = mmd_loss(source_feats, target_feats)
        else:
            source_heatmaps = model.get_heatmaps(source_x)
            target_heatmaps = model.get_heatmaps(target_x)
            l_mmd = mmd_loss(source_heatmaps, target_heatmaps)

        target_log_probs = model(target_x)
        if use_im_loss:
            l_ent = im_loss(target_log_probs)
        else:
            l_ent = entropy_loss(target_log_probs)

        loss = l_mmd + lambda_entropy * l_ent

        if source_params is not None and lambda_l2sp > 0:
            l_l2sp = l2sp_loss(model, source_params)
            loss = loss + lambda_l2sp * l_l2sp

        loss.backward()
        optimizer.step()

        total_mmd += l_mmd.item()
        total_ent += l_ent.item()
        total_loss += loss.item()
        n_batches += 1

    return total_mmd / n_batches, total_ent / n_batches, total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="NoPCFG target adaptation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["digits", "office31", "officehome"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "lenet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-entropy", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--use-im-loss", action="store_true")
    parser.add_argument("--use-bottleneck-mmd", action="store_true")
    parser.add_argument("--lambda-l2sp", type=float, default=0.0)
    parser.add_argument("--adapt-mode", default="full",
                        choices=["full", "bn-only", "bottleneck-only"])

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
    print(f"Dataset: {args.dataset} ({args.source} → {args.target})")

    model = NoPCFGPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(source val acc={checkpoint.get('val_acc', 'N/A')})")

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

    # Freeze classifier (equivalent to freezing grammar in PCFG)
    for param in model.classifier.parameters():
        param.requires_grad = False

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

    adaptable_params = [p for p in model.parameters() if p.requires_grad]
    n_adapt = sum(p.numel() for p in adaptable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable params: {n_adapt:,} | Frozen params: {n_frozen:,}")

    optimizer = Adam(adaptable_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = args.save_path or f"adapted_nopcfg_{args.dataset}_{args.source}_{args.target}.pt"
    if args.use_im_loss:
        print("Using Information Maximization loss (entropy + diversity)")
    if args.use_bottleneck_mmd:
        print("Using bottleneck feature MMD (compact k*3 features)")

    best_target_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_mmd, avg_ent, avg_loss = adapt_epoch_nopcfg(
            model, src_train, tgt_train, optimizer, device,
            args.lambda_entropy, use_im_loss=args.use_im_loss,
            use_bottleneck_mmd=args.use_bottleneck_mmd,
            source_params=source_params, lambda_l2sp=args.lambda_l2sp,
        )
        _, target_acc = evaluate(model, tgt_test, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"MMD={avg_mmd:.4f} ent={avg_ent:.4f} loss={avg_loss:.4f} | "
                  f"target acc={target_acc:.4f} | time={elapsed:.1f}s")

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_acc": target_acc,
            }, save_path)

    # Reload best checkpoint for final eval
    best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, final_acc = evaluate(model, tgt_test, device)

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")
    print(f"Target acc AFTER  adaptation: {final_acc:.4f}")
    print(f"Improvement: {final_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

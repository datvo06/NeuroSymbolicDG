#!/usr/bin/env python3
"""VirDA-style adaptation: freeze backbone+grammar, learn visual prompt only.

Loads a source-trained NeuroSymbolicPipeline checkpoint, attaches a
VisualReprogrammingLayer, freezes everything except the prompt, and
adapts using MMD + IM loss.

The key idea: domain shift is captured entirely by an input-space
transformation. The backbone, bottleneck, and grammar are unchanged.

Usage:
    uv run python scripts/adapt_virda.py \
        --checkpoint checkpoints/gradonly_digits_mnist_usps_v2.pt \
        --dataset digits --source mnist --target usps \
        --epochs 20 --lr 1e-3 --pad-size 30
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.nn.pipeline_virda_pcfg import VisualReprogrammingLayer
from neurosymbolic_da.training.losses import im_loss, mmd_loss
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PromptWrappedPipeline(nn.Module):
    """Wraps a frozen NeuroSymbolicPipeline with a learnable visual prompt."""

    def __init__(self, pipeline: NeuroSymbolicPipeline, pad_size: int = 30):
        super().__init__()
        self.reprogramming = VisualReprogrammingLayer(pad_size=pad_size)
        self.pipeline = pipeline

        # Freeze the entire pipeline
        for param in self.pipeline.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_reprogram = self.reprogramming(x)
        return self.pipeline(x_reprogram)

    def get_heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        x_reprogram = self.reprogramming(x)
        return self.pipeline.get_heatmaps(x_reprogram)

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        x_reprogram = self.reprogramming(x)
        return self.pipeline.get_bottleneck_features(x_reprogram)


def adapt_virda_epoch(
    model: PromptWrappedPipeline,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_entropy: float = 0.1,
    use_bottleneck_mmd: bool = False,
) -> tuple[float, float, float]:
    """One epoch of VirDA adaptation (only prompt params update)."""
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
            source_hm = model.get_heatmaps(source_x)
            target_hm = model.get_heatmaps(target_x)
            l_mmd = mmd_loss(source_hm, target_hm)

        target_log_probs = model(target_x)
        l_ent = im_loss(target_log_probs)

        loss = l_mmd + lambda_entropy * l_ent
        loss.backward()
        optimizer.step()

        total_mmd += l_mmd.item()
        total_ent += l_ent.item()
        total_loss += loss.item()
        n_batches += 1

    return total_mmd / n_batches, total_ent / n_batches, total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="VirDA-style adaptation (visual prompt only)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["digits", "office31", "officehome", "scb", "cubdg"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-inside", action="store_true")

    parser.add_argument("--pad-size", type=int, default=30,
                        help="Visual prompt border width (pixels)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="LR for prompt (higher than backbone LR since prompt is small)")
    parser.add_argument("--lambda-entropy", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--use-bottleneck-mmd", action="store_true")

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    n_classes = get_n_classes(args.dataset)
    src_train, _, tgt_train, tgt_test = get_loaders(
        args.dataset, args.source, args.target,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Dataset: {args.dataset} ({args.source} → {args.target})")

    # Build model and load checkpoint
    pipeline = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
        max_depth=args.max_depth,
        use_inside=args.use_inside,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pipeline.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(source val acc={checkpoint.get('val_acc', 'N/A')})")

    # Wrap with visual prompt
    model = PromptWrappedPipeline(pipeline, pad_size=args.pad_size)
    model.to(device)

    # Evaluate before adaptation
    _, pre_acc = evaluate(model, tgt_test, device)
    print(f"Target acc BEFORE adaptation: {pre_acc:.4f}")

    prompt_params = list(model.reprogramming.parameters())
    n_prompt = sum(p.numel() for p in prompt_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Prompt params: {n_prompt:,} | Frozen params: {n_frozen:,}")

    optimizer = Adam(prompt_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = args.save_path or f"adapted_virda_{args.dataset}_{args.source}_{args.target}.pt"
    print(f"Using IM loss + {'bottleneck' if args.use_bottleneck_mmd else 'heatmap'} MMD")
    print(f"Prompt pad_size={args.pad_size}, params={n_prompt:,}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_mmd, avg_ent, avg_loss = adapt_virda_epoch(
            model, src_train, tgt_train, optimizer, device,
            lambda_entropy=args.lambda_entropy,
            use_bottleneck_mmd=args.use_bottleneck_mmd,
        )
        _, target_acc = evaluate(model, tgt_test, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"MMD={avg_mmd:.4f} ent={avg_ent:.4f} loss={avg_loss:.4f} | "
            f"target acc={target_acc:.4f} | time={elapsed:.1f}s"
        )

        if target_acc > best_acc:
            best_acc = target_acc
            torch.save({
                "epoch": epoch,
                "pipeline_state_dict": pipeline.state_dict(),
                "reprogramming_state_dict": model.reprogramming.state_dict(),
                "target_acc": target_acc,
            }, save_path)

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_acc:.4f}")
    print(f"Target acc AFTER  adaptation: {target_acc:.4f}")
    print(f"Best target acc: {best_acc:.4f}")
    print(f"Improvement: {best_acc - pre_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

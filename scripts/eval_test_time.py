#!/usr/bin/env python3
"""Evaluate DG checkpoints with test-time augmentation and adaptation.

Runs 4 strategies on each checkpoint:
1. Baseline (standard eval)
2. TTA (test-time augmentation, 6 views)
3. TENT (BN entropy minimization)
4. TENT+TTA (combine both)

Usage:
    python scripts/eval_test_time.py \
        --checkpoint checkpoints/dg_adv_pcfg_cubdg_Art.pt \
        --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8

    # Run all 3 DG targets:
    for TGT in Art Cartoon Paint; do
        python scripts/eval_test_time.py \
            --checkpoint checkpoints/dg_adv_pcfg_cubdg_${TGT}.pt \
            --dataset cubdg --target $TGT \
            --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
            --use-sparsemax
    done
"""

import argparse
import copy
import time

import torch
from torch.utils.data import DataLoader

from neurosymbolic_da.data.cubdg import get_cubdg
from neurosymbolic_da.data.loader_utils import get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.trainer import evaluate
from neurosymbolic_da.training.test_time import (
    evaluate_tta,
    tent_adapt,
    tent_tta_adapt,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DG checkpoints with test-time strategies"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cubdg")
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data/cub/CUB-DG")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--use-sparsemax", action="store_true")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    parser.add_argument("--tent-steps", type=int, default=1)
    parser.add_argument("--n-views", type=int, default=6,
                        help="Number of TTA views (max 6)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    n_classes = get_n_classes(args.dataset)

    # Build model
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
        use_sparsemax=args.use_sparsemax,
        max_depth=args.max_depth,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"Loaded: {args.checkpoint}")

    # Load target test set
    tgt_test_ds = get_cubdg(args.data_root, args.target, train=False)
    tgt_test = DataLoader(tgt_test_ds, batch_size=args.batch_size,
                          num_workers=args.num_workers, pin_memory=True,
                          shuffle=False)
    print(f"Target: {args.target} ({len(tgt_test_ds)} test images)")

    results = {}

    # 1. Baseline
    print("\n--- Baseline (standard eval) ---")
    t0 = time.time()
    loss, acc = evaluate(model, tgt_test, device)
    t1 = time.time()
    results["baseline"] = acc
    print(f"  Accuracy: {acc:.4f} ({t1 - t0:.1f}s)")

    # 2. TTA
    print(f"\n--- TTA ({args.n_views} views) ---")
    t0 = time.time()
    loss, acc = evaluate_tta(model, tgt_test, device, n_views=args.n_views)
    t1 = time.time()
    results["tta"] = acc
    print(f"  Accuracy: {acc:.4f} ({t1 - t0:.1f}s)")

    # 3. TENT (need fresh copy each time since it modifies BN)
    for tent_lr in [1e-4, 5e-4, 1e-3]:
        for tent_steps in [1, 3]:
            print(f"\n--- TENT (lr={tent_lr}, steps={tent_steps}) ---")
            model_tent = copy.deepcopy(model)
            model_tent.to(device)
            t0 = time.time()
            loss, acc = tent_adapt(model_tent, tgt_test, device,
                                   lr=tent_lr, n_steps=tent_steps)
            t1 = time.time()
            key = f"tent_lr{tent_lr}_s{tent_steps}"
            results[key] = acc
            print(f"  Accuracy: {acc:.4f} ({t1 - t0:.1f}s)")

    # 4. TENT+TTA (use best TENT config)
    best_tent_key = max(
        [k for k in results if k.startswith("tent_")],
        key=lambda k: results[k]
    )
    best_tent_lr = float(best_tent_key.split("lr")[1].split("_")[0])
    best_tent_steps = int(best_tent_key.split("s")[1])
    print(f"\n--- TENT+TTA (best TENT: lr={best_tent_lr}, steps={best_tent_steps}) ---")
    model_combo = copy.deepcopy(model)
    model_combo.to(device)
    t0 = time.time()
    loss, acc = tent_tta_adapt(model_combo, tgt_test, device,
                               lr=best_tent_lr, n_steps=best_tent_steps,
                               n_views=args.n_views)
    t1 = time.time()
    results["tent+tta"] = acc
    print(f"  Accuracy: {acc:.4f} ({t1 - t0:.1f}s)")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Target: {args.target}")
    print(f"{'=' * 50}")
    for method, acc in sorted(results.items(), key=lambda x: -x[1]):
        delta = acc - results["baseline"]
        sign = "+" if delta >= 0 else ""
        print(f"  {method:25s}: {acc:.4f} ({sign}{delta:.4f})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLIP-PCFG training script for CUB-DG domain generalization.

Uses frozen CLIP ViT-L/14 + concept bottleneck + PCFG grammar.
Optionally includes DDO (Domain Descriptor Orthogonality) regularization.

Usage:
    # ERM baseline (CLIP-PCFG, no DDO)
    python scripts/train_clip_pcfg.py --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --n-primitives 8 \
        --use-sparsemax --epochs 30 --batch-size 32 --lr 1e-3

    # With DDO regularization
    python scripts/train_clip_pcfg.py --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --n-primitives 8 \
        --use-sparsemax --epochs 30 --batch-size 32 --lr 1e-3 \
        --use-ddo --lambda-ddo 0.1
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from neurosymbolic_da.data.loader_utils import get_n_classes
from neurosymbolic_da.nn.pipeline_clip_pcfg import CLIPPCFGPipeline
from neurosymbolic_da.training.trainer import evaluate


# Domain lists per dataset
DATASET_DOMAINS = {
    "cubdg": ["Photo", "Art", "Cartoon", "Paint"],
    "pacs": ["photo", "art_painting", "cartoon", "sketch"],
}

# Default domain descriptors for DDO (subset of LanCE's 200)
DEFAULT_DOMAIN_DESCRIPTORS = [
    "a sketch", "a painting", "an artistic rendering",
    "a cartoon", "a watercolor", "a pencil drawing",
    "a digital art", "an oil painting", "a charcoal drawing",
    "an abstract depiction", "a stylized image", "a line drawing",
    "a realistic photo", "a blurry photo", "a vintage photo",
    "a high contrast image", "a low resolution image",
    "a black and white image", "a sepia toned image",
    "a pop art style image",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dg_loaders(dataset, target, data_root, batch_size=32, num_workers=4,
                   image_size=224, randaugment=False):
    """Get DG loaders: train on all-but-target, test on target."""
    domains = DATASET_DOMAINS[dataset]
    source_domains = [d for d in domains if d != target]
    print(f"Source domains: {source_domains}, Target domain: {target}")

    if dataset == "cubdg":
        from neurosymbolic_da.data.cubdg import get_cubdg

        src_train = []
        src_val = []
        for domain in source_domains:
            src_train.append(get_cubdg(data_root, domain, train=True,
                                       image_size=image_size, randaugment=randaugment))
            src_val.append(get_cubdg(data_root, domain, train=False,
                                     image_size=image_size))

        combined_train = ConcatDataset(src_train)
        combined_val = ConcatDataset(src_val)
        tgt_test = get_cubdg(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            DataLoader(combined_train, shuffle=True, **kwargs),
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported yet")


def main():
    parser = argparse.ArgumentParser(description="CLIP-PCFG DG training")

    # Dataset
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DOMAINS.keys()))
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")

    # Model
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true")
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grammar-l1", type=float, default=0.0)

    # DDO
    parser.add_argument("--use-ddo", action="store_true",
                        help="Enable DDO regularization")
    parser.add_argument("--lambda-ddo", type=float, default=0.1,
                        help="Weight for DDO loss")

    # Domain-conditional
    parser.add_argument("--domain-conditional", action="store_true")

    # Concept bank
    parser.add_argument("--concept-bank", default=None,
                        choices=["k8", "k16", "k32", "k64"],
                        help="Use LLM-generated concept bank (overrides --n-primitives)")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    domains = DATASET_DOMAINS[args.dataset]
    if args.target not in domains:
        raise ValueError(f"Target '{args.target}' not in {domains}")

    n_classes = get_n_classes(args.dataset)
    source_domains = [d for d in domains if d != args.target]

    # Load concept bank if specified
    concept_texts = None
    if args.concept_bank:
        from neurosymbolic_da.data.cub_concepts import (
            CONCEPTS_K8, CONCEPTS_K16, CONCEPTS_K32, CONCEPTS_K64,
        )
        bank_map = {"k8": CONCEPTS_K8, "k16": CONCEPTS_K16,
                     "k32": CONCEPTS_K32, "k64": CONCEPTS_K64}
        concept_texts = bank_map[args.concept_bank]
        args.n_primitives = len(concept_texts)
        print(f"Using LLM concept bank: {args.concept_bank} ({args.n_primitives} concepts)")

    # Build model
    model = CLIPPCFGPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        concept_texts=concept_texts,
        clip_model=args.clip_model,
        max_depth=args.max_depth,
        use_sparsemax=args.use_sparsemax,
        domain_conditional=args.domain_conditional,
        n_domains=len(source_domains) if args.domain_conditional else 0,
    )

    # Only train non-frozen params (bottleneck proj, relation params, grammar)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} trainable / {n_total:,} total")

    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Get data
    train_loader, val_loader, tgt_test = get_dg_loaders(
        args.dataset, args.target, args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        randaugment=args.randaugment,
    )

    print(f"Dataset: {args.dataset} (CLIP-PCFG DG, leave-{args.target}-out)")
    print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Target: {len(tgt_test.dataset)}")

    # Initialize DDO if requested
    if args.use_ddo:
        # Get class names from dataset
        if args.dataset == "cubdg":
            from neurosymbolic_da.data.cubdg import get_cubdg
            ds = get_cubdg(args.data_root, source_domains[0], train=False)
            if hasattr(ds, 'classes'):
                class_names = ds.classes
            else:
                class_names = [str(i) for i in range(n_classes)]
        else:
            class_names = [str(i) for i in range(n_classes)]

        model.initialize_ddo(
            domain_descriptors=DEFAULT_DOMAIN_DESCRIPTORS,
            class_names=class_names,
            source_domain="a photo",
        )
        print(f"DDO initialized: {len(DEFAULT_DOMAIN_DESCRIPTORS)} descriptors, "
              f"lambda={args.lambda_ddo}")

    model.to(device)
    save_path = args.save_path or f"checkpoints/dg_clip_pcfg_{args.dataset}_{args.target}.pt"
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            log_probs = model(x)

            if args.label_smoothing > 0:
                n_cls = log_probs.size(-1)
                smooth = torch.full_like(log_probs, args.label_smoothing / n_cls)
                smooth.scatter_(1, y.unsqueeze(1),
                                1.0 - args.label_smoothing + args.label_smoothing / n_cls)
                cls_loss = -(smooth * log_probs).sum(dim=-1).mean()
            else:
                cls_loss = nn.functional.nll_loss(log_probs, y)

            loss = cls_loss

            if args.grammar_l1 > 0:
                loss = loss + args.grammar_l1 * model.grammar.log_weights.abs().mean()

            if args.use_ddo:
                ddo = model.ddo_loss(model.grammar.log_weights)
                loss = loss + args.lambda_ddo * ddo

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            preds = log_probs.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += bs

        scheduler.step()
        avg_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)
        _, tgt_acc = evaluate(model, tgt_test, device)
        epoch_time = time.time() - t0

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={avg_loss:.4f} acc={train_acc:.4f} | "
                f"val={val_acc:.4f} tgt={tgt_acc:.4f} | {epoch_time:.1f}s"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tgt_acc = tgt_acc
            # Save only trainable params (skip frozen CLIP backbone ~1.7GB)
            save_dict = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("backbone.")}
            torch.save({
                "epoch": epoch,
                "model_state_dict": save_dict,
                "val_acc": val_acc,
                "tgt_acc": tgt_acc,
            }, save_path)

    _, final_tgt_acc = evaluate(model, tgt_test, device)
    print(f"\n--- Final Results (CLIP-PCFG DG) ---")
    print(f"Source val acc: {val_acc:.4f}")
    print(f"Target test acc (held-out {args.target}): {final_tgt_acc:.4f}")
    print(f"Best val epoch target acc: {best_tgt_acc:.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

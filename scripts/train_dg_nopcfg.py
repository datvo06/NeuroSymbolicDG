#!/usr/bin/env python3
"""NoPCFG Domain Generalization training — train on 3 domains, test on held-out target.

Ablation: same DG protocol as train_dg.py but with linear head instead of PCFG grammar.

Usage:
    python scripts/train_dg_nopcfg.py --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
        --epochs 50 --batch-size 32 --lr 1e-3
"""

import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_n_classes
from neurosymbolic_da.nn.pipeline_nopcfg import NoPCFGPipeline
from neurosymbolic_da.training.trainer import train, evaluate
from torch.utils.data import ConcatDataset, DataLoader


DATASET_DOMAINS = {
    "cubdg": ["Photo", "Art", "Cartoon", "Paint"],
    "pacs": ["photo", "art_painting", "cartoon", "sketch"],
}


def get_dg_loaders(dataset, target, data_root, batch_size=32, num_workers=4,
                   image_size=224, strong_aug=False):
    """Get loaders for DG protocol: train on all-but-target, test on target."""
    domains = DATASET_DOMAINS[dataset]
    source_domains = [d for d in domains if d != target]
    print(f"Source domains: {source_domains}, Target domain: {target}")
    if dataset == "cubdg":
        from neurosymbolic_da.data.cubdg import get_cubdg
        src_train_ds, src_val_ds = [], []
        for domain in source_domains:
            src_train_ds.append(get_cubdg(data_root, domain, train=True,
                                          image_size=image_size, strong_aug=strong_aug))
            src_val_ds.append(get_cubdg(data_root, domain, train=False,
                                        image_size=image_size))
        tgt_test = get_cubdg(data_root, target, train=False, image_size=image_size)
        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (DataLoader(ConcatDataset(src_train_ds), shuffle=True, **kwargs),
                DataLoader(ConcatDataset(src_val_ds), shuffle=False, **kwargs),
                DataLoader(tgt_test, shuffle=False, **kwargs))
    elif dataset == "pacs":
        from neurosymbolic_da.data.pacs import get_pacs
        src_train_ds, src_val_ds = [], []
        for domain in source_domains:
            src_train_ds.append(get_pacs(data_root, domain, train=True,
                                         image_size=image_size, strong_aug=strong_aug))
            src_val_ds.append(get_pacs(data_root, domain, train=False,
                                       image_size=image_size))
        tgt_test = get_pacs(data_root, target, train=False, image_size=image_size)
        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (DataLoader(ConcatDataset(src_train_ds), shuffle=True, **kwargs),
                DataLoader(ConcatDataset(src_val_ds), shuffle=False, **kwargs),
                DataLoader(tgt_test, shuffle=False, **kwargs))
    raise ValueError(f"DG not implemented for {dataset}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="NoPCFG DG training (leave-one-domain-out ablation)"
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_DOMAINS.keys())
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    src_train, src_val, tgt_test = get_dg_loaders(
        args.dataset, args.target,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong_aug=args.strong_aug,
    )

    print(f"Dataset: {args.dataset} NoPCFG (DG, leave-{args.target}-out)")
    print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
    print(f"Train samples: {len(src_train.dataset)}, Val samples: {len(src_val.dataset)}")

    model = NoPCFGPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=args.pretrained,
    )

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.bottleneck.parameters())
        + list(model.classifier.parameters())
    )
    backbone_lr_mult = 0.1 if args.pretrained else 1.0
    optimizer = Adam([
        {"params": backbone_params, "lr": args.lr * backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    save_path = (
        args.save_path
        or f"checkpoints/dg_nopcfg_{args.dataset}_{args.target}.pt"
    )

    metrics = train(
        model=model,
        train_loader=src_train,
        val_loader=src_val,
        optimizer=optimizer,
        device=device,
        n_epochs=args.epochs,
        scheduler=scheduler,
        log_interval=args.log_interval,
        save_path=save_path,
    )

    model.to(device)
    tgt_loss, tgt_acc = evaluate(model, tgt_test, device)
    print(f"\n--- Final Results (DG Protocol, NoPCFG) ---")
    print(f"Source val acc (3 domains): {metrics.val_acc:.4f}")
    print(f"Target test acc (held-out {args.target}): {tgt_acc:.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

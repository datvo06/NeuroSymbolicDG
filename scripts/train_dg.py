#!/usr/bin/env python3
"""Domain Generalization training script — train on 3 domains, test on held-out target.

Matches the leave-one-domain-out protocol used by CORAL, CITGM, etc. on CUB-DG.
No target data is used at all (neither labeled nor unlabeled).

Supports two modes:
1. ERM (default): Simple concatenation of 3 source domain train sets
2. Adversarial (--adversarial): 3-way domain alignment via DANN-style GRL
   to learn domain-invariant features across source domains

Usage:
    # ERM baseline
    python scripts/train_dg.py --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
        --use-sparsemax --grammar-l1 0.01 --epochs 50 --batch-size 32 --lr 1e-3

    # With 3-way adversarial alignment (like DANN in DG setting)
    python scripts/train_dg.py --dataset cubdg --target Art \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
        --use-sparsemax --grammar-l1 0.01 --epochs 50 --batch-size 32 --lr 1e-3 \
        --adversarial --lambda-adv 1.0 --lr-disc 1e-3
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
from neurosymbolic_da.training.trainer import train, evaluate


# Domain lists per dataset
DATASET_DOMAINS = {
    "cubdg": ["Photo", "Art", "Cartoon", "Paint"],
    "pacs": ["photo", "art_painting", "cartoon", "sketch"],
    "vlcs": ["CALTECH", "LABELME", "PASCAL", "SUN"],
    "terrainc": ["location_38", "location_43", "location_46", "location_100"],
    "domainnet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dg_loaders(
    dataset: str,
    target: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
):
    """Get loaders for DG protocol: train on all-but-target, test on target.

    Returns:
        (multi_source_train, multi_source_val, target_test)
    """
    domains = DATASET_DOMAINS[dataset]
    source_domains = [d for d in domains if d != target]
    print(f"Source domains: {source_domains}, Target domain: {target}")

    if dataset == "cubdg":
        from neurosymbolic_da.data.cubdg import get_cubdg

        # Combine train sets from all source domains
        src_train_datasets = []
        src_val_datasets = []
        for domain in source_domains:
            src_train_datasets.append(
                get_cubdg(data_root, domain, train=True, image_size=image_size,
                          strong_aug=strong_aug, randaugment=randaugment)
            )
            src_val_datasets.append(
                get_cubdg(data_root, domain, train=False, image_size=image_size)
            )

        combined_train = ConcatDataset(src_train_datasets)
        combined_val = ConcatDataset(src_val_datasets)

        # Target test set (never seen during training)
        tgt_test = get_cubdg(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            DataLoader(combined_train, shuffle=True, **kwargs),
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "pacs":
        from neurosymbolic_da.data.pacs import get_pacs

        src_train_datasets = []
        src_val_datasets = []
        for domain in source_domains:
            src_train_datasets.append(
                get_pacs(data_root, domain, train=True, image_size=image_size,
                         strong_aug=strong_aug, randaugment=randaugment)
            )
            src_val_datasets.append(
                get_pacs(data_root, domain, train=False, image_size=image_size)
            )

        combined_train = ConcatDataset(src_train_datasets)
        combined_val = ConcatDataset(src_val_datasets)

        tgt_test = get_pacs(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            DataLoader(combined_train, shuffle=True, **kwargs),
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )

    elif dataset == "vlcs":
        from neurosymbolic_da.data.vlcs import get_vlcs

        src_train_datasets = []
        src_val_datasets = []
        for domain in source_domains:
            src_train_datasets.append(
                get_vlcs(data_root, domain, train=True, image_size=image_size,
                         strong_aug=strong_aug, randaugment=randaugment)
            )
            src_val_datasets.append(
                get_vlcs(data_root, domain, train=False, image_size=image_size)
            )

        combined_train = ConcatDataset(src_train_datasets)
        combined_val = ConcatDataset(src_val_datasets)

        tgt_test = get_vlcs(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            DataLoader(combined_train, shuffle=True, **kwargs),
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "terrainc":
        from neurosymbolic_da.data.terrainc import get_terra

        src_train_datasets = []
        src_val_datasets = []
        for domain in source_domains:
            src_train_datasets.append(
                get_terra(data_root, domain, train=True, image_size=image_size,
                         strong_aug=strong_aug, randaugment=randaugment)
            )
            src_val_datasets.append(
                get_terra(data_root, domain, train=False, image_size=image_size)
            )

        combined_train = ConcatDataset(src_train_datasets)
        combined_val = ConcatDataset(src_val_datasets)

        tgt_test = get_terra(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            DataLoader(combined_train, shuffle=True, **kwargs),
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "domainnet":
        from neurosymbolic_da.data.domainnet import get_domainnet

        src_train_datasets = []
        src_val_datasets = []
        for domain in source_domains:
            src_train_datasets.append(
                get_domainnet(data_root, domain, train=True, image_size=image_size,
                         strong_aug=strong_aug, randaugment=randaugment)
            )
            src_val_datasets.append(
                get_domainnet(data_root, domain, train=False, image_size=image_size)
            )

        combined_train = ConcatDataset(src_train_datasets)
        combined_val = ConcatDataset(src_val_datasets)

        tgt_test = get_domainnet(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            DataLoader(combined_train, shuffle=True, **kwargs),
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    else:
        raise ValueError(f"DG protocol not implemented for dataset: {dataset}")


def get_dg_per_domain_loaders(
    dataset: str,
    target: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
):
    """Get separate loaders per source domain (for adversarial alignment).

    Returns:
        (domain_train_loaders: list[DataLoader], domain_names: list[str],
         combined_val: DataLoader, target_test: DataLoader)
    """
    domains = DATASET_DOMAINS[dataset]
    source_domains = [d for d in domains if d != target]

    if dataset == "cubdg":
        from neurosymbolic_da.data.cubdg import get_cubdg

        domain_train_loaders = []
        val_datasets = []
        for domain in source_domains:
            train_ds = get_cubdg(data_root, domain, train=True,
                                 image_size=image_size, strong_aug=strong_aug,
                                 randaugment=randaugment)
            val_ds = get_cubdg(data_root, domain, train=False,
                               image_size=image_size)
            kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)
            domain_train_loaders.append(
                DataLoader(train_ds, shuffle=True, **kwargs)
            )
            val_datasets.append(val_ds)

        combined_val = ConcatDataset(val_datasets)
        tgt_test = get_cubdg(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            domain_train_loaders,
            source_domains,
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "pacs":
        from neurosymbolic_da.data.pacs import get_pacs

        domain_train_loaders = []
        val_datasets = []
        for domain in source_domains:
            train_ds = get_pacs(data_root, domain, train=True,
                                image_size=image_size, strong_aug=strong_aug,
                                randaugment=randaugment)
            val_ds = get_pacs(data_root, domain, train=False,
                              image_size=image_size)
            kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)
            domain_train_loaders.append(
                DataLoader(train_ds, shuffle=True, **kwargs)
            )
            val_datasets.append(val_ds)

        combined_val = ConcatDataset(val_datasets)
        tgt_test = get_pacs(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            domain_train_loaders,
            source_domains,
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "vlcs":
        from neurosymbolic_da.data.vlcs import get_vlcs

        domain_train_loaders = []
        val_datasets = []
        for domain in source_domains:
            train_ds = get_vlcs(data_root, domain, train=True,
                                image_size=image_size, strong_aug=strong_aug,
                                randaugment=randaugment)
            val_ds = get_vlcs(data_root, domain, train=False,
                              image_size=image_size)
            kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)
            domain_train_loaders.append(
                DataLoader(train_ds, shuffle=True, **kwargs)
            )
            val_datasets.append(val_ds)

        combined_val = ConcatDataset(val_datasets)
        tgt_test = get_vlcs(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            domain_train_loaders,
            source_domains,
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "terrainc":
        from neurosymbolic_da.data.terrainc import get_terra

        domain_train_loaders = []
        val_datasets = []
        for domain in source_domains:
            train_ds = get_terra(data_root, domain, train=True,
                                image_size=image_size, strong_aug=strong_aug,
                                randaugment=randaugment)
            val_ds = get_terra(data_root, domain, train=False,
                              image_size=image_size)
            kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)
            domain_train_loaders.append(
                DataLoader(train_ds, shuffle=True, **kwargs)
            )
            val_datasets.append(val_ds)

        combined_val = ConcatDataset(val_datasets)
        tgt_test = get_terra(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            domain_train_loaders,
            source_domains,
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    elif dataset == "domainnet":
        from neurosymbolic_da.data.domainnet import get_domainnet

        domain_train_loaders = []
        val_datasets = []
        for domain in source_domains:
            train_ds = get_domainnet(data_root, domain, train=True,
                                image_size=image_size, strong_aug=strong_aug,
                                randaugment=randaugment)
            val_ds = get_domainnet(data_root, domain, train=False,
                              image_size=image_size)
            kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)
            domain_train_loaders.append(
                DataLoader(train_ds, shuffle=True, **kwargs)
            )
            val_datasets.append(val_ds)

        combined_val = ConcatDataset(val_datasets)
        tgt_test = get_domainnet(data_root, target, train=False, image_size=image_size)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        return (
            domain_train_loaders,
            source_domains,
            DataLoader(combined_val, shuffle=False, **kwargs),
            DataLoader(tgt_test, shuffle=False, **kwargs),
        )
    else:
        raise ValueError(f"DG protocol not implemented for dataset: {dataset}")


def dg_adversarial_epoch(
    model: NeuroSymbolicPipeline,
    discriminator: nn.Module,
    grl,
    domain_loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device,
    lambda_adv: float = 1.0,
    align_level: str = "backbone",
    grammar_l1: float = 0.0,
) -> tuple[float, float]:
    """One epoch of DG training with 3-way adversarial alignment.

    Returns:
        (avg_loss, train_acc)
    """
    model.train()
    discriminator.train()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    # Create iterators for each domain
    iters = [iter(loader) for loader in domain_loaders]
    n_domains = len(domain_loaders)

    # Iterate until shortest domain is exhausted
    exhausted = False
    while not exhausted:
        domain_xs = []
        domain_ys = []
        for i, it in enumerate(iters):
            try:
                batch = next(it)
            except StopIteration:
                exhausted = True
                break
            domain_xs.append(batch[0].to(device))
            domain_ys.append(batch[1].to(device))

        if exhausted:
            break

        # ---- Step 1: Update discriminator ----
        optimizer_disc.zero_grad()

        get_feats = (model.get_backbone_features if align_level == "backbone"
                     else model.get_bottleneck_features)

        with torch.no_grad():
            all_feats = []
            all_domain_labels = []
            for i, x in enumerate(domain_xs):
                feats = get_feats(x)
                all_feats.append(feats)
                all_domain_labels.append(
                    torch.full((feats.size(0),), i, dtype=torch.long, device=device)
                )

        disc_input = torch.cat(all_feats, dim=0)
        disc_labels = torch.cat(all_domain_labels, dim=0)
        disc_logits = discriminator(disc_input)
        l_disc = ce(disc_logits, disc_labels)
        l_disc.backward()
        optimizer_disc.step()

        # ---- Step 2: Update model (task + adversarial) ----
        optimizer.zero_grad()

        all_feats = []
        all_domain_labels = []
        l_task_total = 0.0
        batch_correct = 0
        batch_samples = 0

        for i, (x, y) in enumerate(zip(domain_xs, domain_ys)):
            log_probs = model(x)
            l_task = nn.functional.nll_loss(log_probs, y)
            l_task_total += l_task

            # Track accuracy
            preds = log_probs.argmax(dim=-1)
            batch_correct += (preds == y).sum().item()
            batch_samples += y.size(0)

            feats = get_feats(x)
            all_feats.append(feats)
            all_domain_labels.append(
                torch.full((feats.size(0),), i, dtype=torch.long, device=device)
            )

        # Average task loss across domains
        l_task_avg = l_task_total / n_domains

        # Adversarial loss: fool K-way domain discriminator
        feats_all = torch.cat(all_feats, dim=0)
        domain_labels_all = torch.cat(all_domain_labels, dim=0)
        feats_reversed = grl(feats_all)
        disc_logits = discriminator(feats_reversed)
        l_adv = ce(disc_logits, domain_labels_all)

        loss = l_task_avg + lambda_adv * l_adv

        # Grammar L1 sparsity
        if grammar_l1 > 0 and hasattr(model, 'grammar'):
            l1 = model.grammar.log_weights.abs().mean()
            loss = loss + grammar_l1 * l1

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += batch_correct
        total_samples += batch_samples
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(
        description="Domain Generalization training (leave-one-domain-out)"
    )
    # Dataset
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_DOMAINS.keys())
    )
    parser.add_argument("--target", required=True, help="Held-out target domain")
    parser.add_argument("--data-root", default="./data")

    # Model
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--use-sparsemax", action="store_true")
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--randaugment", action="store_true",
                        help="Use RandAugment (num_ops=2, magnitude=9)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.1 = typical)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-relations", action="store_true",
                        help="Use learned residual corrections on top of hand-coded relations")
    parser.add_argument("--learned-relations", action="store_true",
                        help="Use fully learned relation network (MLP) instead of hand-coded relations")
    parser.add_argument("--orthogonal-relations", action="store_true",
                        help="Use orthogonal learned relations (Cayley + L1 sparsity)")
    parser.add_argument("--hourglass", action="store_true",
                        help="Use multiscale FPN hourglass bottleneck (detects parts at multiple scales)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--grammar-lr-mult", type=float, default=1.0)
    parser.add_argument("--grammar-l1", type=float, default=0.0)

    # Adversarial alignment
    parser.add_argument("--adversarial", action="store_true",
                        help="Enable 3-way adversarial domain alignment (like DANN for DG)")
    parser.add_argument("--lambda-adv", type=float, default=1.0,
                        help="Weight for adversarial domain alignment loss")
    parser.add_argument("--lr-disc", type=float, default=1e-3,
                        help="Discriminator learning rate")
    parser.add_argument("--align-level", default="backbone",
                        choices=["bottleneck", "backbone"],
                        help="Feature level for domain discriminator")

    # Production score alignment (MMD between source domains)
    parser.add_argument("--align-productions", action="store_true",
                        help="Add MMD loss on production scores across source domains")
    parser.add_argument("--lambda-align", type=float, default=0.1,
                        help="Weight for production score alignment loss")

    # Domain-conditional grammar
    parser.add_argument("--domain-conditional", action="store_true",
                        help="Learn per-domain production weight offsets (shared base + domain-specific)")
    parser.add_argument("--lambda-domain-reg", type=float, default=0.01,
                        help="L2 regularization on domain offset parameters")

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Validate target domain
    domains = DATASET_DOMAINS[args.dataset]
    if args.target not in domains:
        raise ValueError(f"Target '{args.target}' not in {domains}")

    n_classes = get_n_classes(args.dataset)

    # Build model
    bottleneck_type = "hourglass" if getattr(args, 'hourglass', False) else "conv"
    source_domains = [d for d in domains if d != args.target]
    n_source_domains = len(source_domains)
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=args.pretrained,
        max_depth=args.max_depth,
        use_inside=False,
        use_sparsemax=args.use_sparsemax,
        bottleneck_type=bottleneck_type,
        residual_relations=getattr(args, 'residual_relations', False),
        learned_relations=getattr(args, 'learned_relations', False),
        orthogonal_relations=getattr(args, 'orthogonal_relations', False),
        domain_conditional=args.domain_conditional,
        n_domains=n_source_domains if args.domain_conditional else 0,
    )

    # Separate LRs
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.bottleneck.parameters())
        + list(model.relation_params.parameters())
    )
    grammar_params = list(model.grammar.parameters())
    backbone_lr_mult = 0.1 if args.pretrained else 1.0
    optimizer = Adam([
        {"params": backbone_params, "lr": args.lr * backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
        {"params": grammar_params, "lr": args.lr * args.grammar_lr_mult},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    if args.adversarial:
        # ---- Adversarial DG mode ----
        from neurosymbolic_da.training.adversarial import (
            MultiDomainDiscriminator,
            GradientReversalLayer,
        )

        domain_loaders, domain_names, val_loader, tgt_test = get_dg_per_domain_loaders(
            args.dataset, args.target,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong_aug=args.strong_aug,
            randaugment=args.randaugment,
        )

        n_domains = len(domain_loaders)
        print(f"Dataset: {args.dataset} (DG+adversarial, leave-{args.target}-out)")
        print(f"Source domains: {domain_names} ({n_domains}-way alignment)")
        print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")

        if args.align_level == "backbone":
            feat_dim = model.backbone.out_channels
        else:
            feat_dim = args.n_primitives * 3

        discriminator = MultiDomainDiscriminator(
            feat_dim, n_domains=n_domains, hidden_dim=1024
        ).to(device)
        grl = GradientReversalLayer(lambda_=args.lambda_adv)
        optimizer_disc = Adam(discriminator.parameters(), lr=args.lr_disc)
        scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs)

        n_disc = sum(p.numel() for p in discriminator.parameters())
        print(f"Domain discriminator: {n_disc:,} params ({n_domains}-way, {feat_dim}-dim)")

        model.to(device)
        save_path = (
            args.save_path
            or f"checkpoints/dg_adv_pcfg_{args.dataset}_{args.target}.pt"
        )
        best_val_acc = 0.0

        if args.grammar_l1 > 0:
            print(f"Grammar L1 sparsity: {args.grammar_l1}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # Progressive GRL lambda
            p = epoch / args.epochs
            grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            grl.set_lambda(grl_lambda * args.lambda_adv)

            avg_loss, train_acc = dg_adversarial_epoch(
                model=model,
                discriminator=discriminator,
                grl=grl,
                domain_loaders=domain_loaders,
                optimizer=optimizer,
                optimizer_disc=optimizer_disc,
                device=device,
                lambda_adv=args.lambda_adv,
                align_level=args.align_level,
                grammar_l1=args.grammar_l1,
            )

            val_loss, val_acc = evaluate(model, val_loader, device)
            _, tgt_acc = evaluate(model, tgt_test, device)

            scheduler.step()
            scheduler_disc.step()

            epoch_time = time.time() - t0

            if epoch % args.log_interval == 0:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"loss={avg_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_acc={val_acc:.4f} tgt_acc={tgt_acc:.4f} | "
                    f"grl_l={grl_lambda:.3f} | {epoch_time:.1f}s"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "tgt_acc": tgt_acc,
                }, save_path)

        # Final evaluation
        _, final_tgt_acc = evaluate(model, tgt_test, device)
        print(f"\n--- Final Results (DG+Adversarial Protocol) ---")
        print(f"Source val acc (3 domains): {val_acc:.4f}")
        print(f"Target test acc (held-out {args.target}): {final_tgt_acc:.4f}")
        print(f"Best val epoch target acc: {tgt_acc:.4f}")
        print(f"Checkpoint saved to: {save_path}")

    elif args.domain_conditional:
        # ---- Domain-Conditional Grammar mode ----
        domain_loaders, domain_names, val_loader, tgt_test = get_dg_per_domain_loaders(
            args.dataset, args.target,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong_aug=args.strong_aug,
            randaugment=args.randaugment,
        )

        n_domains = len(domain_loaders)
        print(f"Dataset: {args.dataset} (DG-DomainCond, leave-{args.target}-out)")
        print(f"Source domains: {domain_names} ({n_domains} domains)")
        print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
        print(f"Domain reg lambda: {args.lambda_domain_reg}")

        model.to(device)
        save_path = (
            args.save_path
            or f"checkpoints/dg_domcond_pcfg_{args.dataset}_{args.target}.pt"
        )
        best_val_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            domain_iters = [iter(dl) for dl in domain_loaders]
            max_batches = max(len(dl) for dl in domain_loaders)

            for _batch_idx in range(max_batches):
                domain_xs = []
                domain_ys = []
                domain_id_tensors = []
                for di, diter in enumerate(domain_iters):
                    try:
                        bx, by = next(diter)
                    except StopIteration:
                        domain_iters[di] = iter(domain_loaders[di])
                        bx, by = next(domain_iters[di])
                    domain_xs.append(bx.to(device))
                    domain_ys.append(by.to(device))
                    domain_id_tensors.append(
                        torch.full((bx.size(0),), di, dtype=torch.long, device=device)
                    )

                all_x = torch.cat(domain_xs, dim=0)
                all_y = torch.cat(domain_ys, dim=0)
                all_domain_ids = torch.cat(domain_id_tensors, dim=0)

                optimizer.zero_grad()

                log_probs = model(all_x, domain_ids=all_domain_ids)
                if args.label_smoothing > 0:
                    n_cls = log_probs.size(-1)
                    smooth_targets = torch.full_like(log_probs, args.label_smoothing / n_cls)
                    smooth_targets.scatter_(1, all_y.unsqueeze(1),
                                            1.0 - args.label_smoothing + args.label_smoothing / n_cls)
                    cls_loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
                else:
                    cls_loss = nn.functional.nll_loss(log_probs, all_y)

                loss = cls_loss

                if args.grammar_l1 > 0:
                    loss = loss + args.grammar_l1 * model.grammar.log_weights.abs().mean()

                # L2 regularization on domain offsets
                if args.lambda_domain_reg > 0:
                    domain_reg = model.grammar.domain_proj.weight.pow(2).mean()
                    loss = loss + args.lambda_domain_reg * domain_reg

                loss.backward()
                optimizer.step()

                bs = all_x.size(0)
                total_loss += loss.item() * bs
                preds = log_probs.argmax(dim=-1)
                correct += (preds == all_y).sum().item()
                total += bs

            scheduler.step()
            avg_loss = total_loss / total
            train_acc = correct / total

            # Eval without domain_ids (zero offset = base grammar)
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
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "tgt_acc": tgt_acc,
                    "domain_names": domain_names,
                }, save_path)

        _, final_tgt_acc = evaluate(model, tgt_test, device)
        print(f"\n--- Final Results (DG-DomainCond) ---")
        print(f"Source val acc: {val_acc:.4f}")
        print(f"Target test acc (held-out {args.target}): {final_tgt_acc:.4f}")
        print(f"Best val epoch target acc: {best_tgt_acc:.4f}")
        print(f"Checkpoint saved to: {save_path}")

    elif args.align_productions:
        # ---- ERM + Production Score Alignment mode ----
        from neurosymbolic_da.training.losses import mmd_loss

        domain_loaders, domain_names, val_loader, tgt_test = get_dg_per_domain_loaders(
            args.dataset, args.target,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong_aug=args.strong_aug,
            randaugment=args.randaugment,
        )

        n_domains = len(domain_loaders)
        print(f"Dataset: {args.dataset} (DG-ERM+ProdAlign, leave-{args.target}-out)")
        print(f"Source domains: {domain_names} ({n_domains}-way alignment)")
        print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
        print(f"Production alignment lambda: {args.lambda_align}")

        model.to(device)
        save_path = (
            args.save_path
            or f"checkpoints/dg_align_pcfg_{args.dataset}_{args.target}.pt"
        )
        best_val_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            total_cls_loss = 0.0
            total_align_loss = 0.0
            correct = 0
            total = 0

            # Zip domain loaders (cycle shorter ones)
            domain_iters = [iter(dl) for dl in domain_loaders]
            max_batches = max(len(dl) for dl in domain_loaders)

            for _batch_idx in range(max_batches):
                # Get one batch from each domain
                domain_xs = []
                domain_ys = []
                for di, diter in enumerate(domain_iters):
                    try:
                        bx, by = next(diter)
                    except StopIteration:
                        domain_iters[di] = iter(domain_loaders[di])
                        bx, by = next(domain_iters[di])
                    domain_xs.append(bx.to(device))
                    domain_ys.append(by.to(device))

                # Concatenate for classification loss
                all_x = torch.cat(domain_xs, dim=0)
                all_y = torch.cat(domain_ys, dim=0)

                optimizer.zero_grad()

                # Classification loss
                log_probs = model(all_x)
                if args.label_smoothing > 0:
                    n_cls = log_probs.size(-1)
                    smooth_targets = torch.full_like(log_probs, args.label_smoothing / n_cls)
                    smooth_targets.scatter_(1, all_y.unsqueeze(1),
                                            1.0 - args.label_smoothing + args.label_smoothing / n_cls)
                    cls_loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
                else:
                    cls_loss = nn.functional.nll_loss(log_probs, all_y)

                if args.grammar_l1 > 0:
                    cls_loss = cls_loss + args.grammar_l1 * model.grammar.log_weights.abs().mean()

                # Production score alignment: MMD between all domain pairs
                align_loss = torch.tensor(0.0, device=device)
                domain_prod_scores = []
                offset = 0
                for dx in domain_xs:
                    ps = model.get_production_scores(dx)
                    domain_prod_scores.append(ps)

                for i in range(n_domains):
                    for j in range(i + 1, n_domains):
                        align_loss = align_loss + mmd_loss(
                            domain_prod_scores[i], domain_prod_scores[j]
                        )

                loss = cls_loss + args.lambda_align * align_loss
                loss.backward()
                optimizer.step()

                bs = all_x.size(0)
                total_loss += loss.item() * bs
                total_cls_loss += cls_loss.item() * bs
                total_align_loss += align_loss.item() * bs
                preds = log_probs.argmax(dim=-1)
                correct += (preds == all_y).sum().item()
                total += bs

            scheduler.step()
            avg_loss = total_loss / total
            avg_cls = total_cls_loss / total
            avg_align = total_align_loss / total
            train_acc = correct / total

            val_loss, val_acc = evaluate(model, val_loader, device)
            _, tgt_acc = evaluate(model, tgt_test, device)
            epoch_time = time.time() - t0

            if epoch % args.log_interval == 0:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"cls={avg_cls:.4f} align={avg_align:.4f} acc={train_acc:.4f} | "
                    f"val={val_acc:.4f} tgt={tgt_acc:.4f} | {epoch_time:.1f}s"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_tgt_acc = tgt_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "tgt_acc": tgt_acc,
                }, save_path)

        _, final_tgt_acc = evaluate(model, tgt_test, device)
        print(f"\n--- Final Results (DG-ERM+ProdAlign) ---")
        print(f"Source val acc: {val_acc:.4f}")
        print(f"Target test acc (held-out {args.target}): {final_tgt_acc:.4f}")
        print(f"Best val epoch target acc: {best_tgt_acc:.4f}")
        print(f"Checkpoint saved to: {save_path}")

    else:
        # ---- ERM mode (simple concatenation) ----
        src_train, src_val, tgt_test = get_dg_loaders(
            args.dataset, args.target,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strong_aug=args.strong_aug,
            randaugment=args.randaugment,
        )

        print(f"Dataset: {args.dataset} (DG-ERM, leave-{args.target}-out)")
        print(f"Classes: {n_classes}, Primitives: {args.n_primitives}")
        print(f"Train samples: {len(src_train.dataset)}, Val samples: {len(src_val.dataset)}")
        print(f"Target test samples: {len(tgt_test.dataset)}")

        save_path = (
            args.save_path
            or f"checkpoints/dg_pcfg_{args.dataset}_{args.target}.pt"
        )
        if args.grammar_l1 > 0:
            print(f"Grammar L1 sparsity: {args.grammar_l1}")

        if args.label_smoothing > 0:
            print(f"Label smoothing: {args.label_smoothing}")
        if args.randaugment:
            print(f"RandAugment: num_ops=2, magnitude=9")

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
            grammar_l1=args.grammar_l1,
            label_smoothing=args.label_smoothing,
        )

        # Final evaluation on held-out target (zero-shot)
        model.to(device)
        tgt_loss, tgt_acc = evaluate(model, tgt_test, device)
        print(f"\n--- Final Results (DG-ERM Protocol) ---")
        print(f"Source val acc (3 domains): {metrics.val_acc:.4f}")
        print(f"Target test acc (held-out {args.target}): {tgt_acc:.4f}")
        print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

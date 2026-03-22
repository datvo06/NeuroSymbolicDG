#!/usr/bin/env python3
"""NRC adaptation script (Neighborhood Reciprocal Clustering).

Source-free domain adaptation via neighborhood structure exploitation
(Yang et al., NeurIPS 2021). No source data needed during adaptation.

Algorithm:
  1. Compute target features and build k-NN graph in feature space
  2. Extended neighborhood loss: prediction consistency with k-NN
  3. Compact neighborhood loss: stronger consistency with reciprocal k-NN
  4. Information Maximization loss: confident + diverse predictions

Usage:
    uv run python scripts/adapt_nrc.py \
        --checkpoint checkpoint_office31_amazon_webcam.pt \
        --dataset office31 --source amazon --target webcam \
        --epochs 15 --lr 1e-3
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from neurosymbolic_da.data.loader_utils import get_loaders, get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import freeze_structure, get_adaptable_params
from neurosymbolic_da.training.losses import im_loss
from neurosymbolic_da.training.trainer import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def compute_features_and_predictions(
    model: NeuroSymbolicPipeline,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Compute backbone features and softmax predictions for all samples.

    Returns:
        features: [N, feat_dim] L2-normalized backbone features
        predictions: [N, C] softmax predictions
        indices: sample indices for mapping back to dataset
    """
    model.eval()
    all_feats = []
    all_preds = []

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        feats = model.get_backbone_features(batch_x)  # [B, feat_dim]
        log_probs = model(batch_x)  # [B, C]
        all_feats.append(feats.cpu())
        all_preds.append(log_probs.exp().cpu())  # softmax probs

    features = torch.cat(all_feats, dim=0)
    predictions = torch.cat(all_preds, dim=0)

    # L2 normalize features for cosine similarity
    features = F.normalize(features, dim=1)

    return features, predictions


@torch.no_grad()
def build_knn_graph(
    features: torch.Tensor,
    k: int = 5,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build k-NN graph from features.

    Args:
        features: [N, D] L2-normalized features
        k: number of nearest neighbors
        temperature: softmax temperature for affinity weights

    Returns:
        nn_indices: [N, k] indices of k nearest neighbors
        nn_weights: [N, k] softmax-normalized affinity weights
        reciprocal_mask: [N, k] bool mask for reciprocal nearest neighbors
    """
    N = features.shape[0]

    # Cosine similarity (features are L2-normalized)
    sim = features @ features.T  # [N, N]

    # Zero out self-similarity
    sim.fill_diagonal_(-1.0)

    # Find k nearest neighbors
    nn_weights_raw, nn_indices = sim.topk(k, dim=1)  # [N, k]

    # Softmax-normalized affinity weights
    nn_weights = F.softmax(nn_weights_raw / temperature, dim=1)  # [N, k]

    # Build reciprocal nearest neighbor mask
    # j is reciprocal NN of i if: j in kNN(i) AND i in kNN(j)
    # Create a set-membership matrix: is_nn[i, j] = True if j in kNN(i)
    is_nn = torch.zeros(N, N, dtype=torch.bool)
    row_idx = torch.arange(N).unsqueeze(1).expand(-1, k)
    is_nn[row_idx, nn_indices] = True

    # reciprocal_mask[i, m] = True if nn_indices[i, m] is also a reciprocal neighbor
    reciprocal_mask = torch.zeros(N, k, dtype=torch.bool)
    for m in range(k):
        neighbor_idx = nn_indices[:, m]  # [N]
        # Check if i is in kNN of neighbor_idx[i]
        reciprocal_mask[:, m] = is_nn[neighbor_idx, torch.arange(N)]

    return nn_indices, nn_weights, reciprocal_mask


def nrc_adapt_epoch(
    model: NeuroSymbolicPipeline,
    target_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    nn_indices: torch.Tensor,
    nn_weights: torch.Tensor,
    reciprocal_mask: torch.Tensor,
    all_predictions: torch.Tensor,
    lambda_im: float = 1.0,
    lambda_cn: float = 1.0,
) -> dict[str, float]:
    """Run one NRC adaptation epoch.

    Args:
        model: pipeline (grammar/relation_params frozen)
        target_loader: target data loader
        optimizer: optimizer for feature extractor
        device: torch device
        nn_indices: [N, k] precomputed neighbor indices
        nn_weights: [N, k] precomputed affinity weights
        reciprocal_mask: [N, k] reciprocal neighbor mask
        all_predictions: [N, C] cached predictions from previous epoch
        lambda_im: weight for IM loss
        lambda_cn: weight for compact neighborhood loss

    Returns:
        dict of average losses
    """
    model.train()

    totals = {"en": 0.0, "cn": 0.0, "im": 0.0, "total": 0.0}
    n_batches = 0
    sample_idx = 0

    for batch_x, _ in target_loader:
        batch_x = batch_x.to(device)
        B = batch_x.size(0)
        batch_indices = list(range(sample_idx, sample_idx + B))
        sample_idx += B

        optimizer.zero_grad()

        # Forward pass
        log_probs = model(batch_x)  # [B, C]
        probs = log_probs.exp()  # [B, C]

        # --- Extended Neighborhood Loss ---
        # For each sample in batch, get predictions of its neighbors
        batch_nn_idx = nn_indices[batch_indices]  # [B, k]
        batch_nn_weights = nn_weights[batch_indices].to(device)  # [B, k]
        neighbor_preds = all_predictions[batch_nn_idx].to(device)  # [B, k, C]

        # Prediction agreement: p_i . p_j (dot product of probability vectors)
        # probs: [B, C] -> [B, 1, C], neighbor_preds: [B, k, C]
        agreement = (probs.unsqueeze(1) * neighbor_preds).sum(dim=2)  # [B, k]
        agreement = agreement.clamp(min=1e-8)

        # Weighted log-agreement
        l_en = -(batch_nn_weights * agreement.log()).sum(dim=1).mean()

        # --- Compact Neighborhood Loss ---
        batch_reciprocal = reciprocal_mask[batch_indices].to(device)  # [B, k]
        if batch_reciprocal.any():
            # Only compute for reciprocal neighbors
            recip_agreement = agreement * batch_reciprocal.float()  # zero out non-reciprocal
            recip_count = batch_reciprocal.float().sum(dim=1).clamp(min=1)  # [B]
            l_cn = -(recip_agreement.clamp(min=1e-8).log() * batch_reciprocal.float()).sum(dim=1)
            l_cn = (l_cn / recip_count).mean()
        else:
            l_cn = torch.tensor(0.0, device=device)

        # --- IM Loss ---
        l_im = im_loss(log_probs)

        loss = l_en + lambda_cn * l_cn + lambda_im * l_im
        loss.backward()
        optimizer.step()

        totals["en"] += l_en.item()
        totals["cn"] += l_cn.item()
        totals["im"] += l_im.item()
        totals["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="NRC adaptation (source-free)")

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
    parser.add_argument("--use-sparsemax", action="store_true")
    parser.add_argument("--invariant-coords", action="store_true")
    parser.add_argument("--bottleneck-type", default="conv", choices=["conv", "slot", "moe"])

    # NRC hyperparameters
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature for affinity weights")
    parser.add_argument("--lambda-im", type=float, default=1.0,
                        help="Weight for IM loss")
    parser.add_argument("--lambda-cn", type=float, default=1.0,
                        help="Weight for compact neighborhood loss")

    # Backbone LR
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                        help="LR multiplier for backbone (lower to preserve features)")

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
    _, _, tgt_train, tgt_test = get_loaders(
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
        pretrained_backbone=False,
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

    # Freeze grammar + relation params
    freeze_structure(model, freeze_grammar=True)

    # Set up optimizer with different LR for backbone vs head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    bottleneck_params = [p for p in model.bottleneck.parameters() if p.requires_grad]

    n_adapt = sum(p.numel() for p in backbone_params + bottleneck_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Adaptable params: {n_adapt:,} | Frozen params: {n_frozen:,}")

    optimizer = SGD([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": bottleneck_params, "lr": args.lr},
    ], momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Adaptation loop
    save_path = args.save_path or f"adapted_nrc_{args.dataset}_{args.source}_{args.target}.pt"
    best_target_acc = 0.0

    print(f"\nStarting NRC adaptation for {args.epochs} epochs")
    print(f"  k={args.k}, temperature={args.temperature}")
    print(f"  lambda_im={args.lambda_im}, lambda_cn={args.lambda_cn}")
    print(f"  lr={args.lr}, backbone_lr={args.lr * args.backbone_lr_mult}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Step 1: Compute features and build k-NN graph
        features, predictions = compute_features_and_predictions(
            model, tgt_train, device
        )
        nn_indices, nn_weights, reciprocal_mask = build_knn_graph(
            features, k=args.k, temperature=args.temperature
        )

        t_graph = time.time() - t0

        # Step 2: NRC adaptation epoch
        losses = nrc_adapt_epoch(
            model=model,
            target_loader=tgt_train,
            optimizer=optimizer,
            device=device,
            nn_indices=nn_indices,
            nn_weights=nn_weights,
            reciprocal_mask=reciprocal_mask,
            all_predictions=predictions,
            lambda_im=args.lambda_im,
            lambda_cn=args.lambda_cn,
        )

        # Step 3: Evaluate
        _, target_acc = evaluate(model, tgt_test, device)
        scheduler.step()

        epoch_time = time.time() - t0

        if epoch % args.log_interval == 0:
            n_recip = reciprocal_mask.float().sum().item()
            n_total = reciprocal_mask.numel()
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"en={losses['en']:.4f} cn={losses['cn']:.4f} "
                f"im={losses['im']:.4f} total={losses['total']:.4f} | "
                f"target_acc={target_acc:.4f} recip={n_recip/n_total:.1%} | "
                f"{epoch_time:.1f}s (graph={t_graph:.1f}s)"
            )

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            for _attempt in range(3):
                try:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "target_acc": target_acc,
                        "args": vars(args),
                    }, save_path)
                    break
                except RuntimeError:
                    import time as _t
                    _t.sleep(0.5)

    print(f"\n--- Final Results ---")
    print(f"Target acc BEFORE adaptation: {pre_adapt_acc:.4f}")
    print(f"Target acc AFTER  adaptation: {target_acc:.4f}")
    print(f"Best target acc:              {best_target_acc:.4f}")
    print(f"Improvement (final):  {target_acc - pre_adapt_acc:+.4f}")
    print(f"Improvement (best):   {best_target_acc - pre_adapt_acc:+.4f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

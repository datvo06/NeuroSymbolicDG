"""Adaptation losses: MMD and entropy minimization (Section 4.3).

L_adapt = L_MMD(h_s, h_t) + lambda * L_entropy(x_t)
"""

import torch
from torch import Tensor


def gaussian_kernel(x: Tensor, y: Tensor, bandwidth: float = 1.0) -> Tensor:
    """Compute Gaussian (RBF) kernel between all pairs of rows in x and y.

    Args:
        x: [N, D]
        y: [M, D]
        bandwidth: kernel bandwidth

    Returns:
        kernel matrix [N, M]
    """
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [M, 1]
    dist_sq = x_sq + y_sq.T - 2 * x @ y.T     # [N, M]
    return torch.exp(-dist_sq / (2 * bandwidth ** 2))


def mmd_loss(
    source_heatmaps: Tensor,
    target_heatmaps: Tensor,
    bandwidths: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> Tensor:
    """Maximum Mean Discrepancy between source and target heatmaps.

    Uses a multi-kernel MMD (sum of Gaussian kernels at different bandwidths)
    for robustness.

    Args:
        source_heatmaps: [B_s, k, H, W] source heatmaps
        target_heatmaps: [B_t, k, H, W] target heatmaps
        bandwidths: Gaussian kernel bandwidths

    Returns:
        scalar MMD^2 loss
    """
    # Flatten spatial dims: [B, k*H*W]
    s = source_heatmaps.flatten(start_dim=1)
    t = target_heatmaps.flatten(start_dim=1)

    loss = torch.tensor(0.0, device=s.device, dtype=s.dtype)

    for bw in bandwidths:
        k_ss = gaussian_kernel(s, s, bw)
        k_tt = gaussian_kernel(t, t, bw)
        k_st = gaussian_kernel(s, t, bw)

        n_s = s.size(0)
        n_t = t.size(0)

        # Unbiased MMD^2 estimator
        loss = loss + (
            k_ss.sum() / (n_s * n_s)
            + k_tt.sum() / (n_t * n_t)
            - 2 * k_st.sum() / (n_s * n_t)
        )

    return loss


def entropy_loss(log_probs: Tensor) -> Tensor:
    """Entropy minimization on target predictions.

    Encourages confident (low-entropy) predictions on unlabeled target data.

    Args:
        log_probs: [B, C] log-softmax output

    Returns:
        mean entropy (scalar)
    """
    probs = log_probs.exp()
    # H = -sum(p * log(p))
    return -(probs * log_probs).sum(dim=-1).mean()


def l2sp_loss(model: "torch.nn.Module", source_params: dict[str, Tensor]) -> Tensor:
    """L2-SP regularization (Li et al., 2018).

    Penalizes deviation of current parameters from source-trained values:
        L_L2SP = sum_i ||theta_i - theta_i^source||^2

    Args:
        model: current model being adapted
        source_params: dict of {name: tensor} from source checkpoint

    Returns:
        scalar L2-SP penalty
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in source_params:
            penalty = penalty + ((param - source_params[name]) ** 2).sum()
    return penalty


def heatmap_diversity_loss(heatmaps: Tensor) -> Tensor:
    """Diversity loss: penalize heatmaps that attend to the same spatial location.

    Computes pairwise cosine similarity between flattened heatmaps and
    penalizes high similarity. Forces different primitives to detect
    different parts of the object.

    Args:
        heatmaps: [B, k, H, W] raw heatmaps from concept bottleneck

    Returns:
        scalar diversity loss (lower = more diverse primitives)
    """
    B, k, H, W = heatmaps.shape
    # Normalize heatmaps spatially with softmax to get attention maps
    flat = heatmaps.view(B, k, -1)  # [B, k, H*W]
    attn = torch.softmax(flat, dim=-1)  # [B, k, H*W]

    # Pairwise cosine similarity between all primitive pairs
    # attn is already L1-normalized (softmax), so normalize to unit vectors
    attn_norm = attn / (attn.norm(dim=-1, keepdim=True) + 1e-8)  # [B, k, H*W]
    sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # [B, k, k]

    # Penalize off-diagonal similarities (exclude self-similarity)
    mask = ~torch.eye(k, dtype=torch.bool, device=heatmaps.device).unsqueeze(0)
    loss = sim[mask.expand(B, -1, -1)].mean()
    return loss


def heatmap_concentration_loss(heatmaps: Tensor) -> Tensor:
    """Concentration loss: penalize spatially diffuse heatmaps.

    Each heatmap should be peaked at one location (low spatial entropy),
    like a real part detector. High entropy = spread everywhere = not a part.

    Args:
        heatmaps: [B, k, H, W] raw heatmaps from concept bottleneck

    Returns:
        scalar concentration loss (mean spatial entropy, lower = more peaked)
    """
    B, k, H, W = heatmaps.shape
    flat = heatmaps.view(B, k, -1)  # [B, k, H*W]
    attn = torch.softmax(flat, dim=-1)  # [B, k, H*W]

    # Spatial entropy: H = -sum(p * log(p))
    entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1)  # [B, k]
    return entropy.mean()


def heatmap_orthogonality_loss(heatmaps: Tensor) -> Tensor:
    """Orthogonality loss: force heatmap attention maps to be orthogonal.

    Stronger than cosine diversity — directly penalizes ||A^T A - I||_F^2
    where A is the matrix of flattened attention maps. This forces each
    primitive to attend to a non-overlapping spatial region.

    Args:
        heatmaps: [B, k, H, W] raw heatmaps

    Returns:
        scalar orthogonality loss
    """
    B, k, H, W = heatmaps.shape
    flat = heatmaps.view(B, k, -1)  # [B, k, H*W]
    attn = torch.softmax(flat, dim=-1)  # [B, k, H*W]

    # Gram matrix: [B, k, k]
    gram = torch.bmm(attn, attn.transpose(1, 2))
    # Target: identity (orthogonal attention maps)
    eye = torch.eye(k, device=heatmaps.device, dtype=heatmaps.dtype).unsqueeze(0)
    # Frobenius norm of (Gram - I), focusing on off-diagonal
    loss = ((gram - eye) ** 2).mean()
    return loss


def bottleneck_reg_loss(
    heatmaps: Tensor,
    diversity_weight: float = 1.0,
    concentration_weight: float = 0.1,
) -> Tensor:
    """Combined bottleneck regularization loss.

    Forces the concept bottleneck to learn meaningful part decompositions:
    - Diversity: different primitives attend to different spatial locations
    - Concentration: each primitive's heatmap is spatially peaked

    Args:
        heatmaps: [B, k, H, W] raw heatmaps
        diversity_weight: weight for diversity loss
        concentration_weight: weight for concentration loss

    Returns:
        scalar combined bottleneck regularization loss
    """
    l_div = heatmap_diversity_loss(heatmaps)
    l_conc = heatmap_concentration_loss(heatmaps)
    return diversity_weight * l_div + concentration_weight * l_conc


def bottleneck_reg_loss_v2(
    heatmaps: Tensor,
    orthogonality_weight: float = 1.0,
    concentration_weight: float = 0.5,
) -> Tensor:
    """Stronger bottleneck regularization (v2).

    Uses orthogonality loss instead of cosine diversity (stronger gradient
    signal against collapse) and higher concentration weight.

    Args:
        heatmaps: [B, k, H, W] raw heatmaps
        orthogonality_weight: weight for orthogonality loss
        concentration_weight: weight for concentration loss

    Returns:
        scalar combined bottleneck regularization loss
    """
    l_orth = heatmap_orthogonality_loss(heatmaps)
    l_conc = heatmap_concentration_loss(heatmaps)
    return orthogonality_weight * l_orth + concentration_weight * l_conc


def compute_centroids(
    model: "torch.nn.Module",
    loader: "torch.utils.data.DataLoader",
    device: torch.device,
    n_classes: int,
) -> Tensor:
    """Compute class centroids from labeled data using bottleneck features.

    Args:
        model: NeuroSymbolicPipeline (must have get_bottleneck_features)
        loader: labeled data loader yielding (images, labels)
        device: torch device
        n_classes: number of classes

    Returns:
        centroids: [n_classes, feat_dim] mean feature per class
    """
    model.eval()
    feat_sum: list[Tensor | None] = [None] * n_classes
    counts = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            feats = model.get_bottleneck_features(images)  # [B, k*3]

            for c in range(n_classes):
                mask = labels == c
                if mask.any():
                    c_feats = feats[mask]  # [n_c, k*3]
                    if feat_sum[c] is None:
                        feat_sum[c] = c_feats.sum(dim=0)
                    else:
                        feat_sum[c] = feat_sum[c] + c_feats.sum(dim=0)
                    counts[c] += mask.sum()

    # Build centroids tensor
    feat_dim = feats.shape[1]
    centroids = torch.zeros(n_classes, feat_dim, device=device)
    for c in range(n_classes):
        if counts[c] > 0 and feat_sum[c] is not None:
            centroids[c] = feat_sum[c] / counts[c]

    return centroids


def assign_pseudo_labels(
    features: Tensor,
    centroids: Tensor,
    threshold: float = 0.5,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Assign pseudo-labels to samples by nearest centroid distance.

    Uses softmax over negative distances (with temperature) as confidence.

    Args:
        features: [B, D] bottleneck features
        centroids: [C, D] class centroids
        threshold: confidence threshold; only samples above this are used
        temperature: softmax temperature (lower = sharper, default 0.1)

    Returns:
        pseudo_labels: [B] predicted class indices
        confidence_mask: [B] bool tensor — True for high-confidence samples
    """
    # Compute distances: [B, C]
    dists = torch.cdist(features.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    # Convert distances to confidence via softmax over negative distances
    # Temperature scaling makes distribution sharper so confidence is meaningful
    logits = -dists / temperature
    probs = torch.softmax(logits, dim=-1)  # [B, C]
    confidence, pseudo_labels = probs.max(dim=-1)  # [B], [B]
    confidence_mask = confidence >= threshold
    return pseudo_labels, confidence_mask


def mcc_loss(log_probs: Tensor) -> Tensor:
    """Minimum Class Confusion loss (Jin et al., ECCV 2020).

    Minimizes pairwise class confusion on target predictions. Confident
    samples are weighted more heavily via entropy-based reweighting.

    Args:
        log_probs: [B, C] log-softmax output

    Returns:
        scalar MCC loss (lower = less class confusion)
    """
    import math

    probs = log_probs.exp()  # [B, C]
    K = probs.shape[1]

    # Entropy-based sample weights: confident samples weighted more
    entropy = -(probs * log_probs).sum(dim=1)  # [B]
    max_entropy = math.log(K)
    weights = 1.0 - entropy / max_entropy  # [B], higher for confident
    weights = weights / (weights.sum() + 1e-8)  # normalize to sum to 1

    # Weighted confusion matrix: C[j,k] = sum_i w_i * p_i[j] * p_i[k]
    weighted_probs = probs * weights.unsqueeze(1)  # [B, K]
    confusion = probs.T @ weighted_probs  # [K, K]

    # Minimize off-diagonal (class confusion), normalized by K
    loss = (confusion.sum() - confusion.trace()) / K
    return loss


def im_loss(log_probs: Tensor) -> Tensor:
    """Information Maximization loss (Liang et al., 2020 — SHOT).

    Minimizes per-sample entropy (confident predictions) while maximizing
    marginal entropy (class diversity / balance). This prevents the
    trivial collapse where all samples are assigned to a single class.

        L_IM = E_x[H(p(y|x))] - H(E_x[p(y|x)])
             = (per-sample entropy) - (marginal entropy)

    Args:
        log_probs: [B, C] log-softmax output

    Returns:
        scalar IM loss (lower is better)
    """
    probs = log_probs.exp()  # [B, C]

    # Per-sample entropy: E_x[H(p(y|x))]
    per_sample_ent = -(probs * log_probs).sum(dim=-1).mean()

    # Marginal entropy: H(E_x[p(y|x)])
    mean_probs = probs.mean(dim=0)  # [C]
    marginal_ent = -(mean_probs * (mean_probs + 1e-8).log()).sum()

    # Minimize per-sample entropy, maximize marginal entropy
    return per_sample_ent - marginal_ent

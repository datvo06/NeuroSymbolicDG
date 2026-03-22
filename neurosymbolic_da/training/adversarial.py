"""Adversarial domain adaptation components (CDAN, Long et al. NeurIPS 2018).

Provides:
- GradientReversalLayer: scales gradients by -lambda during backward pass
- DomainDiscriminator: MLP classifier (source vs target)
- cdan_condition: concatenates features with softmax predictions
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function


class _GradientReversal(Function):
    """Gradient reversal autograd function."""

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Reverses gradients during backward pass, scaled by lambda.

    During forward: identity.
    During backward: gradients are multiplied by -lambda.

    Args:
        lambda_: scaling factor for gradient reversal (default 1.0)
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return _GradientReversal.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Update the reversal scaling factor."""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """MLP domain discriminator for adversarial adaptation.

    Architecture: in_dim -> 1024 -> 1024 -> 1 with ReLU + dropout.
    Outputs raw logit (use BCEWithLogitsLoss).

    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension (default 1024)
        dropout: dropout probability (default 0.5)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [B, in_dim] conditioned features

        Returns:
            logits: [B, 1] domain prediction logits
        """
        return self.net(x)


def entropy_weight(log_probs: Tensor) -> Tensor:
    """Compute per-sample entropy weights for CDAN+E.

    High-confidence (low-entropy) samples get higher weight.
    Returns weights in [0, 1] normalized by log(n_classes).

    Args:
        log_probs: [B, C] log-softmax predictions

    Returns:
        weights: [B] per-sample weights
    """
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B]
    max_entropy = torch.log(torch.tensor(log_probs.shape[1], dtype=torch.float32,
                                          device=log_probs.device))
    # Invert: low entropy → high weight
    weights = 1.0 - entropy / max_entropy
    return weights


def domain_mixup(
    source_x: Tensor, target_x: Tensor, alpha: float = 0.3,
) -> tuple[Tensor, Tensor, float]:
    """Mix source and target images with random ratio.

    Creates intermediate-domain samples. Inspired by FixBi.

    Args:
        source_x: [B_s, C, H, W] source images
        target_x: [B_t, C, H, W] target images
        alpha: Beta distribution parameter (lower = more extreme ratios)

    Returns:
        mixed: [B, C, H, W] mixed images
        lam: mixing coefficient (weight of source)
    """
    B = min(source_x.size(0), target_x.size(0))
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    mixed = lam * source_x[:B] + (1 - lam) * target_x[:B]
    return mixed, lam


class MultiDomainDiscriminator(nn.Module):
    """K-way domain discriminator for multi-source DG alignment.

    Classifies which of K source domains an example belongs to.
    Used with GRL so the feature extractor learns domain-invariant features.

    Args:
        in_dim: input feature dimension
        n_domains: number of source domains (K)
        hidden_dim: hidden layer dimension
        dropout: dropout probability
    """

    def __init__(self, in_dim: int, n_domains: int, hidden_dim: int = 1024,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [B, in_dim] features

        Returns:
            logits: [B, n_domains] domain prediction logits
        """
        return self.net(x)


def cdan_condition(features: Tensor, log_probs: Tensor) -> Tensor:
    """Condition features on classifier predictions via concatenation.

    Simple conditioning: concatenate bottleneck features with softmax
    predictions. Suitable when feature_dim + n_classes is small.

    Args:
        features: [B, d_f] bottleneck features
        log_probs: [B, C] log-softmax class predictions

    Returns:
        conditioned: [B, d_f + C] concatenated features
    """
    probs = log_probs.exp()  # [B, C] softmax probabilities
    return torch.cat([features, probs], dim=-1)  # [B, d_f + C]

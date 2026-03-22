"""Ablation: VirDA visual reprogramming + PCFG (Experiment 4f, Section 6.6).

Combines VirDA's visual reprogramming layer (learnable input perturbation
on a frozen backbone) with our concept bottleneck and PCFG scoring head.

VirDA (Nguyen et al., 2025) freezes the backbone and learns a visual
reprogramming layer that transforms inputs. Here we feed the reprogrammed
features through our bottleneck + grammar pipeline.
"""

import torch
import torch.nn as nn
from effectful.ops.semantics import handler
from torch import Tensor

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from neurosymbolic_da.dsl.handlers.inside import get_class_score, make_inside_handler
from neurosymbolic_da.dsl.relations import RelationParams
from neurosymbolic_da.nn.backbone import ResNetBackbone
from neurosymbolic_da.nn.bottleneck import ConceptBottleneck


class VisualReprogrammingLayer(nn.Module):
    """Learnable input-space perturbation (VirDA-style).

    Adds a learnable perturbation pattern to input images.
    The perturbation is applied within a border mask, leaving
    a center window for the original content.

    Args:
        image_size: input image size (assumes square)
        pad_size: width of the reprogramming border
    """

    def __init__(self, image_size: int = 224, pad_size: int = 30):
        super().__init__()
        self.image_size = image_size
        self.pad_size = pad_size

        # Learnable perturbation pattern
        self.perturbation = nn.Parameter(
            torch.randn(1, 3, image_size, image_size) * 0.02
        )

        # Create border mask (1 in border, 0 in center)
        mask = torch.ones(1, 1, image_size, image_size)
        mask[:, :, pad_size:-pad_size, pad_size:-pad_size] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply visual reprogramming to input images.

        Args:
            x: [B, 3, H, W]

        Returns:
            reprogrammed images [B, 3, H, W]
        """
        return x + self.perturbation * self.mask


class VirDAPCFGPipeline(nn.Module):
    """VirDA reprogramming + frozen backbone + concept bottleneck + PCFG.

    The backbone is frozen (as in VirDA). The visual reprogramming layer
    and concept bottleneck are trainable. The PCFG scores over the
    bottleneck's detected primitives.

    Args:
        n_primitives: number of primitive types (k)
        n_classes: number of output classes
        backbone_variant: "resnet18" or "resnet50"
        pretrained_backbone: use ImageNet pretrained weights
        max_depth: max grammar derivation depth
        use_inside: if True, use inside algorithm
        pad_size: VirDA reprogramming border width
    """

    def __init__(
        self,
        n_primitives: int,
        n_classes: int,
        backbone_variant: str = "resnet18",
        pretrained_backbone: bool = True,
        max_depth: int = 2,
        use_inside: bool = False,
        pad_size: int = 30,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes
        self.use_inside = use_inside

        # VirDA: learnable reprogramming + frozen backbone
        self.reprogramming = VisualReprogrammingLayer(pad_size=pad_size)
        self.backbone = ResNetBackbone(backbone_variant, pretrained_backbone)

        # Freeze backbone (VirDA-style)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Our components: bottleneck + grammar
        self.bottleneck = ConceptBottleneck(self.backbone.out_channels, n_primitives)
        self.relation_params = RelationParams()
        self.grammar = LayoutGrammar(n_primitives, n_classes, max_depth)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class scores for a batch of images.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            log_probs: log-softmax class scores [B, n_classes]
        """
        # VirDA reprogramming
        x_reprogram = self.reprogramming(x)

        # Frozen backbone
        features = self.backbone(x_reprogram)  # [B, C, H, W]
        heatmaps, coords, _ = self.bottleneck(features)

        B = x.shape[0]
        conf = torch.sigmoid(
            heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values
        )
        env = self.bottleneck.extract_batched_env(coords, conf, heatmaps)

        if self.use_inside:
            class_scores = []
            for c in range(self.n_classes):
                with handler(make_inside_handler(env, self.relation_params)):
                    table = self.grammar(c)
                s = get_class_score(table, self.n_primitives)
                class_scores.append(s)
            scores = torch.stack(class_scores, dim=-1)
        else:
            scores = self.grammar.forward_vectorized(env, self.relation_params)

        return torch.log_softmax(scores, dim=-1)

    def get_heatmaps(self, x: Tensor) -> Tensor:
        """Extract heatmaps for MMD alignment."""
        x_reprogram = self.reprogramming(x)
        features = self.backbone(x_reprogram)
        heatmaps, _, _ = self.bottleneck(features)
        return heatmaps

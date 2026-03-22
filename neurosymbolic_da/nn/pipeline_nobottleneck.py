"""Ablation: Backbone features directly to PCFG (no concept bottleneck).

Uses global average pooling + learned projection to create pseudo-primitive
features, bypassing the spatial heatmap bottleneck. Used in Experiment 4
(Section 6.6) to isolate the contribution of the concept bottleneck.
"""

import torch
import torch.nn as nn
from effectful.ops.semantics import handler
from torch import Tensor

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from neurosymbolic_da.dsl.handlers.inside import get_class_score, make_inside_handler
from neurosymbolic_da.dsl.primitives import Env, Primitive
from neurosymbolic_da.dsl.relations import RelationParams
from neurosymbolic_da.nn.backbone import ResNetBackbone


class NoBottleneckPipeline(nn.Module):
    """Backbone features projected to pseudo-primitives, then PCFG scoring.

    Instead of a concept bottleneck with spatial heatmaps, this variant
    projects global-pooled backbone features into pseudo-primitive coordinates
    and confidences via a learned linear layer.

    Args:
        n_primitives: number of pseudo-primitive types (k)
        n_classes: number of output classes
        backbone_variant: "resnet18" or "resnet50"
        pretrained_backbone: use ImageNet pretrained weights
        max_depth: max grammar derivation depth
        use_inside: if True, use inside algorithm; else direct eval
    """

    def __init__(
        self,
        n_primitives: int,
        n_classes: int,
        backbone_variant: str = "resnet18",
        pretrained_backbone: bool = True,
        max_depth: int = 2,
        use_inside: bool = False,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes
        self.use_inside = use_inside

        self.backbone = ResNetBackbone(backbone_variant, pretrained_backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Project pooled features to pseudo-primitive parameters:
        # per primitive: cx, cy, x1, y1, x2, y2 (6 coords) + conf (1) = 7
        self.proj = nn.Linear(self.backbone.out_channels, n_primitives * 7)

        self.relation_params = RelationParams()
        self.grammar = LayoutGrammar(n_primitives, n_classes, max_depth)

    def _extract_batched_env(self, prim_params: Tensor) -> Env:
        """Build batched Env from projected primitive parameters [B, k*7]."""
        env: Env = {}
        k = self.n_primitives
        params = prim_params.view(-1, k, 7)  # [B, k, 7]

        for j in range(k):
            p = params[:, j]  # [B, 7]
            env[j] = Primitive(
                cx=torch.tanh(p[:, 0]),
                cy=torch.tanh(p[:, 1]),
                x1=torch.tanh(p[:, 2]),
                y1=torch.tanh(p[:, 3]),
                x2=torch.tanh(p[:, 4]),
                y2=torch.tanh(p[:, 5]),
                conf=torch.sigmoid(p[:, 6]),
                type_idx=j,
            )
        return env

    def forward(self, x: Tensor) -> Tensor:
        """Compute class scores for a batch of images.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            log_probs: log-softmax class scores [B, n_classes]
        """
        features = self.backbone(x)  # [B, C, H, W]
        pooled = self.pool(features).squeeze(-1).squeeze(-1)  # [B, C]
        prim_params = self.proj(pooled)  # [B, k*7]

        env = self._extract_batched_env(prim_params)

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
        """Return pooled features as pseudo-heatmaps for MMD alignment.

        Since there are no real heatmaps, returns the backbone features
        after global average pooling, reshaped to [B, k, 1, 1].
        """
        features = self.backbone(x)
        pooled = self.pool(features).squeeze(-1).squeeze(-1)  # [B, C]
        prim_params = self.proj(pooled)  # [B, k*7]
        # Return the projected params reshaped as pseudo-heatmaps
        return prim_params.view(x.shape[0], self.n_primitives, 7, 1)

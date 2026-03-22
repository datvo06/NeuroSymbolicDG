"""Full neuro-symbolic pipeline (Section 3.1).

Input image → Backbone → Concept Bottleneck → PCFG Scoring Head → class scores.
"""

import torch
import torch.nn as nn
from effectful.ops.semantics import handler
from torch import Tensor

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from neurosymbolic_da.dsl.handlers.inside import make_inside_handler, get_class_score
from neurosymbolic_da.dsl.relations import LearnedRelationParams, OrthogonalLearnedRelationParams, RelationParams, ResidualRelationParams
from neurosymbolic_da.nn.backbone import LeNetBackbone, ResNetBackbone
from neurosymbolic_da.nn.bottleneck import ConceptBottleneck, MoEBottleneck
from neurosymbolic_da.nn.hourglass_bottleneck import HourglassBottleneck
from neurosymbolic_da.nn.slot_bottleneck import SlotAttentionBottleneck


class NeuroSymbolicPipeline(nn.Module):
    """End-to-end neuro-symbolic classification pipeline.

    Args:
        n_primitives: number of primitive types (k)
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
        use_sparsemax: bool = False,
        input_conditional: bool = False,
        bottleneck_type: str = "conv",
        slot_iters: int = 3,
        invariant_coords: bool = False,
        residual_relations: bool = False,
        learned_relations: bool = False,
        orthogonal_relations: bool = False,
        domain_conditional: bool = False,
        n_domains: int = 0,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes
        self.use_inside = use_inside
        self.input_conditional = input_conditional

        self._multiscale = (bottleneck_type == "hourglass")

        if backbone_variant == "lenet":
            self.backbone = LeNetBackbone()
        else:
            self.backbone = ResNetBackbone(
                backbone_variant, pretrained_backbone,
                multiscale=self._multiscale,
            )

        if bottleneck_type == "hourglass":
            self.bottleneck = HourglassBottleneck(
                self.backbone.layer_channels, n_primitives,
            )
        elif bottleneck_type == "slot":
            self.bottleneck = SlotAttentionBottleneck(
                self.backbone.out_channels, n_primitives,
                slot_dim=64, n_iters=slot_iters,
            )
        elif bottleneck_type == "moe":
            self.bottleneck = MoEBottleneck(self.backbone.out_channels, n_primitives)
        else:
            self.bottleneck = ConceptBottleneck(self.backbone.out_channels, n_primitives)
        if orthogonal_relations:
            self.relation_params = OrthogonalLearnedRelationParams()
        elif learned_relations:
            self.relation_params = LearnedRelationParams()
        elif residual_relations:
            self.relation_params = ResidualRelationParams()
        else:
            self.relation_params = RelationParams()
        feature_dim = n_primitives * 3  # k*3: coords + conf
        self.grammar = LayoutGrammar(n_primitives, n_classes, max_depth,
                                     use_sparsemax=use_sparsemax,
                                     input_conditional=input_conditional,
                                     feature_dim=feature_dim,
                                     invariant_coords=invariant_coords,
                                     domain_conditional=domain_conditional,
                                     n_domains=n_domains)
        self.score_temperature = 0.1  # sharpen predictions: lower = more confident

    def forward(self, x: Tensor, domain_ids: Tensor | None = None) -> Tensor:
        """Compute class scores for a batch of images.

        Uses batched env to process all B images simultaneously per class,
        eliminating the per-image loop.

        Args:
            x: input images [B, 3, H, W]
            domain_ids: optional [B] int tensor of domain indices for domain-conditional grammar

        Returns:
            log_probs: log-softmax class scores [B, n_classes]
        """
        features = self.backbone(x)  # [B, C, H, W]
        heatmaps, coords, _ = self.bottleneck(features)

        B = x.shape[0]
        # Batched confidence: [B, k]
        conf = torch.sigmoid(
            heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values
        )
        # Batched env: each Primitive field is [B]
        env = self.bottleneck.extract_batched_env(coords, conf, heatmaps)

        # Compute bottleneck features for input-conditional grammar
        bn_features = None
        if self.input_conditional:
            flat_coords = coords.view(B, -1)  # [B, k*2]
            bn_features = torch.cat([flat_coords, conf], dim=-1)  # [B, k*3]

        if self.use_inside:
            # Inside algorithm: one handler call per class
            class_scores = []
            for c in range(self.n_classes):
                with handler(make_inside_handler(env, self.relation_params)):
                    table = self.grammar(c)
                s = get_class_score(table, self.n_primitives)
                class_scores.append(s)
            scores = torch.stack(class_scores, dim=-1)  # [B, n_classes]
        else:
            # Vectorized eval: single matmul, no effectful overhead
            scores = self.grammar.forward_vectorized(env, self.relation_params,
                                                     features=bn_features,
                                                     domain_ids=domain_ids)

        return torch.log_softmax(scores / self.score_temperature, dim=-1)

    def forward_with_heatmaps(self, x: Tensor, domain_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute class scores and return heatmaps (for bottleneck regularization).

        Args:
            x: input images [B, 3, H, W]

        Returns:
            log_probs: [B, n_classes]
            heatmaps: [B, k, H, W]
        """
        features = self.backbone(x)  # [B, C, H, W]
        heatmaps, coords, _ = self.bottleneck(features)

        B = x.shape[0]
        conf = torch.sigmoid(
            heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values
        )
        env = self.bottleneck.extract_batched_env(coords, conf, heatmaps)

        bn_features = None
        if self.input_conditional:
            flat_coords = coords.view(B, -1)
            bn_features = torch.cat([flat_coords, conf], dim=-1)

        if self.use_inside:
            class_scores = []
            for c in range(self.n_classes):
                with handler(make_inside_handler(env, self.relation_params)):
                    table = self.grammar(c)
                s = get_class_score(table, self.n_primitives)
                class_scores.append(s)
            scores = torch.stack(class_scores, dim=-1)
        else:
            scores = self.grammar.forward_vectorized(env, self.relation_params,
                                                     features=bn_features,
                                                     domain_ids=domain_ids)

        log_probs = torch.log_softmax(scores / self.score_temperature, dim=-1)
        return log_probs, heatmaps

    def get_heatmaps(self, x: Tensor) -> Tensor:
        """Extract heatmaps only (for MMD alignment during adaptation).

        Args:
            x: input images [B, 3, H, W]

        Returns:
            heatmaps: [B, k, H, W]
        """
        features = self.backbone(x)
        heatmaps, _, _ = self.bottleneck(features)
        return heatmaps

    def get_production_scores(self, x: Tensor) -> Tensor:
        """Get raw production scores for distribution alignment.

        Returns [B, n_productions] — the primitive and relation activations
        before grammar weighting. Used for MMD alignment across domains in DG.
        """
        features = self.backbone(x)
        heatmaps, coords, _ = self.bottleneck(features)
        B = x.shape[0]
        conf = torch.sigmoid(
            heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values
        )
        env = self.bottleneck.extract_batched_env(coords, conf, heatmaps)
        return self.grammar.get_production_scores(env, self.relation_params)

    def _get_layer4_features(self, x: Tensor) -> Tensor:
        """Get layer4 features regardless of multiscale mode."""
        features = self.backbone(x)
        if self._multiscale:
            return features["layer4"]
        return features

    def get_backbone_features(self, x: Tensor) -> Tensor:
        """Extract GAP-pooled backbone features for domain alignment.

        Returns high-dimensional features (2048 for ResNet-50, 512 for ResNet-18)
        that are much richer than the 24-dim bottleneck features.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            features: [B, backbone_dim] (2048 for R50, 512 for R18)
        """
        feat_map = self._get_layer4_features(x)  # [B, C, H, W]
        return feat_map.mean(dim=[2, 3])  # Global Average Pooling → [B, C]

    def get_bottleneck_features(self, x: Tensor) -> Tensor:
        """Extract compact bottleneck features (for MMD alignment).

        Returns concatenated [coords, conf] — much lower-dimensional than
        raw heatmaps, reducing risk of feature collapse during adaptation.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            features: [B, k*3] — (cx, cy, conf) per primitive
        """
        features = self.backbone(x)
        heatmaps, coords, _ = self.bottleneck(features)
        B = x.shape[0]
        conf = torch.sigmoid(
            heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values
        )  # [B, k]
        # coords: [B, k, 2] → [B, k*2]
        flat_coords = coords.view(B, -1)
        return torch.cat([flat_coords, conf], dim=-1)  # [B, k*3]

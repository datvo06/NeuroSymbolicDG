"""CLIP-PCFG pipeline for LanCE+PCFG integration.

Frozen CLIP ViT-L/14 → Concept Bottleneck → PCFG Grammar → class scores.
Optionally includes DDO (Domain Descriptor Orthogonality) regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import clip
except ImportError:
    clip = None

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from neurosymbolic_da.dsl.relations import RelationParams
from neurosymbolic_da.nn.clip_backbone import CLIPBackbone
from neurosymbolic_da.nn.clip_concept_bottleneck import CLIPConceptBottleneck


class CLIPPCFGPipeline(nn.Module):
    """CLIP-based neuro-symbolic pipeline with PCFG grammar.

    Architecture:
        CLIP ViT-L/14 (frozen) → patch tokens (16x16)
        → Concept Bottleneck (concept bank → spatial primitives)
        → PCFG Grammar → class scores

    Args:
        n_primitives: number of concept primitives (k)
        n_classes: number of output classes
        concept_texts: list of concept descriptions for the bottleneck
        clip_model: CLIP model variant
        max_depth: max grammar derivation depth
        use_sparsemax: use sparsemax for grammar weights
    """

    def __init__(
        self,
        n_primitives: int = 8,
        n_classes: int = 200,
        concept_texts: list[str] | None = None,
        clip_model: str = "ViT-L/14",
        max_depth: int = 1,
        use_sparsemax: bool = False,
        domain_conditional: bool = False,
        n_domains: int = 0,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes

        # Frozen CLIP backbone
        self.backbone = CLIPBackbone(clip_model, freeze=True)

        # Concept bottleneck
        self.bottleneck = CLIPConceptBottleneck(
            patch_dim=self.backbone.out_channels,
            embed_dim=self.backbone.embed_dim,
            n_primitives=n_primitives,
            concept_texts=concept_texts,
        )

        # Initialize concept embeddings from CLIP text encoder
        self.bottleneck.initialize_concepts(self.backbone.model)

        # Spatial relation parameters (learnable)
        self.relation_params = RelationParams()

        # PCFG grammar
        self.grammar = LayoutGrammar(
            n_primitives, n_classes, max_depth,
            use_sparsemax=use_sparsemax,
            domain_conditional=domain_conditional,
            n_domains=n_domains,
        )

        self.score_temperature = 0.1

        # DDO components (initialized lazily)
        self._ddo_initialized = False
        self._domain_descriptors: list[str] = []
        self._domain_shift_embeddings: Tensor | None = None

    def forward(self, x: Tensor, domain_ids: Tensor | None = None) -> Tensor:
        """Compute class scores.

        Args:
            x: input images [B, 3, 224, 224]
            domain_ids: optional [B] domain indices for domain-conditional grammar

        Returns:
            log_probs: [B, n_classes]
        """
        patch_features = self.backbone(x)  # [B, C, 16, 16]
        heatmaps, coords, conf = self.bottleneck(patch_features)
        env = self.bottleneck.extract_batched_env(coords, conf, heatmaps)

        scores = self.grammar.forward_vectorized(
            env, self.relation_params, domain_ids=domain_ids,
        )

        return torch.log_softmax(scores / self.score_temperature, dim=-1)

    def forward_with_heatmaps(
        self, x: Tensor, domain_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute class scores and return heatmaps."""
        patch_features = self.backbone(x)
        heatmaps, coords, conf = self.bottleneck(patch_features)
        env = self.bottleneck.extract_batched_env(coords, conf, heatmaps)

        scores = self.grammar.forward_vectorized(
            env, self.relation_params, domain_ids=domain_ids,
        )

        log_probs = torch.log_softmax(scores / self.score_temperature, dim=-1)
        return log_probs, heatmaps

    def get_concept_activations(self, x: Tensor) -> Tensor:
        """Get global concept activations (CLS-based, for DDO).

        Returns [B, n_primitives] cosine similarities between
        CLS image embedding and concept embeddings.
        """
        with torch.no_grad():
            img_features = self.backbone.model.encode_image(x.float()).float()
            img_features = F.normalize(img_features, dim=-1)

        concepts = self.bottleneck.concept_embeddings  # [k, C]
        # Need to project through CLIP's visual projection if it exists
        if hasattr(self.backbone.model.visual, 'proj') and self.backbone.model.visual.proj is not None:
            # CLS goes through proj, concepts are in text space
            # Use the projected image features directly
            pass

        return img_features @ concepts.T  # [B, k]

    # ---- DDO (Domain Descriptor Orthogonality) ----

    def initialize_ddo(
        self,
        domain_descriptors: list[str],
        class_names: list[str],
        source_domain: str = "a photo",
    ):
        """Initialize DDO regularization.

        Precomputes domain shift embeddings for all (descriptor, class) pairs.

        Args:
            domain_descriptors: list of domain description phrases
            class_names: list of class names
            source_domain: source domain descriptor (e.g., "a photo")
        """
        if clip is None:
            raise ImportError("openai-clip required")

        device = self.bottleneck.concept_embeddings.device
        concepts = self.bottleneck.concept_embeddings  # [k, C]

        # Compute domain shift text embeddings
        # delta_t(p, y) = E_T([p, y]) - E_T([source, y])
        all_shifts = []  # [n_descriptors * n_classes, k]

        for desc in domain_descriptors:
            for cls_name in class_names:
                target_text = f"{desc} of a {cls_name}"
                source_text = f"{source_domain} of a {cls_name}"

                target_tok = clip.tokenize([target_text]).to(device)
                source_tok = clip.tokenize([source_text]).to(device)

                with torch.no_grad():
                    t_emb = self.backbone.model.encode_text(target_tok).float()
                    s_emb = self.backbone.model.encode_text(source_tok).float()
                    delta = F.normalize(t_emb - s_emb, dim=-1)

                # Simulated domain-specific concept activation
                a_sp = (delta @ concepts.T).squeeze(0)  # [k]
                all_shifts.append(a_sp)

        self._domain_shift_embeddings = torch.stack(all_shifts, dim=0)  # [D*C, k]
        self._domain_descriptors = domain_descriptors
        self._ddo_initialized = True

    def ddo_loss(self, classifier_weights: Tensor) -> Tensor:
        """Compute DDO loss: classifier should be orthogonal to domain shifts.

        Args:
            classifier_weights: [n_classes, k] — the grammar log_weights
                (or a linear head's weight matrix)

        Returns:
            scalar DDO loss
        """
        if not self._ddo_initialized or self._domain_shift_embeddings is None:
            return torch.tensor(0.0, device=classifier_weights.device)

        # L_DDO = E[ |W @ a_sp| ]
        # classifier_weights: [n_classes, n_productions]
        # domain_shift_embeddings: [D*C, k] — but k != n_productions
        # For now, use only the first k (has) production weights
        W = classifier_weights[:, :self.n_primitives]  # [n_classes, k]
        a_sp = self._domain_shift_embeddings  # [D*C, k]

        # [n_classes, k] @ [k, D*C] → [n_classes, D*C]
        projections = W @ a_sp.T
        return projections.abs().mean()

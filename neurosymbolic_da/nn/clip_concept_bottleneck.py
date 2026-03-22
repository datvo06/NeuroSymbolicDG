"""CLIP concept bottleneck for LanCE+PCFG integration.

Maps CLIP patch features to spatial primitives using a concept bank
of text-encoded bird part descriptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import clip
except ImportError:
    clip = None

from kornia.geometry.subpix import spatial_soft_argmax2d

from neurosymbolic_da.dsl.primitives import Primitive


# Default CUB bird part concepts (subset for k=8 primitives)
DEFAULT_CUB_CONCEPTS = [
    "a bird's head with eyes and beak",
    "a bird's breast and belly",
    "a bird's wing feathers",
    "a bird's tail feathers",
    "a bird's legs and feet",
    "a bird's back and nape",
    "a bird's bill or beak shape",
    "a bird's crown and forehead",
]


class CLIPConceptBottleneck(nn.Module):
    """Concept bottleneck using CLIP text embeddings as concept bank.

    Projects CLIP patch features onto concept directions to produce
    per-concept spatial heatmaps, then extracts primitive coordinates
    via spatial_soft_argmax2d.

    Args:
        patch_dim: dimension of CLIP patch features (1024 for ViT-L/14)
        n_primitives: number of primitives/concepts (k)
        concept_texts: list of concept text descriptions
        temperature: spatial softmax temperature
    """

    def __init__(
        self,
        patch_dim: int = 1024,
        embed_dim: int = 768,
        n_primitives: int = 8,
        concept_texts: list[str] | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Concept texts for encoding
        if concept_texts is None:
            concept_texts = DEFAULT_CUB_CONCEPTS[:n_primitives]
        assert len(concept_texts) == n_primitives
        self.concept_texts = concept_texts

        # Learnable projection from patch space to CLIP embed space
        # CLIP ViT-L/14: patch tokens are 1024-dim, text embeddings are 768-dim
        self.proj = nn.Linear(patch_dim, embed_dim, bias=False)

        # Concept embeddings will be registered as buffer after encoding
        # Shape: [n_primitives, embed_dim] (768 for ViT-L/14)
        self.register_buffer(
            "concept_embeddings",
            torch.zeros(n_primitives, embed_dim),
        )
        self._concepts_initialized = False

    def initialize_concepts(self, clip_model: nn.Module):
        """Encode concept texts using CLIP text encoder.

        Must be called once after model creation with the CLIP model.
        """
        if clip is None:
            raise ImportError("openai-clip required")

        tokens = clip.tokenize(self.concept_texts)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens.to(self.concept_embeddings.device))
            text_features = text_features.float()
            # L2 normalize
            text_features = F.normalize(text_features, dim=-1)
        self.concept_embeddings.copy_(text_features)
        self._concepts_initialized = True

    def forward(self, patch_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute concept heatmaps and extract primitive coordinates.

        Args:
            patch_features: [B, C, H, W] from CLIPBackbone

        Returns:
            heatmaps: [B, k, H, W] concept activation heatmaps
            coords: [B, k, 2] normalized (x, y) coordinates
            conf: [B, k] confidence scores
        """
        B, C, H, W = patch_features.shape

        # Project patch features: [B, C, H, W] → [B, H, W, C]
        patches = patch_features.permute(0, 2, 3, 1)
        patches_proj = self.proj(patches)  # [B, H, W, C]
        patches_proj = F.normalize(patches_proj, dim=-1)

        # Concept embeddings: [k, C], already L2-normalized
        concepts = self.concept_embeddings  # [k, C]

        # Cosine similarity: [B, H, W, C] × [k, C]^T → [B, H, W, k]
        heatmaps = torch.einsum('bhwc,kc->bhwk', patches_proj, concepts)
        heatmaps = heatmaps.permute(0, 3, 1, 2)  # [B, k, H, W]

        # Extract coordinates via spatial soft argmax
        temp = torch.tensor(self.temperature, device=heatmaps.device)
        coords = spatial_soft_argmax2d(heatmaps, temperature=temp)  # [B, k, 2]

        # Confidence: max activation per concept
        conf = heatmaps.view(B, self.n_primitives, -1).max(dim=-1).values  # [B, k]
        conf = torch.sigmoid(conf)  # normalize to [0, 1]

        return heatmaps, coords, conf

    def extract_batched_env(
        self, coords: Tensor, conf: Tensor, heatmaps: Tensor
    ) -> dict[int, Primitive]:
        """Build batched environment from concept bottleneck outputs.

        Args:
            coords: [B, k, 2] — (x, y) per primitive
            conf: [B, k] — confidence per primitive
            heatmaps: [B, k, H, W] — concept heatmaps

        Returns:
            env: dict mapping primitive index to Primitive dataclass
        """
        B = coords.shape[0]
        k = self.n_primitives
        H, W = heatmaps.shape[2], heatmaps.shape[3]

        env: dict[int, Primitive] = {}
        for j in range(k):
            cx = coords[:, j, 0]
            cy = coords[:, j, 1]

            # Estimate bounding box from heatmap spatial variance
            hmap = heatmaps[:, j, :, :]  # [B, H, W]
            hmap_soft = F.softmax(hmap.reshape(B, -1), dim=-1).reshape(B, H, W)

            gy = torch.arange(H, device=hmap.device, dtype=hmap.dtype)
            gx = torch.arange(W, device=hmap.device, dtype=hmap.dtype)
            gy, gx = torch.meshgrid(gy, gx, indexing='ij')

            var_x = (hmap_soft * (gx.unsqueeze(0) - cx.view(B, 1, 1)) ** 2).sum(dim=[1, 2])
            var_y = (hmap_soft * (gy.unsqueeze(0) - cy.view(B, 1, 1)) ** 2).sum(dim=[1, 2])

            std_x = var_x.sqrt().clamp(min=0.5)
            std_y = var_y.sqrt().clamp(min=0.5)

            env[j] = Primitive(
                type_idx=j,
                cx=cx, cy=cy,
                conf=conf[:, j],
                x1=cx - std_x, y1=cy - std_y,
                x2=cx + std_x, y2=cy + std_y,
            )

        return env

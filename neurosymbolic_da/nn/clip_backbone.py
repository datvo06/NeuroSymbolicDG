"""CLIP ViT-L/14 backbone for LanCE+PCFG integration.

Extracts patch tokens (16x16 grid) from frozen CLIP ViT-L/14,
providing spatial features for concept-based bottleneck.
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    import clip
except ImportError:
    clip = None


class CLIPBackbone(nn.Module):
    """Frozen CLIP ViT-L/14 that returns patch token features.

    Unlike standard CLIP usage (CLS token only), we extract the
    intermediate patch tokens to preserve spatial information for
    our PCFG grammar.

    Args:
        model_name: CLIP model variant (default "ViT-L/14")
        freeze: whether to freeze all CLIP parameters (default True)
    """

    def __init__(self, model_name: str = "ViT-L/14", freeze: bool = True):
        super().__init__()
        if clip is None:
            raise ImportError(
                "openai-clip is required: pip install git+https://github.com/openai/CLIP.git"
            )
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.model = self.model.float()  # ensure float32

        # ViT-L/14: patch_size=14, image_size=224 → 16x16 = 256 patches
        # Hidden dim = 1024 for ViT-L/14, embed dim = 768 (shared with text)
        self.out_channels = self.model.visual.transformer.width  # 1024
        self.embed_dim = self.model.visual.proj.shape[1] if self.model.visual.proj is not None else self.out_channels  # 768
        self.grid_size = self.model.visual.input_resolution // self.model.visual.conv1.kernel_size[0]  # 16

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text_tokens: Tensor) -> Tensor:
        """Encode text tokens to CLIP text embeddings.

        Args:
            text_tokens: [M, context_length] tokenized text

        Returns:
            text_features: [M, embed_dim] L2-normalized text embeddings
        """
        return self.model.encode_text(text_tokens).float()

    def forward(self, x: Tensor) -> Tensor:
        """Extract patch token features from CLIP ViT.

        Args:
            x: input images [B, 3, 224, 224]

        Returns:
            patch_features: [B, C, H, W] where H=W=grid_size (16 for ViT-L/14)
                           C = transformer width (1024 for ViT-L/14)
        """
        visual = self.model.visual
        B = x.shape[0]

        # Patch embedding: [B, 3, 224, 224] → [B, width, grid, grid]
        x = visual.conv1(x.float())
        # Reshape to sequence: [B, width, grid, grid] → [B, width, grid*grid] → [B, grid*grid, width]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)

        # Prepend class token
        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+HW, width]

        # Add positional embedding
        x = x + visual.positional_embedding.unsqueeze(0)

        # Pre-LN
        x = visual.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)  # [1+HW, B, width] for transformer
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 1+HW, width]

        # Post-LN on patch tokens (skip CLS token at index 0)
        patch_tokens = visual.ln_post(x[:, 1:, :])  # [B, HW, width]

        # Reshape to spatial grid: [B, HW, C] → [B, C, H, W]
        H = W = self.grid_size
        patch_features = patch_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return patch_features  # [B, 1024, 16, 16]

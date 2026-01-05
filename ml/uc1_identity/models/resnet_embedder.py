# ml/uc1_identity/models/resnet_embedder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetEmbedder(nn.Module):
    """
    UC1 Identity Embedder
    --------------------
    - Backbone: ResNet-50 (ImageNet pretrained)
    - Output: 256-D L2-normalized embedding
    - Used for BOTH enrollment and probe images
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # Load ResNet-50 backbone
        backbone = models.resnet50(pretrained=pretrained)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output now: (B, 2048, 1, 1)

        self.embedding_head = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)

        Returns:
            embeddings: Tensor of shape (B, embedding_dim), L2-normalized
        """
        # Feature extraction
        x = self.backbone(x)          # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 2048)

        # Embedding projection
        x = self.embedding_head(x)    # (B, embedding_dim)

        # L2 normalization (CRITICAL)
        x = F.normalize(x, p=2, dim=1)

        return x

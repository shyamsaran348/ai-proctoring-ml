# ml/uc1_identity/losses/triplet_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss for UC1 Identity Verification
    ------------------------------------------
    Enforces:
        d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (B, D) embedding
            positive: (B, D) embedding
            negative: (B, D) embedding

        Returns:
            Scalar triplet loss
        """

        # Euclidean distances
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)

        # Triplet margin loss
        losses = F.relu(d_ap - d_an + self.margin)

        return losses.mean()

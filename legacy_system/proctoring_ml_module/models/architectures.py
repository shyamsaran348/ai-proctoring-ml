import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------
# UC1: ResNet Embedder
# ---------------------------------------------------------
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
        # Note: In a strictly offline/standalone env, we might need to handle 'pretrained=True' failing if no internet.
        # But for now, we assume standard behavior or cached hub.
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


# ---------------------------------------------------------
# UC2: Temporal LSTM
# ---------------------------------------------------------
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, input_dim)
        _, (h_n, _) = self.lstm(x)
        # We take the LAST hidden state to predict instability for this window
        out = self.fc(h_n[-1])
        return out.squeeze(1)


# ---------------------------------------------------------
# UC5: Risk Fusion GRU
# ---------------------------------------------------------
class RiskFusionGRU(nn.Module):
    """
    GRU-based risk fusion model for UC5.

    Input:
        x: Tensor of shape (B, T, 2)
           where features = [UC1_similarity, UC2_impersonation_prob]

    Output:
        risk_traj: Tensor of shape (B, T)
                   continuous risk evidence over time
        final_risk: Tensor of shape (B,)
                    final session-level cheating risk (logit)
    """

    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()

        self.hidden_dim = hidden_dim

        # GRU for temporal risk accumulation
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Linear projection from hidden state → risk evidence
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, 2)

        Returns:
            risk_traj: (B, T)
            final_risk: (B,)
        """

        # h_seq: (B, T, hidden_dim)
        h_seq, _ = self.gru(x)

        # risk_logits: (B, T, 1)
        risk_logits = self.risk_head(h_seq)

        # squeeze last dim → (B, T)
        risk_traj = risk_logits.squeeze(-1)

        # final timestep risk (session-level)
        final_risk = risk_traj[:, -1]

        return risk_traj, final_risk

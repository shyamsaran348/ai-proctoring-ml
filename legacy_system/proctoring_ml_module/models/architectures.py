import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------
# UC1: ResNet Embedder
# ---------------------------------------------------------
class ResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()

        backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding_head = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_head(x)
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
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(1)


# ---------------------------------------------------------
# GAM: Gaze Attentiveness Model (Phase 17)
# ---------------------------------------------------------
class GAM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        out = self.fc(h_seq)
        return self.sigmoid(out)


# ---------------------------------------------------------
# HGDM: Head-Gaze Dynamics Model (Phase 18)
# ---------------------------------------------------------
class HGDM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        out = self.fc(h_seq)
        return self.sigmoid(out)


# ---------------------------------------------------------
# UC6: Audio Anomaly Detection (Phase 19)
# ---------------------------------------------------------
class UC6AudioNet(nn.Module):
    """
    Sequence-based audio anomaly detector.
    Input: (B, T, 1) - amplitude/volume levels
    """
    def __init__(self, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)


# ---------------------------------------------------------
# UC5: Risk Fusion GRU (7-Signal Version - Phase 19)
# ---------------------------------------------------------
class RiskFusionGRU(nn.Module):
    """
    GRU-based risk fusion model (Phase 19).

    Input:
        x: Tensor of shape (B, T, 7)
           features = [
               UC1_similarity,
               UC2_instability_prob,
               UC3_presence_prob,
               UC4_drift_prob,
               GAM_gaze_prob,
               HGDM_head_gaze_prob,
               UC6_audio_prob
           ]

    Output:
        risk_traj: (B, T)
        final_risk: (B,)
    """

    def __init__(self, input_dim=7, hidden_dim=32):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.risk_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):

        h_seq, _ = self.gru(x)

        output = self.risk_head(h_seq)
        mu, log_var = output.chunk(2, dim=-1)

        risk_traj = torch.sigmoid(mu).squeeze(-1)
        uncertainty_traj = torch.exp(0.5 * log_var).squeeze(-1)

        final_risk = risk_traj[:, -1]
        final_uncertainty = uncertainty_traj[:, -1]

        return risk_traj, final_risk, uncertainty_traj, final_uncertainty
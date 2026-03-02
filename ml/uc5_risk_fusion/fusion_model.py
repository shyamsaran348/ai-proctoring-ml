import torch
import torch.nn as nn


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

    def __init__(self, input_dim=4, hidden_dim=32):
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

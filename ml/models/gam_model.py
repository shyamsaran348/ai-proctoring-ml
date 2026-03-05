import torch
import torch.nn as nn

class GAM(nn.Module):
    """
    Gaze Attentiveness Model (GAM) - Phase 17.
    Models temporal eye gaze patterns to output attentiveness probability.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3):
        super(GAM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # BiLSTM hidden dim is doubled
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, T, 6)
        Returns:
            G_t: (B, T, 1) continuous probability trajectory
        """
        h_seq, _ = self.lstm(x)
        out = self.fc(h_seq)
        return self.sigmoid(out)

import torch
import torch.nn as nn

class UC2TemporalLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_final = h_n[-1]
        return self.fc(h_final)

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from pathlib import Path


class PresenceModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(1)


class UC3PresenceEngine:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 60

        model_dir = Path(config["model_dir"])

        self.feature_mean = np.load(model_dir / "feature_mean.npy")
        self.feature_std = np.load(model_dir / "feature_std.npy")

        self.feature_std[self.feature_std == 0] = 1e-6

        self.model = PresenceModel()
        self.model.load_state_dict(
            torch.load(model_dir / "presence_model.pth", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.buffer = deque(maxlen=self.sequence_length)

    def reset(self):
        self.buffer.clear()

    def update(self, feature_vector):
        """
        feature_vector: numpy array shape (6,)
        Returns:
            presence_confidence_t (float) OR None
        """

        self.buffer.append(feature_vector)

        if len(self.buffer) < self.sequence_length:
            return None

        sequence = np.array(self.buffer)

        sequence = (sequence - self.feature_mean) / self.feature_std

        sequence_tensor = torch.tensor(
            sequence, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(sequence_tensor)
            prob = torch.sigmoid(logit).item()

        return prob
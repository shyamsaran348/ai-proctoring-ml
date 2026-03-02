import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.models.architectures import RiskFusionGRU


class UC5Engine:
    def __init__(self, config):

        self.device = torch.device(
            config.get('inference', {}).get('device', 'cpu')
        )

        self.model_config = config['models']['uc5']
        self.model_path = self.model_config['path']
        self.hidden_dim = self.model_config.get('hidden_dim', 32)

        self.max_history = self.model_config.get('max_history', 300)

        self.buffer = deque(maxlen=self.max_history)

        self.model = RiskFusionGRU(
            input_dim=4,
            hidden_dim=self.hidden_dim
        )

        self.load_weights()
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self):
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[UC5] Loaded weights from {self.model_path}")
            except RuntimeError as e:
                print(f"[UC5] WARNING: Checkpoint shape mismatch ({e}). Using random weights for now.")
        else:
            print(f"[UC5] WARNING: Checkpoint not found at {self.model_path}. Using random weights.")

    def reset(self):
        self.buffer.clear()

    def update(self, uc1_sim: float, uc2_prob: float, uc3_presence: float, uc4_drift: float) -> float:

        self.buffer.append([uc1_sim, uc2_prob, uc3_presence, uc4_drift])

        seq_data = list(self.buffer)

        input_tensor = torch.tensor(
            seq_data,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, final_risk = self.model(input_tensor)
            risk_val = torch.sigmoid(final_risk).item()

        return risk_val
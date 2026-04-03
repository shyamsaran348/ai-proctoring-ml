import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.models.architectures import UC6AudioNet


class UC6Engine:
    """
    Sentinel Audio Anomaly Engine (Phase 19).
    Tracks real-time acoustic amplitude and frequency irregularities.
    """
    def __init__(self, config):
        self.device = torch.device(
            config.get('inference', {}).get('device', 'cpu')
        )

        self.audio_config = config.get('models', {}).get('uc6', {})
        self.model_path = self.audio_config.get('path', 'proctoring_ml_module/models/uc6_audio.pth')
        self.hidden_dim = self.audio_config.get('hidden_dim', 32)
        self.max_history = self.audio_config.get('max_history', 150)
        self.threshold = self.audio_config.get('threshold', 0.6)

        # Buffer for volume levels
        self.buffer = deque(maxlen=self.max_history)

        self.model = UC6AudioNet(
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
                print(f"[UC6] Loaded weights from {self.model_path}")
            except Exception as e:
                print(f"[UC6] WARNING: Weight load failed ({e}). Running on heuristics.")
        else:
            print(f"[UC6] Initializing generic Acoustic Anomaly detector.")

    def reset(self):
        self.buffer.clear()

    def update(self, volume: float) -> float:
        """
        Processes a single volume/amplitude data point.
        Returns probability of acoustic anomaly.
        """
        # Feature Engineering: 1D signal
        self.buffer.append([volume])

        if len(self.buffer) < 5:
            return 0.5 # Neutral during warm-up

        seq_data = list(self.buffer)
        input_tensor = torch.tensor(
            seq_data,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model(input_tensor).item()
        
        # Heuristic override for extreme noise
        if volume > 0.8:
            prob = max(prob, 0.9)
            
        return prob

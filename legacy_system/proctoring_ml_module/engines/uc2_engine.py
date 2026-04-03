import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import deque

# Add root to path to find models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.models.architectures import TemporalLSTM

class UC2Engine:
    def __init__(self, config):
        self.device = torch.device(config.get('inference', {}).get('device', 'cpu'))
        self.model_config = config['models']['uc2']
        self.model_path = self.model_config['path']
        self.seq_len = self.model_config.get('sequence_length', 60)
        self.hidden_dim = self.model_config.get('hidden_dim', 64)
        
        # Buffer to hold last N similarity scores
        self.buffer = deque(maxlen=self.seq_len)
        
        self.model = TemporalLSTM(
            input_dim=self.model_config.get('input_dim', 1),
            hidden_dim=self.hidden_dim,
            num_layers=self.model_config.get('num_layers', 2)
        )
        self.load_weights()
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[UC2] Loaded weights from {self.model_path}")
        else:
            print(f"[UC2] WARNING: Checkpoint not found at {self.model_path}. Using random weights.")

    def reset(self):
        """Clear the history buffer."""
        self.buffer.clear()

    def update(self, uc1_similarity: float) -> float:
        """
        Update the engine with the latest UC1 similarity score.
        Returns the instability probability (0.0 to 1.0).
        """
        # Append to buffer
        self.buffer.append(uc1_similarity)
        
        # Prepare sequence
        current_len = len(self.buffer)
        if current_len == 0:
            return 0.0 # Should not happen if update is called
        
        # If buffer is not full, pad with the OLDEST value available (or newest?)
        # Strategy: Replicate the first (oldest) value in the buffer to the left
        # to fill the sequence.
        # e.g., buffer=[0.9, 0.8] (len 2), seq -> [0.9, ..., 0.9, 0.9, 0.8]
        # This simulates "steady state" before the session started.
        
        seq_data = list(self.buffer)
        if current_len < self.seq_len:
            pad_val = seq_data[0] # The earliest known state
            padding = [pad_val] * (self.seq_len - current_len)
            seq_data = padding + seq_data
            
        # Convert to tensor: (B=1, T=60, Input=1)
        input_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)
        
        with torch.no_grad():
            logit = self.model(input_tensor) # (1,)
            prob = torch.sigmoid(logit).item()
            
        return prob

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
        
        # Multi-Res Requirements
        self.base_seq_len = self.model_config.get('sequence_length', 60)
        self.max_buffer_len = 120 # For Long Stream
        
        self.hidden_dim = self.model_config.get('hidden_dim', 64)
        
        # Buffer to hold last N similarity scores
        self.buffer = deque(maxlen=self.max_buffer_len)
        
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
        Returns the instability probability (0.0 to 1.0) using Multi-Resolution Ensemble.
        """
        # Append to buffer
        self.buffer.append(uc1_similarity)
        
        if len(self.buffer) == 0:
            return 0.0

        # Create Parallel Streams
        # All streams must result in [60, 1] tensor for the model
        
        # 1. Short Stream (Last 30 frames) - Micro Instability
        short_seq = self._prepare_stream(window=30, downsample=1)
        
        # 2. Medium Stream (Last 60 frames) - Standard
        med_seq = self._prepare_stream(window=60, downsample=1)
        
        # 3. Long Stream (Last 120 frames) - Drift / Slow Changes
        long_seq = self._prepare_stream(window=120, downsample=2)
        
        # Batch: (3, 60, 1)
        batch = torch.stack([short_seq, med_seq, long_seq]).to(self.device)
        
        with torch.inference_mode():
            logits = self.model(batch) # (3,)
            probs = torch.sigmoid(logits)
            
        # Ensemble: Max Instability (Conservative: if any scale is unstable, flag it)
        # Or mean? Max is safer for proctoring.
        final_prob = torch.max(probs).item()
            
        return final_prob

    def _prepare_stream(self, window, downsample=1):
        """
        Extracts a window from the buffer, pads/processes it to fit base_seq_len (60).
        """
        data = list(self.buffer)
        
        # 1. Slice the relevant window from the END
        # If we want the last 'window' frames
        target_len = window
        start_idx = max(0, len(data) - target_len)
        slice_data = data[start_idx:]
        
        # 2. Pad if insufficient length (pre-pad)
        if len(slice_data) < target_len:
            pad_val = slice_data[0] if slice_data else 0.0
            padding = [pad_val] * (target_len - len(slice_data))
            slice_data = padding + slice_data
            
        # 3. Downsample if needed
        # e.g., if window=120, downsample=2 -> 60 frames
        if downsample > 1:
            slice_data = slice_data[::downsample]
        
        # 4. Final check: Should be exactly 60 now?
        # If window=30, downsample=1 -> 30 frames. Need to pad to 60?
        # Model expects 60.
        # Logic: If result < 60, pad again?
        # For Short (30): We have 30 frames. Need 60.
        # Pad with first value again.
        if len(slice_data) < self.base_seq_len:
            pad_val = slice_data[0]
            padding = [pad_val] * (self.base_seq_len - len(slice_data))
            slice_data = padding + slice_data
        elif len(slice_data) > self.base_seq_len:
            # Should not happen if logic is correct, but clip just in case
            slice_data = slice_data[-self.base_seq_len:]
            
        return torch.tensor(slice_data, dtype=torch.float32).unsqueeze(1) # (60, 1)

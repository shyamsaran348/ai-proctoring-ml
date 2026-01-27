import torch
import torch.nn as nn
import numpy as np
import os
import sys
from collections import deque

# Add root to path to find models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from proctoring_ml_module.models.architectures import RiskFusionGRU

class UC5Engine:
    def __init__(self, config):
        self.device = torch.device(config.get('inference', {}).get('device', 'cpu'))
        self.model_config = config['models']['uc5']
        self.model_path = self.model_config['path']
        self.hidden_dim = self.model_config.get('hidden_dim', 32)
        
        # Max history for risk calculation (sliding window size)
        self.max_history = self.model_config.get('max_history', 300)
        
        # Buffer to hold last N (UC1, UC2) pairs
        self.buffer = deque(maxlen=self.max_history)
        
        # Risk Regularization State
        self.last_risk = 0.0
        self.alpha = 0.2 # Smoothing factor (Lower = Smoother, Higher = More Responsive)
        
        self.model = RiskFusionGRU(
            input_dim=self.model_config.get('input_dim', 2),
            hidden_dim=self.hidden_dim
        )
        self.load_weights()
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[UC5] Loaded weights from {self.model_path}")
        else:
            print(f"[UC5] WARNING: Checkpoint not found at {self.model_path}. Using random weights.")

    def reset(self):
        self.buffer.clear()
        self.last_risk = 0.0

    def update(self, uc1_sim: float, uc2_prob: float) -> float:
        """
        Update with latest frame metrics.
        Returns: Continuous risk score (Smoothed Probability).
        """
        self.buffer.append([uc1_sim, uc2_prob])
        
        # Convert buffer to tensor
        # Shape: (B=1, T=len(buffer), Input=2)
        seq_data = list(self.buffer)
        input_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            _, final_risk = self.model(input_tensor) # final_risk is (B,)
            raw_risk = torch.sigmoid(final_risk).item()
            
        # RISK STABILITY REGULARIZATION (Improvement 3)
        # Apply Exponential Moving Average (EMA) to prevent jitter/oscillations
        # This encourages monotonic behavior and smooth transitions.
        
        # If this is the first frame, init last_risk
        if len(self.buffer) == 1:
            self.last_risk = raw_risk
            
        smoothed_risk = self.alpha * raw_risk + (1 - self.alpha) * self.last_risk
        
        # Update state
        self.last_risk = smoothed_risk
            
        return smoothed_risk

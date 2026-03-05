import torch
import numpy as np
from collections import deque
from ml.preprocessing.gaze_normalizer import normalize_gaze_features

class GAMEngine:
    """
    GAMEngine - Phase 17.
    Wraps the GAM model, manages temporal buffers, and computes G_t probabilities.
    """
    def __init__(self, model, device='cpu', max_history=120):
        self.model = model
        self.device = torch.device(device)
        self.max_history = max_history
        self.buffer = deque(maxlen=max_history)
        
        self.model.to(self.device)
        self.model.eval()

    def reset(self):
        """Clears the temporal buffer."""
        self.buffer.clear()

    def update(self, gaze_features: np.ndarray) -> float:
        """
        Updates the engine with a single frame's gaze features.
        
        Args:
            gaze_features: np.ndarray (6,)
        
        Returns:
            G_t: float (0.0 to 1.0) attentiveness probability
        """
        # 1. Normalize and Append
        norm_features = normalize_gaze_features(gaze_features)
        self.buffer.append(norm_features)

        # 2. Minimum Window Check
        # If buffer not full, return neutral baseline (0.5)
        # This keeps consistency with the temporal accumulation philosophy.
        if len(self.buffer) < self.max_history:
            return 0.5

        # 3. Model Inference
        # Convert state buffer to tensor: (B=1, T=120, C=6)
        sequence = np.array(list(self.buffer), dtype=np.float32)
        sequence_t = torch.tensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get G_t trajectory and return last frame's probability
            g_t_trajectory = self.model(sequence_t)
            g_t_current = g_t_trajectory[0, -1, 0].item()

        return g_t_current

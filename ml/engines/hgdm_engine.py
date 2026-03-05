import torch
import numpy as np
from collections import deque
import sys
import os

# Add parent dir to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.hgdm_model import HGDM
from ml.preprocessing.pose_gaze_builder import PoseGazeBuilder

class HGDMEngine:
    """
    Expert engine for Head-Gaze Dynamics.
    Manages buffering and inference for the HGDM model.
    """
    def __init__(self, model_path=None, device='cpu', seq_len=120):
        self.device = device
        self.seq_len = seq_len
        self.feature_builder = PoseGazeBuilder()
        
        # Initialize model
        self.model = HGDM(input_dim=7).to(device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Buffer for 7D features
        self.buffer = deque(maxlen=seq_len)
        # Pad with zeros initially
        for _ in range(seq_len):
            self.buffer.append(np.zeros(7, dtype=np.float32))

    def update(self, head_pose, gaze_features):
        """
        head_pose: [yaw, pitch, roll]
        gaze_features: [yaw, pitch, ...]
        returns: h_prob (float)
        """
        # Build 7D feature
        feat = self.feature_builder.build(head_pose, gaze_features)
        self.buffer.append(feat)
        
        # Prepare tensor (batch=1, seq_len=120, dim=7)
        seq_tensor = torch.FloatTensor(np.array(self.buffer)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(seq_tensor)
            # Take the probability from the last time step
            h_prob = output[0, -1, 0].item()
            
        return h_prob

    def reset(self):
        self.buffer.clear()
        for _ in range(self.seq_len):
            self.buffer.append(np.zeros(7, dtype=np.float32))
        self.feature_builder.reset()

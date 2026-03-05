import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Ensure we can import architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from proctoring_ml_module.models.architectures import RiskFusionGRU

class AttributionEngine:
    """
    Computes temporal attribution (importance scores) for the Risk Fusion Engine.
    Uses Local Sensitivity Analysis: grad(rho_t) / grad(x_t).
    """
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = RiskFusionGRU(input_dim=6, hidden_dim=32).to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def attribute_session(self, signals_seq):
        """
        signals_seq: (T, 6) numpy array
        Returns: attributions (T, 6) numpy array
        """
        # (1, T, 6)
        x = torch.FloatTensor(signals_seq).unsqueeze(0).to(self.device)
        x.requires_grad = True
        
        # Forward pass: get trajectory
        risk_traj, _ = self.model(x)
        risk_probs = torch.sigmoid(risk_traj).squeeze(0) # (T,)
        
        T = signals_seq.shape[0]
        attributions = np.zeros((T, 6))
        
        # We compute sensitivity for each timestep t
        # Importance of signal i at time t is d(rho_t) / d(x_{t,i})
        for t in range(T):
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
                
            # Current risk at time t
            rho_t = risk_probs[t]
            
            # Backward from rho_t to input x
            rho_t.backward(retain_graph=True)
            
            # Get gradients at time t
            # x.grad shape is (1, T, 6)
            grads = x.grad[0, t, :].detach().cpu().numpy()
            
            # Attribution is absolute magnitude of sensitivity 
            # (or positive influence if we want to show what pushes risk UP)
            attributions[t] = np.abs(grads)
            
        # Normalize per frame so importance sums to 1 (optional, for heatmap clarity)
        row_sums = attributions.sum(axis=1, keepdims=True)
        attributions = np.divide(attributions, row_sums, out=np.zeros_like(attributions), where=row_sums!=0)
        
        return attributions

if __name__ == "__main__":
    # Test
    model_path = "proctoring_ml_module/models/uc5_risk_gru_v3.pth"
    if os.path.exists(model_path):
        engine = AttributionEngine(model_path)
        dummy_seq = np.random.rand(120, 6)
        attr = engine.attribute_session(dummy_seq)
        print(f"Attribution shape: {attr.shape}")
        print(f"Sample attribution (frame 60): {attr[60]}")

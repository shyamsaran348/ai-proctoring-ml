import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

# Ensure we can import engines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from ml.engines.attribution_engine import AttributionEngine

def ar1_process(mean, std, length, phi=0.85):
    noise = np.random.normal(0, std * np.sqrt(1 - phi**2), length)
    signal = np.zeros(length)
    signal[0] = np.random.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return signal

def plot_attribution_heatmap():
    """
    Generates a Signal Importance Heatmap (XAI) for a Substitution session.
    """
    model_path = "proctoring_ml_module/models/uc5_risk_gru_v3.pth"
    if not os.path.exists(model_path):
        print("Error: RFE v3 model not found.")
        return
        
    engine = AttributionEngine(model_path)
    seq_len = 120
    
    # 1. Generate a "Substitution" Session
    # [IdentitySim, Instability, Presence, Drift, Gaze, HGDM]
    s_t = ar1_process(0.45, 0.1, seq_len) # Low Sim
    i_t = ar1_process(0.6, 0.1, seq_len)  # High Instability
    p_t = ar1_process(0.95, 0.02, seq_len) # Present
    d_t = ar1_process(0.7, 0.1, seq_len)  # High Drift
    g_t = ar1_process(0.9, 0.05, seq_len) # Looking at screen
    h_t = ar1_process(0.9, 0.05, seq_len) # Relaxed
    
    signals = np.stack([s_t, i_t, p_t, d_t, g_t, h_t], axis=1)
    signals = np.clip(signals, 0, 1)
    
    # 2. Compute Attribution
    attr = engine.attribute_session(signals)
    
    # 3. Compute Risk Trajectory
    with torch.no_grad():
        x = torch.FloatTensor(signals).unsqueeze(0)
        risk_traj, _ = engine.model(x)
        risk_probs = torch.sigmoid(risk_traj).squeeze(0).cpu().numpy()
        
    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    # Top Plot: Signals
    time = np.arange(seq_len)
    ax1.plot(time, s_t, label='Sim (S)', alpha=0.7)
    ax1.plot(time, i_t, label='Instab (I)', alpha=0.7)
    ax1.plot(time, d_t, label='Drift (D)', alpha=0.7, color='red', linewidth=2)
    ax1.plot(time, risk_probs, label='Risk (ρ)', color='black', linewidth=3)
    ax1.set_title('Signal Trajectories (Substitution Session)')
    ax1.set_ylabel('Intensity / Probability')
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: Heatmap
    # attr is (T, 6) -> needs (6, T) for imshow
    im = ax2.imshow(attr.T, aspect='auto', cmap='YlOrRd')
    ax2.set_yticks(np.arange(6))
    ax2.set_yticklabels(['S', 'I', 'P', 'D', 'G', 'H'])
    ax2.set_title('Temporal Signal Importance (XAI Attribution)')
    ax2.set_xlabel('Time (Frames)')
    ax2.set_ylabel('Signals')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
    cbar.set_label('Importance Weight')
    
    plt.tight_layout()
    
    out_dir = '/Users/shyam/Desktop/ai-proctoring-ml/paper_figures'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'temporal_attribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"Attribution plot saved to {save_path}")

if __name__ == "__main__":
    plot_attribution_heatmap()

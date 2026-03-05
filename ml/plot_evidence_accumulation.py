import matplotlib.pyplot as plt
import numpy as np
import os

# Style
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

def ar1_signal(mean, std, length=120, phi=0.85, rng=None):
    if rng is None: rng = np.random.default_rng()
    noise = rng.normal(0, std * np.sqrt(1 - phi**2), size=length)
    signal = np.zeros(length)
    signal[0] = rng.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return signal

def generate_evidence_diagram():
    """Outputs evidence_accumulation.png"""
    print("Generating Evidence Accumulation Diagram...")
    rng = np.random.default_rng(99)
    t = np.arange(120)
    
    # CASE: Phone Usage (Gaze drop + Identity slight drift)
    # Individual signals are noisy and "weak"
    s_t = ar1_signal(0.7, 0.12, rng=rng) # Weak similarity
    p_t = ar1_signal(0.8, 0.1, rng=rng)  # Presence mostly there
    # Gaze starts high, then drops as student looks at phone
    g_t = np.concatenate([ar1_signal(0.9, 0.05, 40, rng=rng), ar1_signal(0.2, 0.15, 80, rng=rng)])
    
    # Cumulative Risk Trajectory (The Sigma Signal)
    # Starts low, steadily grows as evidence (low G_t) accumulates
    risk = np.zeros(120)
    current_risk = 0.05
    for i in range(120):
        # simplified logic: if gaze is low, increase risk
        if i > 40:
            evidence = (0.9 - g_t[i]) * 0.05
            current_risk += evidence
        risk[i] = 1 / (1 + np.exp(-(current_risk - 0.5) * 10)) # Sigmoid for visualization
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top Plot: Multiple Weak Signals
    ax1.plot(t, s_t, label='Identity Similarity ($S_t$)', color='#3498db', alpha=0.6, lw=1)
    ax1.plot(t, p_t, label='Presence Confidence ($P_t$)', color='#2ecc71', alpha=0.6, lw=1)
    ax1.plot(t, g_t, label='Gaze Attentiveness ($G_t$)', color='#e74c3c', lw=2)
    ax1.axhline(0.5, color='gray', ls='--', alpha=0.3)
    ax1.set_ylabel('Signal Intensity')
    ax1.set_title('Stage 1: Multi-Signal Input (Noisy/Weak Components)', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=8)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(0, 1.1)
    
    # Bottom Plot: The Fused Risk Result
    ax2.fill_between(t, 0, risk, color='#e74c3c', alpha=0.2)
    ax2.plot(t, risk, color='#c0392b', lw=2.5, label='Accumulated Risk ($\\rho_t$)')
    ax2.set_ylabel('Risk Probability')
    ax2.set_xlabel('Time (Session Evolution)')
    ax2.set_title('Stage 2: Temporal Evidence Accumulation (RFE Result)', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(alpha=0.2)
    ax2.legend(loc='upper left', fontsize=8)
    
    # Annotations
    ax1.annotate('Anomalous Gaze Patterns Detected', xy=(80, 0.2), xytext=(90, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), fontsize=8)
    
    ax2.annotate('Cumulative Risk Growth', xy=(100, 0.8), xytext=(60, 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), fontsize=8)

    plt.tight_layout()
    out_dir = 'paper_figures'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'evidence_accumulation.png'), dpi=300)
    print(f"✅ Evidence accumulation diagram saved to {out_dir}/evidence_accumulation.png")

if __name__ == "__main__":
    generate_evidence_diagram()

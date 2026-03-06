import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def generate_evidence_flow_diagram():
    """
    Figure 2: Temporal Evidence Flow — SIMPLE version.
    Three clean horizontal stages with minimal visual clutter.
    """
    np.random.seed(42)
    
    fig = plt.figure(figsize=(14, 6))
    
    # ─── Top row: the pipeline diagram ───
    ax = fig.add_axes([0.02, 0.30, 0.96, 0.65])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ═══ STAGE 1: Signal names (left) ═══
    signals = [
        (r'$S_t$  Identity Similarity',   '#1976D2'),
        (r'$I_t$   Identity Instability',  '#388E3C'),
        (r'$P_t$  Presence Confidence',    '#7B1FA2'),
        (r'$D_t$  Identity Drift',         '#E65100'),
        (r'$G_t$  Gaze Attentiveness',     '#C62828'),
        (r'$H_t$  Head-Gaze Dynamics',     '#00838F'),
    ]
    
    y_pos = np.linspace(0.88, 0.12, len(signals))
    
    for i, (label, color) in enumerate(signals):
        ax.text(0.12, y_pos[i], label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
    
    # Stage 1 bracket / header
    ax.text(0.12, 0.98, 'Weak Signals', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#37474F')
    ax.text(0.12, 0.02, '(AUC: 0.64 – 0.94)', ha='center', va='center',
            fontsize=9, color='#9E9E9E', style='italic')
    
    # ═══ BIG ARROW 1 ═══
    ax.annotate('', xy=(0.37, 0.50), xytext=(0.25, 0.50),
                arrowprops=dict(arrowstyle='->', lw=3, color='#90A4AE'))
    
    # ═══ STAGE 2: Temporal Experts box ═══
    expert_box = mpatches.FancyBboxPatch(
        (0.37, 0.20), 0.18, 0.60,
        boxstyle="round,pad=0.03",
        facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2
    )
    ax.add_patch(expert_box)
    ax.text(0.46, 0.62, 'Temporal\nExperts', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1B5E20')
    ax.text(0.46, 0.42, 'IIM · PAM · LDD\nGAM · HGDM', ha='center', va='center',
            fontsize=10, color='#4CAF50')
    ax.text(0.46, 0.28, '(Bi-LSTM / LSTM)', ha='center', va='center',
            fontsize=8, color='#81C784', style='italic')
    
    # ═══ BIG ARROW 2 ═══
    ax.annotate('', xy=(0.62, 0.50), xytext=(0.56, 0.50),
                arrowprops=dict(arrowstyle='->', lw=3, color='#90A4AE'))
    
    # ═══ STAGE 3: GRU Fusion box ═══
    gru_box = mpatches.FancyBboxPatch(
        (0.62, 0.25), 0.15, 0.50,
        boxstyle="round,pad=0.03",
        facecolor='#FFF3E0', edgecolor='#E65100', lw=2.5
    )
    ax.add_patch(gru_box)
    ax.text(0.695, 0.58, 'GRU', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#BF360C')
    ax.text(0.695, 0.43, 'Risk Fusion\nEngine', ha='center', va='center',
            fontsize=11, color='#E65100')
    ax.text(0.695, 0.32, r'$[\mu_t, \log \sigma_t^2]$', ha='center', va='center',
            fontsize=9, color='#8D6E63', style='italic')
    
    # ═══ BIG ARROW 3 ═══
    ax.annotate('', xy=(0.84, 0.50), xytext=(0.78, 0.50),
                arrowprops=dict(arrowstyle='->', lw=3, color='#90A4AE'))
    
    # ═══ STAGE 4: Output label ═══
    ax.text(0.92, 0.55, r'Risk $\rho_t$', ha='center', va='center',
            fontsize=15, fontweight='bold', color='#C62828')
    ax.text(0.92, 0.42, 'Session\nTrajectory', ha='center', va='center',
            fontsize=11, color='#E53935')
    ax.text(0.92, 0.98, 'Output', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#37474F')
    ax.text(0.92, 0.02, '(AUC: 0.9992)', ha='center', va='center',
            fontsize=9, color='#9E9E9E', style='italic')
    
    # ─── Bottom row: Genuine vs Cheating trajectory ───
    ax_traj = fig.add_axes([0.12, 0.06, 0.76, 0.22])
    
    t = np.linspace(0, 120, 200)
    
    # Genuine: stays flat near 0
    risk_gen = 0.02 + 0.015 * np.sin(0.1 * t) + 0.01 * np.random.randn(200)
    risk_gen = np.clip(risk_gen, 0, 0.12)
    unc_gen = 0.03 + 0.005 * np.random.randn(200)
    
    # Cheating: smooth sigmoid rise
    sigmoid = 1.0 / (1.0 + np.exp(-0.12 * (t - 55)))
    risk_cheat = sigmoid + 0.02 * np.random.randn(200)
    risk_cheat = np.clip(risk_cheat, 0, 1)
    # Uncertainty spikes around t=55 (the transition/ambiguous phase)
    unc_cheat = 0.05 + 0.25 * np.exp(-((t - 55)**2) / 100) + 0.01 * np.random.randn(200)
    unc_cheat = np.clip(unc_cheat, 0.01, 0.4)
    
    ax_traj.fill_between(t, np.clip(risk_cheat - unc_cheat, 0, 1), np.clip(risk_cheat + unc_cheat, 0, 1), alpha=0.25, color='#E53935')
    ax_traj.plot(t, risk_cheat, color='#E53935', lw=2.5, label='Cheating Risk $\\rho_t$')
    
    ax_traj.fill_between(t, np.clip(risk_gen - unc_gen, 0, 1), np.clip(risk_gen + unc_gen, 0, 1), alpha=0.25, color='#2E7D32')
    ax_traj.plot(t, risk_gen, color='#2E7D32', lw=2.5, label=r'Genuine Risk $\rho_t$')
    
    # We add a proxy artist to legend for Uncertainty
    unc_patch = mpatches.Patch(color='gray', alpha=0.3, label=r'Uncertainty $\pm\sigma_t$')
    
    ax_traj.set_xlabel('Frame (t)', fontsize=10)
    ax_traj.set_ylabel(r'$\rho_t$', fontsize=12)
    ax_traj.set_ylim(-0.05, 1.1)
    ax_traj.set_xlim(0, 120)
    ax_traj.set_yticks([0, 0.5, 1.0])
    handles, labels = ax_traj.get_legend_handles_labels()
    handles.append(unc_patch)
    ax_traj.legend(handles=handles, fontsize=9, loc='center left', framealpha=0.9)
    ax_traj.spines['top'].set_visible(False)
    ax_traj.spines['right'].set_visible(False)
    ax_traj.set_title('Accumulated Risk Trajectory', fontsize=10, color='#616161')
    
    # ─── Save ───
    out_dir = '/Users/shyam/Desktop/ai-proctoring-ml/paper_figures'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'evidence_flow.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Evidence flow diagram saved to {save_path}")

if __name__ == "__main__":
    generate_evidence_flow_diagram()

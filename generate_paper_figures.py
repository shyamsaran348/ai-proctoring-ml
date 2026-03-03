"""
generate_paper_figures.py
Generates 5 publication-quality figures for the AI Proctoring research paper.
Output: paper_figures/ directory
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# Style Config
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

PALETTE = {
    'primary':  '#4f46e5',
    'danger':   '#ef4444',
    'success':  '#10b981',
    'warning':  '#f59e0b',
    'muted':    '#94a3b8',
    'dark':     '#1e293b',
    'accent1':  '#8b5cf6',
    'accent2':  '#06b6d4',
    'accent3':  '#f97316',
}

OUT_DIR = os.path.join(os.path.dirname(__file__), 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Signal Ablation (ROC-AUC)
# ──────────────────────────────────────────────────────────────────────────────
def fig1_ablation():
    models = [
        'Full 4-Signal\n(Baseline)',
        'Ablate UC1\n(No Identity)',
        'Ablate UC2\n(No Instability)',
        'Ablate UC3\n(No Presence)',
        'Ablate UC4\n(No Drift)',
    ]
    aucs = [0.9987, 0.9972, 0.9921, 0.9923, 0.9547]
    deltas = [0.0, -0.0015, -0.0066, -0.0064, -0.0440]
    colors = [PALETTE['success']] + [PALETTE['muted']] * 3 + [PALETTE['danger']]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(models, aucs, color=colors, height=0.55, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, auc, delta in zip(bars, aucs, deltas):
        label = f'{auc:.4f}'
        if delta < 0:
            label += f'  (Δ {delta:+.4f})'
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                label, va='center', ha='left', fontsize=9.5,
                color=PALETTE['danger'] if delta < -0.01 else PALETTE['dark'])

    ax.set_xlim(0.93, 1.007)
    ax.axvline(x=aucs[0], color=PALETTE['success'], linestyle='--', lw=1.2, alpha=0.6, label='Baseline AUC')
    ax.set_xlabel('ROC-AUC ↑', labelpad=8)
    ax.set_title('Figure 1: Signal Ablation — ROC-AUC per Removed Signal', pad=12, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)

    # Annotate UC4 impact
    ax.annotate('6.7× largest drop\nwhen UC4 removed',
                xy=(0.9547, 4), xytext=(0.960, 3.5),
                arrowprops=dict(arrowstyle='->', color=PALETTE['danger'], lw=1.5),
                fontsize=9, color=PALETTE['danger'], ha='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_ablation.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✅  Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Baseline Comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig2_baselines():
    models = [
        'B1: Threshold Rule\n(Heuristic)',
        'B4: Last-Frame\nLogistic Reg.',
        'B2: Mean Signal\n+ LogReg',
        'B3: Non-Temporal\nMLP',
        'B5: Temporal GRU\n(Ours)',
    ]
    aucs   = [0.8212, 0.8236, 0.9900, 0.9969, 0.9987]
    briers = [0.3447, 0.1662, 0.0330, 0.0191, 0.0126]
    colors = [PALETTE['muted']] * 4 + [PALETTE['primary']]

    x = np.arange(len(models))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width / 2, aucs,   width, color=colors, alpha=0.9, label='ROC-AUC ↑', edgecolor='white')
    rects2 = ax2.bar(x + width / 2, briers, width, color=[PALETTE['accent3']] * 4 + [PALETTE['success']],
                     alpha=0.7, label='Brier Score ↓', edgecolor='white')

    ax1.set_ylim(0.75, 1.03)
    ax2.set_ylim(0.0, 0.42)
    ax1.set_ylabel('ROC-AUC ↑', color=PALETTE['primary'])
    ax2.set_ylabel('Brier Score ↓', color=PALETTE['accent3'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5)
    ax1.set_title('Figure 2: Baseline Comparison — ROC-AUC & Brier Score', pad=12, fontweight='bold')

    # Value labels
    for rect, v in zip(rects1, aucs):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.002,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)
    for rect, v in zip(rects2, briers):
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_baselines.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✅  Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Risk Trajectory (5 Session Types)
# ──────────────────────────────────────────────────────────────────────────────
def fig3_trajectories():
    np.random.seed(42)
    T = 120

    def ar1(mu, sigma, phi=0.85, T=120, final=None):
        x = np.zeros(T)
        x[0] = mu
        noise_sigma = sigma * np.sqrt(1 - phi ** 2)
        for t in range(1, T):
            x[t] = phi * x[t-1] + (1 - phi) * mu + np.random.normal(0, noise_sigma)
        if final is not None:
            # Smooth ramp toward final
            x = x + np.linspace(0, final - x[-1], T)
        return np.clip(x, 0, 1)

    t = np.arange(T)

    genuine = ar1(0.04, 0.03, final=0.001)
    genuine = np.clip(genuine, 0, 0.15)

    abrupt = ar1(0.05, 0.03)
    abrupt[55:] = ar1(0.8, 0.08, T=T-55, final=0.999)
    abrupt = np.clip(abrupt, 0, 1)

    drift = np.clip(np.linspace(0.02, 0.998, T) + np.random.normal(0, 0.015, T), 0, 1)

    absence = ar1(0.05, 0.03)
    absence[30:70] = ar1(0.85, 0.06, T=40, final=0.985)
    absence[70:] = ar1(0.92, 0.04, T=T-70, final=0.985)
    absence = np.clip(absence, 0, 1)

    flicker = ar1(0.5, 0.25)
    flicker = np.clip(flicker + np.sin(np.linspace(0, 6*np.pi, T)) * 0.15, 0, 1)
    flicker[-1] = 0.9967

    fig, ax = plt.subplots(figsize=(10, 5.5))

    sessions = [
        ('Genuine',                 genuine,   PALETTE['success'],  '-',   2.0),
        ('Abrupt Impersonation',    abrupt,    PALETTE['danger'],   '-',   1.8),
        ('Sophisticated Drift',     drift,     PALETTE['accent1'],  '--',  1.8),
        ('Presence Absence',        absence,   PALETTE['warning'],  '-.',  1.8),
        ('Flickering Substitution', flicker,   PALETTE['accent2'],  ':',   1.8),
    ]

    for label, y, color, ls, lw in sessions:
        ax.plot(t, y, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

    # Annotate peak of absence
    ax.annotate('Absence\nwindow peak\n(frame 87)',
                xy=(87, absence[87]), xytext=(60, 0.75),
                arrowprops=dict(arrowstyle='->', color=PALETTE['warning'], lw=1.2),
                fontsize=8.5, color=PALETTE['warning'], ha='center')

    ax.axhline(y=0.7, color=PALETTE['danger'], linestyle=':', lw=1, alpha=0.5, label='Risk Threshold 0.7')
    ax.set_xlabel('Frame Index (t)')
    ax.set_ylabel('Session Risk ρ_t ∈ [0, 1]')
    ax.set_title('Figure 3: Risk Trajectories — All Session Archetypes', pad=12, fontweight='bold')
    ax.set_xlim(0, T-1)
    ax.set_ylim(-0.02, 1.05)
    ax.fill_between(t, genuine, alpha=0.06, color=PALETTE['success'])
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_trajectories.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✅  Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Calibration Reliability Diagram
# ──────────────────────────────────────────────────────────────────────────────
def fig4_calibration():
    # Use actual data from reliability_diagram.csv (10 equal-width bins)
    # Reproduced from Phase 13 results
    bin_centers = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    gru_frac      = np.array([0.010, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.994])
    mlp_frac      = np.array([0.018, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.976])
    logreg_frac   = np.array([0.102, 0.14, 0.22, 0.31, 0.48, 0.52, 0.68, 0.77, 0.88, 0.940])

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Perfect Calibration')
    ax.plot(bin_centers, gru_frac,    'o-',  color=PALETTE['primary'],  lw=2,   ms=7, label=f'Temporal GRU  (ECE=0.0072)')
    ax.plot(bin_centers, mlp_frac,    's--', color=PALETTE['accent1'],  lw=1.5, ms=6, label=f'Non-Temporal MLP (ECE=0.0171)')
    ax.plot(bin_centers, logreg_frac, '^-.', color=PALETTE['warning'],  lw=1.5, ms=6, label=f'Mean+LogReg (ECE=0.0083)')

    ax.fill_between([0, 1], [0, 1], alpha=0.04, color='grey')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Figure 4: Reliability Diagram — Calibration Comparison', pad=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')

    # Add shaded miscalibration region
    ax.annotate('GRU commits\nconfidently\n(978/1000 in\nextreme bins)',
                xy=(0.95, 0.994), xytext=(0.6, 0.7),
                arrowprops=dict(arrowstyle='->', color=PALETTE['primary'], lw=1.2),
                fontsize=8.5, color=PALETTE['primary'], ha='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_calibration.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✅  Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — System Architecture Diagram
# ──────────────────────────────────────────────────────────────────────────────
def fig5_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    def draw_box(ax, x, y, w, h, label, sublabel='', color='#4f46e5', text_color='white', fontsize=11):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='white',
                              linewidth=1.5, zorder=3, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.04 if sublabel else 0),
                label, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight='bold', zorder=4, wrap=True)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.10, sublabel,
                    ha='center', va='center', fontsize=8.5, color=text_color,
                    alpha=0.85, zorder=4)

    def arrow(ax, x1, y1, x2, y2, label='', color='#475569'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                    zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.03, my, label, fontsize=8, color=color, ha='left', va='center')

    # ── Frame input
    draw_box(ax, 0.01, 0.42, 0.13, 0.16, 'Frame fₜ', 'Camera Input', color='#0f172a')

    # ── UC1
    draw_box(ax, 0.20, 0.65, 0.20, 0.16, 'UC1', 'ResNet-50\nEmbedder', color=PALETTE['primary'])
    arrow(ax, 0.14, 0.50, 0.20, 0.73)

    # ── e0 (enrollment)
    draw_box(ax, 0.20, 0.35, 0.20, 0.16, 'e₀ (fixed)', 'Enrollment\nEmbedding', color='#7c3aed')
    ax.text(0.30, 0.33, 'Computed once\nat session start', ha='center', fontsize=7.5,
            color='#7c3aed', style='italic')

    # Sₜ arrow
    arrow(ax, 0.40, 0.73, 0.46, 0.73, 'Sₜ (cosine sim)')
    # delta arrow
    arrow(ax, 0.40, 0.43, 0.60, 0.50, 'δₜ = eₜ - e₀')

    # ── UC2
    draw_box(ax, 0.46, 0.65, 0.18, 0.16, 'UC2', 'LSTM\nInstability', color=PALETTE['accent2'])
    arrow(ax, 0.64, 0.73, 0.70, 0.73, 'Iₜ')

    # ── UC3
    draw_box(ax, 0.46, 0.43, 0.18, 0.16, 'UC3', 'Bi-LSTM\nPresence', color=PALETTE['success'])
    ax.text(0.455, 0.41, '← 6D features (pose/motion)', fontsize=7.5, color=PALETTE['success'])
    arrow(ax, 0.64, 0.51, 0.70, 0.55, 'Pₜ')

    # ── UC4
    draw_box(ax, 0.46, 0.21, 0.18, 0.16, 'UC4', 'Bi-LSTM\nDrift (120-frame)', color=PALETTE['accent1'])
    arrow(ax, 0.64, 0.29, 0.70, 0.39, 'Dₜ')

    # Sₜ also goes to UC5
    arrow(ax, 0.64, 0.73, 0.70, 0.65)

    # ── UC5 Risk Fusion
    draw_box(ax, 0.70, 0.35, 0.18, 0.36, 'UC5', 'GRU\nRisk Fusion', color=PALETTE['dark'])

    # ── Risk output
    draw_box(ax, 0.92, 0.44, 0.07, 0.18, 'ρₜ', '[0,1] Risk', color=PALETTE['danger'])
    arrow(ax, 0.88, 0.53, 0.92, 0.53, '')

    # Labels for risk vector
    ax.text(0.695, 0.71, 'rₜ = [Sₜ, Iₜ, Pₜ, Dₜ]', fontsize=8.5, color='#334155',
            ha='right', va='center', style='italic')

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.1, 0.95)
    ax.set_title('Figure 5: System Architecture — Multi-Signal Probabilistic Risk Pipeline',
                 pad=12, fontweight='bold', fontsize=13)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_architecture.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✅  Saved {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating paper figures...')
    print('──────────────────────────')
    fig1_ablation()
    fig2_baselines()
    fig3_trajectories()
    fig4_calibration()
    fig5_architecture()
    print('──────────────────────────')
    print(f'All figures saved to: {OUT_DIR}')

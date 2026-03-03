"""
generate_architecture_diagrams.py
==================================
Generates two IEEE-style figures for the AI Proctoring research paper:

  fig6_system_architecture.png  — Technical architecture (UC1–UC5 data flow)
  fig7_workflow.png             — Conceptual phase-by-phase workflow

Output: paper_figures/
Run:    python3 generate_architecture_diagrams.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      9,
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
    'savefig.pad_inches': 0.15,
})

# Clean monochrome palette (IEEE-friendly)
C_DARK   = '#1a1a2e'   # near-black for text / heavy lines
C_MID    = '#374151'   # module box fill
C_LIGHT  = '#f9fafb'   # light box fill
C_BORDER = '#6b7280'   # box border
C_ACCENT = '#1d4ed8'   # blue for key data-flow arrows
C_NOTE   = '#dc2626'   # red for invariant annotations
C_ENROLL = '#14532d'   # dark-green for enrollment boxes
C_PHASE  = '#1e3a5f'   # phase header boxes

# ── Helper: draw a rounded-rect box with centred text ───────────────────────
def box(ax, cx, cy, w, h, text, fontsize=8.5, fc='#374151', ec='#6b7280',
        tc='white', lw=1.2, bold=False, style='round,pad=0.1',
        linestyle='solid'):
    bp = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style,
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(bp)
    weight = 'bold' if bold else 'normal'
    for i, line in enumerate(text.split('\n')):
        n = len(text.split('\n'))
        offset = (n - 1) / 2 * 0.095 - i * 0.095
        ax.text(cx, cy + offset, line,
                ha='center', va='center', fontsize=fontsize,
                color=tc, fontweight=weight, zorder=4)


def arr(ax, x1, y1, x2, y2, color=C_ACCENT, lw=1.4,
        label='', label_side='right', arrowstyle='->', head_w=8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    color=color, lw=lw,
                    mutation_scale=head_w,
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = 0.06 if label_side == 'right' else -0.06
        ax.text(mx + dx, my, label,
                ha='left' if label_side == 'right' else 'right',
                va='center', fontsize=7.2, color=C_DARK,
                style='italic', zorder=5)


def separator(ax, y, xmin, xmax, label='', lc='#d1d5db'):
    ax.plot([xmin, xmax], [y, y], color=lc, lw=0.8, linestyle='--', zorder=1)
    if label:
        ax.text(xmin + 0.01, y + 0.015, label,
                fontsize=7, color=C_BORDER, ha='left', va='bottom')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — System Architecture (Technical View)
# ════════════════════════════════════════════════════════════════════════════
def fig6_architecture():
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Title ────────────────────────────────────────────────────────
    ax.text(0.5, 0.975, 'System Architecture — Multi-Signal Probabilistic Risk Accumulation Engine',
            ha='center', va='top', fontsize=11, fontweight='bold', color=C_DARK)

    # ── Separators / zones ───────────────────────────────────────────
    separator(ax, 0.845, 0.02, 0.98, label='— ENROLLMENT PHASE (executed once) —')
    separator(ax, 0.60,  0.02, 0.98, label='— PER-FRAME PROCESSING (each frame fₜ) —')
    separator(ax, 0.285, 0.02, 0.98, label='— TEMPORAL MODELING —')
    separator(ax, 0.12,  0.02, 0.98, label='— RISK FUSION —')

    # ── ENROLLMENT ───────────────────────────────────────────────────
    box(ax, 0.50, 0.920, 0.22, 0.070,
        'Enrollment Image\n(captured once)', fc=C_ENROLL, ec='#166534',
        fontsize=8.5, bold=True)

    box(ax, 0.50, 0.855, 0.24, 0.055,
        'ResNet-50 Embedder  (shared weights)', fc='#1e3a5f', ec='#1d4ed8',
        fontsize=8)

    # enrollment embedding — wide, highlighted
    box(ax, 0.50, 0.790, 0.44, 0.060,
        'e₀  —  Immutable One-Shot Enrollment Embedding  (L2-normalised, 256-dim)',
        fc='#7f1d1d', ec=C_NOTE, fontsize=8.5, bold=True, lw=1.6)

    # annotation: invariant
    ax.text(0.96, 0.790,
            '← Never updated\n   Never averaged\n   Never thresholded',
            ha='left', va='center', fontsize=7, color=C_NOTE,
            style='italic', zorder=5)

    arr(ax, 0.50, 0.885, 0.50, 0.865)
    arr(ax, 0.50, 0.832, 0.50, 0.822)

    # ── PER-FRAME PROCESSING ─────────────────────────────────────────
    box(ax, 0.18, 0.720, 0.20, 0.060,
        'Live Frame  fₜ\n(webcam input)', fc=C_ENROLL, ec='#166534', fontsize=8)

    box(ax, 0.18, 0.648, 0.24, 0.055,
        'ResNet-50 Embedder  (shared weights)', fc='#1e3a5f', ec='#1d4ed8', fontsize=8)

    box(ax, 0.18, 0.615, 0.18, 0.036,
        'eₜ  (probe embedding)', fc='#1e3a5f', ec='#1d4ed8',
        fontsize=7.5, bold=False)

    arr(ax, 0.18, 0.690, 0.18, 0.675)
    arr(ax, 0.18, 0.621, 0.18, 0.605)   # eₜ downward

    # δₜ = eₜ − e₀
    box(ax, 0.55, 0.615, 0.22, 0.036,
        'δₜ  =  eₜ  −  e₀     (embedding delta)', fc='#312e81', ec='#4338ca',
        fontsize=7.5)
    # e₀ tapped from enrollment box
    ax.annotate('', xy=(0.44, 0.615), xytext=(0.44, 0.760),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.0,
                                connectionstyle='arc3,rad=0.0'), zorder=2)
    arr(ax, 0.27, 0.615, 0.44, 0.615,
        label='Sₜ = eₜ · e₀', label_side='right')
    # separate δₜ branch
    ax.plot([0.27, 0.44], [0.612, 0.612], color='#6b7280', lw=0.7, linestyle=':', zorder=1)

    # UC3 6D features (independent)
    box(ax, 0.82, 0.648, 0.26, 0.060,
        '6D Presence Features\n[face_conf, area, yaw, pitch, roll, motion]',
        fc='#064e3b', ec='#065f46', fontsize=7.8)
    arr(ax, 0.82, 0.688, 0.82, 0.618, label='(per-frame)', label_side='right')

    # ── TEMPORAL MODELS ──────────────────────────────────────────────
    # UC2
    box(ax, 0.18, 0.470, 0.26, 0.080,
        'UC2\nShort-Term Identity Instability\nLSTM  |  window W',
        fc='#78350f', ec='#b45309', fontsize=8, bold=True)
    arr(ax, 0.18, 0.597, 0.18, 0.512,
        label=' Sₜ  (similarity\n  sequence)', label_side='right')

    # UC4
    box(ax, 0.50, 0.470, 0.26, 0.080,
        'UC4\nLong-Term Embedding Drift\nBi-LSTM  |  120-frame buffer',
        fc='#312e81', ec='#4338ca', fontsize=8, bold=True)
    arr(ax, 0.55, 0.597, 0.50, 0.512,
        label=' δₜ  (embedding\n  delta, 120-frame)', label_side='right')

    # UC3
    box(ax, 0.82, 0.470, 0.26, 0.080,
        'UC3\nPresence & Attentiveness\nBi-LSTM  |  window W',
        fc='#064e3b', ec='#065f46', fontsize=8, bold=True)
    arr(ax, 0.82, 0.618, 0.82, 0.512,
        label=' 6D features\n  (window)', label_side='right')

    # Output labels under each UC
    box(ax, 0.18, 0.380, 0.14, 0.038,
        'Iₜ   Instability ∈ [0,1]', fc='#1c1917', ec='#78350f', fontsize=7.5)
    arr(ax, 0.18, 0.430, 0.18, 0.400)

    box(ax, 0.50, 0.380, 0.14, 0.038,
        'Dₜ   Drift ∈ [0,1]', fc='#1c1917', ec='#4338ca', fontsize=7.5)
    arr(ax, 0.50, 0.430, 0.50, 0.400)

    box(ax, 0.82, 0.380, 0.14, 0.038,
        'Pₜ   Presence ∈ [0,1]', fc='#1c1917', ec='#065f46', fontsize=7.5)
    arr(ax, 0.82, 0.430, 0.82, 0.400)

    # Sₜ direct to UC5
    box(ax, 0.50, 0.310, 0.50, 0.042,
        'Risk Vector   rₜ  =  [ Sₜ   Iₜ   Pₜ   Dₜ ]   ∈  ℝ⁴      (no thresholding)',
        fc='#0f172a', ec='#475569', fontsize=8.5, bold=True)
    arr(ax, 0.18, 0.361, 0.26, 0.310)
    arr(ax, 0.50, 0.361, 0.50, 0.332)
    arr(ax, 0.82, 0.361, 0.74, 0.310)
    # Sₜ drop from similarity computation
    ax.annotate('', xy=(0.30, 0.311), xytext=(0.30, 0.597),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=0.9,
                                linestyle='dashed', connectionstyle='arc3,rad=0.0'), zorder=2)
    ax.text(0.305, 0.450, 'Sₜ', fontsize=7, color='#9ca3af', style='italic')

    # ── RISK FUSION ──────────────────────────────────────────────────
    box(ax, 0.50, 0.215, 0.38, 0.072,
        'UC5 — GRU Risk Fusion\nSession-level BCE supervision  |  hidden dim = 32',
        fc='#1a1a2e', ec='#94a3b8', fontsize=9, bold=True, lw=1.8)
    arr(ax, 0.50, 0.289, 0.50, 0.251)

    box(ax, 0.50, 0.138, 0.30, 0.055,
        'Risk Trajectory   ρₜ  ∈  [0, 1]', fc='#1e293b', ec='#38bdf8',
        fontsize=8.5, bold=False)
    arr(ax, 0.50, 0.179, 0.50, 0.167)

    box(ax, 0.50, 0.065, 0.34, 0.058,
        'Final Session Risk   ρ_T\n(session-level probabilistic output)',
        fc='#7f1d1d', ec=C_NOTE, fontsize=9, bold=True, lw=1.8)
    arr(ax, 0.50, 0.110, 0.50, 0.095)

    # ── Invariant annotations (right margin) ─────────────────────────
    ann_x = 0.99
    ax.text(ann_x, 0.470, 'Temporal Modeling\nat All Layers',
            ha='right', va='center', fontsize=7, color=C_NOTE, style='italic')
    ax.text(ann_x, 0.215, 'Session-Level\nSupervision Only',
            ha='right', va='center', fontsize=7, color=C_NOTE, style='italic')
    ax.text(ann_x, 0.065, 'Continuous Probability\n— No Decision Threshold —',
            ha='right', va='center', fontsize=7.5, color=C_NOTE,
            style='italic', fontweight='bold')

    # UC1 label
    ax.text(0.18, 0.958, 'UC1', ha='center', va='bottom',
            fontsize=8, color='#6b7280', fontweight='bold')

    plt.tight_layout(pad=0.3)
    path = os.path.join(OUT_DIR, 'fig6_system_architecture.png')
    plt.savefig(path, facecolor='white')
    plt.close()
    print(f'  ✅  Saved {path}')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — System Workflow (Conceptual View)
# ════════════════════════════════════════════════════════════════════════════
def fig7_workflow():
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(0.5, 0.975, 'System Workflow — Enrollment, Monitoring, and Risk Accumulation',
            ha='center', va='top', fontsize=11, fontweight='bold', color=C_DARK)

    # ── Column config ────────────────────────────────────────────────
    # 3 columns: Phase 1 | Phase 2 | Phase 3
    col = [0.18, 0.52, 0.85]
    bw, bh = 0.28, 0.058

    # ── Phase headers ────────────────────────────────────────────────
    for cx, label, sub in [
        (col[0], 'Phase 1', 'ENROLLMENT'),
        (col[1], 'Phase 2', 'LIVE MONITORING'),
        (col[2], 'Phase 3', 'SESSION END'),
    ]:
        box(ax, cx, 0.920, 0.27, 0.064,
            f'{label}\n{sub}',
            fc=C_PHASE, ec='#1d4ed8', fontsize=9.5, bold=True, lw=1.6)

    # ── Phase 1 (Enrollment) ─────────────────────────────────────────
    steps_p1 = [
        (0.840, 'Capture single\nenrollment image'),
        (0.755, 'UC1: ResNet-50\nextract embedding'),
        (0.670, 'e₀ — immutable embedding\n(L2-normalised, 256-dim)'),
        (0.585, 'Store e₀ as\nsession anchor\n(never updated)'),
    ]
    fcs_p1 = [C_ENROLL, '#1e3a5f', '#7f1d1d', '#7f1d1d']
    ecs_p1 = ['#166534', '#1d4ed8', C_NOTE, C_NOTE]
    for (cy, txt), fc, ec in zip(steps_p1, fcs_p1, ecs_p1):
        box(ax, col[0], cy, bw, bh, txt, fc=fc, ec=ec, fontsize=8)

    # arrows p1
    for i in range(len(steps_p1) - 1):
        y1 = steps_p1[i][0] - bh / 2
        y2 = steps_p1[i + 1][0] + bh / 2
        arr(ax, col[0], y1, col[0], y2)

    # ── Phase 2 (Live Monitoring) ────────────────────────────────────
    steps_p2 = [
        (0.840, 'Receive frame  fₜ'),
        (0.752, 'UC1: extract eₜ\ncompute  Sₜ = eₜ · e₀\ncompute  δₜ = eₜ − e₀'),
        (0.650, 'Extract\n6D presence features'),
        (0.560, 'UC2:  instability  Iₜ\n(window of Sₜ values)'),
        (0.465, 'UC4:  drift  Dₜ\n(120-frame δₜ buffer)'),
        (0.370, 'UC3:  presence  Pₜ\n(window of 6D features)'),
        (0.280, 'Assemble risk vector\nrₜ = [Sₜ, Iₜ, Pₜ, Dₜ]'),
        (0.185, 'UC5 GRU: update\nρₜ from rₜ and h_{t-1}'),
    ]
    fcs_p2 = [C_ENROLL, '#1e3a5f', '#064e3b',
              '#78350f', '#312e81', '#064e3b',
              '#0f172a', '#0f172a']
    ecs_p2 = ['#166534', '#1d4ed8', '#065f46',
              '#b45309', '#4338ca', '#065f46',
              '#475569', '#94a3b8']
    bh2 = [0.052, 0.078, 0.052, 0.058, 0.058, 0.058, 0.052, 0.058]
    for (cy, txt), fc, ec, h in zip(steps_p2, fcs_p2, ecs_p2, bh2):
        box(ax, col[1], cy, bw, h, txt, fc=fc, ec=ec, fontsize=8)

    for i in range(len(steps_p2) - 1):
        h_top = bh2[i]
        h_bot = bh2[i + 1]
        y1 = steps_p2[i][0] - h_top / 2
        y2 = steps_p2[i + 1][0] + h_bot / 2
        arr(ax, col[1], y1, col[1], y2)

    # Loop back arrow: "Repeat for t+1"
    ax.annotate('',
                xy=(col[1] + 0.15, 0.840),
                xytext=(col[1] + 0.15, 0.185 - 0.029),
                arrowprops=dict(arrowstyle='->', color='#6b7280', lw=0.9,
                                connectionstyle='arc3,rad=0.0'), zorder=2)
    ax.text(col[1] + 0.165, 0.510, 'repeat for\nt+1 … T',
            ha='left', va='center', fontsize=7, color='#6b7280', style='italic')

    # ── Phase 3 (Session End) ────────────────────────────────────────
    steps_p3 = [
        (0.840, 'Session ends\n(t = T)'),
        (0.740, 'Final output:\nρ_T  ∈  [0, 1]'),
        (0.630, 'Continuous probability\n— no threshold applied —'),
        (0.520, 'Risk score forwarded\nto human review'),
        (0.400, 'Session risk log\nstored for audit'),
    ]
    fcs_p3 = [C_ENROLL, '#7f1d1d', '#7f1d1d', '#1e3a5f', '#1e3a5f']
    ecs_p3 = ['#166534', C_NOTE, C_NOTE, '#1d4ed8', '#1d4ed8']
    bh3 = 0.070
    for (cy, txt), fc, ec in zip(steps_p3, fcs_p3, ecs_p3):
        box(ax, col[2], cy, bw, bh3, txt, fc=fc, ec=ec, fontsize=8)

    for i in range(len(steps_p3) - 1):
        y1 = steps_p3[i][0] - bh3 / 2
        y2 = steps_p3[i + 1][0] + bh3 / 2
        arr(ax, col[2], y1, col[2], y2)

    # ── Cross-phase arrows ───────────────────────────────────────────
    # e₀ flows into Phase 2
    ax.annotate('',
                xy=(col[1] - 0.14, 0.752),
                xytext=(col[0] + 0.14, 0.670),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.0,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=-0.2'), zorder=2)
    ax.text((col[0] + col[1]) / 2 - 0.01, 0.725,
            'e₀', ha='center', va='center', fontsize=7.5,
            color='#9ca3af', style='italic')

    # Phase 2 final risk → Phase 3
    ax.annotate('',
                xy=(col[2] - 0.14, 0.840),
                xytext=(col[1] + 0.14, 0.185),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.0,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=-0.3'), zorder=2)
    ax.text((col[1] + col[2]) / 2 + 0.02, 0.530,
            'ρ_T', ha='center', va='center', fontsize=7.5,
            color='#9ca3af', style='italic')

    # ── Legend ───────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(fc=C_ENROLL,   ec='#166534', label='Input / Phase boundary'),
        mpatches.Patch(fc='#1e3a5f',  ec='#1d4ed8', label='Feature extraction (UC1)'),
        mpatches.Patch(fc='#78350f',  ec='#b45309', label='Short-window model (UC2)'),
        mpatches.Patch(fc='#312e81',  ec='#4338ca', label='Long-window model (UC4)'),
        mpatches.Patch(fc='#064e3b',  ec='#065f46', label='Presence model (UC3)'),
        mpatches.Patch(fc='#7f1d1d',  ec=C_NOTE,    label='Risk (UC5) / Invariant'),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              fontsize=7.5, framealpha=0.9, edgecolor='#e5e7eb',
              ncol=2, title='Module Type', title_fontsize=7.5)

    plt.tight_layout(pad=0.3)
    path = os.path.join(OUT_DIR, 'fig7_workflow.png')
    plt.savefig(path, facecolor='white')
    plt.close()
    print(f'  ✅  Saved {path}')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating architecture & workflow diagrams...')
    print('──────────────────────────────────────────────')
    fig6_architecture()
    fig7_workflow()
    print('──────────────────────────────────────────────')
    print(f'Saved to: {OUT_DIR}')

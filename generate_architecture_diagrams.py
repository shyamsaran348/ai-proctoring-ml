"""
generate_architecture_diagrams.py  (v2 — IEEE Clean Style)
============================================================
Generates two clean, IEEE-style block diagrams:

  fig6_system_architecture.png  — Technical architecture
  fig7_workflow.png             — Conceptual workflow

Style: white fill, black borders, gray arrows, no colors.
Run:   python3 generate_architecture_diagrams.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          9,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, lines, fontsize=8.5, bold_first=False,
        lw=1.0, ls='solid', fc='white', ec='black'):
    """Draw a plain white rectangle with centred text (supports multi-line)."""
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='square,pad=0',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=3)
    ax.add_patch(rect)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        offset = ((n - 1) / 2 - i) * (h / (n + 1))
        weight = 'bold' if (bold_first and i == 0) else 'normal'
        ax.text(cx, cy + offset, line,
                ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)


def arrow(ax, x1, y1, x2, y2, lw=1.1, label='', lpos='right'):
    """Draw a simple annotate-arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=lw, mutation_scale=9),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx = 0.05 if lpos == 'right' else -0.05
        ax.text(mx+dx, my, label, ha='left' if lpos=='right' else 'right',
                va='center', fontsize=7.5, style='italic', color='#333333', zorder=5)


def dblarrow(ax, x1, y1, x2, y2, lw=1.1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=lw, mutation_scale=9),
                zorder=2)


def hline(ax, y, x0, x1, lw=0.7, ls='--'):
    ax.plot([x0, x1], [y, y], color='#aaaaaa', lw=lw, linestyle=ls, zorder=1)


# ════════════════════════════════════════════════════════════════════════════
# FIG 6: SYSTEM ARCHITECTURE  (two-column block style)
# ════════════════════════════════════════════════════════════════════════════
def fig6_architecture():
    W, H = 10, 11
    fig, ax = plt.subplots(figsize=(W/1.1, H/1.1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(W/2, H - 0.18,
            'System Architecture: Multi-Signal Temporal Risk Accumulation Engine',
            ha='center', va='top', fontsize=10, fontweight='bold')

    # ── Layout constants ─────────────────────────────────────────────
    # Two columns
    Lx, Rx = 2.5, 7.5   # left col center, right col center
    bw, bh  = 3.6, 0.62  # default box width / height

    # ── Left column: ENROLLMENT PATH ────────────────────────────────
    # zone label
    ax.text(Lx, H-0.55, 'Enrollment (one-shot)', ha='center',
            fontsize=8, style='italic', color='#555555')
    hline(ax, H-0.65, 0.3, 4.7)

    L1y = H - 1.2
    box(ax, Lx, L1y, bw, bh, 'Enrollment Image  f_enroll')
    arrow(ax, Lx, L1y - bh/2, Lx, L1y - bh/2 - 0.35)

    L2y = L1y - bh - 0.35
    box(ax, Lx, L2y, bw, bh, 'ResNet-50 Feature Extractor  (UC1)')
    arrow(ax, Lx, L2y - bh/2, Lx, L2y - bh/2 - 0.35)

    L3y = L2y - bh - 0.35
    box(ax, Lx, L3y, bw, bh,
        ['e\u2080  \u2014  Enrollment Embedding',
         '(256-dim, L2-norm, immutable)'],
        lw=1.6, bold_first=True)

    # dashed border annotation
    ax.text(Lx, L3y - bh/2 - 0.22, '\u2014 never updated, never thresholded \u2014',
            ha='center', fontsize=7.5, style='italic', color='#555555')

    # ── Right column: PER-FRAME PATH ─────────────────────────────────
    ax.text(Rx, H-0.55, 'Per-Frame Processing  (t = 1 \u2026 T)', ha='center',
            fontsize=8, style='italic', color='#555555')
    hline(ax, H-0.65, 5.3, 9.7)

    R1y = H - 1.2
    box(ax, Rx, R1y, bw, bh, 'Live Frame  f\u209c  (webcam input)')
    arrow(ax, Rx, R1y - bh/2, Rx, R1y - bh/2 - 0.35)

    R2y = R1y - bh - 0.35
    box(ax, Rx, R2y, bw, bh, 'ResNet-50 Feature Extractor  (UC1)')
    arrow(ax, Rx, R2y - bh/2, Rx, R2y - bh/2 - 0.35)

    R3y = R2y - bh - 0.35
    box(ax, Rx, R3y, bw, bh,
        ['Probe Embedding  e\u209c',
         'Similarity  S\u209c = e\u209c \u00b7 e\u2080       Delta  \u03b4\u209c = e\u209c \u2212 e\u2080'],
        bold_first=True)

    # e0 →  R3y  (cross-column arrow for similarity / delta)
    ax.annotate('', xy=(Rx - bw/2 - 0.05, R3y), xytext=(Lx + bw/2 + 0.05, L3y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.1, mutation_scale=9),
                zorder=2)
    ax.text((Lx + Rx)/2, (L3y + R3y)/2 + 0.12, 'e\u2080 (fixed reference)',
            ha='center', fontsize=7.5, style='italic', color='#333333')

    # ── Temporal model zone ─────────────────────────────────────────
    Ty = R3y - bh/2 - 0.5   # top of temporal zone
    hline(ax, Ty + 0.3, 0.3, 9.7, ls='--')
    ax.text(W/2, Ty + 0.42, 'Temporal Modeling', ha='center',
            fontsize=8, style='italic', color='#555555')

    # Three temporal models side by side
    UC2x, UC3x, UC4x = 1.8, 5.0, 8.2
    UCy = Ty - 0.28
    small_bw = 2.7

    box(ax, UC2x, UCy, small_bw, 0.92,
        ['UC2  \u2014  Identity Instability', 'LSTM  |  window W'],
        fontsize=8, bold_first=True)

    box(ax, UC3x, UCy, small_bw, 0.92,
        ['UC3  \u2014  Presence & Attentiveness', 'Bi-LSTM  |  6D features'],
        fontsize=8, bold_first=True)

    box(ax, UC4x, UCy, small_bw, 0.92,
        ['UC4  \u2014  Embedding Drift', 'Bi-LSTM  |  120-frame buffer'],
        fontsize=8, bold_first=True)

    # Arrows from probe box to temporal models
    # S_t → UC2
    ax.annotate('', xy=(UC2x, UCy + 0.92/2),
                xytext=(Rx - 0.3, R3y - bh/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0,
                                mutation_scale=8,
                                connectionstyle='arc3,rad=0.15'), zorder=2)
    ax.text(UC2x + 0.1, UCy + 0.92/2 + 0.2, 'S\u209c', fontsize=8, style='italic')

    # 6D feats → UC3  (straight down from right col)
    ax.annotate('', xy=(UC3x, UCy + 0.92/2),
                xytext=(Rx, R3y - bh/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0,
                                mutation_scale=8,
                                connectionstyle='arc3,rad=0.05'), zorder=2)
    ax.text(UC3x + 0.08, UCy + 0.92/2 + 0.22, '6D features', fontsize=7.5, style='italic')

    # δ_t → UC4
    ax.annotate('', xy=(UC4x, UCy + 0.92/2),
                xytext=(Rx + 0.3, R3y - bh/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0,
                                mutation_scale=8,
                                connectionstyle='arc3,rad=-0.15'), zorder=2)
    ax.text(UC4x - 0.05, UCy + 0.92/2 + 0.2, '\u03b4\u209c', fontsize=8, style='italic')

    # Outputs: I_t, P_t, D_t
    OUTy = UCy - 0.92/2 - 0.45
    for ucx, lbl in [(UC2x, 'I\u209c'), (UC3x, 'P\u209c'), (UC4x, 'D\u209c')]:
        arrow(ax, ucx, UCy - 0.92/2, ucx, OUTy + 0.18)
        ax.text(ucx, OUTy, lbl, ha='center', va='center',
                fontsize=9, style='italic', fontweight='bold')

    # ── Risk fusion zone ─────────────────────────────────────────────
    RFy = OUTy - 0.55
    hline(ax, OUTy - 0.28, 0.3, 9.7, ls='--')
    ax.text(W/2, OUTy - 0.19, 'Risk Fusion', ha='center',
            fontsize=8, style='italic', color='#555555')

    box(ax, W/2, RFy, 6.5, 0.72,
        ['UC5  \u2014  GRU Risk Fusion',
         'Input: r\u209c = [ S\u209c,  I\u209c,  P\u209c,  D\u209c ]   |   Session-level BCE supervision only'],
        lw=1.6, bold_first=True)

    # Fan-in arrows from I_t, P_t, D_t to UC5
    for ucx in [UC2x, UC3x, UC4x]:
        ax.annotate('', xy=(W/2 + (ucx - W/2)*0.55, RFy + 0.72/2),
                    xytext=(ucx, OUTy - 0.20),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                                    mutation_scale=8), zorder=2)
    # Also S_t direct to UC5
    ax.annotate('', xy=(W/2 - 2.5, RFy + 0.72/2),
                xytext=(Rx - 1.0, R3y - bh/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                                mutation_scale=8,
                                connectionstyle='arc3,rad=0.25'), zorder=2)
    ax.text(W/2 - 2.8, RFy + 0.72/2 + 0.12, 'S\u209c', fontsize=8, style='italic')

    # Output risk
    RTy = RFy - 0.72/2 - 0.5
    arrow(ax, W/2, RFy - 0.72/2, W/2, RTy + 0.30)
    box(ax, W/2, RTy, 5.2, 0.60,
        ['\u03c1\u209c  \u2014  Session Risk Trajectory  \u2208  [0, 1]',
         'Final output: \u03c1_T  (continuous probability, no threshold)'],
        lw=1.6, bold_first=True)

    plt.tight_layout(pad=0.4)
    path = os.path.join(OUT_DIR, 'fig6_system_architecture.png')
    plt.savefig(path, facecolor='white')
    plt.close()
    print(f'  \u2705  Saved {path}')


# ════════════════════════════════════════════════════════════════════════════
# FIG 7: SYSTEM WORKFLOW  (vertical block flow — StudyCorgi style)
# ════════════════════════════════════════════════════════════════════════════
def fig7_workflow():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(3, 8.82,
            'System Workflow: Enrollment, Monitoring, and Risk Accumulation',
            ha='center', va='top', fontsize=9.5, fontweight='bold')

    # ── Boxes (top → bottom) ────────────────────────────────────────
    cx  = 3.0
    bw  = 4.8
    bh  = 0.68
    gap = 0.40

    items = [
        # (y-center, lines, bold_first, thick_border)
        (8.10, ['PHASE 1 — ENROLLMENT'],                              True,  True),
        (7.35, ['Capture single reference image'],                    False, False),
        (6.65, ['UC1: ResNet-50 extract embedding'],                  False, False),
        (5.95, ['e\u2080  \u2014  Immutable Enrollment Embedding',
                '(stored as fixed session anchor)'],                  True,  True),
        (5.12, ['PHASE 2 — LIVE MONITORING  (per frame f\u209c)'],   True,  True),
        (4.37, ['UC1: Extract probe embedding e\u209c',
                'Compute  S\u209c = e\u209c \u00b7 e\u2080   |   \u03b4\u209c = e\u209c \u2212 e\u2080'],  True, False),
        (3.60, ['Extract 6D presence features',
                '[face confidence, area, yaw, pitch, roll, motion]'], False, False),
        (2.82, ['UC2: Instability I\u209c  |  UC3: Presence P\u209c  |  UC4: Drift D\u209c',
                'Short-window LSTM       6D Bi-LSTM        120-frame Bi-LSTM'],       True, False),
        (2.04, ['UC5 GRU: update session risk \u03c1\u209c',
                'Input: r\u209c = [ S\u209c,  I\u209c,  P\u209c,  D\u209c ]'],          True,  False),
        (1.13, ['PHASE 3 — SESSION END'],                             True,  True),
        (0.48, ['Final session risk \u03c1_T  \u2208  [0, 1]',
                'Continuous probability \u2014 no decision threshold applied'],       True,  True),
    ]

    for (cy, lines, bf, thick) in items:
        h = bh * 1.22 if len(lines) > 1 else bh
        box(ax, cx, cy, bw, h, lines, bold_first=bf, lw=1.8 if thick else 1.0,
            fontsize=8.5 if not thick else 9.0)

    # ── Arrows between consecutive boxes ────────────────────────────
    # Compute bottom/top edges
    ys = [cy for (cy, _, _, _) in items]
    hs = [bh*1.22 if len(lines)>1 else bh for (_, lines, _, _) in items]

    for i in range(len(ys)-1):
        bot = ys[i]   - hs[i]/2
        top = ys[i+1] + hs[i+1]/2
        mid = (bot + top) / 2
        arrow(ax, cx, bot, cx, top)

    # ── Phase bracket annotations (left side) ───────────────────────
    for y0, y1, label in [
        (8.44, 7.95, 'Phase 1'),
        (5.47, 5.45, 'Enrollment'),
        (4.74, 1.67, 'Phase 2'),
        (1.47, 0.13, 'Phase 3'),
    ]:
        pass   # keep diagram clean — phase headers in boxes are enough

    # ── Loop annotation ─────────────────────────────────────────────
    # Right-side brace showing the monitoring loop
    loop_top = items[5][0] + hs[5]/2 + 0.05   # top of per-frame section
    loop_bot = items[8][0] - hs[8]/2 - 0.05
    bx = cx + bw/2 + 0.18

    ax.plot([bx, bx+0.22, bx+0.22, bx],
            [loop_top, loop_top, loop_bot, loop_bot],
            color='black', lw=0.9, zorder=2)
    ax.text(bx + 0.28, (loop_top+loop_bot)/2,
            'repeat for\nt+1 \u2026 T',
            ha='left', va='center', fontsize=7.5, style='italic', color='#333333')

    plt.tight_layout(pad=0.4)
    path = os.path.join(OUT_DIR, 'fig7_workflow.png')
    plt.savefig(path, facecolor='white')
    plt.close()
    print(f'  \u2705  Saved {path}')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating IEEE-style diagrams...')
    print('─' * 40)
    fig6_architecture()
    fig7_workflow()
    print('─' * 40)
    print(f'Output: {OUT_DIR}')

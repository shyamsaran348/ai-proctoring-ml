"""
generate_architecture_diagrams.py (v3 — Classic IEEE Academic Style)
======================================================================
Generates two high-quality, classic IEEE-style research diagrams:

  fig6_system_architecture.png — Layered architecture with dashed
                                 grouping boxes and distinct shapes.
  fig7_workflow.png            — Formal flowchart with decision diamonds,
                                 data cylinders, and orthogonal flow.

Style: Serif fonts, orthogonal routing, distinct semantic shapes.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.patches import FancyArrowPatch

# Force serif font for academic look
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shape Drawing Helpers ────────────────────────────────────────────────────

def text_lines(ax, cx, cy, text, fontsize, bold=False):
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=5)

def draw_rect(ax, cx, cy, w, h, text, fc='white', ec='black', lw=1.2, ls='solid', fontsize=8.5):
    rect = mpatches.Rectangle((cx - w/2, cy - h/2), w, h, fill=True,
                              facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3)
    ax.add_patch(rect)
    text_lines(ax, cx, cy, text, fontsize)
    return cx, cy - h/2, cx, cy + h/2

def draw_rounded_rect(ax, cx, cy, w, h, text, fc='white', ec='black', lw=1.2, fontsize=8.5, bold=False):
    box = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    text_lines(ax, cx, cy, text, fontsize, bold=bold)

def draw_cylinder(ax, cx, cy, w, h, text, fc='white', ec='black', lw=1.2, fontsize=8):
    # Cylinder: bottom half-ellipse, body, top full-ellipse
    eh = h * 0.15 # ellipse height
    bot = mpatches.Arc((cx, cy - h/2), w, eh, angle=0, theta1=180, theta2=360,
                       linewidth=lw, color=ec, zorder=4)
    ax.add_patch(bot)
    
    body = mpatches.Rectangle((cx - w/2, cy - h/2), w, h, fill=True,
                              facecolor=fc, edgecolor='none', zorder=3)
    ax.add_patch(body)
    
    # Body borders
    ax.plot([cx - w/2, cx - w/2], [cy - h/2, cy + h/2], color=ec, lw=lw, zorder=4)
    ax.plot([cx + w/2, cx + w/2], [cy - h/2, cy + h/2], color=ec, lw=lw, zorder=4)
    
    top = mpatches.Ellipse((cx, cy + h/2), w, eh, facecolor=fc, edgecolor=ec, lw=lw, zorder=4)
    ax.add_patch(top)
    
    text_lines(ax, cx, cy, text, fontsize)

def draw_diamond(ax, cx, cy, w, h, text, fc='white', ec='black', lw=1.2, fontsize=8):
    path_data = [
        (mpath.Path.MOVETO, (cx, cy + h/2)),
        (mpath.Path.LINETO, (cx + w/2, cy)),
        (mpath.Path.LINETO, (cx, cy - h/2)),
        (mpath.Path.LINETO, (cx - w/2, cy)),
        (mpath.Path.CLOSEPOLY, (cx, cy + h/2)),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=fc, edgecolor=ec, lw=lw, zorder=3)
    ax.add_patch(patch)
    text_lines(ax, cx, cy, text, fontsize)

def draw_parallelogram(ax, cx, cy, w, h, text, fc='white', ec='black', lw=1.2, fontsize=8):
    offset = w * 0.15
    path_data = [
        (mpath.Path.MOVETO, (cx - w/2 + offset, cy + h/2)),
        (mpath.Path.LINETO, (cx + w/2, cy + h/2)),
        (mpath.Path.LINETO, (cx + w/2 - offset, cy - h/2)),
        (mpath.Path.LINETO, (cx - w/2, cy - h/2)),
        (mpath.Path.CLOSEPOLY, (cx - w/2 + offset, cy + h/2)),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=fc, edgecolor=ec, lw=lw, zorder=3)
    ax.add_patch(patch)
    text_lines(ax, cx, cy, text, fontsize)

def draw_oval(ax, cx, cy, w, h, text, fc='white', ec='black', lw=1.2, fontsize=8):
    oval = mpatches.Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=3)
    ax.add_patch(oval)
    text_lines(ax, cx, cy, text, fontsize, bold=True)

def draw_group_box(ax, cx, cy, w, h, title):
    rect = mpatches.Rectangle((cx - w/2, cy - h/2), w, h, fill=False,
                              edgecolor='#555555', linewidth=1.2, linestyle='--', zorder=1)
    ax.add_patch(rect)
    # Title with background to cover dashed line
    ax.text(cx - w/2 + 0.2, cy + h/2, title, ha='left', va='center',
            fontsize=9, fontweight='bold', color='#333333',
            bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=2)

def orth_arrow(ax, p1, p2, lw=1.2, label='', lpos='right', text_offset=(0,0)):
    x1, y1 = p1
    x2, y2 = p2
    
    # Determine orthogonal routing
    if abs(x1 - x2) < 0.05:
        pth = [(x1, y1), (x2, y2)]
    elif abs(y1 - y2) < 0.05:
        pth = [(x1, y1), (x2, y2)]
    else:
        # Default route: vertical then horizontal
        pth = [(x1, y1), (x1, (y1+y2)/2), (x2, (y1+y2)/2), (x2, y2)]
        
    x_coords, y_coords = zip(*pth)
    ax.plot(x_coords, y_coords, color='black', lw=lw, zorder=2)
    
    # Arrow head at the end
    dx = x_coords[-1] - x_coords[-2]
    dy = y_coords[-1] - y_coords[-2]
    # small offset to determine direction
    nx = x_coords[-1] - (0.01 if dx > 0 else (-0.01 if dx < 0 else 0))
    ny = y_coords[-1] - (0.01 if dy > 0 else (-0.01 if dy < 0 else 0))
    
    ax.annotate('', xy=(x_coords[-1], y_coords[-1]), xytext=(nx, ny),
                arrowprops=dict(arrowstyle='->', color='black', lw=lw, mutation_scale=10), zorder=2)
    
    if label:
        # place label on the longest segment
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + text_offset[0], my + text_offset[1], label, 
                ha='center', va='center', fontsize=8, style='italic',
                bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=5)


# ════════════════════════════════════════════════════════════════════════════
# FIG 6: ARCHITECTURE (Layered Grouping Style)
# ════════════════════════════════════════════════════════════════════════════
def fig6_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # ── Layer 1: Data Acquisition & Preprocessing ──
    draw_group_box(ax, 5, 8.0, 9.5, 3.2, "Feature Extraction & Identity Constraints")
    
    # Enrollment
    draw_parallelogram(ax, 2.5, 9.0, 2.5, 0.6, "Enrollment Image\n$f_{enroll}$")
    draw_rounded_rect(ax, 2.5, 7.8, 2.5, 0.7, "ResNet-50\n(Shared Weights)", bold=True)
    draw_cylinder(ax, 2.5, 6.7, 3.0, 0.7, "Immutable Embedding $e_0$\n(Fixed Reference)")
    orth_arrow(ax, (2.5, 8.7), (2.5, 8.15))
    orth_arrow(ax, (2.5, 7.45), (2.5, 7.05))
    
    # Live
    draw_parallelogram(ax, 7.5, 9.0, 2.5, 0.6, "Live Frame\n$f_t$")
    draw_rounded_rect(ax, 7.5, 7.8, 2.5, 0.7, "ResNet-50\n(Shared Weights)", bold=True)
    draw_rect(ax, 7.5, 6.7, 2.5, 0.6, "Probe Embedding\n$e_t$")
    orth_arrow(ax, (7.5, 8.7), (7.5, 8.15))
    orth_arrow(ax, (7.5, 7.45), (7.5, 7.0))
    
    # Comparisons
    draw_rect(ax, 5.0, 6.7, 1.8, 0.6, "Sim $S_t = e_t \\cdot e_0$\nDelta $\\delta_t = e_t - e_0$")
    # Horizontal arrows
    orth_arrow(ax, (4.0, 6.7), (4.1, 6.7)) # From e0
    orth_arrow(ax, (6.25, 6.7), (5.9, 6.7)) # From et
    
    # ── Layer 2: Temporal Modeling ──
    draw_group_box(ax, 5, 4.3, 9.5, 2.4, "Temporal Modeling Layer")
    
    draw_rounded_rect(ax, 2.0, 4.3, 2.4, 1.2, "UC2: Instability\n(LSTM, window $W$)")
    draw_rounded_rect(ax, 5.0, 4.3, 2.4, 1.2, "UC4: Drift\n(Bi-LSTM, 120-buf)")
    draw_rounded_rect(ax, 8.0, 4.3, 2.4, 1.2, "UC3: Presence\n(Bi-LSTM, window $W$)")
    
    # Routing into Layer 2
    # S_t -> UC2
    pth = [(5.0, 6.4), (5.0, 5.8), (2.0, 5.8), (2.0, 4.9)]
    x, y = zip(*pth)
    ax.plot(x, y, color='black', lw=1.2, zorder=2)
    ax.annotate('', xy=(2.0, 4.9), xytext=(2.0, 5.0), arrowprops=dict(arrowstyle='->', color='black', mutation_scale=10))
    ax.text(2.6, 5.8, "Similarity seq $S_t$", ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=0))

    # delta -> UC4
    pth = [(5.0, 6.4), (5.0, 4.9)]
    x, y = zip(*pth)
    ax.plot(x, y, color='black', lw=1.2, zorder=2)
    ax.annotate('', xy=(5.0, 4.9), xytext=(5.0, 5.0), arrowprops=dict(arrowstyle='->', color='black', mutation_scale=10))
    ax.text(5.5, 5.8, "Delta seq $\\delta_t$", ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=0))

    # Live frame -> 6D Features -> UC3
    draw_rect(ax, 8.0, 5.7, 2.0, 0.4, "6D Extractor", ls='--', fc='#f8f8f8')
    orth_arrow(ax, (8.5, 8.7), (8.5, 5.9)) # tap from frame
    orth_arrow(ax, (8.0, 5.5), (8.0, 4.9), label="6D seq", text_offset=(0.6, 0))

    # ── Layer 3: Risk Fusion ──
    draw_group_box(ax, 5, 1.6, 9.5, 2.0, "Session Risk Fusion (Probabilistic)")
    
    draw_rounded_rect(ax, 5.0, 1.6, 3.5, 1.0, "UC5: GRU Fusion Engine\n(Session-level BCE Supervision)", bold=True)
    
    # Routing into Layer 3
    orth_arrow(ax, (2.0, 3.7), (4.0, 2.1))
    ax.text(2.5, 2.9, "Instability $I_t$", fontsize=8, style='italic', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    
    orth_arrow(ax, (5.0, 3.7), (5.0, 2.1))
    ax.text(5.4, 2.9, "Drift $D_t$", fontsize=8, style='italic', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    
    orth_arrow(ax, (8.0, 3.7), (6.0, 2.1))
    ax.text(7.5, 2.9, "Presence $P_t$", fontsize=8, style='italic', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    
    # S_t direct leak to UC5
    pth = [(4.1, 6.7), (0.4, 6.7), (0.4, 1.6), (3.25, 1.6)]
    x, y = zip(*pth)
    ax.plot(x, y, color='black', lw=1.0, ls=':', zorder=2)
    ax.annotate('', xy=(3.25, 1.6), xytext=(3.15, 1.6), arrowprops=dict(arrowstyle='->', color='black', mutation_scale=10))
    
    # Output
    draw_parallelogram(ax, 5.0, 0.4, 3.0, 0.5, "Final Risk Trajectory $\\rho_T$")
    orth_arrow(ax, (5.0, 1.1), (5.0, 0.65))

    plt.savefig(os.path.join(OUT_DIR, 'fig6_system_architecture.png'))
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# FIG 7: WORKFLOW (Formal Flowchart)
# ════════════════════════════════════════════════════════════════════════════
def fig7_workflow():
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    cx = 5.0
    
    # Draw Background Phase Zones
    ax.add_patch(mpatches.Rectangle((0.5, 11.2), 9, 2.5, fill=True, facecolor='#f4f4f4', edgecolor='none', zorder=0))
    ax.text(1.0, 13.4, "Phase 1: Enrollment", fontsize=10, fontweight='bold', color='#555555')
    
    ax.add_patch(mpatches.Rectangle((0.5, 3.2), 9, 7.8, fill=True, facecolor='#fdfdfd', edgecolor='none', zorder=0))
    ax.text(1.0, 10.7, "Phase 2: Live Monitoring", fontsize=10, fontweight='bold', color='#555555')
    
    ax.add_patch(mpatches.Rectangle((0.5, 0.5), 9, 2.5, fill=True, facecolor='#f4f4f4', edgecolor='none', zorder=0))
    ax.text(1.0, 2.7, "Phase 3: Session End", fontsize=10, fontweight='bold', color='#555555')
    
    # Nodes
    draw_oval(ax, cx, 13.0, 2.0, 0.6, "Session Start")
    
    draw_parallelogram(ax, cx, 12.0, 3.5, 0.6, "Capture Enrollment Image")
    orth_arrow(ax, (cx, 12.7), (cx, 12.3))
    
    draw_cylinder(ax, cx, 10.8, 3.5, 0.8, "Compute & Fix\nImmutable Embedding $e_0$")
    orth_arrow(ax, (cx, 11.7), (cx, 11.2))
    
    draw_oval(ax, cx, 9.6, 2.0, 0.6, "Loop $t=1 \\to T$")
    orth_arrow(ax, (cx, 10.4), (cx, 9.9))
    
    draw_parallelogram(ax, cx, 8.6, 3.5, 0.6, "Capture Live Frame $f_t$")
    orth_arrow(ax, (cx, 9.3), (cx, 8.9))
    
    draw_rect(ax, cx, 7.5, 4.5, 0.8, "Extract Probe $e_t$\nCompute $S_t = e_t \\cdot e_0$, $\\delta_t = e_t - e_0$")
    orth_arrow(ax, (cx, 8.3), (cx, 7.9))
    
    draw_rect(ax, cx, 6.4, 4.5, 0.8, "Update Temporal Windows\nUC2($S_t$), UC3(6D), UC4($\\delta_t$)")
    orth_arrow(ax, (cx, 7.1), (cx, 6.8))
    
    draw_rect(ax, cx, 5.3, 4.5, 0.8, "Update Risk State\nUC5 GRU: $\\rho_t = f(S_t, I_t, P_t, D_t)$")
    orth_arrow(ax, (cx, 6.0), (cx, 5.7))
    
    draw_diamond(ax, cx, 4.1, 2.5, 1.0, "Is $t == T$ ?")
    orth_arrow(ax, (cx, 4.9), (cx, 4.6))
    
    # No branch (Loops back)
    pth = [(cx + 1.25, 4.1), (8.5, 4.1), (8.5, 10.0), (cx + 1.0, 10.0)]
    x, y = zip(*pth)
    ax.plot(x, y, color='black', lw=1.2, zorder=2)
    ax.annotate('', xy=(cx + 1.0, 10.0), xytext=(cx + 1.1, 10.0), arrowprops=dict(arrowstyle='->', color='black', mutation_scale=10))
    ax.text(8.5, 7.0, "No", ha='left', va='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2))
    
    # Yes branch
    orth_arrow(ax, (cx, 3.6), (cx, 2.3))
    ax.text(cx, 3.0, "Yes", ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2))
    
    draw_rect(ax, cx, 1.9, 3.5, 0.8, "Commit Final Risk $\\rho_T$\n(No Thresholds Applied)")
    draw_oval(ax, cx, 0.8, 2.0, 0.6, "Session End")
    orth_arrow(ax, (cx, 1.5), (cx, 1.1))

    plt.savefig(os.path.join(OUT_DIR, 'fig7_workflow.png'))
    plt.close()


if __name__ == '__main__':
    print('Generating formal IEEE flowchart and architecture diagrams...')
    fig6_architecture()
    fig7_workflow()
    print(f'Output saved to {OUT_DIR}')

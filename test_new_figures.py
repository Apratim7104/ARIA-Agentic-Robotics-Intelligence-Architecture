"""
test_new_figures.py
Prototyping clean, elegant, spacious, publication-grade figures for CCDM Report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "ccdm_report")
os.makedirs(FIG_DIR, exist_ok=True)

# Colors
NAVY = "#14325c"
SLATE = "#1f4e79"
AMBER = "#8a5a00"
GREEN = "#2c6e2a"
BURGUNDY = "#8a2a2a"
PURPLE = "#5a2a8a"
GREY = "#555555"

def draw_pill(ax, x, y, text, fc="#ffffff", ec="#14325c", fs=7.0, bold=True, pad=0.08):
    """Draws a clean floating badge/pill for arrow annotations."""
    bbox = dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=0.8, alpha=0.95)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=ec, bbox=bbox, zorder=10)

def draw_card(ax, x, y, w, h, title, items, header_fc, header_tc, body_fc, body_ec):
    """Draws a professional card with top title bar and clean left-aligned bullet items."""
    # Main Card Body
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=body_fc, ec=body_ec, lw=1.2, zorder=2))
    # Header Bar
    hh = 0.58
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=header_fc, ec=header_fc, lw=0.1, zorder=3))
    # Patch lower corners of header to make flat connection
    ax.add_patch(Rectangle((x, y + h - hh), w, 0.15, fc=header_fc, ec=header_fc, zorder=3))
    # Title Text
    ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=header_tc, zorder=4)
    # Body Bullets (Left-aligned, spaced cleanly)
    num_items = len(items)
    avail_h = h - hh - 0.25
    dy = avail_h / max(num_items, 1)
    cur_y = y + h - hh - 0.25
    for item in items:
        ax.text(x + 0.15, cur_y, f"\u2022 {item}", ha="left", va="center",
                fontsize=7.2, color="#222222", zorder=4)
        cur_y -= dy

def make_layer_continuation():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 5.0); ax.axis("off")

    # Card 1: Layer 2
    l2_items = [
        "Camera deprojection: (u,v,d) \u2192 p_world",
        "Kalman gating & Hungarian tracking",
        "Initial object locked pose: p_obj_0",
        "Continuous distance: d_rel = ||p_ee - p_obj||",
        "Range rate tracking: dd_rel/dt",
        "DLS-IK & reachability analysis",
        "Normalized 30-D vector: p_spatial"
    ]
    draw_card(ax, 0.4, 0.6, 2.9, 3.8, "LAYER 2: SPATIAL REASONING", l2_items,
              NAVY, "#ffffff", "#f5f8fc", NAVY)

    # Card 2: Layer 3
    l3_items = [
        "Multi-Agent context c_t \u2208 [0, 1]\u2075",
        "Continuous states [x, u, d_rel]",
        "Physical scene facts [m, \u03bc, z_table]",
        "FiLM affine latent modulation",
        "Residual prediction: r_hat \u2208 \u211d\u2076",
        "Heteroscedastic variance: \u03c3_t \u2208 \u211d\u2076",
        "Conformal 2\u03c3 chance constraints"
    ]
    draw_card(ax, 4.05, 0.6, 2.9, 3.8, "LAYER 3: CCDM DYNAMICS NN", l3_items,
              AMBER, "#ffffff", "#fffcf5", AMBER)

    # Card 3: Layer 4
    l4_items = [
        "CasADi + IPOPT multiple shooting",
        "Velocity corridor: v_max(d_rel, c_frag)",
        "Coulomb anti-slip: a_max \u2264 \u03bc\u00b7g",
        "Table safety barrier: z_ee \u2265 z_table + 2\u03c3",
        "Position escalation: Q_pos(d_rel)",
        "Overshoot damping: Q_vel(c_frag)",
        "Anti-jerk slew penalty: S_slew"
    ]
    draw_card(ax, 7.7, 0.6, 2.9, 3.8, "LAYER 4: FRANKA 7-DOF NMPC", l4_items,
              BURGUNDY, "#ffffff", "#fdf6f6", BURGUNDY)

    # Connecting Arrow 1 (L2 -> L3)
    p1_start = (3.3, 2.5); p1_end = (4.05, 2.5)
    ax.add_patch(FancyArrowPatch(p1_start, p1_end, arrowstyle="-|>", mutation_scale=12,
                                color=NAVY, lw=1.6, zorder=5))
    draw_pill(ax, 3.675, 2.85, "p_spatial\nd_rel, p_obj_0", fc="#ffffff", ec=NAVY, fs=6.8)

    # Connecting Arrow 2 (L3 -> L4)
    p2_start = (6.95, 2.5); p2_end = (7.7, 2.5)
    ax.add_patch(FancyArrowPatch(p2_start, p2_end, arrowstyle="-|>", mutation_scale=12,
                                color=AMBER, lw=1.6, zorder=5))
    draw_pill(ax, 7.325, 2.85, "v_max, a_max\nQ_pos, \u03c3_t", fc="#ffffff", ec=AMBER, fs=6.8)

    # Header title banner
    ax.text(5.5, 4.65, "Perception-to-Control Dataflow & Interface Continuity",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)

    fig.tight_layout()
    p = os.path.join(FIG_DIR, "layer_continuation.png")
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG_DIR, "layer_continuation.eps"), format="eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved test layer_continuation!")

if __name__ == "__main__":
    make_layer_continuation()


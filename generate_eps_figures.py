"""
generate_eps_figures.py

Generates vector EPS (.eps) files for all 8 figures in the Spatial Reasoning Layer
Technical Report so that they can be used directly in Overleaf and LaTeX documents:
  1. architecture.eps
  2. timing.eps
  3. tracking.eps
  4. geometry.eps
  5. depth.eps
  6. error.eps
  7. dls.eps
  8. features.eps

Also re-bundles Spatial_Reasoning_Layer_Overleaf.zip to include both vector .eps
and raster .png figures for instant Overleaf compilation.
"""

import os
import zipfile

# Direct MPLCONFIGDIR to avoid permission warnings on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

# Embed fonts properly in PostScript / PDF vector outputs
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")
ZIP_PATH = os.path.join(ROOT, "Spatial_Reasoning_Layer_Overleaf.zip")
TEX_PATH = os.path.join(ROOT, "Spatial_Reasoning_Layer_Technical_Report.tex")
BIB_PATH = os.path.join(ROOT, "references.bib")

os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, base_name):
    """Saves figure in both vector .eps and .png formats."""
    eps_path = os.path.join(FIG_DIR, f"{base_name}.eps")
    png_path = os.path.join(FIG_DIR, f"{base_name}.png")
    
    # Save vector EPS
    fig.savefig(eps_path, format="eps", bbox_inches="tight", facecolor="white", edgecolor="none")
    # Save high-resolution PNG
    fig.savefig(png_path, format="png", dpi=250, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    
    eps_size = os.path.getsize(eps_path)
    png_size = os.path.getsize(png_path)
    print(f"  [SAVED] {base_name}.eps ({eps_size:,} bytes) | {base_name}.png ({png_size:,} bytes)")
    return eps_path

def block(ax, x, y, w, h, text, fc="#dce8f7", ec="#14325c", fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(ax, p, q, text=None, color="#14325c", fs=6.6, off=(0, 0.1), ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.1, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, fontsize=fs, ha="center", color=color)

# ---------------------------------------------------------------- Figures
def gen_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    block(ax, 0.1, 5.0, 2.3, 1.0, "Layer 1: ARIA\n(LLM / VLM / mask agents)\nJSON detections", "#f6e6c8", "#8a5a00", bold=True)
    block(ax, 0.1, 3.3, 2.3, 1.0, "MuJoCo physics loop\nq, \u03c4, scene state", "#e3f2e1", "#2c6e2a", bold=True)
    block(ax, 0.1, 1.6, 2.3, 1.0, "Simulated RGB-D camera\n(overhead_cam)", "#e3f2e1", "#2c6e2a", bold=True)
    ax.add_patch(Rectangle((3.0, 0.5), 4.6, 5.7, fc="#f5f8fc", ec="#14325c", lw=1.6, ls="--"))
    ax.text(5.3, 6.02, "LAYER 2 \u2013 SPATIAL REASONING LAYER", ha="center", fontsize=9, fontweight="bold", color="#14325c")
    block(ax, 3.2, 4.95, 4.2, 0.62, "ObjectDetectionProcessor\nvalidate \u2192 gate \u2192 Hungarian track")
    block(ax, 3.2, 4.1, 4.2, 0.62, "MuJoCoStateManager\nstate snapshot + linearised depth")
    block(ax, 3.2, 3.25, 4.2, 0.62, "CoordinateTransformer\npixel \u2192 camera \u2192 world (SE(3))")
    block(ax, 3.2, 2.4, 4.2, 0.62, "SpatialAnalyzer\nd_rel, workspace, DLS-IK, grasp, difficulty")
    block(ax, 3.2, 1.55, 4.2, 0.62, "FeatureEncoder\n30-D normalised vector p_spatial")
    block(ax, 3.2, 0.7, 4.2, 0.62, "LatestQueue \u2192 worker \u2192 DoubleBuffer\n(lock-free hand-off)", "#fff3d6", "#8a5a00")
    for y0, y1 in [(4.95, 4.72), (4.1, 3.87), (3.25, 3.02), (2.4, 2.17), (1.55, 1.32)]:
        arrow(ax, (5.3, y0), (5.3, y1))
    arrow(ax, (2.4, 5.5), (3.2, 5.26))
    arrow(ax, (2.4, 3.8), (3.2, 4.4))
    arrow(ax, (2.4, 2.1), (3.2, 4.2))
    block(ax, 8.1, 3.9, 1.8, 1.4, "Layer 3\nCCDM\n(context,\nstate, scene)", "#e9dcf5", "#5a2a8a", bold=True)
    block(ax, 8.1, 1.7, 1.8, 1.4, "Layer 4\nNMPC\n(CasADi+IPOPT)", "#f5dcdc", "#8a2a2a", bold=True)
    arrow(ax, (7.4, 1.86), (8.1, 4.3), color="#5a2a8a")
    ax.text(7.75, 1.3, "30-D feature vector", fontsize=6.8, ha="left", color="#5a2a8a")
    arrow(ax, (9.0, 3.9), (9.0, 3.1), color="#5a2a8a")
    ax.text(9.1, 3.5, "constraints,\nweights", fontsize=6.6, ha="left", va="center", color="#5a2a8a")
    save_fig(fig, "architecture")

def gen_geometry():
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.set_xlim(-1, 9); ax.set_ylim(-1.2, 4.6); ax.axis("off"); ax.set_aspect("equal")
    cam = np.array([1.2, 3.8])
    ax.plot(*cam, "ko", ms=6); ax.text(cam[0] - 0.15, cam[1] + 0.3, "camera centre  C", fontsize=8)
    # optical axis
    ax.add_patch(FancyArrowPatch(cam, (1.2 + 5.2, 3.8 - 3.4), arrowstyle="-|>", color="#14325c", lw=1.4, mutation_scale=10))
    ax.text(5.2, 0.8, "optical axis  z_c", fontsize=8, color="#14325c")
    # image plane
    ip_c = cam + 1.3 * np.array([0.837, -0.547]); n = np.array([0.547, 0.837])
    ax.plot([ip_c[0] - 0.85 * n[0], ip_c[0] + 0.85 * n[0]], [ip_c[1] - 0.85 * n[1], ip_c[1] + 0.85 * n[1]], color="#2c6e2a", lw=3)
    ax.text(ip_c[0] - 1.9, ip_c[1] + 0.55, "image plane\n(u, v), focal length f", fontsize=8, color="#2c6e2a")
    # object ray
    P = np.array([6.8, 0.1]); ax.plot(*P, "s", color="#c0392b", ms=9); ax.text(P[0] + 0.15, P[1] + 0.25, "object centroid  p_w", fontsize=8, color="#c0392b")
    ax.plot([cam[0], P[0]], [cam[1], P[1]], "--", color="#c0392b", lw=1.1)
    pi = cam + 1.3 / np.dot(P - cam, [0.837, -0.547]) * (P - cam)
    ax.plot(*pi, "o", color="#c0392b", ms=5); ax.text(pi[0] - 1.65, pi[1] - 0.55, "pixel (u, v)", fontsize=8, color="#c0392b")
    # depth projection
    d_dir = np.array([0.837, -0.547]); proj = cam + np.dot(P - cam, d_dir) * d_dir
    ax.plot([P[0], proj[0]], [P[1], proj[1]], ":", color="#555", lw=1)
    ax.text((P[0] + proj[0]) / 2 + 0.1, (P[1] + proj[1]) / 2 + 0.1, "d  (planar depth)", fontsize=8, color="#555")
    # table
    ax.plot([-0.5, 8.5], [-0.3, -0.3], color="#8a5a00", lw=3); ax.text(7.2, -0.75, "table plane", fontsize=8, color="#8a5a00")
    # world frame
    o = np.array([0.1, -0.1]); ax.add_patch(FancyArrowPatch(o, o + [1.0, 0], arrowstyle="-|>", color="k", mutation_scale=8))
    ax.add_patch(FancyArrowPatch(o, o + [0, 1.0], arrowstyle="-|>", color="k", mutation_scale=8))
    ax.text(o[0] + 1.05, o[1] - 0.12, "x_w", fontsize=8); ax.text(o[0] - 0.1, o[1] + 1.1, "z_w", fontsize=8)
    ax.text(0.0, 2.2, "world frame W\n(T_cw: camera \u2192 world)", fontsize=8)
    save_fig(fig, "geometry")

def gen_depth(n=0.05, f=10.0):
    d = np.linspace(n * 1.001, 3.0, 800)
    db = (1 - n / d) / (1 - n / f)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
    ax[0].plot(d, db, color="#14325c", lw=2); ax[0].set_xlabel("metric depth d (m)"); ax[0].set_ylabel("raw depth-buffer value d_b")
    ax[0].set_title("Non-linear depth buffer (illustrative n = 0.05 m, f = 10 m)", fontsize=8); ax[0].grid(alpha=.3)
    ax[0].axvspan(0.5, 1.0, color="#f6e6c8", alpha=.7); ax[0].text(0.52, 0.2, "tabletop\nworking range", fontsize=7)
    dd = np.linspace(0.05, 3.0, 400)
    for nn, col in [(0.01, "#2c6e2a"), (0.05, "#14325c"), (0.2, "#8a2a2a")]:
        ax[1].plot(dd, dd ** 2 / nn * (1 - nn / 10) * 2 ** -24 * 1e3, label=f"n = {nn} m", color=col, lw=1.8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("metric depth d (m)"); ax[1].set_ylabel("depth resolution (mm), 24-bit buffer")
    ax[1].set_title("Quantisation \u2248 d\u00b2/(n\u00b72\u00b2\u2074)", fontsize=8); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    save_fig(fig, "depth")

def gen_tracking():
    fig, ax = plt.subplots(figsize=(6.8, 3.6)); ax.set_xlim(40, 280); ax.set_ylim(60, 190); ax.invert_yaxis()
    tr = np.array([[110, 95], [190, 140], [235, 85]]); vel = np.array([[8, 3], [-5, 4], [0, 2]])
    pr = tr + vel; det = np.array([[121, 100], [182, 147], [250, 160]])
    for i, (p, name) in enumerate(zip(tr, ["track 0", "track 1", "track 2"])):
        ax.plot(*p, "o", color="#14325c"); ax.text(p[0] - 6, p[1] - 8, name, fontsize=7, color="#14325c")
        ax.annotate("", xy=pr[i], xytext=p, arrowprops=dict(arrowstyle="->", color="#14325c", lw=1))
        ax.plot(*pr[i], "o", mfc="none", color="#14325c")
        ax.add_patch(Circle(pr[i], 40, fc="none", ec="#14325c", ls=":", lw=0.9))
    for j, p in enumerate(det):
        ax.plot(*p, "s", color="#c0392b"); ax.text(p[0] + 4, p[1] + 10, f"det {j}", fontsize=7, color="#c0392b")
    for i, j in [(0, 0), (1, 1)]:
        ax.plot([pr[i][0], det[j][0]], [pr[i][1], det[j][1]], "-", color="#2c6e2a", lw=1.8)
    ax.plot([pr[2][0], det[2][0]], [pr[2][1], det[2][1]], "x--", color="#999", lw=1)
    ax.text(277, 197, "distance > \u03c4 (40 px): assignment\nrejected \u2192 new track spawned", fontsize=7, color="#555", ha="right")
    ax.set_ylim(205, 60)
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title("Constant-velocity prediction (\u25cb), gate radius \u03c4 (dotted) and Hungarian assignment (green)", fontsize=8)
    ax.grid(alpha=.25)
    save_fig(fig, "tracking")

def gen_dls():
    s = np.linspace(0, 1.6, 400)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for lam, col in [(0.0, "#999"), (0.01, "#2c6e2a"), (0.05, "#14325c"), (0.2, "#8a2a2a")]:
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(s > 0, s / (s ** 2 + lam ** 2), 0.0) if lam > 0 else np.where(s > 0, 1 / s, np.nan)
        ax[0].plot(s, np.minimum(g, 60), label=f"\u03bb = {lam}", color=col, lw=1.8)
    ax[0].set_ylim(0, 25); ax[0].set_xlabel("singular value \u03c3\u1d62 of J"); ax[0].set_ylabel("gain  \u03c3/(\u03c3\u00b2+\u03bb\u00b2)")
    ax[0].set_title("Joint-rate gain per singular direction", fontsize=8); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    lams = np.logspace(-4, 0, 300)
    sig = 0.05
    ax[1].loglog(lams, sig / (sig ** 2 + lams ** 2), color="#14325c", lw=2, label="gain at \u03c3 = 0.05 (near-singular)")
    ax[1].loglog(lams, np.full_like(lams, 1 / sig), "--", color="#999", label="pseudo-inverse 1/\u03c3")
    ax[1].axvline(0.05, color="#c0392b", lw=1, ls=":"); ax[1].text(0.055, 1.3, "default \u03bb\u2080 = 0.05", fontsize=7, color="#c0392b")
    ax[1].set_xlabel("damping \u03bb"); ax[1].set_title("Damping bounds the gain: max = 1/(2\u03bb)", fontsize=8); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    save_fig(fig, "dls")

def gen_error():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    d = np.linspace(0.3, 1.5, 100)
    for f, col in [(250, "#14325c"), (500, "#2c6e2a"), (1000, "#8a2a2a")]:
        ax.plot(d, 0.5 * d / f * 1e3, label=f"f = {f} px", color=col, lw=2)
    ax.axvline(0.775, color="#999", ls=":"); ax.text(0.33, 2.2, "dotted line: stock camera,\nd \u2248 0.78 m above cube centre", fontsize=7, color="#555")
    ax.set_xlabel("depth d (m)"); ax.set_ylabel("worst-case lateral error (mm)")
    ax.set_title("Half-pixel quantisation error  \u03b4\u2093\u1d67 = d / (2f)", fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=.3)
    save_fig(fig, "error")

def gen_features():
    fig, ax = plt.subplots(figsize=(9.2, 1.8)); ax.set_xlim(0, 30); ax.set_ylim(0, 2.1); ax.axis("off")
    groups = [("p_obj", 0, 3, "#dce8f7"), ("p_ee", 3, 6, "#e3f2e1"), ("obj_quat", 6, 10, "#f6e6c8"), ("distances", 10, 13, "#e9dcf5"),
              ("grasp_off", 13, 16, "#f5dcdc"), ("flags", 16, 20, "#fff3d6"), ("diff", 20, 21, "#d9efef"), ("conf", 21, 22, "#efe2d9"),
              ("padding (zeros)", 22, 30, "#eeeeee")]
    for name, a, b, col in groups:
        ax.add_patch(Rectangle((a, 0.7), b - a, 0.8, fc=col, ec="k", lw=0.8))
        ax.text((a + b) / 2, 1.1, name, ha="center", va="center", fontsize=6.6 if b - a > 1 else 5.4, rotation=0 if b - a > 1 else 90)
        ax.text((a + b) / 2, 0.45, f"{a}\u2013{b - 1}" if b - a > 1 else f"{a}", ha="center", fontsize=6.4, color="#555")
    ax.text(15, 1.85, "30-D spatial feature vector   p\u209c\u02e2\u1d56\u1d43\u1d57\u1da6\u1d43\u02e1 \u2208 [\u22121, 1]\u00b3\u2070  (float32)", ha="center", fontsize=8.5, fontweight="bold", color="#14325c")
    save_fig(fig, "features")

def gen_timing():
    fig, ax = plt.subplots(figsize=(8.8, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 4.3); ax.axis("off")
    lanes = ["ARIA (async)", "Physics thread", "Layer-2 worker", "NMPC / CCDM"]
    for i, l in enumerate(lanes):
        y = 3.5 - i * 1.0
        ax.text(-1, y + 0.2, l, ha="right", fontsize=7.6, fontweight="bold"); ax.plot([0, 100], [y + 0.2, y + 0.2], color="#ddd", lw=0.8)
    for t in [8, 24, 40, 56, 72, 88]:
        ax.add_patch(Rectangle((t, 3.35), 2.2, 0.32, fc="#f6e6c8", ec="#8a5a00"))
    for t in range(2, 99, 4):
        ax.add_patch(Rectangle((t, 2.35), 1.0, 0.32, fc="#e3f2e1", ec="#2c6e2a"))
    for t in [10, 26, 42, 58, 74, 90]:
        ax.add_patch(Rectangle((t, 1.35), 5, 0.32, fc="#dce8f7", ec="#14325c"))
        ax.annotate("", xy=(t + 5, 0.9), xytext=(t + 5, 1.35), arrowprops=dict(arrowstyle="->", color="#14325c", lw=0.8))
    for t in range(0, 98, 4):
        ax.add_patch(Rectangle((t + 1, 0.35), 2.4, 0.32, fc="#e9dcf5", ec="#5a2a8a"))
    ax.text(50, 4.05, "Producer/consumer decoupling: the controller always reads the newest completed feature vector (double buffer) and never blocks on perception",
            ha="center", fontsize=7.2, color="#14325c")
    ax.text(100, -0.15, "time \u2192", ha="right", fontsize=7, color="#555")
    save_fig(fig, "timing")

def update_overleaf_zip():
    print(f"\nPackaging updated Overleaf project ZIP (with vector .eps & .png): {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(TEX_PATH):
            z.write(TEX_PATH, arcname="Spatial_Reasoning_Layer_Technical_Report.tex")
        if os.path.exists(BIB_PATH):
            z.write(BIB_PATH, arcname="references.bib")
        if os.path.exists(FIG_DIR):
            for fname in sorted(os.listdir(FIG_DIR)):
                fpath = os.path.join(FIG_DIR, fname)
                if os.path.isfile(fpath):
                    z.write(fpath, arcname=os.path.join("figures", fname))
    print(f"  [SUCCESS] {ZIP_PATH} ({os.path.getsize(ZIP_PATH) // 1024} KB)")

def main():
    print("=" * 75)
    print(" GENERATING VECTOR EPS (.eps) AND PNG (.png) FIGURES FOR OVERLEAF")
    print("=" * 75)
    gen_architecture()
    gen_timing()
    gen_tracking()
    gen_geometry()
    gen_depth()
    gen_error()
    gen_dls()
    gen_features()
    update_overleaf_zip()
    print("=" * 75)
    print("All 8 figures successfully generated in vector EPS format!")
    print("=" * 75)

if __name__ == "__main__":
    main()


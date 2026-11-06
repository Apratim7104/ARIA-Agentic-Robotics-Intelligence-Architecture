"""Build 'Spatial_Reasoning_Layer_Technical_Report.pdf' (ReportLab + matplotlib mathtext equations)."""
import io, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether, NextPageTemplate)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Spatial_Reasoning_Layer_Technical_Report.pdf")
FIG = os.path.join(ROOT, "scratch", "report_figs")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- fonts
ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"),
                ("Serif-I", "DejaVuSerif-Italic.ttf"), ("Serif-BI", "DejaVuSerif-BoldItalic.ttf"),
                ("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
                ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(ttf, f)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans", boldItalic="Sans-B")

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "DejaVu Sans"

PAGE_W, PAGE_H = A4
LM = RM = 2.2 * cm
TW = PAGE_W - LM - RM
NAVY = colors.HexColor("#14325c")
GREY = colors.HexColor("#555555")

# ---------------------------------------------------------------- styles
body = ParagraphStyle("body", fontName="Serif", fontSize=9.6, leading=15.2, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=16, leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=12, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5)
h3 = ParagraphStyle("h3", fontName="Sans-B", fontSize=10, leading=13, textColor=colors.HexColor("#333333"), spaceBefore=6, spaceAfter=3)
cap = ParagraphStyle("cap", fontName="Serif-I", fontSize=8.4, leading=11, alignment=TA_CENTER, textColor=GREY, spaceAfter=10)
small = ParagraphStyle("small", fontName="Serif", fontSize=8.4, leading=11.5, alignment=TA_LEFT)
cell = ParagraphStyle("cell", fontName="Serif", fontSize=8.4, leading=11, alignment=TA_LEFT)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-B", textColor=colors.white)
bib = ParagraphStyle("bib", fontName="Serif", fontSize=8.6, leading=11.6, leftIndent=22, firstLineIndent=-22, spaceAfter=3.2, alignment=TA_LEFT)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=3)
box = ParagraphStyle("box", parent=body, backColor=colors.HexColor("#eef3fa"), borderColor=NAVY, borderWidth=0.6,
                     borderPadding=6, spaceBefore=6, spaceAfter=10, fontSize=9.2, leading=13.4)
title_s = ParagraphStyle("title", fontName="Sans-B", fontSize=25, leading=31, textColor=NAVY, alignment=TA_LEFT)
sub_s = ParagraphStyle("sub", fontName="Serif-I", fontSize=12.5, leading=17, textColor=GREY, alignment=TA_LEFT)

# ---------------------------------------------------------------- citations
BIB = {
 "hartley": "Hartley, R. and Zisserman, A. <i>Multiple View Geometry in Computer Vision</i>, 2nd ed. Cambridge University Press, 2004.",
 "zhang": "Zhang, Z. A flexible new technique for camera calibration. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 22(11):1330–1334, 2000.",
 "brown": "Brown, D. C. Close-range camera calibration. <i>Photogrammetric Engineering</i>, 37(8):855–866, 1971.",
 "kuhn": "Kuhn, H. W. The Hungarian method for the assignment problem. <i>Naval Research Logistics Quarterly</i>, 2(1–2):83–97, 1955.",
 "sort": "Bewley, A., Ge, Z., Ott, L., Ramos, F., and Upcroft, B. Simple online and realtime tracking. In <i>IEEE International Conference on Image Processing (ICIP)</i>, pp. 3464–3468, 2016.",
 "deepsort": "Wojke, N., Bewley, A., and Paulus, D. Simple online and realtime tracking with a deep association metric. In <i>IEEE ICIP</i>, pp. 3645–3649, 2017.",
 "kalman": "Kalman, R. E. A new approach to linear filtering and prediction problems. <i>Journal of Basic Engineering</i>, 82(1):35–45, 1960.",
 "wampler": "Wampler, C. W. Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods. <i>IEEE Transactions on Systems, Man, and Cybernetics</i>, 16(1):93–101, 1986.",
 "nakamura": "Nakamura, Y. and Hanafusa, H. Inverse kinematic solutions with singularity robustness for robot manipulator control. <i>Journal of Dynamic Systems, Measurement, and Control</i>, 108(3):163–171, 1986.",
 "buss": "Buss, S. R. Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and damped least squares methods. Technical report, University of California, San Diego, 2004.",
 "levenberg": "Levenberg, K. A method for the solution of certain non-linear problems in least squares. <i>Quarterly of Applied Mathematics</i>, 2(2):164–168, 1944.",
 "marquardt": "Marquardt, D. W. An algorithm for least-squares estimation of nonlinear parameters. <i>Journal of the Society for Industrial and Applied Mathematics</i>, 11(2):431–441, 1963.",
 "mujoco": "Todorov, E., Erez, T., and Tassa, Y. MuJoCo: A physics engine for model-based control. In <i>IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)</i>, pp. 5026–5033, 2012.",
 "craig": "Craig, J. J. <i>Introduction to Robotics: Mechanics and Control</i>, 3rd ed. Pearson Prentice Hall, 2005.",
 "siciliano": "Siciliano, B., Sciavicco, L., Villani, L., and Oriolo, G. <i>Robotics: Modelling, Planning and Control</i>. Springer, 2009.",
 "lynch": "Lynch, K. M. and Park, F. C. <i>Modern Robotics: Mechanics, Planning, and Control</i>. Cambridge University Press, 2017.",
 "khatib": "Khatib, O. A unified approach for motion and force control of robot manipulators: The operational space formulation. <i>IEEE Journal on Robotics and Automation</i>, 3(1):43–53, 1987.",
 "film": "Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. FiLM: Visual reasoning with a general conditioning layer. In <i>AAAI Conference on Artificial Intelligence</i>, 2018.",
 "levine": "Levine, S., Finn, C., Darrell, T., and Abbeel, P. End-to-end training of deep visuomotor policies. <i>Journal of Machine Learning Research</i>, 17(39):1–40, 2016.",
 "qtopt": "Kalashnikov, D., Irpan, A., Pastor, P., et al. QT-Opt: Scalable deep reinforcement learning for vision-based robotic manipulation. In <i>Conference on Robot Learning (CoRL)</i>, 2018.",
 "dexnet": "Mahler, J., Liang, J., Niyaz, S., et al. Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. In <i>Robotics: Science and Systems (RSS)</i>, 2017.",
 "transporter": "Zeng, A., Florence, P., Tompson, J., et al. Transporter networks: Rearranging the visual world for robotic manipulation. In <i>Conference on Robot Learning (CoRL)</i>, 2020 (PMLR 155, 2021).",
 "rt1": "Brohan, A., Brown, N., Carbajal, J., et al. RT-1: Robotics Transformer for real-world control at scale. In <i>Robotics: Science and Systems (RSS)</i>, 2023.",
 "saycan": "Ahn, M., Brohan, A., Brown, N., et al. Do as I can, not as I say: Grounding language in robotic affordances. In <i>Conference on Robot Learning (CoRL)</i>, 2022.",
 "cap": "Liang, J., Huang, W., Xia, F., et al. Code as Policies: Language model programs for embodied control. In <i>IEEE International Conference on Robotics and Automation (ICRA)</i>, 2023.",
 "sam": "Kirillov, A., Mintun, E., Ravi, N., et al. Segment Anything. In <i>IEEE/CVF International Conference on Computer Vision (ICCV)</i>, pp. 4015–4026, 2023.",
 "gdino": "Liu, S., Zeng, Z., Ren, T., et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. arXiv:2303.05499, 2023 (ECCV 2024).",
 "frcnn": "Ren, S., He, K., Girshick, R., and Sun, J. Faster R-CNN: Towards real-time object detection with region proposal networks. In <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2015.",
 "casadi": "Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., and Diehl, M. CasADi: A software framework for nonlinear optimization and optimal control. <i>Mathematical Programming Computation</i>, 11(1):1–36, 2019.",
 "ipopt": "Wächter, A. and Biegler, L. T. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. <i>Mathematical Programming</i>, 106(1):25–57, 2006.",
 "mayne": "Mayne, D. Q., Rawlings, J. B., Rao, C. V., and Scokaert, P. O. M. Constrained model predictive control: Stability and optimality. <i>Automatica</i>, 36(6):789–814, 2000.",
 "barshalom": "Bar-Shalom, Y., Li, X. R., and Kirubarajan, T. <i>Estimation with Applications to Tracking and Navigation</i>. Wiley, 2001.",
 "foley": "Foley, J. D., van Dam, A., Feiner, S. K., and Hughes, J. F. <i>Computer Graphics: Principles and Practice</i>, 2nd ed. Addison-Wesley, 1990.",
 "ros": "Quigley, M., Conley, K., Gerkey, B., et al. ROS: An open-source Robot Operating System. In <i>ICRA Workshop on Open Source Software</i>, 2009.",
 "franka": "Haddadin, S., Parusel, S., Johannsmeier, L., et al. The Franka Emika robot: A standard platform in robotics research. <i>IEEE Robotics &amp; Automation Magazine</i>, 2024.",
}
_order = []
def c(*keys):
    nums = []
    for k in keys:
        assert k in BIB, k
        if k not in _order:
            _order.append(k)
        nums.append(_order.index(k) + 1)
    nums.sort()
    return "[" + ", ".join(map(str, nums)) + "]"

# ---------------------------------------------------------------- math rendering
_eqn = [0]
def _render(latex, size, name=None):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=size, color="#111111")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=330, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    im = PILImage.open(buf)
    w, h = im.size
    buf.seek(0)
    return buf, w / 330 * 72, h / 330 * 72

def eq(latex, size=14, number=True):
    buf, w, h = _render(latex, size)
    maxw = TW - 1.6 * cm
    if w > maxw:
        h *= maxw / w; w = maxw
    img = Image(buf, width=w, height=h)
    if number:
        _eqn[0] += 1
        t = Table([[img, Paragraph(f"({_eqn[0]})", small)]], colWidths=[TW - 1.4 * cm, 1.4 * cm])
    else:
        t = Table([[img, ""]], colWidths=[TW - 1.4 * cm, 1.4 * cm])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (1, 0), (1, 0), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t

def matrix(name, rows, size=14):
    """Render  name = [rows]  as a bracketed matrix sized from measured text extents."""
    def measure(texts):
        f = plt.figure(figsize=(4, 1)); r = f.canvas.get_renderer(); out = []
        for tx in texts:
            t = f.text(0, 0, f"${tx}$", fontsize=size); out.append(t.get_window_extent(r).width / f.dpi)
        plt.close(f); return out
    nr, nc = len(rows), len(rows[0])
    widths = measure([e for r in rows for e in r])
    colw = [max(widths[i * nc + j] for i in range(nr)) + 0.28 for j in range(nc)]
    namew = measure([name])[0] + 0.18 if name else 0.0
    rowh = 0.42
    W = namew + sum(colw) + 0.3; H = rowh * nr + 0.1
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    if name:
        ax.text(0.0, H / 2, f"${name}$", fontsize=size, va="center", ha="left")
    x0 = namew + 0.04; x1 = W - 0.04
    for xs, sgn in ((x0, 1), (x1, -1)):
        ax.plot([xs + sgn * 0.08, xs, xs, xs + sgn * 0.08], [0.03, 0.03, H - 0.03, H - 0.03], color="k", lw=1.1)
    cx = x0 + 0.14
    for j in range(nc):
        for i in range(nr):
            ax.text(cx + colw[j] / 2, H - rowh * (i + 0.5) - 0.05, f"${rows[i][j]}$", ha="center", va="center", fontsize=size)
        cx += colw[j]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=330, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig); buf.seek(0)
    im = PILImage.open(buf); w, h = im.size; buf.seek(0)
    _eqn[0] += 1
    img = Image(buf, width=w / 330 * 72, height=h / 330 * 72)
    t = Table([[img, Paragraph(f"({_eqn[0]})", small)]], colWidths=[TW - 1.4 * cm, 1.4 * cm])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    return t

# ---------------------------------------------------------------- figure helpers
def fig_img(fig, name, width=TW):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    im = PILImage.open(p); w, h = im.size
    return Image(p, width=width, height=width * h / w)

def block(ax, x, y, w, h, text, fc="#dce8f7", ec="#14325c", fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(ax, p, q, text=None, color="#14325c", fs=6.6, off=(0, 0.1), ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.1, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, fontsize=fs, ha="center", color=color)

def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 5.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
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
    return fig_img(fig, "architecture.png")

def fig_geometry():
    fig, ax = plt.subplots(figsize=(8.4, 4.4)); ax.set_xlim(-1, 9); ax.set_ylim(-1.2, 4.6); ax.axis("off"); ax.set_aspect("equal")
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
    return fig_img(fig, "geometry.png", TW * 0.92)

def fig_depth(n=0.05, f=10.0):
    d = np.linspace(n * 1.001, 3.0, 800)
    db = (1 - n / d) / (1 - n / f)           # normalised depth-buffer value (0 at near, 1 at far)
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
    return fig_img(fig, "depth.png")

def fig_tracking():
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
    return fig_img(fig, "tracking.png", TW * 0.85)

def fig_dls():
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
    return fig_img(fig, "dls.png")

def fig_error():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    d = np.linspace(0.3, 1.5, 100)
    for f, col in [(250, "#14325c"), (500, "#2c6e2a"), (1000, "#8a2a2a")]:
        ax.plot(d, 0.5 * d / f * 1e3, label=f"f = {f} px", color=col, lw=2)
    ax.axvline(0.775, color="#999", ls=":"); ax.text(0.33, 2.2, "dotted line: stock camera,\nd \u2248 0.78 m above cube centre", fontsize=7, color="#555")
    ax.set_xlabel("depth d (m)"); ax.set_ylabel("worst-case lateral error (mm)")
    ax.set_title("Half-pixel quantisation error  \u03b4\u2093\u1d67 = d / (2f)", fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=.3)
    return fig_img(fig, "error.png", TW * 0.78)

def fig_features():
    fig, ax = plt.subplots(figsize=(9.2, 1.8)); ax.set_xlim(0, 30); ax.set_ylim(0, 2.1); ax.axis("off")
    groups = [("p_obj", 0, 3, "#dce8f7"), ("p_ee", 3, 6, "#e3f2e1"), ("obj_quat", 6, 10, "#f6e6c8"), ("distances", 10, 13, "#e9dcf5"),
              ("grasp_off", 13, 16, "#f5dcdc"), ("flags", 16, 20, "#fff3d6"), ("diff", 20, 21, "#d9efef"), ("conf", 21, 22, "#efe2d9"),
              ("padding (zeros)", 22, 30, "#eeeeee")]
    for name, a, b, col in groups:
        ax.add_patch(Rectangle((a, 0.7), b - a, 0.8, fc=col, ec="k", lw=0.8))
        ax.text((a + b) / 2, 1.1, name, ha="center", va="center", fontsize=6.6 if b - a > 1 else 5.4, rotation=0 if b - a > 1 else 90)
        ax.text((a + b) / 2, 0.45, f"{a}\u2013{b - 1}" if b - a > 1 else f"{a}", ha="center", fontsize=6.4, color="#555")
    ax.text(15, 1.85, "30-D spatial feature vector   p\u209c\u02e2\u1d56\u1d43\u1d57\u1da6\u1d43\u02e1 \u2208 [\u22121, 1]\u00b3\u2070  (float32)", ha="center", fontsize=8.5, fontweight="bold", color="#14325c")
    return fig_img(fig, "features.png")

def fig_timing():
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
    return fig_img(fig, "timing.png")

# ---------------------------------------------------------------- doc helpers
def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bullet, bulletText="\u2022")
def figcap(img, text): return KeepTogether([img, Paragraph(text, cap)])

def table(data, widths, header=True, zebra=True):
    rows = []
    for i, r in enumerate(data):
        rows.append([Paragraph(str(x), cellb if (header and i == 0) else cell) for x in r])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6fa")))
    t.setStyle(TableStyle(st))
    return t

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(LM, 1.2 * cm, "The Spatial Reasoning Layer \u2013 Technical Report")
    canvas.drawRightString(PAGE_W - RM, 1.2 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#cccccc")); canvas.line(LM, 1.55 * cm, PAGE_W - RM, 1.55 * cm)
    canvas.restoreState()

def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY); canvas.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 1.2 * cm, fill=1, stroke=0)
    canvas.rect(0, 0, PAGE_W, 0.8 * cm, fill=1, stroke=0)
    canvas.restoreState()

# ================================================================= CONTENT
M = json.load(open(os.path.join(ROOT, "scratch", "spatial_metrics.json")))
S = []

# ---------------- cover
S += [Spacer(1, 5.2 * cm), P("The Spatial Reasoning Layer", title_s), Spacer(1, 6),
      P("Philosophy, Architecture and Mathematical Formulation of a Depth-Grounded Perception-to-Control Interface for Context-Conditioned Robotic Manipulation", sub_s),
      Spacer(1, 1.2 * cm),
      table([["Scope", "Layer 2 of the four-layer autonomous manipulation stack (ARIA \u2192 <b>Spatial Reasoning</b> \u2192 CCDM \u2192 NMPC)"],
             ["Platform", "Franka Emika Panda (7-DOF), MuJoCo simulation, overhead RGB-D camera"],
             ["Output", "30-dimensional normalised spatial state vector and structured spatial analysis at every control step"],
             ["Document", "Motivation, design philosophy, architecture, complete mathematical derivation, validation, limitations, bibliography"]],
            [3.0 * cm, TW - 3.0 * cm], header=False, zebra=False),
      Spacer(1, 1.0 * cm),
      P("<b>Abstract.</b> Modern language- and vision-driven manipulation pipelines can decide <i>what</i> to pick and <i>where</i> to place it, yet "
        "a torque-level controller needs <i>metric, continuous, time-aligned</i> geometry: where the object centroid is in the world frame, how far the "
        "gripper is from it right now, whether it is reachable, and how hard the grasp will be. The Spatial Reasoning Layer is the module that converts "
        "sparse, asynchronous, semantic 2-D detections into this dense numerical state. It validates and tracks detections, lifts them to 3-D by "
        "deprojecting through a calibrated pinhole model using a rendered, linearised depth map, synchronises them with the robot state, and computes "
        "relative distances, workspace membership, damped-least-squares inverse-kinematic feasibility, grasp targets, collision clearance and a scalar "
        "difficulty score. These quantities are packed into a fixed 30-D vector for the downstream context-conditioned diffusion/CCDM network and NMPC. "
        "This report motivates the layer, derives each computation from first principles, states its numerical assumptions, and reports measured accuracy "
        f"({M['jit_lat_mean_mm']:.1f} mm mean lateral error under \u00b12 px detector jitter, {M['total_ms_mean']:.1f} ms mean latency on the reference scene).", body),
      NextPageTemplate("main"), PageBreak()]

# ---------------- contents
S += [P("Contents", h1)]
toc = ["1. Introduction and Need", "2. Design Philosophy", "3. System Context and Interface Contract", "4. Architecture",
       "5. Mathematical Formulation", "&nbsp;&nbsp;&nbsp;5.1 Notation and frames  \u2022  5.2 Detection validation  \u2022  5.3 Multi-object tracking",
       "&nbsp;&nbsp;&nbsp;5.4 Camera model  \u2022  5.5 Depth acquisition  \u2022  5.6 Deprojection and error analysis",
       "&nbsp;&nbsp;&nbsp;5.7 Time synchronisation  \u2022  5.8 Relative distance  \u2022  5.9 Workspace, grasp and collision",
       "&nbsp;&nbsp;&nbsp;5.10 Damped-least-squares IK  \u2022  5.11 Difficulty score  \u2022  5.12 Feature encoding",
       "&nbsp;&nbsp;&nbsp;5.13 Concurrency model  \u2022  5.14 Multi-object task sequencing",
       "6. Failure Modes and Graceful Degradation", "7. Validation and Measured Performance", "8. Limitations and Future Work", "9. Bibliography"]
for t in toc:
    S.append(P(t, ParagraphStyle("toc", parent=body, spaceAfter=2, alignment=TA_LEFT)))
S.append(PageBreak())

# ---------------- 1 introduction
S += [P("1. Introduction and Need", h1),
      P("Language-conditioned robotics has advanced rapidly: large language and vision-language models decompose a high-level instruction into sub-goals and "
        f"ground them in affordances {c('saycan')}, emit executable plans {c('cap')}, or output actions directly {c('rt1')}. Open-vocabulary detectors and "
        f"promptable segmenters {c('gdino','sam')} supply the <i>semantic</i> identity of objects in the image. What none of these provide, by themselves, "
        "is the quantity a model-predictive or torque controller actually consumes: a <b>metric, time-aligned, kinematically contextualised state</b>."),
      P("The gap can be stated precisely. A detection is a tuple (class, confidence, bounding box) in <i>pixels</i>, delivered at an irregular rate with latency "
        "and occasional false or missing entries. A controller running at 100 Hz needs, at every cycle, the object position in the robot's world frame "
        "(metres), the end-effector pose, the signed relative displacement that closes during an approach, a verdict on reachability and collision safety, "
        "and a bounded, well-scaled numerical encoding that a neural network can ingest. Bridging pixels to metres is a geometric problem "
        f"{c('hartley')}; bridging asynchronous perception to synchronous control is a systems problem {c('ros')}; deciding whether a pose is attainable is a kinematics "
        f"problem {c('siciliano','lynch')}. The Spatial Reasoning Layer (SRL) solves all three in one deterministic, inspectable module."),
      P("Why a dedicated layer?", h2),
      B("<b>Separation of semantics and geometry.</b> ARIA decides <i>which</i> objects matter; the SRL decides <i>where</i> they are. Neither needs retraining when the other changes."),
      B("<b>Closed-loop relative distance.</b> End-to-end visuomotor policies learn spatial relations implicitly from pixels "
        f"{c('levine','qtopt')}, which is data-hungry and opaque. An explicit, continuously updated relative distance to the grasp is a compact sufficient statistic for approach, slow-down and grasp-trigger logic."),
      B("<b>Safety and constraint generation.</b> Constraint-aware NMPC "
        f"{c('mayne')} requires geometric facts (clearance, workspace, table height). These must be computed, not hallucinated, and be traceable to a measurement."),
      B("<b>Network conditioning.</b> The downstream CCDM network is conditioned through FiLM-style modulation "
        f"{c('film')}; stable conditioning requires a fixed-size, bounded, normalised input. The SRL guarantees a 30-D vector in [\u22121, 1] at every call, even when perception fails."),
      B("<b>Auditability.</b> Each number in the output vector maps to a documented formula (Section 5). This is valuable for debugging, certification and scientific reproducibility."),
      P("Positioning with respect to prior art", h2),
      P(f"Spatial reasoning in manipulation has been approached by learning spatial action maps directly from images, e.g. Transporter Networks {c('transporter')}, and by "
        f"analytic grasp-quality evaluation on point clouds, e.g. Dex-Net {c('dexnet')}. The SRL is complementary: it is a <i>model-based</i> estimator that exposes "
        "interpretable geometry to a learned downstream module, in the tradition of operational-space thinking "
        f"{c('khatib')} where task-space quantities (here the object\u2013gripper displacement) drive control."),
      Spacer(1, 4),
      P("<b>Summary of need.</b> Without the SRL, the controller would either have to infer metric geometry from pixels (unreliable, unsafe) or rely on privileged simulator state "
        "(not deployable). With it, the only privileged input is the depth image that a real RGB-D sensor also provides.", box),
      PageBreak()]

# ---------------- 2 philosophy
S += [P("2. Design Philosophy", h1),
      P("Six principles guided the design. Each is reflected in concrete implementation choices that are cited in later sections.")]
S.append(table([
    ["Principle", "Meaning", "Implementation consequence"],
    ["P1. Geometry before learning", "Use closed-form, calibrated geometry wherever it exists; reserve learning for what geometry cannot express.",
     "Pinhole deprojection, SE(3) transforms, DLS-IK; no learned weights inside the layer."],
    ["P2. Metric, not pixel", "All outputs the controller sees are in SI units or normalised SI units.",
     "Depth linearised to metres; positions divided by fixed scales (2.0 m, 1.5 m, 0.5 m)."],
    ["P3. Continuity over snapshots", "The controller needs a signal defined at every timestep, with stable identity of the tracked object.",
     "Hungarian tracking with persistent IDs; initial object location locked and per-step distance updated."],
    ["P4. Never block the controller", "Perception latency must not stall control.",
     "Drop-oldest inbound queue, private mirror MjData for the worker, lock-free double-buffer output."],
    ["P5. Fail safe and fixed-shape", "Bad detections, missing depth or unreachable targets must degrade gracefully.",
     "Status codes; <i>empty</i> feature vector (flags 0, difficulty 1) instead of exceptions; output always 30-D, clamped."],
    ["P6. Observable and testable", "Every stage is timed and unit-tested against ground truth.",
     "Per-stage microsecond timings; tests bound deprojection error below 3 mm; 28 regression tests."],
], [3.4 * cm, 6.0 * cm, TW - 9.4 * cm]))
S += [Spacer(1, 8),
      P("A guiding idea is the <b>sufficient-statistics view</b>: from the point of view of an approach controller, the high-dimensional scene collapses to a handful of "
        "task-relevant numbers \u2013 the object centroid, the end-effector pose, their relative displacement, and a few binary feasibility predicates. The SRL computes exactly those, "
        "and nothing else, which keeps the CCDM input low-dimensional and the control problem well-conditioned."),
      P("A second idea is <b>explicit uncertainty accounting</b>. Detector confidence enters the difficulty score; depth is median-filtered against outliers; pixel "
        "quantisation is bounded analytically (Section 5.6). The layer does not claim more precision than the sensor supports."),
      PageBreak()]

# ---------------- 3 context
S += [P("3. System Context and Interface Contract", h1),
      P("The SRL occupies the second position of a four-layer stack. Layer 3 (CCDM) maps the SRL state, the task context and the scene to dynamic constraints and cost weights, and Layer 4 is a nonlinear MPC "
        f"formulated in CasADi {c('casadi')} and solved with IPOPT {c('ipopt')}. Figure 1 summarises the data flow; Table 2 lists the formal inputs and outputs.")]
S.append(figcap(fig_architecture(), "<b>Figure 1.</b> Position of the Spatial Reasoning Layer in the autonomous manipulation stack and its internal modules. "
                "Solid arrows are data flow; the dashed boundary is Layer 2."))
S.append(P("Table 2 \u2013 Inputs and outputs", h3))
S.append(table([
    ["Dir.", "Item", "Type / unit", "Source or sink"],
    ["In", "Detection message: timestamp, primary_target, secondary_objects (id, class, confidence, bbox, optional quat)", "JSON; bbox in px, confidence \u2208 [0,1]", "ARIA (Layer 1) \u2013 in simulation, emulated from MuJoCo ground truth"],
    ["In", "Robot state snapshot: q, sim time, end-effector pose, base position", "rad, s, m", "Physics loop, copied after each mj_step"],
    ["In", "Rendered depth buffer from overhead_cam", "raw d<sub>b</sub> \u2208 [0,1]", "MuJoCo renderer (RGB-D sensor in deployment)"],
    ["In", "Calibration: K, distortion, T<sub>cw</sub>; object class size table", "px, m", "config/default_config.yaml"],
    ["Out", "SpatialResult: p<sub>obj</sub>, p<sub>ee</sub>, q<sub>ee</sub>, track id, confidence, SpatialAnalysis, status, timings", "m, unitless, \u00b5s", "Integrated pipeline / logging"],
    ["Out", "Feature vector p<sub>t</sub><super>spatial</super>", "float32[30] in [\u22121, 1]", "CCDM (Layer 3) and NMPC context"],
], [1.9 * cm, 6.2 * cm, 3.6 * cm, TW - 11.7 * cm]))
S += [Spacer(1, 6),
      P("The ARIA message follows a fixed schema: <font name='Mono'>{\"timestamp\", \"primary_target\": {\"id\", \"class\", \"confidence\", \"bbox\"}, \"secondary_objects\": [...]}</font>, "
        "with <font name='Mono'>bbox = [x<sub>min</sub>, y<sub>min</sub>, x<sub>max</sub>, y<sub>max</sub>]</font> in pixels. The <i>primary</i> target is the object the "
        "current sub-task requires; <i>secondary</i> objects are other detections used as collision obstacles. The multi-agent interface additionally provides, outside the SRL, "
        "the semantic task description (pick target, place target, operational context) consumed by the CCDM; the SRL itself is agnostic to it."),
      PageBreak()]

# ---------------- 4 architecture
S += [P("4. Architecture", h1),
      P("The SRL is implemented as a set of small, single-responsibility modules coordinated by a master engine. This section describes each module's role; the "
        "mathematics is deferred to Section 5.")]
S.append(table([
    ["Module", "Responsibility", "Key state"],
    ["<b>ObjectDetectionProcessor</b>", "Parse and validate the JSON message; reject malformed or low-confidence entries; assign persistent track IDs by Hungarian assignment with a constant-velocity predictor.", "Track table (\u2264 5), per-track centroid, EMA velocity, miss counter"],
    ["<b>MuJoCoStateManager</b>", "Snapshot the robot state cheaply on the physics thread; on the worker, apply a snapshot to a private MjData, compute end-effector pose, render depth and linearise it.", "Live and mirror MjData; z<sub>near</sub>, z<sub>far</sub>; depth cache"],
    ["<b>CoordinateTransformer</b>", "Implement the camera model: deprojection, projection, distortion/undistortion, SE(3) extrinsics, robust depth sampling, centroid correction.", "K, T<sub>cw</sub>, T<sub>wc</sub>, distortion"],
    ["<b>SpatialAnalyzer</b>", "Relative distances, workspace test, damped-least-squares IK with warm start, grasp target, clearance check, difficulty score.", "Private IK MjData, last IK solution"],
    ["<b>FeatureEncoder</b>", "Normalise and pack all outputs into the 30-D vector; produce the safe \u2018empty\u2019 vector on failure.", "Scales (pos, dist, grasp)"],
    ["<b>SpatialReasoningLayer</b>", "Master engine: owns the modules, the worker thread, the inbound queue and the outbound double buffer; exposes <font name='Mono'>sync_state</font>, <font name='Mono'>submit_detection</font>, <font name='Mono'>get_latest_features</font>, <font name='Mono'>process_frame</font>.", "Queue, buffer, state history (128)"],
], [4.0 * cm, 8.4 * cm, TW - 12.4 * cm]))
S += [Spacer(1, 6), P("Processing pipeline", h2),
      P("A single call to <font name='Mono'>process_frame</font> executes seven ordered stages, each separately timed (microsecond resolution): "
        "(1) parse, validate and track; (2) select and apply the state snapshot closest in time to the detection; (3) render and linearise depth; "
        "(4) deproject primary and secondary detections; (5) spatial analysis (distance, workspace, IK, grasp, collision, difficulty); "
        "(6) feature encoding; (7) publish to the output double buffer. Two early exits exist: <i>no_valid_primary</i> (no admissible primary detection) and "
        "<i>no_depth</i> (no valid depth near the bounding-box centre). In both cases an empty, shape-correct feature vector is published so the controller never starves."),
      figcap(fig_timing(), "<b>Figure 2.</b> Asynchronous execution model. ARIA publishes irregularly; the physics thread only snapshots state; the SRL worker converts the newest "
             "detection to features; the controller reads the most recent completed result at its own rate."),
      PageBreak()]

# ---------------- 5 math
S += [P("5. Mathematical Formulation", h1)]
S += [P("5.1 Notation and coordinate frames", h2),
      P("Scalars are italic, vectors bold lower-case, matrices bold upper-case. Three frames are used: the world frame <b>W</b> (z up, origin on the robot base plane), the "
        "camera frame <b>C</b> following the OpenCV optical convention (x right, y down, z forward along the optical axis) "
        f"{c('hartley')}, and the image frame with continuous pixel coordinates (u, v). A rigid transform from frame B to frame A is written "
        "<b>T</b><sub>AB</sub> \u2208 SE(3).")]
S.append(table([
    ["Symbol", "Meaning", "Unit"],
    ["(u, v)", "Continuous pixel coordinates; centre of pixel (i, j) is (i+½, j+½)", "px"],
    ["K; f<sub>x</sub>, f<sub>y</sub>, c<sub>x</sub>, c<sub>y</sub>", "Intrinsic matrix; focal lengths and principal point", "px"],
    ["R<sub>cw</sub>, t<sub>cw</sub>", "Camera-to-world rotation and translation (camera origin in world)", "\u2013, m"],
    ["d", "Planar depth along the optical axis (z<sub>c</sub>), <i>not</i> Euclidean range", "m"],
    ["p<sub>obj</sub>, p<sub>ee</sub>, p<sub>base</sub>", "Object centroid, end-effector and robot-base positions in W", "m"],
    ["q \u2208 \u211d\u2077", "Arm joint vector", "rad"],
    ["J(q) \u2208 \u211d\u00b3\u02e3\u2077", "Positional Jacobian of the end-effector body", "m/rad"],
    ["\u03b3<sub>i</sub>, \u03c4", "Detector confidence; tracking gate radius", "\u2013, px"],
], [4.6 * cm, TW - 6.6 * cm, 2.0 * cm]))

# 5.2
S += [P("5.2 Detection validation and admissibility", h2),
      P("A detection <i>i</i> with confidence \u03b3<sub>i</sub>, class label and box b<sub>i</sub> = (x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>, y<sub>2</sub>) is admitted only if it satisfies the "
        "indicator below, where (W, H) is the image size and \u03b3<sub>min</sub> = 0.70 the confidence threshold. Non-finite values are rejected before the comparisons. "
        "An optional orientation quaternion is accepted only if finite, of length four and non-zero, and is then normalised.")]
S.append(eq(r"\mathbb{1}_{\mathrm{ok}}(i)=\left[\gamma_{\min}\leq\gamma_i\leq 1\right]\cdot\left[0\leq x_1<x_2\leq W\right]\cdot\left[0\leq y_1<y_2\leq H\right]".replace(r"\mathbb{1}", r"\mathbf{1}")))
S.append(P("The bounding-box centre and size used throughout are"))
S.append(eq(r"\mathbf{c}_i=\left(\frac{x_1+x_2}{2},\;\frac{y_1+y_2}{2}\right),\qquad \mathbf{s}_i=\left(x_2-x_1,\;y_2-y_1\right)"))
S.append(P("Rejected entries are counted (<font name='Mono'>total_rejected</font>) but never raise exceptions, so a single corrupt entry in a frame cannot discard the valid ones."))

# 5.3 tracking
S += [P("5.3 Multi-object tracking by Hungarian assignment", h2),
      P(f"Persistent identity is required so that the same physical object is followed from frame to frame, and so that a transient mis-detection of a <i>different</i> object is not "
        f"mistaken for the target. The tracker follows the tracking-by-detection paradigm of SORT {c('sort')} and DeepSORT {c('deepsort')} but uses a deliberately minimal "
        f"constant-velocity motion model on the pixel centroid, since the tabletop scene has slow, well-separated objects and no appearance embedding is available."),
      P("<b>State and prediction.</b> Track <i>j</i> stores the last centroid <b>m</b><sub>j</sub> and a smoothed pixel velocity <b>v</b><sub>j</sub>. The one-step prediction is")]
S.append(eq(r"\hat{\mathbf{m}}_j^{\,t+1}=\mathbf{m}_j^{\,t}+\mathbf{v}_j^{\,t}"))
S.append(P(f"This is the deterministic mean-propagation step of a constant-velocity Kalman model {c('kalman','barshalom')} with the covariance machinery omitted. "
           "<b>Cost matrix.</b> For N tracks and M detections the assignment cost is the Euclidean centroid distance, with a large penalty L = 10<super>6</super> forbidding "
           "class-inconsistent pairings:"))
S.append(eq(r"C_{ji}=\begin{cases}\end{cases}".replace(r"\begin{cases}\end{cases}", r"\left\|\hat{\mathbf{m}}_j-\mathbf{c}_i\right\|_2\;\;\mathrm{if\ class}(j)=\mathrm{class}(i),\qquad L\;\;\mathrm{otherwise}")))
S.append(P(f"<b>Optimal assignment.</b> The one-to-one assignment minimising total cost is the linear assignment problem, solved exactly in O(n<super>3</super>) time by the Hungarian method {c('kuhn')}:"))
S.append(eq(r"\pi^{\star}=\arg\min_{\pi}\sum_{j}C_{j,\pi(j)}"))
S.append(P("<b>Gating.</b> An assigned pair is accepted only if C<sub>j,\u03c0*(j)</sub> \u2264 \u03c4 (default 40 px). This prevents a track from jumping to a distant, unrelated detection (Figure 3). "
           "<b>Update.</b> On acceptance, with EMA factor \u03b1 = 0.5:"))
S.append(eq(r"\mathbf{v}_j\leftarrow\alpha\left(\mathbf{c}_i-\mathbf{m}_j\right)+(1-\alpha)\,\mathbf{v}_j,\qquad \mathbf{m}_j\leftarrow\mathbf{c}_i,\qquad \mathrm{missed}_j\leftarrow 0"))
S.append(P("<b>Lifecycle.</b> Unmatched tracks have their miss counter incremented and their velocity halved (v \u2190 v/2) so that a lost track does not extrapolate indefinitely; "
           "a track is deleted when its miss count exceeds 5 frames. Unmatched detections spawn new tracks, primary target first and then by descending confidence. "
           "The table holds at most 5 tracks; if full, a <i>primary</i> detection may evict the track with the largest miss count, but a secondary detection may not."))
S.append(figcap(fig_tracking(), "<b>Figure 3.</b> Tracking geometry. Hollow circles are predicted centroids with the gate radius \u03c4; filled squares are new detections. "
                "Green links are accepted assignments; the third detection lies outside the gate and spawns a new track."))

# 5.4 camera
S += [P("5.4 Camera model", h2),
      P(f"The camera is modelled as a pinhole with optional Brown\u2013Conrady lens distortion {c('brown','zhang','hartley')}. The intrinsic matrix is")]
S.append(matrix(r"\mathbf{K}=", [[r"f_x", "0", r"c_x"], ["0", r"f_y", r"c_y"], ["0", "0", "1"]]))
S.append(P("The stock configuration uses a 320\u00d7240 image with f<sub>x</sub> = f<sub>y</sub> = 250 px and principal point (160, 120). "
           "The extrinsics are the camera-to-world rigid transform"))
S.append(matrix(r"\mathbf{T}_{cw}=", [[r"\mathbf{R}_{cw}", r"\mathbf{t}_{cw}"], [r"\mathbf{0}^{\top}", "1"]]))
S.append(P("For the stock overhead camera, t<sub>cw</sub> = (0.6, 0, 1.2) m and the optical axis points straight down (\u2212z<sub>w</sub>). "
           "Because configuration quaternions are stored with rounded entries (e.g. 0.7071), the rotation obtained from the quaternion is not exactly orthonormal. "
           "The layer therefore projects it onto SO(3) with an SVD (the closest rotation in the Frobenius sense):"))
S.append(eq(r"\mathbf{R}_{cw}\leftarrow\mathbf{U}\mathbf{V}^{\top},\qquad \mathbf{R}_{\mathrm{raw}}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top},\quad \det(\mathbf{U}\mathbf{V}^{\top})=+1"))
S.append(P("(if the determinant is \u22121 the last column of U is negated). The inverse transform follows from the SE(3) structure:"))
S.append(eq(r"\mathbf{T}_{wc}=\mathbf{T}_{cw}^{-1}:\qquad \mathbf{R}_{wc}=\mathbf{R}_{cw}^{\top},\quad \mathbf{t}_{wc}=-\mathbf{R}_{cw}^{\top}\mathbf{t}_{cw}"))
S.append(P("<b>Distortion.</b> With normalised coordinates x<sub>n</sub> = X<sub>c</sub>/Z<sub>c</sub>, y<sub>n</sub> = Y<sub>c</sub>/Z<sub>c</sub> and r<sup>2</sup> = x<sub>n</sub><sup>2</sup> + y<sub>n</sub><sup>2</sup>, the forward distortion model with "
           "radial coefficients k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub> and tangential coefficients p<sub>1</sub>, p<sub>2</sub> is"))
S.append(eq(r"x_d=x_n\,\rho+2p_1x_ny_n+p_2\left(r^2+2x_n^2\right)"))
S.append(eq(r"y_d=y_n\,\rho+p_1\left(r^2+2y_n^2\right)+2p_2x_ny_n,\qquad \rho=1+k_1r^2+k_2r^4+k_3r^6"))
S.append(P("Projection to pixels is u = f<sub>x</sub>x<sub>d</sub> + c<sub>x</sub>, v = f<sub>y</sub>y<sub>d</sub> + c<sub>y</sub>. Deprojection requires the inverse of the distortion map, "
           "which has no closed form; the layer uses the standard fixed-point iteration, initialised at the distorted point and run for 8 iterations (convergent for the mild distortion typical of rectified RGB-D sensors):"))
S.append(eq(r"x^{(k+1)}=\frac{x_d-\Delta_x\!\left(x^{(k)},y^{(k)}\right)}{\rho\!\left(x^{(k)},y^{(k)}\right)},\qquad y^{(k+1)}=\frac{y_d-\Delta_y\!\left(x^{(k)},y^{(k)}\right)}{\rho\!\left(x^{(k)},y^{(k)}\right)}"))
S.append(P("where \u0394<sub>x</sub>, \u0394<sub>y</sub> are the tangential terms. In the default configuration all distortion coefficients are zero and this step is skipped."))
S.append(figcap(fig_geometry(), "<b>Figure 4.</b> Imaging geometry. A world point projects through the camera centre onto the image plane at pixel (u, v); "
                "the <i>planar</i> depth d is the projection of the ray onto the optical axis, which is what the depth buffer stores after linearisation."))

# 5.5 depth
S += [P("5.5 Depth acquisition and linearisation", h2),
      P(f"The simulated RGB-D camera exposes the OpenGL depth buffer. Under perspective projection with near and far clipping planes n and f, the buffer value d<sub>b</sub> \u2208 [0, 1] "
        f"is a <i>non-linear</i> function of metric depth {c('foley')}. Writing the standard mapping with d<sub>b</sub> = 0 at the near plane:")]
S.append(eq(r"d_b=\frac{1-n/d}{1-n/f}\quad\Longleftrightarrow\quad d=\frac{n\,f}{f-d_b\,(f-n)}"))
S.append(P(f"MuJoCo {c('mujoco')} expresses the clipping planes relative to the model extent \u2113 (<font name='Mono'>model.stat.extent</font>): "
           "n = znear\u00b7\u2113 and f = zfar\u00b7\u2113, where znear and zfar are the values in <font name='Mono'>model.vis.map</font>. The layer reads the depth buffer in the "
           "\u2018zero-near\u2019 convention and applies the inverse mapping above element-wise to obtain a metric depth image <b>D</b>. "
           "The buffer is non-uniform: resolution degrades quadratically with distance. Differentiating,"))
S.append(eq(r"\frac{\partial d}{\partial d_b}=\frac{d^{2}\,(f-n)}{n\,f}\;\approx\;\frac{d^{2}}{n}\quad(f\gg n)\qquad\Rightarrow\qquad \delta d\approx\frac{d^{2}}{n\,2^{B}}\;\;(B\text{-bit buffer})".replace(r"\text{-bit buffer}", r"\ \mathrm{bits}")))
S.append(P("so a larger n (tight clipping) coarsens depth, which motivates configuring znear as small as numerically safe (Figure 5). "
           "To suppress isolated outliers and edge artefacts, depth at a continuous pixel (u, v) is estimated as the <b>median</b> over a (2r+1)\u00b2 window "
           "(r = 1, a 3\u00d73 patch) centred at (\u230au\u230b, \u230av\u230b), after discarding values outside (0.05, 10) m:"))
S.append(eq(r"\hat{d}(u,v)=\mathrm{median}\left\{D[r',c']\;:\;|r'-\lfloor v\rfloor|\leq r,\;|c'-\lfloor u\rfloor|\leq r,\;d_{\min}<D[r',c']<d_{\max}\right\}"))
S.append(P("If the valid set is empty the localisation fails with status <i>no_depth</i>. The floor indexing follows from the pixel-centre convention (pixel (i, j) covers [i, i+1)\u00d7[j, j+1)). "
           "<b>Surface-to-centroid correction.</b> A depth sensor measures the visible <i>surface</i>, whereas control targets the object <i>centroid</i>. "
           "For an object with full-extent box s \u2208 \u211d\u00b3 (from the class table; the cube is 5 cm) and orientation R<sub>o</sub>, let a = R<sub>cw</sub>e<sub>3</sub> be the optical axis in W. "
           "The half-thickness of the box along the viewing direction is"))
S.append(eq(r"\delta_c=\frac{1}{2}\sum_{k=1}^{3}\left|\left(\mathbf{R}_o^{\top}\mathbf{a}\right)_k\right|\,s_k,\qquad d\leftarrow\hat{d}+\delta_c"))
S.append(P("For a top-down camera and an axis-aligned cube this evaluates to \u03b4<sub>c</sub> = 0.025 m. Objects with unknown size (default class) receive \u03b4<sub>c</sub> = 0."))
S.append(figcap(fig_depth(), "<b>Figure 5.</b> Left: the depth-buffer value is highly non-linear in metric depth, so it must be linearised before use. "
                "Right: depth resolution of a 24-bit buffer for several near-plane distances (illustrative values)."))

# 5.6 deprojection
S += [P("5.6 Deprojection to the world frame and error analysis", h2),
      P("Given the pixel centre (u, v) of the primary bounding box and the corrected planar depth d, the camera-frame point and world point are")]
S.append(eq(r"\mathbf{p}_c=\left(\frac{(u-c_x)\,d}{f_x},\;\frac{(v-c_y)\,d}{f_y},\;d\right)^{\top},\qquad \mathbf{p}_w=\mathbf{R}_{cw}\,\mathbf{p}_c+\mathbf{t}_{cw}"))
S.append(P("Secondary detections are lifted identically in a vectorised form: for N pixels stacked in <b>U</b> \u2208 \u211d<super>N\u00d72</super> with depths <b>d</b> \u2208 \u211d<super>N</super>, "
           "<b>P</b><sub>w</sub> = <b>P</b><sub>c</sub><b>R</b><sub>cw</sub><super>\u22a4</super> + <b>1</b><b>t</b><sub>cw</sub><super>\u22a4</super>. The inverse map (world to pixel and depth) "
           "is used both for synthesising detections in simulation and for round-trip verification: p<sub>c</sub> = R<sub>wc</sub>p<sub>w</sub> + t<sub>wc</sub>, (u, v) = (f<sub>x</sub>X<sub>c</sub>/Z<sub>c</sub> + c<sub>x</sub>, f<sub>y</sub>Y<sub>c</sub>/Z<sub>c</sub> + c<sub>y</sub>), d = Z<sub>c</sub>."))
S.append(P("<b>First-order error propagation.</b> Let the pixel and depth errors be independent with variances \u03c3<sub>u</sub><super>2</super>, \u03c3<sub>v</sub><super>2</super>, \u03c3<sub>d</sub><super>2</super>. "
           "For the axis-aligned stock overhead camera, R<sub>cw</sub> is a signed permutation matrix, so the world-frame covariance is a re-ordering of the camera-frame covariance, which is obtained from the Jacobian of p<sub>c</sub>:"))
S.append(matrix(r"\mathbf{J}_{p}=\frac{\partial\mathbf{p}_c}{\partial(u,v,d)}=", [[r"d/f_x", "0", r"(u-c_x)/f_x"], ["0", r"d/f_y", r"(v-c_y)/f_y"], ["0", "0", "1"]]))
S.append(eq(r"\boldsymbol{\Sigma}_{w}=\mathbf{R}_{cw}\,\mathbf{J}_{p}\,\mathrm{diag}\!\left(\sigma_u^{2},\sigma_v^{2},\sigma_d^{2}\right)\mathbf{J}_{p}^{\top}\,\mathbf{R}_{cw}^{\top}"))
S.append(P("Two consequences follow. (i) The lateral error scales linearly with depth and inversely with focal length; for half-pixel quantisation "
           "(|\u03b4u|, |\u03b4v| \u2264 \u00bd px) the worst-case lateral error is"))
S.append(eq(r"\left|\delta x\right|,\;\left|\delta y\right|\;\leq\;\frac{d}{2f}\;+\;\frac{|u-c_x|}{f}\,|\delta d|"))
S.append(P("which for f = 250 px and d \u2248 0.78 m is 1.6 mm (Figure 6), consistent with the < 3 mm bound enforced by the test-suite. (ii) The depth error propagates one-to-one "
           "into the vertical coordinate for a top-down camera, so the centroid correction \u03b4<sub>c</sub> must be accurate: an error in the assumed object height maps directly to an error in z."))
S.append(figcap(fig_error(), "<b>Figure 6.</b> Analytic worst-case lateral localisation error due to half-pixel quantisation for several focal lengths."))

# 5.7 time sync
S += [P("5.7 Time synchronisation of perception and state", h2),
      P("Detections carry the capture timestamp t<sub>d</sub>, whereas the robot continues to move while the detection is processed. Pairing an old image with the <i>current</i> "
        "joint state would corrupt the relative geometry. The physics loop therefore pushes a lightweight snapshot s<sub>k</sub> = (t<sub>k</sub>, q<sub>k</sub>, q\u0307<sub>k</sub>) into a ring buffer of capacity 128 after each step, "
        "and the worker selects the snapshot nearest in time:")]
S.append(eq(r"k^{\star}=\arg\min_{k}\left|t_k-t_d\right|"))
S.append(P("The worker applies s<sub>k*</sub> to a <i>private mirror</i> of the simulation data and recomputes forward kinematics there; the live data are never touched, so no locking is needed on the physics thread. "
           "Within the 128-sample window (1.28 s at 100 Hz) the residual synchronisation error is bounded by half the physics sampling interval, \u0394t \u2264 \u00bd\u00b7(1/100 Hz) = 5 ms; "
           "at a peak end-effector speed v<sub>max</sub> the corresponding position error is \u2264 v<sub>max</sub>\u00b75 ms."))

# 5.8 relative distance
S += [P("5.8 Continuous relative distance", h2),
      P("The central output of the layer, as specified for the control stack, is the displacement between the object and the end-effector expressed through three scalar distances, "
        "with \u0394 = p<sub>obj</sub> \u2212 p<sub>ee</sub> = (\u0394<sub>x</sub>, \u0394<sub>y</sub>, \u0394<sub>z</sub>):")]
S.append(eq(r"d_{\mathrm{euc}}=\sqrt{\Delta_x^{2}+\Delta_y^{2}+\Delta_z^{2}},\qquad d_{h}=\sqrt{\Delta_x^{2}+\Delta_y^{2}},\qquad d_{v}=\left|\Delta_z\right|"))
S.append(P("The horizontal/vertical decomposition reflects the structure of a pick: approach in the plane at a safe height (d<sub>h</sub> shrinks), then descend (d<sub>v</sub> shrinks). "
           "Because p<sub>ee</sub> is recomputed from the synchronised state at <i>every</i> frame while p<sub>obj</sub> is either re-measured or held, d is defined at every control step. "
           "The first sufficiently valid localisation of a sub-task <i>locks</i> the object's initial position p<sub>obj</sub><super>0</super>; this is reported separately as the initial location. "
           "The closing speed used for approach logic follows from finite differences of the distance between consecutive steps with period \u0394t:"))
S.append(eq(r"\dot{d}_{\mathrm{euc}}^{(t)}\approx\frac{d_{\mathrm{euc}}^{(t)}-d_{\mathrm{euc}}^{(t-1)}}{\Delta t}"))
S.append(P("A negative value indicates the gripper is converging. During carrying, the grasped object is rigidly attached to the gripper, so the \u2018relative distance to the grasp\u2019 is "
           "naturally near zero and the corresponding quantity of interest becomes the distance to the <i>place</i> receptacle, which the integrated pipeline also reports."))

# 5.9 workspace grasp collision
S += [P("5.9 Workspace validity, grasp target and collision clearance", h2),
      P("<b>Workspace test.</b> The target is declared inside the arm's usable workspace if its radial distance from the base origin lies in an annulus [r<sub>min</sub>, r<sub>max</sub>] = [0.15, 0.85] m. "
        "The inner bound excludes the self-collision region around the base; the outer bound approximates the kinematic reach (the Panda has a nominal reach of 0.855 m "
        f"{c('franka')}):")]
S.append(eq(r"\mathrm{ws}(\mathbf{p}_{obj})=\left[\,r_{\min}\leq\left\|\mathbf{p}_{obj}-\mathbf{p}_{base}\right\|_2\leq r_{\max}\right]"))
S.append(P("<b>Grasp target.</b> The grasp point is the centroid offset by a class-dependent vector o (default 2 cm along +z for a top-down grasp):"))
S.append(eq(r"\mathbf{p}_{g}=\mathbf{p}_{obj}+\mathbf{o}_{\mathrm{class}}"))
S.append(P("<b>Collision clearance.</b> Using the secondary detections {p<sub>k</sub>} as obstacles, a grasp is collision-free if it is above the table plane z<sub>tab</sub> and at least \u03b4<sub>c</sub> = 0.06 m "
           "from every other object (non-finite obstacle positions are ignored):"))
S.append(eq(r"\mathrm{free}(\mathbf{p}_g)=\left[p_{g,z}\geq z_{\mathrm{tab}}\right]\cdot\prod_{k}\left[\left\|\mathbf{p}_k-\mathbf{p}_g\right\|_2\geq\delta_c\right]"))

# 5.10 IK
S += [P("5.10 Reachability by damped-least-squares inverse kinematics", h2),
      P(f"Workspace membership is necessary but not sufficient: joint limits and the arm's kinematic structure can still make a pose unreachable. The layer therefore solves a "
        f"fast position-only inverse-kinematics problem as a <i>feasibility oracle</i>. With forward kinematics f(q) giving the end-effector position, find q such that f(q) = p<sub>obj</sub> subject to q<sub>lo</sub> \u2264 q \u2264 q<sub>hi</sub> "
        f"(limits read from the model). The position Jacobian is J = \u2202f/\u2202q \u2208 \u211d<super>3\u00d77</super> {c('siciliano','craig')}. Because the arm has 7 DOF for a 3-D task, J is redundant (rank \u2264 3) and near singularities its right pseudo-inverse "
        f"has unbounded gain. The remedy is the damped-least-squares (DLS) or Levenberg\u2013Marquardt method {c('wampler','nakamura','levenberg','marquardt')}.")]
S.append(P("<b>Derivation.</b> At each iteration the position error is e = p<sub>obj</sub> \u2212 f(q). The damped step minimises a trade-off between tracking error and step size:"))
S.append(eq(r"\Delta\mathbf{q}^{\star}=\arg\min_{\Delta\mathbf{q}}\;\left\|\mathbf{e}-\mathbf{J}\,\Delta\mathbf{q}\right\|_2^{2}+\lambda^{2}\left\|\Delta\mathbf{q}\right\|_2^{2}"))
S.append(P("Setting the gradient to zero gives the normal equations (J<super>\u22a4</super>J + \u03bb<super>2</super>I<sub>7</sub>)\u0394q = J<super>\u22a4</super>e. By the push-through identity "
           "J<super>\u22a4</super>(JJ<super>\u22a4</super> + \u03bb<super>2</super>I<sub>3</sub>) = (J<super>\u22a4</super>J + \u03bb<super>2</super>I<sub>7</sub>)J<super>\u22a4</super>, this is equivalent to the much smaller 3\u00d73 system actually solved by the layer:"))
S.append(eq(r"\Delta\mathbf{q}=\mathbf{J}^{\top}\left(\mathbf{J}\mathbf{J}^{\top}+\lambda^{2}\mathbf{I}_3\right)^{-1}\mathbf{e}"))
S.append(P(f"<b>Why damping works.</b> Let J = U\u03a3V<super>\u22a4</super> with singular values \u03c3<sub>i</sub>. The DLS inverse acts on each singular direction with gain {c('buss')}"))
S.append(eq(r"g_i(\lambda)=\frac{\sigma_i}{\sigma_i^{2}+\lambda^{2}}\;\leq\;\frac{1}{2\lambda},\qquad \lim_{\lambda\to 0}g_i=\frac{1}{\sigma_i}\;\;(\mathrm{pseudo\text{-}inverse})".replace(r"\mathrm{pseudo\text{-}inverse}", r"\mathrm{pseudo\ inverse}")))
S.append(P("The pseudo-inverse gain 1/\u03c3<sub>i</sub> diverges as \u03c3<sub>i</sub> \u2192 0, whereas the damped gain is uniformly bounded by 1/(2\u03bb), attained at \u03c3<sub>i</sub> = \u03bb (Figure 7). "
           "The price is a small bias in the directions with \u03c3<sub>i</sub> much smaller than \u03bb, which is acceptable for a feasibility test."))
S.append(figcap(fig_dls(), "<b>Figure 7.</b> Effect of damping on the inverse-kinematics gain. Left: gain versus singular value for several \u03bb. Right: near a singularity (\u03c3 = 0.05) the undamped gain is 20 whereas the damped gain is bounded."))
S.append(P("<b>Practical safeguards implemented.</b> (a) <i>Step clamp:</i> if \u2016\u0394q\u2016 > \u0394<sub>max</sub> = 0.5 rad the step is rescaled, \u0394q \u2190 \u0394q\u00b7\u0394<sub>max</sub>/\u2016\u0394q\u2016. "
           "(b) <i>Joint-limit projection:</i> q \u2190 clip(q + \u0394q, q<sub>lo</sub>, q<sub>hi</sub>) every iteration. "
           "(c) <i>Adaptive damping:</i> a step that reduces the error is accepted and \u03bb \u2190 max(\u03bb/2, 10<super>\u22124</super>); a step that does not is rejected and \u03bb \u2190 4\u03bb, the classical Levenberg\u2013Marquardt trust-region heuristic. "
           "(d) <i>Budget:</i> at most 5 iterations or 0.8 ms of wall time, whichever occurs first, with termination also when \u2016e\u2016 \u2264 0.1\u00b7tol. "
           "(e) <i>Success criterion:</i> \u2016e\u2016 \u2264 tol = 0.01 m with all joints within limits."))
S.append(P("<b>Warm start.</b> Consecutive frames have nearly identical targets, so the previous solution is used as the seed when the new target lies within 0.1 m of the last one; "
           "if that fails and more than 25 % of the time budget remains, a second attempt is made from the current pose and the better of the two is kept. The solution is cached only on success. "
           "The solver runs on a private MjData, so it never perturbs the simulation. The overall result of this stage is the feasibility predicate"))
S.append(eq(r"\mathrm{reach}=\mathrm{ws}(\mathbf{p}_{obj})\;\wedge\;\left[\left\|\mathbf{e}_{\mathrm{IK}}\right\|\leq\mathrm{tol}\right]\;\wedge\;\left[\mathbf{q}^{\star}\in[\mathbf{q}_{lo},\mathbf{q}_{hi}]\right]"))

# 5.11 difficulty
S += [P("5.11 Grasp difficulty score", h2),
      P("A scalar in [0, 1] summarises how demanding the grasp is expected to be. It is a convex combination of normalised distance, perceptual uncertainty and kinematic infeasibility, with weights "
        "(0.4, 0.3, 0.3):")]
S.append(eq(r"\mathcal{D}=\mathrm{clip}\!\left(0.4\,\min\!\left(\frac{d_{\mathrm{euc}}}{d_{s}},1\right)+0.3\,(1-\gamma)+0.3\,\left(1-\mathbf{1}_{\mathrm{IK}}\right),\;0,\;1\right),\qquad d_s=1.5\ \mathrm{m}"))
S.append(P("Distance captures travel effort, (1 \u2212 \u03b3) captures detector uncertainty, and the IK indicator penalises poses the arm cannot attain. The score is consumed by the CCDM as a context feature "
           "that can, for example, increase safety margins for difficult grasps. The weights are design constants chosen so that infeasibility alone (0.3) outweighs moderate distance, and can be re-tuned in configuration."))

# 5.12 encoding
S += [P("5.12 Feature encoding", h2),
      P("All outputs are packed into a fixed-length float32 vector. Positions and distances are divided by fixed physical scales chosen to bound typical values in [\u22121, 1], and the result is clamped as a final safeguard. "
        "With scales s<sub>p</sub> = 2.0 m, s<sub>d</sub> = 1.5 m, s<sub>g</sub> = 0.5 m and \u03a0(x) = clip(x, \u22121, 1):")]
S.append(eq(r"\mathbf{p}^{\mathrm{spatial}}=\Pi\!\left(\left[\frac{\mathbf{p}_{obj}}{s_p},\;\frac{\mathbf{p}_{ee}}{s_p},\;\hat{\mathbf{q}}_{o},\;\frac{(d_{\mathrm{euc}},d_h,d_v)}{s_d},\;\frac{\mathbf{o}}{s_g},\;\mathbf{f},\;\mathcal{D},\;\gamma,\;\mathbf{0}_8\right]\right)\in[-1,1]^{30}"))
S.append(P("where the orientation entry is the unit object quaternion (identity if unavailable) and <b>f</b> \u2208 {0, 1}<super>4</super> are the flags (workspace, IK-feasible, collision-free, limits-ok). "
           "The layout is shown in Figure 8 and Table 3. Non-finite values are mapped to 0 (NaN) or \u00b11 (\u00b1\u221e) before clamping. The eight trailing zero entries are reserved for future features (e.g. object velocity) so the interface remains stable."))
S.append(figcap(fig_features(), "<b>Figure 8.</b> Layout of the 30-dimensional spatial feature vector."))
S.append(P("Table 3 \u2013 Feature layout", h3))
S.append(table([
    ["Indices", "Name", "Content", "Normalisation"],
    ["0\u20132", "p_obj", "Object centroid, world frame", "/ 2.0 m"],
    ["3\u20135", "p_ee", "End-effector position, world frame", "/ 2.0 m"],
    ["6\u20139", "obj_quat", "Object orientation [w, x, y, z]", "unit norm"],
    ["10\u201312", "distances", "d<sub>euc</sub>, d<sub>h</sub>, d<sub>v</sub>", "/ 1.5 m"],
    ["13\u201315", "grasp_offset", "Grasp offset vector", "/ 0.5 m"],
    ["16\u201319", "flags", "workspace, ik_feasible, collision_free, limits_ok", "{0, 1}"],
    ["20", "difficulty", "Grasp difficulty score D", "[0, 1]"],
    ["21", "confidence", "Detector confidence \u03b3", "[0, 1]"],
    ["22\u201329", "padding", "Reserved", "0"],
], [1.8 * cm, 2.9 * cm, TW - 8.3 * cm, 3.6 * cm]))
S.append(Spacer(1, 4))
S.append(P("<b>Empty vector.</b> When perception fails the layer emits a valid, shape-correct vector with the end-effector position retained, all flags at 0, quaternion identity and difficulty = 1 (maximal): "
           "the downstream network is told explicitly that there is <i>nothing to grasp</i> rather than receiving stale or undefined data."))

# 5.13 concurrency
S += [P("5.13 Concurrency model and latency bound", h2),
      P("The layer is a single-producer/single-consumer pipeline engineered so that neither side ever waits on the other. Three primitives are used: "
        "(i) a <b>LatestQueue</b> of size 2 holding inbound detection messages, with a drop-oldest policy so the worker always processes the freshest frame; "
        "(ii) a <b>StateHistory</b> ring buffer (Section 5.7); (iii) a <b>DoubleBuffer</b> for outbound results: the worker writes the back buffer and atomically swaps, so a reader obtains a complete, self-consistent result without locking.")]
S.append(P("<b>Latency model.</b> Let the stage times be \u03c4<sub>parse</sub>, \u03c4<sub>state</sub>, \u03c4<sub>depth</sub>, \u03c4<sub>deproj</sub>, \u03c4<sub>ik</sub>, \u03c4<sub>enc</sub>. Then the processing latency and the worst-case age of the features seen by the controller (sampled at period T<sub>c</sub>) are"))
S.append(eq(r"L=\tau_{\mathrm{parse}}+\tau_{\mathrm{state}}+\tau_{\mathrm{depth}}+\tau_{\mathrm{deproj}}+\tau_{\mathrm{ik}}+\tau_{\mathrm{enc}},\qquad A_{\max}=L+\max\!\left(T_{c},\,T_{\mathrm{ARIA}}\right)"))
S.append(P("The IK term is bounded by construction (\u03c4<sub>ik</sub> \u2264 0.8 ms), and the queue bound (size 2, drop-oldest) guarantees that backlog cannot accumulate: under overload the layer sheds load by dropping stale detections "
           "(counter <font name='Mono'>dropped_frames</font>) rather than increasing latency. The worker owns its OpenGL context and releases it from the same thread on shutdown, which is required for correct rendering resource management."))

# 5.14 multi-object
S += [P("5.14 Multi-object task sequencing", h2),
      P("A single Franka arm has one end-effector, so a plan with K objects is executed as K consecutive sub-tasks. For sub-task <i>m</i> the integrated pipeline (i) activates the m-th pick/place pair, "
        "(ii) clears the tracked object memory so that the layer re-locks the new object's initial location p<sub>obj,m</sub><super>0</super>, (iii) starts the reference from the <i>current</i> end-effector pose "
        "so no discontinuity occurs between sub-tasks, and (iv) leaves previously placed objects free to rest physically in their receptacles. "
        "Because identity is tied to the active sub-task, the secondary-object obstacles in the collision test of Section 5.9 automatically include previously placed items and distractors."),
      PageBreak()]

# ---------------- 6 failure
S += [P("6. Failure Modes and Graceful Degradation", h1),
      P("The layer is designed to degrade in a controlled, observable way. Each failure mode produces a <i>status string</i> in the result, preserves the output shape, and continues running.")]
S.append(table([
    ["Failure", "Detection", "Behaviour", "Downstream effect"],
    ["Malformed or low-confidence detection", "Validation (5.2)", "Entry rejected and counted", "Primary missing \u2192 status no_valid_primary"],
    ["No admissible primary target", "frame.primary is None", "Publish empty vector, keep p<sub>ee</sub>", "Flags 0, difficulty 1"],
    ["No valid depth at target", "Empty median window", "Publish empty vector (status no_depth)", "CCDM sees no target"],
    ["Target outside workspace", "ws() false", "Flag 0; IK usually fails", "Difficulty raised; constraints tightened"],
    ["IK non-convergence", "\u2016e\u2016 > tol after budget", "ik_feasible = 0", "Difficulty +0.3"],
    ["Grasp below table / near obstacle", "free() false", "collision_free = 0", "Safety weight can increase"],
    ["Perception slower than control", "Queue full", "Drop oldest frame", "Latency bounded; dropped_frames incremented"],
    ["Exception in worker", "try/except in loop", "Error logged to worker_errors; thread continues", "Previous result retained"],
], [4.0 * cm, 3.3 * cm, 4.6 * cm, TW - 11.9 * cm]))
S += [Spacer(1, 6),
      P("Two limitations of the <i>simulated</i> setting should be noted: detections are emulated from ground-truth body positions projected into the image (with a generous box), so detector errors are not "
        "exercised end-to-end; and the depth image is noise-free. Section 8 discusses how both are addressed before real-hardware deployment."),
      PageBreak()]

# ---------------- 7 validation
S += [P("7. Validation and Measured Performance", h1),
      P("Validation combines analytic bounds, unit tests against simulator ground truth, a randomised localisation experiment, and closed-loop system benchmarks.")]
S.append(P("7.1 Unit tests", h2))
S.append(P("The regression suite contains 28 tests (all passing) covering the transformer (pixel\u2194world round trip to 10<super>\u22126</super> m, batch vs single deprojection, <b>3-D deprojection error < 3 mm</b> at five block positions), "
           "tracking (continuity across frames, bbox/confidence validation, distance gating), feature encoding (layout, clamping, asynchronous double buffer), reachability (in/out of workspace), "
           "scene building and the integrated four-layer pipeline."))
S.append(P("7.2 Randomised localisation experiment", h2))
S.append(P(f"To quantify accuracy, a cube was placed at {M['n']} random positions (x \u2208 [0.42, 0.70], y \u2208 [\u22120.20, 0.20], z = 0.425 m) in the stock scene. Two variants were run through the full layer (forced depth refresh) and compared with the simulator's geometric centre. "
           "<b>(A) Ideal detector:</b> the box centre equals the exact ground-truth projection; the residual is then purely numerical (depth-buffer precision and linearisation). "
           "<b>(B) Noisy detector:</b> the box centre is perturbed uniformly by up to \u00b12 px in u and v, emulating a realistic detector; the lateral (x, y) error is reported and should follow the d/f law of Section 5.6 "
           "(d/f \u2248 3.1 mm per pixel at d \u2248 0.78 m, f = 250 px, i.e. at most 6.2 mm per axis and 8.8 mm radially for \u00b12 px). Timings are per-call totals after a 5-call warm-up (a repeat run on an idle machine gave about 1.0 ms mean; the value below was taken under load)."))
st = M["stages_ms_mean"]
S.append(table([
    ["Metric", "Value"],
    ["Samples", f"{M['n']}"],
    ["(A) Ideal box \u2013 3-D error, mean / p95 / max", f"{M['err_mean_mm']:.4f} / {M['err_p95_mm']:.4f} / {M['err_max_mm']:.4f} mm"],
    ["(B) \u00b12 px box jitter \u2013 lateral error, mean / p95 / max", f"{M['jit_lat_mean_mm']:.2f} / {M['jit_lat_p95_mm']:.2f} / {M['jit_lat_max_mm']:.2f} mm"],
    ["Latency \u2013 mean (total)", f"{M['total_ms_mean']:.2f} ms"],
    ["Latency \u2013 95th percentile", f"{M['total_ms_p95']:.2f} ms"],
] + [[f"Stage mean \u2013 {k}", f"{v:.3f} ms"] for k, v in st.items() if k != "total"], [8.6 * cm, 5.4 * cm]))
S.append(Spacer(1, 4))
S.append(P("<i>Table 4.</i> Measured Layer-2 accuracy and latency on the reference machine and scene. (A) shows the geometric pipeline is exact to numerical precision when the detection is exact, "
           "so the achievable accuracy is set by the detector\u2019s centroid error (B), which scales with depth and inversely with focal length as predicted analytically. The simulated depth map is noise-free.", small))
S.append(P("7.3 Closed-loop system benchmarks", h2))
S.append(P("In the integrated stack, the layer drove two benchmark runs. A single-object task (red cube to green bin) completed with 32.1 mm placement error. A two-object sequential task "
           "(red cube \u2192 green bin, yellow cube \u2192 blue bin, with an obstacle cylinder left undisturbed) completed with placement errors of 32.1 mm and 27.7 mm and a task-space tracking RMSE of 28.0 mm. "
           "These system-level figures include controller tracking error and the object\u2019s settling in the bin; they are therefore larger than the perception-only error in Table 4 and should not be attributed to the SRL alone."))
S.append(PageBreak())

# ---------------- 8 limitations
S += [P("8. Limitations and Future Work", h1),
      B("<b>Noise-free sensing.</b> Simulated depth has no noise, holes or flying pixels. For real RGB-D hardware the median window should be complemented by a depth-noise model (e.g. \u03c3<sub>d</sub> growing quadratically with range) and hole filling; the covariance of Section 5.6 can then be propagated and exposed to the CCDM as an uncertainty feature."),
      B("<b>Detector emulation.</b> Bounding boxes are synthesised from ground truth. Integration with the real ARIA vision agents "
        f"(open-vocabulary detection and mask models {c('gdino','sam','frcnn')}) should use mask centroids or mask-weighted depth rather than the box centre, to reduce bias for non-convex or partially occluded objects."),
      B("<b>Motion model.</b> The tracker is a constant-velocity pixel-space model without covariance or appearance. Full Kalman filtering with Mahalanobis gating "
        f"{c('kalman','barshalom')} and appearance embeddings {c('deepsort')} would improve robustness in cluttered, dynamic scenes."),
      B("<b>Orientation.</b> The layer passes through an orientation quaternion if supplied but does not estimate it. Estimating object pose (e.g. from the segmentation mask and depth) would enable orientation-aware grasp offsets."),
      B("<b>Position-only IK.</b> Reachability is tested for position only. A 6-D test with a task-specific approach direction would be stricter and capture grasp-orientation infeasibility."),
      B("<b>Centroid correction assumes a known box.</b> The class-size table is exact for the simulated cube; for novel objects the offset should be estimated from the segmented point cloud."),
      B("<b>Static camera.</b> A fixed, calibrated overhead camera is assumed. Hand-eye calibration and eye-in-hand viewpoints would require a time-varying T<sub>cw</sub>; the transformer already supports re-setting extrinsics at run time."),
      Spacer(1, 6),
      P("<b>Conclusion.</b> The Spatial Reasoning Layer converts semantic, asynchronous 2-D detections into a metric, time-aligned, bounded 30-D state with explicit feasibility information, using "
        "interpretable closed-form geometry and a bounded-cost damped-least-squares solver. It provides the structural interface between language-driven task understanding and torque-level control, "
        "and its design makes each assumption, bound and failure mode explicit.", box),
      PageBreak()]

# ---------------- 9 bibliography
S.append(P("9. Bibliography", h1))
S.append(P("Numbers in square brackets in the text refer to the entries below, listed in order of first citation.", small))
S.append(Spacer(1, 4))
# bibliography built after content: generate on the fly
BIBSTART = len(S)

# ---------------------------------------------------------------- build
def build():
    for i, k in enumerate(_order):
        S.append(Paragraph(f"[{i + 1}]&nbsp;&nbsp;{BIB[k]}", bib))
    unused = [k for k in BIB if k not in _order]
    if unused:
        print("WARNING uncited refs:", unused)
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=2.0 * cm, bottomMargin=2.2 * cm,
                          title="The Spatial Reasoning Layer \u2013 Technical Report", author="Spatial Reasoning Layer project")
    fr = Frame(LM, 2.2 * cm, TW, PAGE_H - 4.2 * cm, id="f")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[fr], onPage=on_cover),
                          PageTemplate(id="main", frames=[fr], onPage=on_page)])
    doc.build(S)
    print("Wrote", OUT, os.path.getsize(OUT) // 1024, "KB;", len(_order), "references;", _eqn[0], "equations")

build()

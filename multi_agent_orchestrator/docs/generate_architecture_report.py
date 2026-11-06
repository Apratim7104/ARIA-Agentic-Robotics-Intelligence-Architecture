#!/usr/bin/env python3
"""
Generate ARIA Multi-Agent Orchestrator Architecture Report (PDF)
Builds two block diagrams (system architecture + robot perception pipeline)
with matplotlib, then assembles a detailed technical report PDF with reportlab.

Usage:
 python docs/generate_architecture_report.py

Output:
 docs/assets/aria_system_architecture.png
 docs/assets/aria_robot_perception_pipeline.png
 ARIA_Architecture_Report.pdf (repo root)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.path import Path as MplPath
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak,
    Image, Preformatted, ListFlowable, ListItem, KeepTogether, HRFlowable
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
DIAGRAM1_PATH = os.path.join(ASSETS_DIR, "aria_system_architecture.png")
DIAGRAM2_PATH = os.path.join(ASSETS_DIR, "aria_robot_perception_pipeline.png")
OUTPUT_PDF = os.path.join(REPO_ROOT, "ARIA_Architecture_Report.pdf")

# ----------------------------------------------------------------------------
# COLOR PALETTE (shared between diagrams and PDF tables)
# ----------------------------------------------------------------------------
COLOR_IO = "#E0E0E0"         # user-facing input/output
COLOR_LLM = "#BBDEFB"        # Siemens LLM (qwen-3.6-27b) reasoning agents
COLOR_VLM = "#E1BEE7"        # Qwen2.5-VL vision/VLM
COLOR_TOOL = "#FFF9C4"       # web search / tool
COLOR_ROBOT = "#C8E6C9"      # CV / robotics (YOLO, SAM2)
COLOR_EXTERNAL = "#FFE0B2"   # external systems (MPC controller)
COLOR_CRITIC = "#FFCDD2"     # quality-control / critic loop

# ============================================================================
# DIAGRAM 1 — FULL SYSTEM ARCHITECTURE
# ============================================================================
def make_architecture_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 11.5))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(3.9, 16.2)
    ax.axis("off")
    ax.set_title("ARIA Multi-Agent Orchestrator — System Architecture",
                 fontsize=16, fontweight="bold", pad=14)

    def box(x, y, w, h, text, color, fontsize=9.5, style="round", dashed=False):
        bb = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"{style},pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor="#333333", facecolor=color,
            linestyle="dashed" if dashed else "solid", zorder=3,
        )
        ax.add_patch(bb)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                zorder=4, wrap=True)
        return (x, y, w, h)

    def arrow(p1, p2, label=None, rad=0.0, color="#444444", dashed=False,
              label_offset=(0, 0.18), fontsize=8):
        a = FancyArrowPatch(
            p1, p2, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=14, linewidth=1.3,
            color=color, linestyle="dashed" if dashed else "solid", zorder=2,
        )
        ax.add_patch(a)
        if label:
            mx, my = (p1[0] + p2[0]) / 2 + label_offset[0], (p1[1] + p2[1]) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                    color="#222222", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # --- Nodes -----------------------------------------------------------
    b_user = box(5.5, 15.5, 4.2, 0.8, "User Query\n(CLI: main.py --query ...)", COLOR_IO)
    b_planner = box(5.5, 14.1, 4.6, 0.9, "Planner Agent\n(LLM router — qwen-3.6-27b)", COLOR_LLM)
    b_research = box(1.7, 12.3, 2.9, 0.95, "Research Agent\n(Tavily web search)", COLOR_TOOL)
    b_vision = box(5.5, 12.3, 2.9, 0.95, "Vision Agent\n(Qwen2.5-VL-7B-Instruct)", COLOR_VLM)
    b_robotp = box(9.3, 12.3, 2.9, 0.95, "Robot Perception\n(YOLOv8 + SAM2)", COLOR_ROBOT)
    b_nutrition = box(5.5, 10.6, 2.9, 0.9, "Nutrition Agent\n(LLM dietary analysis)", COLOR_LLM)
    b_robotplan = box(9.3, 10.6, 2.9, 0.9, "Robot Planner Agent\n(LLM grasp planning)", COLOR_LLM)
    b_synth = box(5.5, 8.7, 4.8, 0.9, "Synthesis Agent\n(LLM final-answer generation)", COLOR_LLM)
    b_critic = box(5.5, 7.1, 4.8, 0.9, "Critic Agent\n(LLM quality scoring, 1-10)", COLOR_CRITIC)
    b_final = box(3.0, 5.2, 3.4, 0.95, "Final Answer\n(returned to user)", COLOR_IO)
    b_mpc = box(9.0, 5.2, 3.6, 0.95, "MPC Controller\n(external, reads robot_handoff)", COLOR_EXTERNAL, dashed=True)

    # --- Edges -------------------------------------------------------------
    arrow((b_user[0], b_user[1] - b_user[3] / 2), (b_planner[0], b_planner[1] + b_planner[3] / 2))
    arrow((4.6, 13.65), (1.9, 12.78), label="web / excel", rad=0.15)
    arrow((5.5, 13.65), (5.5, 12.78), label="image / video /\nvideo_food", fontsize=7.5)
    arrow((6.4, 13.65), (9.1, 12.78), label="robot_pickup", rad=-0.15)
    arrow((5.5, 11.83), (5.5, 11.05), label="if video_food", fontsize=7.5)
    arrow((6.6, 11.9), (7.0, 9.2), label="else (image/video)", rad=-0.25, fontsize=7.5)
    arrow((9.3, 11.83), (9.3, 11.05))
    arrow((2.2, 11.83), (4.6, 9.1), rad=0.2, color="#777777")
    arrow((5.0, 10.15), (5.3, 9.15))
    arrow((9.3, 10.15), (9.1, 9.15))
    arrow((7.9, 8.7), (8.8, 9.6), label="robot_handoff\n(state passthrough)", rad=0.0,
          dashed=True, color="#B06000", fontsize=7)
    arrow((5.5, 8.25), (5.5, 7.55))
    arrow((4.2, 6.75), (3.5, 5.68), label="score ≥ 7\n(approved)", fontsize=7.5)
    arrow((7.4, 7.3), (7.9, 8.15), label="score < 7 → retry\n(max 2 attempts)", rad=-0.35, fontsize=7.5, color="#B00020")
    arrow((9.3, 10.15), (9.0, 5.68), rad=0.4, color="#B06000", dashed=True)

    # --- Legend --------------------------------------------------------
    legend_items = [
        Patch(facecolor=COLOR_IO, edgecolor="#333", label="User I/O"),
        Patch(facecolor=COLOR_LLM, edgecolor="#333", label="Siemens LLM (qwen-3.6-27b)"),
        Patch(facecolor=COLOR_VLM, edgecolor="#333", label="Local VLM (Qwen2.5-VL-7B)"),
        Patch(facecolor=COLOR_TOOL, edgecolor="#333", label="Web search tool (Tavily)"),
        Patch(facecolor=COLOR_ROBOT, edgecolor="#333", label="CV / Robotics (YOLOv8, SAM2)"),
        Patch(facecolor=COLOR_CRITIC, edgecolor="#333", label="Quality control (Critic)"),
        Patch(facecolor=COLOR_EXTERNAL, edgecolor="#333", label="External system (MPC)"),
    ]
    ax.legend(handles=legend_items, loc="upper center", ncol=2, fontsize=8,
              frameon=True, bbox_to_anchor=(0.5, 0.045))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# DIAGRAM 2 — ROBOT PERCEPTION / YOLOv8 + SAM2 PIPELINE
# ============================================================================
def make_robot_pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 15.5)
    ax.axis("off")
    ax.set_title("Robot Perception Pipeline — YOLOv8 + SAM2 + 3D Back-Projection",
                 fontsize=15, fontweight="bold", pad=12)

    def box(x, y, w, h, text, color, fontsize=9, dashed=False):
        bb = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor="#333333", facecolor=color,
            linestyle="dashed" if dashed else "solid", zorder=3,
        )
        ax.add_patch(bb)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4)
        return (x, y, w, h)

    def arrow(p1, p2, label=None, rad=0.0, color="#444444", fontsize=7.5, dashed=False):
        a = FancyArrowPatch(
            p1, p2, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=color,
            linestyle="dashed" if dashed else "solid", zorder=2,
        )
        ax.add_patch(a)
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=5)

    # Detection branch (left)
    b_rgb = box(3.0, 14.6, 4.0, 0.8, "RGB Frame\n(MuJoCo wrist_cam / RealSense)", "#D7CCC8")
    b_yolo = box(3.0, 13.0, 4.0, 0.9, "YOLOv8 Detector\n(yolov8n.pt, conf ≥ 0.35)", COLOR_ROBOT)
    b_vlmfb = box(3.0, 11.3, 4.2, 1.0, "VLM Open-Vocabulary Fallback\n(Qwen2.5-VL box query)\nused only if target not found by YOLO", COLOR_VLM, fontsize=8)
    b_cand = box(3.0, 9.5, 4.0, 0.9, "Candidate Boxes\n{label, conf, xyxy, detector_src}", "#EEEEEE")

    # Depth / intrinsics branch (right)
    b_depthsrc = box(8.2, 14.6, 4.0, 1.1, "Depth Source (priority)\n1. MuJoCo depth buffer (GT)\n2. RealSense D435i\n3. MiDaS monocular (fallback)", "#D7CCC8", fontsize=8)
    b_intr = box(8.2, 12.7, 4.0, 1.1, "Camera Intrinsics (priority)\n1. MuJoCo model (live)\n2. Precomputed dict\n3. Geometric (FOVY fallback)", "#D7CCC8", fontsize=8)
    b_depthmap = box(8.2, 10.7, 4.0, 0.8, "Metric Depth Map (H×W, metres)", "#EEEEEE")

    # Convergence
    b_sam2 = box(3.0, 7.6, 4.0, 0.9, "SAM2 Image Predictor\n(box-prompted segmentation)", COLOR_ROBOT)
    b_mask = box(3.0, 6.0, 4.0, 0.9, "Binary Mask + Centroid\n(cv2.moments, fallback: bbox centre)", "#EEEEEE")
    b_proj = box(5.6, 4.2, 5.2, 1.1, "3D Back-Projection (pinhole inverse)\nX=(u-cx)·Z/fx Y=(v-cy)·Z/fy Z=depth", "#FFF3E0", fontsize=8.5)
    b_detobj = box(5.6, 2.3, 5.6, 1.1, "DetectedObject list\n{label, bbox, mask, depth_m, centroid_3d,\ndetector_src, centroid_src}", "#E8EAF6", fontsize=8)
    b_out = box(5.6, 0.7, 4.6, 0.8, "→ forwarded to Robot Planner Agent", COLOR_LLM)

    arrow((3.0, 14.2), (3.0, 13.45))
    arrow((3.0, 12.55), (3.0, 11.8), label="target not\nmatched", color="#B00020")
    arrow((3.0, 10.8), (3.0, 9.95))
    arrow((3.0, 12.55), (1.6, 9.95), rad=0.45, fontsize=7)
    arrow((8.2, 14.05), (8.2, 13.25))
    arrow((8.2, 13.25), (8.2, 11.1), rad=0.0)
    arrow((8.2, 12.15), (8.2, 11.1))
    arrow((3.0, 9.05), (3.0, 8.05))
    arrow((3.0, 7.15), (3.0, 6.45))
    arrow((4.3, 5.6), (5.0, 4.75), label="pixel (u, v)")
    arrow((8.2, 10.3), (6.4, 4.6), rad=-0.15, label="depth(u, v), fx, fy, cx, cy", fontsize=7)
    arrow((5.6, 3.65), (5.6, 2.85))
    arrow((5.6, 1.75), (5.6, 1.1))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# PDF REPORT BUILDING
# ============================================================================
def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=26, leading=32, alignment=TA_CENTER,
                             fontName="Helvetica-Bold", spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=14, leading=18, alignment=TA_CENTER,
                             textColor=colors.HexColor("#555555"), spaceAfter=4))
    styles.add(ParagraphStyle(name="H1", fontSize=17, leading=21, fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#0D47A1"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#1565C0"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H3", fontSize=11, leading=14, fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#333333"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body9", parent=styles["BodyText"], fontSize=9.6, leading=13.5,
                             alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=11,
                             textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(name="CodeBlock", fontName="Courier", fontSize=7.4, leading=9.2,
                             backColor=colors.HexColor("#F5F5F5"), borderPadding=6,
                             leftIndent=2, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(name="TableCell", fontSize=8.3, leading=10.2, fontName="Helvetica"))
    styles.add(ParagraphStyle(name="TableHeader", fontSize=8.3, leading=10.2,
                             fontName="Helvetica-Bold", textColor=colors.whitesmoke))
    return styles

def table_style(header_color=colors.HexColor("#1565C0")):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#BBBBBB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])

def bullets(styles, items):
    return ListFlowable(
        [ListItem(Paragraph(it, styles["Body9"]), bulletColor=colors.HexColor("#1565C0")) for it in items],
        bulletType="bullet", start="circle", leftIndent=14, spaceBefore=2, spaceAfter=8,
    )

def fitted_image(path, max_width):
    """Scale a saved PNG to max_width while preserving its real aspect ratio."""
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(path, width=max_width, height=max_width * h / w)

def make_table(data, col_widths, header_color=None, styles=None):
    """Build a Table whose cells are Paragraphs so long text wraps instead of overflowing."""
    header_style = styles["TableHeader"]
    body_style = styles["TableCell"]
    wrapped = [[Paragraph(str(c), header_style) for c in data[0]]]
    for row in data[1:]:
        wrapped.append([Paragraph(str(c), body_style) for c in row])
    t = Table(wrapped, colWidths=col_widths)
    t.setStyle(table_style(header_color) if header_color else table_style())
    return t

def build_pdf():
    styles = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=letter,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title="ARIA Multi-Agent Orchestrator — Architecture Report",
    )
    S = []

    # ------------------------------------------------------------------ COVER
    S.append(Spacer(1, 1.6 * inch))
    S.append(Paragraph("ARIA", styles["ReportTitle"]))
    S.append(Paragraph("Multi-Agent Orchestrator", styles["ReportTitle"]))
    S.append(Spacer(1, 0.15 * inch))
    S.append(Paragraph("Architecture, Workflow &amp; Component Design Report", styles["ReportSubtitle"]))
    S.append(Spacer(1, 0.5 * inch))
    S.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#1565C0"), hAlign="CENTER"))
    S.append(Spacer(1, 0.3 * inch))

    cover_table = Table([
        ["Repository", "multi_agent_orchestrator"],
        ["Framework", "LangGraph (state-graph multi-agent orchestration)"],
        ["Primary LLM", "Siemens-hosted qwen-3.6-27b (OpenAI-compatible API)"],
        ["Local VLM", "Qwen2.5-VL-7B-Instruct (local GPU inference)"],
        ["Object Detection", "YOLOv8 (Ultralytics, yolov8n.pt)"],
        ["Segmentation", "SAM2 (Segment Anything Model 2, Meta AI)"],
        ["Report Date", "2026-10-31"],
    ], colWidths=[1.8 * inch, 4.2 * inch])
    cover_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0D47A1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
    ]))
    S.append(cover_table)
    S.append(PageBreak())

    # ------------------------------------------------------------------ TOC
    S.append(Paragraph("Table of Contents", styles["H1"]))
    toc_items = [
        "1. Executive Summary",
        "2. System Architecture Overview",
        "3. End-to-End Workflow (Request → Response)",
        "4. Shared State Schema",
        "5. Component Deep Dive",
        "   5.1 Planner Agent",
        "   5.2 Vision Agent &amp; Qwen2.5-VL (VLM)",
        "   5.3 Research Agent &amp; Tavily",
        "   5.4 Nutrition Agent",
        "   5.5 Robot Perception Agent — YOLOv8 + SAM2",
        "   5.6 Robot Planner Agent &amp; MPC Handoff",
        "   5.7 Synthesis Agent",
        "   5.8 Critic Agent &amp; Retry Loop",
        "   5.9 LLM Client (Siemens API)",
        "6. Robot Perception Pipeline in Detail",
        "7. Configuration Reference (Environment Variables)",
        "8. Testing &amp; Evaluation",
        "9. Dataset — OCID / OCID-Ref",
        "10. Design Rationale — Why These Models",
        "11. Limitations &amp; Future Work",
    ]
    for it in toc_items:
        S.append(Paragraph(it, styles["Body9"]))
    S.append(PageBreak())

    # ------------------------------------------------------------------ 1. EXEC SUMMARY
    S.append(Paragraph("1. Executive Summary", styles["H1"]))
    S.append(Paragraph(
        "ARIA (this repository, <b>multi_agent_orchestrator</b>) is a modular, graph-driven multi-agent system "
        "built on <b>LangGraph</b>. A single natural-language request — optionally accompanied by an image or "
        "video — is routed by a <b>Planner Agent</b> to one or more specialized agents: a <b>Research Agent</b> "
        "for general web knowledge, a <b>Vision Agent</b> for image/video food understanding, a <b>Nutrition "
        "Agent</b> for dietary advice, and a <b>Robot Perception + Robot Planner</b> pair for embodied "
        "pick-and-place tasks. All branches converge on a <b>Synthesis Agent</b> that drafts a final answer, "
        "which is then quality-checked by a <b>Critic Agent</b> in a bounded retry loop before being returned "
        "to the user.", styles["Body9"]))
    S.append(Paragraph(
        "The system mixes three categories of models: (1) a large, API-hosted reasoning LLM "
        "(<b>qwen-3.6-27b</b>, served behind a Siemens OpenAI-compatible endpoint) used for routing, planning, "
        "dietary reasoning, grasp planning, answer synthesis and quality scoring; (2) a locally-hosted "
        "vision-language model (<b>Qwen2.5-VL-7B-Instruct</b>) used for food recognition and open-vocabulary "
        "object localisation directly on image/video pixels; and (3) classical/learned computer-vision models "
        "— <b>YOLOv8</b> for fast closed-vocabulary object detection and <b>SAM2</b> for pixel-accurate "
        "segmentation — used in the robotic perception pipeline to produce 3-D grasp candidates for a "
        "Franka-Panda-class manipulator simulated in MuJoCo.", styles["Body9"]))
    S.append(Paragraph(
        "This report documents the architecture end-to-end: the orchestration graph, the shared state schema, "
        "every agent's inputs/outputs and prompt design, the YOLOv8 + SAM2 + depth back-projection pipeline "
        "that turns 2-D detections into metric 3-D grasp points, the critic/retry quality loop, and the "
        "configuration surface (environment variables) that controls the whole system.", styles["Body9"]))
    S.append(PageBreak())

    # ------------------------------------------------------------------ 2. ARCHITECTURE
    S.append(Paragraph("2. System Architecture Overview", styles["H1"]))
    S.append(Paragraph(
        "The orchestrator is implemented as a compiled <b>LangGraph</b> state graph (<font face='Courier'>"
        "graph/orchestrator.py</font>) over a single shared <font face='Courier'>OrchestratorState</font> "
        "TypedDict (<font face='Courier'>state/schema.py</font>). Each agent is a plain function "
        "(<font face='Courier'>node(state) -&gt; dict</font>) that reads the fields it needs and returns a "
        "partial-state update that LangGraph merges back in. Routing between nodes is decided by small "
        "conditional-edge functions (e.g. <font face='Courier'>route_after_planner</font>, "
        "<font face='Courier'>route_after_vision</font>, <font face='Courier'>route_after_critic</font>) that "
        "inspect the state and return the name of the next node.", styles["Body9"]))

    S.append(fitted_image(DIAGRAM1_PATH, max_width=6.2 * inch))
    S.append(Paragraph("Figure 1. Full system architecture — agent graph, routing conditions and the critic/synthesis retry loop.",
                       styles["Caption"]))
    S.append(PageBreak())

    S.append(Paragraph("2.1 Routing table", styles["H2"]))
    route_data = [
        ["Route", "Trigger", "Agents invoked"],
        ["web", "General knowledge question", "Research Agent → Synthesis → Critic"],
        ["excel", "Query over an Excel file", "Research Agent (excel_tool) → Synthesis → Critic"],
        ["image", "Single image attached", "Vision Agent → Synthesis → Critic"],
        ["video", "Video attached, non-food", "Vision Agent → Synthesis → Critic"],
        ["video_food", "Food video + dietary request", "Vision Agent → Nutrition Agent → Synthesis → Critic"],
        ["robot_pickup", "“pick up / grab / move X”", "Robot Perception → Robot Planner → Synthesis → Critic (+ MPC handoff)"],
    ]
    t = make_table(route_data, col_widths=[0.9 * inch, 2.15 * inch, 3.15 * inch], styles=styles)
    S.append(t)
    S.append(Spacer(1, 10))
    S.append(Paragraph(
        "Routing decisions are produced by the Planner Agent's LLM call and returned as strict JSON "
        "(<font face='Courier'>route</font>, <font face='Courier'>agents_needed</font>, "
        "<font face='Courier'>target_object</font>, <font face='Courier'>confidence</font>). If the LLM call "
        "fails, a keyword-based fallback (<font face='Courier'>_fallback_intent()</font>) still routes the "
        "request using simple vocabulary matching (e.g. “pick/grab/grasp” → "
        "<font face='Courier'>robot_pickup</font>; “food/buffet/meal” → "
        "<font face='Courier'>video_food</font>; default → <font face='Courier'>web</font>), so the graph "
        "always makes forward progress even if the LLM endpoint is unavailable.", styles["Body9"]))
    S.append(PageBreak())

    # ------------------------------------------------------------------ 3. WORKFLOW
    S.append(Paragraph("3. End-to-End Workflow (Request → Response)", styles["H1"]))
    workflow_steps = [
        ("1. Entry", "main.py parses <font face='Courier'>--query/--video/--image</font> and builds the "
         "initial state: <font face='Courier'>user_input</font>, <font face='Courier'>session_id</font> "
         "(UUID), <font face='Courier'>video_path</font>, <font face='Courier'>image_path</font>, "
         "<font face='Courier'>retry_count=0</font>, then calls "
         "<font face='Courier'>run_pipeline(state)</font>."),
        ("2. Planning", "The Planner Agent calls the Siemens LLM (temperature 0.1) with a system prompt listing "
         "the six routes, and returns route, agents_needed, target_object and a step-by-step plan."),
        ("3. Specialized Processing", "Exactly one of three branches executes depending on the route: Research "
         "(Tavily web search), Vision (Qwen2.5-VL food detection, optionally "
         "followed by Nutrition analysis), or Robot Perception + Robot Planner "
         "(YOLOv8 + SAM2 + 3-D back-projection, then LLM grasp planning)."),
        ("4. Synthesis", "The Synthesis Agent combines whatever upstream outputs are present (research "
         "snippets, detected foods, nutrition analysis, or scene description + grasp plan) "
         "into a single natural-language <font face='Courier'>final_answer</font>."),
        ("5. Critic Review", "The Critic Agent scores the answer 1-10 on completeness, accuracy, clarity, "
         "relevance and tone. Scores ≥ 7 are approved; scores < 7 send control back "
         "to the Synthesis Agent for another attempt, capped at <b>2 retries</b> "
         "(<font face='Courier'>MAX_RETRIES=2</font>) after which the answer is force-approved."),
        ("6. Output", "The graph reaches <font face='Courier'>END</font>; <font face='Courier'>run_pipeline()</font> "
         "returns the complete final state. For robot tasks, the independent "
         "<font face='Courier'>robot_handoff</font> payload (grasp poses, masks, depth map) sits "
         "in state for an external MPC controller to read — it is <i>not</i> a graph node, by design, "
         "since real-time control loops should not be coupled to LLM latency."),
    ]
    for title, text in workflow_steps:
        S.append(Paragraph(title, styles["H3"]))
        S.append(Paragraph(text, styles["Body9"]))
    S.append(PageBreak())

    # ------------------------------------------------------------------ 4. STATE SCHEMA
    S.append(Paragraph("4. Shared State Schema", styles["H1"]))
    S.append(Paragraph(
        "<font face='Courier'>state/schema.py</font> defines every field agents may read or write as a single "
        "<font face='Courier'>OrchestratorState</font> TypedDict (50+ fields), plus smaller typed sub-structures "
        "for clarity and static-analysis friendliness. Using one shared schema means every agent is a pure "
        "function of state, which keeps the graph easy to extend (new agent = new node + one routing edge) "
        "and easy to debug (the whole conversation/trace is visible as one dict).", styles["Body9"]))

    S.append(Paragraph("Route and AgentName constants", styles["H2"]))
    S.append(Preformatted(
        'class Route:\n'
        '    WEB = "web"; EXCEL = "excel"; IMAGE = "image"\n'
        '    VIDEO = "video"; VIDEO_FOOD = "video_food"; ROBOT_PICKUP = "robot_pickup"\n\n'
        'class AgentName:\n'
        '    PLANNER = "planner"; VISION = "vision"; NUTRITION = "nutrition"\n'
        '    RESEARCH = "research"; SYNTHESIZER = "synthesizer"; CRITIC = "critic"\n'
        '    ROBOT_PERC = "robot_perception"; ROBOT_PLAN = "robot_planner"; MPC = "mpc_controller"',
        styles["CodeBlock"]))

    S.append(Paragraph("Key TypedDicts", styles["H2"]))
    S.append(Preformatted(
        'DetectedObject = {\n'
        '    "label": str, "confidence": float, "bbox_xyxy": [x1, y1, x2, y2],\n'
        '    "mask": np.ndarray | None,       # (H, W) uint8 from SAM2\n'
        '    "depth_m": float,                # metric depth at centroid\n'
        '    "centroid_3d": [X, Y, Z],        # camera frame, metres\n'
        '    "centroid_src": "sam2" | "bbox",\n'
        '    "detector_src": "yolo" | "vlm",\n'
        '}\n\n'
        'CriticFeedback = {\n'
        '    "approved": bool, "score": int,  # 1-10, approved if >= 7\n'
        '    "issues": [str], "suggestions": [str], "retry_reason": str,\n'
        '}\n\n'
        'GraspPlan = {\n'
        '    "target_object": str,\n'
        '    "grasp_pose": [x, y, z, roll, pitch, yaw],\n'
        '    "pre_grasp_pose": [x, y, z, roll, pitch, yaw],\n'
        '    "post_grasp_pose": [x, y, z, roll, pitch, yaw],\n'
        '    "gripper_width": float, "confidence": float, "reasoning": str,\n'
        '}',
        styles["CodeBlock"]))
    S.append(PageBreak())

    # ------------------------------------------------------------------ 5. COMPONENT DEEP DIVE
    S.append(Paragraph("5. Component Deep Dive", styles["H1"]))
    S.append(Paragraph("5.1 Planner Agent (agents/planner.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> interpret the raw user request and decide which specialised agent(s) "
                       "should handle it, producing a machine-readable routing decision.", styles["Body9"]))
    S.append(Paragraph("<b>Model &amp; call parameters:</b> Siemens LLM <font face='Courier'>qwen-3.6-27b</font> "
                       "via <font face='Courier'>call_llm_json()</font>, temperature <b>0.1</b> (near-deterministic "
                       "routing), <font face='Courier'>max_tokens=2000</font> to accommodate the plan + subtasks JSON.",
                       styles["Body9"]))

    S.append(Paragraph("5.2 Vision Agent &amp; Qwen2.5-VL (agents/vision_agent.py, llm/vlm_client.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> detect and characterise food items directly from image or video pixels "
                       "(no YOLO/closed vocabulary limitation) so the Nutrition Agent can reason about them.",
                       styles["Body9"]))

    S.append(Paragraph("5.3 Research Agent &amp; Tavily (agents/research_agent.py, llm/tavily_client.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> answer general-knowledge / web / Excel-context queries that do not "
                       "need vision or robotics, by retrieving up-to-date snippets from the web.", styles["Body9"]))

    S.append(Paragraph("5.4 Nutrition Agent (agents/nutrition_agent.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> turn the Vision Agent's structured food list (plus any research "
                       "context and an optional user health profile) into dietary recommendations.", styles["Body9"]))

    S.append(Paragraph("5.5 Robot Perception Agent — YOLOv8 + SAM2 (agents/robot_perception_agent.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> convert a single RGB frame into a list of 3-D grasp candidates "
                       "(<font face='Courier'>DetectedObject</font>) for a target object named by the Planner.",
                       styles["Body9"]))

    S.append(fitted_image(DIAGRAM2_PATH, max_width=5.9 * inch))
    S.append(Paragraph("Figure 2. Robot perception pipeline: parallel detection and depth/intrinsics branches "
                       "converge at the 3-D back-projection step.", styles["Caption"]))
    S.append(PageBreak())

    S.append(Paragraph("5.6 Robot Planner Agent &amp; MPC Handoff (agents/robot_planner_agent.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> turn the list of detected 3-D objects into an executable grasp plan "
                       "for a 6-DOF arm with a parallel-jaw gripper, and package it for an external Model "
                       "Predictive Controller (MPC).", styles["Body9"]))

    S.append(Paragraph("5.7 Synthesis Agent (agents/synthesis_agent.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> produce the single natural-language answer returned to the user, "
                       "tailored to the active route.", styles["Body9"]))

    S.append(Paragraph("5.8 Critic Agent &amp; Retry Loop (agents/critic.py)", styles["H2"]))
    S.append(Paragraph("<b>Purpose:</b> an automated quality gate so a single bad LLM generation does not reach "
                       "the user. Scores the synthesised answer 1-10 on completeness, accuracy, clarity, "
                       "relevance and tone.", styles["Body9"]))

    S.append(Paragraph("5.9 LLM Client — Siemens API (llm/client.py)", styles["H2"]))
    S.append(Paragraph("A thin wrapper around the official <font face='Courier'>openai</font> Python SDK pointed "
                       "at a Siemens-internal OpenAI-compatible endpoint.", styles["Body9"]))
    S.append(PageBreak())

    # ------------------------------------------------------------------ 6. RECAP TABLE
    S.append(Paragraph("6. Robot Perception Pipeline — Stage Summary", styles["H1"]))
    pipeline_table = [
        ["Stage", "Model / Method", "Input", "Output"],
        ["Intrinsics", "MuJoCo model → precomputed → geometric (FOVY)", "Camera name / FOVY", "fx, fy, cx, cy"],
        ["Depth", "MuJoCo depth buffer → RealSense D435i → MiDaS_small", "Sim/sensor buffer", "(H×W) metric depth map"],
        ["Detection", "YOLOv8 (yolov8n.pt), conf ≥ 0.35", "RGB frame", "Candidate boxes (COCO classes)"],
        ["Open-vocab fallback", "Qwen2.5-VL box query", "RGB frame + target phrase", "Extra candidate box (if YOLO misses target)"],
        ["Segmentation", "SAM2 (sam2_hiera_small)", "RGB frame + box prompt", "Binary mask (H×W)"],
        ["Centroid", "cv2.moments() / bbox centre fallback", "Mask", "Pixel centroid (u, v)"],
        ["3-D projection", "Pinhole inverse model", "(u,v), depth, intrinsics", "Camera-frame 3-D point [X,Y,Z]"],
        ["Grasp planning", "Siemens LLM, temperature 0.05", "DetectedObject list + target", "GraspPlan (6-DOF poses)"],
    ]
    t2 = make_table(pipeline_table, col_widths=[1.0 * inch, 1.9 * inch, 1.6 * inch, 1.7 * inch],
                    header_color=colors.HexColor("#2E7D32"), styles=styles)
    S.append(t2)
    S.append(PageBreak())

    # ------------------------------------------------------------------ 7. ENV VARS
    S.append(Paragraph("7. Configuration Reference (Environment Variables)", styles["H1"]))
    env_rows = [
        ["SIEMENS_LLM_API_KEY", "—", "Bearer token for Siemens endpoint"],
        ["SIEMENS_LLM_BASE_URL", "—", "API base URL"],
        ["SIEMENS_LLM_MODEL", "qwen-3.6-27b", "Model identifier for reasoning agents"],
        ["TAVILY_API_KEY", "—", "Tavily search API key"],
        ["VISION_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct", "HF model id for local VLM"],
        ["YOLO_MODEL", "yolov8n.pt", "YOLOv8 weights file"],
        ["YOLO_CONF", "0.35", "Detection confidence threshold"],
        ["SAM2_CONFIG", "sam2_hiera_small.yaml", "SAM2 model config"],
        ["SAM2_CHECKPOINT", "checkpoints/sam2_hiera_small.pt", "SAM2 weights path"],
        ["CAM_WIDTH / CAM_HEIGHT", "640 / 480", "Simulated camera resolution"],
        ["CAM_FOVY", "60.0", "Vertical FOV fallback (degrees)"],
        ["DEPTH_NEAR / DEPTH_FAR", "0.01 / 2.50", "MuJoCo depth range (metres)"],
        ["LOG_LEVEL", "INFO", "Root logger level"],
    ]
    tbl = make_table([["Variable", "Default", "Purpose"]] + env_rows,
                     col_widths=[1.9 * inch, 1.7 * inch, 2.6 * inch], styles=styles)
    S.append(tbl)
    S.append(PageBreak())

    # ------------------------------------------------------------------ 8. SUMMARY
    S.append(Paragraph("8. Summary &amp; Integration", styles["H1"]))
    S.append(Paragraph(
        "The multi-agent orchestrator successfully handles user intent decomposition, open-vocabulary "
        "and closed-vocabulary detection, pixel-accurate segmentation, 3-D back-projection, grasp planning, "
        "and structured MPC handoff payload emission.", styles["Body9"]))

    doc.build(S)
    print(f"✓ PDF report generated: {OUTPUT_PDF}")

def main():
    print("Generating diagrams...")
    make_architecture_diagram(DIAGRAM1_PATH)
    make_robot_pipeline_diagram(DIAGRAM2_PATH)
    print("Building PDF report...")
    build_pdf()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_codebase_pdf.py

Compiles the complete structure, pipeline architecture, and entire source code
of the Spatial Reasoning Layer (Layer 2) repository into a single high-quality,
beautifully styled, printable and downloadable PDF document designed for ingestion
by an AI Agent.
"""

import os
import sys
import html
import subprocess
from pathlib import Path
from datetime import datetime

import pygments
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter

ROOT_DIR = Path(__file__).resolve().parent

# File manifest in structured logical order
FILE_MANIFEST = [
    # 1. Project Configuration & Metadata
    {
        "category": "Project Overview & Configuration",
        "path": "README.md",
        "lang": "markdown",
        "description": "Repository overview, 4-layer architecture role, 30-dim vector spec, directory layout and quickstart."
    },
    {
        "category": "Project Overview & Configuration",
        "path": "requirements.txt",
        "lang": "text",
        "description": "Runtime dependencies: MuJoCo, NumPy, SciPy, PyYAML, PySide6/GLFW, pytest."
    },
    {
        "category": "Project Overview & Configuration",
        "path": "setup.py",
        "lang": "python",
        "description": "Package installation metadata for spatial_layer."
    },
    {
        "category": "Project Overview & Configuration",
        "path": "config/default_config.yaml",
        "lang": "yaml",
        "description": "Master YAML configuration: camera intrinsics/extrinsics, workspace limits, IK solver parameters, and normalization scales."
    },
    {
        "category": "Project Overview & Configuration",
        "path": "models/franka_scene.xml",
        "lang": "xml",
        "description": "MuJoCo MJCF simulation model: Franka Emika Panda 7-DOF arm, overhead camera, table, and movable target cube."
    },

    # 2. Core Spatial Reasoning Engine
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/__init__.py",
        "lang": "python",
        "description": "Package entrypoint, public exports, and version declaration."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/core.py",
        "lang": "python",
        "description": "Master engine & pipeline orchestrator with asynchronous worker thread, state synchronization, and lock-free double buffering."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/state_manager.py",
        "lang": "python",
        "description": "MuJoCo state extraction, robot joint reading, end-effector kinematics, and logarithmic Z-buffer depth linearization."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/detection_processor.py",
        "lang": "python",
        "description": "ARIA JSON detection ingestion, schema validation, bbox filtering, and Hungarian multi-object centroid tracking."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/coordinate_transformer.py",
        "lang": "python",
        "description": "Pinhole camera 2D->3D deprojection (OpenCV frame) and SE(3) camera-to-world coordinate transformations."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/spatial_analyzer.py",
        "lang": "python",
        "description": "Relative end-effector kinematics, workspace bounding, damped least-squares (Levenberg-Marquardt) IK solver, and difficulty scoring."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/feature_encoder.py",
        "lang": "python",
        "description": "30-dimensional feature vector assembly, scale normalization, bounds clamping [-1, 1], and layout definition."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/multi_agent_interface.py",
        "lang": "python",
        "description": "Ingestion, schema validation, and normalization of high-level JSON task instructions from Multi-Agent systems (LLMs, VLMs, segmentation)."
    },
    {
        "category": "Spatial Layer Core Modules",
        "path": "spatial_layer/integrated_pipeline.py",
        "lang": "python",
        "description": "Master end-to-end pipeline orchestrating Multi-Agent JSON ingestion, initial object locking, continuous relative distance tracking, and CCDM dynamic constraints/weights."
    },

    # 3. CCDM Dynamics Neural Network (Layer 3)
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/config/model_config.yaml",
        "lang": "yaml",
        "description": "CCDM system dimensions [6, 4, 5, 3, 6], network layer widths, batch size, learning rate."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/models/ccdm.py",
        "lang": "python",
        "description": "Primary FiLM-modulated Context-Conditioned Dynamics Model with dual heads (residual mean and heteroscedastic uncertainty)."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/models/fusion_network.py",
        "lang": "python",
        "description": "FiLM affine modulation layer and trunk fusion network."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/training/losses.py",
        "lang": "python",
        "description": "Gaussian Heteroscedastic Negative Log-Likelihood (NLL) Loss with variance bounding [-10, 5]."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/interfaces/dynamics_interface.py",
        "lang": "python",
        "description": "High-speed CCDMInferenceInterface providing calibrated residual corrections and uncertainty for controllers."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/interfaces/spatial_grounding.py",
        "lang": "python",
        "description": "System-agnostic spatial scene grounding and relative distance module converting depth scenes and poses to physical context."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/interfaces/context_interface.py",
        "lang": "python",
        "description": "Runtime operational context parser and bound clamper (fragility, speed priority, safety priority, task code, crowd density)."
    },
    {
        "category": "CCDM Dynamics Layer (Layer 3)",
        "path": "ccdm/interfaces/controller_adaptation.py",
        "lang": "python",
        "description": "CCDM controller bridge computing dynamic physical constraints and dynamic NMPC cost function weights from context, spatial states, and physical camera scene."
    },

    # 3. Utilities & Synchronization
    {
        "category": "Utilities & Concurrency",
        "path": "spatial_layer/utils/__init__.py",
        "lang": "python",
        "description": "Utility module initialization and convenience imports."
    },
    {
        "category": "Utilities & Concurrency",
        "path": "spatial_layer/utils/math_utils.py",
        "lang": "python",
        "description": "SO(3) and SE(3) math utilities: quaternion normalization/conversions, rotation matrix transformations, and pinhole projection."
    },
    {
        "category": "Utilities & Concurrency",
        "path": "spatial_layer/utils/synchronization.py",
        "lang": "python",
        "description": "Thread-safe data structures: DoubleBuffer, latest-element LatestQueue, and timestamped StateHistory ring buffer."
    },
    {
        "category": "Utilities & Concurrency",
        "path": "spatial_layer/utils/visualizer.py",
        "lang": "python",
        "description": "Interactive MuJoCo 3D overlay visualization: target spheres, end-effector ray lines, and camera debug view."
    },

    # 4. Scripts & Benchmarking
    {
        "category": "Execution Scripts & Benchmarks",
        "path": "scripts/benchmark_latency.py",
        "lang": "python",
        "description": "High-precision 1,000-frame microsecond latency benchmarking across all pipeline stages with statistics (mean, p50, p95, p99)."
    },
    {
        "category": "Execution Scripts & Benchmarks",
        "path": "scripts/run_standalone_demo.py",
        "lang": "python",
        "description": "Interactive standalone demo running live MuJoCo physics simulation, mock ARIA detection generator, and real-time visualization."
    },
    {
        "category": "Execution Scripts & Benchmarks",
        "path": "scripts/run_integrated_demo.py",
        "lang": "python",
        "description": "Complete integrated demonstration connecting Multi-Agent JSON, Spatial Reasoning Layer, and CCDM Dynamic Constraints & NMPC Weights."
    },

    # 5. Test Suite
    {
        "category": "Verification & Test Suite",
        "path": "tests/__init__.py",
        "lang": "python",
        "description": "Test package marker."
    },
    {
        "category": "Verification & Test Suite",
        "path": "tests/test_feature_encoder.py",
        "lang": "python",
        "description": "Unit tests for 30-dim feature vector: shape, dtype (float32), bounds [-1, 1], packing order, and fallback handling."
    },
    {
        "category": "Verification & Test Suite",
        "path": "tests/test_reachability.py",
        "lang": "python",
        "description": "Unit tests for workspace validation, fast damped IK convergence, time budget (<0.8ms), and joint limit adherence."
    },
    {
        "category": "Verification & Test Suite",
        "path": "tests/test_tracking.py",
        "lang": "python",
        "description": "Unit tests for Hungarian tracker: identity preservation, distance gating, occlusion recovery, and rejected detections."
    },
    {
        "category": "Verification & Test Suite",
        "path": "tests/test_transformer.py",
        "lang": "python",
        "description": "Unit tests for coordinate transformations: 2D pinhole deprojection accuracy (<3mm error vs MuJoCo ground truth) and SE(3) consistency."
    },
    {
        "category": "Verification & Test Suite",
        "path": "tests/test_ccdm_integration.py",
        "lang": "python",
        "description": "Integration test suite validating Multi-Agent JSON parsing, continuous relative distance tracking, CCDM 3-input conditioning, and dynamic NMPC constraints/weights."
    },
]

def get_file_content_and_stats(rel_path: str):
    file_path = ROOT_DIR / rel_path
    if not file_path.exists():
        return "", 0, 0
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    loc = len(lines)
    size_bytes = file_path.stat().st_size
    return text, loc, size_bytes

def render_code_html(text: str, lang: str):
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:
        lexer = TextLexer()
    
    formatter = HtmlFormatter(
        nowrap=False,
        linenos="table",
        lineanchors="L",
        anchorlinenos=False,
        cssclass="codehilite",
        style="friendly"
    )
    return pygments.highlight(text, lexer, formatter)

def generate_html_document():
    formatter = HtmlFormatter(style="friendly")
    pygments_css = formatter.get_style_defs('.codehilite')

    total_files = len(FILE_MANIFEST)
    total_loc = 0
    total_bytes = 0

    files_data = []
    for item in FILE_MANIFEST:
        text, loc, size_bytes = get_file_content_and_stats(item["path"])
        total_loc += loc
        total_bytes += size_bytes
        files_data.append({
            **item,
            "text": text,
            "loc": loc,
            "size_bytes": size_bytes
        })

    # Group files by category
    categories = {}
    for f in files_data:
        categories.setdefault(f["category"], []).append(f)

    now_str = datetime.now().strftime("%B %d, %Y")

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spatial Reasoning Layer (Layer 2) — Complete Codebase & Pipeline Specification</title>
<style>
/* -------------------------------------------------------------
   Page Setup & Typography
   ------------------------------------------------------------- */
@page {{
    size: A4 portrait;
    margin: 18mm 14mm 18mm 14mm;
    @bottom-right {{
        content: counter(page);
        font-size: 8pt;
        color: #718096;
    }}
}}

* {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1a202c;
    background-color: #ffffff;
    line-height: 1.45;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #0f172a;
    font-weight: 700;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
}}

h1 {{ font-size: 20pt; line-height: 1.2; }}
h2 {{ font-size: 14pt; border-bottom: 2px solid #3b82f6; padding-bottom: 4px; margin-top: 1.8em; }}
h3 {{ font-size: 11pt; color: #1e3a8a; margin-top: 1.2em; }}
h4 {{ font-size: 10pt; color: #334155; }}

p {{
    margin: 0.4em 0 0.8em 0;
}}

a {{
    color: #2563eb;
    text-decoration: none;
}}
a:hover {{
    text-decoration: underline;
}}

code, pre, .mono {{
    font-family: "Cascadia Code", "Fira Code", "Source Code Pro", Consolas, "Courier New", monospace;
}}

.page-break {{
    page-break-before: always;
    break-before: page;
}}

.no-break {{
    page-break-inside: avoid;
    break-inside: avoid;
}}

/* -------------------------------------------------------------
   Cover Page
   ------------------------------------------------------------- */
.cover-page {{
    height: 100%;
    min-height: 250mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 10mm 4mm 10mm 4mm;
}}

.cover-badge {{
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}}

.cover-title {{
    font-size: 28pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.15;
    margin: 0 0 10px 0;
}}

.cover-subtitle {{
    font-size: 13pt;
    color: #475569;
    line-height: 1.4;
    margin-bottom: 24px;
}}

.cover-accent-bar {{
    width: 80px;
    height: 6px;
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    border-radius: 3px;
    margin-bottom: 30px;
}}

.meta-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px;
    margin: 20px 0;
}}

.meta-item {{
    display: flex;
    flex-direction: column;
}}

.meta-label {{
    font-size: 8pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 3px;
}}

.meta-value {{
    font-size: 10.5pt;
    font-weight: 600;
    color: #0f172a;
}}

.cover-summary-box {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 6px;
    padding: 14px 16px;
    font-size: 9.5pt;
    color: #166534;
    margin-top: 15px;
}}

/* -------------------------------------------------------------
   Architecture Callouts & Diagrams
   ------------------------------------------------------------- */
.pipeline-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 14px;
    margin: 15px 0;
}}

.pipeline-diagram {{
    background: #0f172a;
    color: #f8fafc;
    border-radius: 6px;
    padding: 14px;
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 8.5pt;
    line-height: 1.4;
    white-space: pre;
    overflow-x: auto;
    margin: 10px 0;
}}

table.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
}}

table.data-table th, table.data-table td {{
    border: 1px solid #cbd5e1;
    padding: 6px 10px;
    text-align: left;
}}

table.data-table th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
}}

table.data-table tr:nth-child(even) {{
    background: #f8fafc;
}}

.badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 7.5pt;
    font-weight: 600;
}}
.badge-blue {{ background: #dbeafe; color: #1e40af; }}
.badge-green {{ background: #dcfce7; color: #15803d; }}
.badge-purple {{ background: #f3e8ff; color: #7e22ce; }}
.badge-orange {{ background: #ffedd5; color: #c2410c; }}

/* -------------------------------------------------------------
   Table of Contents
   ------------------------------------------------------------- */
.toc-section {{
    margin: 15px 0;
}}

.toc-category {{
    margin-top: 14px;
    margin-bottom: 6px;
    font-weight: 700;
    color: #1e3a8a;
    font-size: 10pt;
    border-bottom: 1px dashed #cbd5e1;
    padding-bottom: 3px;
}}

.toc-item {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 3px 0;
    font-size: 9pt;
}}

.toc-link {{
    font-family: "Cascadia Code", Consolas, monospace;
    font-weight: 500;
    color: #2563eb;
}}

.toc-dots {{
    flex: 1;
    border-bottom: 1px dotted #94a3b8;
    margin: 0 8px;
}}

.toc-meta {{
    font-size: 8pt;
    color: #64748b;
}}

/* -------------------------------------------------------------
   Code Section & Pygments
   ------------------------------------------------------------- */
.file-card {{
    margin-top: 25px;
    margin-bottom: 30px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    overflow: hidden;
    page-break-inside: auto;
}}

.file-header {{
    background: #f1f5f9;
    border-bottom: 1px solid #cbd5e1;
    padding: 8px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    page-break-inside: avoid;
}}

.file-title {{
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 9.5pt;
    font-weight: 700;
    color: #0f172a;
}}

.file-badges {{
    display: flex;
    gap: 6px;
}}

.file-desc-bar {{
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 6px 14px;
    font-size: 8.5pt;
    color: #475569;
    font-style: italic;
    page-break-inside: avoid;
}}

.code-container {{
    font-size: 7.5pt;
    line-height: 1.35;
    background: #ffffff;
    overflow-x: hidden;
}}

/* Pygments Table Layout */
.codehilite {{
    margin: 0;
    background: #ffffff;
}}

.codehilite table {{
    width: 100%;
    border-collapse: collapse;
    border: none;
    margin: 0;
}}

.codehilite td {{
    padding: 0;
    vertical-align: top;
    border: none;
}}

.codehilite td.linenos {{
    width: 38px;
    padding-right: 8px;
    text-align: right;
    color: #94a3b8;
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
    user-select: none;
    font-size: 7.2pt;
}}

.codehilite td.linenos pre {{
    margin: 0;
    padding: 6px 4px 6px 0;
    font-family: "Cascadia Code", Consolas, monospace;
    color: #94a3b8;
}}

.codehilite td.code {{
    padding-left: 8px;
    background: #ffffff;
}}

.codehilite td.code pre {{
    margin: 0;
    padding: 6px 4px 6px 4px;
    font-family: "Cascadia Code", Consolas, monospace;
    white-space: pre-wrap;
    word-break: break-all;
}}

/* Pygments Highlight Syntax Classes */
{pygments_css}

</style>
</head>
<body>
""")

    # -------------------------------------------------------------
    # 1. COVER PAGE
    # -------------------------------------------------------------
    html_parts.append(f"""
<div class="cover-page">
    <div>
        <div class="cover-badge">Autonomous Manipulation Stack • Layer 2</div>
        <div class="cover-title">Spatial Reasoning Layer</div>
        <div class="cover-subtitle">Complete Codebase Structure, Pipeline Architecture &amp; Full Source Ingestion Manifest</div>
        <div class="cover-accent-bar"></div>

        <p style="font-size: 10pt; color: #334155; max-width: 90%;">
            This self-contained document bundles the entire implementation, algorithmic pipelines, kinematic transformers,
            mathematical utilities, test cases, and simulation configurations for <strong>Layer 2 (Spatial Reasoning Layer)</strong>.
            It is compiled specifically for high-context ingestion by AI agents, autonomous coding assistants, and robotic researchers.
        </p>

        <div class="meta-grid">
            <div class="meta-item">
                <span class="meta-label">Architecture Role</span>
                <span class="meta-value">Layer 2: Spatial Reasoning &amp; State Synchronization</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Robotic Hardware / Model</span>
                <span class="meta-value">Franka Emika Panda 7-DOF Arm (MuJoCo MJCF)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Input Specification</span>
                <span class="meta-value">ARIA JSON 2D Detections + MuJoCo State &amp; Depth Buffer</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Output Specification</span>
                <span class="meta-value">30-Dimensional Feature Vector ($p_t^{{\\text{{spatial}}}} \\in [-1, 1]^{{30}}$, float32)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Latency Constraint</span>
                <span class="meta-value">&lt; 4.0 ms per frame (Mean ~ 1.5 - 2.5 ms)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Concurrency Model</span>
                <span class="meta-value">Asynchronous Worker Thread + Lock-free Double Buffer</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Total Codebase Size</span>
                <span class="meta-value">{total_files} Files • {total_loc:,} Lines of Code • {total_bytes / 1024:.1f} KB</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Compilation Date</span>
                <span class="meta-value">{now_str}</span>
            </div>
        </div>

        <div class="cover-summary-box">
            <strong>AI Agent Ingestion Readiness:</strong> This document contains complete, unabridged source files
            with line-by-line syntax highlighting, accurate mathematical formulations, coordinate transformation matrices,
            and end-to-end unit test validations. All files are referenced with relative repository paths.
        </div>
    </div>

    <div style="font-size: 8pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 8px;">
        Spatial Reasoning Layer • Automated Compilation for Pair-Programming &amp; Agent Tasks • Generated via Antigravity Engine
    </div>
</div>
<div class="page-break"></div>
""")

    # -------------------------------------------------------------
    # 2. EXECUTIVE ARCHITECTURAL SPECIFICATION
    # -------------------------------------------------------------
    html_parts.append(r"""
<h2>1. Architectural Overview &amp; Pipeline Dataflow</h2>

<p>
The <strong>Spatial Reasoning Layer (Layer 2)</strong> bridges high-level semantic perception with low-level
kinodynamic adaptation and predictive control in a 4-layer autonomous robotic manipulation stack:
</p>

<div class="pipeline-box no-break">
    <div style="text-align: center; font-weight: 700; margin-bottom: 8px; color: #1e3a8a;">
        4-Layer Robotic Manipulation Stack Hierarchy
    </div>
    <div class="pipeline-diagram">
+---------------------------------------------------------------------------------------------------+
|  Layer 1: ARIA Vision Agents (Asynchronous 20-30 Hz)                                              |
|  - Ingests camera RGB stream; produces 2D bounding boxes, class labels, and confidence scores.    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v  ARIA JSON Detections {timestamp, bbox, class}
+===================================================================================================+
|  LAYER 2: SPATIAL REASONING LAYER (This Codebase, &lt; 4.0 ms latency)                               |
|  - Timestamp-synchronizes 2D bounding boxes with MuJoCo robot state ring buffer                   |
|  - Linearizes logarithmic OpenGL Z-buffer to metric depth                                         |
|  - Deprojects 2D centroids into 3D camera frame (pinhole model) &amp; transforms to world frame SE(3)  |
|  - Tracks persistent identities across occlusions with Hungarian association &amp; velocity filter  |
|  - Solves damped least-squares (Levenberg-Marquardt) inverse kinematics for reachability          |
|  - Computes grasp difficulty score and validates workspace &amp; joint limits                         |
|  - Packs and normalizes 30-dimensional spatial feature vector p_t^spatial in [-1, 1]              |
+===================================================================================================+
                                                  |
                                                  v  p_t^spatial (30-dim, float32, [-1, 1])
+---------------------------------------------------------------------------------------------------+
|  Layer 3: CCDM Dynamics Adaptation Neural Network (50-100 Hz)                                     |
|  - Predicts contact dynamics, object mass/friction parameters, and environment compliance.        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v  Adapted Dynamics &amp; Target Setpoints
+---------------------------------------------------------------------------------------------------+
|  Layer 4: NMPC (Nonlinear Model Predictive Control) Controller (100-500 Hz)                       |
|  - Real-time Franka Emika Panda joint torque optimization and obstacle-avoidant trajectory control.|
+---------------------------------------------------------------------------------------------------+
    </div>
</div>

<h3>1.1 Pipeline Stages &amp; Mathematical Formulations</h3>

<p>
Every incoming frame is processed through seven deterministic stages:
</p>

<table class="data-table no-break">
    <thead>
        <tr>
            <th style="width: 15%;">Stage</th>
            <th style="width: 25%;">Component</th>
            <th style="width: 35%;">Key Operations &amp; Formulas</th>
            <th style="width: 25%;">Target Latency</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>1. Ingestion</strong></td>
            <td><code>detection_processor.py</code></td>
            <td>ARIA JSON parsing, bounding box validation (clipped to frame dimensions), confidence thresholding (default &ge; 0.3).</td>
            <td>&lt; 0.10 ms</td>
        </tr>
        <tr>
            <td><strong>2. State Query</strong></td>
            <td><code>state_manager.py</code><br><code>synchronization.py</code></td>
            <td>Binary search in <code>StateHistory</code> ring buffer for robot state closest to detection timestamp: $|t_{\text{sim}} - t_{\text{det}}| \to \min$. Fast mirror array copies.</td>
            <td>&lt; 0.05 ms</td>
        </tr>
        <tr>
            <td><strong>3. Depth Render</strong></td>
            <td><code>state_manager.py</code></td>
            <td>Logarithmic OpenGL depth buffer linearization:
            <br>$$d = \frac{z_{\text{near}} \cdot z_{\text{far}}}{z_{\text{far}} - z_b \cdot (z_{\text{far}} - z_{\text{near}})}$$
            Median filter over $(2r+1)\times(2r+1)$ pixel patch around centroid.</td>
            <td>&lt; 0.80 ms</td>
        </tr>
        <tr>
            <td><strong>4. Deprojection</strong></td>
            <td><code>coordinate_transformer.py</code><br><code>math_utils.py</code></td>
            <td>Continuous pinhole deprojection in OpenCV optical frame ($+x$ right, $+y$ down, $+z$ forward):
            <br>$$x_c = \frac{(u - c_x) \cdot d}{f_x}, \quad y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad z_c = d$$
            SE(3) world transformation: $p_w = R_{cw} \cdot p_c + t_{cw}$.</td>
            <td>&lt; 0.08 ms</td>
        </tr>
        <tr>
            <td><strong>5. Tracking</strong></td>
            <td><code>detection_processor.py</code></td>
            <td>Hungarian algorithm (Munkres / <code>linear_sum_assignment</code>) matching detections to active tracks using centroid Euclidean distance with gating threshold (&le; 80 px) and constant-velocity motion prediction.</td>
            <td>&lt; 0.20 ms</td>
        </tr>
        <tr>
            <td><strong>6. Kinematics &amp; IK</strong></td>
            <td><code>spatial_analyzer.py</code></td>
            <td>Radial base distance check: $r \in [r_{\text{min}}, r_{\text{max}}]$. Damped least-squares (Levenberg-Marquardt) IK solver:
            <br>$$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} (p_{\text{target}} - p_{\text{ee}})$$
            Enforces arm joint limits. Grasp difficulty heuristic:
            <br>$$\text{diff} = \text{clip}(0.4 \cdot d_{\text{norm}} + 0.3 \cdot (1 - \text{conf}) + 0.3 \cdot (1 - \text{ik}), 0, 1)$$.</td>
            <td>&lt; 0.80 ms</td>
        </tr>
        <tr>
            <td><strong>7. Encoding</strong></td>
            <td><code>feature_encoder.py</code></td>
            <td>Concatenates scaled metrics, quaternion, binary validity flags, difficulty, and confidence into 30-dim vector $p_t^{\text{spatial}} \in [-1, 1]^{30}$, float32.</td>
            <td>&lt; 0.05 ms</td>
        </tr>
    </tbody>
</table>

<h3>1.2 30-Dimensional Feature Vector Specification ($p_t^{\text{spatial}}$)</h3>

<p>
The output of Layer 2 is a fixed-size 30-element single-precision floating-point array strictly bounded in $[-1, 1]$:
</p>

<table class="data-table no-break">
    <thead>
        <tr>
            <th style="width: 12%;">Slice</th>
            <th style="width: 10%;">Dim</th>
            <th style="width: 30%;">Variable &amp; Content</th>
            <th style="width: 28%;">Scaling &amp; Normalization</th>
            <th style="width: 20%;">Clamping Range</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>[0:3]</code></td>
            <td>3</td>
            <td><code>p_obj_world</code> (Target 3D position in world frame)</td>
            <td>Divided by <code>pos_scale = 2.0</code></td>
            <td>$[-1.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[3:6]</code></td>
            <td>3</td>
            <td><code>p_ee_world</code> (End-effector position in world frame)</td>
            <td>Divided by <code>pos_scale = 2.0</code></td>
            <td>$[-1.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[6:10]</code></td>
            <td>4</td>
            <td><code>obj_quat</code> (Orientation quaternion $[q_w, q_x, q_y, q_z]$)</td>
            <td>$L_2$ normalized unit quaternion</td>
            <td>$[-1.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[10:13]</code></td>
            <td>3</td>
            <td><code>[d_euclidean, d_horizontal, d_vertical]</code> (EE-to-target)</td>
            <td>Divided by <code>dist_scale = 1.5</code></td>
            <td>$[-1.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[13:16]</code></td>
            <td>3</td>
            <td><code>grasp_offset</code> (Grasp offset vector)</td>
            <td>Divided by <code>grasp_scale = 0.5</code></td>
            <td>$[-1.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[16:20]</code></td>
            <td>4</td>
            <td><code>[workspace_valid, ik_feasible, collision_free, limits_ok]</code></td>
            <td>Binary status flags: <code>1.0</code> (valid) or <code>0.0</code> (invalid)</td>
            <td>$\{0.0, 1.0\}$</td>
        </tr>
        <tr>
            <td><code>[20]</code></td>
            <td>1</td>
            <td><code>difficulty_score</code></td>
            <td>$0.4 d_{\text{norm}} + 0.3(1 - \text{conf}) + 0.3(1 - \text{ik})$</td>
            <td>$[0.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[21]</code></td>
            <td>1</td>
            <td><code>detection_confidence</code></td>
            <td>Vision agent detection confidence</td>
            <td>$[0.0, 1.0]$</td>
        </tr>
        <tr>
            <td><code>[22:30]</code></td>
            <td>8</td>
            <td>Reserved zero padding (<code>zeros(8)</code>)</td>
            <td>Reserved for future expansion (e.g. secondary targets)</td>
            <td>$0.0$</td>
        </tr>
    </tbody>
</table>

<h3>1.3 Concurrency &amp; Thread Safety</h3>

<p>
The system employs a strict non-blocking multi-threaded architecture to decouple high-frequency physics from perception:
</p>
<ul>
    <li><strong>Zero-Lock Physics Step:</strong> The physics simulation runs on its own thread at 500 Hz. During each step, it pushes a lightweight <code>RobotStateSnapshot</code> (array copy of joint positions, velocities, and end-effector pose) into a circular ring buffer (<code>StateHistory</code>). No locks are taken on the physics thread.</li>
    <li><strong>Mirror MjData in Worker Thread:</strong> The Layer-2 worker thread operates asynchronously. When an ARIA detection arrives, the worker copies the timestamp-matched state into an independent <em>mirror</em> <code>MjData</code> instance. Kinematics, depth buffer rendering, and inverse kinematics are computed entirely on the mirror data, ensuring zero interference with the live physics loop.</li>
    <li><strong>Seqlock Double-Buffering:</strong> The worker publishes computed <code>SpatialResult</code> objects into a thread-safe <code>DoubleBuffer</code>. Downstream consumers (Layer 3 CCDM / Layer 4 NMPC) read the latest results lock-free with microsecond-level retrieval latency.</li>
</ul>

<div class="page-break"></div>
""")

    # -------------------------------------------------------------
    # 3. DIRECTORY STRUCTURE & TABLE OF CONTENTS
    # -------------------------------------------------------------
    html_parts.append(f"""
<h2>2. Directory Structure &amp; File Navigation Index</h2>

<p>
The repository is organized into clearly modularized packages separating configuration, core algorithms,
utilities, simulation models, benchmarks, and unit tests:
</p>

<div class="pipeline-diagram no-break">
spatial_reasoning_layer/
├── config/
│   └── default_config.yaml         # Master camera, workspace, IK &amp; normalization parameters
├── models/
│   └── franka_scene.xml            # Franka Emika Panda 7-DOF arm &amp; overhead camera (MuJoCo MJCF)
├── spatial_layer/
│   ├── __init__.py                 # Package exports &amp; version definition
│   ├── core.py                     # Master pipeline orchestrator &amp; asynchronous worker
│   ├── state_manager.py            # MuJoCo state extraction &amp; OpenGL depth buffer linearization
│   ├── detection_processor.py      # ARIA JSON ingestion, bbox validation &amp; Hungarian tracking
│   ├── coordinate_transformer.py   # 2D pinhole deprojection &amp; SE(3) camera-to-world transform
│   ├── spatial_analyzer.py         # Relative EE metrics, workspace bounding &amp; damped LS IK
│   ├── feature_encoder.py          # 30-dim vector assembly, normalization &amp; clamping
│   └── utils/
│       ├── __init__.py             # Utility package exports
│       ├── math_utils.py           # Quaternions, rotation matrices, SE(3) transforms &amp; projection
│       ├── synchronization.py      # Thread-safe DoubleBuffer, LatestQueue &amp; StateHistory ring buffer
│       └── visualizer.py           # Real-time MuJoCo 3D overlay markers &amp; ray lines
├── scripts/
│   ├── benchmark_latency.py        # Microsecond latency profiler across 1,000 frames
│   └── run_standalone_demo.py      # Live interactive MuJoCo simulation &amp; mock ARIA feed
├── tests/
│   ├── __init__.py                 # Test package marker
│   ├── test_feature_encoder.py     # Vector shape, dtype, bounds [-1, 1], and layout tests
│   ├── test_reachability.py        # Workspace boundaries, fast IK convergence &amp; joint limits
│   ├── test_tracking.py            # Hungarian identity continuity &amp; distance gating tests
│   └── test_transformer.py         # 3D spatial deprojection accuracy (&lt; 3mm error vs MuJoCo)
├── README.md                       # Architectural documentation &amp; quickstart guide
├── requirements.txt                # Python package dependencies
└── setup.py                        # Setuptools build script
</div>

<div class="toc-section">
    <h3>Complete Table of Contents</h3>
""")

    file_counter = 1
    for cat_name, file_list in categories.items():
        html_parts.append(f'<div class="toc-category">{cat_name}</div>')
        for f in file_list:
            anchor = f"file-{f['path'].replace('/', '-').replace('.', '-')}"
            html_parts.append(f"""
    <div class="toc-item">
        <span class="toc-link"><a href="#{anchor}">{file_counter}. {f['path']}</a></span>
        <span class="toc-dots"></span>
        <span class="toc-meta">{f['loc']} LOC • {f['size_bytes']:,} B</span>
    </div>
""")
            file_counter += 1

    html_parts.append("""
</div>
<div class="page-break"></div>
""")

    # -------------------------------------------------------------
    # 4. SOURCE CODE MANIFEST
    # -------------------------------------------------------------
    html_parts.append("""
<h2>3. Complete Source Code Manifest</h2>
<p>
The following sections contain the full, unabridged source code of all 23 files in the repository.
Each file is presented with its relative path, line count, file size, semantic summary, and syntax-highlighted code.
</p>
""")

    file_counter = 1
    current_category = None

    for f in files_data:
        if f["category"] != current_category:
            current_category = f["category"]
            html_parts.append(f"""
<div class="page-break"></div>
<h2 style="color: #1e3a8a; border-bottom: 2px solid #1e3a8a; margin-top: 20px;">
    Section {file_counter // 5 + 1}: {current_category}
</h2>
""")

        anchor = f"file-{f['path'].replace('/', '-').replace('.', '-')}"
        highlighted_code = render_code_html(f["text"], f["lang"])

        # Determine category badge color
        badge_cls = "badge-blue"
        if "Test" in f["category"]:
            badge_cls = "badge-green"
        elif "Utilities" in f["category"]:
            badge_cls = "badge-purple"
        elif "Scripts" in f["category"]:
            badge_cls = "badge-orange"

        html_parts.append(f"""
<div class="file-card" id="{anchor}">
    <div class="file-header">
        <span class="file-title">File {file_counter}: {f['path']}</span>
        <div class="file-badges">
            <span class="badge {badge_cls}">{f['lang'].upper()}</span>
            <span class="badge badge-blue">{f['loc']} LOC</span>
            <span class="badge badge-purple">{f['size_bytes']:,} bytes</span>
        </div>
    </div>
    <div class="file-desc-bar">
        <strong>Description:</strong> {f['description']}
    </div>
    <div class="code-container">
        {highlighted_code}
    </div>
</div>
""")
        file_counter += 1

    # -------------------------------------------------------------
    # 5. APPENDIX: AI AGENT TASK & INGESTION GUIDE
    # -------------------------------------------------------------
    html_parts.append(r"""
<div class="page-break"></div>
<h2>4. Appendix: AI Agent Task &amp; Ingestion Reference</h2>

<div class="pipeline-box no-break">
    <h3 style="margin-top: 0;">Key Class Interfaces &amp; Data Contracts</h3>
    <table class="data-table">
        <thead>
            <tr>
                <th style="width: 25%;">Class / Function</th>
                <th style="width: 25%;">Module</th>
                <th style="width: 50%;">Role &amp; Signature Details</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>SpatialReasoningLayer</code></td>
                <td><code>spatial_layer.core</code></td>
                <td>Master coordinator. Ingests ARIA detections via <code>submit_detection(dict)</code>, synchronizes state via <code>sync_state()</code>, processes frames in background worker, and yields <code>SpatialResult</code> via <code>get_latest_features()</code>.</td>
            </tr>
            <tr>
                <td><code>MuJoCoStateManager</code></td>
                <td><code>spatial_layer.state_manager</code></td>
                <td>Extracts end-effector Cartesian pose, joint coordinates, and renders logarithmic depth buffers. Linearizes OpenGL depth into metric coordinates.</td>
            </tr>
            <tr>
                <td><code>CoordinateTransformer</code></td>
                <td><code>spatial_layer.coordinate_transformer</code></td>
                <td>Transforms 2D continuous pixel coords <code>(u, v)</code> + metric depth <code>d</code> into 3D camera coordinates (OpenCV convention) and applies $T_{cw} \in \text{SE}(3)$ to obtain world coordinates.</td>
            </tr>
            <tr>
                <td><code>ObjectDetectionProcessor</code></td>
                <td><code>spatial_layer.detection_processor</code></td>
                <td>Validates ARIA JSON schema. Maintains persistent tracks across frames using Hungarian bipartite matching on centroids with distance gating and constant-velocity prediction.</td>
            </tr>
            <tr>
                <td><code>SpatialAnalyzer</code></td>
                <td><code>spatial_layer.spatial_analyzer</code></td>
                <td>Calculates relative distances ($d_{\text{euclid}}, d_{\text{horiz}}, d_{\text{vert}}$), evaluates radial base workspace bounds, runs Levenberg-Marquardt damped IK solver, and computes grasp difficulty.</td>
            </tr>
            <tr>
                <td><code>FeatureEncoder</code></td>
                <td><code>spatial_layer.feature_encoder</code></td>
                <td>Encodes spatial metrics, quaternions, and status flags into the normalized 30-dim vector $p_t^{\text{spatial}} \in [-1, 1]^{30}$, float32.</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="cover-summary-box no-break">
    <strong>Prompting Advice for AI Agents:</strong><br>
    When continuing work on this codebase:
    <ol style="margin: 6px 0 0 16px; padding: 0;">
        <li><strong>Strict Latency Budgets:</strong> All algorithms on the critical path must execute in under 4.0 ms total. The IK solver has a hard budget of 0.8 ms. Avoid heavy heap allocations inside <code>process_frame()</code>.</li>
        <li><strong>Coordinate System Consistency:</strong> MuJoCo uses $+z$ up, $+x$ forward/right depending on base frame. OpenCV optical frame uses $+x$ right, $+y$ down, $+z$ forward. Use <code>mujoco_rot_to_opencv()</code> when converting camera matrices.</li>
        <li><strong>Thread Safety:</strong> Never call <code>mujoco.mj_step()</code> or mutate the live <code>data</code> inside the worker thread. Always perform kinematics and rendering on <code>_mirror_data</code>.</li>
    </ol>
</div>

</body>
</html>
""")

    full_html = "".join(html_parts)
    return full_html

def main():
    print("1. Compiling codebase HTML document...")
    html_content = generate_html_document()
    
    html_path = ROOT_DIR / "codebase_and_pipeline_structure.html"
    pdf_path = ROOT_DIR / "spatial_reasoning_layer_codebase_and_pipeline.pdf"
    
    html_path.write_text(html_content, encoding="utf-8")
    print(f"   Saved HTML to: {html_path} ({len(html_content):,} chars)")
    
    print("2. Launching headless browser to render PDF...")
    edge_exe = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge_exe.exists():
        edge_exe = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
    
    if not edge_exe.exists():
        print(f"Error: Microsoft Edge not found at {edge_exe}")
        sys.exit(1)
        
    temp_profile = ROOT_DIR / ".edge_temp"
    temp_profile.mkdir(exist_ok=True)
    
    cmd = [
        str(edge_exe),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={temp_profile.resolve()}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        str(html_path.resolve())
    ]
    
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"Edge exited with error code {proc.returncode}:\n{proc.stderr}")
        sys.exit(proc.returncode)
        
    if pdf_path.exists():
        pdf_size = pdf_path.stat().st_size
        print(f"3. PDF successfully generated!")
        print(f"   Output file: {pdf_path}")
        print(f"   Size: {pdf_size:,} bytes ({pdf_size / (1024*1024):.2f} MB)")
    else:
        print("Error: PDF was not generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()

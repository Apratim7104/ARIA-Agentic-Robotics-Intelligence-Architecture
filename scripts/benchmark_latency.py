"""Per-module microsecond latency profiler for the Spatial Reasoning Layer (Layer 2).

Runs 1,000 continuous frames through the full pipeline and prints a formatted
latency table covering:
  - State Query Latency
  - Depth Render & Linearization Latency
  - Hungarian Tracking & Parsing Latency
  - 2D-to-3D Deprojection Latency
  - Fast IK & Reachability Latency
  - Feature Normalization Latency
  - Total End-to-End Latency (Enforces hard assertion: Mean < 4.0 ms)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import mujoco
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from spatial_layer import SpatialReasoningLayer

MODEL_PATH = os.path.join(ROOT_DIR, "models", "franka_scene.xml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default_config.yaml")

STAGE_LABELS = [
    ("state_query", "State Query Latency"),
    ("depth", "Depth Render & Linearization Latency"),
    ("tracking", "Hungarian Tracking & Parsing Latency"),
    ("deprojection", "2D-to-3D Deprojection Latency"),
    ("ik_reachability", "Fast IK & Reachability Latency"),
    ("feature_encoding", "Feature Normalization Latency"),
    ("total", "Total End-to-End Latency"),
]


def run_benchmark(num_frames: int = 1000, force_depth_every: int = 1) -> Dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    qadr = model.jnt_qposadr[jnt_id]

    rng = np.random.default_rng(42)
    layer = SpatialReasoningLayer(model, data, CONFIG_PATH)

    try:
        # Warm-up (initializes OpenGL context, caches, and JITs numpy paths)
        for _ in range(15):
            layer.sync_state()
            gt_pos = data.geom_xpos[geom_id]
            uv, _ = layer.transformer.world_to_pixel(gt_pos)
            layer.process_frame({
                "timestamp": time.time(),
                "primary_target": {
                    "id": 0, "class": "cube", "confidence": 0.96,
                    "bbox": [float(uv[0] - 14), float(uv[1] - 14),
                             float(uv[0] + 14), float(uv[1] + 14)],
                },
                "secondary_objects": [],
            }, force_depth_refresh=True)

        records: Dict[str, List[float]] = {k: [] for k, _ in STAGE_LABELS}
        errors_mm: List[float] = []

        for i in range(num_frames):
            # Move block along a smooth Lissajous curve on the table
            theta = 2.0 * np.pi * i / 250.0
            pos = np.array([
                0.55 + 0.10 * np.sin(theta),
                0.15 * np.sin(2.0 * theta),
                0.425,
            ])
            data.qpos[qadr:qadr + 3] = pos
            mujoco.mj_forward(model, data)
            snap = layer.sync_state()

            gt_pos = data.geom_xpos[geom_id]
            uv, _ = layer.transformer.world_to_pixel(gt_pos)
            noisy_uv = uv + rng.normal(0.0, 0.5, size=2)
            half = 14.0
            bbox = [
                float(np.clip(noisy_uv[0] - half, 0.0, 319.0)),
                float(np.clip(noisy_uv[1] - half, 0.0, 239.0)),
                float(np.clip(noisy_uv[0] + half, 1.0, 320.0)),
                float(np.clip(noisy_uv[1] + half, 1.0, 240.0)),
            ]
            msg = {
                "timestamp": snap.wall_time,
                "primary_target": {
                    "id": 0,
                    "class": "cube",
                    "confidence": float(rng.uniform(0.88, 0.99)),
                    "bbox": bbox,
                },
                "secondary_objects": [],
            }
            force_refresh = (force_depth_every > 0) and (i % force_depth_every == 0)
            res = layer.process_frame(msg, force_depth_refresh=force_refresh, snapshot=snap)

            for k, _ in STAGE_LABELS:
                records[k].append(res.timings_us.get(k, 0.0))
            if res.valid and res.p_obj_world is not None:
                errors_mm.append(float(np.linalg.norm(res.p_obj_world - gt_pos) * 1e3))

        stats = {k: np.asarray(v, dtype=np.float64) for k, v in records.items()}
        stats["errors_mm"] = np.asarray(errors_mm, dtype=np.float64)
        return stats
    finally:
        layer.close()


def print_report(stats: Dict[str, np.ndarray], num_frames: int) -> None:
    line = "=" * 98
    print(f"\n{line}")
    print(f" SPATIAL REASONING LAYER (LAYER 2) - LATENCY BENCHMARK ({num_frames} CONTINUOUS FRAMES)")
    print(line)
    header = f"{'Pipeline Module':<40} | {'Mean (us)':>10} | {'Std (us)':>9} | {'P50 (us)':>9} | {'P99 (us)':>9} | {'Mean (ms)':>9}"
    print(header)
    print("-" * len(line))

    for key, label in STAGE_LABELS:
        arr = stats[key]
        mean_us = float(np.mean(arr))
        std_us = float(np.std(arr))
        p50_us = float(np.percentile(arr, 50))
        p99_us = float(np.percentile(arr, 99))
        mean_ms = mean_us * 1e-3
        if key == "total":
            print("-" * len(line))
        print(f"{label:<40} | {mean_us:10.2f} | {std_us:9.2f} | {p50_us:9.2f} | {p99_us:9.2f} | {mean_ms:9.4f}")

    print(line)
    mean_total_ms = float(np.mean(stats["total"])) * 1e-3
    p99_total_ms = float(np.percentile(stats["total"], 99)) * 1e-3
    mean_err_mm = float(np.mean(stats["errors_mm"])) if stats["errors_mm"].size else float("nan")
    print(f" 3D Localization Error (noisy 2D stream): Mean = {mean_err_mm:.2f} mm")
    print(f" End-to-End Latency Target (< 4.000 ms) : Mean = {mean_total_ms:.4f} ms | P99 = {p99_total_ms:.4f} ms")
    print(f"{line}\n")

    assert mean_total_ms < 4.0, (
        f"Hard latency assertion failed: Mean end-to-end latency {mean_total_ms:.4f} ms >= 4.0 ms"
    )
    print("[PASS] Hard assertion satisfied: Mean End-to-End Latency < 4.0 ms.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Layer-2 microsecond latency.")
    parser.add_argument("--frames", type=int, default=1000, help="Number of continuous frames.")
    parser.add_argument(
        "--depth-refresh-every",
        type=int,
        default=1,
        help="Force fresh offscreen depth render every N frames (0 = rely on 5ms cache).",
    )
    args = parser.parse_args()
    stats = run_benchmark(num_frames=args.frames, force_depth_every=args.depth_refresh_every)
    print_report(stats, args.frames)


if __name__ == "__main__":
    main()

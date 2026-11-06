"""Live MuJoCo simulation with mock ARIA vision stream (20 Hz) and 3D viewer overlays.

Features:
- Separate mock ARIA vision thread at 20 Hz: projects ground-truth red block into
  camera pixel coordinates, adds Gaussian noise (sigma = 1.0 px), and pushes
  JSON into Layer 2.
- Background Layer-2 worker thread computing 3D localization, fast IK, workspace
  validity, and the normalized 30-D feature vector.
- Interactive MuJoCo viewer with 3D overlay markers via ``mujoco.mjvGeom``:
    * Red sphere at ground-truth object position.
    * Blue sphere at estimated reconstructed 3D position from Layer 2.
    * Connector line from End-Effector to estimated target.
- Console readout of the 30-D feature vector and reachability status.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from spatial_layer import SpatialReasoningLayer
from spatial_layer.utils.visualizer import SpatialVisualizer, format_console_status

MODEL_PATH = os.path.join(ROOT_DIR, "models", "franka_scene.xml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default_config.yaml")


class MockAriaVisionStreamer:
    """Generates noisy 20 Hz ARIA JSON detections from ground-truth simulation state."""

    def __init__(self, layer: SpatialReasoningLayer, geom_id: int,
                 rate_hz: float = 20.0, noise_sigma_px: float = 1.0) -> None:
        self.layer = layer
        self.geom_id = geom_id
        self.period = 1.0 / rate_hz
        self.sigma = noise_sigma_px
        self._lock = threading.Lock()
        self._gt_pos = np.array([0.5, 0.0, 0.425], dtype=np.float64)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rng = np.random.default_rng(12345)

    def update_ground_truth(self, gt_pos: np.ndarray) -> None:
        with self._lock:
            self._gt_pos = gt_pos.copy()

    def get_ground_truth(self) -> np.ndarray:
        with self._lock:
            return self._gt_pos.copy()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="MockAriaVision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _loop(self) -> None:
        w = self.layer.transformer.width
        h = self.layer.transformer.height
        while not self._stop.is_set():
            t0 = time.perf_counter()
            gt = self.get_ground_truth()
            uv, depth = self.layer.transformer.world_to_pixel(gt)
            if depth > 0.05:
                noisy_uv = uv + self._rng.normal(0.0, self.sigma, size=2)
                # Apparent half-size in pixels for a 5 cm cube
                half_px = max(8.0, 0.5 * 0.05 * self.layer.transformer.fx / depth)
                x1 = float(np.clip(noisy_uv[0] - half_px, 0.0, w - 2.0))
                y1 = float(np.clip(noisy_uv[1] - half_px, 0.0, h - 2.0))
                x2 = float(np.clip(noisy_uv[0] + half_px, x1 + 1.0, w))
                y2 = float(np.clip(noisy_uv[1] + half_px, y1 + 1.0, h))
                payload = json.dumps({
                    "timestamp": time.time(),
                    "primary_target": {
                        "id": 0,
                        "class": "cube",
                        "confidence": float(np.clip(self._rng.normal(0.95, 0.015), 0.75, 0.99)),
                        "bbox": [x1, y1, x2, y2],
                    },
                    "secondary_objects": [],
                })
                self.layer.submit_detection(payload)

            elapsed = time.perf_counter() - t0
            sleep_s = max(0.0, self.period - elapsed)
            self._stop.wait(sleep_s)


def run_demo(duration: Optional[float] = None, headless: bool = False,
             move_target: bool = True) -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    qadr = model.jnt_qposadr[jnt_id]
    vadr = model.jnt_dofadr[jnt_id]

    visualizer = SpatialVisualizer()
    # Set initial target pose on the figure-8 curve before starting threads
    if move_target:
        data.qpos[qadr + 0] = 0.55
        data.qpos[qadr + 1] = 0.0
        data.qpos[qadr + 2] = 0.425
        data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    layer = SpatialReasoningLayer(model, data, CONFIG_PATH)
    streamer = MockAriaVisionStreamer(layer, geom_id, rate_hz=20.0, noise_sigma_px=1.0)

    layer.sync_state()
    streamer.update_ground_truth(data.geom_xpos[geom_id])

    start_wall = time.perf_counter()
    last_print = 0.0
    home_ctrl = data.ctrl[:7].copy()

    def step_sim() -> np.ndarray:
        t = data.time
        if move_target:
            # Smooth figure-8 trajectory on the table surface (z = 0.425 m)
            data.qpos[qadr + 0] = 0.55 + 0.12 * np.sin(0.6 * t)
            data.qpos[qadr + 1] = 0.18 * np.sin(1.2 * t)
            data.qpos[qadr + 2] = 0.425
            data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[vadr:vadr + 6] = 0.0

        # Gentle observation motion on joint 1 so EE moves without occluding the overhead camera
        data.ctrl[:7] = home_ctrl
        data.ctrl[0] = home_ctrl[0] + 0.25 * np.sin(0.8 * t)

        mujoco.mj_step(model, data)
        layer.sync_state()
        gt_pos = data.geom_xpos[geom_id].copy()
        streamer.update_ground_truth(gt_pos)
        return gt_pos

    try:
        if not headless:
            try:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    layer.start()
                    streamer.start()
                    start_wall = time.perf_counter()
                    print("[SpatialReasoningLayer] Live MuJoCo viewer launched (20 Hz mock ARIA stream).")
                    while viewer.is_running():
                        if duration is not None and (time.perf_counter() - start_wall) >= duration:
                            break
                        step_start = time.perf_counter()
                        gt_pos = step_sim()
                        res = layer.get_latest_result()

                        visualizer.update_scene(viewer.user_scn, res, gt_pos=gt_pos)
                        viewer.sync()

                        now = time.perf_counter()
                        if now - last_print >= 0.5 and res.valid:
                            print(format_console_status(res, gt_pos))
                            last_print = now

                        rem = model.opt.timestep - (time.perf_counter() - step_start)
                        if rem > 0:
                            time.sleep(rem)
                return
            except Exception as exc:
                print(f"[SpatialReasoningLayer] Viewer unavailable ({exc}); falling back to headless loop.")

        # Headless simulation loop (still exercises mujoco.mjvGeom marker population)
        layer.start()
        streamer.start()
        start_wall = time.perf_counter()
        scn = mujoco.MjvScene(model, maxgeom=200)
        run_for = duration if duration is not None else 3.0
        print(f"[SpatialReasoningLayer] Running standalone demo loop for {run_for:.1f}s...")
        while (time.perf_counter() - start_wall) < run_for:
            gt_pos = step_sim()
            res = layer.get_latest_result()
            n_markers = visualizer.update_scene(scn, res, gt_pos=gt_pos)
            now = time.perf_counter()
            if now - last_print >= 0.5 and res.valid:
                print(format_console_status(res, gt_pos) + f" | mjvGeoms={n_markers}")
                last_print = now
            time.sleep(model.opt.timestep)
    finally:
        streamer.stop()
        layer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone Layer-2 MuJoCo demo.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional run duration in seconds (default: run until viewer closed).")
    parser.add_argument("--headless", action="store_true",
                        help="Run without opening a GUI window.")
    parser.add_argument("--static-target", action="store_true",
                        help="Keep the red block stationary instead of moving along a figure-8.")
    args = parser.parse_args()
    run_demo(duration=args.duration, headless=args.headless, move_target=not args.static_target)


if __name__ == "__main__":
    main()

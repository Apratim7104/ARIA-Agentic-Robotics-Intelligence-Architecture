"""Layer-2 master engine and threading coordinator.

Data flow::

    ARIA (Layer 1) --submit_detection()--> LatestQueue(maxsize=2) --+
                                                                    v
    physics loop --sync_state()--> StateHistory  -->  worker thread: process_frame()
                                                                    |
    Layer 3 / NMPC <--get_latest_features()-- DoubleBuffer <--------+

The worker owns a *mirror* ``MjData``: each frame it loads the robot-state
snapshot closest in time to the detection timestamp, recomputes kinematics and
renders depth from it. The physics thread therefore never shares ``MjData``
with the worker and never takes a lock (``sync_state`` is a few array copies).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mujoco
import numpy as np
import yaml

from .coordinate_transformer import CoordinateTransformer
from .detection_processor import Detection, FrameDetections, ObjectDetectionProcessor
from .feature_encoder import FEATURE_DIM, FeatureEncoder
from .spatial_analyzer import SpatialAnalysis, SpatialAnalyzer
from .state_manager import MuJoCoStateManager, RobotStateSnapshot
from .utils.synchronization import DoubleBuffer, LatestQueue, StateHistory

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "config", "default_config.yaml")

STAGES = ("state_query", "depth", "tracking", "deprojection", "ik_reachability",
          "feature_encoding", "total")


def load_config(config_path: Optional[str] = None) -> dict:
    with open(config_path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class SpatialResult:
    """Full output of one Layer-2 pipeline pass."""

    frame_id: int
    timestamp: float                     # detection timestamp (ARIA clock)
    sim_time: float                      # sim time of the state used
    valid: bool                          # a valid primary target was localized
    features: np.ndarray                 # (30,) float32 in [-1, 1]
    p_obj_world: Optional[np.ndarray] = None
    p_ee_world: Optional[np.ndarray] = None
    ee_quat: Optional[np.ndarray] = None
    track_id: int = -1
    confidence: float = 0.0
    analysis: Optional[SpatialAnalysis] = None
    secondary_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    secondary_track_ids: List[int] = field(default_factory=list)
    timings_us: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"


class SpatialReasoningLayer:
    """Standalone Spatial Reasoning Layer (Layer 2).

    Args:
        model: ``MjModel`` of the scene.
        data: live ``MjData`` stepped by the physics loop.
        config_path: YAML config (defaults to ``config/default_config.yaml``).
        config: optional already-loaded config dict (overrides ``config_path``).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 config_path: Optional[str] = None, config: Optional[dict] = None) -> None:
        self.config = config if config is not None else load_config(config_path)
        cam_cfg = self.config["camera"]
        self.model = model
        self.data = data

        # Live-state reader (physics thread) and worker-side mirror.
        self._live = MuJoCoStateManager(model, data, cam_cfg.get("name", "overhead_cam"),
                                        self.config)
        self._mirror_data = mujoco.MjData(model)
        self.state_manager = MuJoCoStateManager(model, self._mirror_data,
                                                cam_cfg.get("name", "overhead_cam"), self.config)
        self.transformer = CoordinateTransformer(cam_cfg["intrinsics"], cam_cfg["extrinsics"])
        self.detector = ObjectDetectionProcessor(self.config)
        self.analyzer = SpatialAnalyzer(model, self.config)
        self.encoder = FeatureEncoder(self.config)

        depth_cfg = self.config.get("depth", {})
        self._patch_radius = int(depth_cfg.get("patch_radius", 1))
        self._dmin = float(depth_cfg.get("min_valid_depth", 0.05))
        self._dmax = float(depth_cfg.get("max_valid_depth", 10.0))
        self._class_sizes = {k: np.asarray(v.get("size", [0, 0, 0]), dtype=np.float64)
                             for k, v in (self.config.get("object_classes") or {}).items()}

        qsize = int(self.config.get("pipeline", {}).get("inbound_queue_size", 2))
        self._inbound: LatestQueue = LatestQueue(maxsize=qsize)
        self._outbound: DoubleBuffer[SpatialResult] = DoubleBuffer()
        self._history: StateHistory[RobotStateSnapshot] = StateHistory(capacity=128)

        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_id = 0
        self.frames_processed = 0
        self.worker_errors: List[str] = []
        self._latency_ema_us = 0.0

        self.sync_state()
        self._publish(self._empty_result(self._history.latest(), time.time(), "init"))

    # ------------------------------------------------------------------ #
    # Producer-side API
    # ------------------------------------------------------------------ #
    def sync_state(self) -> RobotStateSnapshot:
        """Snapshot the live sim state. Call from the physics loop after mj_step."""
        snap = self._live.snapshot()
        self._history.push(snap)
        return snap

    def submit_detection(self, message: Union[str, bytes, dict]) -> bool:
        """Enqueue an ARIA message (non-blocking). False if a stale frame was dropped."""
        return self._inbound.put(message)

    @property
    def dropped_frames(self) -> int:
        return self._inbound.dropped

    # ------------------------------------------------------------------ #
    # Consumer-side API (Layer 3 / NMPC)
    # ------------------------------------------------------------------ #
    def get_latest_features(self) -> np.ndarray:
        """Most recent 30-dim feature vector (lock-free read, returns a copy)."""
        res, _ = self._outbound.read()
        return res.features.copy()

    def get_latest_result(self) -> SpatialResult:
        res, _ = self._outbound.read()
        return res

    @property
    def output_sequence(self) -> int:
        return self._outbound.sequence

    # ------------------------------------------------------------------ #
    # Worker lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> "SpatialReasoningLayer":
        if self._worker is not None and self._worker.is_alive():
            return self
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="SpatialLayerWorker", daemon=True)
        self._worker.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout)
        self._worker = None

    def close(self) -> None:
        self.stop()
        self.state_manager.close()
        self._live.close()

    def __enter__(self) -> "SpatialReasoningLayer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                msg = self._inbound.get_latest(timeout=0.05)
                if msg is None:
                    continue
                try:
                    self.process_frame(msg)
                except Exception as exc:  # keep the worker alive on bad frames
                    self.worker_errors.append(repr(exc))
        finally:
            # GL resources must be released by the thread that owns them.
            self.state_manager.close()

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def _publish(self, result: SpatialResult) -> None:
        self._outbound.write(result)

    def _empty_result(self, snap: Optional[RobotStateSnapshot], ts: float,
                      status: str, p_ee: Optional[np.ndarray] = None) -> SpatialResult:
        if p_ee is None and snap is not None and self._worker is None:
            p_ee = self._live.get_ee_pose()[0]       # only safe outside the worker
        return SpatialResult(frame_id=self._frame_id, timestamp=ts,
                             sim_time=snap.sim_time if snap else 0.0, valid=False,
                             features=self.encoder.empty(p_ee), p_ee_world=p_ee, status=status)

    def _localize(self, det: Detection, depth: np.ndarray) -> Optional[np.ndarray]:
        u, v = det.center
        d = self.transformer.sample_depth(depth, u, v, self._patch_radius, self._dmin, self._dmax)
        if d is None:
            return None
        size = self._class_sizes.get(det.cls, self._class_sizes.get("default", np.zeros(3)))
        d += self.transformer.centroid_depth_offset(size)
        return self.transformer.pixel_to_world(u, v, d)

    def process_frame(self, message: Union[str, bytes, dict],
                      force_depth_refresh: bool = False,
                      snapshot: Optional[RobotStateSnapshot] = None) -> SpatialResult:
        """Run the full Layer-2 pipeline synchronously on one ARIA message."""
        pc = time.perf_counter
        t_start = pc()
        self._frame_id += 1
        timings: Dict[str, float] = {}

        # 1) Parse + validate + Hungarian tracking
        t = pc()
        frame: FrameDetections = self.detector.process(message)
        timings["tracking"] = (pc() - t) * 1e6

        # 2) Robot state synchronised to the detection timestamp
        t = pc()
        snap = snapshot or self._history.closest(frame.timestamp) or self.sync_state()
        sm = self.state_manager
        sm.apply_snapshot(snap)
        joint_state = {"qpos": self._mirror_data.qpos}         # no-copy view for IK
        p_ee, q_ee = sm.get_ee_pose()
        base_pos = sm.get_base_position()
        timings["state_query"] = (pc() - t) * 1e6

        if frame.primary is None:
            res = self._empty_result(snap, frame.timestamp, "no_valid_primary", p_ee)
            res.p_ee_world, res.ee_quat = p_ee, q_ee
            res.features = self.encoder.empty(p_ee)
            for k in ("depth", "deprojection", "ik_reachability", "feature_encoding"):
                timings[k] = 0.0
            timings["total"] = (pc() - t_start) * 1e6
            res.timings_us = timings
            self._publish(res)
            return res

        # 3) Depth render + linearization
        t = pc()
        depth = sm.get_depth_image(force_refresh=force_depth_refresh)
        timings["depth"] = (pc() - t) * 1e6

        # 4) 2D -> 3D deprojection (primary + vectorized secondaries)
        t = pc()
        prim = frame.primary
        p_obj = self._localize(prim, depth)
        sec_world = np.zeros((0, 3))
        if frame.secondary:
            uv = np.stack([s.center for s in frame.secondary])
            depths = self.transformer.batch_sample_depth(depth, uv, self._patch_radius,
                                                         self._dmin, self._dmax)
            offs = np.array([self.transformer.centroid_depth_offset(
                self._class_sizes.get(s.cls, self._class_sizes.get("default", np.zeros(3))))
                for s in frame.secondary])
            sec_world = self.transformer.batch_pixel_to_world(uv, depths + offs)
        timings["deprojection"] = (pc() - t) * 1e6

        if p_obj is None:
            res = self._empty_result(snap, frame.timestamp, "no_depth", p_ee)
            res.p_ee_world, res.ee_quat = p_ee, q_ee
            res.features = self.encoder.empty(p_ee)
            timings["ik_reachability"] = timings["feature_encoding"] = 0.0
            timings["total"] = (pc() - t_start) * 1e6
            res.timings_us = timings
            self._publish(res)
            return res

        # 5) Spatial analysis: distances, workspace, IK, grasp, difficulty
        t = pc()
        analysis = self.analyzer.analyze(p_obj, p_ee, base_pos, joint_state["qpos"],
                                         prim.confidence, prim.cls, sec_world)
        timings["ik_reachability"] = (pc() - t) * 1e6

        # 6) Feature encoding
        t = pc()
        feats = self.encoder.encode_analysis(p_obj, p_ee, analysis, prim.confidence, prim.quat)
        timings["feature_encoding"] = (pc() - t) * 1e6
        timings["total"] = (pc() - t_start) * 1e6

        res = SpatialResult(
            frame_id=self._frame_id, timestamp=frame.timestamp, sim_time=snap.sim_time,
            valid=True, features=feats, p_obj_world=p_obj, p_ee_world=p_ee, ee_quat=q_ee,
            track_id=prim.track_id, confidence=prim.confidence, analysis=analysis,
            secondary_world=sec_world,
            secondary_track_ids=[s.track_id for s in frame.secondary], timings_us=timings)
        self._publish(res)
        self.frames_processed += 1
        self._latency_ema_us = 0.9 * self._latency_ema_us + 0.1 * timings["total"]
        return res

    @property
    def mean_latency_ms(self) -> float:
        return self._latency_ema_us * 1e-3

    @staticmethod
    def feature_dim() -> int:
        return FEATURE_DIM

"""MuJoCo state extraction and metric depth acquisition.

The state manager reads joint / end-effector state from an ``MjData`` and
renders a depth image from a fixed camera with :class:`mujoco.Renderer`. The
raw OpenGL depth buffer (``d_b`` in ``[0, 1]``, 0 at the near plane) is read
back directly and converted to metric depth with

    d = z_near * z_far / (z_far - d_b * (z_far - z_near))

where ``z_near = model.vis.map.znear * model.stat.extent`` and
``z_far = model.vis.map.zfar * model.stat.extent`` (MuJoCo expresses the
clipping planes relative to the model extent).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np


@dataclass
class RobotStateSnapshot:
    """Immutable copy of the simulation state needed to reproduce kinematics."""

    sim_time: float
    wall_time: float          # time.time() at capture (matches ARIA epoch stamps)
    mono_time: float          # time.perf_counter() at capture
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    mocap_quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


class MuJoCoStateManager:
    """Reads robot state and renders linearized depth from a MuJoCo simulation.

    Args:
        model: compiled ``MjModel``.
        data: ``MjData`` to read from. When used inside the Layer-2 worker this
            is a private *mirror* of the physics data, updated through
            :meth:`apply_snapshot`, so rendering never races ``mj_step``.
        camera_name: name of the fixed depth camera.
        config: optional full layer configuration (``robot`` / ``depth`` /
            ``camera.intrinsics`` sections are used).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 camera_name: str = "overhead_cam",
                 config: Optional[dict] = None) -> None:
        cfg = config or {}
        robot_cfg = cfg.get("robot", {})
        depth_cfg = cfg.get("depth", {})
        intr = cfg.get("camera", {}).get("intrinsics", {})

        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")

        ee_name = robot_cfg.get("ee_body", "ee_link")
        base_name = robot_cfg.get("base_body", "link0")
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)
        if self.ee_body_id < 0:
            raise ValueError(f"End-effector body '{ee_name}' not found in model")
        if self.base_body_id < 0:
            raise ValueError(f"Base body '{base_name}' not found in model")

        joint_names = robot_cfg.get("arm_joints") or []
        self.arm_qpos_adr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in joint_names], dtype=np.int64)
        self.arm_dof_adr = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in joint_names], dtype=np.int64)

        self.width = int(intr.get("width", 320))
        self.height = int(intr.get("height", 240))

        # Clipping planes in metres.
        extent = float(model.stat.extent)
        self.z_near = float(model.vis.map.znear) * extent
        self.z_far = float(model.vis.map.zfar) * extent

        self.cache_interval = float(depth_cfg.get("cache_interval_ms", 5.0)) * 1e-3

        # Lazily created, thread-affine render resources.
        self._renderer: Optional[mujoco.Renderer] = None
        self._renderer_thread: Optional[int] = None
        self._mjv_camera = mujoco.MjvCamera()
        self._mjv_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._mjv_camera.fixedcamid = self.camera_id
        self._raw_buf = np.empty((self.height, self.width), dtype=np.float32)
        self._tmp_buf = np.empty((self.height, self.width), dtype=np.float32)
        self._depth_buf = np.empty((self.height, self.width), dtype=np.float32)
        self._low_level = True

        # Depth cache
        self._cache_time = -np.inf
        self._cache_sim_time = None
        self.cache_hits = 0
        self.renders = 0

    # ------------------------------------------------------------------ #
    # Robot state
    # ------------------------------------------------------------------ #
    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Joint positions/velocities with a monotonic timestamp."""
        d = self.data
        out = {
            "qpos": d.qpos.copy(),
            "qvel": d.qvel.copy(),
            "timestamp": time.monotonic(),
            "sim_time": float(d.time),
        }
        if self.arm_qpos_adr.size:
            out["arm_qpos"] = d.qpos[self.arm_qpos_adr].copy()
            out["arm_qvel"] = d.qvel[self.arm_dof_adr].copy()
        return out

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """End-effector world position and quaternion ``[w, x, y, z]``."""
        return (self.data.xpos[self.ee_body_id].copy(),
                self.data.xquat[self.ee_body_id].copy())

    def get_base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_body_id].copy()

    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Camera world position and 3x3 rotation (MuJoCo camera convention)."""
        return (self.data.cam_xpos[self.camera_id].copy(),
                self.data.cam_xmat[self.camera_id].reshape(3, 3).copy())

    def snapshot(self) -> RobotStateSnapshot:
        """Capture a consistent copy of the state (call from the physics thread)."""
        d = self.data
        return RobotStateSnapshot(
            sim_time=float(d.time), wall_time=time.time(), mono_time=time.perf_counter(),
            qpos=d.qpos.copy(), qvel=d.qvel.copy(),
            mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy())

    def apply_snapshot(self, snap: RobotStateSnapshot) -> None:
        """Load a snapshot into ``self.data`` and recompute kinematics/cameras."""
        d = self.data
        d.qpos[:] = snap.qpos
        d.qvel[:] = snap.qvel
        if self.model.nmocap:
            d.mocap_pos[:] = snap.mocap_pos
            d.mocap_quat[:] = snap.mocap_quat
        d.time = snap.sim_time
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)

    # ------------------------------------------------------------------ #
    # Depth
    # ------------------------------------------------------------------ #
    def _ensure_renderer(self) -> mujoco.Renderer:
        """Create the offscreen renderer in the *calling* thread.

        OpenGL contexts are bound to the thread that made them current, so the
        renderer is (re)created whenever a different thread asks for depth.
        """
        tid = threading.get_ident()
        if self._renderer is not None and self._renderer_thread == tid:
            return self._renderer
        self.close()
        r = mujoco.Renderer(self.model, height=self.height, width=self.width)
        # Segmentation mode renders flat-shaded geometry only (no skybox/labels),
        # which is what MuJoCo itself uses for accurate depth.
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        r.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        ctx = getattr(r, "_mjr_context", None)
        self._low_level = ctx is not None and hasattr(r, "_rect")
        if self._low_level:
            # Standard (non-reversed) depth read-back: 0 = near, 1 = far.
            ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZERONEAR
        else:  # pragma: no cover - fallback for alternate renderer backends
            r.enable_depth_rendering()
        self._renderer = r
        self._renderer_thread = tid
        self._cache_time = -np.inf
        return r

    def linearize_depth(self, raw: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert non-linear depth-buffer values ``d_b in [0, 1]`` to metres."""
        n, f = self.z_near, self.z_far
        raw = np.asarray(raw, dtype=np.float32)
        if out is None:
            out = np.empty_like(raw, dtype=np.float32)
        tmp = self._tmp_buf if raw.shape == self._tmp_buf.shape else np.empty_like(out)
        np.multiply(raw, np.float32(-(f - n)), out=tmp)
        tmp += np.float32(f)
        np.divide(np.float32(n * f), tmp, out=out)
        return out

    def render_raw_depth(self) -> np.ndarray:
        """Render and return the raw (non-linear) depth buffer, top row first."""
        r = self._ensure_renderer()
        r.update_scene(self.data, camera=self._mjv_camera)
        gl = getattr(r, "_gl_context", None)
        if gl is not None:
            gl.make_current()
        mujoco.mjr_render(r._rect, r.scene, r._mjr_context)
        mujoco.mjr_readPixels(None, self._raw_buf, r._rect, r._mjr_context)
        if gl is not None:  # GL origin is bottom-left
            return self._raw_buf[::-1]
        return self._raw_buf  # pragma: no cover

    def get_depth_image(self, force_refresh: bool = False) -> np.ndarray:
        """Metric depth image ``(H, W)`` float32 in metres.

        Results are cached: if the previous render happened less than
        ``cache_interval_ms`` ago, the cached image is returned (unless
        ``force_refresh``). The returned array is an internal buffer; copy it
        if it must outlive the next call.
        """
        now = time.perf_counter()
        if (not force_refresh and self._renderer is not None
                and self._renderer_thread == threading.get_ident()
                and (now - self._cache_time) < self.cache_interval):
            self.cache_hits += 1
            return self._depth_buf

        r = self._ensure_renderer()
        if self._low_level:
            raw = self.render_raw_depth()
            self.linearize_depth(raw, out=self._depth_buf)
        else:  # pragma: no cover
            r.update_scene(self.data, camera=self._mjv_camera)
            self._depth_buf[:] = r.render()
        self._cache_time = time.perf_counter()
        self._cache_sim_time = float(self.data.time)
        self.renders += 1
        return self._depth_buf

    def invalidate_cache(self) -> None:
        self._cache_time = -np.inf

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self._renderer = None
        self._renderer_thread = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


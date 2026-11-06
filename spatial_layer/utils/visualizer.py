"""MuJoCo viewer visual markers (EE vector, target point, reachability).

Uses ``mujoco.mjv_initGeom`` and ``mujoco.mjv_connector`` to populate an
``MjvScene`` (typically ``viewer.user_scn``) with 3D overlay markers:
- Semi-transparent red sphere at the ground-truth object position.
- Blue sphere at the estimated 3D object position from Layer 2.
- Green/yellow sphere at the computed grasp target point.
- Connector line from End-Effector to estimated target colored by reachability.
- Optional workspace boundary ring or secondary object markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import mujoco
import numpy as np

if TYPE_CHECKING:
    from ..core import SpatialResult

_IDENTITY_MAT = np.eye(3, dtype=np.float64).flatten()


def add_sphere(scn: mujoco.MjvScene, pos: Sequence[float], radius: float,
               rgba: Sequence[float]) -> bool:
    """Append a visual sphere geom to ``scn`` if capacity allows."""
    if scn.ngeom >= scn.maxgeom:
        return False
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=_IDENTITY_MAT,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return True


def add_line(scn: mujoco.MjvScene, p_from: Sequence[float], p_to: Sequence[float],
             width: float, rgba: Sequence[float]) -> bool:
    """Append a connector capsule/line between ``p_from`` and ``p_to``."""
    if scn.ngeom >= scn.maxgeom:
        return False
    p1 = np.asarray(p_from, dtype=np.float64)
    p2 = np.asarray(p_to, dtype=np.float64)
    if np.linalg.norm(p2 - p1) < 1e-6:
        return False
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=_IDENTITY_MAT,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        width=float(width),
        from_=p1,
        to=p2,
    )
    scn.ngeom += 1
    return True


class SpatialVisualizer:
    """Draws Layer-2 spatial reasoning overlays into a MuJoCo ``MjvScene``."""

    def __init__(self, gt_radius: float = 0.032, est_radius: float = 0.018,
                 line_width: float = 0.004) -> None:
        self.gt_radius = gt_radius
        self.est_radius = est_radius
        self.line_width = line_width

    def update_scene(self, scn: mujoco.MjvScene, result: Optional[SpatialResult],
                     gt_pos: Optional[Sequence[float]] = None) -> int:
        """Clear and populate ``scn`` with markers; returns number of geoms added."""
        scn.ngeom = 0
        # 1. Ground truth object position (red halo sphere)
        if gt_pos is not None:
            add_sphere(scn, gt_pos, self.gt_radius, [1.0, 0.15, 0.15, 0.35])

        if result is None or not result.valid or result.p_obj_world is None:
            return scn.ngeom

        # 2. Estimated 3D object position from Layer 2 (blue sphere)
        add_sphere(scn, result.p_obj_world, self.est_radius, [0.15, 0.45, 1.0, 0.90])

        # 3. Grasp target point & EE-to-target connector colored by reachability
        reachable = bool(result.analysis and result.analysis.reachable)
        line_rgba = [0.15, 0.95, 0.30, 0.85] if reachable else [1.0, 0.55, 0.10, 0.85]

        if result.p_ee_world is not None:
            add_line(scn, result.p_ee_world, result.p_obj_world,
                     self.line_width, line_rgba)

        if result.analysis is not None:
            add_sphere(scn, result.analysis.grasp_point, self.est_radius * 0.65, line_rgba)

        # 4. Secondary tracked objects (cyan markers)
        if result.secondary_world is not None and len(result.secondary_world):
            for p_sec in result.secondary_world:
                if np.all(np.isfinite(p_sec)):
                    add_sphere(scn, p_sec, self.est_radius * 0.75, [0.0, 0.85, 0.85, 0.75])

        return scn.ngeom


def format_console_status(result: SpatialResult, gt_pos: Optional[np.ndarray] = None) -> str:
    """Format a compact status string showing reachability, error, and 30-D vector."""
    if not result.valid or result.p_obj_world is None or result.analysis is None:
        return f"[Frame {result.frame_id:04d}] status={result.status} (no valid target)"

    a = result.analysis
    err_mm = (float(np.linalg.norm(result.p_obj_world - gt_pos) * 1e3)
              if gt_pos is not None else float("nan"))
    vec_str = np.array2string(result.features, precision=3, suppress_small=True,
                              max_line_width=120, separator=", ")
    total_ms = result.timings_us.get("total", 0.0) * 1e-3
    return (
        f"[Frame {result.frame_id:04d} | Track #{result.track_id}] "
        f"Reachable={a.reachable} (WS={a.workspace_valid}, IK={a.ik_feasible} "
        f"in {a.ik.iterations} it/{a.ik.time_ms:.2f}ms, Limits={a.ik.limits_ok}) | "
        f"d_ee={a.d_euclidean:.3f}m | Diff={a.difficulty:.3f} | "
        f"3D_err={err_mm:.2f}mm | Latency={total_ms:.2f}ms\n"
        f"  p_t^spatial (30D): {vec_str}"
    )

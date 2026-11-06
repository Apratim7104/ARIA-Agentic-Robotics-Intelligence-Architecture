"""Relative end-effector metrics, workspace check, fast IK and grasp difficulty."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np


@dataclass
class IKResult:
    q: np.ndarray               # arm joint solution (len = n_arm_joints)
    success: bool               # converged within tolerance
    error: float                # final Cartesian position error (m)
    iterations: int
    time_ms: float
    limits_ok: bool             # q respects model.jnt_range


@dataclass
class SpatialAnalysis:
    d_euclidean: float
    d_horizontal: float
    d_vertical: float
    base_distance: float
    workspace_valid: bool
    ik: IKResult
    collision_free: bool
    grasp_offset: np.ndarray
    grasp_point: np.ndarray
    difficulty: float
    flags: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    @property
    def ik_feasible(self) -> bool:
        return self.ik.success

    @property
    def reachable(self) -> bool:
        return bool(self.workspace_valid and self.ik.success and self.ik.limits_ok)


class SpatialAnalyzer:
    """Computes spatial/kinematic relationships between an object and the EE.

    Workspace validity is evaluated as the radial distance of the object from
    the robot *base* origin lying within ``[reach_min, reach_max]``.

    IK is a position-only damped least-squares (Levenberg-Marquardt) solver on
    the arm chain, run on a private ``MjData`` (never touches the live sim).
    """

    def __init__(self, model: mujoco.MjModel, config: dict) -> None:
        self.model = model
        robot = config.get("robot", {})
        ws = config.get("workspace", {})
        ik = config.get("ik", {})
        norm = config.get("normalization", {})
        grasp = config.get("grasp", {})

        self.reach_min = float(ws.get("reach_min", 0.15))
        self.reach_max = float(ws.get("reach_max", 0.85))
        self.table_height = float(ws.get("table_height", 0.0))
        self.collision_clearance = float(ws.get("collision_clearance", 0.06))
        self.dist_scale = float(norm.get("dist_scale", 1.5))
        self.default_grasp_offset = np.asarray(grasp.get("default_offset", [0.0, 0.0, 0.02]),
                                               dtype=np.float64)
        self.class_grasp_offsets: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.float64) for k, v in (grasp.get("class_offsets") or {}).items()}

        self.max_iterations = int(ik.get("max_iterations", 5))
        self.time_budget = float(ik.get("time_budget_ms", 0.8)) * 1e-3
        self.damping = float(ik.get("damping", 0.05))
        self.tolerance = float(ik.get("tolerance", 0.01))
        self.max_step = float(ik.get("max_step", 0.5))

        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            robot.get("ee_body", "ee_link"))
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                              robot.get("base_body", "link0"))
        names = robot.get("arm_joints") or [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        if any(j < 0 for j in jids):
            raise ValueError(f"Unknown arm joint in {names}")
        self.joint_ids = np.array(jids)
        self.qpos_adr = model.jnt_qposadr[self.joint_ids].astype(np.int64)
        self.dof_adr = model.jnt_dofadr[self.joint_ids].astype(np.int64)
        limited = model.jnt_limited[self.joint_ids].astype(bool)
        rng = model.jnt_range[self.joint_ids].copy()
        self.q_lo = np.where(limited, rng[:, 0], -np.inf)
        self.q_hi = np.where(limited, rng[:, 1], np.inf)
        self.n = len(self.joint_ids)

        self._ik_data = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._eye3 = np.eye(3)
        self._last_solution: Optional[np.ndarray] = None
        self._last_target: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_distances(p_obj: Sequence[float], p_ee: Sequence[float]) -> Tuple[float, float, float]:
        diff = np.asarray(p_obj, dtype=np.float64) - np.asarray(p_ee, dtype=np.float64)
        d_euc = float(np.sqrt(diff @ diff))
        d_h = float(np.hypot(diff[0], diff[1]))
        d_v = float(abs(diff[2]))
        return d_euc, d_h, d_v

    def check_workspace(self, p_obj: Sequence[float], base_pos: Sequence[float]) -> Tuple[bool, float]:
        r = float(np.linalg.norm(np.asarray(p_obj) - np.asarray(base_pos)))
        return (self.reach_min <= r <= self.reach_max), r

    def limits_ok(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(np.all(q >= self.q_lo - tol) and np.all(q <= self.q_hi + tol))

    # ------------------------------------------------------------------ #
    # Inverse kinematics
    # ------------------------------------------------------------------ #
    def _fk_pos(self, d: mujoco.MjData) -> np.ndarray:
        mujoco.mj_kinematics(self.model, d)
        return d.xpos[self.ee_body_id]

    def solve_ik(self, target: Sequence[float], qpos_full: np.ndarray,
                 q_seed: Optional[np.ndarray] = None) -> IKResult:
        """Damped least-squares IK towards ``target`` (EE body origin).

        ``dq = J^T (J J^T + lambda^2 I)^{-1} e``, clamped to ``max_step`` and
        projected onto the joint limits each iteration. Adaptive damping: lambda
        shrinks when the error decreases and grows (with step rejection)
        otherwise. Stops after ``max_iterations`` or the time budget.
        """
        t0 = time.perf_counter()
        m, d = self.model, self._ik_data
        target = np.asarray(target, dtype=np.float64)
        d.qpos[:] = qpos_full
        q = (qpos_full[self.qpos_adr] if q_seed is None else np.asarray(q_seed)).astype(np.float64)
        q = np.clip(q, self.q_lo, self.q_hi)
        d.qpos[self.qpos_adr] = q
        pos = self._fk_pos(d)
        err_vec = target - pos
        err = float(np.linalg.norm(err_vec))
        lam = self.damping
        it = 0
        jac = self._jacp
        while it < self.max_iterations and err > self.tolerance * 0.1:
            if time.perf_counter() - t0 > self.time_budget:
                break
            it += 1
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacBody(m, d, jac, None, self.ee_body_id)
            J = jac[:, self.dof_adr]
            A = J @ J.T + (lam * lam) * self._eye3
            dq = J.T @ np.linalg.solve(A, err_vec)
            nrm = np.linalg.norm(dq)
            if nrm > self.max_step:
                dq *= self.max_step / nrm
            q_new = np.clip(q + dq, self.q_lo, self.q_hi)
            d.qpos[self.qpos_adr] = q_new
            pos = self._fk_pos(d)
            new_err_vec = target - pos
            new_err = float(np.linalg.norm(new_err_vec))
            if new_err < err:
                q, err_vec, err = q_new, new_err_vec, new_err
                lam = max(lam * 0.5, 1e-4)
            else:                                   # reject step, increase damping
                lam *= 4.0
                d.qpos[self.qpos_adr] = q
                self._fk_pos(d)
        elapsed = (time.perf_counter() - t0) * 1e3
        return IKResult(q=q.copy(), success=err <= self.tolerance, error=err, iterations=it,
                        time_ms=elapsed, limits_ok=self.limits_ok(q))

    def solve_ik_warm(self, target: Sequence[float], qpos_full: np.ndarray) -> IKResult:
        """IK warm-started from the previous solution when the target is nearby."""
        target = np.asarray(target, dtype=np.float64)
        seed = None
        if (self._last_solution is not None and self._last_target is not None
                and np.linalg.norm(target - self._last_target) < 0.1):
            seed = self._last_solution
        res = self.solve_ik(target, qpos_full, seed)
        if not res.success and seed is not None:
            remaining = self.time_budget - res.time_ms * 1e-3
            if remaining > 0.25 * self.time_budget:      # retry from the current pose
                res2 = self.solve_ik(target, qpos_full, None)
                res2.time_ms += res.time_ms
                if res2.error < res.error:
                    res = res2
        if res.success:
            self._last_solution, self._last_target = res.q.copy(), target.copy()
        return res

    # ------------------------------------------------------------------ #
    # Grasp, collision, difficulty
    # ------------------------------------------------------------------ #
    def grasp_target(self, p_obj: Sequence[float], cls: str = "") -> Tuple[np.ndarray, np.ndarray]:
        offset = self.class_grasp_offsets.get(cls, self.default_grasp_offset).copy()
        return offset, np.asarray(p_obj, dtype=np.float64) + offset

    def collision_free(self, grasp_point: np.ndarray,
                       obstacles: Optional[np.ndarray] = None) -> bool:
        """Grasp point above the table and clear of other detected objects."""
        if grasp_point[2] < self.table_height:
            return False
        if obstacles is not None and len(obstacles):
            obs = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
            obs = obs[np.all(np.isfinite(obs), axis=1)]
            if obs.size and np.min(np.linalg.norm(obs - grasp_point, axis=1)) < self.collision_clearance:
                return False
        return True

    def difficulty(self, d_euclidean: float, confidence: float, ik_feasible: bool) -> float:
        d_norm = min(max(d_euclidean / self.dist_scale, 0.0), 1.0)
        score = 0.4 * d_norm + 0.3 * (1.0 - confidence) + 0.3 * (1.0 - float(ik_feasible))
        return float(np.clip(score, 0.0, 1.0))

    def analyze(self, p_obj: np.ndarray, p_ee: np.ndarray, base_pos: np.ndarray,
                qpos_full: np.ndarray, confidence: float, cls: str = "",
                obstacles: Optional[np.ndarray] = None) -> SpatialAnalysis:
        d_euc, d_h, d_v = self.compute_distances(p_obj, p_ee)
        ws_ok, r_base = self.check_workspace(p_obj, base_pos)
        ik = self.solve_ik_warm(p_obj, qpos_full)
        offset, grasp_point = self.grasp_target(p_obj, cls)
        coll_ok = self.collision_free(grasp_point, obstacles)
        diff = self.difficulty(d_euc, confidence, ik.success)
        flags = np.array([ws_ok, ik.success, coll_ok, ik.limits_ok], dtype=np.float32)
        return SpatialAnalysis(d_euclidean=d_euc, d_horizontal=d_h, d_vertical=d_v,
                               base_distance=r_base, workspace_valid=ws_ok, ik=ik,
                               collision_free=coll_ok, grasp_offset=offset,
                               grasp_point=grasp_point, difficulty=diff, flags=flags)


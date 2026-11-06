"""2D pixel -> 3D camera -> 3D world transformations.

Pinhole deprojection (OpenCV optical frame: +x right, +y down, +z forward)::

    x_c = (u - c_x) * d / f_x,   y_c = (v - c_y) * d / f_y,   z_c = d
    p_w = R_cw @ p_c + t_cw

Pixel coordinates are *continuous*: the centre of pixel ``(i, j)`` (column,
row) is at ``(i + 0.5, j + 0.5)``, consistent with OpenGL rasterization, so the
depth sample for a continuous point ``(u, v)`` lives at ``[floor(v), floor(u)]``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .utils.math_utils import (intrinsic_matrix, invert_se3, make_se3,
                               mujoco_rot_to_opencv, quat_to_mat,
                               undistort_normalized)


class CoordinateTransformer:
    """Camera model with intrinsics, distortion and camera->world extrinsics.

    Args:
        intrinsics: dict with ``fx, fy, cx, cy, width, height`` and optional
            ``distortion = [k1, k2, p1, p2, k3]``.
        extrinsics: dict with ``pos`` (camera origin in world) and ``quat``
            (``[w, x, y, z]`` camera->world rotation, OpenCV optical frame).
            Alternatively ``rot`` may be a 3x3 matrix.
    """

    def __init__(self, intrinsics: dict, extrinsics: dict) -> None:
        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics["cx"])
        self.cy = float(intrinsics["cy"])
        self.width = int(intrinsics.get("width", 2 * self.cx))
        self.height = int(intrinsics.get("height", 2 * self.cy))
        self.distortion = np.asarray(intrinsics.get("distortion", [0.0] * 5), dtype=np.float64)
        self.has_distortion = bool(np.any(self.distortion != 0.0))
        self.K = intrinsic_matrix(self.fx, self.fy, self.cx, self.cy)
        self.set_extrinsics(extrinsics)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def set_extrinsics(self, extrinsics: dict) -> None:
        if "rot" in extrinsics:
            R = np.asarray(extrinsics["rot"], dtype=np.float64).reshape(3, 3)
        else:
            R = quat_to_mat(extrinsics["quat"])
        # Re-orthonormalise (config quaternions are rounded, e.g. 0.7071).
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        self.R_cw = R
        self.t_cw = np.asarray(extrinsics["pos"], dtype=np.float64).reshape(3)
        self.T_cw = make_se3(self.R_cw, self.t_cw)
        self.T_wc = invert_se3(self.T_cw)
        self.R_wc = self.T_wc[:3, :3]
        self.t_wc = self.T_wc[:3, 3]

    @classmethod
    def from_mujoco_camera(cls, intrinsics: dict, cam_xpos: np.ndarray,
                           cam_xmat: np.ndarray) -> "CoordinateTransformer":
        """Build extrinsics directly from a MuJoCo camera pose."""
        return cls(intrinsics, {"pos": np.asarray(cam_xpos),
                                "rot": mujoco_rot_to_opencv(np.asarray(cam_xmat).reshape(3, 3))})

    @property
    def optical_axis_world(self) -> np.ndarray:
        """Unit vector of the camera +z (viewing) direction in world frame."""
        return self.R_cw[:, 2].copy()

    # ------------------------------------------------------------------ #
    # Single point
    # ------------------------------------------------------------------ #
    @staticmethod
    def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
        return 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])

    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """Deproject continuous pixel ``(u, v)`` at planar depth ``d`` (metres)."""
        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy
        if self.has_distortion:
            xn, yn = undistort_normalized(xn, yn, self.distortion)
        return np.array([xn * depth, yn * depth, depth], dtype=np.float64)

    def camera_to_world(self, p_c: Sequence[float]) -> np.ndarray:
        return self.R_cw @ np.asarray(p_c, dtype=np.float64) + self.t_cw

    def world_to_camera(self, p_w: Sequence[float]) -> np.ndarray:
        return self.R_wc @ np.asarray(p_w, dtype=np.float64) + self.t_wc

    def pixel_to_world(self, u: float, v: float, depth: float) -> np.ndarray:
        return self.camera_to_world(self.pixel_to_camera(u, v, depth))

    def world_to_pixel(self, p_w: Sequence[float]) -> Tuple[np.ndarray, float]:
        """Project a world point; returns ``(uv, depth)`` (distortion applied)."""
        p_c = self.world_to_camera(p_w)
        uv = self._project_cam(p_c[None, :])[0]
        return uv, float(p_c[2])

    # ------------------------------------------------------------------ #
    # Vectorized batch helpers
    # ------------------------------------------------------------------ #
    def batch_pixel_to_camera(self, uv: np.ndarray, depths: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        d = np.asarray(depths, dtype=np.float64).reshape(-1)
        xn = (uv[:, 0] - self.cx) / self.fx
        yn = (uv[:, 1] - self.cy) / self.fy
        if self.has_distortion:
            xn, yn = self._undistort_batch(xn, yn)
        return np.stack([xn * d, yn * d, d], axis=1)

    def batch_camera_to_world(self, P_c: np.ndarray) -> np.ndarray:
        P = np.asarray(P_c, dtype=np.float64).reshape(-1, 3)
        return P @ self.R_cw.T + self.t_cw

    def batch_pixel_to_world(self, uv: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Deproject ``N`` pixels with per-pixel depths -> ``(N, 3)`` world points."""
        return self.batch_camera_to_world(self.batch_pixel_to_camera(uv, depths))

    def batch_world_to_pixel(self, P_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.asarray(P_w, dtype=np.float64).reshape(-1, 3)
        P_c = P @ self.R_wc.T + self.t_wc
        return self._project_cam(P_c), P_c[:, 2]

    def _project_cam(self, P_c: np.ndarray) -> np.ndarray:
        z = np.where(np.abs(P_c[:, 2]) < 1e-12, 1e-12, P_c[:, 2])
        xn = P_c[:, 0] / z
        yn = P_c[:, 1] / z
        if self.has_distortion:
            k1, k2, p1, p2, k3 = self.distortion[:5]
            r2 = xn * xn + yn * yn
            radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
            yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
            xn, yn = xd, yd
        return np.stack([self.fx * xn + self.cx, self.fy * yn + self.cy], axis=1)

    def _undistort_batch(self, xd: np.ndarray, yd: np.ndarray,
                         iterations: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        k1, k2, p1, p2, k3 = self.distortion[:5]
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return x, y

    # ------------------------------------------------------------------ #
    # Depth sampling & object-centroid correction
    # ------------------------------------------------------------------ #
    @staticmethod
    def sample_depth(depth: np.ndarray, u: float, v: float, radius: int = 1,
                     min_depth: float = 0.05, max_depth: float = 10.0) -> Optional[float]:
        """Robust (median) planar depth in a ``(2r+1)^2`` window around ``(u, v)``."""
        h, w = depth.shape
        col, row = int(np.floor(u)), int(np.floor(v))
        if not (0 <= col < w and 0 <= row < h):
            return None
        r0, r1 = max(row - radius, 0), min(row + radius + 1, h)
        c0, c1 = max(col - radius, 0), min(col + radius + 1, w)
        patch = depth[r0:r1, c0:c1].ravel()
        patch = patch[(patch > min_depth) & (patch < max_depth)]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def batch_sample_depth(self, depth: np.ndarray, uv: np.ndarray, radius: int = 1,
                           min_depth: float = 0.05, max_depth: float = 10.0) -> np.ndarray:
        """Vectorized-friendly wrapper; NaN where no valid depth is found."""
        out = np.full(len(uv), np.nan)
        for i, (u, v) in enumerate(np.asarray(uv).reshape(-1, 2)):
            d = self.sample_depth(depth, u, v, radius, min_depth, max_depth)
            if d is not None:
                out[i] = d
        return out

    def centroid_depth_offset(self, size: Sequence[float],
                              obj_rot: Optional[np.ndarray] = None) -> float:
        """Half-thickness of a box along the optical axis.

        The depth camera observes the object's *surface*; adding this offset to
        the sampled depth moves the estimate to the volumetric centroid.
        """
        size = np.asarray(size, dtype=np.float64).reshape(3)
        if not np.any(size):
            return 0.0
        R = np.eye(3) if obj_rot is None else np.asarray(obj_rot).reshape(3, 3)
        axis_in_obj = R.T @ self.optical_axis_world
        return float(0.5 * np.sum(np.abs(axis_in_obj) * size))


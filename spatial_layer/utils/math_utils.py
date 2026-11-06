"""Quaternion math, SE(3) transforms and pinhole projection helpers.

Conventions
-----------
* Quaternions are scalar-first ``[w, x, y, z]`` (MuJoCo convention).
* Rotation matrices map vectors from the child frame to the parent frame,
  i.e. ``p_parent = R @ p_child + t``.
* Camera optical frame follows OpenCV: +x right, +y down, +z forward.
  MuJoCo cameras look along their local -z with +y up, so
  ``R_mujoco = R_opencv @ diag(1, -1, -1)``.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

# Flip between the OpenCV optical frame and the MuJoCo/OpenGL camera frame.
CV_TO_GL = np.diag([1.0, -1.0, -1.0])

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Quaternions
# --------------------------------------------------------------------------- #
def quat_normalize(q: Sequence[float]) -> np.ndarray:
    """Return a unit quaternion with non-negative scalar part."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return -q if q[0] < 0.0 else q


def quat_multiply(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """Hamilton product ``q1 * q2``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to unit quaternion ``[w, x, y, z]`` (Shepperd)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q``."""
    return quat_to_mat(q) @ np.asarray(v, dtype=np.float64)


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #
def make_se3(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def se3_from_pos_quat(pos: Sequence[float], quat: Sequence[float]) -> np.ndarray:
    return make_se3(quat_to_mat(quat), pos)


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_point(T: np.ndarray, p: Sequence[float]) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p, dtype=np.float64) + T[:3, 3]


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Transform an (N, 3) array of points."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #
def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project_points(K: np.ndarray, P_cam: np.ndarray) -> np.ndarray:
    """Project (N, 3) optical-frame points to (N, 2) continuous pixel coords."""
    P = np.asarray(P_cam, dtype=np.float64).reshape(-1, 3)
    z = np.where(np.abs(P[:, 2]) < _EPS, _EPS, P[:, 2])
    u = K[0, 0] * P[:, 0] / z + K[0, 2]
    v = K[1, 1] * P[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def fovy_from_intrinsics(fy: float, height: int) -> float:
    """Vertical field of view (degrees) for a pinhole camera."""
    return float(np.degrees(2.0 * np.arctan(0.5 * height / fy)))


def opencv_rot_to_mujoco(R_cv: np.ndarray) -> np.ndarray:
    """Camera->world rotation in OpenCV optical frame to MuJoCo camera frame."""
    return np.asarray(R_cv).reshape(3, 3) @ CV_TO_GL


def mujoco_rot_to_opencv(R_gl: np.ndarray) -> np.ndarray:
    return np.asarray(R_gl).reshape(3, 3) @ CV_TO_GL


def undistort_normalized(xd: float, yd: float, dist: Iterable[float],
                         iterations: int = 8) -> Tuple[float, float]:
    """Invert Brown-Conrady distortion for a normalized image point.

    ``dist = [k1, k2, p1, p2, k3]`` (OpenCV ordering). Uses fixed-point
    iteration, which converges quickly for moderate distortion.
    """
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0 and k3 == 0.0:
        return xd, yd
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


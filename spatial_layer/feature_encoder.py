"""Assembly and normalization of the 30-dim spatial feature vector p_t^spatial."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .utils.math_utils import quat_normalize

FEATURE_DIM = 30

#: Packing order of the feature vector (slices into the 30-dim array).
FEATURE_LAYOUT: Dict[str, slice] = {
    "p_obj": slice(0, 3),          # object position (world) / pos_scale
    "p_ee": slice(3, 6),           # end-effector position (world) / pos_scale
    "obj_quat": slice(6, 10),      # object orientation [w, x, y, z]
    "distances": slice(10, 13),    # [euclidean, horizontal, vertical] / dist_scale
    "grasp_offset": slice(13, 16), # grasp offset / grasp_scale
    "flags": slice(16, 20),        # [workspace_valid, ik_feasible, collision_free, limits_ok]
    "difficulty": slice(20, 21),
    "confidence": slice(21, 22),
    "padding": slice(22, 30),
}

FLAG_NAMES = ("workspace_valid", "ik_feasible", "collision_free", "limits_ok")


class FeatureEncoder:
    """Packs Layer-2 outputs into a float32 vector clamped to ``[-1, 1]``."""

    def __init__(self, config: dict) -> None:
        norm = config.get("normalization", config)
        self.pos_scale = float(norm.get("pos_scale", 2.0))
        self.dist_scale = float(norm.get("dist_scale", 1.5))
        self.grasp_scale = float(norm.get("grasp_scale", 0.5))
        self.dim = int(config.get("pipeline", {}).get("feature_dim", FEATURE_DIM))
        if self.dim != FEATURE_DIM:
            raise ValueError(f"feature_dim must be {FEATURE_DIM}, got {self.dim}")
        self._buf = np.zeros(FEATURE_DIM, dtype=np.float32)

    def encode(self, p_obj: Sequence[float], p_ee: Sequence[float],
               obj_quat: Optional[Sequence[float]], distances: Sequence[float],
               grasp_offset: Sequence[float], flags: Sequence[float],
               difficulty: float, confidence: float) -> np.ndarray:
        """Return a *new* float32 array of length 30 in ``[-1, 1]``."""
        v = np.zeros(FEATURE_DIM, dtype=np.float64)
        v[0:3] = np.asarray(p_obj, dtype=np.float64) / self.pos_scale
        v[3:6] = np.asarray(p_ee, dtype=np.float64) / self.pos_scale
        v[6:10] = quat_normalize([1.0, 0.0, 0.0, 0.0] if obj_quat is None else obj_quat)
        v[10:13] = np.asarray(distances, dtype=np.float64) / self.dist_scale
        v[13:16] = np.asarray(grasp_offset, dtype=np.float64) / self.grasp_scale
        v[16:20] = [1.0 if f else 0.0 for f in flags]
        v[20] = difficulty
        v[21] = confidence
        # [22:30] zero padding (already zero)
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(v, -1.0, 1.0).astype(np.float32)

    def encode_analysis(self, p_obj: np.ndarray, p_ee: np.ndarray, analysis,
                        confidence: float, obj_quat: Optional[np.ndarray] = None) -> np.ndarray:
        """Convenience wrapper around :meth:`encode` for a ``SpatialAnalysis``."""
        return self.encode(p_obj, p_ee, obj_quat,
                           (analysis.d_euclidean, analysis.d_horizontal, analysis.d_vertical),
                           analysis.grasp_offset, analysis.flags, analysis.difficulty,
                           confidence)

    def empty(self, p_ee: Optional[Sequence[float]] = None) -> np.ndarray:
        """Feature vector for frames with no valid target (all flags 0)."""
        v = np.zeros(FEATURE_DIM, dtype=np.float32)
        if p_ee is not None:
            v[3:6] = np.clip(np.asarray(p_ee) / self.pos_scale, -1.0, 1.0)
        v[6] = 1.0
        v[20] = 1.0           # maximal difficulty: nothing to grasp
        return v

    def decode(self, vec: np.ndarray) -> Dict[str, np.ndarray]:
        """Inverse scaling (for debugging / logging). Clamped values stay clamped."""
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected shape ({FEATURE_DIM},), got {vec.shape}")
        return {
            "p_obj": vec[FEATURE_LAYOUT["p_obj"]] * self.pos_scale,
            "p_ee": vec[FEATURE_LAYOUT["p_ee"]] * self.pos_scale,
            "obj_quat": vec[FEATURE_LAYOUT["obj_quat"]].copy(),
            "distances": vec[FEATURE_LAYOUT["distances"]] * self.dist_scale,
            "grasp_offset": vec[FEATURE_LAYOUT["grasp_offset"]] * self.grasp_scale,
            "flags": vec[FEATURE_LAYOUT["flags"]] > 0.5,
            "difficulty": float(vec[20]),
            "confidence": float(vec[21]),
        }


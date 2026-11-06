"""
ccdm/interfaces/spatial_grounding.py

System-Agnostic Spatial Scene Grounding & Relative Distance Module for CCDM v3.0.
Computes dynamic relative distance metrics between system telemetry (Robot Arm End-Effector)
and target objects/obstacles derived from RGB-D camera depth scenes or 3D poses.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class SystemAgnosticSpatialGrounding:
    """
    Computes dynamic spatial metrics between a robotic system's reference frame
    (e.g., Robotic Arm End-Effector) and environmental targets/obstacles.
    Supports both direct 3D pose inputs and RGB-D camera depth back-projections.
    """

    def __init__(self, default_dt: float = 0.033):
        """
        Args:
            default_dt: Frame delta time in seconds (default ~30 FPS = 0.033s).
        """
        self.default_dt = default_dt
        self.prev_distance: Optional[float] = None

    def estimate_3d_relative_kinematics(
        self,
        system_origin_3d: np.ndarray,
        target_center_3d: np.ndarray,
        system_velocity_3d: Optional[np.ndarray] = None,
        target_velocity_3d: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes 3D relative offset vector, Euclidean scalar distance, range rate, and heading.

        Args:
            system_origin_3d: [x, y, z] of End-Effector / System Reference in meters.
            target_center_3d: [x, y, z] of Target Object or Obstacle in meters.
            system_velocity_3d: Optional [vx, vy, vz] linear velocity of system (m/s).
            target_velocity_3d: Optional [vx, vy, vz] linear velocity of target (m/s).
            dt: Time delta between observations (seconds).

        Returns:
            Dictionary containing relative displacement, distance, range rate, and direction.
        """
        step_dt = dt if dt is not None else self.default_dt
        p_sys = np.asarray(system_origin_3d, dtype=np.float32)
        p_tgt = np.asarray(target_center_3d, dtype=np.float32)

        # 1. Relative 3D Displacement Vector (Target - System)
        delta_vec = p_tgt - p_sys  # [dx, dy, dz] in meters
        euclidean_dist = float(np.linalg.norm(delta_vec))

        # 2. Normalized Direction Unit Vector
        unit_dir = delta_vec / (euclidean_dist + 1e-8)

        # 3. Dynamic Range Rate (Approach Speed: d(d_rel)/dt)
        if system_velocity_3d is not None and target_velocity_3d is not None:
            v_sys = np.asarray(system_velocity_3d, dtype=np.float32)
            v_tgt = np.asarray(target_velocity_3d, dtype=np.float32)
            relative_vel = v_tgt - v_sys
            # Projected velocity along line-of-sight
            range_rate = float(np.dot(relative_vel, unit_dir))
        elif self.prev_distance is not None:
            # Finite difference fallback
            range_rate = float((euclidean_dist - self.prev_distance) / max(step_dt, 1e-5))
        else:
            range_rate = 0.0

        self.prev_distance = euclidean_dist

        return {
            "delta_vector": delta_vec,           # [dx, dy, dz] in SI meters
            "euclidean_distance": euclidean_dist, # d_rel scalar (m)
            "range_rate": range_rate,             # dd_rel/dt (m/s, negative = closing in)
            "unit_direction": unit_dir            # Normalized 3D heading
        }

    def estimate_from_depth_camera(
        self,
        depth_map: np.ndarray,
        target_bbox_2d: Tuple[int, int, int, int],
        camera_intrinsics: Dict[str, float],
        T_system_to_camera: Optional[np.ndarray] = None,
        system_origin_3d: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Estimates 3D target center and relative distance directly from RGB-D Camera Scene Depth.
        """
        xmin, ymin, xmax, ymax = target_bbox_2d
        h_img, w_img = depth_map.shape[:2]

        # Clamp bounding box coordinates
        xmin, xmax = max(0, int(xmin)), min(w_img, int(xmax))
        ymin, ymax = max(0, int(ymin)), min(h_img, int(ymax))

        # Extract depth patch within target bounding box
        depth_patch = depth_map[ymin:ymax, xmin:xmax]
        valid_depths = depth_patch[(depth_patch > 0.05) & (~np.isnan(depth_patch))]

        if len(valid_depths) == 0:
            Z_cam = 1.0
        else:
            # Median filtering for robust depth estimation against specularity/noise
            Z_cam = float(np.median(valid_depths))

        # Pinhole Camera Back-Projection: Pixel (u_c, v_c) -> Camera 3D Point (X_c, Y_c, Z_c)
        u_center = (xmin + xmax) / 2.0
        v_center = (ymin + ymax) / 2.0
        fx, fy = float(camera_intrinsics["fx"]), float(camera_intrinsics["fy"])
        cx, cy = float(camera_intrinsics["cx"]), float(camera_intrinsics["cy"])

        X_cam = (u_center - cx) * Z_cam / fx
        Y_cam = (v_center - cy) * Z_cam / fy
        P_camera = np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float32)

        # Transform 3D point to System Reference Frame (End-Effector)
        if T_system_to_camera is not None:
            P_system = T_system_to_camera @ P_camera
            target_3d_sys = P_system[:3]
        else:
            target_3d_sys = P_camera[:3]

        # Calculate relative distance metrics
        sys_origin = system_origin_3d if system_origin_3d is not None else np.zeros(3, dtype=np.float32)
        metrics = self.estimate_3d_relative_kinematics(sys_origin, target_3d_sys)
        metrics["target_camera_frame_3d"] = P_camera[:3]

        return metrics

    def compute_ccdm_physical_context(
        self,
        system_origin_3d: np.ndarray,
        target_center_3d: np.ndarray,
        object_mass: float = 0.5,
        friction_coeff: float = 0.3
    ) -> np.ndarray:
        """
        Formats relative distance metrics directly into CCDM's Physical Context vector p_t [3-d].
        Per CCDM Spec: p_t = [mass (kg), friction_coeff, relative_distance (m)]
        """
        kinematics = self.estimate_3d_relative_kinematics(system_origin_3d, target_center_3d)
        d_rel = kinematics["euclidean_distance"]

        # Construct 3-dimensional physical context vector p_t
        p_t = np.array([object_mass, friction_coeff, d_rel], dtype=np.float32)
        return p_t


"""
ccdm/interfaces/controller_adaptation.py

Core bridge computing:
1. Physical Constraints for the NMPC Control Layer
2. Dynamic Cost Function Weights for the NMPC Controller

Takes 3 inputs:
- context: from Multi-Agent Pipeline (operational parameters, task plan, fragility, priorities)
- states: from Spatial Reasoning Layer (end-effector state, target location, relative distances, initial object pose)
- physical_scene: from Simulation Camera (depth map, table height, obstacle proximity, mass/friction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .dynamics_interface import CCDMInferenceInterface
from .spatial_grounding import SystemAgnosticSpatialGrounding
from .context_interface import ContextInterface


@dataclass
class PhysicalConstraints:
    """Dynamic physical and chance constraints for the downstream NMPC controller."""
    # Cartesian velocity limits [vx_max, vy_max, vz_max] (m/s) and yaw rate (rad/s)
    velocity_max: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.4, 1.0], dtype=np.float32))
    velocity_min: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.5, -0.4, -1.0], dtype=np.float32))

    # Cartesian acceleration limits [ax_max, ay_max, az_max] (m/s^2)
    acceleration_max: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0], dtype=np.float32))
    acceleration_min: np.ndarray = field(default_factory=lambda: np.array([-2.0, -2.0, -2.0], dtype=np.float32))

    # Cartesian workspace boundaries [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    workspace_bounds: np.ndarray = field(default_factory=lambda: np.array([[0.15, 0.85], [-0.55, 0.55], [0.41, 1.10]], dtype=np.float32))

    # Uncertainty-derived 2-sigma chance constraint margins (m)
    chance_margins: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.01], dtype=np.float32))

    # Table surface safety barrier height (z >= z_table_barrier)
    table_barrier_z: float = 0.415

    # Safe obstacle clearance distance (m)
    safe_obstacle_clearance: float = 0.06

    # Terminal positioning tolerance (m)
    terminal_tolerance: float = 0.015

    # Distance to grasp at current step (m)
    distance_to_grasp: float = 0.0

    # Current task phase
    phase: str = "approach"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "velocity_max": self.velocity_max.tolist(),
            "velocity_min": self.velocity_min.tolist(),
            "acceleration_max": self.acceleration_max.tolist(),
            "acceleration_min": self.acceleration_min.tolist(),
            "workspace_bounds": self.workspace_bounds.tolist(),
            "chance_margins": self.chance_margins.tolist(),
            "table_barrier_z": float(self.table_barrier_z),
            "safe_obstacle_clearance": float(self.safe_obstacle_clearance),
            "terminal_tolerance": float(self.terminal_tolerance),
            "distance_to_grasp": float(self.distance_to_grasp),
            "phase": self.phase
        }


@dataclass
class DynamicNMPCWeights:
    """Dynamic cost function weights for the downstream NMPC controller."""
    # State tracking weights Q = diag(Q_pos, Q_vel)
    Q_pos: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, 60.0], dtype=np.float32))
    Q_vel: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=np.float32))

    # Control input effort penalty R (size 4: vx_cmd, vy_cmd, vz_cmd, yaw_rate)
    R_action: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32))

    # Slew rate / control jerk penalty S (smoothness / anti-jerk)
    S_slew: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0, 1.0], dtype=np.float32))

    # Terminal state weight multiplier Q_terminal = Q_pos * terminal_scale
    Q_terminal: np.ndarray = field(default_factory=lambda: np.array([250.0, 250.0, 300.0], dtype=np.float32))

    # Soft constraint relaxation slack penalty (rho)
    W_slack: float = 10000.0

    # Distance scaling factor currently applied
    distance_scale: float = 1.0

    # Fragility damping factor currently applied
    fragility_damping: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Q_pos": self.Q_pos.tolist(),
            "Q_vel": self.Q_vel.tolist(),
            "R_action": self.R_action.tolist(),
            "S_slew": self.S_slew.tolist(),
            "Q_terminal": self.Q_terminal.tolist(),
            "W_slack": float(self.W_slack),
            "distance_scale": float(self.distance_scale),
            "fragility_damping": float(self.fragility_damping)
        }


class CCDMControllerAdaptation:
    """
    Computes dynamic physical constraints and dynamic NMPC weights conditioned on:
    1. Multi-Agent Context (fragility, speed_priority, safety_priority, task_code, crowd)
    2. Spatial Reasoning Layer States (relative distances, grasp offset, end-effector state, initial locations)
    3. Simulator Camera Physical Scene (depth map, table height, obstacle proximity, mass/friction)
    """

    def __init__(
        self,
        ccdm_inference: Optional[CCDMInferenceInterface] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.ccdm_inference = ccdm_inference or CCDMInferenceInterface()
        self.spatial_grounding = SystemAgnosticSpatialGrounding()
        self.context_parser = ContextInterface()
        self.config = config or {}

        # Default kinematic nominal limits (Franka Emika Panda operational space)
        self.v_nominal_max = np.array([0.45, 0.45, 0.35, 1.0], dtype=np.float32)
        self.a_nominal_max = np.array([2.5, 2.5, 2.0], dtype=np.float32)

        # Baseline NMPC weights
        self.base_Q_pos = np.array([40.0, 40.0, 50.0], dtype=np.float32)
        self.base_Q_vel = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.base_R_action = np.array([0.8, 0.8, 0.8, 0.4], dtype=np.float32)
        self.base_S_slew = np.array([1.5, 1.5, 1.5, 0.8], dtype=np.float32)

    def compute_adaptation(
        self,
        context_input: Union[Dict[str, Any], np.ndarray],
        spatial_state: Dict[str, Any],
        physical_scene: Dict[str, Any]
    ) -> Tuple[PhysicalConstraints, DynamicNMPCWeights, Dict[str, Any]]:
        """
        Main adaptation entrypoint consuming the 3 inputs.

        Args:
            context_input: Context dictionary or 5-d vector from Multi-Agent Pipeline.
            spatial_state: State dictionary from Spatial Reasoning Layer containing:
                - "p_ee": end-effector Cartesian pose [x, y, z]
                - "v_ee": end-effector Cartesian linear velocity [vx, vy, vz]
                - "p_obj": target object position [x, y, z]
                - "p_obj_0": initial object position [x, y, z]
                - "d_rel": relative Euclidean distance from EE grasp to target object
                - "delta_vec": relative displacement vector (p_obj - p_ee)
                - "range_rate": relative approach speed dd_rel/dt
                - "features_30d": 30-dim spatial feature vector p_t^spatial
            physical_scene: Scene dictionary from camera/simulator containing:
                - "depth_map": 2D depth image
                - "table_height": table height (meters, default 0.40)
                - "mass": payload/object mass (kg, default 0.1)
                - "friction": surface friction coefficient (mu, default 0.4)
                - "obstacles": optional list of obstacle centroids

        Returns:
            Tuple of (PhysicalConstraints, DynamicNMPCWeights, ccdm_dynamics_dict)
        """
        # -------------------------------------------------------------
        # 1. PARSE & VALIDATE THE 3 INPUTS
        # -------------------------------------------------------------
        # Input 1: Multi-Agent Context
        c_vec = self.context_parser.encode_context(context_input)
        fragility = float(c_vec[0])
        speed_pri = float(c_vec[1])
        safety_pri = float(c_vec[2])
        task_code = float(c_vec[3])
        crowd = float(c_vec[4])

        # Input 2: Spatial Reasoning States
        p_ee = np.asarray(spatial_state.get("p_ee", [0.4, 0.0, 0.6]), dtype=np.float32)
        v_ee = np.asarray(spatial_state.get("v_ee", [0.0, 0.0, 0.0]), dtype=np.float32)
        p_obj = np.asarray(spatial_state.get("p_obj", [0.5, 0.0, 0.425]), dtype=np.float32)
        p_obj_0 = np.asarray(spatial_state.get("p_obj_0", p_obj), dtype=np.float32)

        # Distance from grasp to target
        if "d_rel" in spatial_state:
            d_rel = float(spatial_state["d_rel"])
        else:
            d_rel = float(np.linalg.norm(p_obj - p_ee))

        range_rate = float(spatial_state.get("range_rate", 0.0))

        # Input 3: Physical Environment Scene
        table_height = float(physical_scene.get("table_height", 0.40))
        object_mass = float(physical_scene.get("mass", 0.10))
        friction = float(physical_scene.get("friction", 0.40))

        # -------------------------------------------------------------
        # 2. RUN CCDM RESIDUAL & UNCERTAINTY INFERENCE
        # -------------------------------------------------------------
        # 6-d state: [x, y, z, vx, vy, vz]
        x_curr = np.concatenate([p_ee, v_ee]).astype(np.float32)
        # 4-d nominal action: commanded velocity
        u_nom = np.concatenate([v_ee, [0.0]]).astype(np.float32)
        # 3-d physical context: [mass, friction, d_rel]
        p_phys = np.array([object_mass, friction, d_rel], dtype=np.float32)

        ccdm_out = self.ccdm_inference.predict(
            state=x_curr,
            action=u_nom,
            context=c_vec,
            physical_context=p_phys
        )
        residual = ccdm_out["residual"]
        uncertainty = ccdm_out["uncertainty"]

        # Conformal 2-sigma chance bounds for safety (95.4% coverage)
        sigma_pos = np.abs(uncertainty[:3])
        chance_margin = np.clip(2.0 * sigma_pos, 0.005, 0.05)

        # -------------------------------------------------------------
        # 3. COMPUTE DYNAMIC PHYSICAL CONSTRAINTS FOR NMPC
        # -------------------------------------------------------------
        # Determine operational phase based on continuous relative distance
        if d_rel > 0.20:
            phase = "approach"
        elif d_rel > 0.06:
            phase = "pre_grasp"
        elif d_rel > 0.015:
            phase = "terminal_alignment"
        else:
            phase = "in_contact_grasp"

        # A. Velocity Constraints
        # Velocity corridor decelerates smoothly as the grasp point is approached:
        # v_cap(d) = v_nominal * tanh(d_rel / 0.15)
        dist_velocity_factor = float(np.clip(math.tanh(d_rel / 0.12) + 0.15, 0.15, 1.0))

        # Fragility dampens maximum velocity to avoid inertial displacement
        fragility_vel_factor = 1.0 - 0.50 * fragility

        # Speed priority scales up velocity headroom
        speed_factor = 0.70 + 0.50 * speed_pri

        v_limit_scale = dist_velocity_factor * fragility_vel_factor * speed_factor
        v_max = self.v_nominal_max * v_limit_scale
        v_min = -v_max

        # B. Acceleration Constraints (Slip & Impact Prevention)
        # Maximum un-slipped acceleration: a_max <= mu * g
        g = 9.81
        slip_a_max = max(0.8, friction * g * (1.0 - 0.40 * fragility))
        a_max = np.minimum(self.a_nominal_max, slip_a_max * (0.8 + 0.4 * speed_pri))
        a_min = -a_max

        # C. Barrier & Clearance Bounds
        # Table barrier: z_ee must stay above table top + safety margin + uncertainty
        table_safety_margin = 0.015 + chance_margin[2]
        table_barrier_z = table_height + table_safety_margin

        # Obstacle clearance
        safe_clearance = 0.05 + float(np.linalg.norm(chance_margin)) + 0.03 * safety_pri

        # Terminal tolerance region around target
        terminal_tol = max(0.008, 0.02 * (1.0 - 0.5 * safety_pri))

        constraints = PhysicalConstraints(
            velocity_max=v_max.astype(np.float32),
            velocity_min=v_min.astype(np.float32),
            acceleration_max=a_max.astype(np.float32),
            acceleration_min=a_min.astype(np.float32),
            workspace_bounds=np.array([[0.15, 0.85], [-0.55, 0.55], [table_barrier_z, 1.10]], dtype=np.float32),
            chance_margins=chance_margin.astype(np.float32),
            table_barrier_z=float(table_barrier_z),
            safe_obstacle_clearance=float(safe_clearance),
            terminal_tolerance=float(terminal_tol),
            distance_to_grasp=float(d_rel),
            phase=phase
        )

        # -------------------------------------------------------------
        # 4. COMPUTE DYNAMIC NMPC COST FUNCTION WEIGHTS
        # -------------------------------------------------------------
        # Position Weight Q_pos:
        # Boosted dramatically in terminal phase (d_rel -> 0) to ensure exact positioning
        # Q_pos(d) = Q_base * (1 + 4.0 * exp(-d_rel / 0.08))
        q_dist_boost = 1.0 + 4.0 * math.exp(-d_rel / 0.08)
        q_pos_scale = q_dist_boost * (0.8 + 0.4 * safety_pri)
        Q_pos = self.base_Q_pos * q_pos_scale

        # Velocity Damping Weight Q_vel:
        # Damping increases when fragile or when closing in on the object to eliminate overshoot
        q_damp_boost = 1.0 + 3.0 * fragility + 2.0 * math.exp(-d_rel / 0.05)
        Q_vel = self.base_Q_vel * q_damp_boost

        # Action Effort Penalty R_action:
        # Low action penalty when high speed is demanded, higher when gentle/fragile
        r_scale = (1.0 + 1.5 * fragility) / (0.6 + 0.6 * speed_pri)
        R_action = self.base_R_action * r_scale

        # Control Slew Rate / Anti-Jerk Penalty S_slew:
        # High jerk penalty when handling fragile payloads
        s_slew_scale = 1.0 + 6.0 * (fragility ** 2) + 2.0 * safety_pri
        S_slew = self.base_S_slew * s_slew_scale

        # Terminal Weight Q_terminal:
        # Multiplied heavily when within reach
        terminal_scale = 5.0 + 10.0 * math.exp(-d_rel / 0.10)
        Q_terminal = Q_pos * terminal_scale

        # Slack Variable Penalty W_slack:
        # Heavily penalized with high safety priority
        W_slack = 5000.0 * (1.0 + 3.0 * safety_pri)

        weights = DynamicNMPCWeights(
            Q_pos=Q_pos.astype(np.float32),
            Q_vel=Q_vel.astype(np.float32),
            R_action=R_action.astype(np.float32),
            S_slew=S_slew.astype(np.float32),
            Q_terminal=Q_terminal.astype(np.float32),
            W_slack=float(W_slack),
            distance_scale=float(q_dist_boost),
            fragility_damping=float(q_damp_boost)
        )

        diagnostics = {
            "residual_mean": residual,
            "uncertainty_sigma": uncertainty,
            "calibrated_margins": chance_margin,
            "phase": phase,
            "distance_to_grasp": d_rel,
            "initial_object_pos": p_obj_0,
            "current_object_pos": p_obj,
            "operational_context": c_vec
        }

        return constraints, weights, diagnostics


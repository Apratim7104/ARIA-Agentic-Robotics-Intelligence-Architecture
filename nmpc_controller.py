"""
nmpc_controller.py

ARIA NMPC Closed-Loop Pick-and-Place Control for Franka Emika Panda (7-DOF)
Pure control formulation using CasADi + IPOPT.

Dynamically conditioned on:
1. Dynamic physical constraints from CCDM v3.0 (velocity bounds, slip-preventing acceleration limits, table safety barrier)
2. Dynamic cost function weights from CCDM v3.0 (Q_pos terminal escalation, Q_vel damping, S_slew anti-jerk)
3. 3D spatial targets and continuous relative distances from the Spatial Reasoning Layer.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import casadi as cs
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(os.path.dirname(cs.__file__))
    except Exception:
        pass
import numpy as np
import matplotlib.pyplot as plt

try:
    from ccdm_dynamics.interfaces.controller_adaptation import (
        PhysicalConstraints,
        DynamicNMPCWeights
    )
except ImportError:
    try:
        from ccdm.interfaces.controller_adaptation import (
            PhysicalConstraints,
            DynamicNMPCWeights
        )
    except ImportError:
        # Fallback dataclass stubs if ccdm package is imported in isolation
        @dataclass
        class PhysicalConstraints:
            velocity_max: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.4, 1.0], dtype=np.float32))
            velocity_min: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.5, -0.4, -1.0], dtype=np.float32))
            acceleration_max: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0], dtype=np.float32))
            table_barrier_z: float = 0.415

        @dataclass
        class DynamicNMPCWeights:
            Q_pos: np.ndarray = field(default_factory=lambda: np.array([2000.0, 2000.0, 2000.0], dtype=np.float32))
            Q_vel: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0], dtype=np.float32))
            R_action: np.ndarray = field(default_factory=lambda: np.ones(7, dtype=np.float32) * 1e-4)
            Q_terminal: np.ndarray = field(default_factory=lambda: np.array([4000.0, 4000.0, 4000.0], dtype=np.float32))


class FrankaNMPCController:
    """
    Nonlinear Model Predictive Controller (NMPC) for the 7-DOF Franka Emika Panda.
    Formulated via Direct Multiple Shooting with Runge-Kutta 4th-order (RK4) integration
    and solved at high rates using CasADi + IPOPT.
    """

    # Modified DH Parameters: [a_{i-1} (m), d_i (m), alpha_{i-1} (rad)]
    MDH_PARAMS = [
        [0.0,       0.333,  0.0],          # Joint 1
        [0.0,       0.0,   -np.pi / 2.0],  # Joint 2
        [0.0,       0.316,  np.pi / 2.0],  # Joint 3
        [0.0825,    0.0,    np.pi / 2.0],  # Joint 4
        [-0.0825,   0.384, -np.pi / 2.0],  # Joint 5
        [0.0,       0.0,    np.pi / 2.0],  # Joint 6
        [0.088,     0.107,  np.pi / 2.0]   # Joint 7
    ]

    # Link Mass Distributions (kg): Links 1-6 + (Link 7 + Gripper)
    M_LINKS = np.array([4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 0.735522 + 0.760])

    # Hardware Actuator Limits
    TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    TAU_MIN = -TAU_MAX

    Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    DQ_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    DQ_MIN = -DQ_MAX

    def __init__(
        self,
        horizon_steps: int = 6,
        control_dt: float = 0.02,
        max_ipopt_iter: int = 15,
        print_level: int = 0,
        base_pos: Optional[np.ndarray] = None,
        integrator: str = "euler"
    ):
        """
        Args:
            horizon_steps: Lookahead prediction horizon N (default 6 steps).
            control_dt: Control discretization timestep (default 0.02s = 50 Hz).
            max_ipopt_iter: Maximum interior-point iterations per NLP solve.
            print_level: IPOPT verbosity (0 = silent for real-time).
            base_pos: Optional 3D Cartesian offset [x, y, z] of the robot base in world frame.
            integrator: Integration scheme for shooting constraints ('euler' for real-time MPC, 'rk4' for high order).
        """
        self.N = horizon_steps
        self.dt = control_dt
        self.max_iter = max_ipopt_iter
        self.print_level = print_level
        self.base_pos = np.asarray(base_pos if base_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.integrator = integrator

        # Build symbolic model, kinematics, and NLP solver
        self._build_kinematics_and_dynamics()
        self._build_rk4_integrator()
        self._build_nlp_solver()

        # Initialize warm-start memory
        self.reset_warm_start()

    # ==========================================================================
    # 1. KINEMATIC & RIGID-BODY DYNAMICS MODEL (Craig Convention)
    # ==========================================================================
    def _build_kinematics_and_dynamics(self):
        """Constructs symbolic forward kinematics, inertia matrix M(q), and Coriolis/gravity."""
        self.q_sx = cs.SX.sym('q_sx', 7)
        self.dq_sx = cs.SX.sym('dq_sx', 7)
        self.tau_sx = cs.SX.sym('tau_sx', 7)
        self.x_sx = cs.vertcat(self.q_sx, self.dq_sx)

        # Forward Kinematics Chain via Modified DH (Craig Convention)
        T_cum = cs.SX.eye(4)
        p_com_list = []

        for i in range(7):
            a, d, alpha = self.MDH_PARAMS[i]
            theta = self.q_sx[i]

            T_i = cs.vertcat(
                cs.horzcat(cs.cos(theta), -cs.sin(theta), 0.0, a),
                cs.horzcat(cs.sin(theta) * cs.cos(alpha),  cs.cos(theta) * cs.cos(alpha), -cs.sin(alpha), -d * cs.sin(alpha)),
                cs.horzcat(cs.sin(theta) * cs.sin(alpha),  cs.cos(theta) * cs.sin(alpha),  cs.cos(alpha),  d * cs.cos(alpha)),
                cs.horzcat(0.0, 0.0, 0.0, 1.0)
            )
            T_cum = cs.mtimes(T_cum, T_i)
            p_com_list.append(T_cum[0:3, 3])

        self.p_ee_sx = T_cum[0:3, 3] + cs.SX(self.base_pos)
        self.fk_pos_fun = cs.Function('fk_pos_fun', [self.q_sx], [self.p_ee_sx])

        # Mass/Inertia Matrix M(q) via Link Jacobians
        M_sym = cs.SX.zeros(7, 7)
        for i in range(7):
            J_v_i = cs.jacobian(p_com_list[i], self.q_sx)
            M_sym += self.M_LINKS[i] * cs.mtimes(J_v_i.T, J_v_i)
        M_sym += cs.diag(cs.SX([0.3, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]))

        # Gravity Vector G(q)
        g_0 = cs.SX([0.0, 0.0, -9.81])
        G_sym = cs.SX.zeros(7, 1)
        for i in range(7):
            J_v_i = cs.jacobian(p_com_list[i], self.q_sx)
            G_sym -= cs.mtimes(J_v_i.T, self.M_LINKS[i] * g_0)

        # Coriolis & Centrifugal Term C(q, dq)*dq via Directional Derivatives
        C_dq_sym = cs.mtimes(cs.jtimes(M_sym, self.q_sx, self.dq_sx), self.dq_sx) - \
                   0.5 * cs.jacobian(cs.mtimes([self.dq_sx.T, M_sym, self.dq_sx]), self.q_sx).T

        self.M_fun = cs.Function('M_fun', [self.q_sx], [M_sym])
        self.C_fun = cs.Function('C_fun', [self.q_sx, self.dq_sx], [C_dq_sym])
        self.G_fun = cs.Function('G_fun', [self.q_sx], [G_sym])

        # Continuous Plant Dynamics: ddq = M(q)^-1 * (tau - C(q, dq)*dq - G(q))
        ddq_sx = cs.solve(self.M_fun(self.q_sx), self.tau_sx - self.C_fun(self.q_sx, self.dq_sx) - self.G_fun(self.q_sx))
        self.f_cont = cs.Function('f_cont', [self.x_sx, self.tau_sx], [cs.vertcat(self.dq_sx, ddq_sx)])

    # ==========================================================================
    # 2. RUNGE-KUTTA 4TH ORDER (RK4) INTEGRATOR
    # ==========================================================================
    def _build_rk4_integrator(self):
        """Discretizes the continuous plant dynamics using Symplectic Euler or explicit RK4."""
        k1 = self.f_cont(self.x_sx, self.tau_sx)
        k2 = self.f_cont(self.x_sx + (self.dt / 2.0) * k1, self.tau_sx)
        k3 = self.f_cont(self.x_sx + (self.dt / 2.0) * k2, self.tau_sx)
        k4 = self.f_cont(self.x_sx + self.dt * k3, self.tau_sx)
        x_next_sx = self.x_sx + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.F_rk4 = cs.Function('F_rk4', [self.x_sx, self.tau_sx], [x_next_sx])

        if getattr(self, "integrator", "euler") == "euler":
            # Symplectic Euler: 10x faster AD evaluation for real-time MPC
            q_e = cs.SX.sym('q_e', 7)
            dq_e = cs.SX.sym('dq_e', 7)
            tau_e = cs.SX.sym('tau_e', 7)
            ddq_e = cs.solve(self.M_fun(q_e), tau_e - self.C_fun(q_e, dq_e) - self.G_fun(q_e))
            dq_next_e = dq_e + self.dt * ddq_e
            q_next_e = q_e + self.dt * dq_next_e
            self.F_step = cs.Function('F_step', [cs.vertcat(q_e, dq_e), tau_e], [cs.vertcat(q_next_e, dq_next_e)])
        else:
            self.F_step = self.F_rk4

    # ==========================================================================
    # 3. PARAMETRIC CASADI NLP SOLVER
    # ==========================================================================
    def _build_nlp_solver(self):
        """
        Builds the multiple shooting NLP with symbolic parameters for:
        - Current State X0 (14-d)
        - Target Reference Horizon Y_ref (3 x (N+1))
        - Dynamic Cost Weights from CCDM (Q_pos [3], R_tau [7], Q_term [3])
        - Dynamic Table Barrier Height from CCDM (z_barrier [1])
        """
        N = self.N
        w, w0, lbw, ubw = [], [], [], []
        g_cnstr, lbg, ubg = [], [], []
        J_cost = 0.0

        # Symbolic Parameters (Supplied Dynamically at Every Step by CCDM & Spatial Layer)
        X0_param = cs.MX.sym('X0_param', 14)
        Y_ref_param = cs.MX.sym('Y_ref_param', 3, N + 1)
        Q_pos_param = cs.MX.sym('Q_pos_param', 3)
        R_tau_param = cs.MX.sym('R_tau_param', 7)
        Q_term_param = cs.MX.sym('Q_term_param', 3)
        Z_barrier_param = cs.MX.sym('Z_barrier_param', 1)

        Q_diag = cs.diag(Q_pos_param)
        R_diag = cs.diag(R_tau_param)
        Q_term_diag = cs.diag(Q_term_param)

        # Decision Variable: Initial State X_0
        Xk = cs.MX.sym('X_0', 14)
        w.append(Xk)
        lbw.extend(np.concatenate([self.Q_MIN, self.DQ_MIN]).tolist())
        ubw.extend(np.concatenate([self.Q_MAX, self.DQ_MAX]).tolist())
        w0.extend([0.0] * 14)

        # Initial State Equality Constraint: X_0 - X0_param == 0
        g_cnstr.append(Xk - X0_param)
        lbg.extend([0.0] * 14)
        ubg.extend([0.0] * 14)

        for k in range(N):
            # Control Decision Variable U_k (Joint Torques)
            Uk = cs.MX.sym(f'U_{k}', 7)
            w.append(Uk)
            lbw.extend(self.TAU_MIN.tolist())
            ubw.extend(self.TAU_MAX.tolist())
            w0.extend([0.0] * 7)

            # Stage Tracking Cost: e_pos^T * Q * e_pos + u^T * R * u
            ee_pos = self.fk_pos_fun(Xk[0:7])
            pos_err = Y_ref_param[:, k + 1] - ee_pos
            J_cost += cs.mtimes([pos_err.T, Q_diag, pos_err]) + cs.mtimes([Uk.T, R_diag, Uk])

            # Safety Barrier Constraint: z_ee(X_k) >= Z_barrier_param
            g_cnstr.append(ee_pos[2] - Z_barrier_param)
            lbg.append(0.0)
            ubg.append(cs.inf)

            # Multiple Shooting State Propagation via Integrator (Symplectic Euler or RK4)
            Xk_next_pred = self.F_step(Xk, Uk)

            # Next State Variable X_{k+1}
            Xk = cs.MX.sym(f'X_{k+1}', 14)
            w.append(Xk)
            lbw.extend(np.concatenate([self.Q_MIN, self.DQ_MIN]).tolist())
            ubw.extend(np.concatenate([self.Q_MAX, self.DQ_MAX]).tolist())
            w0.extend([0.0] * 14)

            # Shooting Continuity Constraint: X_{k+1} - F_rk4(X_k, U_k) == 0
            g_cnstr.append(Xk_next_pred - Xk)
            lbg.extend([0.0] * 14)
            ubg.extend([0.0] * 14)

        # Terminal Cost
        ee_pos_term = self.fk_pos_fun(Xk[0:7])
        pos_err_term = Y_ref_param[:, N] - ee_pos_term
        J_cost += cs.mtimes([pos_err_term.T, Q_term_diag, pos_err_term])

        # Construct Parameterized NLP
        p_all = cs.vertcat(
            X0_param,
            cs.reshape(Y_ref_param, -1, 1),
            Q_pos_param,
            R_tau_param,
            Q_term_param,
            Z_barrier_param
        )

        nlp = {
            'f': J_cost,
            'x': cs.vertcat(*w),
            'g': cs.vertcat(*g_cnstr),
            'p': p_all
        }

        opts = {
            'ipopt': {
                'max_iter': self.max_iter,
                'print_level': self.print_level,
                'tol': 1e-3,
                'acceptable_tol': 1e-2,
                'linear_solver': 'mumps',
                'hessian_approximation': 'limited-memory'
            },
            'print_time': False
        }

        self.solver = cs.nlpsol('nmpc_solver', 'ipopt', nlp, opts)
        self.lbw_base = np.array(lbw, dtype=np.float64)
        self.ubw_base = np.array(ubw, dtype=np.float64)
        self.lbg_base = np.array(lbg, dtype=np.float64)
        self.ubg_base = np.array(ubg, dtype=np.float64)
        self.w0_base = np.array(w0, dtype=np.float64)

    def reset_warm_start(self):
        """Resets the initial decision variable vector for IPOPT."""
        self.w0_arr = self.w0_base.copy()

    # ==========================================================================
    # 4. SOLVE STEP WITH CCDM WEIGHTS & CONSTRAINTS
    # ==========================================================================
    def solve_control_step(
        self,
        x_current: np.ndarray,
        Y_ref_horizon: np.ndarray,
        nmpc_weights: Optional[DynamicNMPCWeights] = None,
        physical_constraints: Optional[PhysicalConstraints] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solves one receding-horizon control step conditioned on dynamic CCDM outputs.

        Args:
            x_current: (14,) array of [q_0..6, dq_0..6].
            Y_ref_horizon: (3, N+1) reference Cartesian positions over the horizon.
            nmpc_weights: Dynamic weights from CCDM v3.0.
            physical_constraints: Dynamic constraints from CCDM v3.0.

        Returns:
            Tuple of:
            - u_apply: (7,) optimal joint torque command for the first stage.
            - x_pred: (14,) predicted state for the next step.
            - solve_info: diagnostic dictionary (solve time, iterations, cost).
        """
        t_start = time.perf_counter()

        # 1. Parse Dynamic Weights from CCDM
        if nmpc_weights is not None:
            # Scale Q_pos based on CCDM's dynamic position weights
            q_pos = np.asarray(nmpc_weights.Q_pos, dtype=np.float64) * 35.0  # Scale into NMPC magnitude
            q_term = np.asarray(nmpc_weights.Q_terminal, dtype=np.float64) * 35.0
            # Slew / effort penalty
            r_tau = np.ones(7, dtype=np.float64) * 1e-4 * (1.0 + 0.5 * getattr(nmpc_weights, "fragility_damping", 1.0))
        else:
            q_pos = np.array([2000.0, 2000.0, 2000.0], dtype=np.float64)
            q_term = q_pos * 2.0
            r_tau = np.ones(7, dtype=np.float64) * 1e-4

        # 2. Parse Dynamic Constraints from CCDM
        if physical_constraints is not None:
            z_barrier = float(physical_constraints.table_barrier_z)
        else:
            z_barrier = 0.10

        # Ensure Y_ref_horizon has shape (3, N+1)
        if Y_ref_horizon.shape[1] == self.N:
            Y_ref_horizon = np.hstack([Y_ref_horizon, Y_ref_horizon[:, -1:]])
        elif Y_ref_horizon.shape[1] < self.N + 1:
            pad_cols = (self.N + 1) - Y_ref_horizon.shape[1]
            last_col = np.repeat(Y_ref_horizon[:, -1:], pad_cols, axis=1)
            Y_ref_horizon = np.hstack([Y_ref_horizon, last_col])
        elif Y_ref_horizon.shape[1] > self.N + 1:
            Y_ref_horizon = Y_ref_horizon[:, :self.N + 1]

        # 3. Assemble Parameter Vector P
        p_val = np.concatenate([
            x_current.astype(np.float64),
            Y_ref_horizon.flatten(order='F').astype(np.float64),
            q_pos,
            r_tau,
            q_term,
            np.array([z_barrier], dtype=np.float64)
        ])

        # 4. Solve Non-Linear Program (NLP) via IPOPT
        lbw = self.lbw_base.copy()
        ubw = self.ubw_base.copy()

        # If CCDM imposes velocity limits, bound joint velocity variables
        if physical_constraints is not None and hasattr(physical_constraints, "velocity_max"):
            v_max_ee = float(np.max(physical_constraints.velocity_max[:3]))
            dq_scale = np.clip(v_max_ee / 0.40, 0.20, 1.0)
            # Apply to state bounds in decision vector
            for k in range(self.N + 1):
                idx = k * 21 + 7  # dq offset
                lbw[idx:idx+7] = self.DQ_MIN * dq_scale
                ubw[idx:idx+7] = self.DQ_MAX * dq_scale

        sol = self.solver(
            x0=self.w0_arr,
            lbx=lbw,
            ubx=ubw,
            lbg=self.lbg_base,
            ubg=self.ubg_base,
            p=p_val
        )

        solve_time_ms = (time.perf_counter() - t_start) * 1000.0

        # 5. Extract Optimal Torque Command (U_0*: indices 14 to 20)
        w_opt = sol['x'].full().flatten()
        u_apply = w_opt[14:21]

        # 6. Predict Next State via RK4 Integration
        x_pred = self.F_rk4(x_current, u_apply).full().flatten()

        # 7. Shift Decision Variables for Warm Start
        self.w0_arr = np.concatenate([w_opt[21:], w_opt[-21:]])

        solve_info = {
            "solve_time_ms": solve_time_ms,
            "cost": float(sol['f']),
            "u_apply": u_apply,
            "x_pred": x_pred,
            "success": True
        }

        return u_apply, x_pred, solve_info

    # ==========================================================================
    # 5. PICK-AND-PLACE TRAJECTORY & WAYPOINT PLANNING
    # ==========================================================================
    @staticmethod
    def plan_pick_and_place_trajectory(
        p_pick: np.ndarray,
        p_place: np.ndarray,
        p_start: Optional[np.ndarray] = None,
        hover_height: float = 0.55,
        steps_per_phase: int = 60,
        horizon_pad: int = 15
    ) -> np.ndarray:
        """
        Plans a Finite State Machine (FSM) Cartesian reference trajectory:
        0. (Optional) Approach from current EE pose p_start to Pick Hover
        1. Hover above Pick Target
        2. Descend to Pick Target
        3. Grasp Hold Phase
        4. Lift Object A
        5. Transfer to Target B
        6. Descend to Drop Target
        7. Retract Arm
        """
        p_pick = np.asarray(p_pick, dtype=np.float64).flatten()
        p_place = np.asarray(p_place, dtype=np.float64).flatten()

        wp_list = []
        if p_start is not None:
            p_start_arr = np.asarray(p_start, dtype=np.float64).flatten()
            wp_list.append([p_start_arr[0], p_start_arr[1], p_start_arr[2]])

        wp_list.extend([
            [p_pick[0],  p_pick[1],  hover_height],  # Hover above Pick Target
            [p_pick[0],  p_pick[1],  p_pick[2]],     # Descend to Pick Target
            [p_pick[0],  p_pick[1],  p_pick[2]],     # Grasp Hold Phase
            [p_pick[0],  p_pick[1],  hover_height],  # Lift Object A
            [p_place[0], p_place[1], hover_height],  # Transfer to Target B
            [p_place[0], p_place[1], p_place[2]],    # Descend to Drop Target
            [p_place[0], p_place[1], hover_height]   # Retract Arm
        ])

        fsm_waypoints = np.array(wp_list).T

        ref_traj_list = []
        for wp in range(fsm_waypoints.shape[1] - 1):
            w_start = fsm_waypoints[:, wp]
            w_end = fsm_waypoints[:, wp + 1]
            for s in range(steps_per_phase):
                alpha = s / float(steps_per_phase)
                ref_traj_list.append((1.0 - alpha) * w_start + alpha * w_end)

        ref_traj_full = np.array(ref_traj_list).T  # Shape: (3, total_steps)

        # Pad trajectory with terminal point for horizon lookahead
        pad = np.tile(fsm_waypoints[:, -1:], (1, horizon_pad))
        ref_traj_full = np.hstack([ref_traj_full, pad])
        return ref_traj_full

    def forward_simulate_step(self, x_current: np.ndarray, u_apply: np.ndarray) -> np.ndarray:
        """Forward steps the rigid body plant using internal RK4 integrator."""
        return self.F_rk4(x_current, u_apply).full().flatten()

    def get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Evaluates end-effector Cartesian 3D position [x, y, z]."""
        return self.fk_pos_fun(q).full().flatten()


# ==============================================================================
# 6. CLOSED-LOOP BENCHMARK SIMULATION & PLOTTING
# ==============================================================================
def run_standalone_nmpc_simulation(
    p_pick: np.ndarray = np.array([0.45, -0.25, 0.12]),
    p_place: np.ndarray = np.array([0.45, 0.25, 0.12]),
    steps_per_phase: int = 50,
    save_plot_path: Optional[str] = "nmpc_tracking_evaluation.png",
    show_plot: bool = False
):
    """
    Executes closed-loop NMPC control loop with simulated CCDM dynamic adaptation
    and visualizes tracking performance.
    """
    print("=" * 80)
    print(" ARIA NMPC CLOSED-LOOP PICK-AND-PLACE CONTROL FOR FRANKA PANDA (7-DOF) ")
    print(" Parameterized Multiple Shooting via CasADi + IPOPT ")
    print("=" * 80)

    controller = FrankaNMPCController(horizon_steps=10, control_dt=0.01)

    hover_height = 0.35
    ref_traj_full = controller.plan_pick_and_place_trajectory(
        p_pick=p_pick,
        p_place=p_place,
        hover_height=hover_height,
        steps_per_phase=steps_per_phase
    )

    total_sim_steps = (7 - 1) * steps_per_phase
    q_home = np.array([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0])
    x_current = np.concatenate([q_home, np.zeros(7)])

    # Logging Buffers
    log_pos_actual = np.zeros((3, total_sim_steps))
    log_pos_ref = np.zeros((3, total_sim_steps))
    log_tau_apply = np.zeros((7, total_sim_steps))
    log_q_actual = np.zeros((7, total_sim_steps))
    solve_times = []

    print(f"\nStarting Receding-Horizon Control Loop ({total_sim_steps} steps at 100 Hz)...")

    for k_sim in range(total_sim_steps):
        # Extract horizon reference window (3 x (N+1))
        Y_ref_horizon = ref_traj_full[:, k_sim : k_sim + controller.N + 1]

        # Current End-Effector Position
        p_ee = controller.get_ee_position(x_current[:7])
        dist_to_pick = float(np.linalg.norm(p_ee - p_pick))

        # Dynamic CCDM Adaptation Simulation
        # (Boost Q_pos near pick grasp, damp velocity for fragile object)
        q_pos_boost = 2000.0 * (1.0 + 3.0 * np.exp(-dist_to_pick / 0.08))
        weights = DynamicNMPCWeights(
            Q_pos=np.array([q_pos_boost, q_pos_boost, q_pos_boost], dtype=np.float32),
            Q_vel=np.array([15.0, 15.0, 15.0], dtype=np.float32),
            R_action=np.ones(7, dtype=np.float32) * 1e-4,
            Q_terminal=np.array([q_pos_boost * 2.0, q_pos_boost * 2.0, q_pos_boost * 2.0], dtype=np.float32)
        )
        constraints = PhysicalConstraints(
            table_barrier_z=0.08,
            velocity_max=np.array([0.45, 0.45, 0.35, 1.0], dtype=np.float32)
        )

        # Solve NMPC Step
        u_apply, x_pred, s_info = controller.solve_control_step(
            x_current=x_current,
            Y_ref_horizon=Y_ref_horizon,
            nmpc_weights=weights,
            physical_constraints=constraints
        )

        # Forward Step Rigid-Body Simulation
        x_current = controller.forward_simulate_step(x_current, u_apply)

        # Log Metrics
        log_pos_actual[:, k_sim] = p_ee
        log_pos_ref[:, k_sim] = Y_ref_horizon[:, 0]
        log_tau_apply[:, k_sim] = u_apply
        log_q_actual[:, k_sim] = x_current[:7]
        solve_times.append(s_info["solve_time_ms"])

        if (k_sim + 1) % steps_per_phase == 0:
            phase_idx = (k_sim + 1) // steps_per_phase
            print(f"  [Phase {phase_idx}/6] Step {k_sim + 1:3d}/{total_sim_steps} | EE Pos: [{p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}] | Solve: {s_info['solve_time_ms']:.1f}ms")

    mean_solve_ms = float(np.mean(solve_times))
    p95_solve_ms = float(np.percentile(solve_times, 95))
    print(f"\nControl execution finished successfully!")
    print(f" - Average NMPC Solve Time : {mean_solve_ms:.2f} ms")
    print(f" - P95 NMPC Solve Time     : {p95_solve_ms:.2f} ms")

    # Tracking Accuracy
    pos_error_mm = (log_pos_ref - log_pos_actual) * 1000.0
    rmse_mm = float(np.sqrt(np.mean(pos_error_mm ** 2)))
    print(f" - Overall Task Space RMSE : {rmse_mm:.2f} mm")

    # Plotting
    time_grid = np.arange(total_sim_steps) * controller.dt
    fig = plt.figure(figsize=(14, 8))

    # 1. 3D Cartesian Tracking Plot
    ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')
    ax1.plot(log_pos_ref[0, :], log_pos_ref[1, :], log_pos_ref[2, :], 'r--', linewidth=2, label='Reference Trajectory')
    ax1.plot(log_pos_actual[0, :], log_pos_actual[1, :], log_pos_actual[2, :], 'b-', linewidth=1.5, label='NMPC Actual Path')
    ax1.scatter(p_pick[0], p_pick[1], p_pick[2], color='red', s=70, marker='s', label='Pick (Object A)')
    ax1.scatter(p_place[0], p_place[1], p_place[2], color='green', s=70, marker='o', label='Place (Target B)')
    ax1.set_title('3D Task Space Pick-and-Place Path (CCDM-Conditioned NMPC)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.grid(True)

    # 2. End-Effector Tracking Error (mm)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_grid, pos_error_mm[0, :], label='e_x')
    ax2.plot(time_grid, pos_error_mm[1, :], label='e_y')
    ax2.plot(time_grid, pos_error_mm[2, :], label='e_z')
    ax2.set_title('Task Space Tracking Error (mm)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (mm)')
    ax2.legend()
    ax2.grid(True)

    # 3. Applied Joint Torques vs Physical Hardware Limits
    ax3 = fig.add_subplot(2, 2, 4)
    for i in range(7):
        ax3.plot(time_grid, log_tau_apply[i, :], label=f'Joint {i+1}')
    ax3.axhline(87.0, color='black', linestyle='--', alpha=0.7, label='Arm Limit (+/- 87 Nm)')
    ax3.axhline(-87.0, color='black', linestyle='--', alpha=0.7)
    ax3.axhline(12.0, color='magenta', linestyle='--', alpha=0.7, label='Wrist Limit (+/- 12 Nm)')
    ax3.axhline(-12.0, color='magenta', linestyle='--', alpha=0.7)
    ax3.set_title('Applied Optimal Joint Torques')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (Nm)')
    ax3.grid(True)

    plt.tight_layout()
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=150)
        print(f" - Performance evaluation plot saved to: {save_plot_path}")

    if show_plot:
        plt.show()

    plt.close()
    return {
        "rmse_mm": rmse_mm,
        "mean_solve_ms": mean_solve_ms,
        "p95_solve_ms": p95_solve_ms,
        "log_pos_actual": log_pos_actual,
        "log_pos_ref": log_pos_ref,
        "log_tau_apply": log_tau_apply
    }


if __name__ == "__main__":
    run_standalone_nmpc_simulation(steps_per_phase=40, show_plot=False)


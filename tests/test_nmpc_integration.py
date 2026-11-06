"""
tests/test_nmpc_integration.py

Comprehensive Unit & Integration Test Suite for:
1. ARIA Franka 7-DOF NMPC Controller (CasADi + IPOPT)
2. Dynamic Parameter Injection from CCDM Layer (Dynamic Weights + Physical Constraints)
3. 4-Layer Master Pipeline Execution (Multi-Agent JSON -> Spatial Layer -> CCDM -> NMPC)
"""

import os
import numpy as np
import pytest
import mujoco

from nmpc_controller import FrankaNMPCController
from ccdm_dynamics.interfaces.controller_adaptation import (
    PhysicalConstraints,
    DynamicNMPCWeights
)
from spatial_layer.integrated_pipeline import (
    IntegratedSpatialCCDMPipeline,
    IntegratedPipelineOutput
)


SCENE_XML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "franka_scene.xml"
)


@pytest.fixture(scope="module")
def nmpc_controller():
    """Initializes a shared FrankaNMPCController instance for fast testing."""
    return FrankaNMPCController(
        horizon_steps=10,
        control_dt=0.01,
        max_ipopt_iter=30,
        print_level=0
    )


@pytest.fixture
def sim_env():
    """Builds the canonical MuJoCo Franka Panda scene for tests."""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    return model, data


def test_franka_kinematics_and_dynamics_functions(nmpc_controller):
    """Verifies Modified DH Craig forward kinematics and RK4 integration."""
    q_zero = np.zeros(7)
    ee_pos = nmpc_controller.fk_pos_fun(q_zero).full().flatten()

    # In Franka zero config, EE position has non-zero height along Z
    assert ee_pos[2] > 0.5, f"Expected EE height > 0.5m, got {ee_pos[2]}"

    # Verify RK4 integrator
    x_curr = np.zeros(14)
    tau_test = np.array([5.0, 5.0, 5.0, 2.0, 0.5, 0.5, 0.5])
    x_next = nmpc_controller.F_rk4(x_curr, tau_test).full().flatten()

    assert x_next.shape == (14,)
    assert not np.isnan(x_next).any()


def test_solve_control_step_with_ccdm_parameters(nmpc_controller):
    """Verifies that dynamic weights and constraints from CCDM are accepted without recompilation."""
    # Synthesize dynamic CCDM weights
    dynamic_weights = DynamicNMPCWeights(
        Q_pos=np.array([2500.0, 2500.0, 3000.0], dtype=np.float32),
        Q_vel=np.array([15.0, 15.0, 15.0], dtype=np.float32),
        R_action=np.ones(7, dtype=np.float32) * 5e-5,
        S_slew=np.ones(7, dtype=np.float32) * 1e-4,
        Q_terminal=np.array([5000.0, 5000.0, 6000.0], dtype=np.float32),
        W_slack=1000.0
    )

    # Dynamic physical constraints
    constraints = PhysicalConstraints(
        velocity_max=np.array([0.4, 0.4, 0.3, 1.0], dtype=np.float32),
        velocity_min=np.array([-0.4, -0.4, -0.3, -1.0], dtype=np.float32),
        acceleration_max=np.array([1.5, 1.5, 1.5], dtype=np.float32),
        table_barrier_z=0.405,
        phase="APPROACH"
    )

    # Initial state and target horizon
    x_current = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.5, 0.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    target_pos = np.array([0.5, 0.0, 0.45])
    Y_ref_horizon = np.repeat(target_pos.reshape(3, 1), nmpc_controller.N, axis=1)

    u_apply, x_pred, solve_info = nmpc_controller.solve_control_step(
        x_current=x_current,
        Y_ref_horizon=Y_ref_horizon,
        nmpc_weights=dynamic_weights,
        physical_constraints=constraints
    )

    assert u_apply.shape == (7,)
    assert x_pred.shape == (14,)
    assert solve_info["success"] is True
    assert solve_info["solve_time_ms"] > 0

    # Actuator limits check
    assert np.all(u_apply >= nmpc_controller.TAU_MIN - 1e-2)
    assert np.all(u_apply <= nmpc_controller.TAU_MAX + 1e-2)


def test_trajectory_planner(nmpc_controller):
    """Verifies that the pick-and-place FSM waypoint trajectory interpolator runs correctly."""
    p_pick = np.array([0.45, -0.15, 0.42])
    p_place = np.array([0.45, 0.25, 0.42])

    ref_traj = nmpc_controller.plan_pick_and_place_trajectory(
        p_pick=p_pick,
        p_place=p_place,
        hover_height=0.35,
        steps_per_phase=20,
        horizon_pad=5
    )

    assert ref_traj.shape[0] == 3
    assert ref_traj.shape[1] > 100
    # First waypoint matches pick hover
    assert np.isclose(ref_traj[2, 0], 0.35, atol=1e-3)


def test_four_layer_integrated_pipeline_step(sim_env, nmpc_controller):
    """Verifies complete 4-layer integration in IntegratedSpatialCCDMPipeline."""
    model, data = sim_env
    pipeline = IntegratedSpatialCCDMPipeline(
        model=model,
        data=data,
        controller=nmpc_controller
    )

    # Ingest Multi-Agent JSON
    mock_task = {
        "task_id": "test_nmpc_task_001",
        "instruction": "Pick up the red cube and place into bin carefully",
        "task_type": "pick_and_place",
        "pick_target": {
            "name": "target_block",
            "class": "cube",
            "initial_position_hint": [0.5, 0.0, 0.425],
            "mass": 0.15,
            "friction": 0.6
        },
        "place_target": {
            "name": "target_bin",
            "class": "bin",
            "target_position": [0.65, 0.20, 0.425]
        },
        "operational_context": {
            "fragility": 0.5,
            "safety_priority": 0.8,
            "speed_priority": 0.5
        }
    }
    pipeline.set_task_from_multi_agent(mock_task)

    # Run one step with NMPC enabled
    output: IntegratedPipelineOutput = pipeline.step(
        force_depth_refresh=False,
        execute_nmpc=True,
        apply_torque_to_sim=True
    )

    assert output.timestep == 1
    assert output.task_id == "test_nmpc_task_001"
    assert output.relative_distance_to_grasp > 0
    assert output.torque_command is not None
    assert output.torque_command.shape == (7,)
    assert output.predicted_next_state is not None
    assert output.predicted_next_state.shape == (14,)
    assert "nmpc_solve_ms" in output.timings_ms
    assert output.status in ("ok", "target_interpolated")

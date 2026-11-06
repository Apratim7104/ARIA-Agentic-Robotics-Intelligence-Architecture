"""
tests/test_ccdm_integration.py

Comprehensive integration test suite verifying:
1. Multi-Agent JSON task message ingestion & parsing.
2. Initial object position locking and continuous relative distance tracking.
3. CCDM neural network FiLM conditioning on context, spatial states, and physical scene.
4. Dynamic physical constraints generation (velocity limits, acceleration limits, table barrier, chance margins).
5. Dynamic NMPC cost function weight adaptation (position escalation near grasp, fragility damping, jerk penalties).
6. Full end-to-end pipeline execution on the MuJoCo Franka Emika Panda scene.
"""

import os
import json
import pytest
import numpy as np
import torch
import mujoco

from spatial_layer.multi_agent_interface import MultiAgentInterface, MultiAgentTaskPlan
from spatial_layer.integrated_pipeline import (
    IntegratedSpatialCCDMPipeline,
    IntegratedPipelineOutput
)
try:
    from ccdm_dynamics.models.ccdm import CCDM
    from ccdm_dynamics.interfaces.controller_adaptation import (
        CCDMControllerAdaptation,
        PhysicalConstraints,
        DynamicNMPCWeights
    )
    from ccdm_dynamics.interfaces.spatial_grounding import SystemAgnosticSpatialGrounding
    from ccdm_dynamics.utils.normalization import StandardScaler
except ImportError:
    from ccdm.models.ccdm import CCDM
    from ccdm.interfaces.controller_adaptation import (
        CCDMControllerAdaptation,
        PhysicalConstraints,
        DynamicNMPCWeights
    )
    from ccdm.interfaces.spatial_grounding import SystemAgnosticSpatialGrounding
    from ccdm.utils.normalization import StandardScaler

SCENE_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "franka_scene.xml")


def test_multi_agent_json_parsing():
    """Verify that multi-agent JSON input is correctly validated and parsed."""
    raw_message = {
        "task_id": "test_pick_001",
        "instruction": "Pick up the fragile glassware block and transfer to bin",
        "task_type": "pick_and_place",
        "pick_target": {
            "name": "glass_cube",
            "class": "cube",
            "mass": 0.25,
            "friction": 0.35,
            "bbox_hint": [100, 100, 160, 160]
        },
        "place_target": {
            "name": "storage_bin",
            "position": [0.70, 0.25, 0.42],
            "tolerance": 0.02
        },
        "operational_context": {
            "fragility": 0.85,
            "speed_priority": 0.30,
            "safety_priority": 0.95,
            "task_code": 0.0,
            "crowd_density": 0.20
        }
    }

    plan: MultiAgentTaskPlan = MultiAgentInterface.parse_task_message(raw_message)

    assert plan.task_id == "test_pick_001"
    assert plan.pick_target.cls == "cube"
    assert plan.pick_target.mass == 0.25
    assert plan.pick_target.friction == 0.35
    assert np.allclose(plan.place_target.target_position, [0.70, 0.25, 0.42])
    assert abs(plan.fragility - 0.85) < 1e-5
    assert abs(plan.speed_priority - 0.30) < 1e-5
    assert abs(plan.safety_priority - 0.95) < 1e-5
    assert len(plan.operational_context) == 5


def test_spatial_grounding_kinematics():
    """Verify relative displacement vector, Euclidean distance, and range rate calculation."""
    grounding = SystemAgnosticSpatialGrounding(default_dt=0.05)

    ee_pos = np.array([0.40, 0.0, 0.50], dtype=np.float32)
    obj_pos = np.array([0.50, 0.0, 0.40], dtype=np.float32)

    kinematics = grounding.estimate_3d_relative_kinematics(
        system_origin_3d=ee_pos,
        target_center_3d=obj_pos,
        system_velocity_3d=np.array([0.2, 0.0, -0.1], dtype=np.float32),
        target_velocity_3d=np.zeros(3, dtype=np.float32)
    )

    # Expected delta: [0.10, 0.0, -0.10]
    expected_dist = float(np.sqrt(0.10**2 + 0.10**2))
    assert abs(kinematics["euclidean_distance"] - expected_dist) < 1e-4
    assert np.allclose(kinematics["delta_vector"], [0.10, 0.0, -0.10], atol=1e-4)
    # Relative range rate along line of sight should be negative (closing in)
    assert kinematics["range_rate"] < 0.0


def test_ccdm_forward_pass():
    """Verify CCDM neural network FiLM modulation and dual-head outputs."""
    model = CCDM(state_dim=6, action_dim=4, context_dim=5, physical_dim=3, residual_dim=6)
    model.eval()

    batch_size = 4
    s = torch.randn(batch_size, 6)
    a = torch.randn(batch_size, 4)
    c = torch.rand(batch_size, 5)
    p = torch.rand(batch_size, 3)

    with torch.no_grad():
        r_mean, log_var = model(s, a, c, p)

    assert r_mean.shape == (batch_size, 6)
    assert log_var.shape == (batch_size, 6)
    assert not torch.isnan(r_mean).any()
    assert not torch.isnan(log_var).any()


def test_ccdm_controller_adaptation_constraints_and_weights():
    """Verify physical constraint deceleration and dynamic NMPC weight scaling."""
    adaptation = CCDMControllerAdaptation()

    # Scenario 1: Far away approach phase (d_rel = 0.50m), low fragility
    ctx_far = {"fragility": 0.1, "speed_priority": 0.8, "safety_priority": 0.5}
    spatial_far = {
        "p_ee": [0.3, 0.0, 0.7],
        "v_ee": [0.2, 0.0, -0.1],
        "p_obj": [0.6, 0.0, 0.425],
        "d_rel": 0.50
    }
    phys_scene = {"table_height": 0.40, "mass": 0.10, "friction": 0.40}

    cons_far, weights_far, _ = adaptation.compute_adaptation(ctx_far, spatial_far, phys_scene)

    assert cons_far.phase == "approach"
    assert cons_far.velocity_max[0] > 0.30 # High speed permitted in approach

    # Scenario 2: Close pre-grasp phase (d_rel = 0.03m), highly fragile object
    ctx_close = {"fragility": 0.9, "speed_priority": 0.2, "safety_priority": 0.95}
    spatial_close = {
        "p_ee": [0.49, 0.0, 0.43],
        "v_ee": [0.02, 0.0, -0.01],
        "p_obj": [0.50, 0.0, 0.425],
        "d_rel": 0.03
    }

    cons_close, weights_close, _ = adaptation.compute_adaptation(ctx_close, spatial_close, phys_scene)

    # 1. Verification of Physical Constraints:
    # Velocity must be decelerated in pre-grasp and dampened by fragility
    assert cons_close.velocity_max[0] < cons_far.velocity_max[0]
    # Acceleration limit must be constrained by friction and fragility to prevent slip
    assert cons_close.acceleration_max[0] <= 0.40 * 9.81
    # Table safety barrier must prevent table collision
    assert cons_close.table_barrier_z > 0.40

    # 2. Verification of Dynamic NMPC Weights:
    # Q_pos must escalate near grasp for sub-millimeter terminal precision
    assert weights_close.Q_pos[0] > weights_far.Q_pos[0]
    # Velocity damping Q_vel must increase for fragile payload
    assert weights_close.Q_vel[0] > weights_far.Q_vel[0]
    # Slew rate jerk penalty S_slew must be significantly higher for fragile payload
    assert weights_close.S_slew[0] > weights_far.S_slew[0]
    # Slack penalty must be higher with safety_priority=0.95
    assert weights_close.W_slack > weights_far.W_slack


def test_end_to_end_integrated_pipeline():
    """Verify live integration on MuJoCo Franka scene across multiple timesteps."""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    pipeline = IntegratedSpatialCCDMPipeline(model, data)

    # Multi-Agent JSON Input
    multi_agent_json = {
        "task_id": "panda_pick_cube_42",
        "instruction": "Pick up the red cube from the table and place in bin",
        "task_type": "pick_and_place",
        "pick_target": {"name": "target_block", "class": "cube", "mass": 0.10, "friction": 0.40},
        "place_target": {"name": "target_bin", "position": [0.65, 0.20, 0.425]},
        "operational_context": {
            "fragility": 0.70,
            "speed_priority": 0.50,
            "safety_priority": 0.90,
            "task_code": 0.0,
            "crowd_density": 0.0
        }
    }

    # Ingest Multi-Agent JSON
    task_plan = pipeline.set_task_from_multi_agent(multi_agent_json)
    assert task_plan.task_id == "panda_pick_cube_42"

    # Execute first step: initial position must be locked
    out1: IntegratedPipelineOutput = pipeline.step(force_depth_refresh=True)
    initial_pos = out1.initial_object_location.copy()

    assert initial_pos is not None
    assert len(initial_pos) == 3
    # Check that initial position is on the table (around z ~ 0.425)
    assert abs(initial_pos[2] - 0.425) < 0.05
    assert out1.relative_distance_to_grasp > 0.0
    assert out1.physical_constraints is not None
    assert out1.nmpc_weights is not None
    assert "total_pipeline_ms" in out1.timings_ms

    # Step simulation forward and run second step
    for _ in range(10):
        mujoco.mj_step(model, data)

    out2: IntegratedPipelineOutput = pipeline.step(force_depth_refresh=False)

    # Initial position must remain locked
    assert np.allclose(out2.initial_object_location, initial_pos)
    # Warm steady-state execution should be fast
    assert out2.timings_ms["total_pipeline_ms"] < 30.0
    # Output must be serializable to dict
    d = out2.to_dict()
    assert d["task_id"] == "panda_pick_cube_42"
    assert "velocity_max" in d["physical_constraints"]
    assert "Q_pos" in d["nmpc_weights"]
    assert len(d["spatial_features"]) == 30

"""
tests/test_multi_agent_integration.py
=====================================
Integration tests for ARIA Layer 1 (Multi-Agent Orchestrator) and its
end-to-end handoff to Layer 2 (Spatial Reasoning Layer), Layer 3 (CCDM),
and Layer 4 (NMPC Controller).
"""

import os
import sys
import numpy as np
import pytest
import mujoco

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

orch_dir = os.path.join(ROOT_DIR, "multi_agent_orchestrator")
if orch_dir not in sys.path:
    sys.path.insert(0, orch_dir)

from spatial_layer.integrated_pipeline import IntegratedSpatialCCDMPipeline, IntegratedPipelineOutput
from spatial_layer.multi_agent_interface import MultiAgentInterface, MultiAgentTaskPlan
from multi_agent_orchestrator.state.schema import Route, AgentName, OrchestratorState
from multi_agent_orchestrator.agents.planner import planner_node
from multi_agent_orchestrator.agents.robot_perception_agent import robot_perception_node
from multi_agent_orchestrator.agents.robot_planner_agent import robot_planner_node
from multi_agent_orchestrator.graph.orchestrator import run_pipeline


def test_multi_agent_planner_routing():
    """Verify planner node routes manipulation prompts to ROBOT_PICKUP with target object extraction."""
    state = {
        "user_input": "Pick up the red cube and place it on the target spot",
        "session_id": "test_session_001",
        "messages": [],
        "current_step": 0,
        "max_steps": 6,
    }
    res = planner_node(state)
    intent = res.get("intent", {})
    assert intent.get("route") == Route.ROBOT_PICKUP
    assert AgentName.ROBOT_PERC in intent.get("agents_needed", [])
    assert AgentName.ROBOT_PLAN in intent.get("agents_needed", [])
    assert "cube" in intent.get("target_object", "").lower()


def test_multi_agent_graph_execution():
    """Verify full orchestrator graph runs and produces robot_handoff payload."""
    initial_state = {
        "user_input": "Pick up the red cube and place it on the target spot",
        "session_id": "test_graph_001",
        "image_path": None,
        "depth_path": None,
    }
    final_state = run_pipeline(initial_state)
    assert "intent" in final_state
    assert final_state["intent"]["route"] == Route.ROBOT_PICKUP
    assert "robot_handoff" in final_state
    handoff = final_state["robot_handoff"]
    assert "selected_items" in handoff
    assert "grasp_poses" in handoff
    assert len(handoff["grasp_poses"]) == 3
    phases = [p["phase"] for p in handoff["grasp_poses"]]
    assert phases == ["pre_grasp", "grasp", "post_grasp"]


def test_spatial_layer_handoff_ingestion():
    """Verify Spatial Reasoning Layer correctly parses multi_agent robot_handoff."""
    interface = MultiAgentInterface()
    mock_handoff_state = {
        "user_input": "Pick up the red cube",
        "session_id": "handoff_test_01",
        "robot_handoff": {
            "selected_items": [{
                "label": "red_cube",
                "centroid_3d": [0.48, -0.15, 0.425],
                "bbox_xyxy": [100.0, 120.0, 160.0, 180.0],
                "mass": 0.15,
                "friction": 0.45
            }],
            "grasp_poses": [
                {"phase": "pre_grasp", "position": {"x": 0.48, "y": -0.15, "z": 0.575}},
                {"phase": "grasp", "position": {"x": 0.48, "y": -0.15, "z": 0.425}},
                {"phase": "post_grasp", "position": {"x": 0.48, "y": -0.15, "z": 0.625}},
            ],
            "timestamp": "2026-11-06T20:00:00Z"
        }
    }
    plan = interface.parse_task_message(mock_handoff_state)
    assert isinstance(plan, MultiAgentTaskPlan)
    assert plan.pick_target.name == "red_cube"
    assert plan.pick_target.cls == "red_cube"
    assert plan.pick_target.bbox_hint == [100.0, 120.0, 160.0, 180.0]
    assert np.allclose(plan.pick_target.position_hint, [0.48, -0.15, 0.425])


def test_end_to_end_multi_agent_to_spatial_pipeline():
    """Verify Layer 1 -> Layer 2 -> Layer 3 data flow and NMPC parameter generation end-to-end."""
    xml_path = os.path.join(ROOT_DIR, "models", "franka_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    pipeline = IntegratedSpatialCCDMPipeline(model, data, enable_nmpc=False)

    prompt = "Pick up the red cube and place it on the target spot"
    agent_result = pipeline.run_multi_agent_layer(user_prompt=prompt)

    assert pipeline.current_task is not None
    assert "cube" in pipeline.current_task.pick_target.cls.lower() or "object" in pipeline.current_task.pick_target.cls.lower()
    assert agent_result.get("robot_handoff") is not None

    # Step the pipeline once to verify Layer 2 + Layer 3 execution
    out: IntegratedPipelineOutput = pipeline.step(execute_nmpc=False)
    assert out.timestep == 1
    assert out.task_prompt == prompt
    assert out.relative_distance_to_grasp > 0.0
    assert out.spatial_features.shape == (30,)
    assert out.physical_constraints is not None
    assert out.nmpc_weights is not None
    assert out.status in ("ok", "target_interpolated")

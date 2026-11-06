# agents/robot_planner_agent.py
"""
Robot Planner Agent
===================
Uses the Siemens LLM API (qwen-3.6-27b) to reason about the detected scene
and produce a structured grasp plan:
 - Which object to pick up
 - Pre-grasp, grasp, and post-grasp 6-DOF poses (as named dicts)
 - Gripper width
 - Confidence + reasoning

This node is the terminal node of the robot branch inside the ARIA
multi-agent graph. After producing the grasp plan it assembles a clean
`robot_handoff` payload — the sole interface contract between the
multi-agent system and the external MPC Controller.

Input state keys: detected_objects, scene_description, target_object,
                  user_input, masks, depth_map
Output state keys: grasp_plan, robot_handoff
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from state.schema import (
    OrchestratorState, GraspPlan, DetectedObject,
    AgentName, add_message,
)
from llm.client import call_llm_json, system_user

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────
ROBOT_PLANNER_SYSTEM = """You are a robot manipulation planning expert.
Given a scene description with detected objects (labels, 3-D positions in metres,
confidence scores) and a target object to pick up, you must produce a precise
grasp plan for a 6-DOF robot arm with a parallel-jaw gripper.

OUTPUT FORMAT (strict JSON, no markdown, no explanation):
{
 "target_object" : "<label of object to grasp>",
 "grasp_pose" : [x, y, z, roll, pitch, yaw],
 "pre_grasp_pose" : [x, y, z, roll, pitch, yaw],
 "post_grasp_pose": [x, y, z, roll, pitch, yaw],
 "gripper_width" : <metres, e.g. 0.08>,
 "confidence" : <0.0-1.0>,
 "reasoning" : "<one paragraph explaining the grasp strategy>"
}

COORDINATE CONVENTIONS:
- All positions in metres, in camera/world frame
- roll/pitch/yaw in radians
- pre_grasp_pose: 15 cm above the grasp pose (approach from above)
- post_grasp_pose: 20 cm above the grasp pose (lift after grasping)
- gripper_width: slightly wider than the object's estimated width

SAFETY RULES:
- Never plan a grasp that would collide with other objects
- If confidence < 0.5, still provide best estimate but note uncertainty
- If no suitable object found, set confidence to 0.0 and explain in reasoning
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_scene_prompt(
    scene_description: str,
    detected_objects: List[DetectedObject],
    target_object: str,
    user_input: str,
) -> str:
    """Build the user prompt with full scene context."""
    objects_json = []
    for obj in (detected_objects or []):
        objects_json.append({
            "label": obj.get("label", "unknown"),
            "confidence": obj.get("confidence", 0.0),
            "depth_m": obj.get("depth_m", 0.0),
            "centroid_3d": obj.get("centroid_3d", [0, 0, 0]),
            "bbox_xyxy": obj.get("bbox_xyxy", []),
        })
    return (
        f"USER REQUEST: {user_input}\n\n"
        f"TARGET OBJECT: {target_object or 'the most graspable object in the scene'}\n\n"
        f"SCENE DESCRIPTION:\n{scene_description}\n\n"
        f"DETECTED OBJECTS (JSON):\n{json.dumps(objects_json, indent=2)}\n\n"
        f"Generate a grasp plan for the target object."
    )

def _pose_list_to_dict(pose: List[float], phase: str, frame: str = "base_link") -> Dict[str, Any]:
    """
    Convert a raw [x, y, z, roll, pitch, yaw] list into a named dict.
    Named dicts are self-documenting and far safer for the MPC Controller
    to consume than positional lists.
    """
    if not pose or len(pose) < 6:
        pose = [0.0] * 6
    return {
        "phase": phase,
        "position": {"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2])},
        "orientation": {"roll": float(pose[3]), "pitch": float(pose[4]), "yaw": float(pose[5])},
        "frame": frame,
    }

def _build_robot_handoff(
    grasp_plan: GraspPlan,
    detected_objects: List[DetectedObject],
    masks: Any,
    depth_map: Any,
) -> Dict[str, Any]:
    """
    Assemble the controller-ready handoff payload.
    This is the sole interface contract between the ARIA multi-agent system
    and the external MPC Controller. The MPC Controller reads this dict
    directly — no LangGraph, no LLM, no agents involved downstream.

    Schema:
    selected_items : list of detected objects chosen for pickup
    grasp_poses : pre_grasp / grasp / post_grasp as named 6-DOF dicts
    masks : SAM2 mask references forwarded from perception state
    depth_map : aligned depth map reference forwarded from perception state
    confidence : overall grasp confidence from the LLM plan
    reasoning : LLM reasoning string for logging / debugging
    timestamp : UTC ISO-8601 string
    """
    target_label = grasp_plan.get("target_object", "")
    selected_items = [
        obj for obj in (detected_objects or [])
        if obj.get("label", "").lower() == target_label.lower()
    ]
    # Fallback: include all objects if no exact match
    if not selected_items:
        selected_items = detected_objects or []

    grasp_poses = [
        _pose_list_to_dict(grasp_plan.get("pre_grasp_pose", [0.0] * 6), phase="pre_grasp"),
        _pose_list_to_dict(grasp_plan.get("grasp_pose", [0.0] * 6), phase="grasp"),
        _pose_list_to_dict(grasp_plan.get("post_grasp_pose", [0.0] * 6), phase="post_grasp"),
    ]

    return {
        "selected_items": selected_items,
        "grasp_poses": grasp_poses,
        "gripper_width": grasp_plan.get("gripper_width", 0.08),
        "masks": masks,
        "depth_map": depth_map,
        "confidence": grasp_plan.get("confidence", 0.0),
        "reasoning": grasp_plan.get("reasoning", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── Node ───────────────────────────────────────────────────────────────────────
def robot_planner_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node: generate a grasp plan using the Siemens LLM, then
    assemble and emit the `robot_handoff` payload for the external MPC Controller.
    This is the terminal node of the robot branch inside the ARIA graph.
    After this node the graph flows to the synthesizer, which produces a
    human-readable summary of the pickup plan. The MPC Controller reads
    `robot_handoff` from the final OrchestratorState independently.
    """
    detected_objects = state.get("detected_objects") or []
    scene_description = state.get("scene_description") or "No scene description available."
    target_object = state.get("target_object") or ""
    # ── FIX: use "user_input" — the canonical key set by main.py / run_pipeline()
    user_input = state.get("user_input") or state.get("user_query") or ""
    masks = state.get("masks") # forwarded from robot_perception
    depth_map = state.get("depth_map") # forwarded from robot_perception

    logger.info(
        "[robot_planner] Planning grasp | target='%s' | objects=%d",
        target_object, len(detected_objects),
    )

    # ── Early exit: nothing to grasp ─────────────────────────────────────────
    if not detected_objects:
        logger.warning("[robot_planner] No detected objects — returning empty plan.")
        empty_plan: GraspPlan = {
            "target_object": target_object,
            "grasp_pose": [0.0] * 6,
            "pre_grasp_pose": [0.0] * 6,
            "post_grasp_pose": [0.0] * 6,
            "gripper_width": 0.08,
            "confidence": 0.0,
            "reasoning": "No objects detected in the scene.",
        }
        robot_handoff = _build_robot_handoff(empty_plan, [], masks, depth_map)
        return {"grasp_plan": empty_plan, "robot_handoff": robot_handoff}

    # ── LLM call ─────────────────────────────────────────────────────────────
    user_prompt = _build_scene_prompt(
        scene_description, detected_objects, target_object, user_input
    )
    try:
        result = call_llm_json(
            messages=system_user(ROBOT_PLANNER_SYSTEM, user_prompt),
            temperature=0.05,
            caller=AgentName.ROBOT_PLAN,
        )
    except Exception as exc:
        logger.error("[robot_planner] LLM call failed: %s — using centroid fallback.", exc)
        obj0 = detected_objects[0]
        cx, cy, cz = obj0.get("centroid_3d", [0.3, 0.0, 0.5])
        result = {
            "target_object": obj0.get("label", "object"),
            "grasp_pose": [cx, cy, cz, 0.0, 1.5708, 0.0],
            "pre_grasp_pose": [cx, cy, cz + 0.15, 0.0, 1.5708, 0.0],
            "post_grasp_pose": [cx, cy, cz + 0.20, 0.0, 1.5708, 0.0],
            "gripper_width": 0.08,
            "confidence": 0.3,
            "reasoning": (
                f"LLM unavailable. Fallback grasp at centroid of "
                f"'{obj0.get('label', 'object')}'."
            ),
        }

    # ── Assemble GraspPlan ────────────────────────────────────────────────────
    grasp_plan: GraspPlan = {
        "target_object": result.get("target_object", target_object),
        "grasp_pose": result.get("grasp_pose", [0.0] * 6),
        "pre_grasp_pose": result.get("pre_grasp_pose", [0.0] * 6),
        "post_grasp_pose": result.get("post_grasp_pose", [0.0] * 6),
        "gripper_width": result.get("gripper_width", 0.08),
        "confidence": result.get("confidence", 0.5),
        "reasoning": result.get("reasoning", ""),
    }

    logger.info(
        "[robot_planner] Grasp plan ready | target=%s | confidence=%.2f | pose=%s",
        grasp_plan["target_object"],
        grasp_plan["confidence"],
        grasp_plan["grasp_pose"],
    )

    # ── Assemble robot_handoff (MPC Controller interface) ────────────────────
    robot_handoff = _build_robot_handoff(
        grasp_plan, detected_objects, masks, depth_map
    )

    logger.info(
        "[robot_planner] robot_handoff emitted | items=%d | poses=%d | ts=%s",
        len(robot_handoff["selected_items"]),
        len(robot_handoff["grasp_poses"]),
        robot_handoff["timestamp"],
    )

    # ── Update message history ────────────────────────────────────────────────
    updated_messages = add_message(
        state=state,
        role=AgentName.ROBOT_PLAN,
        content=(
            f"Grasp plan for '{grasp_plan['target_object']}' ready. "
            f"Confidence={grasp_plan['confidence']:.2f}. "
            f"Grasp pose={grasp_plan['grasp_pose']}. "
            f"robot_handoff emitted with {len(robot_handoff['selected_items'])} selected item(s) "
            f"and {len(robot_handoff['grasp_poses'])} pose phases. "
            f"Reasoning: {grasp_plan['reasoning'][:120]}..."
        ),
    )

    return {
        "grasp_plan": grasp_plan,
        "robot_handoff": robot_handoff,
        "messages": updated_messages,
    }

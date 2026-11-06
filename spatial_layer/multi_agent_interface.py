"""
spatial_layer/multi_agent_interface.py

Ingestion and validation interface for high-level JSON messages received from the
Multi-Agent Perception & Planning Pipeline (LLMs, VLMs, Image Masking Models).

Translates natural language user intent and VLM scene decomposition into structured
robotic manipulation specifications for the Spatial Reasoning Layer and CCDM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


@dataclass
class TargetSpecification:
    """Specification of an object to manipulate or interact with."""
    name: str = "cube"
    cls: str = "cube"
    target_id: int = 0
    bbox_hint: Optional[List[int]] = None           # [xmin, ymin, xmax, ymax]
    position_hint: Optional[List[float]] = None     # [x, y, z] initial estimate
    mass: float = 0.10                              # Estimated mass in kg
    friction: float = 0.40                          # Estimated surface friction coefficient mu


@dataclass
class PlaceSpecification:
    """Specification of target placement location or receptacle."""
    name: str = "target_zone"
    target_position: np.ndarray = field(default_factory=lambda: np.array([0.65, 0.20, 0.425], dtype=np.float32))
    tolerance: float = 0.03                         # Acceptance radius in meters
    orientation_quat: Optional[List[float]] = None  # Desired placement orientation [w, x, y, z]


@dataclass
class MultiAgentTaskPlan:
    """
    Standardized task plan produced by the Multi-Agent System Pipeline
    (LLM Planner + VLM Object Localizer + Semantic Masking Agents).
    """
    task_id: str
    user_prompt: str
    task_type: str = "pick_and_place"              # "pick_and_place", "reach", "lift"
    pick_target: TargetSpecification = field(default_factory=TargetSpecification)
    place_target: PlaceSpecification = field(default_factory=PlaceSpecification)
    operational_context: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.8, 0.0, 0.0], dtype=np.float32))
    context_dict: Dict[str, float] = field(default_factory=dict)
    waypoints: List[str] = field(default_factory=lambda: ["approach", "grasp", "lift", "transport", "place"])
    raw_json: Dict[str, Any] = field(default_factory=dict)
    subtasks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def fragility(self) -> float:
        return float(self.operational_context[0])

    @property
    def speed_priority(self) -> float:
        return float(self.operational_context[1])

    @property
    def safety_priority(self) -> float:
        return float(self.operational_context[2])


class MultiAgentInterface:
    """
    Parses, validates, and normalizes task instructions from multi-agent systems.
    """

    @staticmethod
    def parse_task_message(message: Union[str, bytes, Dict[str, Any]]) -> MultiAgentTaskPlan:
        """
        Ingests a JSON string, byte buffer, or dictionary from the multi-agent system.
        """
        if isinstance(message, (str, bytes)):
            try:
                data = json.loads(message)
            except Exception as e:
                raise ValueError(f"Failed to parse Multi-Agent JSON message: {e}")
        elif isinstance(message, dict):
            data = message
        else:
            raise TypeError(f"Expected JSON string or dict, got {type(message)}")

        task_id = str(data.get("task_id", data.get("session_id", f"task_{int(np.random.randint(1000, 9999))}")))
        prompt = str(data.get("prompt", data.get("user_prompt", data.get("user_input", data.get("instruction", "pick and place object")))))
        task_type = str(data.get("task_type", data.get("task", "pick_and_place"))).lower()

        # Check for robot_handoff from Multi-Agent Orchestrator
        handoff = data.get("robot_handoff")
        if handoff is None and "selected_items" in data and "grasp_poses" in data:
            handoff = data

        selected_items = handoff.get("selected_items", []) if isinstance(handoff, dict) else []
        grasp_poses = handoff.get("grasp_poses", []) if isinstance(handoff, dict) else []

        grasp_pos = None
        for gp in grasp_poses:
            if isinstance(gp, dict) and gp.get("phase") == "grasp":
                pos_d = gp.get("position", {})
                if isinstance(pos_d, dict) and "x" in pos_d:
                    grasp_pos = [float(pos_d["x"]), float(pos_d["y"]), float(pos_d["z"])]
                    break

        # 1. Parse Pick Target
        if selected_items or grasp_pos is not None:
            first_item = selected_items[0] if selected_items else {}
            item_name = str(first_item.get("label", data.get("target_object", "cube")))
            item_pos = first_item.get("centroid_3d", grasp_pos)
            item_bbox = first_item.get("bbox_xyxy", None)
            pick_target = TargetSpecification(
                name=item_name,
                cls=item_name,
                target_id=0,
                bbox_hint=item_bbox,
                position_hint=item_pos,
                mass=float(first_item.get("mass", 0.15)),
                friction=float(first_item.get("friction", 0.40))
            )
        else:
            pick_raw = data.get("pick_target", data.get("required_object", data.get("target_object", {})))
            if isinstance(pick_raw, str):
                pick_target = TargetSpecification(name=pick_raw, cls=pick_raw)
            elif isinstance(pick_raw, dict):
                pick_target = TargetSpecification(
                    name=str(pick_raw.get("name", pick_raw.get("class", "cube"))),
                    cls=str(pick_raw.get("class", "cube")),
                    target_id=int(pick_raw.get("id", 0)),
                    bbox_hint=pick_raw.get("bbox", pick_raw.get("bbox_hint")),
                    position_hint=pick_raw.get("position", pick_raw.get("position_hint")),
                    mass=float(pick_raw.get("mass", 0.10)),
                    friction=float(pick_raw.get("friction", 0.40))
                )
            else:
                pick_target = TargetSpecification()

        # 2. Parse Place Target
        place_raw = data.get("place_target", data.get("place_location", data.get("target_location", {})))
        if isinstance(place_raw, (list, tuple, np.ndarray)):
            pos = np.asarray(place_raw, dtype=np.float32)
            place_target = PlaceSpecification(name="target_pos", target_position=pos)
        elif isinstance(place_raw, dict):
            pos_arr = place_raw.get("position", place_raw.get("target_position", [0.65, 0.20, 0.425]))
            place_target = PlaceSpecification(
                name=str(place_raw.get("name", "target_zone")),
                target_position=np.asarray(pos_arr, dtype=np.float32),
                tolerance=float(place_raw.get("tolerance", 0.03)),
                orientation_quat=place_raw.get("orientation_quat")
            )
        elif isinstance(place_raw, str):
            # Named location e.g. "bin"
            place_target = PlaceSpecification(name=place_raw, target_position=np.array([0.65, 0.20, 0.425], dtype=np.float32))
        else:
            place_target = PlaceSpecification()

        # 3. Parse Operational Context (c_t in [0, 1]^5)
        raw_ctx = data.get("context", data.get("operational_context", {}))
        if isinstance(raw_ctx, dict):
            fragility = float(raw_ctx.get("fragility", 0.5))
            speed_pri = float(raw_ctx.get("speed_priority", 0.5))
            safety_pri = float(raw_ctx.get("safety_priority", 0.8))
            task_code = float(raw_ctx.get("task_code", 0.0 if "pick" in task_type else 1.0))
            crowd = float(raw_ctx.get("crowd_density", raw_ctx.get("obstacle_density", 0.0)))
            c_vec = np.array([fragility, speed_pri, safety_pri, task_code, crowd], dtype=np.float32)
            ctx_dict = {
                "fragility": fragility,
                "speed_priority": speed_pri,
                "safety_priority": safety_pri,
                "task_code": task_code,
                "crowd_density": crowd
            }
        elif isinstance(raw_ctx, (list, tuple, np.ndarray)):
            c_arr = np.asarray(raw_ctx, dtype=np.float32).flatten()
            c_vec = np.zeros(5, dtype=np.float32)
            c_vec[:min(5, len(c_arr))] = c_arr[:min(5, len(c_arr))]
            ctx_dict = {
                "fragility": float(c_vec[0]),
                "speed_priority": float(c_vec[1]),
                "safety_priority": float(c_vec[2]),
                "task_code": float(c_vec[3]),
                "crowd_density": float(c_vec[4])
            }
        else:
            c_vec = np.array([0.5, 0.5, 0.8, 0.0, 0.0], dtype=np.float32)
            ctx_dict = {
                "fragility": 0.5,
                "speed_priority": 0.5,
                "safety_priority": 0.8,
                "task_code": 0.0,
                "crowd_density": 0.0
            }

        c_vec = np.clip(c_vec, 0.0, 1.0)
        c_vec[3] = np.clip(c_vec[3], 0.0, 5.0)

        waypoints = data.get("waypoints", ["approach", "grasp", "lift", "transport", "place"])
        subtasks = data.get("subtasks", data.get("tasks", []))

        return MultiAgentTaskPlan(
            task_id=task_id,
            user_prompt=prompt,
            task_type=task_type,
            pick_target=pick_target,
            place_target=place_target,
            operational_context=c_vec,
            context_dict=ctx_dict,
            waypoints=waypoints,
            raw_json=data,
            subtasks=subtasks
        )

    @staticmethod
    def create_mock_task_message(
        prompt: str = "Pick up the red block and place it in the right basket",
        pick_class: str = "cube",
        place_position: Sequence[float] = (0.65, 0.20, 0.425),
        fragility: float = 0.70,
        speed_priority: float = 0.40,
        safety_priority: float = 0.90
    ) -> Dict[str, Any]:
        """Helper to generate mock Multi-Agent JSON for testing and simulation."""
        return {
            "task_id": "multi_agent_plan_001",
            "prompt": prompt,
            "task_type": "pick_and_place",
            "pick_target": {
                "name": "red_cube",
                "class": pick_class,
                "id": 0,
                "mass": 0.15,
                "friction": 0.45
            },
            "place_target": {
                "name": "target_basket",
                "position": list(place_position),
                "tolerance": 0.025
            },
            "operational_context": {
                "fragility": fragility,
                "speed_priority": speed_priority,
                "safety_priority": safety_priority,
                "task_code": 0.0,
                "obstacle_density": 0.1
            },
            "waypoints": ["approach", "grasp", "lift", "transport", "place"]
        }


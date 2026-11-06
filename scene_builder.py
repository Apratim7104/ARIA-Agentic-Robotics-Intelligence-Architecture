"""
scene_builder.py

Interactive & Programmatic MuJoCo Simulation Scene Builder for the
Spatial Reasoning Layer, CCDM Dynamics, and Franka 7-DOF NMPC Control Stack.

Accepts natural language or generic structured prompts from the user describing:
- Object classes, geometries, and visual appearances (cubes, cylinders, spheres, trays, bins, etc.)
- 3D spatial locations (explicit coordinates [x, y, z] or descriptive table zones like 'center', 'left', 'near robot')
- Physical attributes (mass, friction, movable vs static)
- Roles (pick target, place target / receptacle, obstacles / distractors)

Builds, compiles, validates, and renders the complete MuJoCo XML simulation scene
and generates a synchronized JSON manifest ready for the Multi-Agent interface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

# Safe Matplotlib import for headless rendering
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "/tmp"), "matplotlib_cache"))
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = ROOT_DIR / "models" / "franka_scene.xml"
DEFAULT_OUTPUT_XML = ROOT_DIR / "models" / "generated_scene.xml"
DEFAULT_OUTPUT_MANIFEST = ROOT_DIR / "models" / "generated_scene_manifest.json"
DEFAULT_PREVIEW_IMG = ROOT_DIR / "scene_preview.png"

# Table & Workspace Boundaries (meters)
TABLE_HEIGHT = 0.40
TABLE_X_MIN, TABLE_X_MAX = 0.25, 0.85
TABLE_Y_MIN, TABLE_Y_MAX = -0.45, 0.45

# Standard Color Definitions (RGBA)
COLOR_PALETTE: Dict[str, Tuple[float, float, float, float]] = {
    "red": (0.90, 0.15, 0.15, 1.0),
    "blue": (0.15, 0.45, 0.90, 1.0),
    "green": (0.15, 0.85, 0.25, 1.0),
    "yellow": (0.95, 0.85, 0.10, 1.0),
    "orange": (0.95, 0.55, 0.10, 1.0),
    "purple": (0.65, 0.20, 0.85, 1.0),
    "cyan": (0.10, 0.85, 0.90, 1.0),
    "magenta": (0.85, 0.10, 0.85, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
    "black": (0.15, 0.15, 0.15, 1.0),
    "gray": (0.55, 0.55, 0.55, 1.0),
    "grey": (0.55, 0.55, 0.55, 1.0),
    "wood": (0.62, 0.50, 0.38, 1.0),
    "glass": (0.85, 0.90, 0.95, 0.5),
    "transparent": (0.85, 0.90, 0.95, 0.4)
}

# Qualitative Table Zones (X, Y)
TABLE_ZONES: Dict[str, Tuple[float, float]] = {
    "center": (0.55, 0.00),
    "middle": (0.55, 0.00),
    "left": (0.55, -0.22),
    "left side": (0.55, -0.25),
    "right": (0.55, 0.22),
    "right side": (0.55, 0.25),
    "near": (0.44, 0.00),
    "close": (0.44, 0.00),
    "front": (0.44, 0.00),
    "far": (0.68, 0.00),
    "back": (0.68, 0.00),
    "near left": (0.46, -0.22),
    "front left": (0.46, -0.22),
    "near right": (0.46, 0.22),
    "front right": (0.46, 0.22),
    "far left": (0.68, -0.22),
    "back left": (0.68, -0.22),
    "far right": (0.68, 0.22),
    "back right": (0.68, 0.22)
}


@dataclass
class ObjectSpec:
    """Specification of an object to instantiate in the MuJoCo simulation scene."""
    name: str
    semantic_class: str
    geom_type: str                   # 'box', 'cylinder', 'sphere', 'capsule', 'tray'
    color_name: str
    rgba: Tuple[float, float, float, float]
    pos: np.ndarray                  # [x, y, z] in meters
    size: List[float]                # geom size parameter(s)
    mass: float = 0.10               # in kg
    friction: Tuple[float, float, float] = (1.0, 0.01, 0.001)
    is_freejoint: bool = True        # whether the object can be moved
    role: str = "distractor"         # 'pick_target', 'place_target', 'obstacle', 'distractor'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "semantic_class": self.semantic_class,
            "geom_type": self.geom_type,
            "color": self.color_name,
            "rgba": list(self.rgba),
            "position": [round(float(v), 4) for v in self.pos],
            "size": [round(float(v), 4) for v in self.size],
            "mass": round(float(self.mass), 4),
            "friction": list(self.friction),
            "is_freejoint": self.is_freejoint,
            "role": self.role
        }


class ScenePromptParser:
    """Parses natural language / generic user prompts into structured object specifications."""

    def __init__(self):
        self._used_names: set = set()
        self._assigned_positions: List[np.ndarray] = []
        self.task_pairs: List[Tuple[ObjectSpec, ObjectSpec]] = []

    def split_clauses(self, text: str) -> List[str]:
        """Splits prompt by commas, semicolons, and conjunctions outside coordinate brackets."""
        clauses = []
        tokens = re.split(r"(\s+|,|;|\n)", text)
        buf = ""
        depth = 0

        for token in tokens:
            for char in token:
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth = max(0, depth - 1)

            if depth == 0 and (token in (",", ";", "\n") or token.strip().lower() in ("and", "also", "then")):
                if buf.strip():
                    clauses.append(buf.strip())
                buf = ""
            else:
                buf += token

        if buf.strip():
            clauses.append(buf.strip())

        return clauses

    def parse(self, prompt: str) -> List[ObjectSpec]:
        """
        Extracts objects, geometries, colors, and spatial locations from a prompt string.
        Supports both natural chained prompts and explicit 'Objects: ... Tasks: ...' prompts.
        """
        self._used_names.clear()
        self._assigned_positions.clear()
        self.task_pairs = []
        prompt_clean = prompt.strip()

        if not prompt_clean:
            return self._build_default_scene()

        p_lower = prompt_clean.lower()
        split_m = re.search(r"\b(tasks?|instructions?|steps?)\s*:", prompt_clean, flags=re.IGNORECASE)
        if split_m and any(w in p_lower[:split_m.start()] for w in ["objects:", "scene:", "environment:"]):
            scene_part = prompt_clean[:split_m.start()]
            tasks_part = prompt_clean[split_m.end():]
            scene_part = re.sub(r"^\s*(objects?|scene|environment)\s*:\s*", "", scene_part, flags=re.IGNORECASE)

            clauses = self.split_clauses(scene_part)
            specs: List[ObjectSpec] = []
            for idx, clause in enumerate(clauses):
                clause_str = clause.strip()
                if not clause_str or len(clause_str) < 3:
                    continue
                obj = self._parse_single_clause(clause_str, default_idx=idx + 1)
                if obj is not None:
                    obj.pos = self._resolve_position_collision(obj.pos)
                    self._assigned_positions.append(obj.pos)
                    specs.append(obj)

            if not specs:
                specs = self._build_default_scene()

            # Match task instructions with parsed objects
            task_clauses = self.split_clauses(tasks_part)
            for t_clause in task_clauses:
                pair = self._match_task_clause_to_objects(t_clause, specs)
                if pair:
                    self.task_pairs.append(pair)

            self._assign_default_roles(specs)
            return specs

        clauses = self.split_clauses(prompt_clean)
        specs: List[ObjectSpec] = []

        for idx, clause in enumerate(clauses):
            clause_str = clause.strip()
            # Skip clauses that are purely conversational
            if not clause_str or len(clause_str) < 3:
                continue
            if clause_str.lower() in ("in the scene", "on the table", "in mujoco", "please", "build me a scene"):
                continue

            obj = self._parse_single_clause(clause_str, default_idx=idx + 1)
            if obj is not None:
                # Ensure objects don't spawn overlapping
                obj.pos = self._resolve_position_collision(obj.pos)
                self._assigned_positions.append(obj.pos)
                specs.append(obj)

        if not specs:
            specs = self._build_default_scene()

        # Assign and normalize roles and canonical body names
        self._assign_default_roles(specs)

        # Detect task pairs from alternating pick / place specifications
        pick_objs = [s for s in specs if s.role == "pick_target"]
        place_objs = [s for s in specs if s.role == "place_target"]
        if pick_objs and place_objs:
            for idx, p_pick in enumerate(pick_objs):
                p_place = place_objs[idx] if idx < len(place_objs) else place_objs[-1]
                self.task_pairs.append((p_pick, p_place))

        return specs

    def _match_task_clause_to_objects(self, clause: str, specs: List[ObjectSpec]) -> Optional[Tuple[ObjectSpec, ObjectSpec]]:
        c_lower = clause.lower()
        parts = re.split(r"\b(?:place|put|drop|into)\b", c_lower, maxsplit=1)
        pick_part = parts[0]
        place_part = parts[1] if len(parts) > 1 else ""

        pick_candidate = self._find_best_matching_object(pick_part, specs, role_hint="pick")
        place_candidate = self._find_best_matching_object(place_part, specs, role_hint="place")

        if pick_candidate is not None:
            pick_candidate.role = "pick_target"
            if place_candidate is not None:
                place_candidate.role = "place_target"
                return (pick_candidate, place_candidate)
            else:
                trays = [s for s in specs if s.geom_type == "tray" or s.role == "place_target"]
                if trays:
                    trays[0].role = "place_target"
                    return (pick_candidate, trays[0])
        return None

    def _find_best_matching_object(self, text: str, specs: List[ObjectSpec], role_hint: str = "") -> Optional[ObjectSpec]:
        best_obj = None
        best_score = 0
        for obj in specs:
            score = 0
            if obj.color_name in text:
                score += 5
            if obj.semantic_class in text:
                score += 3
            if obj.geom_type in text:
                score += 2
            if obj.name in text:
                score += 10
            if role_hint == "place" and obj.geom_type == "tray":
                score += 2
            if score > best_score:
                best_score = score
                best_obj = obj
        return best_obj if best_score > 0 else None

    def _parse_single_clause(self, clause: str, default_idx: int) -> Optional[ObjectSpec]:
        c_lower = clause.lower()

        # 1. Identify Shape / Semantic Class
        geom_type = "box"
        semantic_class = "cube"
        size = [0.025, 0.025, 0.025] # 5 cm default cube

        if any(w in c_lower for w in ["tray", "bin", "basket", "receptacle", "plate", "bowl", "drop zone"]):
            geom_type = "tray"
            semantic_class = "bin"
            size = [0.065, 0.065, 0.015]
        elif any(w in c_lower for w in ["cylinder", "can", "bottle", "cup", "mug", "pillar"]):
            geom_type = "cylinder"
            semantic_class = "cylinder"
            size = [0.025, 0.040] # 5 cm diameter, 8 cm tall
        elif any(w in c_lower for w in ["sphere", "ball", "orb", "globe"]):
            geom_type = "sphere"
            semantic_class = "sphere"
            size = [0.030] # 6 cm diameter
        elif any(w in c_lower for w in ["capsule", "pill"]):
            geom_type = "capsule"
            semantic_class = "capsule"
            size = [0.022, 0.030]
        elif any(w in c_lower for w in ["cube", "block", "box", "package", "brick"]):
            geom_type = "box"
            semantic_class = "cube"
            size = [0.025, 0.025, 0.025]
        else:
            # Check if clause specifies an object at all
            if not any(col in c_lower for col in COLOR_PALETTE) and not re.search(r"\[.*?\]", clause):
                return None
            geom_type = "box"
            semantic_class = "object"
            size = [0.025, 0.025, 0.025]

        # 2. Identify Color
        color_name = "red"
        rgba = COLOR_PALETTE["red"]
        for cname, c_rgba in COLOR_PALETTE.items():
            if re.search(r"\b" + re.escape(cname) + r"\b", c_lower):
                color_name = cname
                rgba = c_rgba
                break

        # 3. Identify Position [X, Y, Z]
        pos = self._extract_position(clause, c_lower, geom_type, size, default_idx)

        # 4. Identify Mass & Friction
        mass = 0.10
        mass_match = re.search(r"(\d+\.?\d*)\s*(?:kg|kilos?|g|grams?)", c_lower)
        if mass_match:
            val = float(mass_match.group(1))
            mass = val if "kg" in mass_match.group(0) else val / 1000.0

        friction = 1.0
        if "slippery" in c_lower or "ice" in c_lower or "low friction" in c_lower:
            friction = 0.20
        elif "sticky" in c_lower or "rough" in c_lower or "high friction" in c_lower:
            friction = 1.50
        elif "friction" in c_lower:
            f_match = re.search(r"friction\s*(?:=|:|\b)\s*(\d+\.?\d*)", c_lower)
            if f_match:
                friction = float(f_match.group(1))

        # 5. Determine Role
        role = "distractor"
        if any(w in c_lower for w in ["pick target", "pickup", "primary target", "to pick", "target block", "grab", "target object", "pick"]):
            role = "pick_target"
        elif any(w in c_lower for w in ["place target", "drop target", "destination", "receptacle", "goal", "drop zone", "to place", "drop into"]):
            role = "place_target"
        elif any(w in c_lower for w in ["obstacle", "barrier", "blocker", "avoid", "hazard"]):
            role = "obstacle"
        elif geom_type == "tray":
            role = "place_target"

        # Unique name
        base_name = f"{color_name}_{semantic_class}"
        name = base_name
        counter = 1
        while name in self._used_names:
            counter += 1
            name = f"{base_name}_{counter}"
        self._used_names.add(name)

        # Trays and receptacles are typically fixed or heavy
        is_free = True
        if geom_type == "tray" and ("fixed" in c_lower or "static" in c_lower or role == "place_target"):
            is_free = False

        return ObjectSpec(
            name=name,
            semantic_class=semantic_class,
            geom_type=geom_type,
            color_name=color_name,
            rgba=rgba,
            pos=pos,
            size=size,
            mass=mass,
            friction=(friction, 0.01, 0.001),
            is_freejoint=is_free,
            role=role
        )

    def _extract_position(
        self,
        clause: str,
        c_lower: str,
        geom_type: str,
        size: List[float],
        default_idx: int
    ) -> np.ndarray:
        """Extracts 3D coordinates from brackets, parentheses, or named table zones."""
        # 1. Direct coordinates: [x, y, z] or (x, y, z)
        coord_pattern = r"\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*(?:,\s*(-?\d+\.?\d*))?\s*\]?"
        match = re.search(coord_pattern, clause)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3)) if match.group(3) is not None else None
            # Clamp X and Y to reachable table area
            x = np.clip(x, TABLE_X_MIN, TABLE_X_MAX)
            y = np.clip(y, TABLE_Y_MIN, TABLE_Y_MAX)
        else:
            # 2. Named spatial zones
            zone_found = False
            x, y = 0.55, 0.00
            for zone_name, (zx, zy) in sorted(TABLE_ZONES.items(), key=lambda item: -len(item[0])):
                if zone_name in c_lower:
                    x, y = zx, zy
                    zone_found = True
                    break

            if not zone_found:
                # Default geometric distribution based on index
                preset_positions = [
                    (0.48, -0.15),
                    (0.65, 0.22),
                    (0.55, 0.00),
                    (0.46, 0.18),
                    (0.68, -0.18)
                ]
                pos_pair = preset_positions[(default_idx - 1) % len(preset_positions)]
                x, y = pos_pair

            z = None

        # 3. Calculate safe rest height z directly on top of table surface (z = 0.40)
        half_h = size[2] if geom_type == "box" else (size[1] if geom_type in ("cylinder", "capsule") else size[0])
        z_rest = TABLE_HEIGHT + half_h + 0.001

        if z is None or z < TABLE_HEIGHT:
            z = z_rest

        return np.array([x, y, z], dtype=np.float64)

    def _resolve_position_collision(self, pos: np.ndarray, min_dist: float = 0.08) -> np.ndarray:
        """Slightly shifts coordinates if another object already occupies the exact location."""
        new_pos = pos.copy()
        for existing in self._assigned_positions:
            dist = np.linalg.norm(new_pos[:2] - existing[:2])
            if dist < min_dist:
                # Shift along Y or X to create clear separation
                if new_pos[1] >= 0:
                    new_pos[1] += (min_dist - dist) + 0.03
                else:
                    new_pos[1] -= (min_dist - dist) + 0.03
                new_pos[0] = np.clip(new_pos[0], TABLE_X_MIN, TABLE_X_MAX)
                new_pos[1] = np.clip(new_pos[1], TABLE_Y_MIN, TABLE_Y_MAX)
        return new_pos

    def _assign_default_roles(self, specs: List[ObjectSpec]) -> None:
        """Ensures at least one pick target and one place target are designated, and names match pipeline expectations."""
        pick_objs = [s for s in specs if s.role == "pick_target"]
        if not pick_objs:
            # Designate first movable object as pick target
            for s in specs:
                if s.geom_type != "tray":
                    s.role = "pick_target"
                    pick_objs.append(s)
                    break
        if not pick_objs and specs:
            specs[0].role = "pick_target"
            pick_objs.append(specs[0])

        # If only 1 pick target in the entire scene, use canonical name 'target_block' for test compatibility
        if len(pick_objs) == 1:
            pick_objs[0].name = "target_block"

        place_objs = [s for s in specs if s.role == "place_target"]
        if not place_objs:
            # Prefer trays/bins, otherwise last object
            for s in reversed(specs):
                if s not in pick_objs and s.geom_type == "tray":
                    s.role = "place_target"
                    place_objs.append(s)
                    break
            if not place_objs and len(specs) > 1:
                specs[-1].role = "place_target"
                place_objs.append(specs[-1])

        # If only 1 place target in the entire scene, use canonical name 'target_bin'
        if len(place_objs) == 1:
            place_objs[0].name = "target_bin"

    def _build_default_scene(self) -> List[ObjectSpec]:
        """Provides a standard rich manipulation scene with target block, receptacle, and obstacle."""
        return [
            ObjectSpec(
                name="target_block",
                semantic_class="cube",
                geom_type="box",
                color_name="red",
                rgba=COLOR_PALETTE["red"],
                pos=np.array([0.48, -0.15, 0.425]),
                size=[0.025, 0.025, 0.025],
                mass=0.15,
                friction=(1.0, 0.01, 0.001),
                is_freejoint=True,
                role="pick_target"
            ),
            ObjectSpec(
                name="target_bin",
                semantic_class="bin",
                geom_type="tray",
                color_name="green",
                rgba=COLOR_PALETTE["green"],
                pos=np.array([0.65, 0.22, 0.415]),
                size=[0.065, 0.065, 0.015],
                mass=0.50,
                friction=(1.0, 0.01, 0.001),
                is_freejoint=False,
                role="place_target"
            ),
            ObjectSpec(
                name="blue_cylinder_obstacle",
                semantic_class="cylinder",
                geom_type="cylinder",
                color_name="blue",
                rgba=COLOR_PALETTE["blue"],
                pos=np.array([0.55, 0.05, 0.440]),
                size=[0.025, 0.040],
                mass=0.20,
                friction=(1.0, 0.01, 0.001),
                is_freejoint=True,
                role="obstacle"
            )
        ]


class MuJoCoSceneBuilder:
    """Compiles objects into a valid, physically verified MuJoCo XML scene."""

    def __init__(self, template_xml_path: Optional[Path] = None):
        self.template_path = template_xml_path or DEFAULT_TEMPLATE_PATH

    def generate_scene_xml(self, objects: List[ObjectSpec]) -> str:
        """Constructs the complete MuJoCo XML string."""
        with open(self.template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # 1. Generate Material Assets
        material_lines = []
        for obj in objects:
            mat_name = f"mat_{obj.name}"
            r, g, b, a = obj.rgba
            material_lines.append(
                f'    <material name="{mat_name}" rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"/>'
            )
        materials_xml = "\n".join(material_lines)

        # Insert materials right before </asset>
        if "</asset>" in template:
            template = template.replace("</asset>", f"{materials_xml}\n  </asset>")

        # 2. Generate Object Bodies
        body_lines = []
        free_joint_count = 0
        qpos_free_joints = []

        for obj in objects:
            mat_name = f"mat_{obj.name}"
            px, py, pz = obj.pos
            mu1, mu2, mu3 = obj.friction

            if obj.geom_type == "tray":
                # Build open tray / bin container (base + 4 rims)
                sx, sy, sz = obj.size
                wall_th = 0.008
                rim_h = 0.03
                tray_body = [
                    f'    <!-- Receptacle / Tray: {obj.name} -->',
                    f'    <body name="{obj.name}" pos="{px:.4f} {py:.4f} {pz:.4f}">'
                ]
                if obj.is_freejoint:
                    tray_body.append(f'      <freejoint name="{obj.name}_free"/>')
                    free_joint_count += 1
                    qpos_free_joints.extend([px, py, pz, 1.0, 0.0, 0.0, 0.0])

                # Bottom surface
                tray_body.append(
                    f'      <geom name="{obj.name}_base" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
                    f'material="{mat_name}" mass="{obj.mass * 0.5:.3f}" friction="{mu1} {mu2} {mu3}"/>'
                )
                # 4 surrounding rim walls
                tray_body.append(
                    f'      <geom name="{obj.name}_wall_n" type="box" size="{sx:.4f} {wall_th:.4f} {rim_h:.4f}" '
                    f'pos="0 {sy:.4f} {rim_h:.4f}" material="{mat_name}"/>'
                )
                tray_body.append(
                    f'      <geom name="{obj.name}_wall_s" type="box" size="{sx:.4f} {wall_th:.4f} {rim_h:.4f}" '
                    f'pos="0 {-sy:.4f} {rim_h:.4f}" material="{mat_name}"/>'
                )
                tray_body.append(
                    f'      <geom name="{obj.name}_wall_e" type="box" size="{wall_th:.4f} {sy:.4f} {rim_h:.4f}" '
                    f'pos="{sx:.4f} 0 {rim_h:.4f}" material="{mat_name}"/>'
                )
                tray_body.append(
                    f'      <geom name="{obj.name}_wall_w" type="box" size="{wall_th:.4f} {sy:.4f} {rim_h:.4f}" '
                    f'pos="{-sx:.4f} 0 {rim_h:.4f}" material="{mat_name}"/>'
                )
                tray_body.append('    </body>')
                body_lines.append("\n".join(tray_body))
            else:
                # Standard primitive body
                size_str = " ".join(f"{s:.4f}" for s in obj.size)
                body_xml = [
                    f'    <!-- Simulation Object: {obj.name} ({obj.role}) -->',
                    f'    <body name="{obj.name}" pos="{px:.4f} {py:.4f} {pz:.4f}">'
                ]
                if obj.is_freejoint:
                    body_xml.append(f'      <freejoint name="{obj.name}_free"/>')
                    free_joint_count += 1
                    qpos_free_joints.extend([px, py, pz, 1.0, 0.0, 0.0, 0.0])

                body_xml.append(
                    f'      <geom name="{obj.name}_geom" type="{obj.geom_type}" size="{size_str}" '
                    f'material="{mat_name}" mass="{obj.mass:.3f}" friction="{mu1} {mu2} {mu3}"/>'
                )
                body_xml.append('    </body>')
                body_lines.append("\n".join(body_xml))

        new_bodies_xml = "\n".join(body_lines)

        # Replace existing placeholder target_block if present, or append before </worldbody>
        if '<body name="target_block"' in template:
            # Remove default single target_block body
            template = re.sub(
                r'<body name="target_block".*?</body>',
                new_bodies_xml,
                template,
                flags=re.DOTALL
            )
        else:
            template = template.replace("</worldbody>", f"{new_bodies_xml}\n  </worldbody>")

        # 3. Synchronize Keyframe qpos with exact free joints count
        # Franka has 7 joints: 0 -0.3 0 -2.2 0 1.9 0
        robot_qpos = "0 -0.3 0 -2.2 0 1.9 0"
        if qpos_free_joints:
            free_joints_str = "   " + " ".join(f"{v:.4f}" for v in qpos_free_joints)
            full_qpos = f"{robot_qpos}{free_joints_str}"
        else:
            full_qpos = robot_qpos

        template = re.sub(
            r'qpos=".*?"',
            f'qpos="{full_qpos}"',
            template
        )

        return template

    def validate_and_compile(self, xml_content: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Validates XML syntax, compiles model, and performs physical stability simulation."""
        try:
            model = mujoco.MjModel.from_xml_string(xml_content)
            data = mujoco.MjData(model)
        except Exception as e:
            raise RuntimeError(f"MuJoCo compilation failed on generated XML:\n{e}")

        # Initialize to home keyframe
        home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key)
        else:
            mujoco.mj_resetData(model, data)

        # Run 25 simulation steps to verify physical equilibrium
        for _ in range(25):
            mujoco.mj_step(model, data)

        if np.isnan(data.qpos).any() or np.isinf(data.qpos).any():
            raise RuntimeError("Simulation instability detected: NaN/Inf encountered in qpos.")

        return model, data

    def render_preview(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_path: Path
    ) -> None:
        """Renders an offscreen image preview using the overhead camera and saves to disk."""
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="overhead_cam")
        img = renderer.render()

        plt.imsave(str(output_path), img)
        renderer.close()

    def generate_manifest(
        self,
        objects: List[ObjectSpec],
        output_path: Path,
        prompt: str,
        task_pairs: Optional[List[Tuple[ObjectSpec, ObjectSpec]]] = None
    ) -> Dict[str, Any]:
        """Outputs a structured JSON manifest detailing objects and Multi-Agent task context."""
        pick_targets = [o for o in objects if o.role == "pick_target"]
        place_targets = [o for o in objects if o.role == "place_target"]

        primary_pick = pick_targets[0] if pick_targets else objects[0]
        primary_place = place_targets[0] if place_targets else (objects[1] if len(objects) > 1 else objects[0])

        subtasks = []
        if task_pairs and len(task_pairs) > 0:
            for idx, (p_pick, p_place) in enumerate(task_pairs):
                subtasks.append({
                    "task_id": f"subtask_{idx+1}_{p_pick.name}_to_{p_place.name}",
                    "step_index": idx + 1,
                    "user_prompt": f"Pick up {p_pick.name} and place in {p_place.name}",
                    "pick_target": {
                        "name": p_pick.name,
                        "class": p_pick.semantic_class,
                        "initial_position_hint": [round(float(v), 4) for v in p_pick.pos],
                        "mass": p_pick.mass,
                        "friction": p_pick.friction[0]
                    },
                    "place_target": {
                        "name": p_place.name,
                        "class": p_place.semantic_class,
                        "target_position": [round(float(v), 4) for v in p_place.pos],
                        "tolerance": 0.03
                    },
                    "operational_context": {
                        "fragility": 0.50 if p_pick.color_name != "glass" else 0.90,
                        "speed_priority": 0.60,
                        "safety_priority": 0.85,
                        "task_code": float(idx),
                        "crowd_density": 0.0
                    }
                })
        else:
            for idx, p_pick in enumerate(pick_targets):
                if place_targets:
                    p_place = place_targets[idx] if idx < len(place_targets) else place_targets[-1]
                else:
                    p_place = primary_place
                subtasks.append({
                    "task_id": f"subtask_{idx+1}_{p_pick.name}_to_{p_place.name}",
                    "step_index": idx + 1,
                    "user_prompt": f"Pick up {p_pick.name} and place in {p_place.name}",
                    "pick_target": {
                        "name": p_pick.name,
                        "class": p_pick.semantic_class,
                        "initial_position_hint": [round(float(v), 4) for v in p_pick.pos],
                        "mass": p_pick.mass,
                        "friction": p_pick.friction[0]
                    },
                    "place_target": {
                        "name": p_place.name,
                        "class": p_place.semantic_class,
                        "target_position": [round(float(v), 4) for v in p_place.pos],
                        "tolerance": 0.03
                    },
                    "operational_context": {
                        "fragility": 0.50 if p_pick.color_name != "glass" else 0.90,
                        "speed_priority": 0.60,
                        "safety_priority": 0.85,
                        "task_code": float(idx),
                        "crowd_density": 0.0
                    }
                })

        manifest = {
            "scene_metadata": {
                "created_at": time.time(),
                "prompt": prompt,
                "num_objects": len(objects),
                "num_subtasks": len(subtasks),
                "table_bounds": {
                    "x": [TABLE_X_MIN, TABLE_X_MAX],
                    "y": [TABLE_Y_MIN, TABLE_Y_MAX],
                    "z_surface": TABLE_HEIGHT
                }
            },
            "objects": [obj.to_dict() for obj in objects],
            "multi_agent_task_format": {
                "task_id": f"task_{primary_pick.name}_to_{primary_place.name}" if len(subtasks) == 1 else "multi_object_pick_and_place",
                "user_prompt": prompt,
                "task_type": "pick_and_place",
                "pick_target": subtasks[0]["pick_target"] if subtasks else {
                    "name": primary_pick.name,
                    "class": primary_pick.semantic_class,
                    "initial_position_hint": [round(float(v), 4) for v in primary_pick.pos],
                    "mass": primary_pick.mass,
                    "friction": primary_pick.friction[0]
                },
                "place_target": subtasks[0]["place_target"] if subtasks else {
                    "name": primary_place.name,
                    "class": primary_place.semantic_class,
                    "target_position": [round(float(v), 4) for v in primary_place.pos],
                    "tolerance": 0.03
                },
                "operational_context": subtasks[0]["operational_context"] if subtasks else {
                    "fragility": 0.50 if primary_pick.color_name != "glass" else 0.90,
                    "speed_priority": 0.60,
                    "safety_priority": 0.85,
                    "task_code": 0.0,
                    "crowd_density": 0.0
                },
                "subtasks": subtasks
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


def print_scene_summary(objects: List[ObjectSpec], xml_path: Path, manifest_path: Path, preview_path: Path):
    """Prints a formatted summary table of all generated objects and files."""
    print("\n" + "=" * 90)
    print("                     MUJOCO SIMULATION SCENE GENERATION COMPLETE")
    print("=" * 90)
    print(f"{'Object Name':<25} | {'Shape':<10} | {'Color':<8} | {'Position [x, y, z]':<22} | {'Role':<14} | {'Mass (kg)':<8}")
    print("-" * 90)
    for obj in objects:
        pos_str = f"[{obj.pos[0]:.2f}, {obj.pos[1]:.2f}, {obj.pos[2]:.2f}]"
        print(f"{obj.name:<25} | {obj.geom_type:<10} | {obj.color_name:<8} | {pos_str:<22} | {obj.role:<14} | {obj.mass:<8.2f}")
    print("-" * 90)
    print("\nGenerated Output Assets:")
    print(f"  [1] MuJoCo Scene XML : {xml_path.resolve()}")
    print(f"  [2] JSON Manifest    : {manifest_path.resolve()}")
    print(f"  [3] Visual Preview   : {preview_path.resolve()}")
    print("\nReady for immediate ingestion by:")
    print("  -> MultiAgentInterface.parse_task_message()")
    print("  -> SpatialReasoningLayer.process_frame()")
    print("  -> IntegratedSpatialCCDMPipeline.step()")
    print("  -> FrankaNMPCController (CasADi + IPOPT)")
    print("=" * 90)


def build_scene_from_prompt(
    prompt: str,
    output_xml: Path = DEFAULT_OUTPUT_XML,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    preview_img: Path = DEFAULT_PREVIEW_IMG,
    update_default_model: bool = False
) -> Tuple[mujoco.MjModel, mujoco.MjData, List[ObjectSpec]]:
    """Programmatic API to build a complete scene from a string prompt."""
    parser = ScenePromptParser()
    objects = parser.parse(prompt)

    builder = MuJoCoSceneBuilder()
    xml_str = builder.generate_scene_xml(objects)

    # Validate and simulate equilibrium
    model, data = builder.validate_and_compile(xml_str)

    # Save XML
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_xml, "w", encoding="utf-8") as f:
        f.write(xml_str)

    if update_default_model:
        with open(DEFAULT_TEMPLATE_PATH, "w", encoding="utf-8") as f:
            f.write(xml_str)

    # Save JSON manifest with subtask sequence
    builder.generate_manifest(objects, output_manifest, prompt, task_pairs=parser.task_pairs)

    # Render offscreen preview
    builder.render_preview(model, data, preview_img)

    return model, data, objects


def main():
    arg_parser = argparse.ArgumentParser(
        description="Interactive MuJoCo Simulation Scene Builder for Robotic Manipulation."
    )
    arg_parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Generic natural language or coordinate-based prompt to build the scene."
    )
    arg_parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_XML),
        help="Target MuJoCo XML scene output file path."
    )
    arg_parser.add_argument(
        "--manifest", "-m",
        type=str,
        default=str(DEFAULT_OUTPUT_MANIFEST),
        help="Target JSON scene manifest output file path."
    )
    arg_parser.add_argument(
        "--preview",
        type=str,
        default=str(DEFAULT_PREVIEW_IMG),
        help="Target output image path for the overhead camera rendered snapshot."
    )
    arg_parser.add_argument(
        "--update-default",
        action="store_true",
        help="Update the canonical models/franka_scene.xml model with the generated scene."
    )
    arg_parser.add_argument(
        "--interactive-viewer",
        action="store_true",
        help="Launch MuJoCo 3D interactive viewer window to inspect the generated scene."
    )
    arg_parser.add_argument(
        "--demo",
        action="store_true",
        help="Build a rich multi-object benchmark scene demonstrating pick, place, and obstacle avoidance."
    )
    arg_parser.add_argument(
        "--test-pipeline",
        action="store_true",
        help="Execute an end-to-end test step through Spatial Reasoning, CCDM, and NMPC on the generated scene."
    )

    args = arg_parser.parse_args()

    # If no prompt provided via CLI, interactively ask the user
    user_prompt = args.prompt
    if user_prompt is None and not args.demo:
        print("\n" + "=" * 80)
        print("          MUJOCO SIMULATION SCENE BUILDER (SPATIAL REASONING LAYER)          ")
        print("=" * 80)
        print("Describe the scene you want to build in MuJoCo.")
        print("You can specify object shapes, colors, coordinates, or spatial zones.")
        print("\nSupported shapes: cube / block, cylinder / can / cup, sphere / ball, tray / bin")
        print("Supported colors: red, blue, green, yellow, orange, cyan, purple, gray, wood, glass")
        print("Supported zones : center, left side, right side, near robot, drop target, etc.")
        print("\nExamples:")
        print("  1) 'Place a red cube at [0.5, 0.0, 0.425], a blue cylinder at [0.45, -0.2, 0.425], and a green tray at [0.65, 0.25, 0.425]'")
        print("  2) 'Put a yellow block on the left side of the table, a blue cup near the center, and a red bin on the right'")
        print("  3) 'Pick target: red cube at [0.48, -0.12, 0.425], drop bin: green receptacle at [0.65, 0.22, 0.425], obstacle: gray cylinder in between'")
        print("-" * 80)
        try:
            user_prompt = input("\nEnter scene description prompt (or press Enter for default scene): ").strip()
        except EOFError:
            user_prompt = ""

    if args.demo:
        user_prompt = (
            "Place a red cube pick target at [0.48, -0.15, 0.425], a green tray receptacle at [0.65, 0.22, 0.415], "
            "and a blue cylinder obstacle at [0.55, 0.05, 0.440]"
        )

    out_xml = Path(args.output)
    out_manifest = Path(args.manifest)
    out_preview = Path(args.preview)

    print(f"\n[SceneBuilder] Parsing prompt and synthesizing MuJoCo scene...")
    model, data, objects = build_scene_from_prompt(
        prompt=user_prompt or "",
        output_xml=out_xml,
        output_manifest=out_manifest,
        preview_img=out_preview,
        update_default_model=args.update_default
    )

    print_scene_summary(objects, out_xml, out_manifest, out_preview)

    if args.test_pipeline:
        print("\n" + "=" * 90)
        print("          TESTING 4-LAYER PIPELINE INTEGRATION ON GENERATED SCENE           ")
        print("=" * 90)
        try:
            from spatial_layer.integrated_pipeline import IntegratedSpatialCCDMPipeline

            with open(out_manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            test_pipeline = IntegratedSpatialCCDMPipeline(model, data, enable_nmpc=True)
            test_pipeline.set_task_from_multi_agent(manifest_data["multi_agent_task_format"])

            pipe_out = test_pipeline.step(execute_nmpc=True)

            print("[Pipeline Execution Verified Successfully]")
            print(f"  * Task ID                : {pipe_out.task_id}")
            print(f"  * Initial Object Locked  : {pipe_out.initial_object_location}")
            print(f"  * End-Effector Pose      : {pipe_out.end_effector_position}")
            print(f"  * Relative Grasp Dist    : {pipe_out.relative_distance_to_grasp:.4f} m")
            print(f"  * Active Task Phase      : {pipe_out.phase}")
            print(f"  * CCDM Velocity Limits   : v_max = {pipe_out.physical_constraints.velocity_max[:3]}")
            print(f"  * CCDM Dynamic Q_pos     : {pipe_out.nmpc_weights.Q_pos}")
            if pipe_out.torque_command is not None:
                tau_str = " ".join(f"{t:.2f}" for t in pipe_out.torque_command)
                print(f"  * NMPC Torque tau_apply  : [{tau_str}] N*m")
            print(f"  * Total Latency          : {pipe_out.timings_ms.get('total_pipeline_ms', 0):.2f} ms")
            print("=" * 90)
        except Exception as e:
            print(f"Pipeline verification failed: {e}")

    if args.interactive_viewer:
        print("\nLaunching MuJoCo 3D interactive viewer (close window to exit)...")
        try:
            import mujoco.viewer
            mujoco.viewer.launch(model, data)
        except Exception as e:
            print(f"Interactive viewer could not be opened: {e}")


if __name__ == "__main__":
    main()

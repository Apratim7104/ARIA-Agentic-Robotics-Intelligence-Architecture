"""
tests/test_scene_builder.py

Unit and integration tests for the MuJoCo Simulation Scene Builder.
Verifies prompt parsing, geometric grounding, XML generation, and MuJoCo compilation.
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import mujoco

from scene_builder import (
    ScenePromptParser,
    MuJoCoSceneBuilder,
    build_scene_from_prompt,
    ObjectSpec
)


def test_prompt_parser_explicit_coordinates():
    """Verify parser extracts explicit [x, y, z] coordinates, shapes, colors, and masses."""
    parser = ScenePromptParser()
    prompt = (
        "Place a red cube pick target at [0.48, -0.15, 0.425], "
        "a green tray receptacle at [0.65, 0.22, 0.415] with mass 0.4kg, "
        "and a blue cylinder obstacle at [0.55, 0.05, 0.440]"
    )
    objects = parser.parse(prompt)

    assert len(objects) == 3
    # Check Pick Target
    pick = [o for o in objects if o.role == "pick_target"][0]
    assert pick.color_name == "red"
    assert pick.geom_type == "box"
    assert np.allclose(pick.pos[:2], [0.48, -0.15], atol=1e-3)
    assert pick.name == "target_block"

    # Check Place Target
    place = [o for o in objects if o.role == "place_target"][0]
    assert place.color_name == "green"
    assert place.geom_type == "tray"
    assert np.allclose(place.pos[:2], [0.65, 0.22], atol=1e-3)
    assert place.mass == 0.4

    # Check Obstacle
    obs = [o for o in objects if o.role == "obstacle" or o.role == "distractor"][0]
    assert obs.color_name == "blue"
    assert obs.geom_type == "cylinder"
    assert np.allclose(obs.pos[:2], [0.55, 0.05], atol=1e-3)


def test_prompt_parser_spatial_zones():
    """Verify parser resolves descriptive spatial keywords (center, left side, right side)."""
    parser = ScenePromptParser()
    prompt = "Put a yellow cube on the left side of the table, a blue cup near the center, and a red tray on the right"
    objects = parser.parse(prompt)

    assert len(objects) == 3

    # Left side should have negative y
    left_obj = [o for o in objects if o.color_name == "yellow"][0]
    assert left_obj.pos[1] < -0.15

    # Center should have y close to 0.0
    center_obj = [o for o in objects if o.color_name == "blue"][0]
    assert abs(center_obj.pos[1]) < 0.10

    # Right side should have positive y
    right_obj = [o for o in objects if o.color_name == "red"][0]
    assert right_obj.pos[1] > 0.15


def test_scene_builder_xml_compilation_and_stability():
    """Verify that generated XML compiles in MuJoCo and simulates without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        xml_file = tmp_path / "test_scene.xml"
        manifest_file = tmp_path / "test_manifest.json"
        preview_file = tmp_path / "test_preview.png"

        prompt = (
            "Place a red cube at [0.5, 0.0, 0.425], "
            "a cyan ball at [0.45, -0.18, 0.43], "
            "and a green bin at [0.65, 0.25, 0.415]"
        )

        model, data, objects = build_scene_from_prompt(
            prompt=prompt,
            output_xml=xml_file,
            output_manifest=manifest_file,
            preview_img=preview_file
        )

        assert xml_file.exists()
        assert manifest_file.exists()
        assert preview_file.exists()

        # Check MuJoCo model
        assert model.nbody >= 14 # Robot bodies + 3 new bodies
        assert not np.isnan(data.qpos).any()

        # Step simulation forward for 50 steps
        for _ in range(50):
            mujoco.mj_step(model, data)

        assert not np.isnan(data.qpos).any()


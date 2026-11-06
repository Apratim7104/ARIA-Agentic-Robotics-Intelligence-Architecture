"""Workspace boundary, fast numerical IK, joint limits, and difficulty tests."""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np
import pytest

from spatial_layer import SpatialAnalyzer, SpatialReasoningLayer, load_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "franka_scene.xml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default_config.yaml")


@pytest.fixture(scope="module")
def env():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    cfg = load_config(CONFIG_PATH)
    analyzer = SpatialAnalyzer(model, cfg)
    return model, data, cfg, analyzer


def test_reachable_target_within_workspace(env):
    model, data, _, analyzer = env
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    p_ee = data.xpos[ee_id].copy()
    base_pos = data.xpos[base_id].copy()

    # Target near current EE pose (well inside [0.15, 0.85] m radial reach)
    target = p_ee + np.array([0.04, -0.03, -0.05])
    analysis = analyzer.analyze(target, p_ee, base_pos, data.qpos, confidence=0.95, cls="cube")

    assert analysis.workspace_valid is True
    assert analysis.ik.success is True
    assert analysis.ik.limits_ok is True
    assert analysis.ik.iterations <= 5
    assert analysis.ik.time_ms < 0.8, f"IK took {analysis.ik.time_ms:.3f} ms (>= 0.8 ms)"
    assert analysis.reachable is True
    assert 0.0 <= analysis.difficulty <= 1.0


def test_out_of_workspace_targets(env):
    model, data, _, analyzer = env
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    p_ee = data.xpos[ee_id].copy()
    base_pos = data.xpos[base_id].copy()

    # Too far (> 0.85 m from base)
    far_target = np.array([1.45, 0.0, 0.425])
    far_res = analyzer.analyze(far_target, p_ee, base_pos, data.qpos, confidence=0.95, cls="cube")
    assert far_res.workspace_valid is False
    assert far_res.ik.success is False
    assert far_res.ik.limits_ok is True  # clamped joint configuration still respects limits
    assert far_res.reachable is False
    assert far_res.flags[0] == 0.0 and far_res.flags[1] == 0.0

    # Too close (< 0.15 m from base at [0, 0, 0.4])
    close_target = np.array([0.05, 0.0, 0.42])
    close_res = analyzer.analyze(close_target, p_ee, base_pos, data.qpos, confidence=0.95, cls="cube")
    assert close_res.workspace_valid is False
    assert close_res.reachable is False


def test_pipeline_outside_workspace_behavior():
    """Full Layer-2 pipeline flags workspace_valid=0 and ik_feasible=0 when object is outside reach."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Place target block inside camera FOV but beyond reach_max (0.85 m):
    # Distance from base (0, 0, 0.4) is sqrt(0.85^2 + 0.25^2 + 0.025^2) = 0.886 m > 0.85 m.
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    qadr = model.jnt_qposadr[jnt_id]
    data.qpos[qadr:qadr + 3] = [0.85, 0.25, 0.425]
    mujoco.mj_forward(model, data)

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    gt_pos = data.geom_xpos[geom_id].copy()

    layer = SpatialReasoningLayer(model, data, CONFIG_PATH)
    try:
        layer.sync_state()
        uv, _ = layer.transformer.world_to_pixel(gt_pos)
        msg = {
            "timestamp": time.time(),
            "primary_target": {
                "id": 0, "class": "cube", "confidence": 0.95,
                "bbox": [float(uv[0] - 12), float(uv[1] - 12),
                         float(uv[0] + 12), float(uv[1] + 12)],
            },
            "secondary_objects": [],
        }
        res = layer.process_frame(msg, force_depth_refresh=True)
        assert res.valid is True
        assert res.analysis is not None
        assert res.analysis.workspace_valid is False
        assert res.features[16] == 0.0  # workspace_valid flag in feature vector
        assert res.features[17] == 0.0  # ik_feasible flag in feature vector
    finally:
        layer.close()

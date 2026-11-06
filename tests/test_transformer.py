"""Geometric accuracy tests for 2D->3D deprojection and world transformation."""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np
import pytest

from spatial_layer import CoordinateTransformer, SpatialReasoningLayer, load_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "franka_scene.xml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default_config.yaml")


@pytest.fixture(scope="module")
def sim_env():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    cfg = load_config(CONFIG_PATH)
    return model, data, cfg


def _set_block_pose(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray) -> np.ndarray:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    qadr = model.jnt_qposadr[jnt_id]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    return data.geom_xpos[geom_id].copy()


def test_pixel_world_roundtrip(sim_env):
    _, _, cfg = sim_env
    tf = CoordinateTransformer(cfg["camera"]["intrinsics"], cfg["camera"]["extrinsics"])
    pts_world = np.array([
        [0.50, 0.00, 0.425],
        [0.65, 0.12, 0.425],
        [0.45, -0.15, 0.50],
        [0.72, -0.08, 0.425],
    ])
    for p_w in pts_world:
        uv, d = tf.world_to_pixel(p_w)
        p_rec = tf.pixel_to_world(uv[0], uv[1], d)
        np.testing.assert_allclose(p_rec, p_w, atol=1e-6)


def test_batch_deprojection_matches_single(sim_env):
    _, _, cfg = sim_env
    tf = CoordinateTransformer(cfg["camera"]["intrinsics"], cfg["camera"]["extrinsics"])
    uv = np.array([[160.0, 120.0], [120.0, 90.0], [210.0, 155.0], [85.5, 180.25]])
    depths = np.array([0.75, 0.80, 0.72, 0.68])
    batch_pts = tf.batch_pixel_to_world(uv, depths)
    single_pts = np.stack([tf.pixel_to_world(u, v, d) for (u, v), d in zip(uv, depths)])
    np.testing.assert_allclose(batch_pts, single_pts, atol=1e-9)


@pytest.mark.parametrize(
    "block_pos",
    [
        np.array([0.50, 0.00, 0.425]),
        np.array([0.60, 0.00, 0.425]),
        np.array([0.55, 0.12, 0.425]),
        np.array([0.65, -0.10, 0.425]),
        np.array([0.48, -0.08, 0.425]),
    ],
)
def test_deprojected_3d_error_under_3mm(sim_env, block_pos):
    """Deprojected 3D world coordinate error must be < 3 mm compared to data.geom_xpos."""
    model, data, _ = sim_env
    gt_geom_pos = _set_block_pose(model, data, block_pos)

    layer = SpatialReasoningLayer(model, data, CONFIG_PATH)
    try:
        layer.sync_state()
        uv, _ = layer.transformer.world_to_pixel(gt_geom_pos)
        half_w, half_h = 16.0, 16.0
        msg = {
            "timestamp": time.time(),
            "primary_target": {
                "id": 0,
                "class": "cube",
                "confidence": 0.96,
                "bbox": [
                    float(uv[0] - half_w),
                    float(uv[1] - half_h),
                    float(uv[0] + half_w),
                    float(uv[1] + half_h),
                ],
            },
            "secondary_objects": [],
        }
        result = layer.process_frame(msg, force_depth_refresh=True)
        assert result.valid, f"Frame invalid: {result.status}"
        assert result.p_obj_world is not None
        error_m = float(np.linalg.norm(result.p_obj_world - gt_geom_pos))
        assert error_m < 0.003, f"3D error {error_m * 1e3:.3f} mm exceeds 3.0 mm threshold"
    finally:
        layer.close()


"""Feature vector dimension, bounds [-1, 1], dtype, and packing order tests."""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np
import pytest

from spatial_layer import FEATURE_DIM, FeatureEncoder, SpatialReasoningLayer, load_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "franka_scene.xml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default_config.yaml")


@pytest.fixture
def encoder():
    return FeatureEncoder(load_config(CONFIG_PATH))


def test_feature_vector_layout_and_scaling(encoder: FeatureEncoder):
    p_obj = np.array([0.50, -0.20, 0.425])
    p_ee = np.array([0.46, 0.00, 0.80])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    dists = np.array([0.45, 0.21, 0.375])
    grasp_off = np.array([0.0, 0.0, 0.02])
    flags = [True, True, True, True]
    diff = 0.25
    conf = 0.95

    vec = encoder.encode(p_obj, p_ee, quat, dists, grasp_off, flags, diff, conf)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (30,)
    assert vec.dtype == np.float32
    assert np.all(vec >= -1.0) and np.all(vec <= 1.0)

    np.testing.assert_allclose(vec[0:3], p_obj / 2.0, atol=1e-6)
    np.testing.assert_allclose(vec[3:6], p_ee / 2.0, atol=1e-6)
    np.testing.assert_allclose(vec[6:10], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vec[10:13], dists / 1.5, atol=1e-6)
    np.testing.assert_allclose(vec[13:16], grasp_off / 0.5, atol=1e-6)
    np.testing.assert_allclose(vec[16:20], [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    assert pytest.approx(float(vec[20]), abs=1e-6) == diff
    assert pytest.approx(float(vec[21]), abs=1e-6) == conf
    np.testing.assert_array_equal(vec[22:30], np.zeros(8, dtype=np.float32))


def test_feature_vector_clamping_extreme_values(encoder: FeatureEncoder):
    vec = encoder.encode(
        p_obj=[10.0, -10.0, 5.0],
        p_ee=[-8.0, 9.0, 12.0],
        obj_quat=[0.0, 2.0, 0.0, 0.0],
        distances=[9.0, 5.0, 4.0],
        grasp_offset=[2.0, -2.0, 1.5],
        flags=[0, 1, 0, 1],
        difficulty=2.5,
        confidence=1.2,
    )
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(vec >= -1.0) and np.all(vec <= 1.0)
    np.testing.assert_array_equal(vec[0:3], [1.0, -1.0, 1.0])
    np.testing.assert_array_equal(vec[3:6], [-1.0, 1.0, 1.0])
    np.testing.assert_array_equal(vec[16:20], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(vec[22:30], np.zeros(8, dtype=np.float32))


def test_async_worker_double_buffer():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    with SpatialReasoningLayer(model, data, CONFIG_PATH) as layer:
        layer.sync_state()
        msg = {
            "timestamp": time.time(),
            "primary_target": {"id": 0, "class": "cube", "confidence": 0.95,
                               "bbox": [110, 105, 140, 135]},
            "secondary_objects": [],
        }
        layer.submit_detection(msg)
        deadline = time.perf_counter() + 2.0
        while layer.frames_processed < 1 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert layer.frames_processed >= 1
        feats = layer.get_latest_features()
        assert feats.shape == (30,)
        assert feats.dtype == np.float32
        assert np.all(feats >= -1.0) and np.all(feats <= 1.0)


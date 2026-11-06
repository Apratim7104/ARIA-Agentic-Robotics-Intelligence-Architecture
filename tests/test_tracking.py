"""Hungarian tracking continuity and ARIA JSON validation tests."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spatial_layer import ObjectDetectionProcessor, load_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default_config.yaml")


@pytest.fixture
def processor():
    cfg = load_config(CONFIG_PATH)
    return ObjectDetectionProcessor(cfg)


def _bbox_around(cx: float, cy: float, w: float = 24.0, h: float = 24.0):
    return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]


def test_hungarian_tracking_continuity_across_frames(processor: ObjectDetectionProcessor):
    """Consistent track IDs must be maintained when raw detection IDs shuffle across frames."""
    # Two objects moving smoothly across 15 frames while their input array order alternates.
    traj_a = [(90.0 + 3.0 * k, 80.0 + 2.0 * k) for k in range(15)]
    traj_b = [(210.0 - 2.5 * k, 150.0 - 1.5 * k) for k in range(15)]

    first_frame = processor.process(
        json.dumps({
            "timestamp": 1000.0,
            "primary_target": {"id": 0, "class": "cube", "confidence": 0.95,
                               "bbox": _bbox_around(*traj_a[0])},
            "secondary_objects": [
                {"id": 1, "class": "cube", "confidence": 0.90,
                 "bbox": _bbox_around(*traj_b[0])},
            ],
        })
    )
    assert first_frame.primary is not None
    tid_a = first_frame.primary.track_id
    tid_b = first_frame.secondary[0].track_id
    assert tid_a != tid_b

    for k in range(1, 15):
        # Deliberately scramble raw_id to verify Hungarian assignment uses geometry, not raw_id
        frame = processor.process({
            "timestamp": 1000.0 + 0.05 * k,
            "primary_target": {"id": 99 - k, "class": "cube", "confidence": 0.94,
                               "bbox": _bbox_around(*traj_a[k])},
            "secondary_objects": [
                {"id": 50 + k, "class": "cube", "confidence": 0.88,
                 "bbox": _bbox_around(*traj_b[k])},
            ],
        })
        assert frame.primary is not None
        assert frame.primary.track_id == tid_a
        assert len(frame.secondary) == 1
        assert frame.secondary[0].track_id == tid_b


def test_bbox_and_confidence_validation(processor: ObjectDetectionProcessor):
    """Negative coordinates, out-of-bounds boxes, and low-confidence detections are rejected."""
    msg = {
        "timestamp": 1234.5,
        "primary_target": {
            "id": 0, "class": "cube", "confidence": 0.55,  # below 0.70 threshold
            "bbox": [80, 60, 140, 120],
        },
        "secondary_objects": [
            {"id": 1, "class": "cube", "confidence": 0.95, "bbox": [-5, 20, 40, 60]},     # negative
            {"id": 2, "class": "cube", "confidence": 0.95, "bbox": [10, 20, 350, 60]},    # > width
            {"id": 3, "class": "cube", "confidence": 0.95, "bbox": [100, 100, 80, 120]},  # x2 < x1
            {"id": 4, "class": "cube", "confidence": 0.92, "bbox": [100, 100, 140, 140]}, # valid
        ],
    }
    frame = processor.process(msg)
    assert frame.primary is None
    assert len(frame.secondary) == 1
    assert frame.secondary[0].raw_id == 4
    assert frame.rejected == 4


def test_distance_gating_spawns_new_track(processor: ObjectDetectionProcessor):
    """A sudden jump larger than tracking_distance_threshold (40 px) creates a new track ID."""
    f1 = processor.process({
        "timestamp": 1.0,
        "primary_target": {"id": 0, "class": "cube", "confidence": 0.95,
                           "bbox": _bbox_around(60.0, 60.0)},
        "secondary_objects": [],
    })
    tid1 = f1.primary.track_id

    # Jump by >150 pixels
    f2 = processor.process({
        "timestamp": 1.05,
        "primary_target": {"id": 0, "class": "cube", "confidence": 0.95,
                           "bbox": _bbox_around(240.0, 180.0)},
        "secondary_objects": [],
    })
    assert f2.primary.track_id != tid1


"""ARIA detection ingestion, validation and frame-to-frame Hungarian tracking.

Expected ARIA message (JSON string, bytes or already-decoded dict)::

    {
      "timestamp": 1234567890.123,
      "primary_target": {"id": 0, "class": "cube", "confidence": 0.95,
                         "bbox": [80, 60, 140, 120]},
      "secondary_objects": [ ...same schema... ]
    }

``bbox`` is ``[x_min, y_min, x_max, y_max]`` in pixels. An optional ``quat``
(``[w, x, y, z]``) per object is passed through if the vision agent provides
an orientation estimate.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

_LARGE_COST = 1e6


@dataclass
class Detection:
    """A validated 2D detection (optionally associated with a track)."""

    raw_id: int
    cls: str
    confidence: float
    bbox: np.ndarray                 # [x1, y1, x2, y2]
    is_primary: bool = False
    quat: Optional[np.ndarray] = None
    track_id: int = -1

    @property
    def center(self) -> np.ndarray:
        return np.array([0.5 * (self.bbox[0] + self.bbox[2]),
                         0.5 * (self.bbox[1] + self.bbox[3])])

    @property
    def size(self) -> np.ndarray:
        return np.array([self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]])


@dataclass
class Track:
    track_id: int
    cls: str
    center: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    missed: int = 0
    age: int = 1

    def predict(self) -> np.ndarray:
        """Constant-velocity centroid prediction for the next frame."""
        return self.center + self.velocity


@dataclass
class FrameDetections:
    timestamp: float
    primary: Optional[Detection]
    secondary: List[Detection]
    rejected: int = 0

    @property
    def all(self) -> List[Detection]:
        return ([self.primary] if self.primary is not None else []) + list(self.secondary)


class ObjectDetectionProcessor:
    """Parses/validates ARIA detections and maintains persistent track IDs."""

    def __init__(self, config: dict) -> None:
        tcfg = config.get("tracking", config)
        intr = config.get("camera", {}).get("intrinsics", {})
        self.width = float(intr.get("width", 320))
        self.height = float(intr.get("height", 240))
        self.confidence_threshold = float(tcfg.get("confidence_threshold", 0.70))
        self.distance_threshold = float(tcfg.get("tracking_distance_threshold", 40.0))
        self.max_tracked = int(tcfg.get("max_tracked_objects", 5))
        self.max_missed = int(tcfg.get("max_missed_frames", 5))
        self.velocity_alpha = 0.5           # EMA factor for centroid velocity

        self.tracks: Dict[int, Track] = {}
        self._next_id = 0
        self.frames = 0
        self.total_rejected = 0

    # ------------------------------------------------------------------ #
    # Parsing & validation
    # ------------------------------------------------------------------ #
    def validate_bbox(self, bbox: Any) -> bool:
        """Bbox must be 4 finite, non-negative values inside the image."""
        try:
            if len(bbox) != 4:
                return False
            x1, y1, x2, y2 = (float(b) for b in bbox)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return False
        if x1 < 0.0 or y1 < 0.0 or x2 > self.width or y2 > self.height:
            return False
        return x2 > x1 and y2 > y1

    def _parse_object(self, obj: Any, is_primary: bool) -> Optional[Detection]:
        if not isinstance(obj, dict):
            return None
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(conf) or conf < self.confidence_threshold or conf > 1.0:
            return None
        bbox = obj.get("bbox")
        if not self.validate_bbox(bbox):
            return None
        quat = obj.get("quat")
        if quat is not None:
            q = np.asarray(quat, dtype=np.float64)
            quat = q if q.shape == (4,) and np.all(np.isfinite(q)) and np.linalg.norm(q) > 0 else None
        return Detection(raw_id=int(obj.get("id", -1)), cls=str(obj.get("class", "unknown")),
                         confidence=conf, bbox=np.asarray(bbox, dtype=np.float64),
                         is_primary=is_primary, quat=quat)

    def parse(self, message: Union[str, bytes, dict]) -> FrameDetections:
        """Decode and validate one ARIA message (no tracking)."""
        if isinstance(message, (str, bytes, bytearray)):
            msg = json.loads(message)
        elif isinstance(message, dict):
            msg = message
        else:
            raise TypeError(f"Unsupported ARIA message type: {type(message)!r}")

        ts = msg.get("timestamp")
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = time.time()

        rejected = 0
        primary = None
        if msg.get("primary_target") is not None:
            primary = self._parse_object(msg["primary_target"], True)
            rejected += primary is None
        secondary: List[Detection] = []
        for obj in msg.get("secondary_objects") or []:
            det = self._parse_object(obj, False)
            if det is None:
                rejected += 1
            else:
                secondary.append(det)
        return FrameDetections(timestamp=ts, primary=primary, secondary=secondary,
                               rejected=rejected)

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #
    def _new_track(self, det: Detection) -> None:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(track_id=tid, cls=det.cls, center=det.center,
                                 bbox=det.bbox.copy())
        det.track_id = tid

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Assign persistent track IDs to ``detections`` (in place).

        Builds a centroid-distance cost matrix between predicted track
        positions and the new detections, solves it with the Hungarian
        algorithm, and rejects pairs beyond ``tracking_distance_threshold``
        pixels or with mismatched classes.
        """
        self.frames += 1
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        unmatched_dets = list(range(len(detections)))

        if track_ids and detections:
            preds = np.stack([self.tracks[t].predict() for t in track_ids])
            centers = np.stack([d.center for d in detections])
            cost = np.linalg.norm(preds[:, None, :] - centers[None, :, :], axis=2)
            for i, t in enumerate(track_ids):
                for j, d in enumerate(detections):
                    if self.tracks[t].cls != d.cls:
                        cost[i, j] = _LARGE_COST
            rows, cols = linear_sum_assignment(cost)
            unmatched = set(unmatched_dets)
            for r, c in zip(rows, cols):
                if cost[r, c] > self.distance_threshold:
                    continue
                tr = self.tracks[track_ids[r]]
                det = detections[c]
                new_center = det.center
                tr.velocity = (self.velocity_alpha * (new_center - tr.center)
                               + (1.0 - self.velocity_alpha) * tr.velocity)
                tr.center = new_center
                tr.bbox = det.bbox.copy()
                tr.hits += 1
                tr.missed = 0
                det.track_id = tr.track_id
                matched_tracks.add(tr.track_id)
                unmatched.discard(c)
            unmatched_dets = sorted(unmatched)

        # Age / prune unmatched tracks.
        for t in track_ids:
            tr = self.tracks[t]
            tr.age += 1
            if t not in matched_tracks:
                tr.missed += 1
                tr.velocity *= 0.5
                if tr.missed > self.max_missed:
                    del self.tracks[t]

        # Spawn new tracks: primary first, then by confidence.
        order = sorted(unmatched_dets,
                       key=lambda j: (not detections[j].is_primary, -detections[j].confidence))
        for j in order:
            if len(self.tracks) >= self.max_tracked:
                # Evict the stalest track to make room for a primary target.
                if not detections[j].is_primary:
                    continue
                stale = [t for t in self.tracks.values() if t.missed > 0]
                if not stale:
                    continue
                victim = max(stale, key=lambda t: t.missed)
                del self.tracks[victim.track_id]
            self._new_track(detections[j])
        return detections

    def process(self, message: Union[str, bytes, dict]) -> FrameDetections:
        """Parse, validate and track one ARIA message."""
        frame = self.parse(message)
        self.total_rejected += frame.rejected
        self.update(frame.all)
        # Untracked secondaries (capacity exhausted) are dropped.
        frame.secondary = [d for d in frame.secondary if d.track_id >= 0][: self.max_tracked]
        return frame

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0
        self.frames = 0


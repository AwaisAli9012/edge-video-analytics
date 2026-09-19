"""
Multi-Object Tracker (MOT) Wrapper using ByteTrack via Supervision.
Associates detected bounding boxes across consecutive frames with stable track IDs.
"""

import numpy as np
from typing import List, Dict, Any
import logging

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectTracker:
    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        """
        Initialize ByteTrack Multi-Object Tracker.
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        if SUPERVISION_AVAILABLE:
            # Modern Supervision API instantiation
            if hasattr(sv, 'ByteTrack'):
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_thresh,
                    lost_track_buffer=self.track_buffer,
                    minimum_matching_threshold=self.match_thresh,
                    frame_rate=self.frame_rate
                )
            else:
                self.tracker = None
            logger.info("[INFO] ByteTrack tracker initialized successfully using supervision.")
        else:
            self.tracker = None
            logger.warning("[WARNING] Supervision library not found. Running in fallback mode.")

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update object tracks using frame detections.
        """
        if not detections:
            return []

        if not SUPERVISION_AVAILABLE or self.tracker is None:
            for idx, det in enumerate(detections):
                det["track_id"] = idx + 1
            return detections

        xyxy = np.array([d["box"] for d in detections], dtype=np.float32)
        confidence = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_id = np.array([d["class_id"] for d in detections], dtype=int)

        sup_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        tracked_detections = self.tracker.update_with_detections(sup_detections)

        results = []
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
            for box, conf, cls_id, track_id in zip(
                tracked_detections.xyxy,
                tracked_detections.confidence,
                tracked_detections.class_id,
                tracked_detections.tracker_id
            ):
                results.append({
                    "box": [float(b) for b in box],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "track_id": int(track_id)
                })
        else:
            for det in detections:
                det["track_id"] = -1
                results.append(det)

        return results
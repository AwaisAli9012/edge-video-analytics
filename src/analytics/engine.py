"""
Analytics Engine for Line-Crossing Counters, ROI Geofencing, and Dwell-Time Tracking.
"""

from typing import List, Dict, Any, Tuple
import time
import logging
from src.analytics.spatial import is_point_in_polygon, intersect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsEngine:
    def __init__(
        self,
        line_counter: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        roi_polygon: List[Tuple[float, float]] = None,
        fps: int = 30
    ):
        self.line_counter = line_counter
        self.roi_polygon = roi_polygon
        self.fps = fps

        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.roi_dwell_frames: Dict[int, int] = {}
        self.line_cross_count = 0
        self.crossed_ids = set()

    def _get_centroid(self, box: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y2)

    def process(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        active_ids = set()
        active_in_roi = []
        dwell_times_sec = {}

        for obj in tracked_objects:
            track_id = obj.get("track_id", -1)
            if track_id == -1:
                continue

            active_ids.add(track_id)
            centroid = self._get_centroid(obj["box"])

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(centroid)

            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Line Crossing Verification
            if self.line_counter and track_id not in self.crossed_ids and len(self.track_history[track_id]) >= 2:
                prev_pos = self.track_history[track_id][-2]
                curr_pos = self.track_history[track_id][-1]
                
                line_p1, line_p2 = self.line_counter
                if intersect(prev_pos, curr_pos, line_p1, line_p2):
                    self.line_cross_count += 1
                    self.crossed_ids.add(track_id)
                    logger.info(f"[ANALYTICS] Track ID {track_id} crossed the line! Total count: {self.line_cross_count}")

            # ROI & Dwell Time Calculation
            if self.roi_polygon:
                if is_point_in_polygon(centroid, self.roi_polygon):
                    active_in_roi.append(track_id)
                    self.roi_dwell_frames[track_id] = self.roi_dwell_frames.get(track_id, 0) + 1
                    dwell_sec = self.roi_dwell_frames[track_id] / self.fps
                    dwell_times_sec[track_id] = round(dwell_sec, 2)

        return {
            "total_line_crossings": self.line_cross_count,
            "objects_in_roi_count": len(active_in_roi),
            "roi_active_track_ids": active_in_roi,
            "dwell_times_seconds": dwell_times_sec
        }

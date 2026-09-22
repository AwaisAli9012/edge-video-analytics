"""
Frame Renderer and Visualizer for Edge Analytics.
Draws bounding boxes, track IDs, ROI geofences, line gates, and HUD metrics onto video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class FrameVisualizer:
    def __init__(
        self,
        line_counter: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        roi_polygon: List[Tuple[float, float]] = None,
        class_names: Dict[int, str] = None
    ):
        self.line_counter = line_counter
        self.roi_polygon = roi_polygon
        self.class_names = class_names or {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # Palette for track ID colors (BGR)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=int)

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a consistent color per Track ID."""
        if track_id < 0:
            return (200, 200, 200)
        idx = track_id % len(self.colors)
        return int(self.colors[idx][0]), int(self.colors[idx][1]), int(self.colors[idx][2])

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        analytics_metrics: Dict[str, Any],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Render all visual elements onto the provided frame.
        """
        canvas = frame.copy()

        # 1. Draw ROI Polygon Zone
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            pts = np.array(self.roi_polygon, np.int32).reshape((-1, 1, 2))
            
            # Semi-transparent fill
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))  # Yellow tint
            cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)

            # Zone label
            cv2.putText(canvas, "ROI GEOFENCE", (pts[0][0][0], max(15, pts[0][0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Virtual Line Counter Gate
        if self.line_counter:
            p1 = (int(self.line_counter[0][0]), int(self.line_counter[0][1]))
            p2 = (int(self.line_counter[1][0]), int(self.line_counter[1][1]))
            cv2.line(canvas, p1, p2, (0, 0, 255), 3)  # Red line
            cv2.circle(canvas, p1, 5, (0, 0, 255), -1)
            cv2.circle(canvas, p2, 5, (0, 0, 255), -1)
            cv2.putText(canvas, "COUNT GATE", (p1[0] + 5, p1[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 3. Draw Tracked Objects & Bounding Boxes
        for obj in tracked_objects:
            box = [int(v) for v in obj["box"]]
            track_id = obj.get("track_id", -1)
            class_id = obj.get("class_id", 0)
            conf = obj.get("confidence", 0.0)

            color = self._get_color(track_id)
            label_text = self.class_names.get(class_id, f"cls:{class_id}")
            
            display_label = f"ID:{track_id} {label_text} {conf:.2f}" if track_id != -1 else f"{label_text} {conf:.2f}"

            # Bounding Box
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Centroid Point
            centroid_x = int((box[0] + box[2]) / 2)
            centroid_y = box[3]
            cv2.circle(canvas, (centroid_x, centroid_y), 4, color, -1)

            # Label Background & Text
            (w, h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (box[0], box[1] - h - 6), (box[0] + w + 6, box[1]), color, -1)
            cv2.putText(canvas, display_label, (box[0] + 3, box[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw Dashboard HUD (Top-Left Metrics Overlay)
        hud_w, hud_h = 280, 100
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, canvas, 0.4, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (255, 255, 255), 1)

        # HUD Text
        line_count = analytics_metrics.get("total_line_crossings", 0)
        roi_count = analytics_metrics.get("objects_in_roi_count", 0)

        cv2.putText(canvas, f"EDGE ANALYTICS ENGINE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Line Crossings: {line_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Objects in ROI: {roi_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return canvas

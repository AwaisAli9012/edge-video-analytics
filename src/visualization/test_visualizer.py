"""
Test Suite for FrameVisualizer.
Renders synthetic frame overlays and exports test_output.jpg.
"""

import cv2
import numpy as np
from src.visualization.draw import FrameVisualizer


def test_visualization():
    print("[INFO] Initializing Frame Visualizer Test...")

    # Create synthetic blank video frame (720p)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8) + 40  # Dark grey background

    line_gate = ((640.0, 0.0), (640.0, 720.0))
    roi_zone = [(800.0, 200.0), (1100.0, 200.0), (1100.0, 600.0), (800.0, 600.0)]

    visualizer = FrameVisualizer(line_counter=line_gate, roi_polygon=roi_zone)

    # Synthetic tracked detections
    tracked_objects = [
        {"box": [400.0, 250.0, 500.0, 450.0], "confidence": 0.92, "class_id": 0, "track_id": 1},
        {"box": [850.0, 300.0, 950.0, 500.0], "confidence": 0.88, "class_id": 0, "track_id": 2}
    ]

    metrics = {
        "total_line_crossings": 3,
        "objects_in_roi_count": 1,
        "dwell_times_seconds": {2: 4.5}
    }

    # Render overlays
    rendered_frame = visualizer.draw(frame, tracked_objects, metrics, fps=29.8)

    output_filename = "test_visualizer_output.jpg"
    cv2.imwrite(output_filename, rendered_frame)
    print(f"[SUCCESS] Test frame successfully rendered and saved to '{output_filename}'!")


if __name__ == "__main__":
    test_visualization()

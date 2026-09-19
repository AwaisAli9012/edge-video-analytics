"""
Test Suite for AnalyticsEngine.
Simulates trajectory motion through virtual line gates and ROI polygons.
"""

from src.analytics.engine import AnalyticsEngine


def test_analytics_simulation():
    print("[INFO] Initializing Analytics Engine...")

    line_gate = ((150.0, 0.0), (150.0, 300.0))
    roi_zone = [(200.0, 100.0), (400.0, 100.0), (400.0, 300.0), (200.0, 300.0)]

    analytics = AnalyticsEngine(line_counter=line_gate, roi_polygon=roi_zone, fps=30)

    print("[INFO] Simulating Object (Track ID 1) trajectory across Line and into ROI...")
    
    for frame in range(1, 31):
        x_pos = 100.0 + (frame * 10.0)
        box = [x_pos - 15.0, 150.0, x_pos + 15.0, 200.0]

        tracked_objects = [{
            "box": box,
            "confidence": 0.90,
            "class_id": 0,
            "track_id": 1
        }]

        metrics = analytics.process(tracked_objects)

        if frame % 10 == 0:
            print(f"  Frame {frame:02d} | Line Crossings: {metrics['total_line_crossings']} | "
                  f"Objects in ROI: {metrics['objects_in_roi_count']} | "
                  f"Dwell Time: {metrics['dwell_times_seconds'].get(1, 0.0)}s")

    print("[SUCCESS] Analytics simulation complete.")
    assert metrics["total_line_crossings"] == 1, "Line crossing failed to detect!"
    assert metrics["objects_in_roi_count"] == 1, "ROI detection failed!"
    print("[PASS] All Spatial Analytics asserts passed successfully!")


if __name__ == "__main__":
    test_analytics_simulation()

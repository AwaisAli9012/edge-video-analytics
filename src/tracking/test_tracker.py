"""
Test Suite for ObjectTracker (ByteTrack Integration).
Simulates object movement across synthetic frames to verify ID persistence.
"""

import time
from src.tracking.tracker import ObjectTracker


def test_tracker_pipeline():
    print("[INFO] Initializing ByteTrack tracker...")
    tracker = ObjectTracker(track_thresh=0.2, track_buffer=30)

    base_box = [100.0, 100.0, 200.0, 200.0]  # x1, y1, x2, y2
    num_frames = 10

    print(f"[INFO] Running tracker simulation across {num_frames} synthetic frames...")
    
    start_time = time.perf_counter()
    assigned_ids = []

    for frame_idx in range(num_frames):
        offset = frame_idx * 5.0
        simulated_box = [
            base_box[0] + offset,
            base_box[1],
            base_box[2] + offset,
            base_box[3]
        ]

        detections = [{
            "box": simulated_box,
            "confidence": 0.85,
            "class_id": 0
        }]

        tracked_results = tracker.update(detections)
        
        if tracked_results:
            track_id = tracked_results[0].get("track_id", -1)
            assigned_ids.append(track_id)
            print(f"  Frame {frame_idx + 1:02d} -> Box: [{simulated_box[0]:.1f}, {simulated_box[1]:.1f}] | Track ID: {track_id}")

    elapsed = (time.perf_counter() - start_time) * 1000
    avg_latency = elapsed / num_frames

    print(f"[SUCCESS] Multi-Object Tracking Complete.")
    print(f"[METRICS] Average Tracker Overhead: {avg_latency:.3f} ms / frame")
    print(f"[OUTPUT] Assigned Track IDs across frames: {assigned_ids}")

    if len(set(assigned_ids)) == 1 and list(set(assigned_ids))[0] != -1:
        print("[PASS] Object tracking ID persistence verified! (ID remained stable)")
    else:
        print("[WARN] ID flipped or non-persistent during tracking simulation.")


if __name__ == "__main__":
    test_tracker_pipeline()
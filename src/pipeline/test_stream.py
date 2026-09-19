import time
import cv2
from src.pipeline.stream import AsyncVideoStream

def run_test():
    # Generate a dummy synthetic video for verification if no file is present
    print("[INFO] Testing AsyncVideoStream pipeline...")
    
    # Using a 0 index for webcam test or standard synthetic video
    # Initializing stream
    video_path = "data/raw/sample.mp4"
    
    # Create a quick synthetic video file for testing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    for i in range(90):  # 3-second test clip
        frame = (np.random.rand(480, 640, 3) * 255).astype('uint8')
        out.write(frame)
    out.release()
    
    stream = AsyncVideoStream(src=video_path, queue_size=32).start()
    frame_count = 0
    start_time = time.time()

    while stream.is_running():
        ret, frame = stream.read()
        if ret:
            frame_count += 1
            # Simulate slight inference workload (10ms)
            time.sleep(0.01)

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"[SUCCESS] Processed {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
    stream.stop()

if __name__ == "__main__":
    import numpy as np
    run_test()

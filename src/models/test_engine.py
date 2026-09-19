import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import time
from src.models.engine import InferenceEngine

def run_test():
    model_path = "yolov8n.onnx"
    if not Path(model_path).exists():
        print(f"[ERROR] ONNX model file not found at {model_path}. Please check Step 3 export.")
        return

    print(f"[INFO] Initializing InferenceEngine with {model_path}...")
    engine = InferenceEngine(model_path=model_path, confidence_threshold=0.25)

    # Generate synthetic image for testing pipeline (640x480 RGB)
    dummy_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    # Warmup pass
    _ = engine.infer(dummy_frame)

    # Benchmark 20 inference passes
    start_time = time.time()
    num_runs = 20
    for _ in range(num_runs):
        detections = engine.infer(dummy_frame)
    
    elapsed = time.time() - start_time
    avg_latency = (elapsed / num_runs) * 1000
    fps = num_runs / elapsed

    print(f"[SUCCESS] Executed {num_runs} inference passes.")
    print(f"[METRICS] Average Latency: {avg_latency:.2f} ms | Throughput: {fps:.2f} FPS")
    print(f"[OUTPUT] Detections count on dummy frame: {len(detections)}")

if __name__ == "__main__":
    run_test()

import argparse
from pathlib import Path
from ultralytics import YOLO

def export_model(model_name: str, export_format: str, dynamic: bool, half: bool):
    """
    Downloads base YOLO weights and exports to ONNX/TensorRT formats for low-latency edge inference.
    """
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading model: {model_name}...")
    model = YOLO(f"{model_name}.pt")
    
    print(f"[INFO] Exporting to format={export_format} (dynamic={dynamic}, half={half})...")
    exported_path = model.export(
        format=export_format,
        dynamic=dynamic,
        half=half,
        simplify=True
    )
    
    print(f"[SUCCESS] Model exported successfully to: {exported_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Model Exporter")
    parser.add_argument("--model", type=str, default="yolov8n", help="Base model name (e.g., yolov8n, yolov8s)")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine"], help="Export format")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shape")
    parser.add_argument("--half", action="store_true", help="Enable FP16 half precision")
    args = parser.parse_args()

    export_model(args.model, args.format, args.dynamic, args.half)

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class InferenceEngine:
    """
    High-performance ONNX Runtime engine wrapper for YOLOv8 models.
    Handles image preprocessing, dynamic tensor execution, and output parsing.
    """
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Initialize ONNX Runtime Session prioritizing CUDA/GPU with CPU fallback
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Retrieve model input metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.img_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocesses BGR image for YOLOv8 model input (Resize, Letterbox, Normalization, NCHW conversion).
        """
        h, w = image.shape[:2]
        scale = min(self.img_width / w, self.img_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_height, self.img_width, 3), 114, dtype=np.uint8)

        pad_x = (self.img_width - new_w) // 2
        pad_y = (self.img_height - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalize BGR to RGB, scale [0, 1], transpose to (1, 3, H, W)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, (pad_x, pad_y)

    def infer(self, image: np.ndarray) -> List[Tuple[List[int], float, int]]:
        """
        Runs model inference on a raw frame and returns parsed detections:
        List of tuples: [ [x1, y1, x2, y2], confidence, class_id ]
        """
        blob, scale, (pad_x, pad_y) = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        predictions = np.squeeze(outputs[0]).T  # Shape: (8400, 84) for YOLOv8 standard

        boxes, confidences, class_ids = [], [], []

        for pred in predictions:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >= self.conf_threshold:
                cx, cy, w, h = pred[:4]
                # Scale coordinates back to original image space
                x1 = int((cx - w / 2 - pad_x) / scale)
                y1 = int((cy - h / 2 - pad_y) / scale)
                x2 = int((cx + w / 2 - pad_x) / scale)
                y2 = int((cy + h / 2 - pad_y) / scale)

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append(([x, y, x + w, y + h], confidences[i], class_ids[i]))

        return results

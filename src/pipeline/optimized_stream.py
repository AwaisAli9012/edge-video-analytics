"""
Optimized Real-Time Edge Analytics Pipeline.
Includes Frame Downscaling, Frame Skipping for Detection, and Optimized Tracking.
"""

import time
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.models.engine import InferenceEngine
from src.tracking.tracker import ObjectTracker
from src.analytics.engine import AnalyticsEngine
from src.visualization.draw import FrameVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedVideoPipeline:
    def __init__(
        self,
        model_path: str,
        source: str,
        output_path: Optional[str] = None,
        line_counter: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
        confidence_threshold: float = 0.25,
        target_width: int = 1280,
        detect_interval: int = 2,
        show_window: bool = True
    ):
        self.source = source
        self.output_path = output_path
        self.show_window = show_window
        self.target_width = target_width
        self.detect_interval = detect_interval

        logger.info("[OPTIMIZED PIPELINE] Initializing Inference Engine...")
        self.engine = InferenceEngine(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        logger.info("[OPTIMIZED PIPELINE] Initializing ByteTrack...")
        self.tracker = ObjectTracker(frame_rate=30)

        logger.info("[OPTIMIZED PIPELINE] Initializing Analytics & Visualizer...")
        self.analytics = AnalyticsEngine(line_counter=line_counter, roi_polygon=roi_polygon, fps=30)
        self.visualizer = FrameVisualizer(line_counter=line_counter, roi_polygon=roi_polygon)

    def run(self, max_frames: Optional[int] = None):
        cap_source = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            logger.error(f"[PIPELINE ERROR] Unable to open source: {self.source}")
            return

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if self.target_width and orig_width > self.target_width:
            scale = self.target_width / float(orig_width)
            proc_width = self.target_width
            proc_height = int(orig_height * scale)
        else:
            scale = 1.0
            proc_width, proc_height = orig_width, orig_height

        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, input_fps, (proc_width, proc_height))
            logger.info(f"[OPTIMIZED PIPELINE] Writer setup: {proc_width}x{proc_height} @ {input_fps:.1f} FPS")

        frame_count = 0
        fps_measured = 0.0
        last_detections = np.empty((0, 6))

        logger.info("[OPTIMIZED PIPELINE] Running pipeline loop...")

        try:
            while cap.isOpened():
                start_time = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if scale != 1.0:
                    frame_proc = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_proc = frame

                if frame_count % self.detect_interval == 0 or len(last_detections) == 0:
                    detections = self.engine.infer(frame_proc)
                    last_detections = detections
                else:
                    detections = last_detections

                tracked_objects = self.tracker.update(detections)
                metrics = self.analytics.process(tracked_objects)
                annotated_frame = self.visualizer.draw(frame_proc, tracked_objects, metrics, fps=fps_measured)

                if writer:
                    writer.write(annotated_frame)

                if self.show_window:
                    try:
                        cv2.imshow("Optimized Edge Video Analytics", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("[PIPELINE] User exit requested.")
                            break
                    except cv2.error as e:
                        logger.warning(f"[GUI WARNING] GUI unavailable ({e}). Continuing headless...")
                        self.show_window = False

                elapsed = time.perf_counter() - start_time
                fps_measured = 1.0 / elapsed if elapsed > 0 else 0.0

                if frame_count % 30 == 0:
                    logger.info(f"[PIPELINE STATS] Frame {frame_count} | FPS: {fps_measured:.1f} | Active Tracks: {len(tracked_objects)}")

                if max_frames and frame_count >= max_frames:
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            logger.info(f"[PIPELINE COMPLETE] Total frames processed: {frame_count}")

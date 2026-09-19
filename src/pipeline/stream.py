import cv2
import queue
import threading
import time
from typing import Optional, Tuple
import numpy as np

class AsyncVideoStream:
    """
    Asynchronous, multi-threaded video stream reader for RTSP/Video file streams.
    Prevents frame drop and I/O bottlenecks by decoding frames on a background thread.
    """
    def __init__(self, src: str, queue_size: int = 128):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {src}")
        
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        """Starts the background frame reading thread."""
        self.thread.start()
        return self

    def _update(self):
        """Internal background loop to continuously read and enqueue frames."""
        while not self.stopped:
            if not self.q.full():
                grabbed, frame = self.cap.read()
                if not grabbed:
                    self.stopped = True
                    break
                self.q.put(frame)
            else:
                time.sleep(0.001)  # Prevent CPU burn when queue is full

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the next frame from the queue."""
        if not self.q.empty():
            return True, self.q.get()
        return False, None

    def stop(self):
        """Stops the reader thread and releases video source from caller thread."""
        self.stopped = True
        if threading.current_thread() != self.thread:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
        self.cap.release()

    def is_running(self) -> bool:
        """Returns True if stream is active or queue still has frames."""
        return not self.stopped or not self.q.empty()

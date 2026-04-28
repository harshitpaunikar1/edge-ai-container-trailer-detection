"""
End-to-end edge AI inference pipeline for container and trailer detection.
Handles RTSP/video source reading, frame skipping, detection, and result publishing.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Video capture will use stub frames.")

from detector import DetectionBox, LogisticsDetector, ROIFilter


@dataclass
class PipelineConfig:
    source: str = "0"             # camera index, RTSP URL, or video file
    frame_skip: int = 2           # process every Nth frame
    max_queue_size: int = 10
    display: bool = False
    output_callback: Optional[Callable] = None
    roi_coords: Optional[tuple] = None  # (x1, y1, x2, y2)
    model_path: str = "yolov8n.pt"
    confidence: float = 0.45
    target_classes: List[str] = field(default_factory=lambda: ["container", "trailer", "truck"])


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: List[DetectionBox]
    processing_time_ms: float

    def summary(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "num_detections": len(self.detections),
            "processing_ms": round(self.processing_time_ms, 2),
            "objects": [d.to_dict() for d in self.detections],
        }


class VideoCapture:
    """Thread-safe OpenCV video capture wrapper with stub support."""

    def __init__(self, source: str):
        self.source = source
        self.cap = None
        self._open()

    def _open(self) -> None:
        if not CV2_AVAILABLE:
            return
        try:
            src = int(self.source) if self.source.isdigit() else self.source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                logger.warning("Could not open source: %s", self.source)
                self.cap = None
        except Exception as exc:
            logger.error("VideoCapture open error: %s", exc)
            self.cap = None

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None or not CV2_AVAILABLE:
            return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class DetectionPipeline:
    """
    Orchestrates frame capture, detection, and result dispatch for edge logistics vision.
    Supports synchronous and background-threaded operation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        roi = None
        if config.roi_coords:
            x1, y1, x2, y2 = config.roi_coords
            roi = ROIFilter(x1=x1, y1=y1, x2=x2, y2=y2)
        self.detector = LogisticsDetector(
            model_path=config.model_path,
            confidence_threshold=config.confidence,
            target_classes=config.target_classes,
            roi=roi,
        )
        self.result_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._frame_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FrameResult(
            frame_id=self._frame_id,
            timestamp=time.time(),
            detections=detections,
            processing_time_ms=elapsed_ms,
        )

    def run_sync(self, max_frames: int = 100) -> List[FrameResult]:
        """Process up to max_frames synchronously and return all results."""
        self.detector.load_model()
        capture = VideoCapture(self.config.source)
        results = []
        for _ in range(max_frames):
            frame = capture.read()
            if frame is None:
                break
            self._frame_id += 1
            if self._frame_id % self.config.frame_skip != 0:
                continue
            result = self._process_frame(frame)
            results.append(result)
            if self.config.output_callback:
                self.config.output_callback(result)
        capture.release()
        return results

    def _run_loop(self, max_frames: int) -> None:
        self.detector.load_model()
        capture = VideoCapture(self.config.source)
        processed = 0
        while self._running and processed < max_frames:
            frame = capture.read()
            if frame is None:
                break
            self._frame_id += 1
            if self._frame_id % self.config.frame_skip != 0:
                continue
            result = self._process_frame(frame)
            processed += 1
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                self.result_queue.get_nowait()
                self.result_queue.put_nowait(result)
            if self.config.output_callback:
                self.config.output_callback(result)
        capture.release()
        self._running = False

    def start_background(self, max_frames: int = 1000) -> None:
        """Start detection pipeline in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, args=(max_frames,), daemon=True)
        self._thread.start()
        logger.info("Background detection pipeline started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_result(self, timeout: float = 1.0) -> Optional[FrameResult]:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def pipeline_stats(self) -> Dict:
        return {
            "frames_seen": self._frame_id,
            "detector_stats": self.detector.detection_stats(),
            "queue_size": self.result_queue.qsize(),
        }


if __name__ == "__main__":
    def print_result(result: FrameResult):
        if result.detections:
            print(f"Frame {result.frame_id}: {len(result.detections)} detections "
                  f"({result.processing_time_ms:.1f}ms)")

    config = PipelineConfig(
        source="0",
        frame_skip=2,
        model_path="yolov8n.pt",
        confidence=0.45,
        target_classes=["container", "trailer", "truck"],
        roi_coords=(50, 50, 1200, 700),
        output_callback=print_result,
    )
    pipeline = DetectionPipeline(config)
    print("Running 10 frames synchronously...")
    results = pipeline.run_sync(max_frames=10)
    print(f"Processed {len(results)} frames.")
    for r in results[:3]:
        print(r.summary())
    print("Stats:", pipeline.pipeline_stats())

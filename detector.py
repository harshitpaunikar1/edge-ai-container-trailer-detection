"""
Container and trailer detection engine for edge AI logistics systems.
Wraps YOLOv8 inference with class filtering, ROI masking, and detection confidence logic.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Detection will use stub mode.")


LOGISTICS_CLASSES = {
    "container": 0,
    "trailer": 1,
    "truck": 2,
    "forklift": 3,
    "pallet": 4,
    "person": 5,
}


@dataclass
class DetectionBox:
    class_name: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    timestamp: float = field(default_factory=time.time)

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> Dict:
        return {
            "class": self.class_name,
            "conf": round(self.confidence, 3),
            "bbox": [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
            "area": round(self.area, 1),
            "timestamp": self.timestamp,
        }


class ROIFilter:
    """Filters detections to a region of interest polygon or bounding rectangle."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def contains_center(self, box: DetectionBox) -> bool:
        cx, cy = box.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def filter(self, detections: List[DetectionBox]) -> List[DetectionBox]:
        return [d for d in detections if self.contains_center(d)]


class LogisticsDetector:
    """
    YOLOv8-based detector for logistics objects: containers, trailers, trucks, forklifts.
    Supports confidence thresholding, ROI filtering, and class-specific alerts.
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.45,
                 target_classes: Optional[List[str]] = None,
                 roi: Optional[ROIFilter] = None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or list(LOGISTICS_CLASSES.keys())
        self.roi = roi
        self.model = None
        self._frame_count = 0

    def load_model(self) -> bool:
        if not YOLO_AVAILABLE:
            logger.warning("YOLO unavailable. Detector in stub mode.")
            return False
        try:
            self.model = YOLO(self.model_path)
            logger.info("Loaded YOLO model from %s", self.model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    def _stub_detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Return synthetic detections for testing when YOLO is unavailable."""
        h, w = frame.shape[:2]
        detections = []
        for i, cls in enumerate(self.target_classes[:3]):
            x1 = np.random.uniform(0.1, 0.4) * w
            y1 = np.random.uniform(0.1, 0.4) * h
            x2 = x1 + np.random.uniform(0.1, 0.3) * w
            y2 = y1 + np.random.uniform(0.1, 0.3) * h
            conf = np.random.uniform(self.confidence_threshold, 0.99)
            detections.append(DetectionBox(
                class_name=cls,
                class_id=LOGISTICS_CLASSES.get(cls, i),
                confidence=conf,
                x1=x1, y1=y1, x2=min(x2, w), y2=min(y2, h),
            ))
        return detections

    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Run detection on a single frame and return filtered DetectionBox list."""
        self._frame_count += 1
        if self.model is None:
            detections = self._stub_detect(frame)
        else:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = r.names.get(cls_id, str(cls_id))
                    if cls_name not in self.target_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(DetectionBox(
                        class_name=cls_name, class_id=cls_id,
                        confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2,
                    ))
        if self.roi is not None:
            detections = self.roi.filter(detections)
        return detections

    def count_by_class(self, detections: List[DetectionBox]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def generate_alerts(self, detections: List[DetectionBox],
                        alert_classes: Optional[List[str]] = None,
                        max_allowed: int = 0) -> List[str]:
        """
        Return alert strings for detections of specified classes exceeding max_allowed count.
        Default alert class is 'person' (unauthorized personnel in restricted zone).
        """
        if alert_classes is None:
            alert_classes = ["person"]
        counts = self.count_by_class(detections)
        alerts = []
        for cls in alert_classes:
            cnt = counts.get(cls, 0)
            if cnt > max_allowed:
                alerts.append(f"ALERT: {cnt} '{cls}' detected in zone (limit {max_allowed})")
        return alerts

    def detection_stats(self) -> Dict:
        return {"frames_processed": self._frame_count}


if __name__ == "__main__":
    np.random.seed(42)
    detector = LogisticsDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.45,
        target_classes=["container", "trailer", "truck", "forklift"],
        roi=ROIFilter(x1=100, y1=50, x2=900, y2=700),
    )
    detector.load_model()

    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  {d.class_name} conf={d.confidence:.2f} bbox={d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}")

    counts = detector.count_by_class(detections)
    print("\nCount by class:", counts)
    alerts = detector.generate_alerts(detections, alert_classes=["person"])
    if alerts:
        for a in alerts:
            print(a)
    print("Stats:", detector.detection_stats())

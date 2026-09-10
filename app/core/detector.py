"""YOLO-based person detection, including optional segmentation masks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import time
from ultralytics import YOLO


@dataclass
class Detection:
    """One person detection passed from YOLO to the tracker.

    ``mask`` is stored in bounding-box-local coordinates. Keeping it local
    avoids resizing and copying a full camera frame for every detected person.
    """
    box: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray | None = None


class PersonDetector:
    """Wrap YOLO and expose only person boxes, scores, and local masks."""
    def __init__(self, model_name: str, confidence: float = 0.35, device: str = ''):
        """Load YOLO once and retain the device/confidence settings per run."""
        self.model = YOLO(model_name)
        self.confidence, self.device = confidence, device
        self.last_profile = {}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection and convert tensors into tracker records.

        The detector keeps timing data in ``last_profile`` so the pipeline can
        distinguish YOLO time from mask conversion time.
        """
        started = time.perf_counter()
        result = self.model(
            frame,
            # COCO class 0 is a person; filtering at the model call avoids
            # constructing tracker records for unrelated objects.
            classes=[0],
            conf=self.confidence,
            iou=0.7,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]
        yolo_ms = (time.perf_counter() - started) * 1000.
        self.last_profile = {'yolo_ms': yolo_ms, 'mask_transfer_ms': 0., 'mask_resize_ms': 0., 'mask_count': 0}
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        detections = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            mask = None
            if result.masks is not None:
                mask_started = time.perf_counter()
                x1, y1, x2, y2 = box
                mask = np.zeros((max(1, y2-y1), max(1, x2-x1)), dtype=np.uint8)
                polygons = getattr(result.masks, 'xy', None)
                if polygons is not None and i < len(polygons):
                    # Polygon coordinates are frame-relative. Translate them
                    # into the local mask's coordinate system before rasterizing.
                    polygon = np.asarray(polygons[i], dtype=np.float32).copy()
                    polygon[:, 0] -= x1; polygon[:, 1] -= y1
                    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
                else:
                    # Some result adapters expose only mask tensors. Resize
                    # those tensors directly to this detection's crop as a
                    # compatibility fallback.
                    raw_mask = result.masks.data[i].cpu().numpy()
                    mask = cv2.resize(raw_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    self.last_profile['mask_resize_ms'] += (time.perf_counter() - mask_started) * 1000.
                self.last_profile['mask_count'] += 1
            detections.append(Detection(tuple(box), float(score), mask))
        return detections

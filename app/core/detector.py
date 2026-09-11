"""YOLO11-seg detection + Ultralytics built-in multi-object tracking adapter."""
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Person observation emitted by YOLO11-seg + its built-in tracker."""
    box: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray | None = None
    yolo_track_id: int | None = None


class PersonDetector:
    """Run YOLO11-seg with an Ultralytics tracker and expose local masks.

    YOLO owns short-term geometric association/track IDs. V08 Re-IDTracker
    remains responsible for conservative appearance identity, occlusion
    recovery, gallery protection, face/person arbitration, and metrics.
    """
    def __init__(self, model_name: str, confidence: float = 0.35,
                 device: str = '', tracker: str = 'bytetrack.yaml'):
        self.model = YOLO(model_name)
        self.confidence, self.device, self.tracker = confidence, device, tracker
        self.last_profile = {}
        self._started = False

    def reset(self) -> None:
        """Reset the persistent Ultralytics tracker state."""
        self._started = False
        predictor = getattr(self.model, 'predictor', None)
        if predictor is not None:
            predictor.trackers = None
            predictor.vid_path = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Track persons with YOLO11-seg and return boxes, masks and YOLO IDs."""
        started = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            conf=self.confidence,
            iou=0.7,
            imgsz=640,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        yolo_ms = (time.perf_counter() - started) * 1000.0
        self.last_profile = {
            'yolo_ms': yolo_ms, 'mask_transfer_ms': 0.0,
            'mask_resize_ms': 0.0, 'mask_count': 0,
            'yolo_tracker_ids': 0,
        }
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)
        self.last_profile['yolo_tracker_ids'] = sum(x is not None for x in track_ids)

        detections: list[Detection] = []
        polygons = getattr(result.masks, 'xy', None) if result.masks is not None else None
        for i, (box, score) in enumerate(zip(boxes, scores)):
            mask = None
            if result.masks is not None:
                mask_started = time.perf_counter()
                x1, y1, x2, y2 = box
                mask = np.zeros((max(1, y2-y1), max(1, x2-x1)), dtype=np.uint8)
                if polygons is not None and i < len(polygons):
                    polygon = np.asarray(polygons[i], dtype=np.float32).copy()
                    polygon[:, 0] -= x1
                    polygon[:, 1] -= y1
                    if len(polygon) >= 3:
                        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
                else:
                    raw_mask = result.masks.data[i].cpu().numpy()
                    mask = cv2.resize(raw_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    self.last_profile['mask_resize_ms'] += (time.perf_counter() - mask_started) * 1000.0
                self.last_profile['mask_count'] += 1
            detections.append(Detection(tuple(box), float(score), mask, track_ids[i]))
        return detections

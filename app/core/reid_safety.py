"""Centralized Re-ID validation and track-state hardening for V08."""
from __future__ import annotations
from collections import Counter
import logging
import time
import numpy as np
from app.core import tracker as _tracker

logger = logging.getLogger(__name__)
EXPECTED_BODY_DIM = 512


def validate_embedding(embedding, expected_dim: int | None = None):
    """Return ``(valid, embedding, reason)`` without returning bad data."""
    if embedding is None:
        return False, None, "NONE"
    try:
        value = np.asarray(embedding)
    except Exception:
        return False, None, "MALFORMED"
    if value.size == 0:
        return False, None, "EMPTY"
    if not np.issubdtype(value.dtype, np.floating):
        return False, None, "UNEXPECTED_DTYPE"
    value = value.reshape(-1)
    if expected_dim is not None and value.shape[0] != expected_dim:
        return False, None, "WRONG_DIMENSION"
    if not np.isfinite(value).all():
        return False, None, "NAN_OR_INF"
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-8:
        return False, None, "ZERO_NORM" if norm <= 1e-8 else "NAN_OR_INF"
    value = value.astype(np.float32, copy=False)
    return True, value / norm, "OK"


_BaseTrack = _tracker.Track


class SafeTrack(_BaseTrack):
    """Track that never lets an invalid ReID result overwrite valid state."""
    def __post_init__(self):
        super().__post_init__()
        if not validate_embedding(self.body_embedding, EXPECTED_BODY_DIM)[0]:
            self.body_embedding = None
        self.last_body_embedding = None
        self.reid_validation_stats = Counter()

    def _validated(self, embedding, current, expected_dim=None):
        current_ok, current_value, _ = validate_embedding(current, expected_dim)
        if current_ok:
            expected_dim = len(current_value)
        valid, value, reason = validate_embedding(embedding, expected_dim)
        self.reid_validation_stats[reason] += 1
        if not valid:
            logger.debug("REID_INVALID track_id=%s reason=%s", self.tracker_id, reason)
            return None
        return value

    def update(self, box, body, face, confidence, update_gallery=False,
               recovered=False, detection_confidence=1.0):
        body = self._validated(body, self.body_embedding, EXPECTED_BODY_DIM)
        face = self._validated(face, self.face_embedding)
        return super().update(box, body, face, confidence,
                              update_gallery=update_gallery,
                              recovered=recovered,
                              detection_confidence=detection_confidence)


# The old V08 implementation refreshed every detection every N frames. This
# replacement schedules fresh appearance by track condition instead.
def _needs_fresh_appearance(self, boxes):
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    if not active:
        return bool(boxes)
    for box in boxes:
        candidates = []
        for track in active:
            iou = self._compute_iou(track.predicted_bbox, box)
            distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
            candidates.append((iou - 0.02 * distance, iou, distance, track))
        _, best_iou, best_distance, best = max(candidates, key=lambda x: x[0])
        if best.state in (_tracker.TrackState.TENTATIVE, _tracker.TrackState.OCCLUDED, _tracker.TrackState.LOST):
            return True
        if best.reid_age >= self.reid_interval or best.prediction_uncertainty >= .25:
            return True
        if best_iou < self.association_min_iou and best_distance > self.weak_motion_distance:
            return True
        if any(self._compute_iou(box, other) >= .10 for other in boxes if not np.array_equal(other, box)):
            return True
    return False


def _selective_extract_embeddings_batch(self, image, boxes, masks):
    """Extract appearance only for new, uncertain, stale, or crossing tracks."""
    started = time.perf_counter()
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    selected = []
    for index, (box, mask) in enumerate(zip(boxes, masks)):
        needs = not active
        if active:
            candidates = []
            for track in active:
                iou = self._compute_iou(track.predicted_bbox, box)
                distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
                candidates.append((iou - .02 * distance, iou, distance, track))
            _, best_iou, best_distance, best = max(candidates, key=lambda x: x[0])
            needs = (best.state in (_tracker.TrackState.TENTATIVE, _tracker.TrackState.OCCLUDED, _tracker.TrackState.LOST)
                     or best.reid_age >= self.reid_interval
                     or best.prediction_uncertainty >= .25
                     or (best_iou < self.association_min_iou and best_distance > self.weak_motion_distance))
            if any(self._compute_iou(box, other) >= .10 for other in boxes if not np.array_equal(other, box)):
                needs = True
        if needs:
            crop = self._prepare_reid_crop(image, box, mask)
            if crop is not None and crop.size:
                selected.append((index, crop))
    prep_ms = (time.perf_counter() - started) * 1000.0
    bodies = [None for _ in boxes]
    inference_started = time.perf_counter()
    if selected:
        try:
            crops = [crop for _, crop in selected]
            extracted = self.extractor.batch_extract(crops) if hasattr(self.extractor, 'batch_extract') else [self.extractor.extract(c) for c in crops]
            if len(extracted) != len(selected):
                raise ValueError(f'ReID output count mismatch: expected {len(selected)}, got {len(extracted)}')
            for (index, _), body in zip(selected, extracted):
                ok, value, reason = validate_embedding(body, EXPECTED_BODY_DIM)
                if ok:
                    bodies[index] = value
                else:
                    logger.debug('REID_INVALID detection=%s reason=%s', index, reason)
        except Exception:
            logger.exception('Selective batch Re-ID extraction failed; preserving cached descriptors.')
    self.last_profile['reid_preprocessing_ms'] = prep_ms
    self.last_profile['reid_inference_ms'] = (time.perf_counter() - inference_started) * 1000.0
    self.last_profile['reid_calls'] = 1 if selected else 0
    self.last_profile['reid_crops'] = len(selected)
    return bodies


# Activate the hardened class and selective scheduler for every normal tracker
# construction path after app.core imports this module.
_tracker.Track = SafeTrack
Track = SafeTrack
ReIDTracker = _tracker.ReIDTracker
TrackState = _tracker.TrackState
IdentityState = _tracker.IdentityState
GateResult = _tracker.GateResult
FacePersonMatch = _tracker.FacePersonMatch
ReIDTracker._appearance_refresh_required = _needs_fresh_appearance
ReIDTracker._extract_embeddings_batch = _selective_extract_embeddings_batch

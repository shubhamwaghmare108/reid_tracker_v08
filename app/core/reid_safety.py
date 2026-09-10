"""Centralized Re-ID embedding validation and track-state hardening for V08."""
from __future__ import annotations
from collections import Counter
import logging
import numpy as np
from app.core import tracker as _tracker

logger = logging.getLogger(__name__)


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
    if not np.isfinite(norm):
        return False, None, "NAN_OR_INF"
    if norm <= 1e-8:
        return False, None, "ZERO_NORM"
    return True, value.astype(np.float32, copy=False), "OK"


_BaseTrack = _tracker.Track


class SafeTrack(_BaseTrack):
    """Track that never lets an invalid ReID result overwrite valid state."""

    def __post_init__(self):
        super().__post_init__()
        self.reid_validation_stats = Counter()

    def _validated(self, embedding, current):
        current_ok, current_value, _ = validate_embedding(current)
        expected_dim = len(current_value) if current_ok else None
        valid, value, reason = validate_embedding(embedding, expected_dim)
        self.reid_validation_stats[reason] += 1
        if not valid:
            logger.debug("REID_INVALID track_id=%s reason=%s", self.tracker_id, reason)
            return None
        return value

    def update(self, box, body, face, confidence, update_gallery=False,
               recovered=False, detection_confidence=1.0):
        body = self._validated(body, self.body_embedding)
        face = self._validated(face, self.face_embedding)
        return super().update(
            box, body, face, confidence,
            update_gallery=update_gallery,
            recovered=recovered,
            detection_confidence=detection_confidence,
        )


# ReIDTracker resolves Track from its module globals at runtime. Replacing the
# module-global class therefore hardens all normal Track construction paths.
_tracker.Track = SafeTrack
Track = SafeTrack
ReIDTracker = _tracker.ReIDTracker
TrackState = _tracker.TrackState
IdentityState = _tracker.IdentityState
GateResult = _tracker.GateResult
FacePersonMatch = _tracker.FacePersonMatch

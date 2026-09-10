"""Centralized Re-ID validation and track-state hardening for V08."""
from __future__ import annotations
from collections import Counter
import logging
import numpy as np
from app.core import tracker as _tracker

logger = logging.getLogger(__name__)

# TorchReID OSNet x1_0 returns 512-D descriptors. Keep this explicit so a
# legacy 256-D zero placeholder can never masquerade as a valid body feature.
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
    if not np.isfinite(norm):
        return False, None, "NAN_OR_INF"
    if norm <= 1e-8:
        return False, None, "ZERO_NORM"
    value = value.astype(np.float32, copy=False)
    value = value / max(float(np.linalg.norm(value)), 1e-12)
    return True, value, "OK"


_BaseTrack = _tracker.Track


class SafeTrack(_BaseTrack):
    """Track that never lets an invalid ReID result overwrite valid state."""

    def __post_init__(self):
        super().__post_init__()
        # The base V08 class historically inserted a 256-D zero vector for a
        # missing first body descriptor. Remove that sentinel completely.
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
        # Body must be OSNet x1_0-compatible. Invalid results are converted to
        # None, so BaseTrack.update leaves the previous valid descriptor intact.
        body = self._validated(body, self.body_embedding, EXPECTED_BODY_DIM)
        # Face dimensions are discovered from the existing valid face embedding;
        # this avoids hardcoding an InsightFace model-specific size.
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

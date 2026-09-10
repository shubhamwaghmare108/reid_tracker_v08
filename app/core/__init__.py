"""Core package initialization.

Import the ReID safety layer before tracker consumers construct tracks. The
layer validates descriptors and installs the V08.3 association/lifecycle fixes.
"""
from app.core import reid_safety as _reid_safety


def _safe_new_track_conflicts(self, box, body, face):
    """Delegate to the original duplicate gate and audit fragmentation risk."""
    conflict = _reid_safety._original_new_track_conflicts(self, box, body, face)
    if not conflict:
        for track in self.tracks:
            if track.state in (_reid_safety.TrackState.OCCLUDED, _reid_safety.TrackState.LOST):
                distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
                if distance <= self.recovery_motion_gate * 1.25:
                    self._record_metric('TRACK_FRAGMENT_CANDIDATE', track, rejection_reason='NEW_ID_NEAR_LOST_TRACK')
                    break
    return conflict


_reid_safety.ReIDTracker._new_track_conflicts = _safe_new_track_conflicts

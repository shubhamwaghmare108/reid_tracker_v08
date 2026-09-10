"""Core package initialization and V08.3.1 runtime tuning."""
from app.core import reid_safety as _reid_safety
from app.core.metrics import MetricsCollector

_Tracker = _reid_safety.ReIDTracker
_BaseGate = _Tracker._passes_association_gates
_BaseAssociate = _Tracker._associate_detections
_BaseMetricsRecord = MetricsCollector.record
_OriginalNewTrackConflicts = _reid_safety._original_new_track_conflicts


def _safe_new_track_conflicts(self, box, body, face):
    """Block a strong near-lost duplicate from creating a fresh ID."""
    if _OriginalNewTrackConflicts(self, box, body, face):
        return True
    for track in self.tracks:
        if track.state not in (_reid_safety.TrackState.OCCLUDED, _reid_safety.TrackState.LOST):
            continue
        distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
        iou = self._compute_iou(track.predicted_bbox, box)
        _, b, f = self._gallery_similarity(track, body, face)
        if distance <= self.recovery_motion_gate * 1.5 and ((max(b, f) >= 0.50 and track.hits >= 3) or iou >= 0.15):
            self._record_metric('TRACK_FRAGMENT_CANDIDATE', track, rejection_reason='NEW_ID_NEAR_LOST_TRACK')
            return True
    return False


def _face_refresh_needed(self):
    """Do not run expensive face inference merely because two people cross."""
    active = [t for t in self.tracks if t.state != _reid_safety.TrackState.DELETED]
    if not active or self.frame_count <= 2:
        return True
    if self.frame_count % 30 == 0:
        return True
    return any((t.state in (_reid_safety.TrackState.OCCLUDED, _reid_safety.TrackState.LOST) and t.missed_frames in (1, 6)) or (t.identity_state == _reid_safety.IdentityState.CANDIDATE and t.identity_candidate_frames in (2, 5)) for t in active)


def _passes_association_gates(self, track, box, body, face, recovery=False):
    """Relax only short-gap recovery geometry; retain V08.3 safety otherwise."""
    if recovery:
        iou = self._compute_iou(track.predicted_bbox, box)
        distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
        if track.missed_frames <= 2 and distance <= self.recovery_motion_gate * 1.35 and (iou >= .03 or distance <= .90):
            return _reid_safety.GateResult(True)
        if track.missed_frames <= 6 and distance <= self.recovery_motion_gate * 1.15 and iou >= .10:
            return _reid_safety.GateResult(True)
        old_threshold, old_scale = self.recovery_reid_threshold, self.association_max_scale_change
        try:
            self.recovery_reid_threshold = min(old_threshold, .50)
            self.association_max_scale_change = max(old_scale, 3.25)
            return _BaseGate(self, track, box, body, face, recovery=True)
        finally:
            self.recovery_reid_threshold, self.association_max_scale_change = old_threshold, old_scale
    return _BaseGate(self, track, box, body, face, recovery=False)


def _associate_detections(self, tracks, boxes, bodies, faces, recovery=False):
    """Skip recovery cooldown tracks instead of evaluating doomed pairs."""
    if not recovery:
        return _BaseAssociate(self, tracks, boxes, bodies, faces, recovery=False)
    eligible = [(idx, track) for idx, track in enumerate(tracks) if track.missed_frames <= self.max_recovery_frames and (track.missed_frames <= 1 or track.missed_frames % 5 == 1)]
    if not eligible:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    local_tracks = [track for _, track in eligible]
    matches, local_unmatched, unmatched_dets = _BaseAssociate(self, local_tracks, boxes, bodies, faces, recovery=True)
    local_to_global = {local: global_idx for local, (global_idx, _) in enumerate(eligible)}
    matches = [(local_to_global[r], c, score) for r, c, score in matches]
    eligible_global = set(local_to_global.values())
    unmatched_tracks = [local_to_global[r] for r in local_unmatched]
    unmatched_tracks.extend(idx for idx in range(len(tracks)) if idx not in eligible_global)
    return matches, sorted(set(unmatched_tracks)), unmatched_dets


def _metrics_record(self, event_type, frame, track=None, **values):
    """Prevent duplicate deletion events from inflating lifecycle metrics."""
    if event_type == 'TRACK_DELETED' and track is not None:
        seen = getattr(self, '_v0831_deleted_events', None)
        if seen is None:
            seen = set()
            self._v0831_deleted_events = seen
        key = (int(frame), int(track.tracker_id))
        if key in seen:
            return
        seen.add(key)
    return _BaseMetricsRecord(self, event_type, frame, track=track, **values)


def _identity_assignment(self):
    """Run normal identity assignment and record a one-time gallery diagnostic."""
    if self.frame_count == 1:
        self._record_metric('GALLERY_STATUS', count=len(getattr(self.gallery, 'names', [])),
                            body_count=int(getattr(getattr(self.gallery, 'body_means', []), 'shape', [0])[0]) if getattr(self.gallery, 'body_means', None) is not None and getattr(self.gallery.body_means, 'ndim', 0) > 0 else 0)
    return _BaseIdentityAssignment(self)


_BaseIdentityAssignment = _Tracker._frame_level_identity_assignment

# reid_safety._association_score resolves _passes_association_gates by module
# global lookup, so replace that module reference as well as the class method.
_reid_safety._passes_association_gates = _passes_association_gates
_Tracker._face_refresh_needed = _face_refresh_needed
_Tracker._passes_association_gates = _passes_association_gates
_Tracker._associate_detections = _associate_detections
_Tracker._new_track_conflicts = _safe_new_track_conflicts
_Tracker._frame_level_identity_assignment = _identity_assignment
MetricsCollector.record = _metrics_record

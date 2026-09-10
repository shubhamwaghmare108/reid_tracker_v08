"""V08.3 safety and association layer.

The module keeps the existing detector/Kalman/OSNet/face/gallery interfaces but
replaces the fragile V08.2 seams with bounded track lifecycle, adaptive
association, event-driven face inference, and identity continuity.
"""
from __future__ import annotations

from collections import Counter, deque
import logging
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core import tracker as _tracker
from app.core.metrics import MetricsCollector

logger = logging.getLogger(__name__)
_BaseTrack = _tracker.Track


def validate_embedding(embedding, expected_dim=None):
    if embedding is None:
        return False, None, "NONE"
    try:
        value = np.asarray(embedding)
    except Exception:
        return False, None, "MALFORMED"
    if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
        return False, None, "EMPTY_OR_DTYPE"
    value = value.reshape(-1)
    if expected_dim is not None and value.size != expected_dim:
        return False, None, "WRONG_DIMENSION"
    if not np.isfinite(value).all():
        return False, None, "NAN_OR_INF"
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-8:
        return False, None, "ZERO_NORM"
    value = value.astype(np.float32, copy=False)
    return True, value / norm, "OK"


class SafeTrack(_BaseTrack):
    """Track that never lets invalid appearance overwrite valid evidence."""

    def __post_init__(self):
        super().__post_init__()
        ok, body, _ = validate_embedding(self.body_embedding)
        self.body_embedding = body if ok else None
        self.last_body_embedding = body.copy() if ok else None
        self.reid_validation_stats = Counter()
        self._recovery_last_attempt_frame = -10**9
        self._identity_candidate_scores = deque(maxlen=8)
        self._identity_candidate_name = None
        self._last_crossing_frame = -10**9
        self._identity_confirmed_once = False

    def _validated(self, embedding, current):
        current_ok, current_value, _ = validate_embedding(current)
        expected_dim = len(current_value) if current_ok else None
        ok, value, reason = validate_embedding(embedding, expected_dim)
        self.reid_validation_stats[reason] += 1
        return value if ok else None

    def update(self, box, body, face, confidence, update_gallery=False,
               recovered=False, detection_confidence=1.0):
        body = self._validated(body, self.body_embedding)
        face = self._validated(face, self.face_embedding)
        previous_identity = self.name
        previous_identity_state = self.identity_state
        result = super().update(box, body, face, confidence,
                                update_gallery=update_gallery,
                                recovered=recovered,
                                detection_confidence=detection_confidence)
        if previous_identity != 'Unknown' and self.name == 'Unknown':
            self.name = previous_identity
            self.identity_state = previous_identity_state
        return result


def _crossing_pair(self, first, second):
    if first.state == _tracker.TrackState.DELETED or second.state == _tracker.TrackState.DELETED:
        return False
    c1 = first.centre(first.predicted_bbox)
    c2 = second.centre(second.predicted_bbox)
    scale = max(self._scale(first.last_reliable_bbox), self._scale(second.last_reliable_bbox))
    distance = float(np.linalg.norm(c1 - c2))
    iou = self._compute_iou(first.predicted_bbox, second.predicted_bbox)
    if iou >= 0.12:
        return True
    if distance > max(0.80 * scale, 100.0):
        return False
    if not first.velocity_history or not second.velocity_history:
        return False
    v1, v2 = first.velocity_history[-1], second.velocity_history[-1]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1.0 or n2 < 1.0:
        return False
    direction_dot = float(np.dot(v1 / n1, v2 / n2))
    return direction_dot < -0.20 and float(np.linalg.norm(v1 - v2)) > 1.0


def _crossing_for_detection(self, track, box):
    for other in self.tracks:
        if other is track or other.state == _tracker.TrackState.DELETED:
            continue
        if self._crossing_pair(track, other):
            d = self._distance(other.predicted_bbox, box) / self._scale(other.last_reliable_bbox)
            if self._compute_iou(other.predicted_bbox, box) >= 0.02 or d <= 1.5:
                track._last_crossing_frame = self.frame_count
                return True
    return False


def _appearance_refresh_required(self, boxes):
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    if not boxes:
        return False
    if not active or self.frame_count <= 2:
        return True
    if self.frame_count % self.reid_interval == 0:
        return True
    if any(t.reid_age >= self.reid_interval for t in active):
        return True
    if any(t.prediction_uncertainty >= 0.65 or t.motion_confidence < 0.25 for t in active):
        return True
    return any(self._crossing_pair(a, b) for i, a in enumerate(active) for b in active[i + 1:])


def _face_refresh_needed(self):
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    if not active or self.frame_count <= 2:
        return True
    if self.frame_count % 30 == 0:
        return True
    for t in active:
        if t.state in (_tracker.TrackState.OCCLUDED, _tracker.TrackState.LOST) and t.missed_frames in (1, 6):
            return True
        if t.identity_state == _tracker.IdentityState.CANDIDATE and t.identity_candidate_frames in (2, 5):
            return True
    for i, first in enumerate(active):
        for second in active[i + 1:]:
            if self._crossing_pair(first, second) and self.frame_count - getattr(first, '_last_crossing_frame', -10**9) >= 10:
                return True
    return False


def _update_v083(self, image, detections):
    original_face = self.face_processor
    # V08.2 allowed configuration values to keep dead tracks for 90+ frames.
    # V08.3 enforces a bounded lifecycle even when an old environment overrides
    # those settings, preventing stale tracks from exploding association cost.
    original_limits = (self.max_lost_frames, self.max_occlusion_frames, self.max_recovery_frames)
    self.max_lost_frames = min(int(self.max_lost_frames), 8)
    self.max_occlusion_frames = min(int(self.max_occlusion_frames), 6)
    self.max_recovery_frames = min(int(self.max_recovery_frames), 24)
    if not _face_refresh_needed(self):
        self.face_processor = None
    try:
        result = _original_update(self, image, detections)
    finally:
        self.face_processor = original_face
        self.max_lost_frames, self.max_occlusion_frames, self.max_recovery_frames = original_limits
    return result


def _selective_extract_embeddings_batch(self, image, boxes, masks):
    started = time.perf_counter()
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    recoverable = [t for t in active if t.state in (_tracker.TrackState.OCCLUDED, _tracker.TrackState.LOST)]
    selected = []
    periodic = self.frame_count <= 2 or self.frame_count % self.reid_interval == 0
    for index, (box, mask) in enumerate(zip(boxes, masks)):
        best = None
        best_iou, best_d = -1.0, 1e9
        for t in active:
            iou = self._compute_iou(t.predicted_bbox, box)
            d = self._distance(t.predicted_bbox, box) / self._scale(t.last_reliable_bbox)
            if iou > best_iou or (iou == best_iou and d < best_d):
                best, best_iou, best_d = t, iou, d
        needs = periodic
        if best is not None:
            needs = needs or best.state in (_tracker.TrackState.TENTATIVE, _tracker.TrackState.OCCLUDED, _tracker.TrackState.LOST)
            needs = needs or best.reid_age >= self.reid_interval or best.prediction_uncertainty >= 0.65 or best_d > self.motion_gate_threshold
            if any(self._crossing_pair(best, other) for other in active if other is not best):
                needs = True
        for t in recoverable:
            d = self._distance(t.predicted_bbox, box) / self._scale(t.last_reliable_bbox)
            if self._compute_iou(t.predicted_bbox, box) >= 0.01 or d <= self.recovery_motion_gate * 1.25:
                needs = True
                break
        if needs:
            crop = self._prepare_reid_crop(image, box, mask)
            if crop is not None and getattr(crop, 'size', 0):
                selected.append((index, crop))
    bodies = [None] * len(boxes)
    inference_started = time.perf_counter()
    if selected:
        try:
            crops = [crop for _, crop in selected]
            extracted = self.extractor.batch_extract(crops) if hasattr(self.extractor, 'batch_extract') else [self.extractor.extract(c) for c in crops]
            if len(extracted) != len(selected):
                raise ValueError(f'ReID output count mismatch: expected {len(selected)}, got {len(extracted)}')
            for (index, _), body in zip(selected, extracted):
                ok, value, reason = validate_embedding(body)
                if ok:
                    bodies[index] = value
                else:
                    logger.debug('Invalid ReID descriptor detection=%s reason=%s', index, reason)
        except Exception:
            logger.exception('Selective batch ReID failed; cached descriptors remain intact.')
    self.last_profile['reid_preprocessing_ms'] = (time.perf_counter() - started) * 1000.
    self.last_profile['reid_inference_ms'] = (time.perf_counter() - inference_started) * 1000.
    self.last_profile['reid_calls'] = 1 if selected else 0
    self.last_profile['reid_crops'] = len(selected)
    return bodies


def _passes_association_gates(self, track, box, body, face, recovery=False):
    iou = self._compute_iou(track.predicted_bbox, box)
    distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
    _, b, f = self._gallery_similarity(track, body, face)
    appearance = max(b, f)
    crossing = self._crossing_for_detection(track, box)
    strong = appearance >= self.strong_reid_gate
    has_appearance = appearance >= 0.0
    if recovery:
        if track.missed_frames > self.max_recovery_frames:
            return _tracker.GateResult(False, 'RECOVERY_EXPIRED')
        if track.missed_frames > 1 and track.missed_frames % 5 != 1:
            return _tracker.GateResult(False, 'RECOVERY_COOLDOWN')
        gate = self.recovery_motion_gate * (1.50 if strong or crossing else 1.0)
        if distance > gate:
            return _tracker.GateResult(False, 'MOTION_GATE_FAIL')
        old = track.last_reliable_bbox
        wr = (box[2] - box[0]) / max(old[2] - old[0], 1.)
        hr = (box[3] - box[1]) / max(old[3] - old[1], 1.)
        limit = self.association_max_scale_change * (1.50 if strong or crossing else 1.0)
        if not (1 / limit <= wr <= limit and 1 / limit <= hr <= limit):
            return _tracker.GateResult(False, 'SCALE_GATE_FAIL')
        if track.missed_frames >= 3 and not has_appearance:
            return _tracker.GateResult(False, 'RECOVERY_REID_GATE_FAIL')
        if has_appearance and appearance < self.recovery_reid_threshold and not (iou >= 0.30 and distance <= 0.8):
            return _tracker.GateResult(False, 'RECOVERY_REID_GATE_FAIL')
        return _tracker.GateResult(True)
    gate = self.motion_gate_threshold * (1.50 if strong or crossing else 1.0)
    if distance > gate:
        return _tracker.GateResult(False, 'MOTION_GATE_FAIL')
    old = track.last_reliable_bbox
    wr = (box[2] - box[0]) / max(old[2] - old[0], 1.)
    hr = (box[3] - box[1]) / max(old[3] - old[1], 1.)
    limit = self.association_max_scale_change * (1.25 if strong else 1.0)
    if not (1 / limit <= wr <= limit and 1 / limit <= hr <= limit):
        return _tracker.GateResult(False, 'SCALE_GATE_FAIL')
    if iou < self.association_min_iou and not strong and distance > self.weak_motion_distance:
        return _tracker.GateResult(False, 'IOU_GATE_FAIL')
    if self.valid(body) and b < 0.18 and iou < 0.12 and not crossing:
        return _tracker.GateResult(False, 'REID_GATE_FAIL')
    if not has_appearance and iou < 0.02 and distance > 1.5:
        return _tracker.GateResult(False, 'NO_SAFE_MATCH')
    return _tracker.GateResult(True)


def _association_score(self, track, box, body, face, recovery=False):
    gate = _passes_association_gates(self, track, box, body, face, recovery)
    iou = self._compute_iou(track.predicted_bbox, box)
    distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
    _, b, f = self._gallery_similarity(track, body, face)
    if not gate.accepted:
        track.association_reason = gate.reason
        self._record_metric('ASSOCIATION_REJECTED', track, iou_score=iou, body_score=b, face_score=f,
                            motion_score=max(0., 1. - distance / max(self.motion_gate_threshold, 1e-6)),
                            rejection_reason=gate.reason)
        return None
    crossing = self._crossing_for_detection(track, box)
    has_body, has_face = self.valid(body), self.valid(face)
    appearance = .75 * max(b, 0.) + .25 * max(f, 0.) if has_face else max(b, 0.)
    motion_gate = (self.recovery_motion_gate if recovery else self.motion_gate_threshold) * (1.5 if crossing else 1.0)
    motion = max(0., 1. - distance / max(motion_gate, 1e-6))
    old = track.last_reliable_bbox
    wr = (box[2] - box[0]) / max(old[2] - old[0], 1.)
    hr = (box[3] - box[1]) / max(old[3] - old[1], 1.)
    scale_score = float(np.exp(-0.5 * (abs(np.log(max(wr, 1e-3))) + abs(np.log(max(hr, 1e-3))))))
    if recovery:
        if not (has_body or has_face):
            return .45 * motion + .40 * iou + .15 * scale_score
        return .62 * appearance + .23 * motion + .10 * iou + .05 * scale_score
    if crossing:
        if has_body or has_face:
            return .58 * appearance + .25 * motion + .10 * iou + .07 * scale_score
        return .45 * motion + .35 * iou + .20 * scale_score
    if has_body or has_face:
        return .45 * appearance + .33 * motion + .15 * iou + .07 * scale_score
    return .40 * motion + .42 * iou + .18 * scale_score


def _associate_detections(self, tracks, boxes, bodies, faces, recovery=False):
    if not tracks or not boxes:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    pairs = []
    scores = {}
    for i, track in enumerate(tracks):
        radius = self.recovery_motion_gate * 1.25 if recovery else self.motion_gate_threshold * 1.25
        for j, box in enumerate(boxes):
            iou = self._compute_iou(track.predicted_bbox, box)
            d = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
            if iou >= 0.01 or d <= radius:
                pairs.append((i, j))
    if not pairs:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    cost = np.full((len(tracks), len(boxes)), 1e6, dtype=np.float32)
    for i, j in pairs:
        self._record_metric('ASSOCIATION_ATTEMPT', tracks[i], detection_id='', track_state=tracks[i].state, identity_state=tracks[i].identity_state)
        score = _association_score(self, tracks[i], boxes[j], bodies[j], faces[j], recovery)
        if score is not None:
            scores[(i, j)] = float(score)
            cost[i, j] = 1.0 - float(score)
    rows, cols = linear_sum_assignment(cost)
    minimum = 0.30 if not recovery else 0.45
    margin = max(0.05, self.recovery_margin * (0.75 if not recovery else 1.0))
    matches, unmatched_tracks, unmatched_dets = [], set(range(len(tracks))), set(range(len(boxes)))
    for r, c in zip(rows, cols):
        score = scores.get((r, c))
        if score is None or score < minimum:
            continue
        alternatives = [v for (rr, cc), v in scores.items() if (rr == r and cc != c) or (cc == c and rr != r)]
        second = max(alternatives, default=-1.)
        if second >= 0. and score - second < margin:
            reason = 'RECOVERY_AMBIGUOUS' if recovery else 'ASSOCIATION_AMBIGUOUS'
            tracks[r].association_reason = reason
            self._record_metric('ASSOCIATION_REJECTED', tracks[r], rejection_reason=reason,
                                best_score=score, second_best_score=second, association_margin=score-second)
            if recovery:
                self._record_metric('RECOVERY_AMBIGUOUS', tracks[r], best_score=score,
                                    second_best_score=second, recovery_margin=score-second,
                                    recovery_result='RECOVERY_AMBIGUOUS', rejection_reason=reason)
            continue
        matches.append((r, c, score))
        unmatched_tracks.discard(r)
        unmatched_dets.discard(c)
    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def _new_track_conflicts(self, box, body, face):
    conflict = _tracker.ReIDTracker._new_track_conflicts(self, box, body, face)
    if not conflict:
        # A new ID beside a recently lost/occluded track is a measurable
        # fragmentation candidate. It is not automatically rejected because
        # two nearby people can legitimately be different identities.
        for t in self.tracks:
            if t.state in (_tracker.TrackState.OCCLUDED, _tracker.TrackState.LOST):
                d = self._distance(t.predicted_bbox, box) / self._scale(t.last_reliable_bbox)
                if d <= self.recovery_motion_gate * 1.25:
                    self._record_metric('TRACK_FRAGMENT_CANDIDATE', t, detection_confidence=0.0,
                                        rejection_reason='NEW_ID_NEAR_LOST_TRACK')
                    break
    return conflict


def _frame_level_identity_assignment(self):
    active = [t for t in self.tracks if t.state in (_tracker.TrackState.CONFIRMED, _tracker.TrackState.RECOVERED)]
    if not active or not self.gallery.has_body():
        return
    proposals = []
    for track in active:
        body_scores, face_scores = self._identity_scores(track)
        bi, bs, bm = self._best_identity(body_scores)
        fi, fs, fm = self._best_identity(face_scores)
        track.body_score, track.face_score = bs, fs
        track.body_score_history.append(bs); track.face_score_history.append(fs)
        if fi is not None and fs >= self.identity_face_threshold and fm >= self.identity_margin:
            proposals.append((track, fi, fs, fm, 'FACE'))
            continue
        candidate = self.gallery.names[bi] if bi is not None and bs >= self.identity_body_candidate_threshold and bm >= self.identity_margin else None
        if candidate != getattr(track, '_identity_candidate_name', None):
            track._identity_candidate_name = candidate
            track._identity_candidate_scores.clear()
        if candidate is not None:
            track._identity_candidate_scores.append(bs)
            track.identity_candidate = candidate
            track.identity_candidate_frames = len(track._identity_candidate_scores)
            track.identity_margin = bm
            track.identity_source = 'BODY'
            track.identity_state = _tracker.IdentityState.CANDIDATE
            stable = (len(track._identity_candidate_scores) >= max(3, min(self.identity_candidate_min_frames, 4))
                      and float(np.mean(track._identity_candidate_scores)) >= max(.74, self.identity_body_confirm_threshold - .06)
                      and bs >= max(.68, self.identity_body_confirm_threshold - .10))
            if stable:
                proposals.append((track, bi, float(np.mean(track._identity_candidate_scores)), bm, 'BODY_HISTORY'))
        elif track.identity_state not in (_tracker.IdentityState.CONFIRMED, _tracker.IdentityState.RETAINED):
            track.identity_candidate = None
            track.identity_candidate_frames = 0
            track.identity_state = _tracker.IdentityState.UNKNOWN
    for track, index, score, margin, source in sorted(proposals, key=lambda x: (x[4] == 'FACE', x[2]), reverse=True):
        name = self.gallery.names[index]
        owner_id = self.identity_owners.get(name)
        owner = next((t for t in self.tracks if t.tracker_id == owner_id and t.state != _tracker.TrackState.DELETED), None)
        if owner is not None and owner is not track:
            track.identity_rejection_reason = 'CANONICAL_OWNER_EXISTS'
            track.identity_state = _tracker.IdentityState.CANDIDATE
            self._record_metric('IDENTITY_OWNER_CONFLICT', track, identity=name, rejection_reason='IDENTITY_OWNER_CONFLICT')
            continue
        previous = track.name
        track.name, track.score = name, score
        track.identity_score, track.identity_margin = score, margin
        track.identity_state = _tracker.IdentityState.CONFIRMED
        track.identity_source = source
        track.identity_rejection_reason = ''
        track.identity_candidate = name
        track.identity_candidate_frames = max(track.identity_candidate_frames, self.identity_candidate_min_frames)
        self.identity_owners[name] = track.tracker_id
        if not getattr(track, '_identity_confirmed_once', False):
            self._record_metric('IDENTITY_CONFIRMED', track, previous_identity=previous, current_identity=name)
            track._identity_confirmed_once = True
        elif previous != name and previous != 'Unknown':
            track.identity_switch_count += 1
            self._record_metric('IDENTITY_SWITCH', track, previous_identity=previous, current_identity=name)


_original_update = _tracker.ReIDTracker.update
_original_start_session = MetricsCollector.start_session
_original_new_track_conflicts = _tracker.ReIDTracker._new_track_conflicts


def _start_session_v083(self, video_name, fps, configuration):
    _original_start_session(self, video_name, fps, configuration)
    self.session['tracker_version'] = 'V08.3'
    if isinstance(configuration, dict):
        configuration['effective_lifecycle'] = {'max_lost_frames': 8, 'max_occlusion_frames': 6, 'max_recovery_frames': 24}
        configuration['face_schedule'] = 'event-driven; first 2 frames, recovery/candidate events, crossing cooldown, 30-frame safety refresh'


_tracker.Track = SafeTrack
Track = SafeTrack
ReIDTracker = _tracker.ReIDTracker
TrackState = _tracker.TrackState
IdentityState = _tracker.IdentityState
GateResult = _tracker.GateResult
FacePersonMatch = _tracker.FacePersonMatch

ReIDTracker._crossing_pair = _crossing_pair
ReIDTracker._crossing_for_detection = _crossing_for_detection
ReIDTracker._appearance_refresh_required = _appearance_refresh_required
ReIDTracker._extract_embeddings_batch = _selective_extract_embeddings_batch
ReIDTracker._passes_association_gates = _passes_association_gates
ReIDTracker._compute_association_cost = _association_score
ReIDTracker._associate_detections = _associate_detections
ReIDTracker._new_track_conflicts = _new_track_conflicts
ReIDTracker._frame_level_identity_assignment = _frame_level_identity_assignment
ReIDTracker.update = _update_v083
MetricsCollector.start_session = _start_session_v083

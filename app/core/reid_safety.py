"""V08.1 tracking hardening.

This module is intentionally small: it patches the existing tracker at its
public association/appearance seams while keeping the detector, Kalman state,
gallery, identity ownership, database, and output pipeline intact.

V08.1 fixes the root causes seen in the crowded benchmark:
- no scene-wide periodic ReID refresh;
- no scene-wide face inference on stable frames;
- scale/IoU are soft association cues instead of brittle hard gates;
- missing appearance is neutral rather than a negative score;
- crossings increase appearance weight and ambiguity protection;
- normal matches also require a best-vs-second-best margin;
- invalid descriptors never overwrite a valid track descriptor.
"""
from __future__ import annotations

from collections import Counter
import logging
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core import tracker as _tracker
from app.core.metrics import MetricsCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descriptor safety
# ---------------------------------------------------------------------------
def validate_embedding(embedding, expected_dim: int | None = None):
    """Return ``(valid, normalized_embedding, reason)`` without bad data."""
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
        return False, None, "ZERO_NORM"
    value = value.astype(np.float32, copy=False)
    return True, value / norm, "OK"


_BaseTrack = _tracker.Track


class SafeTrack(_BaseTrack):
    """Track that never lets an invalid descriptor erase valid evidence."""
    def __post_init__(self):
        super().__post_init__()
        ok, value, _ = validate_embedding(self.body_embedding)
        if ok:
            self.body_embedding = value
            self.last_body_embedding = value.copy()
        else:
            self.body_embedding = None
            self.last_body_embedding = None
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
        # Validate before the base Track can assign anything. None means
        # "descriptor unavailable", not "replace the previous descriptor".
        body = self._validated(body, self.body_embedding)
        face = self._validated(face, self.face_embedding)
        return super().update(box, body, face, confidence,
                              update_gallery=update_gallery,
                              recovered=recovered,
                              detection_confidence=detection_confidence)


# ---------------------------------------------------------------------------
# Event-driven appearance scheduling
# ---------------------------------------------------------------------------
def _crossing_pair(self, first, second):
    """Return True when two tracks are plausibly approaching/crossing."""
    iou = self._compute_iou(first.predicted_bbox, second.predicted_bbox)
    c1, c2 = first.centre(first.predicted_bbox), second.centre(second.predicted_bbox)
    distance = float(np.linalg.norm(c1 - c2))
    scale = max(self._scale(first.last_reliable_bbox), self._scale(second.last_reliable_bbox))
    close = distance <= max(0.65 * scale, 80.0)
    if iou >= 0.10:
        return True
    if not close or not first.velocity_history or not second.velocity_history:
        return False
    v1 = first.velocity_history[-1]
    v2 = second.velocity_history[-1]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-3 or n2 < 1e-3:
        return False
    # Negative dot product means opposing motion, the common crossing case.
    approaching = float(np.dot(v1 / n1, v2 / n2)) < -0.20
    relative = c2 - c1
    closing = float(np.dot(v1 - v2, relative)) > 0.0
    return approaching and closing


def _appearance_refresh_required_v081(self, boxes):
    """Refresh appearance only for uncertain/new/occluded/crossing situations.

    There is deliberately no ``frame_count % reid_interval`` scene cadence.
    ``reid_interval`` remains a per-track stale-descriptor age limit.
    """
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    if not active:
        return bool(boxes)

    for track in active:
        if track.state in (_tracker.TrackState.TENTATIVE,
                           _tracker.TrackState.OCCLUDED,
                           _tracker.TrackState.LOST):
            return True
        if track.reid_age >= self.reid_interval:
            # A stale descriptor is enough to request a body refresh, but this
            # does not imply a face refresh on every frame; face work is still
            # performed only when this refresh is actually needed.
            return True
        if track.prediction_uncertainty >= 0.40 or track.motion_confidence < 0.30:
            return True

    # Crossing is the one scene interaction that justifies an appearance pass.
    for i, first in enumerate(active):
        for second in active[i + 1:]:
            if self._crossing_pair(first, second):
                return True

    # If a detection has no plausible motion candidate, appearance can rescue
    # it; stable detections otherwise stay motion/geometry-only.
    for box in boxes:
        if not any(self._distance(t.predicted_bbox, box) /
                   self._scale(t.last_reliable_bbox) <= self.motion_gate_threshold
                   for t in active):
            return True
    return False


def _selective_extract_embeddings_batch_v081(self, image, boxes, masks):
    """Batch body descriptors only for detections that need appearance."""
    started = time.perf_counter()
    active = [t for t in self.tracks if t.state != _tracker.TrackState.DELETED]
    selected = []

    for index, (box, mask) in enumerate(zip(boxes, masks)):
        needs = not active
        best = None
        if active:
            candidates = []
            for track in active:
                iou = self._compute_iou(track.predicted_bbox, box)
                distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
                candidates.append((iou - 0.02 * distance, iou, distance, track))
            _, best_iou, best_distance, best = max(candidates, key=lambda x: x[0])
            needs = (best.state in (_tracker.TrackState.TENTATIVE,
                                    _tracker.TrackState.OCCLUDED,
                                    _tracker.TrackState.LOST)
                     or best.reid_age >= self.reid_interval
                     or best.prediction_uncertainty >= 0.40
                     or best.motion_confidence < 0.30
                     or best_distance > self.motion_gate_threshold)

            if any(self._crossing_pair(best, other) for other in active if other is not best):
                needs = True

            # A strong cached descriptor is preferable to extracting body ReID
            # for every stable detection just because another person is nearby.
            if best_iou >= 0.20 and best_distance <= 0.60 and best.state == _tracker.TrackState.CONFIRMED:
                if best.reid_age < self.reid_interval and best.prediction_uncertainty < 0.40:
                    needs = False

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
            if hasattr(self.extractor, 'batch_extract'):
                extracted = self.extractor.batch_extract(crops)
            else:
                extracted = [self.extractor.extract(crop) for crop in crops]
            if len(extracted) != len(selected):
                raise ValueError(
                    f'ReID output count mismatch: expected {len(selected)}, got {len(extracted)}')
            for (index, _), body in zip(selected, extracted):
                ok, value, reason = validate_embedding(body)
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


# ---------------------------------------------------------------------------
# Robust association
# ---------------------------------------------------------------------------
def _crossing_for_detection(self, track, box):
    """Check whether this track is in a local crossing interaction."""
    for other in self.tracks:
        if other is track or other.state == _tracker.TrackState.DELETED:
            continue
        if self._crossing_pair(track, other):
            # Require the detection to be near one member of the interaction.
            if (self._compute_iou(other.predicted_bbox, box) >= 0.02 or
                    self._distance(other.predicted_bbox, box) /
                    self._scale(other.last_reliable_bbox) < 1.25):
                return True
    return False


def _passes_association_gates_v081(self, track, box, body, face, recovery=False):
    """Use gates to remove impossible candidates, not normal detector jitter."""
    iou = self._compute_iou(track.predicted_bbox, box)
    distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
    app, b, f = self._gallery_similarity(track, body, face)
    crossing = self._crossing_for_detection(track, box)
    strong_app = max(b, f) >= self.strong_reid_gate
    usable_app = max(b, f) >= self.normal_reid_gate

    motion_gate = self.recovery_motion_gate if recovery else self.motion_gate_threshold
    if crossing or strong_app:
        motion_gate *= 1.50
    if distance > motion_gate:
        return _tracker.GateResult(False, 'MOTION_GATE_FAIL')

    old_box = track.last_reliable_bbox
    old_w = max(float(old_box[2] - old_box[0]), 1.0)
    old_h = max(float(old_box[3] - old_box[1]), 1.0)
    new_w = max(float(box[2] - box[0]), 1.0)
    new_h = max(float(box[3] - box[1]), 1.0)
    wr, hr = new_w / old_w, new_h / old_h
    max_scale = self.association_max_scale_change
    # Normal scale range is kept, but strong appearance/crossing may tolerate
    # perspective-driven box changes. Only an extreme jump remains a hard fail.
    hard_scale = max_scale * (1.50 if (crossing or strong_app) else 1.0)
    if wr < 1.0 / hard_scale or wr > hard_scale or hr < 1.0 / hard_scale or hr > hard_scale:
        return _tracker.GateResult(False, 'SCALE_GATE_FAIL')

    if recovery:
        if max(b, f) < self.recovery_reid_threshold:
            return _tracker.GateResult(False, 'RECOVERY_REID_GATE_FAIL')
        return _tracker.GateResult(True)

    # A disjoint detection is acceptable when reliable appearance agrees.
    # Otherwise require at least weak geometric continuity.
    if iou < self.association_min_iou and not usable_app and distance > self.weak_motion_distance:
        return _tracker.GateResult(False, 'IOU_GATE_FAIL')
    if iou < self.normal_iou_gate and not usable_app and distance > self.weak_motion_distance:
        return _tracker.GateResult(False, 'NO_SAFE_MATCH')

    # Do not hard-reject on a weak body descriptor during a crossing; let the
    # global assignment + ambiguity margin decide instead.
    if self.valid(body) and b < self.normal_reid_gate and not crossing and iou < self.normal_iou_gate:
        return _tracker.GateResult(False, 'REID_GATE_FAIL')
    return _tracker.GateResult(True)


def _compute_association_cost_v081(self, track, box, body, face, recovery=False):
    """Fuse motion, IoU, scale and available appearance without missing-feature penalties."""
    self._record_metric('ASSOCIATION_ATTEMPT', track, detection_id='',
                        track_state=track.state, identity_state=track.identity_state)
    gate = _passes_association_gates_v081(self, track, box, body, face, recovery)
    iou = self._compute_iou(track.predicted_bbox, box)
    distance = self._distance(track.predicted_bbox, box) / self._scale(track.last_reliable_bbox)
    app, b, f = self._gallery_similarity(track, body, face)
    if not gate.accepted:
        track.association_reason = gate.reason
        self._record_metric('ASSOCIATION_REJECTED', track, iou_score=iou,
                            body_score=b, face_score=f,
                            motion_score=max(0., 1. - distance / max(self.motion_gate_threshold, 1e-6)),
                            rejection_reason=gate.reason)
        return None

    crossing = self._crossing_for_detection(track, box)
    motion_gate = self.recovery_motion_gate if recovery else self.motion_gate_threshold
    if crossing or max(b, f) >= self.strong_reid_gate:
        motion_gate *= 1.50
    motion = max(0., 1. - distance / max(motion_gate, 1e-6))

    # Scale consistency is a soft score. A value near 1 means little size
    # change; it is intentionally not a hard gate here.
    old = track.last_reliable_bbox
    wr = (box[2] - box[0]) / max(old[2] - old[0], 1.0)
    hr = (box[3] - box[1]) / max(old[3] - old[1], 1.0)
    scale_error = abs(np.log(max(wr, 1e-3))) + abs(np.log(max(hr, 1e-3)))
    scale_score = float(np.exp(-0.5 * scale_error))

    has_body = self.valid(body)
    has_face = self.valid(face)
    has_appearance = has_body or has_face

    if recovery:
        appearance = (0.65 * max(b, 0.) + 0.35 * max(f, 0.)) if has_appearance else 0.
        return 0.60 * appearance + 0.25 * motion + 0.10 * iou + 0.05 * scale_score

    if crossing:
        # During a crossing, appearance must dominate. Motion and IoU are still
        # useful but cannot overpower a reliable appearance match.
        if has_appearance:
            appearance = 0.70 * max(b, 0.) + 0.30 * max(f, 0.) if has_face else max(b, 0.)
            return 0.62 * appearance + 0.20 * motion + 0.10 * iou + 0.08 * scale_score
        return 0.50 * motion + 0.25 * iou + 0.25 * scale_score

    if has_appearance:
        appearance = 0.70 * max(b, 0.) + 0.30 * max(f, 0.) if has_face else max(b, 0.)
        return 0.42 * appearance + 0.30 * motion + 0.20 * iou + 0.08 * scale_score

    # Missing ReID is neutral: do not subtract a score merely because no
    # descriptor was produced on this frame.
    return 0.50 * motion + 0.32 * iou + 0.18 * scale_score


def _associate_detections_v081(self, tracks, boxes, bodies, faces, recovery=False):
    """Hungarian assignment with ambiguity rejection for normal and recovery matches."""
    if not tracks or not boxes:
        return [], list(range(len(tracks))), list(range(len(boxes)))

    cost = np.full((len(tracks), len(boxes)), 1e6, dtype=np.float32)
    scores = {}
    for i, track in enumerate(tracks):
        for j, box in enumerate(boxes):
            score = _compute_association_cost_v081(self, track, box, bodies[j], faces[j], recovery)
            if score is not None:
                cost[i, j] = 1.0 - float(score)
                scores[(i, j)] = float(score)

    rows, cols = linear_sum_assignment(cost)
    minimum = self.recovery_reid_threshold if recovery else 0.28
    matches, unmatched_tracks, unmatched_dets = [], list(range(len(tracks))), list(range(len(boxes)))

    # Configurable ambiguity margin. Recovery keeps its stricter margin; normal
    # crossings use the same safety principle with a slightly smaller margin.
    margin = self.recovery_margin if recovery else max(0.05, self.recovery_margin * 0.75)
    for row, col in zip(rows, cols):
        score = scores.get((row, col))
        if score is None or score < minimum:
            continue
        alternatives = [value for (r, c), value in scores.items()
                        if (r == row and c != col) or (c == col and r != row)]
        second = max(alternatives, default=-1.0)
        if second >= 0.0 and score - second < margin:
            tracks[row].association_reason = 'RECOVERY_AMBIGUOUS' if recovery else 'ASSOCIATION_AMBIGUOUS'
            self._record_metric('ASSOCIATION_REJECTED', tracks[row],
                                rejection_reason=tracks[row].association_reason,
                                best_score=score, second_best_score=second,
                                association_margin=score - second)
            if recovery:
                self._record_metric('RECOVERY_AMBIGUOUS', tracks[row],
                                    best_score=score, second_best_score=second,
                                    recovery_margin=score - second,
                                    recovery_result='RECOVERY_AMBIGUOUS',
                                    rejection_reason='RECOVERY_AMBIGUOUS')
            continue
        matches.append((row, col, score))
        unmatched_tracks.remove(row)
        unmatched_dets.remove(col)

    return matches, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# Metrics version label
# ---------------------------------------------------------------------------
_original_start_session = MetricsCollector.start_session


def _start_session_v081(self, video_name, fps, configuration):
    _original_start_session(self, video_name, fps, configuration)
    self.session['tracker_version'] = 'V08.1'


# ---------------------------------------------------------------------------
# Install patches
# ---------------------------------------------------------------------------
_tracker.Track = SafeTrack
Track = SafeTrack
ReIDTracker = _tracker.ReIDTracker
TrackState = _tracker.TrackState
IdentityState = _tracker.IdentityState
GateResult = _tracker.GateResult
FacePersonMatch = _tracker.FacePersonMatch

ReIDTracker._crossing_pair = _crossing_pair
ReIDTracker._appearance_refresh_required = _appearance_refresh_required_v081
ReIDTracker._extract_embeddings_batch = _selective_extract_embeddings_batch_v081
ReIDTracker._passes_association_gates = _passes_association_gates_v081
ReIDTracker._compute_association_cost = _compute_association_cost_v081
ReIDTracker._associate_detections = _associate_detections_v081
MetricsCollector.start_session = _start_session_v081

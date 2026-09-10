"""Occlusion-aware multi-object tracking and identity assignment."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.core.gallery import Gallery
from app.utils.pipeline_logging import get_pipeline_logger

logger = get_pipeline_logger()


class KalmanFilter:
    """Eight-state constant-velocity filter for ``x, y, area, ratio`` boxes.

    The first four state values describe the measured box and the last four
    values describe their velocity. Process uncertainty is increased during
    uncertain motion so recovery gates can distinguish confident prediction
    from extrapolation that needs appearance evidence.
    """
    def __init__(self):
        self.f = np.eye(8); self.h = np.eye(4, 8)
        for i in range(4): self.f[i, i + 4] = 1.
        self.q, self.r, self.x, self.p = np.eye(8) * .01, np.eye(4) * .1, None, None

    @staticmethod
    def measurement(box):
        x1, y1, x2, y2 = box; w, h = max(x2-x1, 1e-3), max(y2-y1, 1e-3)
        return np.array([(x1+x2)/2, (y1+y2)/2, w*h, w/h], dtype=np.float32)

    def initiate(self, box):
        self.x = np.r_[self.measurement(box), np.zeros(4, dtype=np.float32)]; self.p = np.eye(8) * 10

    def predict(self, uncertainty=1.):
        self.x = self.f @ self.x; self.p = self.f @ self.p @ self.f.T + self.q * max(1., uncertainty)
        return self.get_bbox()

    def update(self, box):
        innovation = self.measurement(box) - self.h @ self.x
        covariance = self.h @ self.p @ self.h.T + self.r
        gain = self.p @ self.h.T @ np.linalg.inv(covariance)
        self.x += gain @ innovation; self.p = (np.eye(8) - gain @ self.h) @ self.p

    def get_bbox(self):
        cx, cy, area, ratio = self.x[:4]; area, ratio = max(area, 1e-6), max(ratio, .1)
        w = np.sqrt(area * ratio); h = area / w
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)


class TrackState:
    # Track lifecycle states describe visibility and temporal continuity.
    # They are intentionally separate from identity evidence states below.
    TENTATIVE, CONFIRMED, OCCLUDED, RECOVERED, LOST, DELETED = range(6)


class IdentityState:
    """Known-identity evidence state, intentionally separate from TrackState."""
    UNKNOWN, CANDIDATE, CONFIRMED, RETAINED = range(4)


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str = ''


@dataclass(frozen=True)
class FacePersonMatch:
    face_index: int
    person_index: int | None
    score: float
    margin: float
    result: str
    reason: str


@dataclass
class Track:
    """Mutable state for one person track and its identity evidence.

    A track owns motion history, cached appearance descriptors, galleries,
    confidence state, and identity ownership metadata. Keeping these together
    lets short occlusions recover the same track without creating a new ID.
    """
    tracker_id: int; bbox: np.ndarray; body_embedding: np.ndarray | None; face_embedding: np.ndarray | None = None
    state: int = TrackState.TENTATIVE; age: int = 0; hits: int = 0; time_since_update: int = 0
    name: str = 'Unknown'; score: float = 0.; max_trajectory_history: int = 30; max_body_gallery: int = 12; max_face_gallery: int = 6; turn_threshold: float = .7

    def __post_init__(self):
        # A missing first descriptor is represented by a zero vector so gallery
        # and similarity code can use one consistent numeric shape.
        self.body_embedding = (np.zeros(256, dtype=np.float32) if self.body_embedding is None
                               else np.asarray(self.body_embedding, dtype=np.float32).reshape(-1))
        self.face_embedding = None if self.face_embedding is None else np.asarray(self.face_embedding, dtype=np.float32).reshape(-1)
        self.kalman = KalmanFilter(); self.kalman.initiate(self.bbox); self.predicted_bbox = self.bbox.copy(); self.last_reliable_bbox = self.bbox.copy()
        self.trajectory, self.velocity_history, self.direction_history = deque([self.centre(self.bbox)], maxlen=self.max_trajectory_history), deque(maxlen=self.max_trajectory_history), deque(maxlen=self.max_trajectory_history)
        self.velocity = self.acceleration = np.zeros(2, dtype=np.float32); self.motion_confidence = 1.
        self.missed_frames = self.occlusion_frames = 0; self.occlusion_start_frame = self.occluded_by = None; self.recovery_score = self.association_confidence = 0.
        self.body_gallery, self.face_gallery = deque(maxlen=self.max_body_gallery), deque(maxlen=self.max_face_gallery)
        self.append_gallery(self.body_gallery, self.body_embedding); self.append_gallery(self.face_gallery, self.face_embedding)
        self.identity_state = IdentityState.UNKNOWN; self.identity_candidate = None
        self.identity_candidate_frames = self.identity_retention_count = self.identity_switch_count = 0
        self.identity_score = self.identity_margin = self.face_score = self.body_score = 0.
        self.identity_source = self.identity_rejection_reason = ''
        # V06 ownership locks survive tracker fragmentation. The lock is kept
        # internal so the established public identity-state vocabulary remains.
        self.identity_locked = False; self.identity_lock_frame = None
        self.last_detection_confidence = 0.; self.association_reason = ''
        self.face_score_history, self.body_score_history = deque(maxlen=8), deque(maxlen=8)
        self.prediction_displacement = self.prediction_total_displacement = 0.; self.prediction_uncertainty = 0.; self.prediction_visible = True
        self.last_body_embedding = self.body_embedding.copy() if self.body_embedding is not None else None
        self.last_reid_frame = 0; self.last_reid_time = 0.; self.reid_age = 0

    @staticmethod
    def centre(box): return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=np.float32)

    @staticmethod
    def append_gallery(gallery, embedding, duplicate=.995):
        """Append only finite, nonzero, sufficiently novel evidence."""
        if embedding is None: return
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not np.isfinite(emb).all() or np.linalg.norm(emb) < 1e-8: return
        emb = emb.astype(np.float32, copy=True); emb /= np.linalg.norm(emb) + 1e-12
        if not gallery or max(float(np.dot(emb, member.reshape(-1))) for member in gallery) < duplicate: gallery.append(emb)

    def predict(self, velocity_decay=.7, max_displacement_per_frame=80., max_total_displacement=240., uncertainty_growth=.25):
        """Advance the track through a missed frame using damped motion."""
        self.time_since_update += 1; self.missed_frames += 1
        self.reid_age += 1
        self.kalman.x[4:6] = self.velocity * (velocity_decay ** self.missed_frames)
        raw_bbox = self.kalman.predict(1 + (1-self.motion_confidence)*4 + self.occlusion_frames*.1)
        previous_center, raw_center = self.centre(self.bbox), self.centre(raw_bbox)
        displacement = raw_center - previous_center
        distance = float(np.linalg.norm(displacement)); clamped = distance > max_displacement_per_frame
        if clamped: displacement *= max_displacement_per_frame / max(distance, 1e-6)
        from_last = previous_center + displacement - self.centre(self.last_reliable_bbox)
        total_distance = float(np.linalg.norm(from_last))
        if total_distance > max_total_displacement:
            displacement = self.centre(self.last_reliable_bbox) - previous_center
            clamped = True
        correction = displacement - (raw_center - previous_center)
        self.predicted_bbox = raw_bbox + np.tile(correction, 2); self.bbox = self.predicted_bbox.copy()
        self.prediction_displacement = float(np.linalg.norm(self.centre(self.predicted_bbox) - previous_center))
        self.prediction_total_displacement = float(np.linalg.norm(self.centre(self.predicted_bbox) - self.centre(self.last_reliable_bbox)))
        self.prediction_uncertainty = min(1., self.prediction_uncertainty + uncertainty_growth * self.missed_frames)
        self.prediction_visible = self.prediction_uncertainty < .75 and self.prediction_total_displacement < max_total_displacement
        return {'damped': self.missed_frames > 1, 'clamped': clamped, 'expired': self.prediction_total_displacement >= max_total_displacement, 'visible': self.prediction_visible}

    def update(self, box, body, face, confidence, update_gallery=False, recovered=False, detection_confidence=1.):
        """Commit an observed detection and optionally accept gallery evidence."""
        previous = self.last_reliable_bbox.copy(); self.kalman.update(box); self.bbox = self.predicted_bbox = box.copy()
        observed = self.centre(box) - self.centre(previous); self.acceleration = observed-self.velocity; self.velocity = observed
        speed = float(np.linalg.norm(observed))
        if speed > 1e-4:
            direction = observed/speed
            if self.direction_history and float(np.dot(direction, self.direction_history[-1])) < self.turn_threshold:
                self.motion_confidence *= .7; logger.debug('Track %s made a sharp direction change.', self.tracker_id)
            self.direction_history.append(direction)
        self.velocity_history.append(observed); self.trajectory.append(self.centre(box)); self.last_reliable_bbox = box.copy()
        if body is not None:
            self.body_embedding = np.asarray(body, dtype=np.float32).reshape(-1)
            if np.isfinite(self.body_embedding).all() and np.linalg.norm(self.body_embedding) > 1e-8:
                self.last_body_embedding = self.body_embedding.copy(); self.last_reid_time = time.time(); self.reid_age = 0
        if face is not None:
            self.face_embedding = np.asarray(face, dtype=np.float32).reshape(-1)
        self.time_since_update = self.missed_frames = self.occlusion_frames = 0; self.occlusion_start_frame = self.occluded_by = None
        self.prediction_displacement = self.prediction_total_displacement = 0.; self.prediction_uncertainty = 0.; self.prediction_visible = True
        self.association_confidence = confidence; self.recovery_score = confidence if recovered else 0.; self.motion_confidence = min(1., self.motion_confidence+.2); self.age += 1; self.hits += 1
        self.last_detection_confidence = detection_confidence; self.association_reason = 'RECOVERY' if recovered else 'ASSOCIATED'
        self.identity_locked = False; self.identity_lock_frame = None
        self.state = TrackState.RECOVERED if recovered else TrackState.CONFIRMED
        if update_gallery: self.append_gallery(self.body_gallery, body); self.append_gallery(self.face_gallery, face)


class ReIDTracker:
    """Hungarian multi-cue tracker with conservative identity recovery.

    The update pipeline is deliberately ordered: filter detections, predict
    tracks, refresh appearance only when needed, gate by motion and geometry,
    associate with Hungarian assignment, recover lost tracks, arbitrate
    duplicates, and finally assign open-set identities.
    """
    def __init__(self, gallery: Gallery, extractor, face_processor=None, threshold=.75, margin=.15, max_lost_frames=30, min_hits_to_confirm=3, iou_threshold=.3, appearance_weight=.5, w_body=.5, w_face=.5, *, max_occlusion_frames=45, max_recovery_frames=180, max_trajectory_history=30, max_body_gallery=12, max_face_gallery=6, occlusion_iou_threshold=.15, recovery_reid_threshold=.78, motion_gate_threshold=4., recovery_motion_gate=8., turn_threshold=.75, motion_confidence_threshold=.45, gallery_update_threshold=.72, normal_reid_gate=.35, strong_reid_gate=.55, normal_iou_gate=.05, strong_reid_iou_bypass=.70, low_motion_reid_gate=.50, weak_motion_distance=1., face_person_min_match_score=.35, face_person_ambiguity_margin=.10, prediction_velocity_decay=.70, max_prediction_displacement_per_frame=80., max_total_prediction_displacement=240., prediction_uncertainty_growth=.25, detection_dedup_iou=.75, min_detection_confidence=.35, min_detection_width=20, min_detection_height=40, min_detection_area=1200, duplicate_track_iou=.70, duplicate_appearance_threshold=.80, identity_face_threshold=.72, identity_body_candidate_threshold=.72, identity_body_confirm_threshold=.84, identity_margin=.08, identity_candidate_min_frames=4, identity_retention_frames=45, identity_lock_timeout=None, recovery_margin=.10, association_min_iou=.01, association_max_scale_change=2.85, gallery_min_detection_confidence=.70, reid_interval=5):
        self.gallery, self.extractor, self.face_processor = gallery, extractor, face_processor; self.threshold, self.margin = threshold, margin; self.max_lost_frames, self.min_hits_to_confirm = max_lost_frames, min_hits_to_confirm
        self.iou_threshold, self.appearance_weight, self.w_body, self.w_face = iou_threshold, appearance_weight, w_body, w_face; self.max_occlusion_frames, self.max_recovery_frames = max_occlusion_frames, max_recovery_frames
        self.max_trajectory_history, self.max_body_gallery, self.max_face_gallery = max_trajectory_history, max_body_gallery, max_face_gallery; self.occlusion_iou_threshold, self.recovery_reid_threshold = occlusion_iou_threshold, recovery_reid_threshold
        self.motion_gate_threshold, self.recovery_motion_gate, self.turn_threshold = motion_gate_threshold, recovery_motion_gate, turn_threshold; self.motion_confidence_threshold, self.gallery_update_threshold, self.normal_reid_gate = motion_confidence_threshold, gallery_update_threshold, normal_reid_gate
        self.strong_reid_gate, self.normal_iou_gate = strong_reid_gate, normal_iou_gate; self.strong_reid_iou_bypass, self.low_motion_reid_gate = strong_reid_iou_bypass, low_motion_reid_gate; self.weak_motion_distance = weak_motion_distance
        self.face_person_min_match_score, self.face_person_ambiguity_margin = face_person_min_match_score, face_person_ambiguity_margin
        self.prediction_velocity_decay, self.max_prediction_displacement_per_frame = prediction_velocity_decay, max_prediction_displacement_per_frame
        self.max_total_prediction_displacement, self.prediction_uncertainty_growth = max_total_prediction_displacement, prediction_uncertainty_growth
        self.detection_dedup_iou, self.min_detection_confidence = detection_dedup_iou, min_detection_confidence
        self.min_detection_width, self.min_detection_height, self.min_detection_area = min_detection_width, min_detection_height, min_detection_area
        self.duplicate_track_iou, self.duplicate_appearance_threshold = duplicate_track_iou, duplicate_appearance_threshold
        self.identity_face_threshold, self.identity_body_candidate_threshold = identity_face_threshold, identity_body_candidate_threshold
        self.identity_body_confirm_threshold, self.identity_margin = identity_body_confirm_threshold, identity_margin
        self.identity_candidate_min_frames, self.identity_retention_frames = identity_candidate_min_frames, identity_retention_frames
        self.identity_lock_timeout = max_recovery_frames if identity_lock_timeout is None else identity_lock_timeout
        self.recovery_margin, self.association_min_iou = recovery_margin, association_min_iou
        self.association_max_scale_change = association_max_scale_change
        self.gallery_min_detection_confidence = gallery_min_detection_confidence
        self.reid_interval = max(1, int(reid_interval))
        self.identity_owners = {}
        self.metrics = None
        self.tracks: List[Track] = []; self.next_id = self.frame_count = 0
        self.last_profile = {}
        self.debug_counters = {key: 0 for key in ('detections_raw', 'detections_after_filter', 'detections_after_dedup', 'tracks_created', 'tracks_confirmed', 'tracks_occluded', 'tracks_recovered', 'tracks_lost', 'tracks_deleted', 'duplicate_tracks_removed', 'recovery_attempts', 'recovery_successes', 'recovery_failures')}

    @staticmethod
    def valid(emb): return emb is not None and np.isfinite(emb).all() and np.linalg.norm(emb) > 1e-8

    def _record_metric(self, event_type, track=None, **values):
        if self.metrics is not None: self.metrics.record(event_type, self.frame_count, track=track, **values)

    def _hybrid_similarity(self, b1, f1, b2, f2):
        value = weight = 0.
        if self.valid(b1) and self.valid(b2): value += self.w_body*float(np.dot(b1,b2)); weight += self.w_body
        if self.valid(f1) and self.valid(f2): value += self.w_face*float(np.dot(f1,f2)); weight += self.w_face
        return value/weight if weight else -1.

    def _gallery_similarity(self, track, body, face):
        b = max((float(np.dot(x, body)) for x in track.body_gallery), default=-1.) if self.valid(body) else -1.; f = max((float(np.dot(x, face)) for x in track.face_gallery), default=-1.) if self.valid(face) else -1.
        hybrid = self._hybrid_similarity(track.body_embedding, track.face_embedding, body, face)
        return max(hybrid, b if f < 0 else (self.w_body*b+self.w_face*f)/(self.w_body+self.w_face)), b, f

    def _prepare_reid_crop(self, image, bbox, mask):
        x1,y1,x2,y2 = bbox.astype(int); x1,y1,x2,y2 = max(x1,0),max(y1,0),min(x2,image.shape[1]),min(y2,image.shape[0])
        if x2 <= x1 or y2 <= y1: return None
        crop=image[y1:y2,x1:x2].copy()
        if mask is not None:
            local_mask = np.asarray(mask)
            if local_mask.shape[:2] != (y2-y1, x2-x1):
                local_mask = cv2.resize(local_mask.astype(np.float32), (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            crop[local_mask <= .5] = 0
        return cv2.resize(crop, (128,256))

    def _extract_embeddings_batch(self, image, boxes, masks):
        """Prepare indexed crops and run exactly one OSNet batch when requested.

        ``crop_indices`` preserves detection order even if an invalid crop is
        skipped. This prevents a failed crop from shifting every later person's
        embedding onto the wrong detection.
        """
        started = time.perf_counter()
        crops = []; crop_indices = []
        for detection_index, (box, mask) in enumerate(zip(boxes, masks)):
            crop = self._prepare_reid_crop(image, box, mask)
            if crop is not None and crop.size:
                crop_indices.append(detection_index); crops.append(crop)
        preparation_ms = (time.perf_counter() - started) * 1000.
        bodies = [np.zeros(256, dtype=np.float32) for _ in boxes]
        inference_started = time.perf_counter()
        if crops:
            try:
                if hasattr(self.extractor, 'batch_extract'):
                    extracted = self.extractor.batch_extract(crops)
                else:
                    extracted = [self.extractor.extract(crop) for crop in crops]
                for detection_index, body in zip(crop_indices, extracted):
                    bodies[detection_index] = body
            except Exception:
                logger.exception('Batch Re-ID extraction failed; preserving empty descriptors.')
        self.last_profile['reid_preprocessing_ms'] = preparation_ms
        self.last_profile['reid_inference_ms'] = (time.perf_counter() - inference_started) * 1000.
        self.last_profile['reid_calls'] = 1 if crops else 0
        self.last_profile['reid_crops'] = len(crops)
        return bodies

    def _appearance_refresh_required(self, boxes):
        """Decide whether this frame needs fresh body and face evidence.

        Stable tracks reuse cached descriptors. Refreshes are forced for new
        scenes, periodic cadence, uncertainty, occlusion/recovery, and
        overlapping detections that may represent a crossing.
        """
        active = [track for track in self.tracks if track.state != TrackState.DELETED]
        if not active or self.frame_count % self.reid_interval == 0:
            return True
        if any(track.state in (TrackState.OCCLUDED, TrackState.LOST) or track.prediction_uncertainty >= .25
               for track in active):
            return True
        return any(self._compute_iou(first, second) >= .10
                   for index, first in enumerate(boxes) for second in boxes[index + 1:])

    def _extract_embeddings(self, image, bbox, mask):
        crop = self._prepare_reid_crop(image, bbox, mask)
        try: body=self.extractor.extract(crop) if crop is not None and crop.size else np.zeros(256,dtype=np.float32)
        except Exception: body=np.zeros(256,dtype=np.float32)
        if crop is None: return body,None
        face=None
        if self.face_processor is not None and crop.size and crop.shape[0]>=30 and crop.shape[1]>=30:
            cached_faces = getattr(self, '_frame_face_results', None)
            if cached_faces is not None:
                index = getattr(self, '_frame_detection_index', 0)
                assignments = getattr(self, '_frame_face_assignments', {})
                face = assignments.get(index)
                self._frame_detection_index = index + 1
            else:
                started = time.perf_counter()
                try: _,face=self.face_processor.extract(crop)
                except Exception: pass
                self._record_metric('FACE_INFERENCE', count=1, inference_time_ms=(time.perf_counter() - started) * 1000., face_count=1 if face is not None else 0)
        return body,face

    @staticmethod
    def _compute_iou(a,b):
        x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0.,x2-x1)*max(0.,y2-y1); union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
        return inter/union if union>0 else 0.

    @classmethod
    def _face_bbox_relationship(cls, face_box, person_box):
        """Measure how one detected face relates to one person bounding box."""
        face = np.asarray(face_box, dtype=np.float32)
        person = np.asarray(person_box, dtype=np.float32)
        x1, y1 = max(face[0], person[0]), max(face[1], person[1])
        x2, y2 = min(face[2], person[2]), min(face[3], person[3])
        intersection = max(0., x2 - x1) * max(0., y2 - y1)
        face_area = max(0., face[2] - face[0]) * max(0., face[3] - face[1])
        center_x, center_y = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
        face_center_y_ratio = (center_y - person[1]) / max(person[3] - person[1], 1e-6)
        return {
            'iou': cls._compute_iou(face, person),
            'intersection_over_face': intersection / face_area if face_area else 0.,
            'face_center_inside': bool(person[0] <= center_x <= person[2] and person[1] <= center_y <= person[3]),
            'face_center_y_ratio': float(face_center_y_ratio),
        }

    @classmethod
    def _calculate_face_bbox_relationships(cls, face_results, person_boxes):
        """Calculate relationships for every detected face and person box pair."""
        return [[cls._face_bbox_relationship(cls._face_result_box(face), person_box) for person_box in person_boxes]
                for face in (face_results or [])]

    @staticmethod
    def _face_result_box(face_result):
        if hasattr(face_result, 'bbox'):
            return face_result.bbox
        return face_result[0] if isinstance(face_result, (tuple, list)) else face_result

    @staticmethod
    def _face_result_embedding(face_result):
        if hasattr(face_result, 'embedding'):
            return face_result.embedding
        return face_result[1] if isinstance(face_result, (tuple, list)) else None

    @staticmethod
    def _face_result_confidence(face_result):
        return float(getattr(face_result, 'confidence', 1.0))

    @classmethod
    def _face_person_score(cls, relationship):
        """Score valid face/person geometry, prioritizing containment and upper position."""
        center = 1.0 if relationship['face_center_inside'] else 0.0
        upper = relationship['face_center_y_ratio']
        return .55 * center + .30 * relationship['intersection_over_face'] + .15 * max(0., 1. - upper / .75)

    def _associate_faces_to_persons(self, face_results, person_boxes, min_confidence=.0, min_face_size=1., min_match_score=None, ambiguity_margin=None):
        """Assign faces to at most one person using geometry and ambiguity gates.

        A face is rejected when it is weak, outside a person box, too small, or
        too close to another person's geometry. Refusing uncertain assignments
        protects face-confirmed identities from cross-person contamination.
        """
        face_results = face_results or []
        min_match_score = self.face_person_min_match_score if min_match_score is None else min_match_score
        ambiguity_margin = self.face_person_ambiguity_margin if ambiguity_margin is None else ambiguity_margin
        relationships = self._calculate_face_bbox_relationships(face_results, person_boxes)
        candidates = []
        match_results = []
        for face_index, row in enumerate(relationships):
            valid = []
            face_box = self._face_result_box(face_results[face_index])
            face_size = min(face_box[2] - face_box[0], face_box[3] - face_box[1])
            face_valid = (np.isfinite(face_box).all() and face_box[2] > face_box[0] and face_box[3] > face_box[1]
                          and self._face_result_embedding(face_results[face_index]) is not None
                          and self.valid(self._face_result_embedding(face_results[face_index]))
                          and self._face_result_confidence(face_results[face_index]) >= min_confidence
                          and face_size >= min_face_size)
            if not face_valid:
                match_results.append(FacePersonMatch(face_index, None, 0., 0., 'UNASSIGNED', 'FACE_PERSON_INVALID_FACE'))
                continue
            for person_index, relationship in enumerate(row):
                score = self._face_person_score(relationship)
                if relationship['iou'] > 0. and relationship['face_center_y_ratio'] <= .75:
                    valid.append((score, person_index))
            valid.sort(reverse=True)
            if not valid:
                match_results.append(FacePersonMatch(face_index, None, 0., 0., 'UNASSIGNED', 'FACE_PERSON_NO_CANDIDATE'))
                continue
            best_score, best_person = valid[0]
            second_score = valid[1][0] if len(valid) > 1 else 0.
            margin = best_score - second_score
            if best_score < min_match_score:
                match_results.append(FacePersonMatch(face_index, None, best_score, margin, 'UNASSIGNED', 'FACE_PERSON_LOW_SCORE'))
            elif len(valid) > 1 and margin < ambiguity_margin:
                match_results.append(FacePersonMatch(face_index, None, best_score, margin, 'UNASSIGNED', 'FACE_PERSON_AMBIGUOUS'))
            else:
                candidates.append((best_score, face_index, best_person, margin))
                match_results.append(FacePersonMatch(face_index, best_person, best_score, margin, 'ACCEPTED', 'FACE_PERSON_ACCEPTED'))
        assignments = {}
        used_persons = set()
        accepted_faces = set()
        for _, face_index, person_index, _ in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if person_index not in used_persons:
                assignments[person_index] = self._face_result_embedding(face_results[face_index])
                used_persons.add(person_index)
                accepted_faces.add(face_index)
        for match in match_results:
            if match.result == 'ACCEPTED' and match.face_index not in accepted_faces:
                result_index = next(index for index, item in enumerate(match_results) if item is match)
                match_results[result_index] = FacePersonMatch(match.face_index, None, match.score, match.margin, 'UNASSIGNED', 'FACE_PERSON_PERSON_CONFLICT')
        ambiguous = sum(match.reason == 'FACE_PERSON_AMBIGUOUS' for match in match_results)
        return assignments, relationships, match_results, ambiguous, len(face_results) - len(assignments)

    @staticmethod
    def _distance(a,b): return float(np.linalg.norm(Track.centre(a)-Track.centre(b)))
    @staticmethod
    def _scale(box): return max(10.,float(np.hypot(box[2]-box[0],box[3]-box[1])))

    def _valid_detection(self, det, shape):
        """Reject weak or implausible boxes before neural feature extraction."""
        confidence = float(getattr(det, 'confidence', 1.0))
        x1, y1, x2, y2 = map(float, det.box); height, width = shape[:2]
        box_width, box_height = x2-x1, y2-y1
        return (confidence >= self.min_detection_confidence and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                and box_width >= self.min_detection_width and box_height >= self.min_detection_height
                and box_width * box_height >= self.min_detection_area and .15 <= box_width / box_height <= 2.5)

    def _prepare_detections(self, detections, shape):
        """Confidence-aware NMS for tracking input; YOLO NMS alone is not enough here."""
        self.debug_counters['detections_raw'] += len(detections)
        self._record_metric('DETECTION_RAW', count=len(detections))
        valid = [det for det in detections if self._valid_detection(det, shape)]
        self._record_metric('DETECTIONS_AFTER_CONFIDENCE_FILTER', count=sum(float(getattr(det, 'confidence', 1.)) >= self.min_detection_confidence for det in detections))
        self._record_metric('DETECTIONS_AFTER_BBOX_FILTER', count=len(valid))
        self.debug_counters['detections_after_filter'] += len(valid)
        kept = []
        for det in sorted(valid, key=lambda item: float(getattr(item, 'confidence', 1.0)), reverse=True):
            box = np.asarray(det.box, dtype=np.float32)
            if all(self._compute_iou(box, np.asarray(other.box, dtype=np.float32)) < self.detection_dedup_iou for other in kept): kept.append(det)
        self.debug_counters['detections_after_dedup'] += len(kept)
        self._record_metric('DETECTIONS_AFTER_DEDUPLICATION', count=len(kept))
        return kept

    def _new_track_conflicts(self, box, body, face):
        """Avoid creating a competing ID beside an existing canonical track."""
        for track in self.tracks:
            if track.state == TrackState.DELETED: continue
            overlap = self._compute_iou(track.bbox, box)
            app, _, _ = self._gallery_similarity(track, body, face)
            if overlap >= self.duplicate_track_iou and (app >= self.duplicate_appearance_threshold or track.state != TrackState.TENTATIVE): return True
        return False

    def _arbitrate_duplicates(self):
        """Retire only highly-overlapping, appearance-consistent competing tracks."""
        active = [track for track in self.tracks if track.state != TrackState.DELETED]
        for index, first in enumerate(active):
            for second in active[index+1:]:
                if first.state == TrackState.DELETED or second.state == TrackState.DELETED: continue
                overlap = self._compute_iou(first.bbox, second.bbox)
                app, _, _ = self._gallery_similarity(first, second.body_embedding, second.face_embedding)
                if overlap < self.duplicate_track_iou or app < self.duplicate_appearance_threshold: continue
                def rank(track):
                    confirmed = track.state in (TrackState.CONFIRMED, TrackState.RECOVERED)
                    return (confirmed, track.hits, track.age, track.association_confidence)
                winner, loser = (first, second) if rank(first) >= rank(second) else (second, first)
                # Two mature tracks may be legitimate nearby people with similar clothes.
                if loser.state != TrackState.TENTATIVE and winner.state != TrackState.TENTATIVE: continue
                loser.state = TrackState.DELETED; self.debug_counters['duplicate_tracks_removed'] += 1
                logger.debug('Duplicate track removed: canonical=%s duplicate=%s.', winner.tracker_id, loser.tracker_id)

    def _passes_association_gates(self, track, box, body, face, recovery=False):
        """Apply hard scale, motion, overlap, and appearance safety gates.

        These gates run before Hungarian scoring so impossible candidates do not
        consume association work or force an identity swap during a crossing.
        Recovery uses stricter appearance requirements than normal tracking.
        """
        iou=self._compute_iou(track.predicted_bbox,box); distance=self._distance(track.predicted_bbox,box)/self._scale(track.last_reliable_bbox); w=(box[2]-box[0])/max(track.last_reliable_bbox[2]-track.last_reliable_bbox[0],1.); h=(box[3]-box[1])/max(track.last_reliable_bbox[3]-track.last_reliable_bbox[1],1.); app,b,f=self._gallery_similarity(track,body,face)
        gate=self.recovery_motion_gate if recovery else self.motion_gate_threshold*(1+(1-track.motion_confidence)*.75)
        max_scale = self.association_max_scale_change
        if not 1/max_scale<=w<=max_scale or not 1/max_scale<=h<=max_scale:
            return GateResult(False, 'SCALE_GATE_FAIL')
        if distance > gate:
            return GateResult(False, 'MOTION_GATE_FAIL')
        # A completely disjoint box can still be recovered only with strong Re-ID;
        # ordinary association must not bridge arbitrary nearby people.
        if not recovery and iou < self.association_min_iou and app < self.strong_reid_gate:
            return GateResult(False, 'IOU_GATE_FAIL')
        if recovery:
            if b < self.recovery_reid_threshold and f < max(.9,self.recovery_reid_threshold):
                return GateResult(False, 'RECOVERY_REID_GATE_FAIL')
            return GateResult(True)
        # When a usable body descriptor contradicts the track gallery, overlap alone
        # is not enough: that is the common person-to-person occlusion ID-swap case.
        body_gate = self.normal_reid_gate + (1 - track.motion_confidence) * (self.low_motion_reid_gate - self.normal_reid_gate)
        if self.valid(body) and b < body_gate: return GateResult(False, 'REID_GATE_FAIL')
        if app < body_gate and iou < self.normal_iou_gate and distance > self.weak_motion_distance:
            return GateResult(False, 'COMBINED_GATE_FAIL')
        if app < body_gate and iou < self.normal_iou_gate: return GateResult(False, 'NO_SAFE_MATCH')
        return GateResult(True)

    def _compute_association_cost(self, track, box, body, face, recovery=False):
        """Compute availability-aware multi-cue association score after hard gates."""
        self._record_metric('ASSOCIATION_ATTEMPT', track, detection_id='', track_state=track.state, identity_state=track.identity_state)
        gate_result = self._passes_association_gates(track, box, body, face, recovery)
        iou=self._compute_iou(track.predicted_bbox,box); distance=self._distance(track.predicted_bbox,box)/self._scale(track.last_reliable_bbox)
        app,b,f=self._gallery_similarity(track,body,face)
        if not gate_result.accepted:
            track.association_reason = gate_result.reason
            self._record_metric('ASSOCIATION_REJECTED', track, iou_score=iou, body_score=b, face_score=f, motion_score=max(0., 1-distance / max(self.motion_gate_threshold, 1e-6)), rejection_reason=gate_result.reason)
            return None
        gate=self.recovery_motion_gate if recovery else self.motion_gate_threshold*(1+(1-track.motion_confidence)*.75)
        if recovery:
            # Face is strongest when available; body remains a supporting recovery cue.
            fused = .72*max(b,0)+.18*max(f,0)+.10*max(0,1-distance/gate)
            return max(fused, f if f >= max(.9, self.recovery_reid_threshold) else -1.)
        motion=max(0,1-distance/gate); mw=(.10 if track.motion_confidence < self.motion_confidence_threshold else .35)*track.motion_confidence
        return mw*motion+.25*iou+(1-mw-.25)*max(app,0)

    def _associate_detections(self, tracks, boxes, bodies, faces, recovery=False):
        """Assign surviving track/detection candidates with Hungarian matching."""
        if not tracks or not boxes: return [],list(range(len(tracks))),list(range(len(boxes)))
        cost=np.full((len(tracks),len(boxes)),1e6,dtype=np.float32); scores={}
        for i,t in enumerate(tracks):
            for j,box in enumerate(boxes):
                score=self._compute_association_cost(t,box,bodies[j],faces[j],recovery)
                if score is not None: cost[i,j]=1-score; scores[i,j]=score
        rows,cols=linear_sum_assignment(cost); matches=[]; ut=list(range(len(tracks))); ud=list(range(len(boxes))); minimum=self.recovery_reid_threshold if recovery else .30
        for r,c in zip(rows,cols):
            if cost[r,c]<1e6 and scores[r,c]>=minimum: matches.append((r,c,scores[r,c]));ut.remove(r);ud.remove(c)
        # Recovery needs separation from alternatives, not merely the highest
        # score; an ambiguous recovery is safer than an ID switch.
        if recovery:
            safe=[]
            for r,c,score in matches:
                alternatives=sorted((value for (row,col),value in scores.items()
                                     if (row == r and col != c) or (col == c and row != r)), reverse=True)
                if alternatives and score-alternatives[0] < self.recovery_margin:
                    tracks[r].association_reason = 'RECOVERY_AMBIGUOUS'; self._record_metric('ASSOCIATION_REJECTED', tracks[r], rejection_reason='RECOVERY_AMBIGUOUS'); self._record_metric('RECOVERY_AMBIGUOUS', tracks[r], best_score=score, second_best_score=alternatives[0], recovery_margin=score-alternatives[0], recovery_result='RECOVERY_AMBIGUOUS', rejection_reason='RECOVERY_AMBIGUOUS'); self._record_metric('RECOVERY_REJECTED', tracks[r], rejection_reason='RECOVERY_AMBIGUOUS'); ut.append(r); ud.append(c)
                else: safe.append((r,c,score))
            matches=safe
        return matches,ut,ud

    def cleanup(self, final=False):
        """Remove deleted tracks and, at end of input, never-confirmed fragments."""
        removed = []
        for track in self.tracks:
            if track.state == TrackState.DELETED or (final and track.state == TrackState.TENTATIVE and track.hits < self.min_hits_to_confirm):
                track.state = TrackState.DELETED
                self._release_identity_owner(track, force=True)
                removed.append(track)
                self.debug_counters['tracks_deleted'] += 1
                self._record_metric('TRACK_DELETED', track, rejection_reason='TENTATIVE_NOT_CONFIRMED' if final else '')
        self.tracks = [track for track in self.tracks if track.state != TrackState.DELETED]
        return removed

    def _find_occluder(self, track):
        options=[(self._compute_iou(track.predicted_bbox,o.bbox),o.tracker_id) for o in self.tracks if o is not track and o.state in (TrackState.CONFIRMED,TrackState.RECOVERED) and o.time_since_update==0]
        best=max(options,default=(0.,None)); return best[1] if best[0]>=self.occlusion_iou_threshold else None

    @staticmethod
    def _best_identity(scores):
        if not len(scores): return None, -1., 0.
        ranked = np.sort(scores)[::-1]; index = int(np.argmax(scores))
        return index, float(scores[index]), float(scores[index] - ranked[1]) if len(ranked) > 1 else 1.

    def _identity_scores(self, track):
        body = np.array([float(np.dot(track.body_embedding, ref)) if self.valid(track.body_embedding) else -1.
                         for ref in self.gallery.body_means])
        face = np.array([float(np.dot(track.face_embedding, ref)) if self.valid(track.face_embedding) and self.valid(ref) else -1.
                         for ref in self.gallery.face_means])
        return body, face

    def _lock_identity(self, track):
        """Reserve a confirmed identity while its track is occluded or lost."""
        if track.name == 'Unknown' or track.identity_state not in (IdentityState.CONFIRMED, IdentityState.RETAINED):
            return
        if not track.identity_locked:
            track.identity_locked = True; track.identity_lock_frame = self.frame_count
        track.identity_state = IdentityState.RETAINED; track.identity_source = 'IDENTITY_LOCKED'
        self.identity_owners[track.name] = track.tracker_id
        self._record_metric('IDENTITY_LOCKED', track)

    def _release_identity_owner(self, track, force=False):
        """Release only after the configured lock lifetime (or explicit expiry)."""
        if not track.identity_locked and self.identity_owners.get(track.name) != track.tracker_id:
            return False
        if track.identity_locked and not force and track.identity_lock_frame is not None:
            if self.frame_count - track.identity_lock_frame < self.identity_lock_timeout:
                return False
        if track.name != 'Unknown' and self.identity_owners.get(track.name) == track.tracker_id:
            self.identity_owners.pop(track.name, None)
        track.identity_locked = False
        self._record_metric('IDENTITY_RELEASED', track)
        return True

    def _frame_level_identity_assignment(self):
        """Assign known identities using face evidence and body history.

        Face evidence may confirm immediately when it clears the strong gate.
        Body-only evidence must satisfy threshold, margin, and repeated-frame
        requirements. Existing owners always win over competing claims.
        """
        for track in self.tracks:
            if track.state == TrackState.DELETED: self._release_identity_owner(track)
        active = [track for track in self.tracks if track.state in (TrackState.CONFIRMED, TrackState.RECOVERED)]
        if not active or not self.gallery.has_body(): return
        proposals = []
        for track in active:
            body, face = self._identity_scores(track)
            body_index, body_score, body_margin = self._best_identity(body)
            face_index, face_score, face_margin = self._best_identity(face)
            track.body_score, track.face_score = body_score, face_score
            track.body_score_history.append(body_score); track.face_score_history.append(face_score)
            # A decisive face is the only one-frame path to a confirmed identity.
            if face_index is not None and face_score >= self.identity_face_threshold and face_margin >= self.identity_margin:
                track.identity_retention_count = 0
                proposals.append((track, face_index, face_score, face_margin, 'FACE'))
                continue
            candidate_index = body_index if body_index is not None and body_score >= self.identity_body_candidate_threshold and body_margin >= self.identity_margin else None
            candidate = self.gallery.names[candidate_index] if candidate_index is not None else None
            if candidate == track.identity_candidate:
                track.identity_candidate_frames += 1
            else:
                track.identity_candidate, track.identity_candidate_frames = candidate, 1 if candidate else 0
            track.identity_margin = body_margin if candidate is not None else max(face_margin, body_margin)
            track.identity_source = 'BODY' if candidate is not None else ''
            track.identity_rejection_reason = '' if candidate is not None else 'AMBIGUOUS_OR_BELOW_THRESHOLD'
            # Retain a face-confirmed identity through temporary face loss, but never
            # promote a new body-only candidate from a single frame.
            if track.identity_state in (IdentityState.CONFIRMED, IdentityState.RETAINED):
                track.identity_retention_count += 1
                if track.identity_retention_count <= self.identity_retention_frames:
                    track.identity_state = IdentityState.RETAINED
                    track.identity_source = 'HISTORY'
                    self._record_metric('IDENTITY_RETAINED', track)
                    continue
                self._release_identity_owner(track)
                track.name, track.score, track.identity_state = 'Unknown', 0., IdentityState.UNKNOWN
                track.identity_rejection_reason = 'RETENTION_EXPIRED'
                continue
            if candidate is not None and body_score >= self.identity_body_confirm_threshold and track.identity_candidate_frames >= self.identity_candidate_min_frames:
                proposals.append((track, candidate_index, body_score, body_margin, 'BODY_HISTORY'))
            else:
                track.name = 'Unknown'; track.score = 0.
                track.identity_state = IdentityState.CANDIDATE if candidate is not None else IdentityState.UNKNOWN
                if candidate is not None: self._record_metric('IDENTITY_CANDIDATE', track, identity=candidate)
        # Process face evidence first, then conservative body-history proposals.
        for track, index, score, margin, source in sorted(proposals, key=lambda item: (item[4] == 'FACE', item[2]), reverse=True):
            name = self.gallery.names[index]; owner_id = self.identity_owners.get(name)
            owner = next((item for item in self.tracks if item.tracker_id == owner_id and item.state != TrackState.DELETED), None)
            if owner is not None and owner is not track:
                # V06: never transfer an identity between live tracks.  A locked
                # owner is deliberately just as authoritative as a visible owner.
                track.name = 'Unknown'; track.score = 0.; track.identity_state = IdentityState.CANDIDATE
                # Keep the established diagnostic spelling for compatibility; this
                # is the V06 identity-owner conflict gate.
                track.identity_rejection_reason = 'CANONICAL_OWNER_EXISTS'; self._record_metric('IDENTITY_OWNER_CONFLICT', track, identity=name, rejection_reason='IDENTITY_OWNER_CONFLICT'); continue
            previous = track.name
            track.name, track.score = name, score; track.identity_score, track.identity_margin = score, margin
            track.identity_state, track.identity_source, track.identity_rejection_reason = IdentityState.CONFIRMED, source, ''
            track.identity_candidate, track.identity_candidate_frames = name, max(track.identity_candidate_frames, self.identity_candidate_min_frames)
            if previous not in ('Unknown', name): track.identity_switch_count += 1
            if previous not in ('Unknown', name): self._record_metric('IDENTITY_SWITCH', track, previous_identity=previous, current_identity=name)
            self._record_metric('IDENTITY_CONFIRMED', track, previous_identity=previous, current_identity=name)
            self.identity_owners[name] = track.tracker_id

    def update(self,image,detections):
        """Process one frame while preserving track and identity continuity.

        Expensive appearance work is conditional; geometry-only frames reuse
        cached evidence while Kalman prediction and hard gates continue to run.
        The method also records stage timings for benchmark comparison.
        """
        self.frame_count+=1
        self.last_profile = {'reid_preprocessing_ms': 0., 'reid_inference_ms': 0., 'reid_calls': 0,
                             'reid_crops': 0, 'face_detection_ms': 0., 'association_ms': 0.,
                             'identity_ms': 0., 'gallery_ms': 0., 'face_calls': 0, 'face_detections': 0}
        self._frame_face_results = None
        for t in self.tracks:
            if t.state!=TrackState.DELETED:
                prediction = t.predict(self.prediction_velocity_decay, self.max_prediction_displacement_per_frame, self.max_total_prediction_displacement, self.prediction_uncertainty_growth)
                self._record_metric('PREDICTION_USED', t)
                if prediction['damped']: self._record_metric('PREDICTION_DAMPED', t)
                if prediction['clamped']: self._record_metric('PREDICTION_CLAMPED', t)
                if prediction['expired']: self._record_metric('PREDICTION_EXPIRED', t)
                if not prediction['visible']: self._record_metric('PREDICTION_HIDDEN', t)
                if t.identity_locked and t.identity_lock_frame is not None and self.frame_count - t.identity_lock_frame >= self.identity_lock_timeout:
                    self._release_identity_owner(t)
        detections = self._prepare_detections(detections, image.shape)
        boxes=[np.asarray(det.box,dtype=np.float32) for det in detections]
        refresh_appearance = self._appearance_refresh_required(boxes)
        face_started = time.perf_counter()
        face_inference_calls = 0
        if refresh_appearance and boxes and self.face_processor is not None and hasattr(self.face_processor, 'extract_faces'):
            try:
                face_inference_calls = 1
                self._frame_face_results = self.face_processor.extract_faces(image)
            except Exception:
                logger.debug('Frame-wide face extraction failed; continuing without face evidence.', exc_info=True)
        self._record_metric('FACE_INFERENCE', count=face_inference_calls,
                            inference_time_ms=(time.perf_counter() - face_started) * 1000.,
                            face_count=len(self._frame_face_results or []))
        self.last_profile['face_detection_ms'] = (time.perf_counter() - face_started) * 1000.
        self.last_profile['face_calls'] = face_inference_calls
        self.last_profile['face_detections'] = len(self._frame_face_results or [])
        self._frame_face_assignments, self._frame_face_relationships, self._frame_face_person_matches, _, face_unassigned = self._associate_faces_to_persons(
            self._frame_face_results, boxes, min_confidence=.5, min_face_size=8.,
            min_match_score=self.face_person_min_match_score, ambiguity_margin=self.face_person_ambiguity_margin)
        self._record_metric('FACE_PERSON_MATCH_ATTEMPT', count=len(self._frame_face_results or []) * len(boxes))
        self._record_metric('FACE_PERSON_MATCH', count=len(self._frame_face_assignments))
        for item in self._frame_face_person_matches:
            self._record_metric('FACE_PERSON_RESULT', face_index=item.face_index, person_index=item.person_index,
                                best_face_person_score=item.score, face_person_margin=item.margin,
                                face_person_result=item.result, rejection_reason=item.reason)
            if item.reason != 'FACE_PERSON_ACCEPTED':
                self._record_metric(item.reason, count=1, face_index=item.face_index, person_index=item.person_index,
                                    best_face_person_score=item.score, face_person_margin=item.margin)
        self._record_metric('FACE_PERSON_UNASSIGNED', count=face_unassigned)
        self._frame_detection_index = 0
        masks = [det.mask for det in detections]
        if getattr(self._extract_embeddings, '__func__', None) is not ReIDTracker._extract_embeddings:
            # Preserve the narrow injection point used by custom integrations and tests.
            bodies=[];faces=[]
            for det, box in zip(detections, boxes):
                body,face=self._extract_embeddings(image,box,det.mask);bodies.append(body);faces.append(face)
        elif refresh_appearance:
            bodies = self._extract_embeddings_batch(image, boxes, masks)
            faces = [self._frame_face_assignments.get(index) for index in range(len(boxes))]
        else:
            bodies = [None for _ in boxes]
            faces = [None for _ in boxes]
        detection_confidences=[float(getattr(det, 'confidence', 1.)) for det in detections]
        self._frame_face_bbox_relationships = self._calculate_face_bbox_relationships(self._frame_face_results, boxes)
        association_started = time.perf_counter()
        normal=[t for t in self.tracks if t.state in (TrackState.TENTATIVE,TrackState.CONFIRMED,TrackState.RECOVERED)]; matches,unmatched,unused=self._associate_detections(normal,boxes,bodies,faces)
        for ti,di,score in matches:
            t=normal[ti]; quality = (score >= self.gallery_update_threshold and detection_confidences[di] >= self.gallery_min_detection_confidence and t.occlusion_frames == 0)
            self._record_metric('TRACK_MATCHED', t, association_score=score, detection_confidence=detection_confidences[di]); self._record_metric('GALLERY_UPDATE_ATTEMPT', t)
            gallery_reason = '' if quality else ('GALLERY_REJECT_LOW_CONFIDENCE' if detection_confidences[di] < self.gallery_min_detection_confidence else 'GALLERY_REJECT_LOW_ASSOCIATION')
            self._record_metric('GALLERY_UPDATE_ACCEPTED' if quality else 'GALLERY_UPDATE_REJECTED', t, rejection_reason=gallery_reason)
            t.update(boxes[di],bodies[di],faces[di],score,quality,detection_confidence=detection_confidences[di])
            if bodies[di] is not None and self.valid(bodies[di]): t.last_reid_frame = self.frame_count
            if t.hits>=self.min_hits_to_confirm:
                if t.state != TrackState.CONFIRMED: self.debug_counters['tracks_confirmed'] += 1
                t.state=TrackState.CONFIRMED
        for ti in unmatched:
            t=normal[ti]
            if t.state==TrackState.TENTATIVE and t.missed_frames > 1:
                t.state=TrackState.DELETED; self.debug_counters['tracks_deleted'] += 1
            elif t.state!=TrackState.TENTATIVE:
                owner=self._find_occluder(t)
                if owner is not None and t.occlusion_frames<self.max_occlusion_frames:
                    if t.state!=TrackState.OCCLUDED:logger.debug('Track %s entered OCCLUDED by %s.',t.tracker_id,owner)
                    if t.state != TrackState.OCCLUDED: self.debug_counters['tracks_occluded'] += 1
                    t.state,t.occluded_by=TrackState.OCCLUDED,owner;t.occlusion_frames+=1;t.occlusion_start_frame=t.occlusion_start_frame or self.frame_count
                    self._record_metric('TRACK_OCCLUDED', t)
                    self._lock_identity(t)
                elif t.missed_frames>self.max_lost_frames:
                    if t.state!=TrackState.LOST:logger.debug('Track %s became LOST.',t.tracker_id)
                    if t.state != TrackState.LOST: self.debug_counters['tracks_lost'] += 1
                    t.state=TrackState.LOST; self._lock_identity(t)
                    self._record_metric('TRACK_LOST', t)
        recoverable=[t for t in self.tracks if t.state in (TrackState.OCCLUDED,TrackState.LOST) and t.missed_frames<=self.max_recovery_frames]
        recovery_possible = any(
            self._passes_association_gates(track, boxes[di], bodies[di], faces[di], recovery=True).accepted
            for track in recoverable for di in unused
        ) if unused and recoverable else False
        if unused and recoverable and recovery_possible:
            self.debug_counters['recovery_attempts'] += 1
            for track in recoverable: self._record_metric('RECOVERY_ATTEMPT', track)
            logger.debug('Re-ID recovery attempted: tracks=%s detections=%s.',len(recoverable),len(unused))
        rmatches,_,unused=self._associate_detections(recoverable,boxes,bodies,faces,True) if recovery_possible else ([],recoverable,unused)
        if unused and recoverable and not rmatches:
            self.debug_counters['recovery_failures'] += 1
            for track in recoverable:
                if track.association_reason != 'REID_MARGIN_FAIL': self._record_metric('RECOVERY_FAILED', track)
            logger.debug('Re-ID recovery rejected all eligible candidates.')
        for ti,di,score in rmatches:
            t=recoverable[ti];self.debug_counters['recovery_successes'] += 1; self.debug_counters['tracks_recovered'] += 1; logger.debug('Re-ID recovery accepted: track=%s confidence=%.3f.',t.tracker_id,score)
            self._record_metric('RECOVERY_SUCCESS', t, recovery_result='SUCCESS', association_score=score); self._record_metric('TRACK_RECOVERED', t)
            quality = score >= self.gallery_update_threshold and detection_confidences[di] >= self.gallery_min_detection_confidence
            t.update(boxes[di],bodies[di],faces[di],score,quality,True,detection_confidences[di])
            if bodies[di] is not None and self.valid(bodies[di]): t.last_reid_frame = self.frame_count
        for di in unused:
            if self._new_track_conflicts(boxes[di], bodies[di], faces[di]):
                logger.debug('New-track gate rejected spatial duplicate detection.'); continue
            created=Track(self.next_id,boxes[di],bodies[di],faces[di],TrackState.CONFIRMED if self.min_hits_to_confirm<=1 else TrackState.TENTATIVE,hits=1,max_trajectory_history=self.max_trajectory_history,max_body_gallery=self.max_body_gallery,max_face_gallery=self.max_face_gallery,turn_threshold=self.turn_threshold)
            created.last_detection_confidence=detection_confidences[di]; self.tracks.append(created);self.next_id+=1;self.debug_counters['tracks_created'] += 1
            self._record_metric('TRACK_CREATED', created, detection_confidence=detection_confidences[di])
        self._arbitrate_duplicates()
        for t in self.tracks:
            if t.state in (TrackState.OCCLUDED,TrackState.LOST) and t.missed_frames>self.max_recovery_frames:
                t.state=TrackState.DELETED;self.debug_counters['tracks_deleted'] += 1;logger.debug('Track %s deleted after recovery/lock lifetime.',t.tracker_id)
                self._record_metric('TRACK_DELETED', t)
            if t.state == TrackState.DELETED: self._release_identity_owner(t)
        self.last_profile['association_ms'] = (time.perf_counter() - association_started) * 1000.
        self.cleanup()
        identity_started = time.perf_counter()
        self._frame_level_identity_assignment()
        self.last_profile['identity_ms'] = (time.perf_counter() - identity_started) * 1000.
        self.last_profile['gallery_ms'] = self.last_profile['identity_ms']
        return self.tracks

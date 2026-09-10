import unittest
from types import SimpleNamespace

import numpy as np

from app.core.tracker import IdentityState, ReIDTracker, Track, TrackState


def emb(index):
    value = np.zeros(4, dtype=np.float32); value[index] = 1.
    return value


class EmptyGallery:
    def has_body(self): return False


class IdentityGallery:
    names = ['alice', 'bob']
    body_means = np.stack([emb(0), emb(1)])
    face_means = np.stack([emb(0), emb(1)])
    def has_body(self): return True


class DummyExtractor:
    def extract(self, image): return emb(0)


class FrameFaceProcessor:
    def __init__(self): self.calls = 0

    def extract_faces(self, image):
        self.calls += 1
        return [(np.array([15, 15, 35, 40], dtype=np.float32), emb(0))]


def detection(box, confidence=1.0): return SimpleNamespace(box=box, confidence=confidence, mask=None)


class TrackerTests(unittest.TestCase):
    def make_tracker(self, **kwargs):
        defaults = dict(min_hits_to_confirm=1, max_lost_frames=1, max_occlusion_frames=5,
                        max_recovery_frames=8, recovery_reid_threshold=.75,
                        motion_gate_threshold=4, recovery_motion_gate=8)
        defaults.update(kwargs)
        return ReIDTracker(EmptyGallery(), DummyExtractor(), **defaults)

    def feed(self, tracker, boxes, embeddings):
        values = iter(embeddings)
        tracker._extract_embeddings = lambda image, box, mask: (next(values), None)
        return tracker.update(np.zeros((300, 300, 3), dtype=np.uint8), [detection(box) for box in boxes])

    def test_normal_tracking_keeps_id(self):
        tracker = self.make_tracker()
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [(14, 10, 44, 90)], [emb(0)])
        self.assertEqual([track.tracker_id for track in tracker.tracks], [0])

    def test_short_occlusion_recovers_same_id(self):
        tracker = self.make_tracker()
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [], [])
        self.feed(tracker, [], [])
        self.assertEqual(tracker.tracks[0].state, TrackState.LOST)
        self.feed(tracker, [(16, 10, 46, 90)], [emb(0)])
        self.assertEqual(tracker.tracks[0].tracker_id, 0)
        self.assertEqual(tracker.tracks[0].state, TrackState.RECOVERED)

    def test_person_occlusion_records_soft_owner_and_recovers(self):
        tracker = self.make_tracker(occlusion_iou_threshold=.05)
        self.feed(tracker, [(10, 10, 40, 90), (50, 10, 80, 90)], [emb(0), emb(1)])
        self.feed(tracker, [(15, 10, 45, 90), (45, 10, 85, 90)], [emb(0), emb(1)])
        self.feed(tracker, [(15, 10, 55, 90)], [emb(1)])
        hidden = next(t for t in tracker.tracks if t.tracker_id == 0)
        self.assertEqual(hidden.state, TrackState.OCCLUDED)
        self.assertEqual(hidden.occluded_by, 1)
        self.feed(tracker, [(20, 10, 50, 90), (15, 10, 55, 90)], [emb(0), emb(1)])
        self.assertEqual(next(t for t in tracker.tracks if t.tracker_id == 0).state, TrackState.RECOVERED)

    def test_turn_after_occlusion_uses_reid_not_precise_prediction(self):
        tracker = self.make_tracker(recovery_motion_gate=8)
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [(30, 10, 60, 90)], [emb(0)])
        self.feed(tracker, [], [])
        self.feed(tracker, [(45, 65, 75, 145)], [emb(0)])
        self.assertEqual(tracker.tracks[0].tracker_id, 0)

    def test_velocity_change_does_not_fragment(self):
        tracker = self.make_tracker()
        for x in (10, 13, 35, 38): self.feed(tracker, [(x, 10, x+30, 90)], [emb(0)])
        self.assertEqual(len(tracker.tracks), 1)

    def test_crossing_does_not_swap_appearance_ids(self):
        tracker = self.make_tracker()
        self.feed(tracker, [(10, 10, 40, 90), (100, 10, 130, 90)], [emb(0), emb(1)])
        self.feed(tracker, [(60, 10, 90, 90), (50, 10, 80, 90)], [emb(0), emb(1)])
        self.assertGreater(float(tracker.tracks[0].body_embedding[0]), .9)
        self.assertGreater(float(tracker.tracks[1].body_embedding[1]), .9)

    def test_long_disappearance_recovers_within_lifetime(self):
        tracker = self.make_tracker(max_lost_frames=1, max_recovery_frames=5)
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [], []); self.feed(tracker, [], [])
        self.feed(tracker, [(20, 10, 50, 90)], [emb(0)])
        self.assertEqual(tracker.tracks[0].tracker_id, 0)

    def test_weak_reid_cannot_steal_lost_id(self):
        tracker = self.make_tracker()
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [], [])
        self.feed(tracker, [], [])
        self.feed(tracker, [(16, 10, 46, 90)], [np.array([.7, .714, 0, 0], dtype=np.float32)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_low_confidence_association_does_not_contaminate_gallery(self):
        tracker = self.make_tracker(gallery_update_threshold=.95)
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [(12, 10, 42, 90)], [np.array([.8, .6, 0, 0], dtype=np.float32)])
        self.assertEqual(len(tracker.tracks[0].body_gallery), 1)

    def test_expired_track_is_deleted(self):
        tracker = self.make_tracker(max_recovery_frames=2)
        self.feed(tracker, [(10, 10, 40, 90)], [emb(0)])
        self.feed(tracker, [], []); self.feed(tracker, [], []); self.feed(tracker, [], [])
        self.assertEqual(tracker.tracks, [])

    def test_overlapping_detections_keep_only_the_highest_confidence_box(self):
        tracker = self.make_tracker(detection_dedup_iou=.70)
        values = iter([emb(0)])
        tracker._extract_embeddings = lambda image, box, mask: (next(values), None)
        tracker.update(np.zeros((300, 300, 3), dtype=np.uint8), [
            detection((10, 10, 50, 100), .95), detection((12, 10, 52, 100), .40)])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.debug_counters['detections_after_dedup'], 1)

    def test_weak_unmatched_detection_is_rejected_before_track_creation(self):
        tracker = self.make_tracker(min_detection_confidence=.8)
        tracker.update(np.zeros((300, 300, 3), dtype=np.uint8), [detection((10, 10, 50, 100), .5)])
        self.assertEqual(tracker.tracks, [])

    def test_two_nearby_people_are_not_merged(self):
        tracker = self.make_tracker(detection_dedup_iou=.70)
        self.feed(tracker, [(10, 10, 50, 100), (55, 10, 95, 100)], [emb(0), emb(1)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_tentative_duplicate_loses_to_confirmed_track(self):
        tracker = self.make_tracker(min_hits_to_confirm=2)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        self.feed(tracker, [(12, 10, 52, 100)], [emb(0)])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].state, TrackState.CONFIRMED)

    def test_long_occlusion_stays_recoverable_without_new_id(self):
        tracker = self.make_tracker(max_lost_frames=1, max_recovery_frames=6)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        self.feed(tracker, [], []); self.feed(tracker, [], [])
        self.feed(tracker, [(16, 10, 56, 100)], [emb(0)])
        self.assertEqual([track.tracker_id for track in tracker.tracks], [0])

    def make_identity_tracker(self, **kwargs):
        defaults = dict(min_hits_to_confirm=1, identity_candidate_min_frames=3,
                        identity_body_candidate_threshold=.70, identity_body_confirm_threshold=.90,
                        identity_face_threshold=.80, identity_margin=.08)
        defaults.update(kwargs)
        return ReIDTracker(IdentityGallery(), DummyExtractor(), **defaults)

    def test_strong_face_confirms_known_identity(self):
        tracker = self.make_identity_tracker()
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        track = tracker.tracks[0]; track.face_embedding = emb(0)
        tracker._frame_level_identity_assignment()
        self.assertEqual((track.name, track.identity_state, track.identity_source), ('alice', IdentityState.CONFIRMED, 'FACE'))

    def test_one_body_only_match_is_candidate_not_known_identity(self):
        tracker = self.make_identity_tracker()
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        track = tracker.tracks[0]
        self.assertEqual((track.name, track.identity_state, track.identity_candidate), ('Unknown', IdentityState.CANDIDATE, 'alice'))

    def test_repeated_strong_body_evidence_can_confirm_identity(self):
        tracker = self.make_identity_tracker()
        for x in (10, 12, 14): self.feed(tracker, [(x, 10, x+40, 100)], [emb(0)])
        track = tracker.tracks[0]
        self.assertEqual((track.name, track.identity_state, track.identity_source), ('alice', IdentityState.CONFIRMED, 'BODY_HISTORY'))

    def test_canonical_face_owner_blocks_body_only_identity_theft(self):
        tracker = self.make_identity_tracker(identity_candidate_min_frames=1)
        self.feed(tracker, [(10, 10, 50, 100), (150, 10, 190, 100)], [emb(0), emb(0)])
        first, second = tracker.tracks; first.face_embedding = emb(0)
        tracker._frame_level_identity_assignment()
        self.assertEqual(first.name, 'alice')
        self.assertEqual(second.name, 'Unknown')
        self.assertEqual(second.identity_rejection_reason, 'CANONICAL_OWNER_EXISTS')

    def test_confirmed_identity_is_retained_when_face_disappears(self):
        tracker = self.make_identity_tracker()
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        track = tracker.tracks[0]; track.face_embedding = emb(0); tracker._frame_level_identity_assignment()
        track.face_embedding = None; tracker.body_embedding = np.array([.8, .6, 0, 0], dtype=np.float32)
        tracker._frame_level_identity_assignment()
        self.assertEqual((track.name, track.identity_state, track.identity_source), ('alice', IdentityState.RETAINED, 'HISTORY'))

    def test_v06_active_owner_cannot_be_replaced_by_competing_face_claim(self):
        tracker = self.make_identity_tracker()
        self.feed(tracker, [(10, 10, 50, 100), (150, 10, 190, 100)], [emb(0), emb(0)])
        first, second = tracker.tracks
        first.face_embedding = emb(0)
        second.face_embedding = emb(0)
        tracker._frame_level_identity_assignment()
        self.assertEqual(first.name, 'alice')
        self.assertEqual(second.name, 'Unknown')
        self.assertEqual(second.identity_rejection_reason, 'CANONICAL_OWNER_EXISTS')

    def test_v06_lost_identity_lock_blocks_competing_track_until_timeout(self):
        tracker = self.make_identity_tracker(max_lost_frames=1, identity_lock_timeout=3)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        locked = tracker.tracks[0]
        locked.face_embedding = emb(0)
        tracker._frame_level_identity_assignment()
        self.feed(tracker, [], [])
        self.feed(tracker, [], [])
        self.assertEqual(locked.state, TrackState.LOST)
        self.assertTrue(locked.identity_locked)
        competing = Track(99, np.array([150, 10, 190, 100], dtype=np.float32), emb(0), emb(0), TrackState.CONFIRMED, hits=1)
        tracker.tracks.append(competing)
        tracker._frame_level_identity_assignment()
        self.assertEqual(competing.name, 'Unknown')
        self.assertEqual(competing.identity_rejection_reason, 'CANONICAL_OWNER_EXISTS')
        tracker.frame_count += 3
        locked.state = TrackState.DELETED
        self.assertTrue(tracker._release_identity_owner(locked))
        self.assertNotIn('alice', tracker.identity_owners)

    def test_identity_lock_frame_does_not_move_during_misses(self):
        tracker = self.make_identity_tracker(identity_lock_timeout=20)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        track = tracker.tracks[0]; track.face_embedding = emb(0)
        tracker._frame_level_identity_assignment()
        self.feed(tracker, [], [])
        tracker._lock_identity(track)
        lock_frame = tracker.frame_count
        for _ in range(4):
            self.feed(tracker, [], [])
        self.assertEqual(track.identity_lock_frame, lock_frame)

    def test_identity_ownership_expires_before_track_recovery_lifetime(self):
        tracker = self.make_identity_tracker(max_lost_frames=1, max_recovery_frames=8, identity_lock_timeout=2)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        track = tracker.tracks[0]; track.face_embedding = emb(0)
        tracker._frame_level_identity_assignment()
        self.feed(tracker, [], []); self.feed(tracker, [], [])
        self.assertTrue(track.identity_locked)
        self.feed(tracker, [], [])
        self.feed(tracker, [], [])
        self.assertFalse(track.identity_locked)
        self.assertNotIn('alice', tracker.identity_owners)
        self.assertIn(track, tracker.tracks)

    def test_weak_spatial_and_appearance_match_is_rejected(self):
        tracker = self.make_tracker(normal_reid_gate=.35, normal_iou_gate=.05)
        track = Track(0, np.array([10, 10, 50, 100], dtype=np.float32), emb(0))
        track.predicted_bbox = np.array([100, 100, 140, 190], dtype=np.float32)
        track.motion_confidence = 1.
        result = tracker._passes_association_gates(track, np.array([150, 150, 190, 240], dtype=np.float32), emb(1), None)
        self.assertFalse(result.accepted)
        self.assertIn(result.reason, ('IOU_GATE_FAIL', 'MOTION_GATE_FAIL', 'COMBINED_GATE_FAIL'))

    def test_strong_reid_can_pass_low_iou_with_valid_motion(self):
        tracker = self.make_tracker(normal_reid_gate=.35, strong_reid_gate=.55, normal_iou_gate=.05)
        track = Track(0, np.array([10, 10, 50, 100], dtype=np.float32), emb(0))
        track.predicted_bbox = np.array([12, 10, 52, 100], dtype=np.float32)
        result = tracker._passes_association_gates(track, np.array([52, 10, 92, 100], dtype=np.float32), emb(0), None)
        self.assertTrue(result.accepted)

    def test_recovery_margin_rejects_ambiguous_best_candidate(self):
        tracker = self.make_tracker(recovery_reid_threshold=.65, recovery_margin=.10)
        track = Track(0, np.array([10, 10, 50, 100], dtype=np.float32), emb(0), state=TrackState.LOST)
        tracker._compute_association_cost = lambda current, box, body, face, recovery=False: .71 if box[0] == 10 else .69
        matches, unmatched_tracks, unmatched_detections = tracker._associate_detections(
            [track], [np.array([10, 10, 50, 100], dtype=np.float32), np.array([20, 10, 60, 100], dtype=np.float32)],
            [emb(0), emb(0)], [None, None], recovery=True)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(sorted(unmatched_detections), [0, 1])

    def test_face_relationships_cover_every_face_and_person_box(self):
        tracker = self.make_tracker()
        face_results = [(np.array([12, 12, 28, 36], dtype=np.float32), emb(0)),
                        (np.array([120, 20, 140, 48], dtype=np.float32), emb(1))]
        person_boxes = [np.array([10, 10, 50, 100], dtype=np.float32),
                        np.array([100, 10, 160, 110], dtype=np.float32),
                        np.array([200, 10, 250, 110], dtype=np.float32)]
        relationships = tracker._calculate_face_bbox_relationships(face_results, person_boxes)
        self.assertEqual(len(relationships), 2)
        self.assertEqual([len(row) for row in relationships], [3, 3])
        self.assertTrue(relationships[0][0]['face_center_inside'])
        self.assertTrue(relationships[1][1]['face_center_inside'])
        self.assertFalse(relationships[0][2]['face_center_inside'])

    def test_face_person_results_distinguish_clear_ambiguous_weak_and_missing(self):
        tracker = self.make_tracker()
        person_boxes = [np.array([0, 0, 100, 200], dtype=np.float32), np.array([0, 0, 100, 200], dtype=np.float32)]
        clear, _, results, _, _ = tracker._associate_faces_to_persons(
            [(np.array([10, 10, 30, 40], dtype=np.float32), emb(0))], person_boxes,
            min_match_score=.35, ambiguity_margin=.10)
        self.assertEqual(results[0].reason, 'FACE_PERSON_AMBIGUOUS')
        self.assertEqual(clear, {})
        weak, _, results, _, _ = tracker._associate_faces_to_persons(
            [(np.array([99, 10, 119, 30], dtype=np.float32), emb(0))], [person_boxes[0]],
            min_match_score=.35)
        self.assertEqual(results[0].reason, 'FACE_PERSON_LOW_SCORE')
        self.assertEqual(weak, {})
        missing, _, results, _, _ = tracker._associate_faces_to_persons(
            [(np.array([150, 10, 170, 30], dtype=np.float32), emb(0))], [person_boxes[0]],
            min_match_score=.35)
        self.assertEqual(results[0].reason, 'FACE_PERSON_NO_CANDIDATE')
        self.assertEqual(missing, {})

    def test_face_person_conflict_keeps_strongest_face(self):
        tracker = self.make_tracker()
        faces = [(np.array([10, 10, 40, 50], dtype=np.float32), emb(0)),
                 (np.array([20, 20, 60, 70], dtype=np.float32), emb(1))]
        assignments, _, results, _, _ = tracker._associate_faces_to_persons(
            faces, [np.array([0, 0, 100, 200], dtype=np.float32)], min_match_score=.35)
        self.assertEqual(list(assignments), [0])
        self.assertEqual(results[0].reason, 'FACE_PERSON_ACCEPTED')
        self.assertEqual(results[1].reason, 'FACE_PERSON_PERSON_CONFLICT')

    def test_prediction_is_limited_after_one_missed_frame(self):
        track = Track(0, np.array([10, 10, 50, 100], dtype=np.float32), emb(0))
        track.velocity = np.array([200., 0.], dtype=np.float32)
        result = track.predict(velocity_decay=.7, max_displacement_per_frame=12., max_total_displacement=30.)
        self.assertLessEqual(track.prediction_displacement, 12.)
        self.assertFalse(result['expired'])

    def test_repeated_prediction_misses_damp_velocity_and_raise_uncertainty(self):
        track = Track(0, np.array([10, 10, 50, 100], dtype=np.float32), emb(0))
        track.velocity = np.array([20., 0.], dtype=np.float32)
        track.predict(velocity_decay=.5, uncertainty_growth=.2)
        first_velocity = float(np.linalg.norm(track.kalman.x[4:6]))
        first_uncertainty = track.prediction_uncertainty
        track.predict(velocity_decay=.5, uncertainty_growth=.2)
        self.assertLess(float(np.linalg.norm(track.kalman.x[4:6])), first_velocity)
        self.assertGreater(track.prediction_uncertainty, first_uncertainty)

    def test_prediction_hides_at_high_uncertainty_without_deleting_track(self):
        tracker = self.make_tracker(max_recovery_frames=6)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        track = tracker.tracks[0]
        for _ in range(4): self.feed(tracker, [], [])
        self.assertFalse(track.prediction_visible)
        self.assertIn(track, tracker.tracks)
        self.assertEqual(track.tracker_id, 0)

    def test_final_cleanup_removes_unconfirmed_tentative_tracks(self):
        tracker = self.make_tracker(min_hits_to_confirm=3)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        self.assertEqual(len(tracker.tracks), 1)
        tracker.cleanup(final=True)
        self.assertEqual(tracker.tracks, [])

    def test_recovery_is_not_counted_when_all_candidates_fail_hard_gates(self):
        tracker = self.make_tracker(max_lost_frames=1, max_recovery_frames=5, recovery_reid_threshold=.99)
        self.feed(tracker, [(10, 10, 50, 100)], [emb(0)])
        self.feed(tracker, [], []); self.feed(tracker, [], [])
        self.feed(tracker, [(16, 10, 56, 100)], [emb(1)])
        self.assertEqual(tracker.debug_counters['recovery_attempts'], 0)

    def test_configured_face_person_minimum_is_enforced_by_association_api(self):
        tracker = self.make_tracker(face_person_min_match_score=.99)
        assignments, _, results, _, _ = tracker._associate_faces_to_persons(
            [(np.array([10, 10, 30, 40], dtype=np.float32), emb(0))],
            [np.array([0, 0, 100, 200], dtype=np.float32)])
        self.assertEqual(assignments, {})
        self.assertEqual(results[0].reason, 'FACE_PERSON_LOW_SCORE')

    def test_frame_face_api_runs_once_and_reaches_person_track(self):
        face_processor = FrameFaceProcessor()
        tracker = ReIDTracker(EmptyGallery(), DummyExtractor(), face_processor=face_processor, min_hits_to_confirm=1)
        tracker.update(np.zeros((300, 300, 3), dtype=np.uint8), [detection((10, 10, 50, 100))])
        self.assertEqual(face_processor.calls, 1)
        self.assertTrue(np.array_equal(tracker.tracks[0].face_embedding, emb(0)))

    def test_empty_frame_skips_frame_face_inference(self):
        face_processor = FrameFaceProcessor()
        tracker = ReIDTracker(EmptyGallery(), DummyExtractor(), face_processor=face_processor, min_hits_to_confirm=1)
        tracker.update(np.zeros((300, 300, 3), dtype=np.uint8), [])
        self.assertEqual(face_processor.calls, 0)

    def test_tracking_config_is_exposed_and_reid_is_conditional(self):
        settings = __import__('app.config', fromlist=['Settings']).Settings()
        self.assertTrue(settings.tracking_config.use_segmentation_for_reid)
        self.assertGreater(settings.tracking_config.reid_interval, 0)
        self.assertGreater(settings.tracking_config.reid_batch_size, 0)

    def test_feature_extractor_supports_batch_reid(self):
        extractor = __import__('app.core.reid', fromlist=['FeatureExtractor']).FeatureExtractor(model_name='osnet_x1_0', device='cpu')
        extractor.extractor = SimpleNamespace(__call__=lambda images, *args, **kwargs: np.ones((2, 1), dtype=np.float32))
        result = extractor.batch_extract([np.ones((10, 10, 3), dtype=np.uint8), np.ones((10, 10, 3), dtype=np.uint8)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (1,))


if __name__ == '__main__': unittest.main()

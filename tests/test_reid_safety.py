import unittest
import numpy as np

from app.core.reid_safety import SafeTrack, validate_embedding


class ReIDSafetyTests(unittest.TestCase):
    def test_valid_embedding_is_normalized(self):
        ok, value, reason = validate_embedding(np.array([3., 4.], dtype=np.float32), 2)
        self.assertTrue(ok)
        self.assertEqual(reason, 'OK')
        self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=6)

    def test_zero_embedding_is_rejected(self):
        ok, value, reason = validate_embedding(np.zeros(512, dtype=np.float32), 512)
        self.assertFalse(ok)
        self.assertIsNone(value)
        self.assertEqual(reason, 'ZERO_NORM')

    def test_wrong_dimension_is_rejected(self):
        ok, value, reason = validate_embedding(np.ones(256, dtype=np.float32), 512)
        self.assertFalse(ok)
        self.assertIsNone(value)
        self.assertEqual(reason, 'WRONG_DIMENSION')

    def test_nonfinite_embedding_is_rejected(self):
        value = np.ones(512, dtype=np.float32)
        value[10] = np.nan
        ok, result, reason = validate_embedding(value, 512)
        self.assertFalse(ok)
        self.assertIsNone(result)
        self.assertEqual(reason, 'NAN_OR_INF')

    def test_missing_initial_body_descriptor_has_no_zero_sentinel(self):
        track = SafeTrack(0, np.array([10, 10, 50, 100], dtype=np.float32), None)
        self.assertIsNone(track.body_embedding)
        self.assertIsNone(track.last_body_embedding)
        self.assertEqual(len(track.body_gallery), 0)

    def test_invalid_update_preserves_previous_valid_body(self):
        original = np.zeros(512, dtype=np.float32)
        original[0] = 1.0
        track = SafeTrack(0, np.array([10, 10, 50, 100], dtype=np.float32), original)
        previous = track.body_embedding.copy()
        invalid = np.zeros(256, dtype=np.float32)
        track.update(np.array([12, 10, 52, 100], dtype=np.float32), invalid, None, .9)
        np.testing.assert_allclose(track.body_embedding, previous)
        np.testing.assert_allclose(track.last_body_embedding, previous)
        self.assertEqual(track.reid_validation_stats['WRONG_DIMENSION'], 1)

    def test_nan_update_preserves_previous_valid_body(self):
        original = np.zeros(512, dtype=np.float32)
        original[1] = 1.0
        track = SafeTrack(0, np.array([10, 10, 50, 100], dtype=np.float32), original)
        invalid = original.copy()
        invalid[4] = np.inf
        track.update(np.array([12, 10, 52, 100], dtype=np.float32), invalid, None, .9)
        np.testing.assert_allclose(track.body_embedding, original)
        np.testing.assert_allclose(track.last_body_embedding, original)


if __name__ == '__main__':
    unittest.main()

"""Regression tests for observation-only discovery and separate evaluation."""
import inspect
import unittest
import numpy as np
from brain_states.state_detection import DetectorConfig, detect, boolean_product
from brain_states.evaluation import match_states, evaluate


class DetectionTests(unittest.TestCase):
    def test_observations_only_and_overlap(self):
        # Independent fixture: two groups with shared channel, not the research defaults.
        fs = 100
        t = np.arange(600)/fs
        first = ((t >= 0.5) & (t < 2.5)) | ((t >= 4) & (t < 5.5))
        second = (t >= 1.5) & (t < 4.5)
        x = np.zeros((5, len(t)))
        x[[0, 1, 2]] += first*np.sin(2*np.pi*3*t)
        x[[2, 3, 4]] += second*np.sin(2*np.pi*5*t)
        result = detect(x, fs)
        self.assertEqual(set(inspect.signature(detect).parameters), {'signals', 'sample_rate', 'config'})
        self.assertEqual(result['memberships'].shape, (5, 2))
        self.assertTrue(np.any(result['state_activations'].sum(axis=0) > 1))
        self.assertTrue(np.any(result['memberships'].sum(axis=1) > 1))
        self.assertTrue(np.all(np.diff(result['objective_history']) < 0))
        self.assertTrue(np.all(result['channel_activity'][0, 60:240]))
        np.testing.assert_array_equal(result['reconstruction'],
                                      boolean_product(result['memberships'], result['state_activations']))
        repeat = detect(x, fs)
        np.testing.assert_array_equal(result['memberships'], repeat['memberships'])
        high_penalty = detect(x, fs, DetectorConfig(state_penalty_seconds=1000))
        self.assertEqual(high_penalty['memberships'].shape[1], 0)

    def test_silence_and_validation(self):
        result = detect(np.zeros((4, 100)), 100)
        self.assertEqual(result['state_activations'].shape, (0, 100))
        self.assertFalse(result['channel_activity'].any())
        for x, fs in [(np.zeros((2, 0)), 100), (np.full((2, 100), np.nan), 100), (np.zeros((2, 100)), 0)]:
            with self.assertRaises(ValueError):
                detect(x, fs)

    def test_matching_is_optimal_and_label_invariant(self):
        self.assertEqual(match_states(np.array([[0.9, 0.8], [0.85, 0.1]])), [(0, 1), (1, 0)])
        self.assertEqual(match_states(np.zeros((2, 3))), [])
        w = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
        h = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=bool)
        activity = boolean_product(w, h)
        truth = dict(state_memberships=w, state_activations=h, channel_activations=activity)
        result = dict(memberships=w[:, ::-1], state_activations=h[::-1],
                      channel_activity=activity, reconstruction=activity)
        metrics = evaluate(result, truth)
        self.assertEqual(metrics['temporal_iou_with_unmatched_zero'], 1)
        self.assertEqual(metrics['matches'][0]['discovered_state'], 2)
        result.update(memberships=w[:, :1], state_activations=h[:1], reconstruction=boolean_product(w[:, :1], h[:1]))
        metrics = evaluate(result, truth)
        self.assertEqual(metrics['unmatched_true_states'], [2])
        self.assertEqual(metrics['temporal_iou_with_unmatched_zero'], 0.5)
        result.update(memberships=np.column_stack((w, [1, 0, 0])),
                      state_activations=np.vstack((h, [0, 0, 0, 1])))
        metrics = evaluate(result, truth)
        self.assertEqual(metrics['unmatched_discovered_states'], [3])


if __name__ == '__main__':
    unittest.main()

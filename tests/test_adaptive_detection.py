"""Observation-only adaptive estimation and paired experiment regression tests."""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from brain_states.activity_detection import AdaptiveConfig, estimate_adaptive_activity, detect_adaptive
from brain_states.state_detection import estimate_activity
from brain_states.experiments.adaptive_rms import paired_detect, run_comparison, verify_comparison, detector_hash, BASELINE_HASH


class AdaptiveDetectionTests(unittest.TestCase):
    def test_independent_thresholds_and_identical_windows(self):
        x = np.random.default_rng(8).normal(size=(3, 200)) * np.array([0.1, 0.3, 0.8])[:, None]
        config = AdaptiveConfig(percentile=30, scale=1.6, threshold_floor=0.04)
        result = estimate_adaptive_activity(x, 200, config)
        rms, _ = estimate_activity(x, 200, 0.1, 0.05)
        np.testing.assert_array_equal(result['rms'], rms)
        background = np.percentile(rms, 30, axis=1)
        np.testing.assert_array_equal(result['background_rms'], background)
        np.testing.assert_array_equal(result['thresholds'], np.maximum(0.04, 1.6*background))
        np.testing.assert_array_equal(result['channel_activity'], rms >= result['thresholds'][:, None])
        self.assertEqual(len(np.unique(result['thresholds'])), 3)
        for width in (1, 3, 22):
            recording = x[:, :width]
            adapted = estimate_adaptive_activity(recording, 200)
            fixed_rms, _ = estimate_activity(recording, 200)
            np.testing.assert_array_equal(adapted['rms'], fixed_rms)

    def test_floor_and_quantile_assumption_failure(self):
        result = detect_adaptive(np.zeros((3, 100)), 200)
        self.assertFalse(result['channel_activity'].any())
        np.testing.assert_array_equal(result['thresholds'], 0.05)
        # With no background, the estimator cannot tell a sustained signal from noise.
        always_on = estimate_adaptive_activity(np.ones((2, 100)), 200)
        np.testing.assert_allclose(always_on['thresholds'], 1.5)
        self.assertFalse(always_on['channel_activity'].any())
        for config in (AdaptiveConfig(percentile=-1), AdaptiveConfig(scale=1),
                       AdaptiveConfig(threshold_floor=0), AdaptiveConfig(percentile=float('nan'))):
            with self.assertRaises(ValueError):
                estimate_adaptive_activity(np.zeros((2, 10)), 200, config)

    def test_paired_boundary_receives_same_readonly_array(self):
        import brain_states.experiments.adaptive_rms as experiment_adaptive
        fixed_original = experiment_adaptive.detect
        adaptive_original = experiment_adaptive.detect_adaptive
        seen = []
        def fixed(x, fs, config):
            self.assertFalse(x.flags.writeable)
            seen.append(x)
            return fixed_original(x, fs, config)
        def adaptive(x, fs, config, factor_config):
            self.assertIs(x, seen[0])
            self.assertEqual(fs, 200)
            return adaptive_original(x, fs, config, factor_config)
        with patch.object(experiment_adaptive, 'detect', fixed), patch.object(experiment_adaptive, 'detect_adaptive', adaptive):
            paired_detect(np.zeros((4, 200)), 200, AdaptiveConfig())
        self.assertEqual(detector_hash(), BASELINE_HASH)

    def test_small_reproducible_paired_pipeline(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rows, pairs, summaries, paired_summary, _, _, metadata = run_comparison(repeats=2, base_seed=8)
            repeat = run_comparison(repeats=2, base_seed=8)
        self.assertEqual(rows, repeat[0])
        self.assertEqual(pairs, repeat[1])
        self.assertEqual(len(rows), 32)
        self.assertEqual(len(pairs), 16)
        self.assertEqual(len(summaries), 16)
        self.assertEqual(len(paired_summary), 8)
        verify_comparison(rows, 2)
        for pair in pairs:
            records = {r['detector']: r for r in rows if r['sigma'] == pair['sigma'] and r['run'] == pair['run']}
            self.assertAlmostEqual(pair['activity_f1'], records['adaptive']['activity_f1'] - records['fixed']['activity_f1'])
        self.assertEqual(metadata['adaptive'], {'percentile': 20.0, 'scale': 1.5, 'threshold_floor': 0.05})


if __name__ == '__main__':
    unittest.main()

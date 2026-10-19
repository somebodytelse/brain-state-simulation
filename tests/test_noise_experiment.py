"""Noise generation, reproducibility and repeated-experiment checks."""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import brain_states.experiments.noise_robustness as experiment_noise
from brain_states.simulation import simulate, add_gaussian_noise
from brain_states.state_detection import detect


class NoiseExperimentTests(unittest.TestCase):
    def test_noise_everywhere_and_clean_compatibility(self):
        clean = simulate()
        noisy = simulate(noise_std=0.2, seed=123)
        self.assertEqual(set(clean), {'time', 'signals', 'state_activations',
                                     'channel_activations', 'state_memberships'})
        np.testing.assert_array_equal(noisy['clean_signals'], clean['signals'])
        for key in ('time', 'state_activations', 'channel_activations', 'state_memberships'):
            np.testing.assert_array_equal(noisy[key], clean[key])
        residual = noisy['signals'] - clean['signals']
        inactive = clean['channel_activations'] == 0
        self.assertTrue(np.all(residual[inactive] != 0))
        self.assertTrue(np.all(residual[~inactive] != 0))
        for mask in (inactive, ~inactive):
            self.assertAlmostEqual(float(np.std(residual[mask])), 0.2, delta=0.01)
            self.assertAlmostEqual(float(np.mean(residual[mask])), 0, delta=0.01)
        expected = np.random.default_rng(123).normal(0, 0.2, clean['signals'].shape)
        np.testing.assert_allclose(residual, expected, atol=1e-15)
        np.testing.assert_array_equal(noisy['signals'], simulate(noise_std=0.2, seed=123)['signals'])
        self.assertFalse(np.array_equal(noisy['signals'], simulate(noise_std=0.2, seed=124)['signals']))
        np.testing.assert_array_equal(add_gaussian_noise(clean['signals'], 0), clean['signals'])
        for sigma in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                add_gaussian_noise(clean['signals'], sigma)

    def test_small_pipeline_and_no_truth_at_detector_boundary(self):
        calls = []
        def observed_only(signals, fs, config):
            self.assertIsInstance(signals, np.ndarray)
            self.assertEqual(signals.shape, (8, 2400))
            self.assertEqual(fs, 200)
            calls.append(signals.copy())
            return detect(signals, fs, config)
        with patch.object(experiment_noise, 'detect', observed_only), contextlib.redirect_stdout(io.StringIO()):
            rows, examples, metadata = experiment_noise.run_trials(repeats=2, base_seed=17)
        self.assertEqual(len(calls), 16)
        self.assertFalse(np.array_equal(calls[2], calls[3]))
        experiment_noise.verify_trials(rows, 2)
        self.assertEqual(set(examples), {0, 0.2, 0.5})
        self.assertEqual(metadata['detector']['threshold'], 0.05)
        summary = experiment_noise.summarize(rows)
        self.assertEqual(len(summary), 8)
        self.assertTrue(all(row['runs'] == 2 for row in summary))
        self.assertTrue(all(row['temporal_iou'] <= row['matched_temporal_iou'] for row in rows))
        self.assertTrue(all(row['membership_jaccard'] <= row['matched_membership_jaccard'] for row in rows))
        with contextlib.redirect_stdout(io.StringIO()):
            repeat, _, _ = experiment_noise.run_trials(repeats=2, base_seed=17)
        self.assertEqual(rows, repeat)


if __name__ == '__main__':
    unittest.main()

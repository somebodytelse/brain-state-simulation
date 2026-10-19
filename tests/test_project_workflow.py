"""Packaging/refactor checks independent of local generated output directories."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from brain_states.config import SIMULATION, DETECTOR, ADAPTIVE, CONNECTIVITY
from brain_states.simulation import simulate
from brain_states.state_detection import detect
from brain_states.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


class ProjectWorkflowTests(unittest.TestCase):
    def test_baseline_matches_portable_pre_refactor_snapshot(self):
        truth = simulate(**SIMULATION.kwargs())
        inferred = detect(truth['signals'], SIMULATION.sample_rate, DETECTOR)
        actual = evaluate(inferred, truth)
        expected = json.loads((ROOT/'docs/reference_results/baseline_metrics.json').read_text(encoding='utf-8'))
        for key in ('temporal_iou_with_unmatched_zero', 'membership_jaccard_with_unmatched_zero',
                    'boolean_reconstruction_error', 'reconstruction_error_against_ground_truth'):
            self.assertAlmostEqual(actual[key], expected[key], places=14)
        for key in ('precision', 'recall', 'f1'):
            self.assertAlmostEqual(actual['channel_activity'][key], expected['channel_activity'][key], places=14)
        self.assertEqual(actual['matches'], expected['matches'])
        self.assertEqual(actual['discovered_state_count'], 3)

    def test_configuration_registry_retains_defaults(self):
        self.assertEqual((SIMULATION.sample_rate,SIMULATION.duration,SIMULATION.n_channels),(200,12,8))
        self.assertEqual((ADAPTIVE.percentile,ADAPTIVE.scale,ADAPTIVE.threshold_floor),(20,1.5,0.05))
        self.assertEqual((DETECTOR.min_channels,DETECTOR.min_duration_seconds),(2,0.2))
        self.assertEqual((CONNECTIVITY.window_seconds,CONNECTIVITY.step_seconds),(1,0.1))

    def test_old_imports_alias_canonical_modules(self):
        import state_detection
        import brain_states.state_detection
        import overlapping_signals
        import brain_states.simulation
        self.assertIs(state_detection, brain_states.state_detection)
        self.assertIs(overlapping_signals, brain_states.simulation)
        # Importing the old test.py no longer executes a clustering experiment.
        result = subprocess.run([sys.executable,'-c','import test'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'')

    def test_documented_modules_offer_help(self):
        for module in ('brain_states.simulation','brain_states.experiments.baseline',
                       'brain_states.experiments.noise_robustness','brain_states.experiments.adaptive_rms',
                       'brain_states.experiments.connectivity','brain_states.experiments.presentation',
                       'brain_states.experiments.noisy_presentation'):
            with self.subTest(module=module):
                result=subprocess.run([sys.executable,'-m',module,'--help'],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('usage:',result.stdout)


if __name__=='__main__':
    unittest.main()

"""Jaccard, window-grid, reference separation and connectivity regression tests."""
import inspect
import unittest
from unittest.mock import patch
import numpy as np
from brain_states.connectivity import (ConnectivityConfig, estimate_connectivity, jaccard_adjacency,
                          dynamic_adjacency, validate_adjacency)
from brain_states.connectivity_reference import build_references
from brain_states.simulation import simulate


class ConnectivityTests(unittest.TestCase):
    def test_known_jaccard_and_empty_union(self):
        a = np.array([[1,1,0,0], [1,0,1,0], [0,0,0,0], [0,0,0,0]])
        adjacency = jaccard_adjacency(a)
        self.assertAlmostEqual(adjacency[0,1], 1/3)
        self.assertEqual(adjacency[2,3], 0)
        validate_adjacency(adjacency)
        np.testing.assert_array_equal(np.diag(adjacency), 0)
        # More than 255 samples must not overflow uint8 intersection counts.
        many = jaccard_adjacency(np.ones((2, 1000), dtype=np.uint8))
        self.assertEqual(many[0,1], 1)

    def test_uniform_windows_independent_of_transitions(self):
        a = np.zeros((2, 250), dtype=bool)
        a[:, 37:183] = True  # Neither transition lies on the step grid.
        result = dynamic_adjacency(a, 100, 1, 0.1)
        np.testing.assert_array_equal(result['window_starts'], np.arange(0,151,10))
        np.testing.assert_allclose(result['window_centers'], np.arange(16)/10+0.5)
        self.assertEqual(result['dynamic_adjacency'].shape, (16,2,2))
        self.assertEqual(result['dynamic_adjacency'][0,0,1], 1)
        # A single exact-width window is valid.
        self.assertEqual(dynamic_adjacency(a[:,:100],100)['dynamic_adjacency'].shape,(1,2,2))
        for window, step in [(3,0.1), (1,0), (0.001,0.1), (1,0.015)]:
            with self.assertRaises(ValueError):
                dynamic_adjacency(a,100,window,step)

    def test_independent_coactivity_is_not_a_shared_state(self):
        # Two independent states are continuously active together.
        w = np.array([[1,0],[1,0],[0,1],[0,1]])
        h = np.ones((2,100), dtype=int)
        estimated = jaccard_adjacency(w @ h > 0)
        refs = build_references(w, h, np.array([0]), np.array([100]))
        self.assertEqual(estimated[0,2], 1)
        self.assertFalse(refs['static_reference'][0,2])
        self.assertFalse(refs['center_reference'][0,0,2])
        self.assertEqual(refs['window_fraction_reference'][0,0,2], 0)
        self.assertTrue(refs['samplewise_reference'][:,0,1].all())
        for matrix in refs.values():
            validate_adjacency(matrix)

    def test_reference_occupancy_is_separate_from_jaccard(self):
        refs = build_references(np.ones((2,1)), np.array([[0,0,1,1]]),
                                np.array([0]), np.array([4]))
        self.assertEqual(refs['window_fraction_reference'][0,0,1],0.5)
        self.assertTrue(refs['center_reference'][0,0,1])
        self.assertEqual(jaccard_adjacency(np.array([[0,0,1,1],[0,0,1,1]]))[0,1],1)

    def test_observed_only_api_and_no_factorization(self):
        self.assertEqual(set(inspect.signature(estimate_connectivity).parameters),
                         {'signals','sample_rate','config'})
        x = np.random.default_rng(31).normal(size=(4,400))
        original = x.copy()
        with patch('brain_states.state_detection.factorize', side_effect=AssertionError('No state discovery allowed')):
            for method in ('fixed','adaptive'):
                config = ConnectivityConfig(activity_method=method)
                first = estimate_connectivity(x,200,config)
                second = estimate_connectivity(x.copy(),200,config)
                for name in first:
                    np.testing.assert_array_equal(first[name],second[name])
                validate_adjacency(first['dynamic_adjacency'])
        np.testing.assert_array_equal(x,original)

    def test_default_experiment(self):
        data = simulate()
        result = estimate_connectivity(data['signals'],200)
        self.assertEqual(result['static_adjacency'].shape,(8,8))
        self.assertEqual(result['dynamic_adjacency'].shape,(111,8,8))
        self.assertEqual(result['window_centers'][0],0.5)
        self.assertEqual(result['window_centers'][-1],11.5)
        refs = build_references(data['state_memberships'],data['state_activations'],
                                result['window_starts'], result['window_ends'])
        for center in (3.0,9.5):
            index = np.argmin(np.abs(result['window_centers']-center))
            self.assertEqual(result['dynamic_adjacency'][index,0,3],1)
            self.assertFalse(refs['center_reference'][index,0,3])
        self.assertEqual(result['static_adjacency'][0,1],1)
        self.assertEqual(result['static_adjacency'][3,4],1)
        self.assertGreater(result['static_adjacency'][0,3],0)

    def test_invalid_inputs_and_matrix_checks(self):
        for activity in (np.zeros((2,0)), np.array([[0,2]]), np.ones(3)):
            with self.assertRaises(ValueError):
                jaccard_adjacency(activity)
        for matrix in (np.ones((2,2)), np.array([[0,0.2],[0.3,0]]),
                       np.array([[0,np.nan],[np.nan,0]]), np.array([[0,-1],[-1,0]])):
            with self.assertRaises(ValueError):
                validate_adjacency(matrix)
        with self.assertRaises(ValueError):
            estimate_connectivity(np.zeros((2,400)),200,ConnectivityConfig(activity_method='unknown'))


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from brain_states.simulation import State, simulate, visualize, save_data


class OverlappingSignalTests(unittest.TestCase):
    def test_specification(self):
        data = simulate()
        expected_shapes = {'time': (2400,), 'signals': (8, 2400),
                           'state_activations': (3, 2400),
                           'channel_activations': (8, 2400), 'state_memberships': (8, 3)}
        for key, shape in expected_shapes.items():
            self.assertEqual(data[key].shape, shape)
        t = data['time']
        expected = np.array([((t >= 1) & (t < 4)) | ((t >= 8) & (t < 10)),
                             ((t >= 2) & (t < 6)) | ((t >= 9) & (t < 12)),
                             (t >= 5) & (t < 9)])
        np.testing.assert_array_equal(data['state_activations'], expected)
        memberships = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0],
                                [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        np.testing.assert_array_equal(data['state_memberships'], memberships)
        np.testing.assert_array_equal(data['channel_activations'], memberships @ expected.astype(int) > 0)
        np.testing.assert_array_equal(data['signals'][data['channel_activations'] == 0], 0)
        overlap = (t >= 8) & (t < 9)
        np.testing.assert_allclose(data['signals'][2, overlap],
                                   np.sin(2*np.pi*2*t[overlap]) + np.sin(2*np.pi*3*t[overlap]))
        self.assertGreater(np.max(np.abs(data['signals'][2, overlap])), 1)
        # Active channels remain labelled ON at zero-valued sine crossings.
        self.assertEqual(data['channel_activations'][0, 200], 1)

    def test_configurable_phase(self):
        data = simulate(states=(State('X', (1,), ((0, 12),), phase=np.pi/2),))
        self.assertAlmostEqual(data['signals'][0, 0], 1)

    def test_saved_artifacts(self):
        plt.switch_backend('Agg')
        data = simulate()
        figure = visualize(data, 12)
        with tempfile.TemporaryDirectory() as directory:
            npz, png = save_data(data, figure, Path(directory))
            self.assertGreater(png.stat().st_size, 1000)
            with np.load(npz, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(data))
                for key in data:
                    np.testing.assert_array_equal(saved[key], data[key])
        plt.close(figure)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from brain_states.legacy.four_channel import generate_signals


class SignalTests(unittest.TestCase):
    def test_ground_truth(self):
        t, x, states = generate_signals()
        self.assertEqual(x.shape, (4500, 4))
        self.assertEqual(t[2500], 2.5)
        self.assertAlmostEqual(t[-1] + 0.001, 4.5)
        np.testing.assert_array_equal(states[:2500], 1)
        np.testing.assert_array_equal(states[2500:], 2)
        np.testing.assert_array_equal(x[:2500, 2:], 0)
        np.testing.assert_array_equal(x[2500:, :2], 0)
        np.testing.assert_array_equal(x[2500], 0)
        for start, stop, channel, frequency, cycles in [(0, 2500, 0, 2, 5), (2500, 4500, 2, 4, 8)]:
            wave = x[start:stop, channel]
            peak = np.argmax(np.abs(np.fft.rfft(wave)))
            self.assertEqual(peak, cycles)
            self.assertEqual(peak * 1000 / len(wave), frequency)
        np.testing.assert_array_equal(x[:, 0], x[:, 1])
        np.testing.assert_array_equal(x[:, 2], x[:, 3])

    def test_invalid_parameters(self):
        for rate in [0, 8, 999, float('nan')]:
            with self.assertRaises(ValueError):
                generate_signals(rate)
        with self.assertRaises(ValueError):
            generate_signals(amplitude=-1)


if __name__ == '__main__':
    unittest.main()

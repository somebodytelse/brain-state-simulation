# Configuration

`src/brain_states/config.py` is the entry point for experimental defaults.
`python -m brain_states.config` prints the complete registry as JSON.

- `SimulationConfig` / `SIMULATION`: sample rate, duration, amplitude, channel
  count, noise standard deviation, generator seed and original states.
- `DETECTOR`: `DetectorConfig` from `state_detection.py`; RMS window, fixed
  threshold, minimum channels/duration, state and membership penalties.
- `ADAPTIVE`: `AdaptiveConfig` from `activity_detection.py`; lower percentile,
  scale and threshold floor.
- `CONNECTIVITY`: `ConnectivityConfig`; activity method, RMS settings, sliding
  window and step.
- `NOISE_LEVELS`, `NOISE_EXPERIMENT_SEED`, `PRESENTATION_NOISE_SEED`: fixed
  experiment grids and seeds. Original state definitions remain in
  `simulation.DEFAULT_STATES`, shared by the registry rather than copied.

Use the existing CLI options for run-specific overrides, and save their values
with results. Programmatic callers can pass these dataclasses/functions directly.
The low-level detector signatures retain their historical default values; pass
an edited configuration explicitly rather than expecting global changes to
silently alter a scientific API. The original detector source remains byte-for-
byte preserved to honor historical fingerprint checks.

No file in this folder contains credentials. The reference constraints record
the tested Python 3.12 dependency environment. They are for reproducing the
reference runs, not a universal lockfile for every Python/platform combination.

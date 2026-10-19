"""Central experiment defaults, assembled from the existing scientific parameter classes.

No configuration contains ground-truth-derived tuning. Observation-only detector
settings remain separate from the simulation's known state specification.
"""
from dataclasses import asdict, dataclass
from brain_states.simulation import DEFAULT_STATES, State
from brain_states.state_detection import DetectorConfig
from brain_states.activity_detection import AdaptiveConfig
from brain_states.connectivity import ConnectivityConfig


@dataclass(frozen=True)
class SimulationConfig:
    """Recording settings; X has (n_channels, sample_rate * duration) samples."""
    sample_rate: float = 200.0
    duration: float = 12.0
    amplitude: float = 1.0
    n_channels: int = 8
    noise_std: float = 0.0
    seed: int = 42
    states: tuple[State, ...] = DEFAULT_STATES

    def kwargs(self) -> dict:
        """Arguments for simulate(), retaining the original immutable State objects."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


SIMULATION = SimulationConfig()
DETECTOR = DetectorConfig()
ADAPTIVE = AdaptiveConfig()
CONNECTIVITY = ConnectivityConfig()
NOISE_LEVELS = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
NOISE_EXPERIMENT_SEED = 20261012
PRESENTATION_NOISE_SEED = 20261015


def describe() -> dict:
    """Return JSON-serializable defaults for inspection and experiment provenance."""
    return dict(simulation=asdict(SIMULATION), detector=asdict(DETECTOR),
                adaptive=asdict(ADAPTIVE), connectivity=asdict(CONNECTIVITY),
                noise_levels=NOISE_LEVELS, noise_experiment_seed=NOISE_EXPERIMENT_SEED,
                presentation_noise_seed=PRESENTATION_NOISE_SEED)


if __name__ == '__main__':
    import json
    print(json.dumps(describe(), indent=2))

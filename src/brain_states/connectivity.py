"""Observed-only static and sliding-window Jaccard functional connectivity."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from brain_states.state_detection import estimate_activity
from brain_states.activity_detection import AdaptiveConfig, estimate_adaptive_activity


@dataclass(frozen=True)
class ConnectivityConfig:
    """Activity settings and a uniform, transition-independent connectivity grid."""
    activity_method: str = 'fixed'
    rms_window_seconds: float = 0.1
    fixed_threshold: float = 0.05
    adaptive: AdaptiveConfig = AdaptiveConfig()
    window_seconds: float = 1.0
    step_seconds: float = 0.1


def jaccard_adjacency(activity: NDArray) -> NDArray:
    """Pairwise active-time intersection/union; two empty rows have similarity 0.

    Diagonal is always zero. Shared inactivity does not count as coordination.
    This measures binary coactivity, not anatomical or state-specific links.
    """
    a = np.asarray(activity)
    if a.ndim != 2 or min(a.shape) < 1 or not np.all((a == 0) | (a == 1)):
        raise ValueError('Activity must be a nonempty binary channels-by-samples matrix.')
    counts = a.astype(np.int64)
    intersection = counts @ counts.T
    total = counts.sum(axis=1)
    union = total[:, None] + total[None, :] - intersection
    adjacency = np.divide(intersection, union, out=np.zeros(intersection.shape, dtype=float), where=union > 0)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def _sample_count(seconds: float, sample_rate: float) -> int:
    """Require exact sample-aligned window sizes instead of silently rounding."""
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Sampling rate and window/step durations must be finite and positive.')
    value = seconds * sample_rate
    if not np.isclose(value, round(value), rtol=0, atol=1e-9) or round(value) < 1:
        raise ValueError('Window/step duration must contain a positive whole number of samples.')
    return round(value)


def dynamic_adjacency(activity: NDArray, sample_rate: float,
                      window_seconds: float = 1.0, step_seconds: float = 0.1) -> dict[str, NDArray]:
    """Use full [start,end) windows on a fixed sample grid, with no truth input.

    Center = (start + width/2)/fs. Adjacent windows overlap; partial edge
    windows are omitted. Connectivity aggregates over the whole window, so
    changes are smeared across roughly half a window either side of a center,
    in addition to prior RMS blur. Transitions need not coincide with steps.
    """
    # Reuse validation for binary data and zero-union handling.
    jaccard_adjacency(activity)
    a = np.asarray(activity)
    width = _sample_count(window_seconds, sample_rate)
    step = _sample_count(step_seconds, sample_rate)
    if width > a.shape[1]:
        raise ValueError('Connectivity window exceeds recording length.')
    starts = np.arange(0, a.shape[1]-width+1, step, dtype=int)
    matrices = np.stack([jaccard_adjacency(a[:, start:start+width]) for start in starts])
    return dict(window_centers=(starts + width/2)/sample_rate,
                window_starts=starts, window_ends=starts+width, dynamic_adjacency=matrices)


def estimate_connectivity(signals: NDArray, sample_rate: float,
                          config: ConnectivityConfig = ConnectivityConfig()) -> dict[str, NDArray]:
    """Accept only recorded signals, sampling rate and observation-only settings."""
    if config.activity_method == 'fixed':
        rms, active = estimate_activity(signals, sample_rate, config.rms_window_seconds, config.fixed_threshold)
        extra = dict(thresholds=np.full(rms.shape[0], config.fixed_threshold))
    elif config.activity_method == 'adaptive':
        estimated = estimate_adaptive_activity(signals, sample_rate, config.adaptive, config.rms_window_seconds)
        rms, active = estimated['rms'], estimated['channel_activity']
        extra = {key: estimated[key] for key in ('thresholds', 'background_rms')}
    else:
        raise ValueError('Activity method must be fixed or adaptive.')
    result = dynamic_adjacency(active, sample_rate, config.window_seconds, config.step_seconds)
    return dict(rms=rms, channel_activity=active, static_adjacency=jaccard_adjacency(active), **result, **extra)


def validate_adjacency(matrices: NDArray) -> None:
    """Check matrix invariants for one matrix or an entire stack."""
    a = np.asarray(matrices)
    if a.ndim not in (2, 3) or a.shape[-2] != a.shape[-1] or a.shape[-1] < 1:
        raise ValueError('Expected a square adjacency matrix or stack of matrices.')
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1):
        raise ValueError('Connectivity must be finite and in [0,1].')
    if not np.allclose(a, a.swapaxes(-1, -2), rtol=0, atol=1e-12):
        raise ValueError('Adjacency must be symmetric.')
    if np.any(np.diagonal(a, axis1=-2, axis2=-1) != 0):
        raise ValueError('Adjacency diagonal must be zero.')

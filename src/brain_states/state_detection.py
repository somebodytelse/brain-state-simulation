"""Observed-data-only RMS and greedy Boolean factorization baseline."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DetectorConfig:
    """Penalties are channel-sample error equivalents per second of sampling."""
    window_seconds: float = 0.1
    threshold: float = 0.05
    min_channels: int = 2
    min_duration_seconds: float = 0.2
    state_penalty_seconds: float = 0.5
    membership_penalty_seconds: float = 0.02


def boolean_product(w: NDArray, h: NDArray) -> NDArray:
    """Boolean OR of state contributions; support shared channels and overlap."""
    return (w.astype(np.int64) @ h.astype(np.int64)) > 0


def estimate_activity(signals: NDArray, sample_rate: float,
                      window_seconds: float = 0.1, threshold: float = 0.05) -> tuple[NDArray, NDArray]:
    """Centered odd-window RMS, using edge padding. Offline, not causal.

    Smoothing bridges zero crossings but spreads transitions by about half a
    window. Smaller thresholds increase this spread; larger thresholds can
    miss weak activity. Low-frequency or cancelling waves may require longer
    windows. No ground truth is consulted.
    """
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError('Signals must be a nonempty finite channels-by-samples matrix.')
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('Sample rate must be finite and positive.')
    if not np.isfinite(window_seconds) or window_seconds <= 0 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Window and threshold must be finite and positive.')
    width = max(1, round(window_seconds * sample_rate))
    width += (width % 2 == 0)
    padded = np.pad(x*x, ((0, 0), (width//2, width//2)), mode='edge')
    sums = np.cumsum(np.pad(padded, ((0, 0), (1, 0))), axis=1)
    rms = np.sqrt(np.maximum((sums[:, width:] - sums[:, :-width]) / width, 0))
    return rms, rms >= threshold


def persistent_runs(active: NDArray, min_samples: int) -> NDArray:
    """Retain runs of at least min_samples; do not fill inactive gaps."""
    result = np.zeros_like(active, dtype=bool)
    edges = np.diff(np.r_[False, active, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= min_samples:
            result[start:end] = True
    return result


def factorize(activity: NDArray, sample_rate: float,
              config: DetectorConfig) -> dict:
    """Greedily minimize mismatches + fs*(state penalty*K + membership penalty*|W|).

    Enumerate all channel subsets with minimum membership, activate a subset
    wherever all its channels are ON, and remove short runs. Add the candidate
    with largest positive objective reduction. Ties use enumeration order.
    Each timeline includes all qualifying recurrences. Candidates cannot
    create false positives relative to input activity. Exhaustive subset
    enumeration is limited to 16 channels; this is a deterministic small-data
    baseline, not a global optimization or identifiable generative model.
    """
    a = np.asarray(activity, dtype=bool)
    channels, samples = a.shape
    if channels > 16:
        raise ValueError('Exhaustive version 1 supports at most 16 channels.')
    if not isinstance(config.min_channels, int) or config.min_channels < 2:
        raise ValueError('min_channels must be an integer of at least 2.')
    for value in (config.min_duration_seconds, config.state_penalty_seconds, config.membership_penalty_seconds):
        if not np.isfinite(value) or value < 0:
            raise ValueError('Durations and penalties must be finite and nonnegative.')
    candidates = []
    for size in range(config.min_channels, channels + 1):
        for group in combinations(range(channels), size):
            h = persistent_runs(np.all(a[list(group)], axis=0), max(1, int(np.ceil(config.min_duration_seconds*sample_rate))))
            if h.any():
                w = np.zeros(channels, dtype=bool)
                w[list(group)] = True
                cost = sample_rate*(config.state_penalty_seconds + config.membership_penalty_seconds*size)
                candidates.append((w, h, cost))
    reconstruction = np.zeros_like(a)
    selected = []
    objective_history = [float(a.sum())]
    while candidates:
        gains = [np.count_nonzero(a & ~reconstruction & (w[:, None] & h)) - cost
                 for w, h, cost in candidates]
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        w, h, cost = candidates.pop(best)
        reconstruction |= w[:, None] & h
        selected.append((w, h, cost))
        objective_history.append(objective_history[-1] - float(gains[best]))
    # Remove factors made redundant by later selections, if objective improves.
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            remaining = selected[:i] + selected[i+1:]
            trial = np.zeros_like(a)
            for w, h, _ in remaining:
                trial |= w[:, None] & h
            objective = float(np.count_nonzero(a != trial) + sum(cost for _, _, cost in remaining))
            if objective < objective_history[-1]:
                selected = remaining
                objective_history.append(objective)
                changed = True
                break
    w = np.column_stack([item[0] for item in selected]) if selected else np.zeros((channels, 0), dtype=bool)
    h = np.stack([item[1] for item in selected]) if selected else np.zeros((0, samples), dtype=bool)
    return dict(memberships=w, state_activations=h, reconstruction=boolean_product(w, h),
                objective_history=np.array(objective_history))


def detect(signals: NDArray, sample_rate: float,
           config: DetectorConfig = DetectorConfig()) -> dict:
    """Accept observations and fs only, plus optional ground-truth-free settings."""
    rms, activity = estimate_activity(signals, sample_rate, config.window_seconds, config.threshold)
    result = factorize(activity, sample_rate, config)
    result.update(rms=rms, channel_activity=activity)
    return result

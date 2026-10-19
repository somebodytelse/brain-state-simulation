"""Adaptive lower-quantile RMS detection using observations only."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from brain_states.state_detection import DetectorConfig, estimate_activity, factorize


@dataclass(frozen=True)
class AdaptiveConfig:
    """Fixed across noise levels; floor prevents an all-ON zero threshold."""
    percentile: float = 20.0
    scale: float = 1.5
    threshold_floor: float = 0.05


def estimate_adaptive_activity(signals: NDArray, sample_rate: float,
                               config: AdaptiveConfig = AdaptiveConfig(),
                               window_seconds: float = 0.1) -> dict[str, NDArray]:
    """Estimate each threshold from that channel's entire RMS distribution.

    Background = percentile(RMS, q); threshold = max(floor, scale*background).
    The baseline's RMS routine gives identical odd centered windows and edge
    padding. Assumes a sufficiently long low-activity population per channel.
    A quantile can reflect active signal when OFF time is scarce; overlapping
    distributions also bias it. This estimates a background RMS statistic, not
    known noise sigma. No labels, state intervals or noise parameters are inputs.
    """
    if not np.isfinite(config.percentile) or not 0 <= config.percentile <= 100:
        raise ValueError('Percentile must be finite and between 0 and 100.')
    if not np.isfinite(config.scale) or config.scale <= 1:
        raise ValueError('Background scaling factor must be finite and greater than one.')
    if not np.isfinite(config.threshold_floor) or config.threshold_floor <= 0:
        raise ValueError('Threshold floor must be finite and positive.')
    rms, _ = estimate_activity(signals, sample_rate, window_seconds, config.threshold_floor)
    background = np.percentile(rms, config.percentile, axis=1)
    thresholds = np.maximum(config.threshold_floor, config.scale*background)
    return dict(rms=rms, channel_activity=rms >= thresholds[:, None],
                background_rms=background, thresholds=thresholds)


def detect_adaptive(signals: NDArray, sample_rate: float,
                    config: AdaptiveConfig = AdaptiveConfig(),
                    factor_config: DetectorConfig = DetectorConfig()) -> dict:
    """Run adaptive activity estimation followed by the unchanged factorizer."""
    activity = estimate_adaptive_activity(signals, sample_rate, config, factor_config.window_seconds)
    result = factorize(activity['channel_activity'], sample_rate, factor_config)
    result.update(activity)
    return result

"""Run observed-only discovery, then separately load truth for evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from brain_states.config import SIMULATION, DETECTOR

from brain_states.state_detection import DetectorConfig, detect
from brain_states.evaluation import evaluate
from brain_states.visualization.detection import detection_figure, evaluation_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('output/overlapping_signals.npz'))
    parser.add_argument('--sample-rate', type=float, default=SIMULATION.sample_rate)
    parser.add_argument('--output-dir', type=Path, default=Path('output/detection_v1'))
    parser.add_argument('--window-seconds', type=float, default=DETECTOR.window_seconds)
    parser.add_argument('--threshold', type=float, default=DETECTOR.threshold)
    parser.add_argument('--min-channels', type=int, default=DETECTOR.min_channels)
    parser.add_argument('--min-duration-seconds', type=float, default=DETECTOR.min_duration_seconds)
    parser.add_argument('--state-penalty-seconds', type=float, default=DETECTOR.state_penalty_seconds)
    parser.add_argument('--membership-penalty-seconds', type=float, default=DETECTOR.membership_penalty_seconds)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    config = DetectorConfig(args.window_seconds, args.threshold, args.min_channels,
                            args.min_duration_seconds, args.state_penalty_seconds,
                            args.membership_penalty_seconds)
    # NPZ is lazy: access ONLY observations before detection has completed.
    with np.load(args.input, allow_pickle=False) as source:
        signals = source['signals'].copy()
    try:
        result = detect(signals, args.sample_rate, config)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / 'discovered_states.npz', **result)
    detection_plot = detection_figure(signals, args.sample_rate, result)
    detection_plot.savefig(args.output_dir / 'detection.png', dpi=160)
    # Ground-truth arrays are first accessed here, after detection and saving.
    with np.load(args.input, allow_pickle=False) as source:
        truth = {key: source[key].copy() for key in ('time', 'state_activations',
                 'state_memberships', 'channel_activations')}
    if not np.allclose(np.diff(truth['time']), 1/args.sample_rate, rtol=1e-7, atol=1e-10):
        parser.error('Supplied sampling rate does not match the recording time vector.')
    metrics = evaluate(result, truth)
    report = dict(input=str(args.input.resolve()), sample_rate=args.sample_rate,
                  config=asdict(config), metrics=metrics,
                  objective_history=result['objective_history'].tolist(),
                  matching='Maximum-weight one-to-one: mean temporal IoU and membership Jaccard; zero-score pairs unmatched.',
                  limitation='Exploratory baseline. Boolean reconstruction does not establish unique state recovery. RMS blurs transitions; greedy factors need not be globally optimal.')
    (args.output_dir / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    evaluation_plot = evaluation_figure(truth, result, metrics, args.sample_rate)
    evaluation_plot.savefig(args.output_dir / 'evaluation.png', dpi=160)
    print(json.dumps(metrics, indent=2))
    print(f'Saved results to {args.output_dir.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()

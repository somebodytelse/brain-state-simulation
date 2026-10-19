"""Paired adaptive-versus-fixed RMS experiment; no ground-truth threshold tuning."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np

from brain_states.config import SIMULATION, DETECTOR, ADAPTIVE, NOISE_LEVELS, NOISE_EXPERIMENT_SEED

from brain_states.activity_detection import AdaptiveConfig, detect_adaptive
from brain_states.state_detection import detect
from brain_states.simulation import simulate, add_gaussian_noise, DEFAULT_STATES
from brain_states.evaluation import evaluate
from brain_states.experiments.noise_robustness import NOISE_LEVELS, SUMMARY_METRICS, summarize, write_csv

EXAMPLE_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.75, 1.0)
BASELINE_HASH = 'e4c9fea33d5bf3a94e1c8dc0f1f97ddc14a291e630b9c1d37b9157b6b9c48093'


def detector_hash() -> str:
    """Fingerprint the untouched fixed detector and Boolean factorizer module."""
    return hashlib.sha256((Path(__file__).resolve().parents[1] / 'state_detection.py').read_bytes()).hexdigest()


def paired_detect(signals: np.ndarray, fs: float, adaptive: AdaptiveConfig) -> dict:
    """Truth-free boundary: give both detectors the same read-only observations."""
    x = signals.copy()
    x.setflags(write=False)
    digest = hashlib.sha256(x.tobytes()).hexdigest()
    fixed = detect(x, fs, DETECTOR)
    adapted = detect_adaptive(x, fs, adaptive, DETECTOR)
    np.testing.assert_array_equal(fixed['rms'], adapted['rms'])
    assert hashlib.sha256(x.tobytes()).hexdigest() == digest
    return dict(fixed=fixed, adaptive=adapted, recording_sha256=digest)


def metric_row(metrics: dict, detector: str, sigma: float, run: int, seed: int) -> dict:
    """Keep unmatched-penalized and matched-only metrics separately labelled."""
    matches = metrics['matches']
    return dict(detector=detector, sigma=sigma, run=run, seed=seed,
        activity_precision=metrics['channel_activity']['precision'],
        activity_recall=metrics['channel_activity']['recall'], activity_f1=metrics['channel_activity']['f1'],
        temporal_iou=metrics['temporal_iou_with_unmatched_zero'],
        membership_jaccard=metrics['membership_jaccard_with_unmatched_zero'],
        matched_temporal_iou=float(np.mean([m['temporal_iou'] for m in matches])) if matches else 0,
        matched_membership_jaccard=float(np.mean([m['membership_jaccard'] for m in matches])) if matches else 0,
        true_states=metrics['true_state_count'], discovered_states=metrics['discovered_state_count'],
        state_count_recovery=int(metrics['true_state_count'] == metrics['discovered_state_count']),
        reconstruction_error=metrics['boolean_reconstruction_error'],
        reconstruction_error_against_truth=metrics['reconstruction_error_against_ground_truth'],
        unmatched_true_count=len(metrics['unmatched_true_states']),
        unmatched_discovered_count=len(metrics['unmatched_discovered_states']),
        unmatched_true_states=json.dumps(metrics['unmatched_true_states']),
        unmatched_discovered_states=json.dumps(metrics['unmatched_discovered_states']),
        matches=json.dumps(matches))


def run_comparison(repeats: int = 30, base_seed: int = NOISE_EXPERIMENT_SEED,
                   adaptive: AdaptiveConfig = ADAPTIVE) -> tuple:
    """Generate identical paired data with Experiment A's original seed sequence."""
    if repeats < 2:
        raise ValueError('At least two repeats are required.')
    before = detector_hash()
    truth = simulate(**SIMULATION.kwargs())
    clean = truth['signals']
    children = np.random.SeedSequence(base_seed).spawn(len(NOISE_LEVELS)*repeats)
    rows, pairs, examples = [], [], {}
    for level_index, sigma in enumerate(NOISE_LEVELS):
        for run in range(repeats):
            seed = int(children[level_index*repeats+run].generate_state(1, dtype=np.uint64)[0])
            signals = add_gaussian_noise(clean, sigma, seed)
            results = paired_detect(signals, SIMULATION.sample_rate, adaptive)
            pair_rows = {}
            evaluated = {}
            for name in ('fixed', 'adaptive'):
                # Neither detector has access to the evaluator's truth argument.
                metrics = evaluate(results[name], truth)
                evaluated[name] = metrics
                row = metric_row(metrics, name, sigma, run+1, seed)
                row['recording_sha256'] = results['recording_sha256']
                row['thresholds'] = json.dumps(results[name].get('thresholds', np.full(8, 0.05)).tolist())
                row['background_rms'] = json.dumps(results[name]['background_rms'].tolist()) if name == 'adaptive' else ''
                pair_rows[name] = row
                rows.append(row)
            pair = dict(sigma=sigma, run=run+1, seed=seed, recording_sha256=results['recording_sha256'])
            for key in SUMMARY_METRICS:
                pair[key] = pair_rows['adaptive'][key] - pair_rows['fixed'][key]
            pairs.append(pair)
            if sigma in EXAMPLE_LEVELS and run == 0:
                examples[sigma] = dict(signals=signals, results=results, metrics=evaluated, seed=seed)
        print(f'Completed sigma={sigma:.2f}: {repeats} identical pairs', flush=True)
    assert before == detector_hash() == BASELINE_HASH
    summaries = []
    for name in ('fixed', 'adaptive'):
        for row in summarize([r for r in rows if r['detector'] == name]):
            summaries.append(dict(detector=name, **row))
    pair_summary = summarize(pairs)
    metadata = dict(base_seed=base_seed, repeats=repeats, noise_levels=NOISE_LEVELS,
        fixed=asdict(DETECTOR), adaptive=asdict(adaptive),
        adaptive_rule='background=per-channel percentile(RMS,q); threshold=max(floor,scale*background)',
        sample_rate=200, duration=12, amplitude=1, n_channels=8, states=[asdict(s) for s in DEFAULT_STATES],
        noise='Independent N(0,sigma^2) everywhere; sigma is not supplied to either detector.',
        seed_scheme='Same as Experiment A: SeedSequence(base_seed).spawn(8*repeats), uint64 per child, level-major order.',
        selection='Run 1 examples selected in advance; channels 1 and 3 for envelope/distribution analysis.',
        detector_sha256=before, detector_unchanged=True, identical_inputs_verified=True, identical_rms_verified=True,
        numpy_version=np.__version__, matplotlib_version=plt.matplotlib.__version__, python_version=platform.python_version(),
        statistics='Means and sample SD (ddof=1); paired differences = adaptive minus fixed. Error bars are not confidence intervals.',
        scores='Existing one-to-one matching; unmatched state scores contribute zero with max(true, discovered) denominator.',
        assumptions='Lower quantile requires sufficient OFF time. Floor avoids zero thresholds. No truth or known sigma used to estimate thresholds; one adaptive configuration across levels.')
    return rows, pairs, summaries, pair_summary, examples, truth, metadata


def verify_comparison(rows: list[dict], repeats: int, previous: Path | None = None) -> int:
    """Check pairing and reproduce prior fixed metrics using saved seeds if available."""
    assert len(rows) == 2*len(NOISE_LEVELS)*repeats
    fixed = {(r['sigma'], r['run'], r['seed']): r for r in rows if r['detector'] == 'fixed'}
    adaptive = {(r['sigma'], r['run'], r['seed']): r for r in rows if r['detector'] == 'adaptive'}
    assert set(fixed) == set(adaptive)
    assert len({key[2] for key in fixed}) == len(fixed)
    for sigma in NOISE_LEVELS:
        assert sum(key[0] == sigma for key in fixed) == repeats
    for key, row in fixed.items():
        assert row['recording_sha256'] == adaptive[key]['recording_sha256']
        if key[0] == 0:
            np.testing.assert_allclose([row['activity_f1'], row['temporal_iou'], row['membership_jaccard']],
                [0.9897341405633062, 0.9783083395943947, 1.0], rtol=0, atol=1e-12)
    checked = 0
    if previous is not None and previous.exists():
        with previous.open(newline='', encoding='utf-8') as handle:
            old = list(csv.DictReader(handle))
        for old_row in old:
            key = (float(old_row['sigma']), int(old_row['run']), int(old_row['seed']))
            if key in fixed:
                for metric in SUMMARY_METRICS:
                    np.testing.assert_allclose(fixed[key][metric], float(old_row[metric]), rtol=0, atol=1e-12)
                checked += 1
    return checked


def write_report(path: Path, summaries: list[dict], metadata: dict) -> None:
    """Report measured improvement and failures, without selecting settings afterward."""
    lookup = {(r['detector'], r['sigma']): r for r in summaries}
    lines = ['# Paired adaptive RMS comparison', '',
        '240 paired recordings, 480 detection/evaluation runs. The fixed detector and Boolean factorizer '
        'are unchanged. Adaptive settings were selected before evaluation and kept constant at every noise level.', '',
        f'Adaptive rule: max({metadata["adaptive"]["threshold_floor"]}, '
        f'{metadata["adaptive"]["scale"]} × per-channel {metadata["adaptive"]["percentile"]}th-percentile RMS). '
        'Both use the same centered 21-sample RMS window at 200 Hz (105 ms), with identical edge padding. '
        'The floor handles zero background in noiseless data. The method does not know noise sigma.', '',
        '| Sigma | Fixed F1 | Adaptive F1 ± SD | Fixed IoU | Adaptive IoU ± SD | Fixed Jaccard | Adaptive Jaccard ± SD | Adaptive exact count | Adaptive states ± SD |',
        '|---|---|---|---|---|---|---|---|---|']
    for sigma in NOISE_LEVELS:
        f, a = lookup['fixed', sigma], lookup['adaptive', sigma]
        def score(key):
            return f'{a[key+"_mean"]:.4f} ± {a[key+"_std"]:.4f}'
        lines.append(f'| {sigma:.2f} | {f["activity_f1_mean"]:.4f} | {score("activity_f1")} | '
            f'{f["temporal_iou_mean"]:.4f} | {score("temporal_iou")} | '
            f'{f["membership_jaccard_mean"]:.4f} | {score("membership_jaccard")} | '
            f'{a["state_count_recovery_mean"]:.1%} | {score("discovered_states")} |')
    lines += ['', '## Findings and warnings', '']
    for sigma in NOISE_LEVELS:
        f, a = lookup['fixed', sigma], lookup['adaptive', sigma]
        delta = a['activity_f1_mean'] - f['activity_f1_mean']
        state_delta = a['temporal_iou_mean'] - f['temporal_iou_mean']
        lines.append(f'- sigma={sigma:.2f}: adaptive minus fixed mean F1 {delta:+.4f}, '
                     f'temporal IoU {state_delta:+.4f}; adaptive precision '
                     f'{a["activity_precision_mean"]:.4f}, recall {a["activity_recall_mean"]:.4f}.')
        worse = [label for key, label in [('activity_f1', 'activity F1'), ('temporal_iou', 'temporal IoU'),
            ('membership_jaccard', 'membership Jaccard'), ('state_count_recovery', 'exact count recovery')]
            if a[key+'_mean'] < f[key+'_mean'] - 1e-12]
        if worse:
            lines.append(f'  **Warning:** Adaptive performs worse than fixed on {", ".join(worse)} at sigma={sigma:.2f}.')
    lines += ['', 'Compare activity precision/recall with the state scores separately. Adaptive thresholds '
        'can reject background noise while also rejecting low-amplitude portions of active oscillations. '
        'With high noise, active and inactive RMS distributions overlap, fragmenting estimated activity; '
        'the factorizer keeps runs only if they last 0.2 seconds and may lose true groups or return no states. '
        'Improved activity F1 therefore need not imply good state recovery. Reconstruction error is measured '
        'against estimated activity; ground-truth reconstruction error is a separately recorded diagnostic.', '',
        'At sigma=0.20–0.30, activity F1 remains above 0.98 but aggregate state scores drop. '
        'In the sigma=0.30 run-1 example, a separate {1,2} factor complements a fragmented {1,2,3} '
        'factor when channel 3 falls below threshold. Brief channel disagreement can therefore produce '
        'additional states even when overall activity classification is accurate. This is a sensitivity '
        'of the existing AND-timeline factorization, not evidence of a new underlying brain state.', '',
        'The lower quantile is not a guaranteed noise estimate. Channel 3 is active much of the recording, '
        'leaving a relatively small background population, and its shared-state signal may cancel. '
        'If a channel is continuously active, its lower quantile is an active-signal statistic. '
        'RMS distribution plots are labelled using truth only after both detectors finish. Histogram overlap '
        'is measured with shared bins as sum(min(active_probability, inactive_probability)); it is descriptive, '
        'not a tuning criterion. It includes RMS transition blur. Noise is only Gaussian white noise; '
        'these results are not evidence of robustness to realistic brain-recording artifacts.', '',
        'The state count is not forced. Existing one-to-one matching and unmatched penalties are preserved. '
        'Per-pair matched-only scores and missed/extra state IDs are saved alongside aggregate scores. '
        'Different latent structures may give the same binary activity; correct count or low reconstruction '
        'error alone is insufficient evidence of recovery.', '',
        '## Reproduction and verification', '',
        '`python experiment_adaptive.py --no-show`', '',
        f'Base seed {metadata["base_seed"]}; all per-run seeds and recording hashes are saved. '
        f'{metadata["previous_fixed_runs_verified"]} fixed runs matched the saved Experiment A metrics '
        'to absolute tolerance 1e-12. Both detectors received the same read-only array and returned '
        'identical RMS envelopes for every pair. state_detection.py SHA256 remained unchanged: '
        f'`{metadata["detector_sha256"]}`. Detector APIs accept observations, sampling rate and configuration '
        'only; evaluation receives truth afterward.', '',
        'CSV outputs: per_run_metrics.csv (480 rows), summary_statistics.csv (16 rows), '
        'paired_comparison.csv (240 differences), paired_summary.csv (8 mean/SD rows), '
        'rms_distribution_overlap.csv. parameters.json, seeds.csv, example_runs.npz, five PNGs and this '
        'report preserve configuration, examples and results. Standard deviations are sample SD, not '
        'confidence intervals. Examples use run 1, never selected on performance.']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-seed', type=int, default=NOISE_EXPERIMENT_SEED)
    parser.add_argument('--percentile', type=float, default=ADAPTIVE.percentile)
    parser.add_argument('--scale', type=float, default=ADAPTIVE.scale)
    parser.add_argument('--threshold-floor', type=float, default=ADAPTIVE.threshold_floor)
    parser.add_argument('--output-dir', type=Path, default=Path('output/adaptive_comparison'))
    parser.add_argument('--previous-results', type=Path, default=Path('output/noise_experiment_a/per_run_metrics.csv'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    config = AdaptiveConfig(args.percentile, args.scale, args.threshold_floor)
    rows, pairs, summaries, pair_summary, examples, truth, metadata = run_comparison(base_seed=args.base_seed, adaptive=config)
    checked = verify_comparison(rows, 30, args.previous_results)
    metadata['previous_fixed_runs_verified'] = checked
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'per_run_metrics.csv', rows)
    write_csv(out/'summary_statistics.csv', summaries)
    write_csv(out/'paired_comparison.csv', pairs)
    write_csv(out/'paired_summary.csv', pair_summary)
    write_csv(out/'seeds.csv', [{key: row[key] for key in ('sigma', 'run', 'seed', 'recording_sha256')} for row in pairs])
    (out/'parameters.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    from brain_states.visualization.adaptive import create_figures
    figures, overlap = create_figures(examples, summaries, truth)
    for name, figure in figures.items():
        figure.savefig(out/name, dpi=160)
    write_csv(out/'rms_distribution_overlap.csv', overlap)
    saved = dict(time=truth['time'], clean_signals=truth['signals'])
    for sigma, example in examples.items():
        prefix = f'sigma_{sigma:.2f}'
        saved[prefix+'_signals'] = example['signals']
        for name in ('fixed', 'adaptive'):
            for key, array in example['results'][name].items():
                saved[prefix+'_'+name+'_'+key] = array
    np.savez_compressed(out/'example_runs.npz', **saved)
    write_report(out/'report.md', summaries, metadata)
    print(f'Verified 240 pairs, {checked} previous fixed runs, unchanged factorizer. Saved {out.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()

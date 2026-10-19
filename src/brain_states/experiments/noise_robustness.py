"""Experiment A: unchanged-detector robustness to independent Gaussian noise."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np

from brain_states.config import SIMULATION, DETECTOR, NOISE_LEVELS, NOISE_EXPERIMENT_SEED

from brain_states.simulation import DEFAULT_STATES, add_gaussian_noise, simulate
from brain_states.state_detection import detect
from brain_states.evaluation import evaluate
from brain_states.visualization.detection import timelines, membership_plot

EXAMPLE_LEVELS = (0.00, 0.20, 0.50)
SUMMARY_METRICS = ('activity_precision', 'activity_recall', 'activity_f1',
                   'temporal_iou', 'membership_jaccard', 'matched_temporal_iou',
                   'matched_membership_jaccard', 'discovered_states',
                   'state_count_recovery', 'reconstruction_error',
                   'reconstruction_error_against_truth', 'unmatched_true_count',
                   'unmatched_discovered_count')


def run_trials(repeats: int = 30, base_seed: int = NOISE_EXPERIMENT_SEED) -> tuple[list[dict], dict, dict]:
    """Execute all levels without changing detector settings or using truth in detection."""
    if repeats < 2:
        raise ValueError('At least two repeats are required for sample standard deviations.')
    clean_data = simulate(**SIMULATION.kwargs())  # The underlying structure stays identical for every run.
    clean = clean_data['signals']
    config = DETECTOR
    seeds = np.random.SeedSequence(base_seed).spawn(len(NOISE_LEVELS)*repeats)
    rows, examples = [], {}
    for level_index, sigma in enumerate(NOISE_LEVELS):
        for run in range(repeats):
            seed = int(seeds[level_index*repeats+run].generate_state(1, dtype=np.uint64)[0])
            noisy = add_gaussian_noise(clean, sigma, seed)
            # Only observations and sampling frequency enter the detector.
            result = detect(noisy, SIMULATION.sample_rate, DETECTOR)
            metrics = evaluate(result, clean_data)  # Truth used only after detection.
            matched = metrics['matches']
            residual = noisy - clean
            inactive = clean_data['channel_activations'] == 0
            row = dict(sigma=sigma, run=run+1, seed=seed,
                       activity_precision=metrics['channel_activity']['precision'],
                       activity_recall=metrics['channel_activity']['recall'],
                       activity_f1=metrics['channel_activity']['f1'],
                       temporal_iou=metrics['temporal_iou_with_unmatched_zero'],
                       membership_jaccard=metrics['membership_jaccard_with_unmatched_zero'],
                       matched_temporal_iou=float(np.mean([m['temporal_iou'] for m in matched])) if matched else 0.0,
                       matched_membership_jaccard=float(np.mean([m['membership_jaccard'] for m in matched])) if matched else 0.0,
                       true_states=metrics['true_state_count'], discovered_states=metrics['discovered_state_count'],
                       state_count_recovery=int(metrics['true_state_count'] == metrics['discovered_state_count']),
                       reconstruction_error=metrics['boolean_reconstruction_error'],
                       reconstruction_error_against_truth=metrics['reconstruction_error_against_ground_truth'],
                       unmatched_true_count=len(metrics['unmatched_true_states']),
                       unmatched_discovered_count=len(metrics['unmatched_discovered_states']),
                       unmatched_true_states=json.dumps(metrics['unmatched_true_states']),
                       unmatched_discovered_states=json.dumps(metrics['unmatched_discovered_states']),
                       matches=json.dumps(matched),
                       inactive_noise_std=float(np.std(residual[inactive], ddof=1)),
                       active_noise_std=float(np.std(residual[~inactive], ddof=1)))
            rows.append(row)
            if sigma in EXAMPLE_LEVELS and run == 0:
                examples[sigma] = dict(signals=noisy, result=result, metrics=metrics, seed=seed)
        print(f'Completed sigma={sigma:.2f}: {repeats} runs', flush=True)
    metadata = dict(base_seed=base_seed, repeats=repeats, noise_levels=NOISE_LEVELS,
                    sample_rate=200, duration=12, amplitude=1, n_channels=8,
                    noise_model='Independent additive N(0,sigma^2) on all channels and samples; sigma is standard deviation, not dB.',
                    states=[asdict(s) for s in DEFAULT_STATES], detector=asdict(config),
                    numpy_version=np.__version__, matplotlib_version=plt.matplotlib.__version__,
                    python_version=platform.python_version(),
                    seed_scheme='SeedSequence(base_seed).spawn(8*repeats); one uint64 seed per child, level-major order; default_rng(seed).normal(0,sigma,(8,2400)).',
                    example_selection='First run at each specified example level, never selected by performance.',
                    standard_deviation='Sample standard deviation, ddof=1; bands describe run variability, not confidence intervals.',
                    matching='Existing maximum-weight one-to-one mean temporal IoU/membership Jaccard matching.',
                    aggregate='Sum of matched scores / max(true state count, discovered state count); unmatched contribute zero.',
                    matched_only='Mean across matched pairs; 0 if no matches. Does not replace aggregate scores.')
    return rows, examples, metadata


def summarize(rows: list[dict]) -> list[dict]:
    """Compute means and sample standard deviations separately for each level."""
    summary = []
    for sigma in NOISE_LEVELS:
        selected = [row for row in rows if row['sigma'] == sigma]
        row = dict(sigma=sigma, runs=len(selected))
        for key in SUMMARY_METRICS:
            values = np.array([item[key] for item in selected])
            row[key+'_mean'] = float(values.mean())
            row[key+'_std'] = float(values.std(ddof=1))
        summary.append(row)
    return summary


def verify_trials(rows: list[dict], repeats: int) -> None:
    """Check coverage, seed uniqueness, baseline reproduction and noise presence."""
    assert len(rows) == len(NOISE_LEVELS)*repeats
    assert len({row['seed'] for row in rows}) == len(rows)
    expected = dict(activity_f1=0.9897341405633062, temporal_iou=0.9783083395943947,
                    membership_jaccard=1.0, discovered_states=3)
    for sigma in NOISE_LEVELS:
        selected = [row for row in rows if row['sigma'] == sigma]
        assert len(selected) == repeats
        assert {row['run'] for row in selected} == set(range(1, repeats+1))
        for row in selected:
            if sigma == 0:
                for key, value in expected.items():
                    np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-12)
                assert row['active_noise_std'] == row['inactive_noise_std'] == 0
            else:
                assert row['active_noise_std'] > 0 and row['inactive_noise_std'] > 0


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save numeric values, seeds and explicit matching/unmatched information."""
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signals_figure(examples: dict):
    """Plot identical channels/time across levels with common amplitude limits."""
    fig, axes = plt.subplots(8, 3, figsize=(16, 12), sharex=True, sharey=True, layout='constrained')
    time = np.arange(2400)/200
    limit = max(np.max(np.abs(examples[s]['signals'])) for s in EXAMPLE_LEVELS)*1.05
    for col, sigma in enumerate(EXAMPLE_LEVELS):
        for channel in range(8):
            ax = axes[channel, col]
            ax.plot(time, examples[sigma]['signals'][channel], color='#294768', lw=0.55)
            ax.set_ylim(-limit, limit)
            ax.set_xlim(0, 12)
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel(f'Ch {channel+1}')
            if channel == 0:
                ax.set_title(f'sigma = {sigma:.2f} | run 1')
            if channel == 7:
                ax.set_xlabel('Time (s)')
    fig.suptitle('Example recordings — same clean signals; common y-axis limits')
    return fig


def performance_figure(summary: list[dict]):
    """Show means ± sample SD; retain errors from unmatched states."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout='constrained')
    for ax, (key, label) in zip(axes.flat, [('activity_f1', 'Channel activity F1'),
             ('temporal_iou', 'Temporal IoU (unmatched penalized)'),
             ('membership_jaccard', 'Membership Jaccard (unmatched penalized)'),
             ('state_count_recovery', 'Exact state-count recovery')]):
        ax.errorbar(NOISE_LEVELS, [row[key+'_mean'] for row in summary],
                    yerr=[row[key+'_std'] for row in summary], fmt='o-', capsize=4, color='#294768')
        ax.set_title(label)
        ax.set_ylim(-0.08, 1.08)
        ax.grid(alpha=0.2)
        ax.set_xlabel('Gaussian noise standard deviation sigma')
    fig.suptitle('Unchanged detector — mean ± sample SD across 30 runs')
    return fig


def states_figure(examples: dict, truth: dict):
    """Compare three fixed examples; explicitly identify unmatched/extra states."""
    fig = plt.figure(figsize=(18, 13), layout='constrained')
    grid = fig.add_gridspec(3, 3, height_ratios=[1.5, 2, 3])
    first = None
    for col, sigma in enumerate(EXAMPLE_LEVELS):
        item = examples[sigma]
        metrics, result = item['metrics'], item['result']
        matches = sorted(metrics['matches'], key=lambda m: m['true_state'])
        order = [m['discovered_state']-1 for m in matches] + [i-1 for i in metrics['unmatched_discovered_states']]
        labels = [f"D{m['discovered_state']} -> T{m['true_state']}" for m in matches]
        labels += [f'D{i} extra' for i in metrics['unmatched_discovered_states']]
        ax = fig.add_subplot(grid[0, col], sharex=first)
        if first is None:
            first = ax
        timelines(ax, truth['state_activations'], 200, ['T1 (A)', 'T2 (B)', 'T3 (C)'])
        ax.set_title(f'sigma={sigma:.2f} | true activity\nUnmatched true: {metrics["unmatched_true_states"]}')
        ax.tick_params(labelbottom=False)
        discovered = fig.add_subplot(grid[1, col], sharex=first)
        timelines(discovered, result['state_activations'][order], 200, labels)
        discovered.set_title(f'Discovered activity | extra: {metrics["unmatched_discovered_states"]}')
        discovered.set_xlabel('Time (s)')
        subgrid = grid[2, col].subgridspec(1, 2)
        true_ax, found_ax = fig.add_subplot(subgrid[0]), fig.add_subplot(subgrid[1])
        membership_plot(true_ax, truth['state_memberships'], ['T1', 'T2', 'T3'])
        membership_plot(found_ax, result['memberships'][:, order], labels)
        true_ax.set_title('True memberships')
        found_ax.set_title('Discovered memberships')
        found_ax.tick_params(axis='x', labelrotation=60)
    first.set_xlim(0, 12)
    fig.suptitle('State recovery — run 1 at each level, label matching performed after detection')
    return fig


def write_report(path: Path, summary: list[dict], metadata: dict) -> None:
    """Report actual performance without tuning or equating reconstruction with recovery."""
    first_noisy = summary[1]
    lines = ['# Experiment A: Gaussian white noise robustness', '',
             'Eight noise standard deviations, 30 independently seeded runs per level. '
             'Clean amplitude 1, 8 channels, 200 Hz, 12 seconds. Underlying states are fixed. '
             'The original detector and Boolean algorithm are unchanged; no ground-truth tuning.', '',
             '| Sigma | Activity F1 mean ± SD | Temporal IoU mean ± SD | Membership Jaccard mean ± SD | Count recovery | States mean ± SD | Reconstruction error |',
             '|---|---|---|---|---|---|---|']
    for row in summary:
        def score(key):
            return f'{row[key+"_mean"]:.4f} ± {row[key+"_std"]:.4f}'
        lines.append(f'| {row["sigma"]:.2f} | {score("activity_f1")} | {score("temporal_iou")} | {score("membership_jaccard")} | '
                     f'{row["state_count_recovery_mean"]:.1%} | {score("discovered_states")} | {row["reconstruction_error_mean"]:.4f} |')
    lines += ['', 'Temporal IoU and membership Jaccard penalize unmatched states using the existing '
              'sum/max(true count, discovered count) rule. Matched-only scores, unmatched IDs, per-pair '
              'scores and all seeds are separately recorded in per_run_metrics.csv. Summary deviations '
              'are sample SD (ddof=1), not confidence intervals. Count recovery is merely the fraction '
              'of runs with the correct count; it does not establish correct groups.', '',
              '## Interpretation', '',
              'Deterioration starts at sigma=0.05 on this tested grid, where noise magnitude reaches '
              'the fixed RMS threshold (0.05). The method initially mistakes inactive noisy channels '
              'for active ones, reducing activity precision. At sigma>=0.10, estimated activity becomes '
              'almost entirely ON. The complexity penalty then favors an all-channel state rather '
              'than recovering independent groups. The group recovery failure follows activity-estimation '
              'failure; the factorizer can perfectly reconstruct the wrong activity matrix.', '',
              f'At sigma=0.05, the matched-only membership Jaccard is '
              f'{first_noisy["matched_membership_jaccard_mean"]:.4f} and matched temporal IoU '
              f'is {first_noisy["matched_temporal_iou_mean"]:.4f}, but the mean discovered count '
              f'is {first_noisy["discovered_states_mean"]:.2f}, including '
              f'{first_noisy["unmatched_discovered_count_mean"]:.2f} extra states. '
              'Thus the lower aggregate scores at this level primarily reflect excess factors; '
              'reporting matched-only scores would hide the failure.', '',
              'The noiseless baseline was reproduced to 1e-12 in all 30 runs. High-noise results must '
              'not be interpreted as robust state recovery even if reconstruction error is zero. '
              'No threshold or factorization parameters were altered after observing results. '
              'We did not independently benchmark factorization under an oracle activity matrix in '
              'this experiment, so these results do not isolate its general robustness to activity errors.', '',
              'Gaussian white noise is a first approximation, not a complete model of brain-recording '
              'noise. Results apply to this fixed synthetic structure and parameter configuration; '
              'the coarse noise grid does not establish an exact failure threshold. Boolean state '
              'representations remain potentially non-identifiable.', '',
              '## Reproduction', '',
              '`python experiment_noise.py --no-show`', '',
              f'Base seed: {metadata["base_seed"]}. Every per-run seed appears in the CSV. '
              'parameters.json includes detector settings, generator structure, environment versions, '
              'seed derivation and metric definitions. Example plots use run 1, never a best-performing run.', '',
              'Outputs: per_run_metrics.csv, summary_statistics.csv, parameters.json, report.md, '
              'example_signals.png, performance_vs_noise.png, state_examples.png, example_runs.npz.', '',
              'Noise-presence checks cover active and inactive periods; all 240 runs and baseline '
              'values are checked by the runner before saving summaries.']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-seed', type=int, default=NOISE_EXPERIMENT_SEED)
    parser.add_argument('--output-dir', type=Path, default=Path('output/noise_experiment_a'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    rows, examples, metadata = run_trials(base_seed=args.base_seed)
    verify_trials(rows, 30)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'per_run_metrics.csv', rows)
    write_csv(args.output_dir/'summary_statistics.csv', summary)
    (args.output_dir/'parameters.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    truth = simulate(**SIMULATION.kwargs())
    figures = [signals_figure(examples), performance_figure(summary), states_figure(examples, truth)]
    for figure, name in zip(figures, ['example_signals.png', 'performance_vs_noise.png', 'state_examples.png']):
        figure.savefig(args.output_dir/name, dpi=160)
    saved = {'time': truth['time'], 'clean_signals': truth['signals']}
    for sigma, example in examples.items():
        prefix = f'sigma_{sigma:.2f}'
        saved[prefix+'_signals'] = example['signals']
        for key, value in example['result'].items():
            saved[prefix+'_'+key] = value
    np.savez_compressed(args.output_dir/'example_runs.npz', **saved)
    write_report(args.output_dir/'report.md', summary, metadata)
    print(f'Verified 240 trials and saved results to {args.output_dir.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()

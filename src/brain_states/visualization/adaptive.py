"""Post-detection adaptive comparison figures; truth used only for analysis."""
import numpy as np
import matplotlib.pyplot as plt
from brain_states.experiments.noise_robustness import NOISE_LEVELS
from brain_states.visualization.detection import timelines, membership_plot

COLORS = {'fixed': '#a54935', 'adaptive': '#246f99'}
CHANNELS = (0, 2)
ENVELOPE_LEVELS = (0.0, 0.1, 0.3, 0.5)
DISTRIBUTION_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.75, 1.0)


def envelopes_figure(examples: dict, truth: dict):
    """Signal/RMS/truth triplets with truth on separate reference axes."""
    fig, axes = plt.subplots(6, 4, figsize=(19, 14), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [2, 2, 0.6, 2, 2, 0.6]})
    time = truth['time']
    for col, sigma in enumerate(ENVELOPE_LEVELS):
        item = examples[sigma]
        for ci, channel in enumerate(CHANNELS):
            row = ci*3
            axes[row, col].plot(time, item['signals'][channel], color='#596579', lw=0.6)
            axes[row, col].set_title(f'sigma={sigma:.2f}, channel {channel+1}')
            axes[row, col].set_ylabel('Signal')
            ax = axes[row+1, col]
            rms = item['results']['fixed']['rms'][channel]
            threshold = item['results']['adaptive']['thresholds'][channel]
            ax.plot(time, rms, color='#344556', lw=0.8, label='RMS')
            ax.axhline(0.05, color=COLORS['fixed'], ls='--', label='Fixed threshold')
            ax.axhline(threshold, color=COLORS['adaptive'], ls=':', lw=2,
                       label=f'Adaptive {threshold:.3f}')
            ax.set_ylabel('RMS amplitude')
            ax.legend(fontsize=8, loc='upper right')
            reference = axes[row+2, col]
            reference.step(time, truth['channel_activations'][channel], where='post', color='#26784b')
            reference.set_yticks([0, 1])
            reference.set_ylim(-0.2, 1.2)
            reference.set_ylabel('Truth')
            for a in axes[row:row+3, col]:
                a.set_xlim(0, 12)
                a.grid(alpha=0.15)
        axes[-1, col].set_xlabel('Time (s)')
    fig.suptitle('Run 1 envelopes and thresholds — truth shown separately, never used for estimation')
    return fig


def metric_curves(ax, summaries, metric, title):
    """Plot both methods with sample standard deviation error bars."""
    for name in ('fixed', 'adaptive'):
        selected = [r for r in summaries if r['detector'] == name]
        ax.errorbar(NOISE_LEVELS, [r[metric+'_mean'] for r in selected],
                    yerr=[r[metric+'_std'] for r in selected], fmt='o-', capsize=3,
                    label=name.capitalize(), color=COLORS[name])
    ax.set_title(title)
    ax.set_xlabel('Noise standard deviation sigma')
    ax.grid(alpha=0.2)
    ax.legend()
    if metric != 'discovered_states':
        ax.set_ylim(-0.08, 1.08)
    else:
        ax.set_ylim(bottom=-0.1)


def activity_figure(summaries):
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    metric_curves(ax, summaries, 'activity_f1', 'Activity detection — mean ± sample SD over 30 paired runs')
    ax.set_ylabel('Channel activity F1')
    return fig


def discovery_figure(summaries):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, metric, title in zip(axes.flat,
        ('temporal_iou', 'membership_jaccard', 'state_count_recovery', 'discovered_states'),
        ('Temporal IoU (unmatched penalized)', 'Membership Jaccard (unmatched penalized)',
         'Exact state-count recovery', 'Number of discovered states')):
        metric_curves(ax, summaries, metric, title)
    fig.suptitle('State discovery — unchanged factorizer, mean ± sample SD over 30 pairs')
    return fig


def ordered_states(item, method):
    """Return all discoveries in matching order, retaining extra states."""
    result, metrics = item['results'][method], item['metrics'][method]
    matches = sorted(metrics['matches'], key=lambda m: m['true_state'])
    order = [m['discovered_state']-1 for m in matches] + [i-1 for i in metrics['unmatched_discovered_states']]
    labels = [f"D{m['discovered_state']} -> T{m['true_state']}" for m in matches]
    labels += [f'D{i} extra' for i in metrics['unmatched_discovered_states']]
    return result['state_activations'][order], result['memberships'][:, order], labels


def recovery_figure(examples, truth):
    fig = plt.figure(figsize=(18, 15), layout='constrained')
    grid = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.7, 1.7, 2.5])
    first = None
    for col, sigma in enumerate((0.1, 0.3)):
        item = examples[sigma]
        ax = fig.add_subplot(grid[0, col], sharex=first)
        if first is None:
            first = ax
        timelines(ax, truth['state_activations'], 200, ['T1 (A)', 'T2 (B)', 'T3 (C)'])
        ax.set_title(f'sigma={sigma:.2f} | ground-truth states')
        ax.tick_params(labelbottom=False)
        matrices = []
        for row, method in enumerate(('fixed', 'adaptive'), start=1):
            h, w, labels = ordered_states(item, method)
            metrics = item['metrics'][method]
            timelines_ax = fig.add_subplot(grid[row, col], sharex=first)
            timelines(timelines_ax, h, 200, labels)
            timelines_ax.set_title(f'{method.capitalize()} | missed: {metrics["unmatched_true_states"]} | extra: {metrics["unmatched_discovered_states"]}')
            timelines_ax.tick_params(labelbottom=row == 2)
            if row == 2:
                timelines_ax.set_xlabel('Time (s)')
            matrices.append((w, labels))
        matrix_grid = grid[3, col].subgridspec(1, 3)
        for j, (w, labels, title) in enumerate([
                (truth['state_memberships'], ['T1', 'T2', 'T3'], 'True'),
                (*matrices[0], 'Fixed'), (*matrices[1], 'Adaptive')]):
            ax = fig.add_subplot(matrix_grid[j])
            membership_plot(ax, w, labels)
            ax.tick_params(axis='x', labelrotation=60)
            ax.set_title(title+' memberships')
    first.set_xlim(0, 12)
    fig.suptitle('Example state recovery — paired run 1; no state-count constraint')
    return fig


def distributions_figure(examples, truth):
    """Truth-labelled distributions with common bins per comparison, after detection."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), layout='constrained')
    rows = []
    for ci, channel in enumerate(CHANNELS):
        active = truth['channel_activations'][channel].astype(bool)
        for col, sigma in enumerate(DISTRIBUTION_LEVELS):
            item = examples[sigma]
            values = item['results']['fixed']['rms'][channel]
            bins = np.linspace(0, max(float(values.max())*1.001, 0.1), 51)
            on_counts, _ = np.histogram(values[active], bins)
            off_counts, _ = np.histogram(values[~active], bins)
            overlap = float(np.minimum(on_counts/on_counts.sum(), off_counts/off_counts.sum()).sum())
            ax = axes[ci, col]
            ax.hist(values[~active], bins=bins, density=True, histtype='step', color='#687d99', label='True OFF')
            ax.hist(values[active], bins=bins, density=True, histtype='step', color='#26784b', label='True ON')
            ax.axvline(0.05, color=COLORS['fixed'], ls='--', label='Fixed')
            adaptive = item['results']['adaptive']['thresholds'][channel]
            ax.axvline(adaptive, color=COLORS['adaptive'], ls=':', label='Adaptive')
            ax.set_title(f'Ch {channel+1}, sigma={sigma:.2f}\nHistogram overlap={overlap:.3f}')
            ax.set_xlabel('RMS amplitude')
            ax.set_ylabel('Density')
            ax.grid(alpha=0.15)
            if col == 0:
                ax.legend(fontsize=8)
            rows.append(dict(sigma=sigma, run=1, seed=item['seed'], channel=channel+1,
                             histogram_overlap=overlap, bins=50,
                             true_active_samples=int(active.sum()), true_inactive_samples=int((~active).sum()),
                             fixed_threshold=0.05, adaptive_threshold=float(adaptive)))
    fig.suptitle('Post-detection RMS distribution analysis — truth labels used only for this comparison')
    return fig, rows


def create_figures(examples, summaries, truth):
    distribution, overlap = distributions_figure(examples, truth)
    figures = {'rms_envelopes.png': envelopes_figure(examples, truth),
               'activity_comparison.png': activity_figure(summaries),
               'state_discovery_comparison.png': discovery_figure(summaries),
               'state_recovery_examples.png': recovery_figure(examples, truth),
               'rms_distributions.png': distribution}
    return figures, overlap

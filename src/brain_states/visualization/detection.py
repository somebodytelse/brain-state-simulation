"""Figures for observation-only detection and subsequent ground-truth evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BINARY = ListedColormap(['#f1f3f6', '#294768'])


def timelines(ax, h, fs, labels):
    """Plot each state independently, retaining simultaneous activations."""
    if not len(h):
        ax.text(0.5, 0.5, 'No discovered states', transform=ax.transAxes, ha='center')
    for k, row in enumerate(h):
        edges = np.diff(np.r_[False, row, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        ax.broken_barh([(a/fs, (b-a)/fs) for a, b in zip(starts, ends)],
                       (k-0.3, 0.6), facecolors=plt.get_cmap('tab10')(k % 10))
    ax.set_yticks(range(len(h)), labels)
    ax.set_ylim(max(len(h)-0.5, 0.5), -0.5)
    ax.grid(axis='x', alpha=0.15)


def membership_plot(ax, w, labels):
    """Show membership columns, including explicit zero-state behavior."""
    if w.shape[1]:
        ax.imshow(w, cmap=BINARY, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    else:
        ax.text(0.5, 0.5, 'No discovered memberships', transform=ax.transAxes, ha='center')
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(w.shape[0]), [f'Ch {i+1}' for i in range(w.shape[0])])
    ax.set_xlabel('State')


def detection_figure(signals, fs, result):
    """Plot signals, activity, state timelines and memberships without truth."""
    channels, samples = signals.shape
    fig = plt.figure(figsize=(14, 12), layout='constrained')
    grid = fig.add_gridspec(4, 1, height_ratios=[4, 2, 1.8, 2])
    ax = fig.add_subplot(grid[0])
    time = np.arange(samples)/fs
    spacing = max(1.0, float(np.max(np.abs(signals)))) * 2.5
    for c in range(channels):
        ax.plot(time, signals[c] + (channels-1-c)*spacing, lw=0.65, color='#294768')
    ax.set_yticks(np.arange(channels)*spacing, [f'Ch {i}' for i in range(channels, 0, -1)])
    ax.set_title('1. Observed signals', loc='left')
    ax.grid(alpha=0.15)
    ax.tick_params(labelbottom=False)
    activity_ax = fig.add_subplot(grid[1], sharex=ax)
    activity_ax.imshow(result['channel_activity'], cmap=BINARY, vmin=0, vmax=1,
                       extent=(0, samples/fs, channels+0.5, 0.5), aspect='auto', interpolation='nearest')
    activity_ax.set_yticks(range(1, channels+1))
    activity_ax.set_ylabel('Channel')
    activity_ax.set_title('2. RMS activity estimate — light OFF, dark ON', loc='left')
    activity_ax.tick_params(labelbottom=False)
    state_ax = fig.add_subplot(grid[2], sharex=ax)
    labels = [f'D{i+1}' for i in range(len(result['state_activations']))]
    timelines(state_ax, result['state_activations'], fs, labels)
    state_ax.set_title('3. Discovered state activation', loc='left')
    state_ax.set_xlabel('Time (s)')
    ax.set_xlim(0, samples/fs)
    matrix_ax = fig.add_subplot(grid[3])
    membership_plot(matrix_ax, result['memberships'], labels)
    matrix_ax.set_title('4. Discovered memberships — dark = member', loc='left')
    return fig


def evaluation_figure(truth, result, metrics, fs):
    """Order discoveries by optimal matches, retaining unmatched states."""
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 2])
    true_h = truth['state_activations']
    matches = sorted(metrics['matches'], key=lambda m: m['true_state'])
    order = [m['discovered_state']-1 for m in matches] + [i-1 for i in metrics['unmatched_discovered_states']]
    labels = [f"D{m['discovered_state']} ↔ True {m['true_state']}" for m in matches]
    labels += [f'D{i} (unmatched)' for i in metrics['unmatched_discovered_states']]
    top = fig.add_subplot(grid[0, :])
    timelines(top, true_h, fs, [f'True {i+1}' for i in range(len(true_h))])
    top.set_title('Ground-truth activation', loc='left')
    top.tick_params(labelbottom=False)
    bottom = fig.add_subplot(grid[1, :], sharex=top)
    timelines(bottom, result['state_activations'][order], fs, labels)
    bottom.set_title('Discovered activation — matched labels; unmatched states retained', loc='left')
    bottom.set_xlabel('Time (s)')
    top.set_xlim(0, true_h.shape[1]/fs)
    left, right = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    membership_plot(left, truth['state_memberships'], [f'True {i+1}' for i in range(len(true_h))])
    membership_plot(right, result['memberships'][:, order], labels)
    left.set_title('True memberships')
    right.set_title('Discovered memberships')
    fig.suptitle('Post-detection evaluation | unmatched true: ' + str(metrics['unmatched_true_states'])
                 + ' | unmatched discovered: ' + str(metrics['unmatched_discovered_states']))
    return fig

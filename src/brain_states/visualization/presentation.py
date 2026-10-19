"""Reusable presentation layouts; no simulation or inference is run here."""
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

COLORS = ('#0072B2', '#D55E00', '#009E73')


INK = '#283442'


STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#aeb6be', 'axes.linewidth': 0.6,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'path.simplify': False,
}


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timeline(ax, activations, fs, labels, colors):
    """Draw exact saved binary run intervals without rounding transition times."""
    for row, (active, label, color) in enumerate(zip(activations, labels, colors)):
        changes = np.diff(np.r_[False, active, False].astype(int))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        ax.broken_barh([(a/fs, (b-a)/fs) for a,b in zip(starts,ends)],
                       (row-0.25,0.5), facecolors=color, edgecolors='none')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-0.5,-0.5)
    ax.set_xlim(0,12)
    ax.set_xticks(np.arange(0,13,1))
    ax.grid(axis='x', color='#dfe3e8', lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, pad=12)
    for spine in ('top','left','right'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#c8cfd6')


def synthetic_figure(data, fs):
    """Show eight saved signals and independent state timelines in a 16:9 canvas."""
    fig = plt.figure(figsize=(16,9))
    grid = fig.add_gridspec(10,1, left=0.115, right=0.975, bottom=0.085,
                           top=0.875, hspace=0.26, height_ratios=[1]*8+[0.28,2.15])
    fig.suptitle('Synthetic Multichannel Signals with Overlapping States',
                 fontsize=21, fontweight='semibold', x=0.5, y=0.96)
    axes = []
    for channel, wave in enumerate(data['signals']):
        ax = fig.add_subplot(grid[channel], sharex=axes[0] if axes else None)
        axes.append(ax)
        membership = np.flatnonzero(data['state_memberships'][channel])
        color = COLORS[membership[0]] if len(membership) == 1 else INK
        ax.plot(data['time'], wave, lw=0.8, color=color)
        # A common amplitude scale preserves the observed overlap increase.
        ax.set_ylim(-2.15,2.15)
        label = f'Ch {channel+1}'
        if len(membership) > 1:
            label += ' ('+' + '.join(chr(65+int(k)) for k in membership)+')'
        ax.set_yticks([0], [label])
        ax.tick_params(axis='both', length=0, labelbottom=False)
        ax.tick_params(axis='y', pad=12)
        ax.grid(axis='x', color='#dfe3e8', lw=0.6, alpha=0.8)
        ax.axhline(0, color='#e9edf1', lw=0.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    state_ax = fig.add_subplot(grid[9], sharex=axes[0])
    timeline(state_ax, data['state_activations'], fs, ['State A','State B','State C'], COLORS)
    state_ax.set_xlabel('Time (s)', labelpad=9)
    state_ax.set_title('Ground-truth state activation', loc='left', fontsize=13, pad=12)
    return fig


def membership_matrix(ax, memberships, labels, colors):
    """Use vector cells and state-specific colors; common OFF fill and cell geometry."""
    n_channels,n_states = memberships.shape
    for channel in range(n_channels):
        for state in range(n_states):
            ax.add_patch(Rectangle((state-0.5,channel-0.5),1,1,
                         facecolor=colors[state] if memberships[channel,state] else '#f0f3f6',
                         edgecolor='white',linewidth=1.5))
    ax.set_xlim(-0.5,n_states-0.5)
    ax.set_ylim(n_channels-0.5,-0.5)
    ax.set_xticks(range(n_states),labels)
    ax.set_yticks(range(n_channels),[f'Ch {i+1}' for i in range(n_channels)])
    ax.tick_params(length=0,pad=8)
    for tick,color in zip(ax.get_xticklabels(),colors):
        tick.set_color(color)
        tick.set_fontweight('semibold')
    for spine in ax.spines.values():
        spine.set_visible(False)


def detection_figure(truth,detected,metrics,fs):
    """Reorder existing detections using post-detection one-to-one matches only."""
    matches = sorted(metrics['matches'],key=lambda match:match['true_state'])
    if metrics['unmatched_true_states'] or metrics['unmatched_discovered_states']:
        raise ValueError('Saved results contain unmatched states; revise the figure layout to show them explicitly.')
    order = [m['discovered_state']-1 for m in matches]
    # Indexing reorders saved arrays; nothing is rebuilt from ground truth.
    aligned_h = detected['state_activations'][order]
    aligned_w = detected['memberships'][:,order]
    names = [chr(64+m['true_state']) for m in matches]
    detected_labels = [f'Detected {name} (D{m["discovered_state"]})' for name,m in zip(names,matches)]
    fig = plt.figure(figsize=(16,9))
    fig.suptitle('Unsupervised Recovery of Overlapping States',
                 fontsize=21,fontweight='semibold',y=0.96)
    grid = fig.add_gridspec(3,2,left=0.13,right=0.965,bottom=0.085,top=0.87,
                           height_ratios=[1.05,1.05,2.25],hspace=0.65,wspace=0.38)
    true_ax = fig.add_subplot(grid[0,:])
    detected_ax = fig.add_subplot(grid[1,:],sharex=true_ax)
    timeline(true_ax,truth['state_activations'],fs,[f'State {name}' for name in names],COLORS)
    timeline(detected_ax,aligned_h,fs,detected_labels,COLORS)
    true_ax.set_title('A   Ground-truth state activation',loc='left',fontweight='semibold',pad=10)
    detected_ax.set_title('B   Detected state activation',loc='left',fontweight='semibold',pad=10)
    true_ax.tick_params(labelbottom=False)
    detected_ax.set_xlabel('Time (s)',labelpad=7)
    left,right = fig.add_subplot(grid[2,0]),fig.add_subplot(grid[2,1])
    membership_matrix(left,truth['state_memberships'],[f'State {name}' for name in names],COLORS)
    membership_matrix(right,aligned_w,[f'{name} (D{m["discovered_state"]})' for name,m in zip(names,matches)],COLORS)
    left.set_title('C   Ground-truth channel membership',loc='left',fontweight='semibold',pad=12)
    right.set_title('D   Detected channel membership',loc='left',fontweight='semibold',pad=12)
    fig.text(0.965,0.02,'Filled cells indicate state membership',ha='right',fontsize=10,color='#647080')
    return fig


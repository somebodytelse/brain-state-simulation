"""Connectivity visualizations with common [0,1] scales and separate references."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from brain_states.visualization.detection import timelines

SNAPSHOT_TIMES = (0.5, 1.5, 3.0, 6.5, 9.5)
PAIRS = ((0, 1), (3, 4), (2, 6), (0, 3))


def matrix_heatmap(ax, matrix, title, annotate=True):
    """Show weighted connectivity with fixed scale, labelled axes and cell values."""
    n = matrix.shape[0]
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    ax.set_xticks(range(n), range(1, n+1))
    ax.set_yticks(range(n), range(1, n+1))
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    ax.set_title(title)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                        color='black' if matrix[i,j] > 0.6 else 'white', fontsize=8)
    return image


def static_figure(result, refs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    image = matrix_heatmap(axes[0], result['static_adjacency'], 'Estimated full-recording Jaccard')
    matrix_heatmap(axes[1], refs['static_reference'], 'Reference: any shared state (binary)')
    fig.colorbar(image, ax=axes, label='Connectivity [0,1]', shrink=0.8)
    fig.suptitle('Static functional coactivity and state-membership reference — different definitions')
    return fig


def dynamic_figure(result, refs):
    fig, axes = plt.subplots(2, 5, figsize=(21, 9), layout='constrained')
    for col, requested in enumerate(SNAPSHOT_TIMES):
        index = int(np.argmin(np.abs(result['window_centers']-requested)))
        center = result['window_centers'][index]
        image = matrix_heatmap(axes[0, col], result['dynamic_adjacency'][index], f'Estimated | center {center:.1f} s')
        matrix_heatmap(axes[1, col], refs['center_reference'][index], f'Shared active state at {center:.1f} s')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Connectivity [0,1]', shrink=0.65)
    fig.suptitle('1-second window estimates versus instantaneous reference — common scale throughout')
    return fig


def pairwise_figure(result, refs, states, fs):
    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.3]})
    descriptions = ('same state A', 'same state B', 'shared membership via C', 'independent states A and B')
    for ax, (i,j), label in zip(axes[:4], PAIRS, descriptions):
        ax.plot(result['window_centers'], result['dynamic_adjacency'][:, i,j],
                color='#294768', lw=1.6, label='Estimated window Jaccard')
        ax.step(result['window_centers'], refs['center_reference'][:, i,j], where='mid',
                color='#a95f37', ls='--', label='Shared-active-state reference at center')
        ax.set_title(f'Channels {i+1}–{j+1}: {label}', loc='left')
        ax.set_ylabel('Connectivity')
        ax.set_ylim(-0.06, 1.08)
        ax.grid(alpha=0.15)
    axes[0].legend(loc='lower right', fontsize=9)
    timelines(axes[4], states, fs, ['State A', 'State B', 'State C'])
    axes[4].set_title('Ground-truth activation — interpretation only', loc='left')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_xlim(0, states.shape[1]/fs)
    fig.suptitle('Connectivity follows coactivity; independent simultaneous states can appear fully connected')
    return fig


def save_animation(path: Path, result: dict, fps: int = 8) -> None:
    """Animate uniform sliding windows using Matplotlib's existing Pillow writer."""
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    image = matrix_heatmap(ax, result['dynamic_adjacency'][0], '', annotate=False)
    fig.colorbar(image, ax=ax, label='Jaccard connectivity')
    def update(index):
        image.set_data(result['dynamic_adjacency'][index])
        ax.set_title(f'Window center {result["window_centers"][index]:.1f} s')
        return image,
    animation = FuncAnimation(fig, update, frames=len(result['window_centers']), interval=1000/fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)

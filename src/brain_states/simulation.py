"""Generate multichannel observations and overlapping truth, with optional noise."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class State:
    """State membership uses one-based channel numbers; intervals are [start, end)."""
    name: str
    channels: tuple[int, ...]
    intervals: tuple[tuple[float, float], ...]
    frequency: float = 3.0
    phase: float = 0.0


DEFAULT_STATES = (
    State('A', (1, 2, 3), ((1, 4), (8, 10)), 2.0),
    State('B', (4, 5, 6), ((2, 6), (9, 12)), 4.0),
    State('C', (3, 7, 8), ((5, 9),), 3.0),
)


def create_state_activations(time: NDArray, states: tuple[State, ...]) -> NDArray:
    """Create binary state timelines, retaining independent overlapping intervals."""
    activations = np.zeros((len(states), time.size), dtype=np.uint8)
    for k, state in enumerate(states):
        for start, end in state.intervals:
            activations[k, (time >= start) & (time < end)] = 1
    return activations


def compute_channel_activity(memberships: NDArray, activations: NDArray) -> NDArray:
    """Return 1 wherever any member state is active, even at sine zero crossings."""
    return (memberships.astype(np.int64) @ activations.astype(np.int64) > 0).astype(np.uint8)


def generate_signals(time: NDArray, memberships: NDArray, activations: NDArray,
                     states: tuple[State, ...], amplitude: float = 1.0) -> NDArray:
    """Sum gated state sine waves using global recording time and configured phases."""
    frequencies = np.array([s.frequency for s in states])[:, None]
    phases = np.array([s.phase for s in states])[:, None]
    contributions = amplitude * activations * np.sin(2 * np.pi * frequencies * time + phases)
    return memberships @ contributions


def add_gaussian_noise(signals: NDArray, noise_std: float = 0.0,
                       seed: int | None = 42) -> NDArray:
    """Add independent Gaussian samples everywhere without altering clean input."""
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError('Noise standard deviation must be finite and nonnegative.')
    if noise_std == 0:
        return signals.copy()
    return signals + np.random.default_rng(seed).normal(0, noise_std, signals.shape)


def simulate(sample_rate: float = 200.0, duration: float = 12.0,
             amplitude: float = 1.0, states: tuple[State, ...] = DEFAULT_STATES,
             n_channels: int = 8, noise_std: float = 0.0,
             seed: int | None = 42) -> dict[str, NDArray]:
    """Build observations and truth on [0, duration); preserve clean default output.

    Positive noise_std adds noise everywhere and stores clean_signals separately.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('Sample rate must be finite and positive.')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Duration must be finite and positive.')
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError('Amplitude must be finite and nonnegative.')
    if n_channels < 1 or not states:
        raise ValueError('At least one channel and state are required.')
    count = duration * sample_rate
    if not np.isclose(count, round(count), rtol=0, atol=1e-9):
        raise ValueError('Duration must contain a whole number of samples.')
    memberships = np.zeros((n_channels, len(states)), dtype=np.uint8)
    for k, state in enumerate(states):
        if (not np.isfinite(state.frequency) or state.frequency <= 0
                or sample_rate <= 2 * state.frequency or not np.isfinite(state.phase)):
            raise ValueError('State frequencies must be positive and below Nyquist; phases must be finite.')
        for channel in state.channels:
            if not isinstance(channel, int) or not 1 <= channel <= n_channels:
                raise ValueError('State channel number is outside the recording.')
            memberships[channel - 1, k] = 1
        for start, end in state.intervals:
            if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= duration):
                raise ValueError('State intervals must lie within the recording.')
    time = np.arange(round(count)) / sample_rate
    activations = create_state_activations(time, states)
    clean = generate_signals(time, memberships, activations, states, amplitude)
    data = dict(time=time,
                signals=add_gaussian_noise(clean, noise_std, seed),
                state_activations=activations,
                channel_activations=compute_channel_activity(memberships, activations),
                state_memberships=memberships)
    if noise_std > 0:
        data['clean_signals'] = clean
    return data


def visualize(data: dict[str, NDArray], duration: float,
              states: tuple[State, ...] = DEFAULT_STATES) -> Figure:
    """Plot channel traces, state timelines, and binary activity on aligned axes."""
    time, signals = data['time'], data['signals']
    n_channels = signals.shape[0]
    fig = plt.figure(figsize=(14, 13), layout='constrained')
    grid = fig.add_gridspec(n_channels + 2, 1, height_ratios=[1] * n_channels + [1.8, 2.8])
    axes = []
    limit = max(1.0, float(np.max(np.abs(signals)))) * 1.12
    for c in range(n_channels):
        ax = fig.add_subplot(grid[c], sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(time, signals[c], color='#253f64', lw=0.9)
        ax.set_ylabel(f'Ch {c + 1}')
        ax.set_ylim(-limit, limit)
        ax.grid(alpha=0.15)
        ax.tick_params(labelbottom=False)
    axes[0].set_title('1. Multichannel signals — summed state contributions', loc='left')
    state_ax = fig.add_subplot(grid[n_channels], sharex=axes[0])
    edges = np.append(time, duration)
    colors = plt.get_cmap('tab10').colors
    for k, state in enumerate(states):
        active = data['state_activations'][k].astype(bool)
        changes = np.diff(np.r_[False, active, False].astype(int))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        state_ax.broken_barh([(edges[a], edges[b] - edges[a]) for a, b in zip(starts, ends)],
                            (k - 0.3, 0.6), facecolors=colors[k % len(colors)])
    state_ax.set_yticks(range(len(states)), [f'State {s.name}' for s in states])
    state_ax.set_ylim(len(states) - 0.5, -0.5)
    state_ax.set_title('2. Ground-truth state activation', loc='left')
    state_ax.tick_params(labelbottom=False)
    state_ax.grid(axis='x', alpha=0.15)
    heat_ax = fig.add_subplot(grid[n_channels + 1], sharex=axes[0])
    heat_ax.pcolormesh(edges, np.arange(n_channels + 1) + 0.5,
                       data['channel_activations'], shading='flat',
                       cmap=ListedColormap(['#f2f4f7', '#253f64']), vmin=0, vmax=1)
    heat_ax.set_yticks(range(1, n_channels + 1), [f'Ch {c}' for c in range(1, n_channels + 1)])
    heat_ax.set_ylim(n_channels + 0.5, 0.5)
    heat_ax.set_title('3. Ground-truth channel activity — light: 0 (OFF), dark: 1 (ON)', loc='left')
    heat_ax.set_xlabel('Time (s)')
    heat_ax.set_xlim(0, duration)
    heat_ax.set_xticks(np.arange(0, duration + 0.001, 1))
    return fig


def save_data(data: dict[str, NDArray], figure: Figure, output_dir: Path) -> tuple[Path, Path]:
    """Save the five named arrays in one NPZ and the visualization as PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, png_path = output_dir / 'overlapping_signals.npz', output_dir / 'overlapping_signals.png'
    np.savez_compressed(npz_path, **data)
    figure.savefig(png_path, dpi=160)
    return npz_path, png_path


def main() -> None:
    """Run the configured experiment and save its observations and ground truth."""
    from brain_states.config import SIMULATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-rate', type=float, default=SIMULATION.sample_rate)
    parser.add_argument('--duration', type=float, default=SIMULATION.duration)
    parser.add_argument('--amplitude', type=float, default=SIMULATION.amplitude)
    parser.add_argument('--noise-std', type=float, default=SIMULATION.noise_std)
    parser.add_argument('--seed', type=int, default=SIMULATION.seed)
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    try:
        data = simulate(args.sample_rate, args.duration, args.amplitude,
                        noise_std=args.noise_std, seed=args.seed,
                        n_channels=SIMULATION.n_channels, states=SIMULATION.states)
    except ValueError as error:
        parser.error(str(error))
    figure = visualize(data, args.duration)
    paths = save_data(data, figure, args.output_dir)
    for name, array in data.items():
        print(f'{name}: {array.shape}')
    for path in paths:
        print(f'Saved {path.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == '__main__':
    main()

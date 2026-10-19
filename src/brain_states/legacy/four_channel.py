"""Four noiseless signals with two consecutive ground-truth states."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate_signals(sample_rate=1000, amplitude=1.0):
    """Return time (N,), signals (N, 4), states (N,); cover [0, 4.5)."""
    if not np.isfinite(sample_rate) or sample_rate <= 8:
        raise ValueError('Sample rate must be finite and greater than 8 Hz.')
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError('Amplitude must be finite and nonnegative.')
    counts = np.array([2.5, 2.0]) * sample_rate
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-9):
        raise ValueError('Both states must contain whole sample counts.')
    n1, n2 = np.rint(counts).astype(int)
    time = np.arange(n1 + n2) / sample_rate
    signals = np.zeros((n1 + n2, 4))
    signals[:n1, :2] = (amplitude * np.sin(2*np.pi*2*np.arange(n1)/sample_rate))[:, None]
    signals[n1:, 2:] = (amplitude * np.sin(2*np.pi*4*np.arange(n2)/sample_rate))[:, None]
    states = np.concatenate((np.ones(n1, dtype=int), np.full(n2, 2, dtype=int)))
    return time, signals, states


def plot_signals(time, signals, states, amplitude=1.0):
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.6]})
    # Display the mathematical endpoint without adding an extra data sample.
    display_time = np.append(time, 4.5)
    for channel, ax in enumerate(axes[:4]):
        ax.plot(display_time, np.append(signals[:, channel], 0), color='#263b65', lw=1)
        ax.set_ylabel(f'Signal {channel + 1}')
        ax.set_ylim(-1.2*max(amplitude, 1), 1.2*max(amplitude, 1))
    axes[4].step(display_time, np.append(states, 2), where='post', color='#263b65')
    axes[4].set_yticks([1, 2], ['State 1', 'State 2'])
    axes[4].set_ylim(0.7, 2.3)
    for ax in axes:
        ax.axvspan(0, 2.5, color='#c7e4f5', alpha=0.5)
        ax.axvspan(2.5, 4.5, color='#ccebd8', alpha=0.5)
        ax.axvline(2.5, color='#555555', linestyle='--', lw=1)
        ax.set_xlim(0, 4.5)
        ax.grid(alpha=0.2)
    axes[0].set_title('Synthetic brain signals: 2 Hz / 5 cycles, then 4 Hz / 8 cycles')
    axes[4].set_xlabel('Time (s)')
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-rate', type=float, default=1000)
    parser.add_argument('--amplitude', type=float, default=1)
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    try:
        time, signals, states = generate_signals(args.sample_rate, args.amplitude)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / 'brain_signals.csv',
               np.column_stack((time, signals, states)), delimiter=',',
               header='time_s,signal_1,signal_2,signal_3,signal_4,state', comments='',
               fmt=['%.9f', '%.17g', '%.17g', '%.17g', '%.17g', '%d'])
    fig = plot_signals(time, signals, states, args.amplitude)
    fig.savefig(args.output_dir / 'brain_signals.png', dpi=180)
    print(f'Saved {len(time)} samples covering 4.5 seconds to {args.output_dir.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

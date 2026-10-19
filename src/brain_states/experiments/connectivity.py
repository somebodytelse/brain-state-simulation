"""Experiment C: descriptive static/dynamic adjacency, without state discovery."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib.pyplot as plt

from brain_states.config import CONNECTIVITY

from brain_states.simulation import simulate, DEFAULT_STATES
from brain_states.connectivity import ConnectivityConfig, estimate_connectivity, validate_adjacency
from brain_states.connectivity_reference import build_references, compare_connectivity
from brain_states.visualization.connectivity import SNAPSHOT_TIMES, PAIRS, static_figure, dynamic_figure, pairwise_figure, save_animation


CORE_FILES = ('overlapping_signals.py', 'state_detection.py', 'adaptive_detection.py', 'state_evaluation.py')


def source_hashes() -> dict[str, str]:
    """Fingerprint existing simulation/detection code without modifying it."""
    import importlib
    module_names = {
        'overlapping_signals.py': 'brain_states.simulation',
        'state_detection.py': 'brain_states.state_detection',
        'adaptive_detection.py': 'brain_states.activity_detection',
        'state_evaluation.py': 'brain_states.evaluation',
    }
    return {name: hashlib.sha256(Path(importlib.import_module(module_names[name]).__file__).read_bytes()).hexdigest()
            for name in CORE_FILES}



def write_report(path: Path, result: dict, refs: dict, comparison: dict, metadata: dict) -> None:
    """Summarize measured coactivity and its state-identifiability limitations."""
    static = result['static_adjacency']
    lines = ['# Experiment C: static and dynamic adjacency', '',
        'Noiseless signals, 8 channels, 12 seconds, 200 Hz. The estimator receives only '
        'recorded signals, sampling frequency and configuration; references are built afterward. '
        'No state detection, clustering or changes to Boolean factorization were introduced.', '',
        f'Activity estimator: {metadata["config"]["activity_method"]}. Requested RMS window 0.1 s '
        '(existing centered 21-sample/105 ms calculation, edge padding). Fixed threshold 0.05; '
        'adaptive settings are saved in parameters.json and used only if selected. '
        f'Connectivity windows are {metadata["config"]["window_seconds"]} seconds long, stepped '
        f'by {metadata["config"]["step_seconds"]} seconds. Full windows only, with sample-grid '
        'positions independent of state transitions.', '',
        '## Arrays and definitions', '',
        f'- static_adjacency.npy: {static.shape}, full-recording active-time Jaccard.',
        f'- dynamic_adjacency.npy: {result["dynamic_adjacency"].shape}.',
        f'- window_centers.npy: {result["window_centers"].shape}; first center '
        f'{result["window_centers"][0]:.3f} s, last {result["window_centers"][-1]:.3f} s.',
        '- activity_estimates.npz: RMS, binary activity, thresholds, time and window start/end indices.',
        '- observed_signals.npz: observations and time, separated from references.',
        '- reference_adjacency.npz: static membership links; samplewise shared-active-state links; '
        'reference at window centers; mean reference occupancy in the same windows; memberships and state timelines.', '',
        'Zero union means connectivity zero, including two fully inactive channels. Diagonals '
        'are zero even for an active channel. Shared OFF samples do not increase Jaccard. '
        'The binary static reference connects any channels sharing a state. The time-dependent '
        'reference connects channels only when a state they share is currently active. '
        'Window-fraction reference is the mean of these binary links within each estimation '
        'window; it is not Jaccard. Center reference is instantaneous. These references are '
        'not used to choose or tune any estimator setting.', '',
        '## Measured findings', '',
        '| Channel pair | Interpretation | Full-recording Jaccard | Membership reference |',
        '|---|---|---|---|']
    labels = ('same state A', 'same state B', 'membership shared via C', 'independent states A/B')
    for (i,j), label in zip(PAIRS, labels):
        lines.append(f'| {i+1}–{j+1} | {label} | {static[i,j]:.6f} | {int(refs["static_reference"][i,j])} |')
    lines += ['', '| Requested / selected center (s) | Jaccard 1–2 | Jaccard 4–5 | Jaccard 3–7 | Jaccard 1–4 | Shared-active-state reference 1–4 |',
              '|---|---|---|---|---|---|']
    for requested in SNAPSHOT_TIMES:
        index = int(np.argmin(np.abs(result['window_centers']-requested)))
        values = [result['dynamic_adjacency'][index, i,j] for i,j in PAIRS]
        lines.append(f'| {requested:.1f} / {result["window_centers"][index]:.1f} | '
                     + ' | '.join(f'{value:.4f}' for value in values)
                     + f' | {int(refs["center_reference"][index,0,3])} |')
    lines += ['', '**Dynamic Jaccard does not distinguish independent simultaneously active states.** '
        'Channels 1 and 4 never share a state. During a window where both are continuously '
        'active, their intersection equals their union and Jaccard is 1, just as for a '
        'within-state pair. At the default centers 3.0 s and 9.5 s this produces an apparent '
        'connection despite a zero shared-state reference. This is statistical coactivity, '
        'not an anatomical connection or recovery of state-specific connectivity.', '',
        'Channel 3 mixes contributions from A and C. Its full-recording activity includes A-only '
        'periods when channel 7 is OFF, lowering their full-recording Jaccard despite shared '
        'membership through C. A fixed global state group is therefore not encoded uniquely '
        'by a coactivity similarity. Agreement with reference edges cannot prove latent-state recovery.', '',
        '## Temporal smoothing and boundary effects', '',
        'A 1-second connectivity window mixes activity before and after transitions. Neighbouring '
        'windows share 90% of samples at the defaults, so curves are strongly dependent. A '
        'window centered within half a window of a transition can include both states; '
        '0.1-second steps are not 0.1-second independent temporal resolution. Centered RMS '
        'adds about 50 ms of transition spread. Even a brief leaked activation can give '
        'Jaccard 1 for two identical channels in a mostly inactive window: inspect center '
        '0.5 s, whose [0,1) window lies before the true activation. The zero-union convention '
        'would give 0 if there were no leaked activity. Full windows omit the first and last '
        'half-window of center-time coverage; no partial-window padding is applied.', '',
        '## Reference comparison', '',
        'The following errors compare distinct definitions and are descriptive discrepancies, '
        'not state-recovery accuracy. Only unique off-diagonal pairs are included:', '',
        f'- Static MAE versus membership reference: {comparison["static_mae_vs_membership_reference"]:.6f}.',
        f'- Dynamic MAE versus reference at center: {comparison["dynamic_mae_vs_center_reference"]:.6f}.',
        f'- Dynamic MAE versus window-mean reference occupancy: {comparison["dynamic_mae_vs_window_fraction_reference"]:.6f}.',
        f'- Positive full-recording Jaccard on structural nonedges: '
        f'{comparison["positive_jaccard_on_structural_nonedges"]} of {comparison["structural_nonedge_pairs"]} pairs.', '',
        '## Reproduction and checks', '',
        '`python experiment_connectivity.py --no-show --animate`', '',
        'Use `--activity-method adaptive` to select the existing adaptive estimator. Use a separate '
        '`--output-dir` to preserve both configurations. Connectivity window and step can be changed '
        'with `--window-seconds` and `--step-seconds`; durations must map to whole samples. '
        'The connectivity API also exposes RMS and adaptive settings through ConnectivityConfig.', '',
        'All estimated/reference matrices were checked for shape, symmetry, zero diagonal, '
        'finite values and [0,1] bounds. Re-estimation from the same observation-only input '
        'was bitwise identical. Existing simulation/detection files were fingerprinted before '
        'and after execution and remained unchanged. parameters.json records source hashes, '
        'environment versions, configuration and window conventions. The GIF is optional; '
        'PNG figures and numerical arrays do not require animation.']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activity-method', choices=('fixed', 'adaptive'), default='fixed')
    parser.add_argument('--window-seconds', type=float, default=CONNECTIVITY.window_seconds)
    parser.add_argument('--step-seconds', type=float, default=CONNECTIVITY.step_seconds)
    parser.add_argument('--output-dir', type=Path, default=Path('output/connectivity_experiment_c'))
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    before = source_hashes()
    data = simulate(noise_std=0)
    fs = 200
    config = ConnectivityConfig(activity_method=args.activity_method,
                                window_seconds=args.window_seconds, step_seconds=args.step_seconds)
    try:
        result = estimate_connectivity(data['signals'], fs, config)
    except ValueError as error:
        parser.error(str(error))
    # Post-estimation only: known memberships and timelines enter reference construction.
    refs = build_references(data['state_memberships'], data['state_activations'],
                            result['window_starts'], result['window_ends'])
    for array in (result['static_adjacency'], result['dynamic_adjacency'], *refs.values()):
        validate_adjacency(array)
    count = (data['signals'].shape[1]-round(config.window_seconds*fs))//round(config.step_seconds*fs)+1
    assert result['static_adjacency'].shape == (8,8)
    assert result['dynamic_adjacency'].shape == (count,8,8)
    repeated = estimate_connectivity(data['signals'].copy(), fs, config)
    for name in result:
        np.testing.assert_array_equal(result[name], repeated[name])
    assert before == source_hashes()
    comparison = compare_connectivity(result, refs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in ('static_adjacency', 'dynamic_adjacency', 'window_centers'):
        np.save(out/(name+'.npy'), result[name], allow_pickle=False)
    np.savez_compressed(out/'activity_estimates.npz', time=data['time'], **{key: array for key, array in result.items()
                       if key not in ('static_adjacency', 'dynamic_adjacency')})
    np.savez_compressed(out/'observed_signals.npz', time=data['time'], signals=data['signals'])
    np.savez_compressed(out/'reference_adjacency.npz', **refs,
                        state_memberships=data['state_memberships'], state_activations=data['state_activations'])
    metadata = dict(config=asdict(config), sample_rate=fs, duration=12, noise_std=0,
                    amplitude=1, states=[asdict(state) for state in DEFAULT_STATES],
                    source_hashes=before, existing_code_unchanged=True,
                    numpy_version=np.__version__, matplotlib_version=plt.matplotlib.__version__,
                    python_version=platform.python_version(), animation=args.animate,
                    windows='Full half-open [start,end) windows on sample grid; center=(start+width/2)/fs.',
                    estimated_shapes={name: list(result[name].shape) for name in ('static_adjacency','dynamic_adjacency','window_centers')},
                    verification=dict(matrix_invariants=True, observed_only_estimation=True, bitwise_reproducible=True))
    (out/'parameters.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out/'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    figures = {'static_adjacency.png': static_figure(result, refs),
               'dynamic_snapshots.png': dynamic_figure(result, refs),
               'connectivity_over_time.png': pairwise_figure(result, refs, data['state_activations'], fs)}
    for name, figure in figures.items():
        figure.savefig(out/name, dpi=160)
    if args.animate:
        save_animation(out/'dynamic_adjacency.gif', result)
    write_report(out/'report.md', result, refs, comparison, metadata)
    print(json.dumps(comparison, indent=2))
    print(f'Static {result["static_adjacency"].shape}; dynamic {result["dynamic_adjacency"].shape}; saved {out.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()

"""Presentation comparison: one Gaussian realization scaled at four noise levels."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from brain_states.activity_detection import AdaptiveConfig, detect_adaptive
from brain_states.config import PRESENTATION_NOISE_SEED
from brain_states.state_detection import DetectorConfig
from brain_states.evaluation import evaluate
from brain_states.simulation import simulate, add_gaussian_noise, DEFAULT_STATES
from brain_states.paths import PROJECT_ROOT as ROOT
from brain_states.visualization.presentation import COLORS, INK, STYLE, fingerprint, timeline, membership_matrix

LEVELS = (0.05, 0.10, 0.15, 0.20)
EXTRA_COLOR = '#8B5FBF'


def run_inference(saved_config: dict, seed: int) -> tuple:
    """Reuse the generator; detection gets observations and saved settings only."""
    adaptive = AdaptiveConfig(**saved_config['adaptive'])
    factor = DetectorConfig(**saved_config['fixed'])
    # The current default structure must agree with the saved experiment.
    if json.loads(json.dumps([asdict(s) for s in DEFAULT_STATES])) != saved_config['states']:
        raise ValueError('Generator states differ from the saved experiment.')
    fs = saved_config['sample_rate']
    truth = simulate(sample_rate=fs, duration=saved_config['duration'],
                     amplitude=saved_config['amplitude'], n_channels=saved_config['n_channels'])
    clean = truth['signals']
    standard_normal = add_gaussian_noise(np.zeros_like(clean), noise_std=1.0, seed=seed)
    inputs, results = {}, {}
    # Finish every actual inference before post-detection truth matching.
    for sigma in LEVELS:
        noisy = clean + sigma*standard_normal
        noisy.setflags(write=False)
        inputs[sigma] = noisy
        results[sigma] = detect_adaptive(noisy, fs, adaptive, factor)
    metrics = {sigma:evaluate(results[sigma], truth) for sigma in LEVELS}
    for sigma in LEVELS:
        np.testing.assert_allclose((inputs[sigma]-clean)/sigma,standard_normal,rtol=0,atol=1e-14)
        assert np.any((inputs[sigma]-clean)[truth['channel_activations'] == 0] != 0)
        assert np.any((inputs[sigma]-clean)[truth['channel_activations'] == 1] != 0)
    return truth, standard_normal, inputs, results, metrics


def aligned_display(result: dict, metrics: dict) -> dict:
    """Match saved inference for display, retaining missed slots and every extra factor.

    Empty slots for missed true states are visual placeholders only; they are
    never detector outputs, and are explicitly labelled missing.
    """
    matches = {m['true_state']-1:m['discovered_state']-1 for m in metrics['matches']}
    rows, columns, labels, matrix_labels, colors = [], [], [], [], []
    for state in range(metrics['true_state_count']):
        name = chr(65+state)
        if state in matches:
            found = matches[state]
            rows.append(result['state_activations'][found])
            columns.append(result['memberships'][:,found])
            labels.append(f'{name} (D{found+1})')
            matrix_labels.append(f'{name}\n(D{found+1})')
        else:
            rows.append(np.zeros(result['state_activations'].shape[1],dtype=bool))
            columns.append(np.zeros(result['memberships'].shape[0],dtype=bool))
            labels.append(f'{name} missing')
            matrix_labels.append(f'{name}\nmissing')
        colors.append(COLORS[state])
    for found in metrics['unmatched_discovered_states']:
        rows.append(result['state_activations'][found-1])
        columns.append(result['memberships'][:,found-1])
        labels.append(f'Extra D{found}')
        matrix_labels.append(f'Extra\nD{found}')
        colors.append(EXTRA_COLOR)
    return dict(h=np.stack(rows), w=np.column_stack(columns), labels=labels,
                matrix_labels=matrix_labels, colors=colors)


def signals_comparison(truth, inputs, seed):
    """2x2 panel view with the same time, amplitude and channel offset scales."""
    fig,axes = plt.subplots(2,2,figsize=(16,9),sharex=True,sharey=True)
    fig.subplots_adjust(left=0.065,right=0.985,top=0.86,bottom=0.09,hspace=0.28,wspace=0.10)
    fig.suptitle('Synthetic Multichannel Signals at Different Noise Levels',fontsize=21,fontweight='bold',y=0.965)
    maximum = max(float(np.max(np.abs(x))) for x in inputs.values())
    spacing = 2.2*maximum
    offsets = np.arange(7,-1,-1)*spacing
    for ax,sigma in zip(axes.flat,LEVELS):
        for channel in range(8):
            memberships = np.flatnonzero(truth['state_memberships'][channel])
            color = COLORS[memberships[0]] if len(memberships) == 1 else INK
            ax.plot(truth['time'],inputs[sigma][channel]+offsets[channel],color=color,lw=0.65)
        ax.set_yticks(offsets,[f'Ch {c+1}' for c in range(8)])
        ax.set_ylim(-maximum*1.1,offsets[0]+maximum*1.1)
        ax.set_xlim(0,12)
        ax.set_xticks(np.arange(0,13,2))
        ax.set_title(r'$\sigma = '+f'{sigma:.2f}'+r'$',loc='left',fontsize=15,pad=10)
        ax.grid(axis='x',color='#dfe3e8',lw=0.6,alpha=0.8)
        ax.tick_params(axis='y',length=0)
        for spine in ('top','left','right'):
            ax.spines[spine].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.text(0.5,0.02,f'One shared noise realization (seed {seed}), scaled by sigma; identical amplitude and time scales',
             ha='center',fontsize=10,color='#647080')
    return fig


def detection_comparison(truth, results, metrics, fs, seed):
    """Four columns, matched-state colors and every additional state included."""
    fig=plt.figure(figsize=(16,9))
    fig.suptitle('Recovery of Overlapping States Under Increasing Noise',fontsize=21,fontweight='bold',y=0.975)
    grid=fig.add_gridspec(4,4,left=0.08,right=0.985,top=0.845,bottom=0.10,
                         height_ratios=[0.9,1.2,1.55,1.55],hspace=0.67,wspace=0.55)
    first=None
    display = {sigma:aligned_display(results[sigma],metrics[sigma]) for sigma in LEVELS}
    # Maintain one state-row scale, even where extra inferred factors appear.
    display_rows = max(item['h'].shape[0] for item in display.values())
    for col,sigma in enumerate(LEVELS):
        item=display[sigma]
        true_ax=fig.add_subplot(grid[0,col],sharex=first)
        if first is None:
            first=true_ax
        timeline(true_ax,truth['state_activations'],fs,['A','B','C'],COLORS)
        true_ax.set_title(r'$\sigma = '+f'{sigma:.2f}'+r'$'+'\nGround-truth activation',fontsize=12,pad=10)
        true_ax.tick_params(axis='x',labelbottom=False)
        true_ax.tick_params(axis='y',pad=5,labelsize=10)
        true_ax.set_xticks([0,4,8,12])
        found_ax=fig.add_subplot(grid[1,col],sharex=first)
        timeline(found_ax,item['h'],fs,item['labels'],item['colors'])
        found_ax.set_ylim(display_rows-0.5,-0.5)
        found_ax.set_title('Detected activation',fontsize=12,pad=9)
        found_ax.tick_params(axis='both',labelsize=9)
        found_ax.tick_params(axis='y',pad=5)
        found_ax.set_xlabel('Time (s)',fontsize=10,labelpad=3)
        true_matrix=fig.add_subplot(grid[2,col])
        found_matrix=fig.add_subplot(grid[3,col])
        membership_matrix(true_matrix,truth['state_memberships'],['A','B','C'],COLORS)
        membership_matrix(found_matrix,item['w'],item['matrix_labels'],item['colors'])
        true_matrix.set_title('True memberships',fontsize=12,pad=9)
        found_matrix.set_title('Detected memberships',fontsize=12,pad=9)
        for ax in (true_matrix,found_matrix):
            ax.tick_params(axis='both',labelsize=9,pad=4)
    first.set_xticks([0,4,8,12])
    fig.text(0.5,0.026,f'Adaptive RMS + Boolean factorization | Single illustrative realization per level (shared seed {seed})',
             ha='center',fontsize=10,color='#647080')
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=PRESENTATION_NOISE_SEED)
    parser.add_argument('--config',type=Path,default=ROOT/'output/adaptive_comparison/parameters.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'presentation_figures/noise_comparison')
    args=parser.parse_args()
    protected = [ROOT/'presentation_figures'/f'{name}.{ext}' for name in ('01_synthetic_signals','02_state_detection') for ext in ('png','pdf')]
    protected += [ROOT/name for name in ('overlapping_signals.py','state_detection.py','adaptive_detection.py')]
    protected = [path for path in protected if path.exists()]
    before={str(path):fingerprint(path) for path in protected}
    saved=json.loads(args.config.read_text(encoding='utf-8'))
    truth,noise,inputs,results,metrics=run_inference(saved,args.seed)
    # Confirm the generated structure agrees exactly with the original saved data.
    with np.load(ROOT/'output/overlapping_signals.npz',allow_pickle=False) as original:
        for key in ('time','signals','state_activations','channel_activations','state_memberships'):
            np.testing.assert_array_equal(truth[key],original[key])
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sigma in LEVELS:
        m=metrics[sigma]
        rows.append(dict(sigma=sigma,seed=args.seed,detector='adaptive RMS',
            channel_activity_f1=m['channel_activity']['f1'],temporal_iou=m['temporal_iou_with_unmatched_zero'],
            membership_jaccard=m['membership_jaccard_with_unmatched_zero'],discovered_states=m['discovered_state_count'],
            unmatched_true_states=json.dumps(m['unmatched_true_states']),
            unmatched_discovered_states=json.dumps(m['unmatched_discovered_states'])))
    with (out/'noise_comparison_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context(STYLE):
        figures=[('01_noisy_signals_comparison',signals_comparison(truth,inputs,args.seed)),
                 ('02_noisy_detection_comparison',detection_comparison(truth,results,metrics,saved['sample_rate'],args.seed))]
        for name,figure in figures:
            for ext in ('png','pdf'):
                figure.savefig(out/f'{name}.{ext}',dpi=300,facecolor='white')
            plt.close(figure)
    arrays=dict(time=truth['time'],clean_signals=truth['signals'],standard_normal_noise=noise,
                true_state_activations=truth['state_activations'],true_state_memberships=truth['state_memberships'])
    for sigma in LEVELS:
        prefix=f'sigma_{sigma:.2f}'
        arrays[prefix+'_signals']=inputs[sigma]
        for key,value in results[sigma].items():
            arrays[prefix+'_'+key]=value
    np.savez_compressed(out/'noise_comparison_data.npz',**arrays)
    for path in protected:
        assert fingerprint(path)==before[str(path)]
    provenance=dict(seed=args.seed,noise_levels=LEVELS,detector='adaptive RMS',
        adaptive_config=saved['adaptive'],factor_config=saved['fixed'],
        source_config=str(args.config.resolve()),source_config_sha256=fingerprint(args.config),
        experiment='Single illustrative realization, scaled identically across levels; not repeated-run averages.',
        simulation=dict(sample_rate=saved['sample_rate'],duration=saved['duration'],amplitude=saved['amplitude'],states=saved['states']),
        unchanged_artifact_sha256=before,
        verification=dict(truth_identical=True,shared_noise_verified=True,unchanged_settings=True,
                          real_inference=True,ground_truth_used_only_for_reference_and_post_detection_matching=True),
        metrics=metrics,detector_version='Saved validated adaptive configuration',size_inches=[16,9],dpi=300)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Single illustrative realization per sigma; adaptive settings unchanged; seed',args.seed)
    print('sigma    activity F1    temporal IoU    membership Jaccard    states')
    for row in rows:
        print(f'{row["sigma"]:.2f}     {row["channel_activity_f1"]:.4f}         {row["temporal_iou"]:.4f}          {row["membership_jaccard"]:.4f}                {row["discovered_states"]}')
    print('Saved',out.resolve())


if __name__=='__main__':
    main()

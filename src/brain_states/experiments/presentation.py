"""Presentation figures from saved noiseless observations and detector outputs only."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from brain_states.visualization.presentation import (COLORS, INK, STYLE, fingerprint, timeline,
    synthetic_figure, membership_matrix, detection_figure)
from brain_states.evaluation import evaluate

from brain_states.paths import PROJECT_ROOT as ROOT

__all__ = ["ROOT", "COLORS", "INK", "STYLE", "fingerprint", "timeline", "synthetic_figure",
           "membership_matrix", "detection_figure", "main"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signals',type=Path,default=ROOT/'output/overlapping_signals.npz')
    parser.add_argument('--detections',type=Path,default=ROOT/'output/detection_v1/discovered_states.npz')
    parser.add_argument('--evaluation',type=Path,default=ROOT/'output/detection_v1/evaluation.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'presentation_figures')
    args = parser.parse_args()
    inputs = [args.signals,args.detections,args.evaluation]
    hashes = {str(p.resolve()):fingerprint(p) for p in inputs}
    with np.load(args.signals,allow_pickle=False) as saved:
        truth = {key:saved[key].copy() for key in saved.files}
    with np.load(args.detections,allow_pickle=False) as saved:
        detected = {key:saved[key].copy() for key in saved.files}
    previous = json.loads(args.evaluation.read_text(encoding='utf-8'))
    fs = previous['sample_rate']
    if truth['signals'].shape != (8,2400) or not np.allclose(np.diff(truth['time']),1/fs):
        raise ValueError('Expected the saved 8-channel, 12-second, 200 Hz recording.')
    if np.any(truth['signals'][truth['channel_activations'] == 0] != 0):
        raise ValueError('Input must be the saved noiseless recording.')
    metrics = evaluate(detected,truth)  # Matching only; no detector is imported or executed.
    if metrics['matches'] != previous['metrics']['matches']:
        raise ValueError('Saved results do not match the existing evaluation record.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        figures = [('01_synthetic_signals',synthetic_figure(truth,fs)),
                   ('02_state_detection',detection_figure(truth,detected,metrics,fs))]
        for name,fig in figures:
            for extension in ('png','pdf'):
                path = args.output_dir/f'{name}.{extension}'
                fig.savefig(path,dpi=300,facecolor='white')
                print(path.resolve())
            plt.close(fig)
    for path in inputs:
        if fingerprint(path) != hashes[str(path.resolve())]:
            raise RuntimeError('An input artifact changed during plotting.')
    provenance = dict(source_sha256=hashes,detector_rerun=False,dpi=300,
                      size_inches=[16,9],colors=dict(zip(('A','B','C'),COLORS)),
                      post_detection_matches=metrics['matches'])
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

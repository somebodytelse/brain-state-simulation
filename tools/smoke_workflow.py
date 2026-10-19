"""Run the documented workflow in a new isolated directory, preserving saved results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


COMMANDS = (
    'brain_states.simulation',
    'brain_states.experiments.baseline',
    'brain_states.experiments.noise_robustness',
    'brain_states.experiments.adaptive_rms',
    'brain_states.experiments.connectivity',
    'brain_states.experiments.presentation',
    'brain_states.experiments.noisy_presentation',
)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args()
    work=args.work_dir.resolve()
    work.mkdir(parents=True,exist_ok=True)
    # An installed wheel resolves presentation inputs relative to this directory.
    # Refuse editable mode, which intentionally uses the source repository root.
    from brain_states.paths import PROJECT_ROOT
    if PROJECT_ROOT.resolve() != work:
        raise SystemExit('Run this helper from --work-dir using a non-editable wheel installation.')
    environment=dict(os.environ,MPLBACKEND='Agg')
    results=[]
    for module in COMMANDS:
        command=[sys.executable,'-m',module]
        if module not in COMMANDS[-2:]:
            command.append('--no-show')
        print('Running',module,flush=True)
        completed=subprocess.run(command,cwd=work,env=environment,capture_output=True,text=True)
        (work/(module.rsplit('.',1)[-1]+'.log')).write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        results.append(dict(module=module,returncode=completed.returncode))
        if completed.returncode:
            raise RuntimeError(f'{module} failed; inspect its log in {work}')
    (work/'workflow_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('All seven documented workflows passed.',flush=True)


if __name__=='__main__':
    main()

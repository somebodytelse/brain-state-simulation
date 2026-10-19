# Project structure and migration

## Before

The repository had 23 top-level Python files: reusable algorithms, experiment
runners, plotting functions, six test modules and a historical exploratory
`test.py` script mixed together. The README accumulated sequential experiment
notes. Two virtual environments, IDE files, caches, temporary PDF renders,
`output/` and `presentation_figures/` lived alongside the source. There were 58
saved artifacts, approximately 14 MB in total. Git had no commits, with five
IDE files and the original main/test scripts already staged.

## After

```text
.
├── README.md, CONTRIBUTING.md
├── pyproject.toml, requirements.txt
├── .gitignore, .gitattributes
├── .github/workflows/tests.yml
├── configs/
│   ├── README.md
│   └── reference_constraints.txt
├── src/brain_states/
│   ├── config.py, paths.py
│   ├── simulation.py
│   ├── state_detection.py
│   ├── activity_detection.py
│   ├── evaluation.py
│   ├── connectivity.py, connectivity_reference.py
│   ├── visualization/
│   ├── experiments/
│   └── legacy/
├── tests/
├── tools/smoke_workflow.py
├── docs/
│   ├── methods.md, outputs.md, reproducibility.md
│   ├── structure.md, github.md, experiment_notes.md
│   ├── assets/
│   └── reference_results/
├── output/                 # retained locally, ignored
├── presentation_figures/  # retained locally, ignored
└── root *.py launchers    # backward compatibility, no copied algorithms
```

The complete experiment commands are in the README. `src/brain_states/experiments`
keeps run orchestration within the installed package, rather than adding a
second source tree of duplicate script implementations. Shared plotting code
is in `visualization`; the presentation layout was separated from its CLI.

## Module names

| Original import/script | Canonical module |
|---|---|
| overlapping_signals | brain_states.simulation |
| state_detection | brain_states.state_detection |
| adaptive_detection | brain_states.activity_detection |
| state_evaluation | brain_states.evaluation |
| connectivity / connectivity_reference | brain_states.connectivity / connectivity_reference |
| detection_visualization | brain_states.visualization.detection |
| adaptive_visualization | brain_states.visualization.adaptive |
| connectivity_visualization | brain_states.visualization.connectivity |
| run_detection | brain_states.experiments.baseline |
| experiment_noise | brain_states.experiments.noise_robustness |
| experiment_adaptive | brain_states.experiments.adaptive_rms |
| experiment_connectivity | brain_states.experiments.connectivity |
| plot_presentation_figures | brain_states.experiments.presentation |
| plot_noise_presentation_figures | brain_states.experiments.noisy_presentation |
| main | brain_states.legacy.four_channel |
| test.py | brain_states.legacy.three_state_clustering |

Compatibility imports alias the canonical module, preserving monkeypatching
and ordinary function imports. Legacy test-module commands still run through
small loaders. Discover the canonical suite with `unittest discover -s tests`.
Importing the old `test.py` no longer executes a noisy clustering experiment
accidentally; explicitly running it retains the historical behavior. The
legacy three-state script uses SciPy/scikit-learn and is not a current detector.

## Deliberate preservation

All existing numerical artifacts and presentation exports remain untouched.
The original accumulated notes are preserved in experiment_notes.md. The
original detector file is byte-identical, including its historical mixed final
newline. `.gitattributes` disables newline normalization for that one file so
its pinned source hash survives Git checkouts. Other Python files use LF.

Imports, paths, configuration plumbing and documentation changed; scientific
functions and experimental defaults did not. Source-hash verification now
locates canonical modules, so it does not accidentally hash a launcher.
Existing roots `output/` and `presentation_figures/` were retained to avoid
breaking output schemas and provenance. They are not renamed to `outputs/`.

## Duplication and unused material

Repeated implementation copies were not introduced: root modules only delegate.
Shared presentation helpers moved into one visualization module. Existing
shared CSV writers, metric calculation and matching functions are reused.
Unused imports introduced or exposed by the move were removed. Earlier main.py
and test.py experiments were retained as documented legacy code, not deleted
as supposedly unused. Virtual environments, caches and intermediate renders
are ignored and remain local. No Git history or user staging was rewritten.

# Reproducibility and verification

## Running and inspecting configuration

Install the project and run the README commands in order from the repository
root. `python -m brain_states.config` prints all defaults. Use the reference
constraints on Python 3.12 to reproduce the saved dependency environment.
NumPy's RNG seed handling, original state definitions and detector settings
are unchanged. Research provenance includes run settings and seeds.

The baseline has no stochastic initialization. Noise experiments spawn one
reproducible independent RNG per run. The paired adaptive study reuses each
recording across detectors; the presentation comparison instead scales one
shared standard-normal recording across its four sigma levels. These are
intentionally different experimental designs.

## Checks performed for the refactor

- The 21 original tests passed before restructuring and after relocation.
- Four additional workflow/configuration tests make the current suite 25 tests.
- A fresh Python 3.12 environment installed the project with its reference
  dependency constraints; both editable and ordinary wheel installation work.
- All seven README workflows were executed with a non-editable wheel in an
  isolated working directory, without reading existing result files.
- Both full noise studies retained 30 runs at each of eight levels.
- Scientific function/class definitions in simulation, adaptive activity,
  evaluation and connectivity were compared by AST (23 definitions unchanged).
- The complete original detector implementation remains byte-identical with
  SHA256 e4c9fea33d5bf3a94e1c8dc0f1f97ddc14a291e630b9c1d37b9157b6b9c48093.
- Regenerated 182 NPZ arrays, 3 NPY arrays and 1,248 CSV rows matched the saved
  results (arrays exactly, numeric tables within 1e-12). All 18 matching PNG
  figures were pixel-identical.
- Every one of the 58 pre-existing output/figure artifacts was fingerprinted
  before refactoring and verified unchanged afterward.

The numerical baseline remains activity F1 0.9897341405633062, temporal IoU
0.9783083395943947, membership Jaccard 1.0 and three discovered states. The
portable baseline reference under docs/reference_results is used for regression
checks, never for initializing or tuning the algorithm.

## Run checks yourself

```bash
python -m unittest discover -s tests -v
```

For a complete isolated workflow, build and install a non-editable wheel in a
separate environment. Create a new working directory, change into it, then run
`tools/smoke_workflow.py` by its path in your source checkout, with that
installed interpreter and `--work-dir .`. The helper refuses editable mode
because source-checkout presentation paths intentionally point to the repo.
It saves logs and regenerates all seven workflows only inside that directory.

Existing root-script commands remain available in a checkout. Installed console
scripts or `python -m brain_states...` avoid relying on the old imports.

## Provenance and limitations

Saved artifacts retain original provenance rather than being rewritten during
refactoring. Different file locations and current source hashes may appear in
new metadata; source relocation alone is not a change in scientific results.
PDF timestamps and machine-specific provenance paths are not numeric regression
criteria. Binary arrays and numeric metrics are the comparison targets.

The local environment was Python 3.12 on Windows. CI is configured for Python
3.10 and 3.12 on Linux; hosted CI has not been run locally. Optional legacy
SciPy/scikit-learn execution is preserved but is outside the modern pipeline's
reference environment. A license still needs an explicit owner decision.

A portable verification summary is saved in `refactor_verification.json`.

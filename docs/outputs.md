# Output schemas and artifact policy

Defaults preserve the established `output/` and `presentation_figures/` paths.
They are generated automatically, retained locally and ignored by Git. No
existing data or figure was deleted during restructuring.

| Producer | Key files |
|---|---|
| simulation | `overlapping_signals.npz`: time (T,), signals (8,T), state_activations (3,T), channel_activations (8,T), state_memberships (8,3); positive-noise runs also save clean_signals |
| baseline | `discovered_states.npz`: memberships (C,K), state_activations (K,T), reconstruction (C,T), rms (C,T), channel_activity (C,T), objective_history; evaluation.json and two PNGs |
| noise robustness | per_run_metrics.csv (240 rows), summary_statistics.csv (8 rows), parameters.json, example_runs.npz, report and three PNGs |
| adaptive comparison | per_run_metrics.csv (480 rows), summary_statistics.csv (16 rows), paired_comparison.csv (240 differences), paired_summary.csv (8 rows), seeds/configuration, example NPZ, five PNGs and report |
| connectivity | static_adjacency.npy (8,8), dynamic_adjacency.npy (111,8,8), window_centers.npy (111,), activity_estimates.npz, observed_signals.npz, reference_adjacency.npz, three PNGs, optional GIF, comparison/configuration and report |
| presentation | two 16:9 figures, PNG at 300 DPI and vector PDF; provenance.json |
| noise presentation | four-level shared-noise comparison, two PNG/PDF pairs, metrics CSV, replayable data and provenance |

Load NumPy archives with `allow_pickle=False`. Shapes are channel-major except
for the preserved historical four-channel `main.py` experiment, whose signals
are sample-major (T,4). That exception is documented and not silently changed.

State labels are arbitrary; figures align them after detection. Raw discoveries
are never replaced by reference arrays. Histograms labelled with truth and
reference connectivity comparisons are post-detection analyses only.

Compact, unchanged numeric reference tables are included in
`docs/reference_results/`. Example PNGs in `docs/assets/` are unchanged copies
of existing figures. Historical provenance in local generated directories can
contain absolute paths and old source hashes; it remains an original record,
not an input-portability requirement. Fresh runs save the current provenance.

Large recordings, environments, temporary files and generated presentation
exports are intentionally absent from the version-control candidate set.

# Synthetic brain signals

Install with `python -m pip install -r requirements.txt`, then run
`python main.py`. Use `python main.py --no-show` to save without opening a
window. Options: `--sample-rate`, `--amplitude`, `--output-dir`.

Default: 1,000 Hz, amplitude 1, no noise. State 1 covers [0, 2.5):
channels 1 and 2 complete five cycles at 2 Hz. State 2 covers [2.5, 4.5):
channels 3 and 4 complete eight cycles at 4 Hz. Inactive channels are
exactly zero. Both active pairs start at phase zero and thus have identical
values in separate columns.

Outputs: `output/brain_signals.csv` and `output/brain_signals.png`.
CSV columns: time_s, signal_1, signal_2, signal_3, signal_4, state.
There are 4,500 rows at timestamps 0 through 4.499 seconds. Each sample
represents a 0.001-second interval, giving exactly 4.5 seconds of coverage.
The 2.5-second sample belongs to state 2. The plot includes the mathematical
endpoint at 4.5 seconds for display only. There are no additional OFF states.
ON signals cross zero; use ground-truth labels to identify the state.

`generate_signals()` returns NumPy arrays: time (N,), signals (N, 4), and
integer state labels (N,). Custom sampling rates must exceed 8 Hz and yield
whole sample counts for both state durations.

The existing test.py remains a separate older clustering experiment.

## Overlapping-state experiment

Run `.\.signal-venv\Scripts\python.exe overlapping_signals.py --no-show`
(or `python overlapping_signals.py` in an environment with requirements.txt
installed). This creates output/overlapping_signals.npz and
output/overlapping_signals.png. The earlier main.py experiment is preserved.

Defaults: 8 channels, 200 Hz, 12 seconds, amplitude 1, no noise.
State A: channels 1–3, 2 Hz, [1,4) and [8,10).
State B: channels 4–6, 4 Hz, [2,6) and [9,12).
State C: channels 3,7,8, 3 Hz, [5,9).
Shared channel 3 sums both state contributions, without normalization.
Sines use global recording time and zero phase. Configure state memberships,
intervals, frequencies and phases in DEFAULT_STATES or pass State objects to
simulate(). State's default frequency is 3 Hz. CLI parameters include
--sample-rate, --duration, --amplitude, --output-dir and --no-show.
Intervals must fit the configured duration; edit them for shorter recordings.

The NPZ contains time (2400,), signals (8,2400), state_activations (3,2400),
channel_activations (8,2400) and state_memberships (8,3). Binary matrices
use uint8. Time covers [0,12), with final timestamp 11.995 seconds.
Load with np.load('output/overlapping_signals.npz', allow_pickle=False).
Supply only the signals array to future detection algorithms; ground-truth
arrays are separate entries for evaluation. Activity labels reflect state
membership, not whether a sine sample is nonzero.

Verify with `.\.signal-venv\Scripts\python.exe -m unittest test_signals test_overlapping_signals`.

## Unsupervised state detection, version 1

Run the complete pipeline:

```powershell
.\.signal-venv\Scripts\python.exe overlapping_signals.py --no-show
.\.signal-venv\Scripts\python.exe run_detection.py --no-show
.\.signal-venv\Scripts\python.exe -m unittest test_signals test_overlapping_signals test_state_detection
```

Existing simulation scripts are unchanged. New modules:
- state_detection.py: observed-data-only RMS estimation and Boolean discovery.
- state_evaluation.py: post-detection exact one-to-one matching and metrics.
- detection_visualization.py: detection and evaluation figures.
- run_detection.py: load observations, detect and save, then load ground truth.
- test_state_detection.py: independent fixtures, overlap, shared memberships,
  penalties, silence, label permutations, optimal assignment and unmatched states.

### Detector inputs and settings

`detect(signals, sample_rate, config=DetectorConfig())` accepts only observations,
sampling rate and optional settings. It neither imports the simulator nor reads
files or ground truth. The default runner uses the existing NPZ lazily: only its
signals entry is accessed before discovery finishes. The sampling frequency is
supplied separately (default 200 Hz); the time vector is checked afterward.
The number of states is chosen by the objective, never supplied by the generator.
All operations are deterministic; no random initialization or seed is needed.

Centered RMS uses a 0.1-second requested window and a 0.05 amplitude threshold.
The window is rounded up to an odd number of samples (21 at 200 Hz, 105 ms),
with edge padding. It bridges ordinary sine zero crossings on this experiment.
It is offline and uses future samples. Transition timing can be spread by about
half a window; low thresholds activate earlier and deactivate later. Larger
thresholds miss weak activity, and shorter windows can fragment low-frequency
waves. Cancellation on shared channels can still conceal activity. Parameters
were selected before evaluation and were not tuned using ground truth.

### Boolean factorization objective

Let A be estimated binary channel activity, W channel memberships and H state
activity. Reconstruction is R[c,t] = OR_k (W[c,k] AND H[k,t]). Minimize:

`J = count(A != R) + fs * (state_penalty_seconds * K + membership_penalty_seconds * count(W))`.

Defaults: state penalty 0.5, membership penalty 0.02, minimum group size 2,
minimum active run duration 0.2 seconds. Penalty units are channel-sample error
equivalents, scaled by fs so changes in sampling rate preserve their time scale.
The state penalty controls model size; the membership penalty discourages larger
unnecessary groups. A high state penalty may choose no states; low penalties can
split patterns into many factors. There is no fixed target number of states.

Enumerate all channel subsets of size at least two, form their AND timelines,
remove runs shorter than the minimum duration, and greedily add the factor with
largest positive reduction in J. Each timeline includes all qualifying recurring
runs. This favors sustained coverage and reuses one factor across recurrences;
it does not impose a minimum number of recurrences because single sustained
activations can also be meaningful. Remove redundant factors if doing so further
reduces J. Full AND timelines cannot produce false positives against A, but they
can inherit errors in RMS activity. Short activity and one-channel states are
intentionally unmodelled. This restricted greedy search is not a global optimum.
Subset enumeration is exponential and capped at 16 channels for this baseline.

### Evaluation and saved outputs

output/detection_v1/discovered_states.npz contains rms, channel_activity,
memberships (W), state_activations (H), reconstruction and objective_history.
output/detection_v1/detection.png shows observations and detected structures.
output/detection_v1/evaluation.png shows aligned true/discovered timelines and
memberships, retaining unmatched states.
output/detection_v1/evaluation.json records input, configuration, metrics,
matching rule, objective history and limitations.

Evaluation uses exact maximum-weight one-to-one assignment with pair score
(mean temporal IoU and channel-membership Jaccard). Bitmask dynamic programming
supports at most 16 states on the smaller side; zero-score pairs remain unmatched.
Labels are arbitrary. Per-match scores and both lists of unmatched states are
reported. Overall IoU and Jaccard sums are divided by max(true count, discovered
count), so unmatched states contribute zero. Channel scores are micro-averaged.
Boolean reconstruction error is the fraction of channel-samples mismatched
against estimated activity; ground-truth reconstruction error is separately named.

Default experiment results (initial settings, no ground-truth tuning):
- 3 true states and 3 discovered states; no unmatched states.
- D1: channels 4,5,6; matched True 2 (State B), temporal IoU 0.982456.
- D2: channels 1,2,3; matched True 1 (State A), temporal IoU 0.970874.
- D3: channels 3,7,8; matched True 3 (State C), temporal IoU 0.981595.
- Membership Jaccard: 1.0 for each match; mean temporal IoU 0.978308.
- Activity precision 0.979677, recall 1.0, F1 0.989734.
- 195 false-positive channel-samples, no false negatives; transition smoothing
  explains the discrepancy in this noiseless experiment.
- Boolean reconstruction error against estimated activity: 0.0.
- Reconstruction error against true activity: 0.01015625.
- All 8 tests passed; both saved figures were visually inspected.

These results describe this synthetic recording, not a validated neuroscience
method. Matching evaluation confirms the expected groups here, but exact Boolean
reconstruction alone does not establish state recovery. If two groups always
activate together, a merged group and separate states can produce identical
activity. RMS also discards frequency and phase information. Different latent
configurations can therefore remain indistinguishable, and the complexity
penalty selects one explanation rather than proving it is the true one.

## Experiment A: noise robustness

The generator now supports `simulate(noise_std=0.2, seed=42)` and CLI options
`--noise-std` and `--seed`. Noise is independent Gaussian N(0, sigma^2), added
to every channel at every time point. Noise-free output and existing call
signatures retain their default behavior. Noisy simulations additionally store
clean_signals; truth remains unchanged. To generate a standalone noisy recording
while keeping the original output files, use:

```powershell
.\.signal-venv\Scripts\python.exe overlapping_signals.py --noise-std 0.2 --seed 42 --output-dir output/noisy_example --no-show
```

Run the full eight-level, 30-run-per-level study:

```powershell
.\.signal-venv\Scripts\python.exe experiment_noise.py --no-show
```

The unchanged RMS and factorization algorithms use the same DetectorConfig()
for all 240 trials. Ground truth is passed only to post-detection evaluation.
The experiment runner reuses a fixed clean simulation and the generator's noise
function; each trial has its own SeedSequence-derived seed. The zero-noise
trials are necessarily identical, though their assigned seeds are distinct.
Change --base-seed to repeat with different realizations; default 20261012.

Outputs in output/noise_experiment_a:
- per_run_metrics.csv: all metrics, seeds, per-pair matching scores, unmatched
  state IDs, and active/inactive residual-noise standard deviations.
- summary_statistics.csv: per-level means and sample standard deviations.
- parameters.json: generation and detection parameters, software versions,
  seed generation, matching and score definitions.
- example_signals.png, performance_vs_noise.png, state_examples.png: the three
  requested figures, with fixed run-1 examples and comparable signal axes.
- example_runs.npz: example observations, clean signals, detector outputs.
- report.md: experiment results and interpretation.

At sigma=0.05, activity F1 drops to 0.8027, with 6.47 discovered states on average
and 0% exact count recovery. The matched true groups still have perfect membership
Jaccard, but extra factors lower aggregate membership Jaccard to 0.4717.
At sigma>=0.10, discovery collapses to one all-channel state. At sigma>=0.20,
reconstruction of estimated activity is perfect even though ground-truth activity
error is 0.5104. This exposes failure of activity estimation under the original
fixed threshold; it is not evidence of successful state recovery. Matched-only
scores and unmatched-penalized scores are kept separate.

Verify with:

```powershell
.\.signal-venv\Scripts\python.exe -m unittest test_signals test_overlapping_signals test_state_detection test_noise_experiment
```

## Adaptive RMS paired comparison

New modules: adaptive_detection.py, experiment_adaptive.py,
adaptive_visualization.py and test_adaptive_detection.py. The original
state_detection.py (fixed RMS and Boolean factorization) remains unchanged.

```powershell
.\.signal-venv\Scripts\python.exe experiment_adaptive.py --no-show
```

The adaptive API accepts only signals, sampling rate and configuration:
`detect_adaptive(signals, sample_rate, config=AdaptiveConfig())`.
For each channel, it uses `max(0.05, 1.5 * percentile(RMS, 20))`.
The percentile, scaling factor and threshold floor are configurable using
--percentile, --scale and --threshold-floor; settings stay identical across
all noise levels. Defaults were selected before evaluation, not optimized
using ground truth. The 0.05 floor handles zero noiseless background.
Lower quantiles require enough OFF samples and can be contaminated by active
signal; neither known sigma nor true activity is used to set the threshold.
Both detectors reuse the same centered odd RMS window and edge padding.

All 240 paired recordings use the previous Experiment A seeds. Both detectors
receive the same read-only array, and the runner verifies identical RMS
values. Ground truth enters evaluation only after both runs finish. The
factorizer is reused unchanged, with original settings and no target count.
The runner checks its source SHA256 and compares every fixed run against
saved Experiment A metrics for matching seeds (240 for the default run).

Outputs under output/adaptive_comparison:
- per_run_metrics.csv: 480 detector/run rows, thresholds, backgrounds, seeds,
  recording hashes, all scores, per-state matches and unmatched IDs.
- summary_statistics.csv: 16 detector/noise rows of mean and sample SD.
- paired_comparison.csv: 240 adaptive-minus-fixed differences, with seeds.
- paired_summary.csv: 8 mean/sample-SD rows of paired differences.
- parameters.json and seeds.csv: configuration, versions and reproducibility.
- Five PNGs: rms_envelopes, activity_comparison, state_discovery_comparison,
  state_recovery_examples and rms_distributions.
- rms_distribution_overlap.csv: post-detection histogram overlap analysis.
- example_runs.npz: run-1 signals and detector outputs at selected levels.
- report.md: measured improvements, warnings and scientific limitations.

Adaptive restores correct groups/count in every run at sigma=0.05 and 0.10.
At sigma=0.10, F1 is 0.9888 and temporal IoU is 0.9835, versus fixed 0.6574
and 0.1944. At sigma=0.20–0.30, adaptive activity F1 exceeds 0.98, but
extra factors reduce state recovery. At sigma=0.50, adaptive activity F1 is
better while temporal IoU is worse than fixed. At sigma=0.75 and 1.00,
adaptive activity F1 and state scores are worse than fixed; at sigma=1.00
it discovers no states. High thresholds sacrifice recall, and fragmentary
activity does not yield persistent Boolean factors. No parameters were
changed to conceal these failures. See report.md for full comparisons.

Tests:

```powershell
.\.signal-venv\Scripts\python.exe -m unittest test_signals test_overlapping_signals test_state_detection test_noise_experiment test_adaptive_detection
```

## Experiment C: static and dynamic functional adjacency

Run the separate descriptive connectivity experiment:

```powershell
.\.signal-venv\Scripts\python.exe experiment_connectivity.py --no-show --animate
```

The animation is optional; omit --animate to save only arrays and PNG figures.
The input is the original noiseless, 8-channel, 200 Hz, 12-second simulation.
No state-discovery algorithm is run or modified.

New files:
- connectivity.py: observed-only estimation API, pairwise Jaccard, uniform
  sliding windows and matrix validation.
- connectivity_reference.py: post-estimation membership/active-state references.
- connectivity_visualization.py: static/dynamic matrices, selected pair curves,
  separate truth reference panel and optional Matplotlib GIF.
- experiment_connectivity.py: execution, verification, saving and report.
- test_connectivity.py: Jaccard arithmetic/empty unions, overflow protection,
  transition-independent window placement, shared-state references, coactivity
  ambiguity, observed-only boundary and reproducibility.

`estimate_connectivity(signals, sample_rate, config=ConnectivityConfig())`
accepts observations and settings only. It reuses existing RMS activity
estimation. Select `--activity-method adaptive` for the existing adaptive
method and choose a separate --output-dir to preserve both results. The
ConnectivityConfig API also exposes RMS window, fixed threshold and adaptive
configuration. No ground truth is passed to either activity estimator.

Default windows are full half-open [start,end) windows of 1 second every
0.1 seconds, giving 111 matrices with centers 0.5 through 11.5 seconds. Window
positions never depend on state transitions. Zero union gives zero Jaccard;
all diagonals are zero. All matrices are symmetric and bounded by [0,1].

Outputs under output/connectivity_experiment_c:
- static_adjacency.npy (8,8), dynamic_adjacency.npy (111,8,8),
  window_centers.npy (111,).
- activity_estimates.npz and observed_signals.npz for independent reproduction.
- reference_adjacency.npz: static shared memberships, samplewise shared active
  states, reference at window centers and window-mean reference occupancy.
- static_adjacency.png, dynamic_snapshots.png, connectivity_over_time.png,
  and optional dynamic_adjacency.gif.
- parameters.json, comparison.json and report.md.

References are constructed only after estimation. The static reference is
binary shared membership; dynamic references represent sharing a currently
active state. These definitions differ from windowed coactivity Jaccard, so
reported off-diagonal mean absolute differences are descriptive, not state
recovery accuracy.

Default findings: static Jaccard is 1.0 for channels 1–2 and 4–5, 0.5 for
3–7, and 0.346681 for 1–4. During simultaneous A/B activity, dynamic 1–4
connectivity is 1.0 at centers 3.0 and 9.5 seconds despite zero reference
connectivity. Thus dynamic Jaccard does not distinguish different states that
are continuously active in the same window. One-second aggregation and
centered RMS smear transitions; even brief leaked activity can produce a
score of 1 for identical channels in a mostly inactive window. Functional
coactivity is neither anatomical connectivity nor uniquely recovered latent
state structure.

Run all current regression tests:

```powershell
.\.signal-venv\Scripts\python.exe -m unittest test_signals test_overlapping_signals test_state_detection test_noise_experiment test_adaptive_detection test_connectivity
```

## Research meeting figures

Regenerate the two 16:9 figures directly from saved noiseless observations and
saved detection outputs, without running simulation or detection:

```powershell
.\.signal-venv\Scripts\python.exe plot_presentation_figures.py
```

The script reads output/overlapping_signals.npz,
output/detection_v1/discovered_states.npz and its evaluation.json. It uses the
existing post-detection matching procedure to order the saved discoveries,
retaining their original D IDs and exact sample-level activation intervals.
No states are recreated from ground truth and no detector settings change.

Outputs in presentation_figures:
- 01_synthetic_signals.png and 01_synthetic_signals.pdf.
- 02_state_detection.png and 02_state_detection.pdf.

PNGs are 4800 x 2700 pixels at 300 DPI. PDFs are single-page 16 x 9 inch
vector figures with embedded TrueType fonts and no raster image objects.
A shared color scheme uses blue for A, orange for B and green for C;
channel 3 is labelled A + C. Membership fill indicates binary membership,
with the same state colors in both reference and detected matrices.
provenance.json records input hashes, post-detection matching and export
settings; input artifacts are checked unchanged after plotting. Optional
--signals, --detections, --evaluation and --output-dir arguments allow explicit
input/output paths. The default presentation layout expects the original
8-channel recording with three fully matched states and refuses to silently
hide unmatched states.

## White-noise presentation figures

Run `.\.signal-venv\Scripts\python.exe plot_noise_presentation_figures.py`
to generate presentation_figures/noise_comparison. It uses one standard-normal
noise realization (seed 20261015), scaled to sigma 0.05, 0.10, 0.15 and 0.20,
with the original clean signals. Adaptive and factorization settings are loaded
from output/adaptive_comparison/parameters.json and stay fixed across levels.
All displayed discoveries come from separate real inferences; ground truth is
used only afterward for display matching and evaluation. The original noiseless
presentation figures are preserved.

Two 16:9 overview figures are saved as 300 DPI PNG and vector PDF, alongside
noise_comparison_metrics.csv, replayable data and provenance. See the output
folder's README.md for method, seed and results. These figures illustrate one
shared noise realization, not repeated-run averages.

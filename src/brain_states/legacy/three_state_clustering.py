

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Simulation parameters
# ==========================================

fs = 250               # Sampling frequency (Hz)
duration = 12           # Total duration (seconds)
n_signals = 5           # Number of brain signals
noise_std = 0.3         # White noise standard deviation
spacing = 3             # Vertical spacing between signals

rng = np.random.default_rng(42)

# Time vector
t = np.arange(0, duration, 1 / fs)
n_samples = len(t)

# ==========================================
# 2. Define known brain states
# ==========================================

# State 1: 0-4 seconds, low frequency
# State 2: 4-8 seconds, high frequency
# State 3: 8-12 seconds, increased amplitude

true_states = np.zeros(n_samples, dtype=int)

true_states[t < 4] = 1
true_states[(t >= 4) & (t < 8)] = 2
true_states[t >= 8] = 3

frequencies = {
    1: 2.0,
    2: 8.0,
    3: 3.0
}

amplitudes = {
    1: 1.0,
    2: 1.0,
    3: 1.8
}

# ==========================================
# 3. Generate synthetic brain signals
# ==========================================

signals = np.zeros((n_signals, n_samples))

for i in range(n_signals):

    # Different phase for each signal
    phase = rng.uniform(0, 2 * np.pi)

    # Continuous oscillation phase
    instantaneous_frequency = np.array(
        [frequencies[state] for state in true_states]
    )

    oscillation_phase = (
        2 * np.pi *
        np.cumsum(instantaneous_frequency) / fs
        + phase
    )

    instantaneous_amplitude = np.array(
        [amplitudes[state] for state in true_states]
    )

    sine = instantaneous_amplitude * np.sin(
        oscillation_phase
    )

    noise = rng.normal(
        0, noise_std, size=n_samples
    )

    signals[i] = sine + noise

# ==========================================
# 4. Plot signals and known states
# ==========================================

fig, ax = plt.subplots(figsize=(14, 7))

state_colors = {
    1: "lightblue",
    2: "lightgreen",
    3: "mistyrose"
}

# State background shading
for state, start, end in [
    (1, 0, 4),
    (2, 4, 8),
    (3, 8, 12)
]:
    ax.axvspan(
        start, end,
        color=state_colors[state],
        alpha=0.35
    )

# Plot stacked signals
for i in range(n_signals):

    offset = i * spacing

    ax.plot(
        t,
        signals[i] + offset,
        color="black",
        linewidth=0.7
    )

# State boundaries
for boundary in [4, 8]:
    ax.axvline(
        boundary,
        color="red",
        linestyle="--",
        linewidth=1.2
    )

ax.set_yticks(np.arange(n_signals) * spacing)
ax.set_yticklabels(np.arange(1, n_signals + 1))

ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal number")
ax.set_title("Synthetic brain activity with known states")
ax.set_xlim(0, duration)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Sliding-window parameters
# ==========================================

window_sec = 1.0
step_sec = 0.1

window_size = int(window_sec * fs)
step_size = int(step_sec * fs)

features = []
window_centers = []

# ==========================================
# 2. Extract features from each window
# ==========================================

for start in range(
    0, signals.shape[1] - window_size + 1, step_size
):

    end = start + window_size
    window = signals[:, start:end]

    # Root mean square amplitude per channel
    rms = np.sqrt(np.mean(window**2, axis=1))

    # Power spectral density
    freqs, psd = welch(
        window,
        fs=fs,
        nperseg=window_size,
        axis=1
    )

    # Average spectral power in two frequency bands
    low_band = (freqs >= 1) & (freqs <= 4)
    high_band = (freqs >= 6) & (freqs <= 10)

    low_power = np.mean(psd[:, low_band], axis=1)
    high_power = np.mean(psd[:, high_band], axis=1)

    # Combine features from all channels
    feature_vector = np.concatenate([
        rms,
        low_power,
        high_power
    ])

    features.append(feature_vector)
    window_centers.append(
        (start + window_size / 2) / fs
    )

features = np.array(features)
window_centers = np.array(window_centers)

# ==========================================
# 3. Normalize and cluster windows
# ==========================================

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

predicted_states = kmeans.fit_predict(features_scaled)

# ==========================================
# 4. Temporal smoothing
# ==========================================

# Median filter removes short isolated changes
smoothed_states = median_filter(
    predicted_states,
    size=5,
    mode="nearest"
)

# ==========================================
# 5. Match cluster IDs to true state IDs
# ==========================================

# Cluster numbers are arbitrary.
# Align them with true labels for evaluation.

true_window_states = true_states[
    np.minimum(
        (window_centers * fs).astype(int),
        len(true_states) - 1
    )
]

cm = confusion_matrix(
    true_window_states,
    smoothed_states,
    labels=[0, 1, 2, 3]
)[1:, :3]

row_ind, col_ind = linear_sum_assignment(-cm)

mapping = {
    col: row + 1
    for row, col in zip(row_ind, col_ind)
}

aligned_states = np.array([
    mapping[s] for s in smoothed_states
])

accuracy = np.mean(
    aligned_states == true_window_states
)

print(f"Window-level accuracy: {accuracy:.2%}")

# ==========================================
# 6. Visualize the results
# ==========================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(14, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1]}
)

# Original signals
spacing = 3

for i in range(signals.shape[0]):
    axes[0].plot(
        t,
        signals[i] + i * spacing,
        color="black",
        linewidth=0.6
    )

axes[0].set_ylabel("Signal")
axes[0].set_title("Synthetic brain activity")

# True states
axes[1].step(
    t,
    true_states,
    where="post",
    color="blue"
)
axes[1].set_ylabel("True state")
axes[1].set_yticks([1, 2, 3])

# Predicted states
axes[2].step(
    window_centers,
    aligned_states,
    where="mid",
    color="red"
)
axes[2].set_ylabel("Predicted")
axes[2].set_yticks([1, 2, 3])
axes[2].set_xlabel("Time (s)")

for ax in axes:
    ax.set_xlim(0, duration)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

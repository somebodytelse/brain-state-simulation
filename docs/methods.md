# Methods and matrix conventions

## Synthetic state structure

At 200 Hz, a 12-second recording has 2,400 samples, covering `[0,12)` with a
last timestamp of 11.995 seconds. Each active state contributes
`amplitude * sin(2*pi*frequency*time + phase)` using global recording time.

| State | Channels (one-based) | Half-open intervals (s) | Frequency |
|---|---|---|---|
| A | 1, 2, 3 | [1,4), [8,10) | 2 Hz |
| B | 4, 5, 6 | [2,6), [9,12) | 4 Hz |
| C | 3, 7, 8 | [5,9) | 3 Hz |

Channel 3 sums A and C when both are ON. Wave cancellation can occur; activity
labels therefore describe participation, not nonzero sample amplitude.
Optional noise is independent `N(0,sigma^2)` at every channel/sample. Sigma is
a standard deviation, not SNR in dB. Clean values and truth are preserved.

## Observation-only activity and factorization

`signals X`: C by T observed values. `channel_activity`: C by T binary values.
The requested RMS window is rounded to an odd sample count, 21 at 200 Hz.
It is centered, noncausal and uses edge padding, spreading transitions by about
half a window. The fixed threshold is 0.05. Adaptive background is a channel's
20th-percentile RMS; threshold is `max(0.05, 1.5*background)`. Enough low-activity
time is assumed. Both weak activity and overlapping background distributions
can invalidate that assumption.

`W` is C by K binary membership and `H` is K by T binary state activation.
Boolean reconstruction is `R[c,t] = OR_k (W[c,k] AND H[k,t])`.
The original objective is:

`count(channel_activity != R) + fs*(0.5*K + 0.02*count(W))`.

Enumerate all channel subsets of size at least two. Their activation is the
AND of member-channel activity; discard runs shorter than 0.2 s. Greedily add
positive-gain factors, include recurrences and remove newly redundant factors
if the objective improves. There is no supplied true count. Enumeration is
limited to 16 channels and is not a global optimizer. Detector and factorizer
inputs exclude true memberships, true activation intervals and known noise.

## Evaluation

Activity scores are micro-averaged over channel-samples. State matching uses
exact maximum-weight one-to-one assignment with pair score equal to the mean
of temporal IoU and channel-membership Jaccard. Zero-score pairs are unmatched.
For aggregates, sum matched scores and divide by `max(true_count, found_count)`;
unmatched states contribute zero. Matched-only means and unmatched IDs are
also reported. The assignment's smaller side is limited to 16 states.

Reconstruction error against estimated activity and error against true activity
are distinct diagnostics. Correct state count is not sufficient evidence of
correct memberships or timelines. Matching is evaluation, never initialization.

## Functional adjacency

`static_adjacency` is C by C; dynamic adjacency is windows by C by C. Pairwise
Jaccard is active intersection/union. Empty union and the diagonal give zero.
The default 1-second windows advance by 0.1 s, yielding 111 centers from 0.5
to 11.5 s. They are fixed-grid full windows, independent of true transitions.
RMS blur and window aggregation both smooth time; adjacent windows are not
independent observations.

Reference adjacency is constructed afterward from shared memberships. Dynamic
references indicate a shared currently active state; window-mean reference
occupancy and instantaneous center references are separate arrays. They differ
conceptually from Jaccard, so MAE comparisons are descriptive, not recovery
accuracy. Independent simultaneously active channels can have Jaccard 1.

## Scope

These are exploratory synthetic experiments. Real noise, anatomy, physiological
interpretability and unique decomposition are not established by these results.

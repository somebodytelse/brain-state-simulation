"""Post-detection evaluation; never imported by the detector."""
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray


def jaccard(a: NDArray, b: NDArray) -> float:
    """Intersection-over-union, with two empty sets treated as identical."""
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def match_states(scores: NDArray) -> list[tuple[int, int]]:
    """Exact maximum-weight one-to-one matching by bitmask dynamic programming.

    Zero-score pairs are left unmatched. Intended for small experiments.
    """
    transposed = scores.shape[0] < scores.shape[1]
    matrix = scores.T if transposed else scores
    rows, columns = matrix.shape
    if columns > 16:
        raise ValueError('Evaluation matching supports at most 16 states on the smaller side.')

    @lru_cache(None)
    def solve(row: int, used: int) -> tuple[float, tuple]:
        if row == rows:
            return 0.0, ()
        best = solve(row + 1, used)
        for col in range(columns):
            if not used & (1 << col) and matrix[row, col] > 0:
                value, pairs = solve(row + 1, used | (1 << col))
                option = (value + float(matrix[row, col]), ((row, col),) + pairs)
                if option[0] > best[0]:
                    best = option
        return best
    pairs = solve(0, 0)[1]
    return sorted((col, row) if transposed else (row, col) for row, col in pairs)


def evaluate(result: dict, truth: dict) -> dict:
    """Compute activity scores, label-invariant state comparison and residuals."""
    estimated = result['channel_activity'].astype(bool)
    actual = truth['channel_activations'].astype(bool)
    tp = int(np.count_nonzero(estimated & actual))
    fp = int(np.count_nonzero(estimated & ~actual))
    fn = int(np.count_nonzero(~estimated & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    true_w, true_h = truth['state_memberships'].astype(bool), truth['state_activations'].astype(bool)
    w, h = result['memberships'], result['state_activations']
    temporal = np.array([[jaccard(a, b) for b in h] for a in true_h]).reshape(len(true_h), len(h))
    membership = np.array([[jaccard(a, b) for b in w.T] for a in true_w.T]).reshape(true_w.shape[1], w.shape[1])
    pairs = match_states((temporal + membership) / 2)
    matched = [dict(true_state=i + 1, discovered_state=j + 1,
                    temporal_iou=float(temporal[i, j]), membership_jaccard=float(membership[i, j]))
               for i, j in pairs]
    denominator = max(len(true_h), len(h), 1)
    return dict(channel_activity=dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                true_positives=tp, false_positives=fp, false_negatives=fn),
                true_state_count=len(true_h), discovered_state_count=len(h), matches=matched,
                unmatched_true_states=[i+1 for i in range(len(true_h)) if i not in {a for a, _ in pairs}],
                unmatched_discovered_states=[j+1 for j in range(len(h)) if j not in {b for _, b in pairs}],
                temporal_iou_with_unmatched_zero=sum(m['temporal_iou'] for m in matched)/denominator,
                membership_jaccard_with_unmatched_zero=sum(m['membership_jaccard'] for m in matched)/denominator,
                boolean_reconstruction_error=float(np.mean(estimated != result['reconstruction'])),
                boolean_reconstruction_mismatches=int(np.count_nonzero(estimated != result['reconstruction'])),
                reconstruction_error_against_ground_truth=float(np.mean(actual != result['reconstruction'])))

"""Ground-truth reference construction strictly separate from estimation."""
import numpy as np
from numpy.typing import NDArray


def membership_reference(memberships: NDArray) -> NDArray:
    """Connect channels if they share any state, independent of activation time."""
    w = np.asarray(memberships, dtype=np.int64)
    adjacency = w @ w.T > 0
    np.fill_diagonal(adjacency, False)
    return adjacency


def build_references(memberships: NDArray, state_activations: NDArray,
                     starts: NDArray, ends: NDArray) -> dict[str, NDArray]:
    """Construct samplewise shared-active-state links, then align for comparison.

    Center reference is instantaneous; window_fraction is the fraction of
    samples sharing an active state in each SAME estimation window. Neither
    equals coactivity Jaccard in general; comparison is descriptive, not a
    state-recovery accuracy score. This function runs after estimation.
    """
    w = np.asarray(memberships, dtype=np.int64)
    h = np.asarray(state_activations, dtype=np.int64)
    if w.ndim != 2 or h.ndim != 2 or w.shape[1] != h.shape[0]:
        raise ValueError('Memberships and state activations must have compatible shapes.')
    if not np.all((w == 0) | (w == 1)) or not np.all((h == 0) | (h == 1)):
        raise ValueError('Reference membership and activation matrices must be binary.')
    samplewise = np.einsum('ik,kt,jk->tij', w, h, w, optimize=True) > 0
    indices = np.arange(w.shape[0])
    samplewise[:, indices, indices] = False
    center_indices = (starts+ends)//2
    fractions = np.stack([samplewise[a:b].mean(axis=0) for a, b in zip(starts, ends)])
    return dict(static_reference=membership_reference(w), samplewise_reference=samplewise,
                center_reference=samplewise[center_indices], window_fraction_reference=fractions)


def compare_connectivity(estimated: dict, references: dict) -> dict:
    """Compare off-diagonal entries without treating different definitions as truth recovery."""
    channels = estimated['static_adjacency'].shape[0]
    upper = np.triu_indices(channels, 1)
    if not len(upper[0]):
        return dict(note='No inter-channel pairs available.')
    static = estimated['static_adjacency'][upper]
    dynamic = estimated['dynamic_adjacency'][:, upper[0], upper[1]]
    static_ref = references['static_reference'][upper]
    center_ref = references['center_reference'][:, upper[0], upper[1]]
    fraction_ref = references['window_fraction_reference'][:, upper[0], upper[1]]
    return dict(static_mae_vs_membership_reference=float(np.mean(np.abs(static-static_ref))),
                dynamic_mae_vs_center_reference=float(np.mean(np.abs(dynamic-center_ref))),
                dynamic_mae_vs_window_fraction_reference=float(np.mean(np.abs(dynamic-fraction_ref))),
                structural_nonedge_pairs=int(np.count_nonzero(~static_ref)),
                positive_jaccard_on_structural_nonedges=int(np.count_nonzero((static > 0) & ~static_ref)),
                note='Different connectivity definitions. MAE is descriptive, not state detection accuracy; pairs are counted once with the diagonal excluded.')

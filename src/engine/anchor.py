"""Shared, model-independent latent ground truth for the click simulator.

The original runner derived ``U_star``/``V_star`` from the very model being
evaluated, so each recommender generated clicks from its own geometry. That is
self-confirming and makes cross-model comparisons meaningless. This module
builds one deterministic anchor from the baseline interaction matrix and every
adapter/seed reuses it.
"""
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def build_shared_anchor(
    base_csr: sp.csr_matrix,
    dim: int = 32,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic truncated-SVD factorization of ``D_0``.

    The factorization satisfies ``U @ V.T ~= D_0`` in the least-squares sense,
    so dot products are on the same scale as binary interactions. Signs are
    canonicalised per component to keep the anchor stable across runs and
    library versions.

    Args:
        base_csr: Baseline CSR interaction matrix of shape (|U|, |I|).
        dim: Maximum latent dimensionality.
        seed: Random state for the randomized SVD.

    Returns:
        U_anchor: (|U|, d) float32 user factors.
        V_anchor: (|I|, d) float32 item factors.
    """
    n_users, n_items = base_csr.shape
    if base_csr.nnz == 0:
        raise ValueError("Cannot build a shared anchor from an empty interaction matrix")

    effective_dim = int(min(dim, n_users, n_items))
    if effective_dim < 1:
        raise ValueError("Anchor dimension must be at least 1")

    matrix = base_csr.astype(np.float32).tocsr()
    U, S, Vt = randomized_svd(
        matrix,
        n_components=effective_dim,
        n_iter=7,
        random_state=seed,
    )

    # Canonical sign convention: largest-magnitude entry of each right singular
    # vector is positive. Without this, equivalent factorizations can differ by
    # a global sign and produce different click traces.
    for k in range(effective_dim):
        pivot = int(np.argmax(np.abs(Vt[k])))
        if Vt[k, pivot] < 0:
            U[:, k] *= -1.0
            Vt[k, :] *= -1.0

    sqrt_s = np.sqrt(np.maximum(S, 0.0)).astype(np.float32)
    U_anchor = (U * sqrt_s[None, :]).astype(np.float32)
    V_anchor = (Vt.T * sqrt_s[None, :]).astype(np.float32)

    assert U_anchor.shape == (n_users, effective_dim)
    assert V_anchor.shape == (n_items, effective_dim)
    assert np.isfinite(U_anchor).all() and np.isfinite(V_anchor).all()
    return U_anchor, V_anchor

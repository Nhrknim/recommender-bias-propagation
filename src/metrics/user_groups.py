import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple

def partition_user_cohorts(
    base_matrix: sp.csr_matrix, 
    long_tail_mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Partitions users into 3 equal cohorts (U_Pop, U_Div, U_Niche) based on
    baseline long-tail interaction ratio gamma_u.

    Args:
        base_matrix: Initial sparse interaction matrix D_0 of shape (|U|, |I|).
        long_tail_mask: 1D boolean array of shape (|I|,) indicating long-tail items.

    Returns:
        cohorts: Dict mapping cohort names ('pop', 'div', 'niche') to 1D user index arrays.
    """
    n_users, n_items = base_matrix.shape
    gamma_u = np.zeros(n_users, dtype=np.float32)

    for u in range(n_users):
        user_items = base_matrix[u].indices
        if len(user_items) > 0:
            tail_count = np.sum(long_tail_mask[user_items])
            gamma_u[u] = float(tail_count / len(user_items))

    # Sort users by long-tail preference ratio gamma_u ascending
    sorted_user_indices = np.argsort(gamma_u)
    third = n_users // 3

    pop_users = sorted_user_indices[:third]
    div_users = sorted_user_indices[third: 2 * third]
    niche_users = sorted_user_indices[2 * third:]

    return {
        "pop": pop_users.astype(np.int32),
        "div": div_users.astype(np.int32),
        "niche": niche_users.astype(np.int32)
    }


def compute_kl_divergence_drift(
    base_user_dist: np.ndarray, 
    current_rec_dist: np.ndarray, 
    epsilon: float = 1e-9
) -> np.ndarray:
    """
    Computes category taste drift D_KL(P_u^{(0)} || Q_u^{(t)}) per user.

    D_KL = sum_c P_u^{(0)}(c) * log( P_u^{(0)}(c) / Q_u^{(t)}(c) )

    Args:
        base_user_dist: Baseline category probability distribution P_u^{(0)}, shape (N_users, C).
        current_rec_dist: Generation t recommendation distribution Q_u^{(t)}, shape (N_users, C).
        epsilon: Numerical cushion to prevent log(0) and division by zero.

    Returns:
        kl_drifts: 1D array of shape (N_users,) containing KL divergence per user.
    """
    assert base_user_dist.shape == current_rec_dist.shape, "Category distributions shape misaligned"

    # Add epsilon cushion and normalize probability distributions
    p_smooth = base_user_dist + epsilon
    p_smooth = p_smooth / np.sum(p_smooth, axis=1, keepdims=True)

    q_smooth = current_rec_dist + epsilon
    q_smooth = q_smooth / np.sum(q_smooth, axis=1, keepdims=True)

    # KL Divergence element-wise evaluation
    kl_terms = p_smooth * np.log(p_smooth / q_smooth)
    kl_drifts = np.sum(kl_terms, axis=1)

    # Non-negativity assertion (D_KL >= 0.0)
    kl_drifts = np.maximum(0.0, kl_drifts)
    assert not np.isnan(kl_drifts).any(), "NaN detected in KL divergence calculation"

    return kl_drifts.astype(np.float32)


def compute_subgroup_utility_gap(ndcg_pop: float, ndcg_niche: float) -> float:
    """
    Computes Subgroup Utility Gap: Delta NDCG@K = NDCG(U_Pop) - NDCG(U_Niche).

    Args:
        ndcg_pop: Recommendation utility NDCG@K for Popularity cohort.
        ndcg_niche: Recommendation utility NDCG@K for Niche cohort.

    Returns:
        gap: Subgroup utility gap value.
    """
    return float(ndcg_pop - ndcg_niche)

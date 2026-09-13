import numpy as np
import random
from typing import Tuple
from src.engine.exposure import compute_position_exposure
from src.engine.preference import compute_intrinsic_preference
from src.engine.conformity import compute_social_conformity

def simulate_user_clicks(
    recs: np.ndarray, 
    U_star: np.ndarray, 
    V_star: np.ndarray, 
    pop_counts: np.ndarray, 
    alpha: float = 0.5, 
    tau: float = 1.0, 
    poisson_lambda: float = 3.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates user browsing, position decay, intrinsic relevance matching,
    social conformity, and Poisson session budget truncations.

    Args:
        recs: 2D integer array of recommended item indices, shape (N_active, K).
        U_star: Ground-truth user factor anchor matrix, shape (|U|, d).
        V_star: Ground-truth item factor anchor matrix, shape (|I|, d).
        pop_counts: 1D item interaction count array, shape (|I|,).
        alpha: Weight balancing intrinsic relevance vs social conformity (0.0 <= alpha <= 1.0).
        tau: Temperature scaling for intrinsic preference.
        poisson_lambda: Mean click budget parameter lambda for Poisson distribution.
        seed: Deterministic random seed for reproducibility.

    Returns:
        clicked_u: 1D integer array of active user indices who clicked.
        clicked_i: 1D integer array of global item indices clicked.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Popularity weight alpha must be in range [0.0, 1.0], got {alpha}")

    # Set deterministic random state
    np.random.seed(seed)
    random.seed(seed)

    n_active, k = recs.shape

    # 1. Position Exposure Discount: shape (K,) -> broadcast to (N_active, K)
    p_exposed = compute_position_exposure(k)

    # 2. Intrinsic Preference & Conformity: shapes (N_active, K)
    p_relevance = compute_intrinsic_preference(U_star, V_star, recs, tau=tau)
    p_conformity = compute_social_conformity(pop_counts, recs)

    # 3. Joint Click Probability Matrix: shape (N_active, K)
    p_click = p_exposed * ((1.0 - alpha) * p_relevance + alpha * p_conformity)

    # 4. Vectorized Bernoulli Sampling across all (user, rank) pairs
    rand_vals = np.random.uniform(size=(n_active, k))
    clicked_mask = rand_vals < p_click

    # 5. Poisson Session Budgeting per active user
    budgets = np.random.poisson(lam=poisson_lambda, size=n_active)

    clicked_u, clicked_i = [], []
    for u_idx in range(n_active):
        items_clicked = recs[u_idx, clicked_mask[u_idx]]
        # Enforce maximum session budget truncation
        if len(items_clicked) > budgets[u_idx]:
            items_clicked = items_clicked[:budgets[u_idx]]
            
        clicked_u.extend([u_idx] * len(items_clicked))
        clicked_i.extend(items_clicked)

    u_arr = np.array(clicked_u, dtype=np.int32)
    i_arr = np.array(clicked_i, dtype=np.int32)

    # Invariant assertions
    assert len(u_arr) == len(i_arr), "User and Item click arrays length misaligned"
    assert not np.isnan(u_arr).any(), "NaN detected in clicked user indices"
    assert not np.isnan(i_arr).any(), "NaN detected in clicked item indices"

    return u_arr, i_arr

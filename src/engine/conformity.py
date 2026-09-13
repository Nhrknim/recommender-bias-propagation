import numpy as np

def compute_social_conformity(
    pop_counts: np.ndarray, 
    recs: np.ndarray
) -> np.ndarray:
    """
    Computes sub-linear logarithmic social conformity probabilities.

    P(Conformity | i, t) = log(1 + f_i) / log(1 + max_j f_j)

    Args:
        pop_counts: 1D array of cumulative item interaction counts, shape (|I|,).
        recs: 2D integer array of recommended item indices, shape (N_users, K).

    Returns:
        p_conformity: 2D float array of shape (N_users, K) with conformity probabilities.
    """
    n_active, k = recs.shape
    max_pop = np.max(pop_counts) if len(pop_counts) > 0 else 0.0

    if max_pop == 0.0:
        # Initial baseline state with zero interactions across catalog
        p_conformity = np.zeros((n_active, k), dtype=np.float32)
    else:
        # Sub-linear logarithmic scaling
        numerator = np.log1p(pop_counts[recs].astype(np.float32))
        denominator = np.log1p(float(max_pop))
        p_conformity = np.clip(numerator / denominator, 0.0, 1.0)

    # Invariant assertions
    assert p_conformity.shape == (n_active, k), f"Conformity shape misaligned: {p_conformity.shape}"
    assert np.all(p_conformity >= 0.0) and np.all(p_conformity <= 1.0), "Conformity probabilities out of bounds [0, 1]"
    assert not np.isnan(p_conformity).any(), "NaN detected in social conformity calculation"

    return p_conformity.astype(np.float32)

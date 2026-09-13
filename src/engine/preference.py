import numpy as np

def compute_intrinsic_preference(
    U_star: np.ndarray, 
    V_star: np.ndarray, 
    recs: np.ndarray, 
    tau: float = 1.0
) -> np.ndarray:
    """
    Computes temperature-scaled intrinsic user-item relevance probabilities.

    P(Relevance | u, i) = sigmoid( clip( (u_star^T v_star) / (tau * sqrt(d)), -15.0, 15.0 ) )

    Args:
        U_star: Ground-truth user embedding matrix of shape (|U|, d).
        V_star: Ground-truth item embedding matrix of shape (|I|, d).
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        tau: Temperature scaling parameter (tau > 0).

    Returns:
        p_relevance: 2D float array of shape (N_users, K) with intrinsic relevance probabilities.
    """
    if tau <= 0.0:
        raise ValueError(f"Temperature parameter tau must be positive, got {tau}")

    n_active, k = recs.shape
    d = U_star.shape[1]

    assert V_star.shape[1] == d, f"Embedding dimension mismatch: U_star {d} vs V_star {V_star.shape[1]}"

    # Vectorized batch indexing across active users and recommended items
    user_indices = np.repeat(np.arange(n_active, dtype=np.int32), k)
    item_indices = recs.flatten()

    u_vecs = U_star[user_indices]
    i_vecs = V_star[item_indices]

    # Batch dot product
    dot_products = np.sum(u_vecs * i_vecs, axis=1).reshape(n_active, k)

    # Temperature scaling & numerical stability logit clipping
    scaled_logits = np.clip(dot_products / (tau * np.sqrt(d)), -15.0, 15.0)

    # Numerically stable sigmoid calculation
    p_relevance = 1.0 / (1.0 + np.exp(-scaled_logits))

    # Invariant assertions
    assert p_relevance.shape == (n_active, k), f"Relevance shape misaligned: {p_relevance.shape}"
    assert np.all(p_relevance >= 0.0) and np.all(p_relevance <= 1.0), "Relevance probabilities out of bounds [0, 1]"
    assert not np.isnan(p_relevance).any(), "NaN detected in intrinsic preference calculation"

    return p_relevance.astype(np.float32)

import numpy as np

def compute_position_exposure(k: int) -> np.ndarray:
    """
    Computes position-discounted exposure probabilities for ranks 0..K-1.

    P(Exposed | r) = 1.0 / log2(r + 2)

    Args:
        k: Number of recommendation list positions K.

    Returns:
        p_exposed: 1D NumPy array of shape (K,) containing exposure decay probabilities.
    """
    if k <= 0:
        raise ValueError(f"Recommendation list length k must be positive, got {k}")

    ranks = np.arange(k, dtype=np.float32)
    p_exposed = 1.0 / np.log2(ranks + 2.0)

    # Invariant assertions
    assert p_exposed.shape == (k,), f"Exposure array shape misaligned: {p_exposed.shape}"
    assert np.all(p_exposed > 0.0) and np.all(p_exposed <= 1.0), "Exposure probabilities out of bounds (0, 1]"
    assert p_exposed[0] == 1.0, "Top rank r=0 must have exposure probability 1.0"

    return p_exposed

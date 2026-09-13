import numpy as np

def compute_exposure_gini(recs: np.ndarray, n_items: int) -> float:
    """
    Computes Exposure Gini coefficient across catalog item recommendation counts.

    G = (|I| + 1) / |I| - (2 * sum_{k=1}^{|I|} (|I| - k + 1) c_{(k)}) / (|I| * sum_{i=1}^{|I|} c_i)

    Args:
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        n_items: Total count of items in the catalog |I|.

    Returns:
        gini: Exposure Gini coefficient in range [0.0, 1.0].
    """
    if n_items <= 0:
        raise ValueError(f"Total catalog items n_items must be positive, got {n_items}")

    # Count exposure frequencies per catalog item
    item_counts = np.bincount(recs.flatten(), minlength=n_items).astype(np.float64)
    total_exposure = np.sum(item_counts)

    if total_exposure == 0.0:
        return 0.0

    # Sort exposure counts in ascending order c_{(1)} <= c_{(2)} <= ... <= c_{(|I|)}
    sorted_counts = np.sort(item_counts)
    i_arr = np.arange(1, n_items + 1, dtype=np.float64)

    # Formula evaluation
    numerator = 2.0 * np.sum((n_items - i_arr + 1.0) * sorted_counts)
    denominator = n_items * total_exposure

    gini = float((n_items + 1.0) / n_items - numerator / denominator)
    gini = float(np.clip(gini, 0.0, 1.0))

    assert 0.0 <= gini <= 1.0, f"Exposure Gini out of bounds: {gini}"
    return gini


def compute_catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """
    Computes Catalog Coverage@K metric.

    Coverage@K = |Unique items recommended across all users| / |I|

    Args:
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        n_items: Total count of items in catalog |I|.

    Returns:
        coverage: Ratio of catalog items recommended in range [0.0, 1.0].
    """
    if n_items <= 0:
        raise ValueError(f"Total catalog items n_items must be positive, got {n_items}")

    unique_items = np.unique(recs)
    coverage = float(len(unique_items) / n_items)
    coverage = float(np.clip(coverage, 0.0, 1.0))

    assert 0.0 <= coverage <= 1.0, f"Catalog coverage out of bounds: {coverage}"
    return coverage


def compute_aplt(recs: np.ndarray, long_tail_mask: np.ndarray) -> float:
    r"""
    Computes Average Percentage of Long-Tail@K (APLT@K).

    APLT@K = (1 / |U|) * sum_u ( |TopK(u) \cap LongTail| / K )

    Args:
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        long_tail_mask: 1D boolean array of shape (|I|,) where True indicates a long-tail item.

    Returns:
        aplt: Average percentage of long-tail items in recommendation lists [0.0, 1.0].
    """
    n_users, k = recs.shape
    if n_users == 0 or k == 0:
        return 0.0

    # Vectorized boolean lookup of long-tail items in recs
    is_long_tail = long_tail_mask[recs]
    user_tail_counts = np.sum(is_long_tail, axis=1)

    aplt = float(np.mean(user_tail_counts / float(k)))
    aplt = float(np.clip(aplt, 0.0, 1.0))

    assert 0.0 <= aplt <= 1.0, f"APLT@K out of bounds: {aplt}"
    return aplt

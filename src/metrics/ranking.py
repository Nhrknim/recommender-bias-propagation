import numpy as np
import scipy.sparse as sp
from typing import Any, Optional

def compute_precision_at_k(recs: np.ndarray, test_matrix: sp.csr_matrix, k: int) -> float:
    """
    Computes Precision@K evaluated against held-out test matrix D_test.

    Args:
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        test_matrix: Sparse CSR ground-truth test matrix of shape (|U|, |I|).
        k: Recommendation list cut-off rank.

    Returns:
        precision: Mean Precision@K score across users [0.0, 1.0].
    """
    n_users = recs.shape[0]
    if n_users == 0 or k == 0:
        return 0.0

    precisions = []
    for u in range(n_users):
        test_items = set(test_matrix[u].indices)
        if len(test_items) == 0:
            continue
        hits = sum(1 for item in recs[u, :k] if item in test_items)
        precisions.append(hits / float(k))

    return float(np.mean(precisions)) if len(precisions) > 0 else 0.0


def compute_recall_at_k(recs: np.ndarray, test_matrix: sp.csr_matrix, k: int) -> float:
    """
    Computes Recall@K evaluated against held-out test matrix D_test.

    Args:
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        test_matrix: Sparse CSR ground-truth test matrix of shape (|U|, |I|).
        k: Recommendation list cut-off rank.

    Returns:
        recall: Mean Recall@K score across users [0.0, 1.0].
    """
    n_users = recs.shape[0]
    if n_users == 0 or k == 0:
        return 0.0

    recalls = []
    for u in range(n_users):
        test_items = set(test_matrix[u].indices)
        if len(test_items) == 0:
            continue
        hits = sum(1 for item in recs[u, :k] if item in test_items)
        recalls.append(hits / float(len(test_items)))

    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


def compute_ndcg_at_k(recs: np.ndarray, test_matrix: sp.csr_matrix, k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG@K = DCG@K / IDCG@K

    Args:
        recs: 2D integer array of recommended item indices, shape (N_users, K).
        test_matrix: Sparse CSR ground-truth test matrix of shape (|U|, |I|).
        k: Recommendation list cut-off rank.

    Returns:
        ndcg: Mean NDCG@K score across users [0.0, 1.0].
    """
    n_users = recs.shape[0]
    if n_users == 0 or k == 0:
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcgs = []

    for u in range(n_users):
        test_items = set(test_matrix[u].indices)
        n_rel = len(test_items)
        if n_rel == 0:
            continue

        # DCG@K
        hits = np.array([1.0 if recs[u, r] in test_items else 0.0 for r in range(k)], dtype=np.float32)
        dcg = float(np.sum(hits * discounts))

        # Ideal DCG@K
        idcg_hits = np.zeros(k, dtype=np.float32)
        idcg_hits[:min(n_rel, k)] = 1.0
        idcg = float(np.sum(idcg_hits * discounts))

        if idcg > 0.0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if len(ndcgs) > 0 else 0.0


def compute_vectorized_auc(
    score_matrix: Any,
    test_matrix: sp.csr_matrix,
    exclude_matrix: Optional[sp.csr_matrix] = None,
) -> float:
    """
    Computes tie-aware global Mann-Whitney Pairwise AUC.

    Positive items are the held-out test interactions. Negatives are every
    catalog item that is not a test positive and not present in
    ``exclude_matrix`` (typically the training matrix D_t). Tied scores receive
    average ranks, which is equivalent to awarding 0.5 credit per tied pair —
    an all-tied score vector therefore yields exactly 0.5, not 0.0.

    Args:
        score_matrix: Dense 2D array or sparse CSR matrix of full-catalog scores,
            shape (|U|, |I|). Must contain real scores, not a top-K truncation.
        test_matrix: Sparse CSR matrix of ground-truth test interactions.
        exclude_matrix: Optional sparse matrix of items to remove from the
            negative candidate pool (e.g. observed training interactions).

    Returns:
        auc: Mean tie-aware AUC across users with at least one positive and one
            negative candidate [0.0, 1.0].
    """
    from scipy.stats import rankdata

    n_users, n_items = score_matrix.shape
    if test_matrix.shape != (n_users, n_items):
        raise ValueError("score_matrix and test_matrix shapes must match")

    auc_scores = []
    is_sparse = sp.issparse(score_matrix)
    all_items = np.arange(n_items)

    for u in range(n_users):
        pos_items = test_matrix[u].indices
        n_pos = len(pos_items)
        if n_pos == 0 or n_pos == n_items:
            continue

        scores = score_matrix[u].toarray().ravel() if is_sparse else np.asarray(score_matrix[u]).ravel()

        if exclude_matrix is not None:
            excluded = exclude_matrix[u].indices
            keep = np.ones(n_items, dtype=bool)
            keep[excluded] = False
            # A test positive must never be excluded by upstream leakage.
            keep[pos_items] = True
            candidate_scores = scores[keep]
            pos_mask = np.isin(all_items[keep], pos_items)
        else:
            candidate_scores = scores
            pos_mask = np.zeros(n_items, dtype=bool)
            pos_mask[pos_items] = True

        n_candidates = len(candidate_scores)
        n_pos_c = int(pos_mask.sum())
        n_neg = n_candidates - n_pos_c
        if n_pos_c == 0 or n_neg <= 0:
            continue

        # Average ranks implement the tie correction: tied positives and
        # negatives each receive the mean rank of their tied block.
        ranks = rankdata(candidate_scores, method="average")
        rank_sum = float(np.sum(ranks[pos_mask]))
        auc_u = (rank_sum - n_pos_c * (n_pos_c + 1.0) / 2.0) / (n_pos_c * n_neg)
        auc_scores.append(float(np.clip(auc_u, 0.0, 1.0)))

    auc_val = float(np.mean(auc_scores)) if len(auc_scores) > 0 else 0.5
    auc_val = float(np.clip(auc_val, 0.0, 1.0))

    assert 0.0 <= auc_val <= 1.0, f"Vectorized AUC out of bounds: {auc_val}"
    return auc_val

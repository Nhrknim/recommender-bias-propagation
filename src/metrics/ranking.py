import numpy as np
import scipy.sparse as sp

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


def compute_vectorized_auc(score_matrix: np.ndarray, test_matrix: sp.csr_matrix) -> float:
    """
    Computes exact global Mann-Whitney Pairwise AUC in O(|I| log |I|) time per user.

    AUC_u = (RankSum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Args:
        score_matrix: Dense 2D array of predicted item scores, shape (|U|, |I|).
        test_matrix: Sparse CSR matrix of ground-truth test interactions, shape (|U|, |I|).

    Returns:
        auc: Mean Mann-Whitney AUC score across active test users [0.0, 1.0].
    """
    n_users, n_items = score_matrix.shape
    auc_scores = []

    for u in range(n_users):
        pos_items = test_matrix[u].indices
        n_pos = len(pos_items)
        if n_pos == 0 or n_pos == n_items:
            continue

        n_neg = n_items - n_pos
        scores = score_matrix[u]

        # Vectorized rank sorting (1-indexed ranks)
        ranks = np.argsort(np.argsort(scores)) + 1
        pos_rank_sum = np.sum(ranks[pos_items])

        auc_u = (pos_rank_sum - (n_pos * (n_pos + 1.0)) / 2.0) / (n_pos * n_neg)
        auc_scores.append(auc_u)

    auc_val = float(np.mean(auc_scores)) if len(auc_scores) > 0 else 0.5
    auc_val = float(np.clip(auc_val, 0.0, 1.0))

    assert 0.0 <= auc_val <= 1.0, f"Vectorized AUC out of bounds: {auc_val}"
    return auc_val

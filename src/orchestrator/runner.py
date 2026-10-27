import gc
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
import scipy.sparse as sp

from src.data.buffer import merge_and_deduplicate
from src.engine.anchor import build_shared_anchor
from src.engine.simulator import simulate_user_clicks
from src.orchestrator.seed_control import set_deterministic_seed
from src.metrics import (
    compute_exposure_gini,
    compute_catalog_coverage,
    compute_aplt,
    partition_user_cohorts,
    compute_kl_divergence_drift,
    compute_subgroup_utility_gap,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_vectorized_auc,
    compute_reflection_corrected_procrustes,
    compute_relative_norm_shift,
    compute_effective_rank,
)


def _compute_baseline_user_dist(
    base_csr: sp.csr_matrix,
    item_category_matrix: np.ndarray
) -> np.ndarray:
    """Vectorized baseline user preference distribution over categories."""
    n_categories = item_category_matrix.shape[1]
    user_cat_counts = base_csr.dot(item_category_matrix)
    row_sums = user_cat_counts.sum(axis=1, keepdims=True)
    return np.where(
        row_sums > 0,
        user_cat_counts / np.maximum(row_sums, 1e-9),
        np.full((1, n_categories), 1.0 / n_categories, dtype=np.float32),
    )


def _category_distribution_from_pairs(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
    item_category_matrix: np.ndarray,
    fallback: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-user category distributions from clicked (user, item) pairs."""
    n_categories = item_category_matrix.shape[1]
    if len(item_ids) == 0:
        return fallback.copy(), np.zeros(n_users, dtype=bool)

    click_matrix = sp.csr_matrix(
        (np.ones(len(item_ids), dtype=np.float32), (user_ids, item_ids)),
        shape=(n_users, n_items),
    )
    counts = np.asarray(click_matrix.dot(item_category_matrix), dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    dist = np.where(row_sums > 0, counts / np.maximum(row_sums, 1e-9), fallback)
    has_click = np.bincount(user_ids, minlength=n_users) > 0
    return dist, has_click


def _run_single_seed(
    seed: int,
    adapter_cls: Type[Any],
    kwargs: Dict[str, Any],
    base_csr: sp.csr_matrix,
    test_csr: sp.csr_matrix,
    T: int,
    k: int,
    alpha: float,
    tau: float,
    long_tail_mask: np.ndarray,
    cohorts: Dict[str, np.ndarray],
    base_user_dist: np.ndarray,
    item_category_matrix: np.ndarray,
    U_anchor: np.ndarray,
    V_anchor: np.ndarray,
) -> Dict[str, List[float]]:
    """Executes a single simulation trajectory for one seed."""
    set_deterministic_seed(seed)
    n_users, n_items = base_csr.shape
    n_categories = item_category_matrix.shape[1]
    uniform_dist = np.full((1, n_categories), 1.0 / n_categories, dtype=np.float32)

    D_t = base_csr.copy()
    active_users = np.arange(n_users, dtype=np.int32)

    model_adapter = adapter_cls(**kwargs)
    model_adapter.fit(D_t, seed=seed)

    # Geometry baseline: the model's own t=0 manifold. This is deliberately
    # separate from the shared click anchor (U_anchor/V_anchor), which is
    # model-independent and frozen for the whole run.
    _, V_model_0 = model_adapter.get_embeddings()

    seed_metrics: Dict[str, List[float]] = {
        "gini": [], "coverage": [], "aplt": [],
        "kl_drift": [], "rec_kl": [], "subgroup_gap": [],
        "precision": [], "recall": [], "ndcg": [], "auc": [],
        "r_norm": [], "r_norm_mean": [], "r_norm_valid_frac": [], "s_eff": []
    }

    for t in range(T):
        # 1. Recommendations
        recs = model_adapter.recommend(active_users, k=k, remove_seen=True)
        assert recs.shape == (n_users, k), f"Recommendation shape misaligned: {recs.shape}"

        # 2. Vectorized User Interaction Simulation (global user coordinates)
        pop_counts = np.asarray(D_t.sum(axis=0)).flatten()
        u_clicks, i_clicks = simulate_user_clicks(
            recs=recs,
            U_star=U_anchor,
            V_star=V_anchor,
            pop_counts=pop_counts,
            user_indices=active_users,
            alpha=alpha,
            tau=tau,
            seed=seed + t
        )

        # 3. Diagnostics
        gini = compute_exposure_gini(recs, n_items)
        cov = compute_catalog_coverage(recs, n_items)
        aplt = compute_aplt(recs, long_tail_mask)

        # Recommendation-category mix vs baseline user taste.
        safe_recs = np.clip(recs, 0, n_items - 1)
        u_recs_cat = item_category_matrix[safe_recs]
        u_recs_cat[recs < 0] = 0.0
        cat_sum = u_recs_cat.sum(axis=1)
        sum_val = cat_sum.sum(axis=1, keepdims=True)
        current_rec_dist = np.where(sum_val > 0, cat_sum / np.maximum(sum_val, 1e-9), uniform_dist)
        rec_kl = float(np.mean(compute_kl_divergence_drift(base_user_dist, current_rec_dist)))

        # Realized taste drift: what users actually clicked this generation.
        click_dist, has_click = _category_distribution_from_pairs(
            u_clicks, i_clicks, n_users, n_items, item_category_matrix, base_user_dist
        )
        click_kl = compute_kl_divergence_drift(base_user_dist, click_dist)
        mean_kl_drift = float(np.mean(click_kl[has_click])) if np.any(has_click) else 0.0

        # Subgroup disparity
        ndcg_pop = compute_ndcg_at_k(recs[cohorts["pop"]], test_csr[cohorts["pop"]], k=k) if len(cohorts["pop"]) > 0 else 0.0
        ndcg_niche = compute_ndcg_at_k(recs[cohorts["niche"]], test_csr[cohorts["niche"]], k=k) if len(cohorts["niche"]) > 0 else 0.0
        subgroup_gap = compute_subgroup_utility_gap(ndcg_pop, ndcg_niche)

        # Utility
        prec = compute_precision_at_k(recs, test_csr, k=k)
        rec = compute_recall_at_k(recs, test_csr, k=k)
        ndcg = compute_ndcg_at_k(recs, test_csr, k=k)

        # Tie-aware AUC over real full-catalog scores (never a top-K matrix).
        full_scores = model_adapter.score_all(active_users)
        auc = compute_vectorized_auc(full_scores, test_csr, exclude_matrix=D_t)

        # Geometry
        _, V_t = model_adapter.get_embeddings()
        if V_t is not None and V_model_0 is not None:
            V_t_aligned = compute_reflection_corrected_procrustes(V_model_0, V_t)
            r_norm_arr = compute_relative_norm_shift(V_model_0, V_t_aligned)
            valid_norm = np.isfinite(r_norm_arr)
            r_norm_valid_frac = float(np.mean(valid_norm))
            if np.any(valid_norm):
                r_norm_median = float(np.median(r_norm_arr[valid_norm]))
                r_norm_mean = float(np.mean(r_norm_arr[valid_norm]))
            else:
                r_norm_median, r_norm_mean = 1.0, 1.0
            s_eff = compute_effective_rank(V_t_aligned)
        else:
            r_norm_median, r_norm_mean, r_norm_valid_frac = 1.0, 1.0, 0.0
            s_eff = 1.0

        seed_metrics["gini"].append(gini)
        seed_metrics["coverage"].append(cov)
        seed_metrics["aplt"].append(aplt)
        seed_metrics["kl_drift"].append(mean_kl_drift)
        seed_metrics["rec_kl"].append(rec_kl)
        seed_metrics["subgroup_gap"].append(subgroup_gap)
        seed_metrics["precision"].append(prec)
        seed_metrics["recall"].append(rec)
        seed_metrics["ndcg"].append(ndcg)
        seed_metrics["auc"].append(auc)
        seed_metrics["r_norm"].append(r_norm_median)
        seed_metrics["r_norm_mean"].append(r_norm_mean)
        seed_metrics["r_norm_valid_frac"].append(r_norm_valid_frac)
        seed_metrics["s_eff"].append(s_eff)

        # 4. Ingestion
        D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))

        # 5. Cold retrain for the next generation only.
        if t < T - 1:
            model_adapter = adapter_cls(**kwargs)
            model_adapter.fit(D_t, seed=seed + t + 1)

    del model_adapter, D_t, U_anchor, V_anchor
    gc.collect()
    return seed_metrics


def run_simulation_pipeline(
    adapter_cls: Type[Any],
    base_csr: sp.csr_matrix,
    test_csr: sp.csr_matrix,
    seeds: List[int] = [42, 123, 456],
    T: int = 5,
    k: int = 10,
    alpha: float = 0.8,
    tau: float = 1.0,
    item_category_matrix: Optional[np.ndarray] = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
    anchor_dim: int = 32,
) -> Dict[str, Any]:
    n_users, n_items = base_csr.shape
    kwargs = adapter_kwargs.copy() if adapter_kwargs else {}

    base_pop = np.asarray(base_csr.sum(axis=0)).flatten()
    sorted_item_indices = np.argsort(base_pop)
    n_tail = int(n_items * 0.8)
    long_tail_mask = np.zeros(n_items, dtype=bool)
    long_tail_mask[sorted_item_indices[:n_tail]] = True

    cohorts = partition_user_cohorts(base_csr, long_tail_mask)

    if item_category_matrix is None:
        item_category_matrix = np.ones((n_items, 1), dtype=np.float32)

    base_user_dist = _compute_baseline_user_dist(base_csr, item_category_matrix)

    # One shared, model-independent anchor for every adapter and seed.
    U_anchor, V_anchor = build_shared_anchor(base_csr, dim=anchor_dim, seed=42)

    # Parallelize across seeds using processes
    worker_args = [
        (
            seed,
            adapter_cls,
            kwargs,
            base_csr,
            test_csr,
            T,
            k,
            alpha,
            tau,
            long_tail_mask,
            cohorts,
            base_user_dist,
            item_category_matrix,
            U_anchor,
            V_anchor,
        )
        for seed in seeds
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_seed, *args) for args in worker_args]
        seed_runs_history = [f.result() for f in futures]

    aggregated_results: Dict[str, Dict[str, List[float]]] = {}
    metric_keys = seed_runs_history[0].keys()

    for key in metric_keys:
        trajectory_matrix = np.array([run[key] for run in seed_runs_history], dtype=np.float64)
        aggregated_results[key] = {
            "mean": np.mean(trajectory_matrix, axis=0).tolist(),
            "std": np.std(trajectory_matrix, axis=0).tolist(),
        }

    return {
        "seeds": seeds,
        "generations": T,
        "config": {
            "anchor_method": "truncated_svd",
            "anchor_dim": int(U_anchor.shape[1]),
            "alpha": alpha,
            "tau": tau,
            "k": k,
            "T": T,
        },
        "aggregated": aggregated_results,
        "seed_runs": seed_runs_history,
    }

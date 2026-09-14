import gc
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Type, Optional
from src.data.buffer import merge_and_deduplicate
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
    compute_mag_ang_decomposition,
    compute_effective_rank,
)

def run_simulation_pipeline(
    adapter_cls: Type[Any],
    base_csr: sp.csr_matrix,
    test_csr: sp.csr_matrix,
    seeds: List[int] = [42, 123, 456],
    T: int = 5,
    k: int = 10,
    alpha: float = 0.5,
    tau: float = 1.0,
    item_category_matrix: Optional[np.ndarray] = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes closed-loop recommendation bias & latent dynamics simulation
    across multiple generations and random seeds.

    Args:
        adapter_cls: Recommender adapter class (e.g. CornacMFAdapter).
        base_csr: Initial training interaction matrix D_0 of shape (|U|, |I|).
        test_csr: Held-out ground-truth test matrix D_test of shape (|U|, |I|).
        seeds: List of random seeds for multi-seed experiment execution.
        T: Total number of simulation generations to run (t = 0..T-1).
        k: Top-K recommendation cut-off length.
        alpha: Weight balancing intrinsic preference vs social conformity (0.0 <= alpha <= 1.0).
        tau: Temperature parameter for intrinsic preference.
        item_category_matrix: Optional 2D array of shape (|I|, C) mapping items to category distributions.
        adapter_kwargs: Optional dict of keyword arguments for adapter initialization.

    Returns:
        results: Dict containing raw seed logs and aggregated metrics (mean, std) across generations.
    """
    n_users, n_items = base_csr.shape
    kwargs = adapter_kwargs.copy() if adapter_kwargs else {}

    # Define long-tail mask (bottom 80% baseline popularity items)
    base_pop = np.array(base_csr.sum(axis=0)).flatten()
    sorted_item_indices = np.argsort(base_pop)
    n_tail = int(n_items * 0.8)
    long_tail_mask = np.zeros(n_items, dtype=bool)
    long_tail_mask[sorted_item_indices[:n_tail]] = True

    # User cohort partitioning based on baseline long-tail ratio
    cohorts = partition_user_cohorts(base_csr, long_tail_mask)

    # Item category distributions for KL divergence calculation
    if item_category_matrix is None:
        # Default one-hot categories or dummy 1-category matrix
        item_category_matrix = np.ones((n_items, 1), dtype=np.float32)

    n_categories = item_category_matrix.shape[1]

    # Baseline user category distributions P_u^{(0)}
    base_user_dist = np.zeros((n_users, n_categories), dtype=np.float32)
    for u in range(n_users):
        u_items = base_csr[u].indices
        if len(u_items) > 0:
            cat_sum = np.sum(item_category_matrix[u_items], axis=0)
            sum_val = np.sum(cat_sum)
            base_user_dist[u] = cat_sum / sum_val if sum_val > 0 else np.ones(n_categories) / n_categories
        else:
            base_user_dist[u] = np.ones(n_categories) / n_categories

    seed_runs_history: List[Dict[str, List[float]]] = []

    for seed in seeds:
        set_deterministic_seed(seed)
        D_t = base_csr.copy()
        active_users = np.arange(n_users, dtype=np.int32)

        # Instantiate and fit initial baseline model at t=0
        model_adapter = adapter_cls(**kwargs)
        model_adapter.fit(D_t, seed=seed)

        # Extract baseline ground-truth anchor embeddings (U*, V*)
        U_star, V_star = model_adapter.get_embeddings()

        # Fallback dummy anchor vectors if model is non-latent
        if U_star is None or V_star is None:
            np.random.seed(seed)
            dim = getattr(model_adapter, "dim", 32)
            U_star = np.random.randn(n_users, dim).astype(np.float32)
            V_star = np.random.randn(n_items, dim).astype(np.float32)

        seed_metrics: Dict[str, List[float]] = {
            "gini": [], "coverage": [], "aplt": [],
            "kl_drift": [], "subgroup_gap": [],
            "precision": [], "recall": [], "ndcg": [], "auc": [],
            "r_norm": [], "s_eff": []
        }

        for t in range(T):
            # 1. Recommendation Generation
            recs = model_adapter.recommend(active_users, k=k, remove_seen=True)

            # 2. Vectorized User Interaction Simulation
            pop_counts = np.array(D_t.sum(axis=0)).flatten()
            u_clicks, i_clicks = simulate_user_clicks(
                recs=recs,
                U_star=U_star,
                V_star=V_star,
                pop_counts=pop_counts,
                alpha=alpha,
                tau=tau,
                seed=seed + t
            )

            # 3. Diagnostic Auditing for Generation t
            # Tier 1 Macro
            gini = compute_exposure_gini(recs, n_items)
            cov = compute_catalog_coverage(recs, n_items)
            aplt = compute_aplt(recs, long_tail_mask)

            # Tier 2 Cohorts & KL Drift (Vectorized)
            u_recs_cat = item_category_matrix[recs]  # (n_users, k, n_categories)
            cat_sum = u_recs_cat.sum(axis=1)          # (n_users, n_categories)
            sum_val = cat_sum.sum(axis=1, keepdims=True)
            uniform_dist = np.ones((1, n_categories), dtype=np.float32) / n_categories
            current_rec_dist = np.where(sum_val > 0, cat_sum / np.maximum(sum_val, 1e-9), uniform_dist)

            kl_drifts = compute_kl_divergence_drift(base_user_dist, current_rec_dist)
            mean_kl_drift = float(np.mean(kl_drifts))

            ndcg_pop = compute_ndcg_at_k(recs[cohorts["pop"]], test_csr[cohorts["pop"]], k=k) if len(cohorts["pop"]) > 0 else 0.0
            ndcg_niche = compute_ndcg_at_k(recs[cohorts["niche"]], test_csr[cohorts["niche"]], k=k) if len(cohorts["niche"]) > 0 else 0.0
            subgroup_gap = compute_subgroup_utility_gap(ndcg_pop, ndcg_niche)

            # Tier 3 Ranking Utility
            prec = compute_precision_at_k(recs, test_csr, k=k)
            rec = compute_recall_at_k(recs, test_csr, k=k)
            ndcg = compute_ndcg_at_k(recs, test_csr, k=k)

            # Construct dummy score matrix for AUC computation (Vectorized)
            score_matrix = np.zeros((n_users, n_items), dtype=np.float32)
            user_rows = np.arange(n_users)[:, None]
            score_matrix[user_rows, recs] = np.linspace(1.0, 0.1, num=k)
            auc = compute_vectorized_auc(score_matrix, test_csr)

            # Tier 4 Geometry
            U_t, V_t = model_adapter.get_embeddings()
            if V_t is not None:
                V_t_aligned = compute_reflection_corrected_procrustes(V_star, V_t)
                r_norm_arr = compute_relative_norm_shift(V_star, V_t)
                mean_r_norm = float(np.mean(r_norm_arr))
                s_eff = compute_effective_rank(V_t)
            else:
                mean_r_norm = 1.0
                s_eff = 1.0

            # Record metrics
            seed_metrics["gini"].append(gini)
            seed_metrics["coverage"].append(cov)
            seed_metrics["aplt"].append(aplt)
            seed_metrics["kl_drift"].append(mean_kl_drift)
            seed_metrics["subgroup_gap"].append(subgroup_gap)
            seed_metrics["precision"].append(prec)
            seed_metrics["recall"].append(rec)
            seed_metrics["ndcg"].append(ndcg)
            seed_metrics["auc"].append(auc)
            seed_metrics["r_norm"].append(mean_r_norm)
            seed_metrics["s_eff"].append(s_eff)

            # 4. Ingest new clicks into Data Buffer D_{t+1} = D_t + ΔD_t
            D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))

            # 5. Cold Retrain model adapter on updated matrix D_{t+1}
            model_adapter = adapter_cls(**kwargs)
            model_adapter.fit(D_t, seed=seed + t + 1)

        seed_runs_history.append(seed_metrics)

        # Memory safety cleanup
        del model_adapter, D_t, U_star, V_star
        gc.collect()

    # Aggregate metric trajectories across seeds (mean & std)
    aggregated_results: Dict[str, Dict[str, List[float]]] = {}
    metric_keys = seed_runs_history[0].keys()

    for key in metric_keys:
        trajectory_matrix = np.array([run[key] for run in seed_runs_history], dtype=np.float64)
        aggregated_results[key] = {
            "mean": np.mean(trajectory_matrix, axis=0).tolist(),
            "std": np.std(trajectory_matrix, axis=0).tolist()
        }

    return {
        "seeds": seeds,
        "generations": T,
        "aggregated": aggregated_results,
        "seed_runs": seed_runs_history
    }

from src.metrics.macro import (
    compute_exposure_gini,
    compute_catalog_coverage,
    compute_aplt,
)
from src.metrics.user_groups import (
    partition_user_cohorts,
    compute_kl_divergence_drift,
    compute_subgroup_utility_gap,
)
from src.metrics.ranking import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_vectorized_auc,
)
from src.metrics.geometry import (
    compute_reflection_corrected_procrustes,
    compute_relative_norm_shift,
    compute_mag_ang_decomposition,
    compute_effective_rank,
)

__all__ = [
    # Tier 1
    "compute_exposure_gini",
    "compute_catalog_coverage",
    "compute_aplt",
    # Tier 2
    "partition_user_cohorts",
    "compute_kl_divergence_drift",
    "compute_subgroup_utility_gap",
    # Tier 3
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_ndcg_at_k",
    "compute_vectorized_auc",
    # Tier 4
    "compute_reflection_corrected_procrustes",
    "compute_relative_norm_shift",
    "compute_mag_ang_decomposition",
    "compute_effective_rank",
]

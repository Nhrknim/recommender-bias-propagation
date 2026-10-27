# Correctness Fixes Applied to the Closed-Loop Simulation

**Date:** 2026-10-27  
**Status:** Implemented and covered by the unit-test suite. The canonical 50-generation benchmark (`run_20261027_170352_60a757`, T=50, 3 seeds) and the separate Phase 6 LightFM benchmark (`run_20261027_160811_c80465`) have been re-run from the corrected code; all figures and notes now cite those artifacts. Historical run artifacts are preserved but superseded.

---

## Data Pipeline

- Collapsed duplicate `(user, item)` interactions (keep last) before splitting, and applied `sum_duplicates()` to sparse matrices.
- Guaranteed every user has at least one training interaction; removed any train/test overlap and added assertions for leakage.
- Forced interaction matrices to binary values and validated click-array lengths and index bounds before buffer ingestion.
- Rejected the `-1` padding sentinel before it can enter the accumulated training matrix.

## Adapters

- **Base contract:** added `score_all()` for real full-catalog scoring; documented that `recommend()` pads unused slots with `-1`.
- **LightFM:** fixed 1-D/2-D positive-pair checks, rejected observed positives during WARP/BPR negative sampling with bounded retries, skipped all-seen users, and added `score_all()`.
- **LightGCN:** summed duplicate adjacency entries, fixed positive-pair masking, skipped users with no valid negative, added `score_all()`, and enforced `-1` padding.
- **Cornac:** retained a canonical training matrix for seen-item masking, added `score_all()`, fixed the MostPop popularity vector, and enforced padding.
- **Cornac scoring:** `score_all()` now includes the learned user/item biases used by MF and PMF (`score = u·v + b_u + b_i`) and masks items absent from `D_t` as unscored instead of treating them as zero-scored candidates. The previous dot-product-only path made MF/PMF near-random rankers.
- **BPR configuration:** set `use_bias=False` so the adapter implements standard bias-free BPR-MF and its factor-only scoring matches training exactly. Cornac defaults to `use_bias=True`, which lets scalar item biases absorb popularity.

## Simulator and Metrics

- **Shared click anchor:** replaced the model's own `t=0` embeddings with a single truncated-SVD factorization of `D_0`, computed once with sign canonicalization and frozen across every adapter and seed. Geometry metrics now use each model's own `t=0` embeddings as a separate baseline.
- **User indexing:** preference and click simulation now use global user IDs throughout; clicks are emitted in global coordinates.
- **Padding:** `-1` slots are never clicked and are excluded from Gini, coverage, and APLT; APLT normalizes by valid slots.
- **AUC:** computed from full-catalog `score_all()` outputs with tie-aware average ranks, excluding the accumulated training matrix `D_t` from negatives.
- **Norm shift:** `r_norm` returns `NaN` for zero-norm baselines and is aggregated by median with a valid-fraction counter.
- **KL drift:** split into `kl_drift` (realized click-category drift) and `rec_kl` (recommendation-slate category drift).

## Orchestration and Figures

- Removed the unnecessary final retrain; added run configuration metadata and git SHA to JSON/report artifacts.
- Updated core and LightFM benchmarks to `T=50`, 3 seeds, and the shared anchor.
- Regenerated trajectory animations from measured per-generation PCA coordinates instead of interpolating between `t=0` and `t=T-1`.
- Updated visual scripts to use 5 LightGCN epochs, align manifolds with reflection-corrected Procrustes, and load the latest JSON artifacts.
- Checkpoint plots now write to `docs/benchmarks/checkpoint_trajectories.png`; partial runs no longer overwrite the publication dashboard.

## Validation

- `venv/bin/python -m unittest discover tests`: **64 tests pass**.
- Dataset check: 6,038 users, 3,533 items, 457,819 train interactions, 117,462 test interactions, zero train/test leakage, zero zero-train users.

# System Engineering Specification: Closed-Loop Recommendation Bias & Latent Dynamics Simulator

**Document:** `SPEC.md`  
**Target System:** Closed-Loop Recommender Bias Propagation & Latent Dynamics Simulator  
**Status:** Approved Engineering Ground of Truth  
**Date:** September 2026  

---

## 1. System Objectives & Architecture Overview

This specification establishes the canonical software and mathematical architecture for a closed-loop simulator designed to measure popularity bias propagation (the Matthew Effect) and latent space representation degradation in collaborative filtering systems over iterative recommendation cycles.

### 1.1 High-Level Architectural Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP SIMULATION ARCHITECTURE                      │
│                                                                             │
│  ┌───────────────────────────┐         ┌─────────────────────────────────┐  │
│  │ Accumulating Train Matrix │         │  Shared Frozen Click Anchor      │  │
│  │   D_t ∈ {0, 1}^(|U| x |I|)│         │   (U*, V*) = TruncatedSVD(D_0)   │  │
│  └─────────────┬─────────────┘         └────────────────┬────────────────┘  │
│                │                                        │                   │
│                ▼                                        ▼                   │
│  ┌───────────────────────────┐         ┌─────────────────────────────────┐  │
│  │ Architecture-Agnostic     │         │ Vectorized User Interaction     │  │
│  │ Recommender Adapter       │────────>│ Engine                          │  │
│  │ Generates Top-K Lists     │         │ Simulates Bernoulli Clicks      │  │
│  └─────────────┬─────────────┘         └────────────────┬────────────────┘  │
│                │                                        │                   │
│                │                                        ▼                   │
│                │                       ┌─────────────────────────────────┐  │
│                │                       │ Data Buffer Ingestion           │  │
│                │                       │ D_{t+1} = Binarize(D_t ∪ ΔD_t)  │  │
│                │                       └────────────────┬────────────────┘  │
│                │                                        │                   │
│                ▼                                        ▼                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                  Multi-Tier Diagnostic Engine                         │  │
│  │  Tier 1: Exposure Gini, Coverage@K, APLT@K                            │  │
│  │  Tier 2: User Cohorts, Calibration Drift D_KL, Subgroup Utility Gap   │  │
│  │  Tier 3: Precision@K, Recall@K, NDCG@K, Vectorized Mann-Whitney AUC   │  │
│  │  Tier 4: Procrustes Alignment, Norm Shift R_norm, Effective Rank S_eff│  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Subsystem 1: Invariant State & Data Buffer Subsystem

### 2.1 Continuous Zero-Indexed Entity Register
* All unique users $u \in [0, |\mathcal{U}|-1]$ and items $i \in [0, |\mathcal{I}|-1]$ are assigned continuous integer indices at baseline initialization ($t=0$).
* Coordinate index assignments are **immutable**. Sparse matrix dimensions $(|\mathcal{U}| \times |\mathcal{I}|)$ remain fixed across all simulation generations $t \in [0, T]$.

### 2.2 Dual-Matrix State Architecture
* **Accumulating Interaction Matrix ($\mathcal{D}_t \in \{0, 1\}^{|\mathcal{U}| \times |\mathcal{I}|}$):** CSR sparse binary matrix storing historical interactions. Ingests new clicks $\Delta \mathcal{D}_t$ per generation and deduplicates values:

```python
import numpy as np
import scipy.sparse as sp
from typing import Tuple

def merge_and_deduplicate(D_t: sp.csr_matrix, clicks_coo: Tuple[np.ndarray, np.ndarray]) -> sp.csr_matrix:
    """Merges new clicks ΔD_t into D_t and strictly binarizes non-zero values."""
    u_idx, i_idx = clicks_coo
    n_users, n_items = D_t.shape
    
    delta_coo = sp.coo_matrix((np.ones_like(u_idx), (u_idx, i_idx)), shape=(n_users, n_items))
    D_next = (D_t + delta_coo).tocsr()
    D_next.data = np.ones_like(D_next.data, dtype=np.float32)
    return D_next
```

* **Shared Frozen Click Anchor ($U^* \in \mathbb{R}^{|\mathcal{U}| \times d}, V^* \in \mathbb{R}^{|\mathcal{I}| \times d}$):** Read-only user and item factor matrices estimated **once** from the baseline matrix $\mathcal{D}_0$ by truncated SVD (with per-component sign canonicalisation). The anchor is model-independent: every adapter and seed in a run shares exactly the same $U^*, V^*$, so click traces are not generated from the geometry of the model under evaluation. Geometry metrics use a separate baseline: each model's own $t=0$ item factors.

### 2.3 Catalog Popularity Register
* Dense 1D array $f_i^{(t)} = \sum_{u=0}^{|\mathcal{U}|-1} \mathcal{D}_t[u, i]$ tracking total cumulative interactions per item.

### 2.4 Shared Click Anchor Construction
The click anchor is computed once per run from the baseline matrix:

$$\mathcal{D}_0 \approx U \Sigma V^\top, \qquad U^* = U \Sigma^{1/2}, \quad V^* = V \Sigma^{1/2},$$

using a deterministic truncated SVD with $d = 32$ components (or $\min(32, |\mathcal{U}|, |\mathcal{I}|)$). Per-component signs are canonicalised so the largest-magnitude entry of each right singular vector is positive. The anchor is passed read-only to every worker process; it never depends on the adapter class, seed, or generation index. This prevents the self-confirming loop in which a model's own embeddings define the preferences used to evaluate it.

---

## 3. Subsystem 2: Architecture-Agnostic Recommender Interface

### 3.1 Abstract Facade Adapter Protocol

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseRecommenderAdapter(ABC):
    """Abstract Facade Interface for Recommendation Backbones."""
    
    @abstractmethod
    def fit(self, train_matrix: sp.csr_matrix, seed: int) -> 'BaseRecommenderAdapter':
        """Cold retrains model on interaction matrix D_t with fixed seed."""
        pass

    @abstractmethod
    def recommend(self, user_indices: np.ndarray, k: int = 10, remove_seen: bool = True) -> np.ndarray:
        """Generates top-K recommendation lists of shape (|users|, K).

        Unused slots when K > |I| are padded with the sentinel -1.
        """
        pass

    @abstractmethod
    def score_all(self, user_indices: np.ndarray) -> np.ndarray:
        """Returns real full-catalog scores of shape (|users|, |I|).

        This is the canonical input for tie-aware AUC; it must not return a
        truncated top-K matrix.
        """
        pass

    @abstractmethod
    def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts learned user and item latent matrices (U_t, V_t) aligned to global indices."""
        pass
```

### 3.2 Framework Coordinate Alignment Mapping
For model backbones with internal indexing (e.g., Cornac), adapters translate internal arrays to global coordinates:

```python
def extract_aligned_embeddings(model_cornac, n_users: int, n_items: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Translates framework-internal factor arrays into global matrix coordinates."""
    U_global = np.zeros((n_users, dim), dtype=np.float32)
    V_global = np.zeros((n_items, dim), dtype=np.float32)
    
    for internal_u, global_u in enumerate(model_cornac.user_uid_map):
        U_global[global_u] = model_cornac.u_factors[internal_u]
        
    for internal_i, global_i in enumerate(model_cornac.item_uid_map):
        V_global[global_i] = model_cornac.i_factors[internal_i]
        
    return U_global, V_global
```

---

## 4. Subsystem 3: Vectorized User Interaction Engine

### 4.1 Probability Formulations
* **Position-Discounted Exposure Model:**
  $$P(\text{Exposed} \mid r) = \frac{1}{\log_2(r + 2)}, \quad r \in \{0, \dots, K-1\}$$
* **Temperature-Scaled Intrinsic Preference Matching:**
  $$P(\text{Relevance} \mid u, i) = \sigma\left( \text{clip}\left( \frac{\mathbf{u}_u^{*T} \mathbf{v}_i^*}{\tau \cdot \sqrt{d}}, -15.0, 15.0 \right) \right)$$
  *Parameters:* Temperature $\tau = 1.0$, Dimension $d = 32$.
* **Sub-Linear Social Conformity Model:**
  $$P(\text{Conformity} \mid i, t) = \frac{\log(1 + f_i^{(t)})}{\log\left(1 + \max_{j \in \mathcal{I}} f_j^{(t)}\right)}$$
* **Joint Click Probability:**
  $$P(\text{Click} \mid u, i, r, t) = P(\text{Exposed} \mid r) \cdot \left[ (1 - \alpha) P(\text{Relevance} \mid u, i) + \alpha P(\text{Conformity} \mid i, t) \right]$$

### 4.2 Vectorized Execution Implementation

```python
def simulate_user_clicks(
    recs: np.ndarray,          # (N_active, K), -1 marks padding
    U_star: np.ndarray,        # shared anchor, (|U|, d)
    V_star: np.ndarray,        # shared anchor, (|I|, d)
    pop_counts: np.ndarray,    # (|I|,)
    user_indices: np.ndarray,  # GLOBAL user ids aligned to recs rows
    alpha: float = 0.5,
    tau: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates browsing, position decay, relevance, conformity, and session budgets."""
    rng = np.random.default_rng(seed)
    n_active, k = recs.shape
    d = U_star.shape[1]

    # 1. Position Discount
    p_exposed = compute_position_exposure(k)

    # 2. Relevance & Conformity (user_indices selects the correct global row)
    p_relevance = compute_intrinsic_preference(U_star, V_star, recs, user_indices, tau=tau)
    p_conformity = compute_social_conformity(pop_counts, recs)

    # 3. Joint Click Probability, with -1 padding forced to zero
    p_click = p_exposed * ((1.0 - alpha) * p_relevance + alpha * p_conformity)
    p_click[recs < 0] = 0.0

    # 4. Bernoulli Sampling & Poisson Session Budgeting (λ = 3)
    clicked_mask = (rng.random((n_active, k)) < p_click) & (recs >= 0)
    budgets = rng.poisson(lam=3.0, size=n_active)

    clicked_u, clicked_i = [], []
    for row in range(n_active):
        items_clicked = recs[row, clicked_mask[row]]
        items_clicked = items_clicked[: budgets[row]]
        clicked_u.extend([int(user_indices[row])] * len(items_clicked))
        clicked_i.extend(items_clicked)

    return np.asarray(clicked_u, dtype=np.int32), np.asarray(clicked_i, dtype=np.int32)
```

**Coordinate contract:** rows of `recs` correspond to `user_indices`, which are **global** zero-indexed user IDs. The returned click arrays emit those same global IDs, so `merge_and_deduplicate` can add them directly to $\mathcal{D}_{t+1}$ without local-row remapping.

---

## 5. Subsystem 4: Multi-Tier Metric & Diagnostic Engine

### 5.1 Tier 1: Macro-Catalog Distribution
* **Exposure Gini ($G^{(t)}$):**
  $$G^{(t)} = \frac{|\mathcal{I}| + 1}{|\mathcal{I}|} - \frac{2 \sum_{k=1}^{|\mathcal{I}|} (|\mathcal{I}| - k + 1) c_{(k)}^{(t)}}{|\mathcal{I}| \sum_{i=1}^{|\mathcal{I}|} c_i^{(t)}}$$
* **Catalog Coverage@K:** $\text{Cov}@K = \frac{\left| \bigcup_{u \in \mathcal{U}} \text{TopK}^{(t)}(u) \right|}{|\mathcal{I}|}$.
* **Average Percentage of Long-Tail@K:** $\text{APLT}@K = \frac{1}{|\mathcal{U}|} \sum_{u} \frac{|\text{TopK}^{(t)}(u) \cap \mathcal{I}_{\text{Long-Tail}}|}{\text{valid slots}(u)}$, where padding sentinels are excluded from both numerator and denominator.

### 5.2 Tier 2: User-Subgroup Mesoscale Dynamics
* **User Cohort Partitioning:** Users sorted by baseline long-tail interaction ratio $\gamma_u$ and split into three equal groups ($\mathcal{U}_{\text{Pop}}, \mathcal{U}_{\text{Div}}, \mathcal{U}_{\text{Niche}}$).
* **Category Calibration Drift ($D_{\text{KL}}$):** Two distinct quantities are recorded:
  * `kl_drift` — **realized click-category drift**: divergence between the baseline user category distribution $P_u^{(0)}$ and the category mix of the user's actual clicks in generation $t$;
  * `rec_kl` — **recommendation-category drift**: divergence between $P_u^{(0)}$ and the category mix of the recommended slate.
  $D_{\text{KL}}\left(P_u^{(0)} \parallel Q_u^{(t)}\right) = \sum_{c \in \mathcal{C}} \tilde{P}_u^{(0)}(c) \log \left( \frac{\tilde{P}_u^{(0)}(c)}{\tilde{Q}_u^{(t)}(c)} \right)$
* **Subgroup Utility Gap:** $\Delta \text{NDCG}@K = \text{NDCG}@K(\mathcal{U}_{\text{Pop}}) - \text{NDCG}@K(\mathcal{U}_{\text{Niche}})$.

### 5.3 Tier 3: Ranking & Global Catalog Utility
* **Retrieval Metrics:** Precision@K, Recall@K, NDCG@K evaluated against test set.
* **Vectorized Mann-Whitney Global Pairwise AUC:** evaluated on **real full-catalog scores** (`score_all`), not the top-K matrix. Ties receive average ranks, so an all-tied score vector yields $0.5$. Items in the accumulating training matrix $\mathcal{D}_t$ are excluded from the negative pool; a test positive is never excluded.

```python
def compute_vectorized_auc(score_matrix, test_matrix, exclude_matrix=None) -> float:
    """Tie-aware global pairwise AUC over real full-catalog scores."""
    from scipy.stats import rankdata

    n_users, n_items = score_matrix.shape
    auc_scores = []
    for u in range(n_users):
        pos_items = test_matrix[u].indices
        if len(pos_items) == 0 or len(pos_items) == n_items:
            continue

        scores = np.asarray(score_matrix[u]).ravel()
        keep = np.ones(n_items, dtype=bool)
        if exclude_matrix is not None:
            keep[exclude_matrix[u].indices] = False
            keep[pos_items] = True  # never drop a test positive
        candidate_scores = scores[keep]
        pos_mask = np.isin(np.flatnonzero(keep), pos_items)

        n_pos = int(pos_mask.sum())
        n_neg = len(candidate_scores) - n_pos
        if n_pos == 0 or n_neg <= 0:
            continue

        ranks = rankdata(candidate_scores, method="average")
        rank_sum = float(np.sum(ranks[pos_mask]))
        auc_scores.append((rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))

    return float(np.mean(auc_scores)) if auc_scores else 0.5
```

### 5.4 Tier 4: Latent Space Geometry & Magnitude Dynamics

#### Reflection-Corrected Orthogonal Procrustes Alignment
```python
def compute_reflection_corrected_procrustes(V_0: np.ndarray, V_t: np.ndarray) -> np.ndarray:
    """Solves R* = U V_svd^T subject to det(R*) = +1 to guarantee proper rigid rotation."""
    M = V_0.T @ V_t
    U, _, Vt_svd = np.linalg.svd(M)
    
    R_star = U @ Vt_svd
    if np.linalg.det(R_star) < 0:
        Vt_svd[-1, :] *= -1.0
        R_star = U @ Vt_svd
        
    return V_t @ R_star
```

#### Magnitude & Geometric Formulas
* **Smooth Relative $L_2$ Vector Norm Shift ($R_{\text{norm}}$):**
  $$R_{\text{norm}}(i, t) = \frac{\|v_i^{(t)}\|_2 + \epsilon}{\|v_i^{(0)}\|_2 + \epsilon}, \quad \epsilon = 10^{-9}$$
  *Zero-norm guard:* when $\|v_i^{(0)}\|_2 = 0$ the ratio is undefined. Such entries are returned as `NaN` and excluded from aggregate statistics; trajectories report the **median** over finite entries alongside the valid fraction, rather than letting a single cold item dominate the mean.
* **Decomposed Magnitude vs. Angular Drift:**
  $$\|v_i^{(0)} - v_i^{(t, \text{aligned})}\|_2^2 = \underbrace{\left( \|v_i^{(0)}\|_2 - \|v_i^{(t)}\|_2 \right)^2}_{\Delta_{\text{mag}}(i, t)} + \underbrace{2 \|v_i^{(0)}\|_2 \|v_i^{(t)}\|_2 \left( 1 - \cos \theta_i \right)}_{\Delta_{\text{ang}}(i, t)}$$
* **Embedding Drift Velocity ($v_{\text{drift}}$):** Average angular shift $1 - \cos\theta_i$ per item cohort.
* **Effective Rank ($S_{\text{eff}}$):** Continuous rank measured via spectral entropy over singular values of $V_t$:
  $$\tilde{\sigma}_k = \frac{\sigma_k}{\sum_{j=1}^d \sigma_j}, \quad S_{\text{eff}}(t) = \exp\left( -\sum_{k=1}^d \tilde{\sigma}_k \ln \tilde{\sigma}_k \right)$$

---

## 6. Subsystem 5: Multi-Seed Orchestrator & Lifecycle

### 6.1 Lifecycle Execution Routine

```python
import random

def set_deterministic_seed(seed: int):
    """Sets deterministic seed state across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def run_simulation_pipeline(adapter_cls, base_csr, test_csr, seeds=[42, 123, 456], T=5):
    """Executes multi-seed simulation lifecycle across T generations."""
    # One shared, model-independent anchor for every seed/adapter.
    U_star, V_star = build_shared_anchor(base_csr, dim=32, seed=42)

    for seed in seeds:
        set_deterministic_seed(seed)
        D_t = base_csr.copy()
        active_users = np.arange(D_t.shape[0], dtype=np.int32)

        model = adapter_cls().fit(D_t, seed=seed)
        _, V_model_0 = model.get_embeddings()

        for t in range(T):
            recs = model.recommend(active_users, k=10, remove_seen=True)
            u_clicks, i_clicks = simulate_user_clicks(
                recs, U_star, V_star, np.asarray(D_t.sum(axis=0)).flatten(),
                user_indices=active_users, alpha=0.8, seed=seed + t,
            )
            audit_generation_metrics(D_t, recs, U_star, V_star, V_model_0, model, generation=t)

            D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))

            # Cold retrain only if another generation will be evaluated.
            if t < T - 1:
                model = adapter_cls().fit(D_t, seed=seed + t + 1)

        del model, D_t, U_star, V_star
        import gc; gc.collect()
```

---

## 7. System Diagnostic Metric Reference

| Tier | Metric Name | Mathematical Formulation | Diagnostic Purpose |
| --- | --- | --- | --- |
| **Tier 1** | Exposure Gini ($G$) | Formula in Section 5.1 | Quantifies catalog recommendation inequality. |
| **Tier 1** | Catalog Coverage@K | $\|\bigcup \text{TopK}(u)\| / |\mathcal{I}|$ | Percentage of unique catalog items recommended. |
| **Tier 1** | APLT@K | Average percentage of tail items in TopK | Tracks long-tail item exposure. |
| **Tier 2** | Calibration Drift ($D_{\text{KL}}$) | $\sum \tilde{P}_u^{(0)} \log(\tilde{P}_u^{(0)} / \tilde{Q}_u^{(t)})$ | Measures user taste distortion per cohort. |
| **Tier 2** | Subgroup Utility Gap | $\text{NDCG}_{\text{Pop}} - \text{NDCG}_{\text{Niche}}$ | Measures marginalization of niche users. |
| **Tier 3** | NDCG@K | $\text{DCG}@K / \text{IDCG}@K$ | Position-discounted retrieval accuracy. |
| **Tier 3** | Vectorized Pairwise AUC | Formula in Section 5.3 | Catalog-wide ranking capacity ($\mathcal{O}(|\mathcal{I}| \log |\mathcal{I}|)$). |
| **Tier 4** | Relative Norm Shift ($R_{\text{norm}}$) | $(\|v_t\|_2 + \epsilon) / (\|v_0\|_2 + \epsilon)$ | Tracks origin collapse vs. periphery dispersion. |
| **Tier 4** | Mag/Ang Error Decomposition | $(\|v_0\| - \|v_t\|)^2 + 2\|v_0\|\|v_t\|(1-\cos\theta)$ | Disentangles vector length shift from rotation. |
| **Tier 4** | Effective Rank ($S_{\text{eff}}$) | $\exp(-\sum \tilde{\sigma}_k \ln \tilde{\sigma}_k)$ | Detects latent manifold dimensional collapse. |

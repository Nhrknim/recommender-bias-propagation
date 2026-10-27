# Recommender Bias Propagation Engine (`src/`)

This directory contains the core Python library for simulating, auditing, and analyzing closed-loop recommendation bias propagation and latent space dynamics. 

The framework models the feedback loop where recommender systems influence user interaction choices, which in turn retrain future models, resulting in popularity concentration, taste homogenization, and latent manifold rank collapse.

---

## Part 1: How to Run & Configure Simulations

Simulations can be executed either via the top-level benchmark script or programmatically within Python scripts and notebooks.

### 1. Running via CLI Benchmark Script

To run the complete benchmark suite across multiple recommender models:

```bash
python scripts/run_full_benchmark.py
```

### 2. Programmatic Execution

To run customized simulation trials, import and call `run_simulation_pipeline` from `src.orchestrator.runner`:

```python
import numpy as np
import scipy.sparse as sp
from src.data.dataset import load_movielens_1m, load_movielens_genres
from src.adapters.cornac_adapters import CornacMFAdapter
from src.orchestrator.runner import run_simulation_pipeline

# 1. Load dataset and genre metadata
base_csr, test_csr, user2code, item2code = load_movielens_1m(
    "ml-1m/ratings.dat", threshold=4.0, split_ratio=0.8
)
item_categories = load_movielens_genres("ml-1m/movies.dat", item2code=item2code)

# 2. Run simulation with custom hyperparameter settings
results = run_simulation_pipeline(
    adapter_cls=CornacMFAdapter,
    base_csr=base_csr,
    test_csr=test_csr,
    seeds=[42, 123, 456],
    T=50,
    k=10,
    alpha=0.8,
    tau=1.0,
    item_category_matrix=item_categories,
    adapter_kwargs={"dim": 32, "max_iter": 10},
    anchor_dim=32,
)

# 3. Access aggregated metric trajectories
gini_mean = results["aggregated"]["gini"]["mean"]
ndcg_mean = results["aggregated"]["ndcg"]["mean"]
rec_kl_mean = results["aggregated"]["rec_kl"]["mean"]
print(f"Exposure Gini Trajectory: {gini_mean}")
print(f"NDCG@10 Trajectory:       {ndcg_mean}")
print(f"Recommendation KL Drift:  {rec_kl_mean}")
```

### 3. Simulation Configuration Settings

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `adapter_cls` | `Type[BaseRecommenderAdapter]` | **Required** | Model adapter class (`CornacMFAdapter`, `CornacBPRAdapter`, `CornacMostPopAdapter`, `CornacUserKNNAdapter`). |
| `base_csr` | `sp.csr_matrix` | **Required** | Initial training interaction matrix $D_0$ of shape $(|U|, |I|)$. |
| `test_csr` | `sp.csr_matrix` | **Required** | Ground-truth held-out test matrix $D_{\text{test}}$ of shape $(|U|, |I|)$. |
| `seeds` | `List[int]` | `[42, 123, 456]` | List of integer seeds for multi-seed statistical aggregation. |
| `T` | `int` | `5` | Total number of simulation feedback loop generations ($t = 0 \dots T-1$). |
| `k` | `int` | `10` | Recommendation list cut-off length $K$. |
| `alpha` | `float` | `0.8` | Conformity weight balancing social proof vs intrinsic relevance ($0.0 \le \alpha \le 1.0$). |
| `tau` | `float` | `1.0` | Temperature scaling parameter for intrinsic user relevance ($\tau > 0$). |
| `item_category_matrix` | `np.ndarray` | `None` | Optional 2D float array of shape $(|I|, C)$ for user genre taste drift calculations. |
| `adapter_kwargs` | `Dict[str, Any]` | `{}` | Keyword arguments forwarded to the recommender model constructor (`dim`, `max_iter`, `learning_rate`). |
| `anchor_dim` | `int` | `32` | Truncated-SVD rank of the shared, model-independent click anchor built from $D_0$. |

---

## Part 2: Project Architecture & Subsystem Layout

The codebase is organized into five modular subsystems, each handling a distinct stage of the simulation life cycle:

```
src/
├── data/           # Dataset ingestion, binarization, and interaction buffering
├── adapters/       # Recommender model facades & embedding matrix alignment
├── engine/         # User browsing, exposure decay, and click simulation
├── metrics/        # 4-Tier Diagnostic Auditing System
└── orchestrator/   # Deterministic seed control and pipeline execution
```

---

### Subsystem 1: Data Pipeline (`src/data/`)

* **`dataset.py`**: Handles dataset loading (`load_movielens_1m`), binarization ($r_{ui} \in \{0, 1\}$), train/test splitting (`parse_and_split_dataframe`), continuous 0-indexing mapping, and genre matrix extraction (`load_movielens_genres`).
* **`buffer.py`**: Manages feedback loop interaction accumulation (`merge_and_deduplicate`). Combines existing interaction matrix $D_t$ with newly generated simulated clicks $\Delta D_t$, enforcing coordinate bounds and deduplication.

### Subsystem 2: Recommender Adapters (`src/adapters/`)

* **`base.py`**: Defines abstract base interface `BaseRecommenderAdapter` enforcing standardized `fit()`, `recommend()`, `score_all()`, and `get_embeddings()` methods across model families. `recommend()` pads unused slots with `-1`; `score_all()` returns real full-catalog scores for tie-aware AUC.
* **`cornac_adapters.py`**: Implements framework adapters for Cornac models (`CornacMFAdapter`, `CornacBPRAdapter`, `CornacMostPopAdapter`, `CornacUserKNNAdapter`). 
  * *Coordinate Alignment (`extract_aligned_embeddings`)*: Translates framework-internal factor indices into global matrix coordinates $(|U|, d)$ and $(|I|, d)$.
  * *Fast Vectorized Scoring*: Vectorizes batch score computation $U_{\text{active}} \cdot V^T$ to eliminate per-user scoring loops.

### Subsystem 3: User Interaction Engine (`src/engine/`)

Simulates stochastic user browsing, attention decay, and item selection:

* **`exposure.py`**: Position-Based Model (PBM) exposure discount $P(\text{Exposed} \mid r) = \frac{1}{\log_2(r + 2)}$.
* **`anchor.py`**: Builds the single shared, model-independent click anchor from $D_0$ via deterministic truncated SVD with sign canonicalisation.
* **`preference.py`**: Temperature-scaled intrinsic user relevance $P(\text{Relevance} \mid u, i) = \sigma\left(\frac{u^* \cdot v^*}{\tau \sqrt{d}}\right)$ using the shared anchor and **global** user row indices.
* **`conformity.py`**: Sub-linear social proof / popularity bandwagon probability $P(\text{Conformity} \mid i, t) = \frac{\log(1 + f_i)}{\log(1 + \max f)}$.
* **`simulator.py`**: Combines probabilities via $P_{\text{click}} = P_{\text{exposed}} \cdot [(1-\alpha) P_{\text{relevance}} + \alpha P_{\text{conformity}}]$, executing Bernoulli sampling and Poisson budget truncation ($\lambda = 3.0$). Emits global user IDs and masks `-1` padding.

### Subsystem 4: 4-Tier Diagnostic Auditing (`src/metrics/`)

* **`macro.py` (Tier 1: Macro Exposure)**: Evaluates catalog exposure inequality using Exposure Gini ($G$), Catalog Coverage@K, and Average Popularity of Recommended Items (APLT@K), ignoring `-1` padding.
* **`user_groups.py` (Tier 2: Cohorts & Drift)**: Partitions users into popularity-seeking ($\mathcal{U}_{\text{Pop}}$), diverse ($\mathcal{U}_{\text{Div}}$), and niche ($\mathcal{U}_{\text{Niche}}$) cohorts. Computes realized click-category drift (`kl_drift`), recommendation-category drift (`rec_kl`), and subgroup utility gaps.
* **`ranking.py` (Tier 3: Utility & Accuracy)**: Computes standard recommendation accuracy metrics on held-out test data $D_{\text{test}}$ (Precision@K, Recall@K, NDCG@K) plus a tie-aware full-catalog Mann-Whitney AUC that excludes observed training items from the negative pool.
* **`geometry.py` (Tier 4: Latent Dynamics)**: Audits latent manifold geometry evolution using Orthogonal Procrustes rigid alignment ($\det(R^*) = +1$), relative norm shift ($R_{\text{norm}}$, with `NaN` for zero-norm baselines), and effective spectral rank ($S_{\text{eff}}$).

### Subsystem 5: Orchestration & Seed Control (`src/orchestrator/`)

* **`seed_control.py`**: Enforces strict multi-library deterministic reproducibility across Python `random`, NumPy, and PyTorch (if present).
* **`runner.py`**: Manages multi-generation simulation loops ($t = 0 \dots T-1$), retraining models on accumulated interactions $D_{t+1} = D_t + \Delta D_t$, collecting metrics across seeds, and computing mean ($\mu$) and standard deviation ($\sigma$) trajectories.
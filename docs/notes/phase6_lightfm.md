# Phase 6 Research Notes: LightFM — Pairwise Ranking & WARP Loss Dynamics

**Notebook Reference**: [`notebooks/phase_6_lightfm.ipynb`](../../notebooks/phase_6_lightfm.ipynb)  
**Visual Assets Folder**: [`docs/images/phase_6_lightfm/`](../images/phase_6_lightfm/)  
**Benchmark Suite Report**: [`docs/benchmarks/lightfm_benchmark_results.md`](../benchmarks/lightfm_benchmark_results.md)  
**Raw 50-Gen Trajectory Log**: [`docs/benchmarks/lightfm_run_20261027_160811_c80465.json`](../benchmarks/lightfm_run_20261027_160811_c80465.json)  
**32-D Quantitative Drift Metrics**: [`docs/benchmarks/lightfm_quantitative_drift_metrics.json`](../benchmarks/lightfm_quantitative_drift_metrics.json)

> **Backend note:** Phase 6 is a supplementary run executed with the repository's native NumPy fallback (`_NativeLightFM`) because the compiled LightFM C library is unavailable in this environment. Cross-library comparisons with the five-model core benchmark are indicative only.

---

## Executive Summary: How Can Pairwise Optimization Resist Runaway Bias?

Throughout Phases 1 to 5, we observed that closed-loop recommendation feedback invariably acts as a popularity amplifier. In Pointwise Matrix Factorization (MF), item vectors contract radially toward the origin (median $R_{\text{norm}} = 0.704$ by $t=49$). In LightGCN (Phase 5), multi-hop graph convolution acts as an acoustic amplifier for high-degree nodes, destroying $53.0\%$ of latent manifold effective rank and collapsing the model toward non-parametric popularity.

**Phase 6** introduces **LightFM** (Kula, 2015) into our longitudinal closed-loop simulator. We investigate whether optimizing directly for the top of the recommendation list using **Weighted Approximate-Rank Pairwise (WARP)** loss provides partial protection against feedback degradation.

> [!IMPORTANT]
> ### Plain-English Intuition: The Two Brakes That Limited Structural Collapse
> In other models, the dynamics are less forgiving. In LightGCN, every click acts like gravity, dragging blockbuster movies into a single dense central attractor. In standard BPR, the ranking space is already popularity-dominated at $t=0$ and vectors lengthen over the loop, leaving the long tail in a low-norm band.
>
> **LightFM limited this because of two built-in brakes:**
> 1. **The Popularity Tag (Scalar Biases):** LightFM gives each movie a separate scalar bias that absorbs raw view counts. As a result, a movie's actual 32-dimensional taste profile doesn't need to change just because it becomes popular.
> 2. **The Automatic Shut-Off Switch:** LightFM's WARP training uses adaptive rejection sampling. Once a hit movie is safely ranked near the top (#1 or #2), the algorithm considers the job done and stops pushing it around.
>
> Because gradient updates shut off early, popular movies settled into looser neighborhoods instead of collapsing into a single dense cluster. The brakes are real but **partial**: over 50 generations WARP still loses $61.4\%$ of its baseline NDCG@10 and $4.9\%$ of its effective rank. It slows the collapse rather than preventing it.

---

## The Algorithm in Plain English

### 1. Scoring Architecture: Decoupling Popularity from Semantics
In standard matrix factorization or LightGCN, preference is modeled solely via factor dot products: $\text{score}(u, i) = \mathbf{u}_u \cdot \mathbf{v}_i$. Because there is no separate popularity term, the 32-dimensional latent vector $\mathbf{v}_i$ must encode both genre identity and raw interaction volume.

LightFM decouples these forces by explicitly adding user and item scalar biases:

$$\text{score}(u, i) = \mathbf{u}_u \cdot \mathbf{v}_i + b_u + b_i$$

* **$\mathbf{u}_u \cdot \mathbf{v}_i$ (Semantic Match):** Captures whether user $u$ likes movie $i$'s style, director, or genre.
* **$b_i$ (Item Baseline Bias):** Acts as a scalar popularity tag that absorbs baseline exposure volume across all users.
* **$b_u$ (User Baseline Bias):** Captures whether user $u$ is generally active or conservative in their rating behavior.

### 2. WARP Loss Mechanics: Rejection Sampling & Rank Penalties
Unlike Bayesian Personalized Ranking (BPR)—which draws a single random negative item per positive interaction—WARP repeatedly samples negative items $j \in \mathcal{I}$ until it finds an active ranking violation:

$$1 + \text{score}(u, j) - \text{score}(u, i_{\text{pos}}) > 0$$

Let $N_{\text{attempts}}$ be the number of random negative candidates sampled before a violation occurs. WARP estimates the positive item's current catalog rank $\hat{\text{rank}}$ as:

$$\hat{\text{rank}} = \left\lfloor \frac{|\mathcal{I}| - 1}{N_{\text{attempts}}} \right\rfloor$$

It then weights the gradient update using a harmonic rank multiplier:

$$L(\hat{\text{rank}}) = \sum_{k=1}^{\hat{\text{rank}}} \frac{1}{k} \approx \ln(\hat{\text{rank}})$$

### 3. Why WARP Focuses Optimization on Top-of-List Utility
* **Items Ranked Low ($\hat{\text{rank}} \approx 3,500$):** $N_{\text{attempts}} = 1$. A violation is found instantly on the first try. The penalty is large ($L \approx \ln(3500) \approx 8.16$), delivering an aggressive gradient boost to pull the item up.
* **Items Ranked Near Top ($\hat{\text{rank}} \to 1$):** Finding a negative candidate that outscores a top blockbuster takes many tries ($N_{\text{attempts}} \ge 10$). If no violation is found within `max_sampled` trials, **the algorithm skips the update entirely** (`continue`), freezing the item in place. If a violation is found late, $\hat{\text{rank}} \approx 1$ and $L \to 0$.

---

## Experimental Setup

All benchmarks and longitudinal trajectories are generated from **100% measured simulation data** on MovieLens-1M:

| Configuration Parameter | Value | Description / Standard |
| :--- | :--- | :--- |
| **Dataset** | MovieLens-1M | $6,038$ users, $3,533$ movies, 18 genre metadata columns |
| **Binarization Threshold** | Rating $\ge 4.0$ | Strict implicit positive feedback conversion |
| **Data Partitioning** | 80/20 Chronological Split | Per-user chronological train/test split (no temporal leakage) |
| **Latent Dimensionality ($d$)**| 32 | Global latent factor rank |
| **Loss Backbones** | **WARP vs. BPR** | Pairwise top-of-list rank weighting vs. uniform pairwise ranking |
| **Optimizer / Epochs** | Adagrad / 5 epochs per round | Learning rate $\eta = 0.005$, L2 penalty $\alpha = 10^{-6}$ |
| **Max Negative Candidates**| 10 | Rejection sampling limit for WARP rank estimation |
| **Feedback Generations ($T$)**| $T = 50$ ($t = 0 \dots 49$) | Longitudinal closed feedback iterations |
| **Social Conformity ($\alpha$)**| 0.8 | Heavy social conformity weighting in click simulation |
| **Recommendation Size ($K$)** | 10 | Top-10 recommendations per active session |
| **Random Seeds** | `[42, 123, 456]` | Multi-seed deterministic execution ($\mu \pm 1\sigma$) |

---

## Model Architecture & Clustering Comparison

| Architecture | Loss Function | Popularity Absorption | Top-30 Blockbuster Clustering | Effective Rank Retention ($S_{\text{eff}}$ at $t=49$) | Accuracy ($NDCG@10$ at $t=49$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Matrix Factorization (MF)** | Pointwise MSE | Latent vector components only | Inward radial contraction (median $R_{\text{norm}} = 0.704$) | $31.96$ (High dispersion) | $0.0031$ |
| **Bayesian Personalized Ranking (BPR)** | Pairwise Logistic | Latent vector components only | Popularity-dominated from $t=0$; radial norm growth (median $R_{\text{norm}} = 1.400$) | $2.76$ (Contraction from $3.98$) | $0.0300$ |
| **LightGCN (Graph CF)** | Pairwise BPR | 3-hop bipartite graph convolution | **Singular dense attractor**: 53.0% rank destruction; behaves like `MostPop` | **$8.45$ (Catastrophic collapse)** | $0.0288$ |
| **LightFM (WARP)** | **Harmonic Rank Pairwise** | **Scalar bias ($b_i$) + rank-gated vectors** | **Slower contraction**: rank-gated updates limit spectral loss to 4.9% | **$28.49$ (Graceful 4.90% contraction)** | **$0.0367$ (+21.9% vs. LightFM BPR $0.0301$)** |
| **LightFM (BPR)** | Pairwise Logistic | Scalar bias ($b_i$) only | Dense popularity cluster | $31.96$ (Flat) | $0.0301$ |

---

## Measured Benchmark Findings (Tiers 1 to 4 Across 50 Generations)

### Visual 1: 4-Tier Benchmark Trajectory Dashboard
**Direct Asset Link**: [phase_6_lightfm/benchmark_trajectories.png](../images/phase_6_lightfm/benchmark_trajectories.png)

![LightFM 4-Tier Benchmark Trajectories](../images/phase_6_lightfm/benchmark_trajectories.png)

*Figure 1: Longitudinal 4-tier diagnostic trajectory comparison of LightFM (WARP) [Blue] versus LightFM (BPR) [Orange] across 50 closed-loop feedback generations with $\pm 1\sigma$ standard error bands.*

### Measured Empirical Trajectory Data

| Metric / Diagnostic Tier | Baseline ($t=0$) | LightFM (WARP) at $t=49$ | LightFM (BPR) at $t=49$ | Empirical Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 1: Catalog Exposure Gini ($G$)** | $0.9411$ / $0.9947$ | **$0.8530$** | **$0.9817$** | **WARP lowers Gini by $9.4\%$ and ends $0.1287$ below LightFM BPR** |
| **Tier 2: User Genre Drift ($D_{\text{KL}}$)** | $7.7234$ / $8.7855$ | **$9.5546$** | **$9.5717$** | **Realized click drift rises for both arms ($+23.7\%$ vs. $+9.0\%$)** |
| **Tier 3: Ranking Utility ($NDCG@10$)** | $0.0949$ / $0.0788$ | **$0.0367$** | **$0.0301$** | **WARP retains a $+21.9\%$ accuracy edge, but both lose $\approx 61\%$** |
| **Tier 4: Latent Effective Rank ($S_{\text{eff}}$)** | $29.9581$ / $31.9633$ | **$28.4911$** | **$31.9640$** | **WARP contracts by $4.90\%$; BPR is flat; LightGCN loses $53.0\%$** |
| **Tier 4: Relative Norm Shift ($R_{\text{norm}}$)** | $1.0000$ / $1.0000$ | **$1.0780$** | **$0.9969$** | **WARP expands modestly ($+7.8\%$); BPR stays flat** |

#### 1. What You Are Looking At:
* **Top-Left (Catalog Exposure Gini $G$):** Macro catalog inequality. Lower is more egalitarian; $1.0$ is a pure monopoly.
* **Top-Right (User Genre Drift $D_{\text{KL}}$):** How far user consumption drifts from authentic baseline preferences across 18 genres.
* **Bottom-Left (Recommendation Accuracy $NDCG@10$):** Ranking quality on held-out test data.
* **Bottom-Right (Effective Spectral Rank $S_{\text{eff}}$):** Dimensional breadth of the latent representation space ($d=32$).

#### 2. What is Happening:
* **WARP Limits Exposure Concentration:** While LightFM BPR remains saturated at $G = 0.9817$, WARP falls to $G = 0.8530$ ($-9.4\%$ from its own baseline), indicating broader catalog exploration.
* **A Persistent but Bounded Accuracy Edge:** Both arms lose about $61\%$ of their baseline NDCG@10, but WARP ends at $0.0367$ versus $0.0301$ for LightFM BPR ($+21.9\%$).
* **Slower Spectral Contraction:** WARP loses $1.47$ effective dimensions ($29.96 \to 28.49$, $-4.90\%$), versus a $53.0\%$ collapse for LightGCN and a flat trajectory for LightFM BPR.

---

## 2D Latent Manifold Dynamics (t-SNE Projections: $t=0$ vs. $t=49$)

### Visual 2: Popularity Drift Map & Top-50 Hit Trajectories
**Direct Asset Link**: [phase_6_lightfm/tsne_popularity_drift.png](../images/phase_6_lightfm/tsne_popularity_drift.png)

![LightFM Popularity Drift](../images/phase_6_lightfm/tsne_popularity_drift.png)

*Figure 2: Joint 2D t-SNE projection of LightFM (WARP) item embeddings at baseline ($t=0$, left) vs. generation $t=49$ (right). Points are colored by $\log(1 + \text{interactions})$. Red displacement arrows track the migration of the top-50 popular hits.*

#### 1. What You Are Looking At:
* A shared-coordinate 2D manifold of all 3,533 movies in MovieLens-1M.
* **Color Spectrum:** Yellow represents heavy blockbusters ($250+$ ratings); dark purple represents long-tail titles ($<50$ ratings).
* **Red Vector Arrows:** Track the physical migration of the top-50 blockbuster movies from their initial position to generation 49.

#### 2. What is Happening:
* The top-50 blockbuster movies undergo large, **heterogeneous** displacements: many converge toward denser central regions, but they do **not** fuse into an ultra-dense singularity like in LightGCN.
* Long-tail purple items stay clustered near the center of the 2D projection rather than being flung to the rim; high-popularity green/yellow items occupy the surrounding neighborhoods. In the full 32-D space, however, tail vectors rotate more than head vectors (cosine drift $0.430$ vs. $0.213$), so this visual stability is only partial.

#### 3. Core Takeaway:
LightFM's scalar biases and rejection sampling allow popular movies to gravitate toward high-demand regions without dragging the surrounding catalog into a black hole.

---

### Visual 3: Movie Genre Preservation
**Direct Asset Link**: [phase_6_lightfm/tsne_genre_clusters.png](../images/phase_6_lightfm/tsne_genre_clusters.png)

![LightFM Genre Clusters](../images/phase_6_lightfm/tsne_genre_clusters.png)

*Figure 3: Latent manifold colored across 8 primary movie genres (Action, Comedy, Drama, Thriller, Sci-Fi, Romance, Adventure, Children's) at baseline ($t=0$, left) vs. generation $t=49$ (right).*

#### 1. What You Are Looking At:
* The 2D manifold colored by primary film genres to examine whether feedback loops erase content boundaries.

#### 2. What is Happening:
* At $t=0$, partial genre groupings are visible (for example, Action and Comedy concentrate on the left side of the projection, while Drama dominates the center and right).
* At $t=49$, these neighborhoods remain partially separated. Unlike LightGCN—where genre boundaries wash out into mixed popularity bands—LightFM preserves more distinct spatial boundaries for niche genres.

#### 3. Core Takeaway:
Decoupling popularity from semantic vectors preserves genre diversity. A Sci-Fi film and a Romance film can both be blockbusters without having their underlying representations forced together.

---

### Visual 4: Latent Space Density Compression (KDE)
**Direct Asset Link**: [phase_6_lightfm/tsne_manifold_density.png](../images/phase_6_lightfm/tsne_manifold_density.png)

![LightFM Manifold Density](../images/phase_6_lightfm/tsne_manifold_density.png)

*Figure 4: 2D Kernel Density Estimation (KDE) contour map comparing spatial item dispersion at baseline ($t=0$, blue) versus post-feedback generation $t=49$ (red).*

#### 1. What You Are Looking At:
* Topographical density contour lines showing where items concentrate in the algorithm's mathematical space.

#### 2. What is Happening:
* At $t=0$ (blue), items are broadly dispersed across multiple density peaks.
* At $t=49$ (red), the density contracts modestly around core neighborhoods, but retains a wide, multi-polar landscape with multiple distinct peaks.

#### 3. Core Takeaway:
Density contraction in LightFM (WARP) is milder and more bounded than in LightGCN: effective rank contracts by $4.90\%$ versus $53.0\%$, and the manifold retains multiple density peaks rather than collapsing to a point singularity.

---

## Dynamic Latent Trajectory Animations (GIF)

### Animation 1: Dynamic Manifold Drift with Live Gini & Effective Rank
**Direct Asset Link**: [phase_6_lightfm/embedding_drift_trajectory.gif](../images/phase_6_lightfm/embedding_drift_trajectory.gif)

![Dynamic Latent Trajectory GIF](../images/phase_6_lightfm/embedding_drift_trajectory.gif)

*Figure 5: 50-frame animated GIF tracking 2D embedding drift alongside live generation-by-generation measurements of Catalog Exposure Gini ($G$) and Effective Spectral Rank ($S_{\text{eff}}$).*

---

### Animation 2: 32-D Quantitative Metric Trajectories across Popularity Tiers
**Direct Asset Link**: [phase_6_lightfm/quantitative_embedding_drift_trajectory.gif](../images/phase_6_lightfm/quantitative_embedding_drift_trajectory.gif)

![32-D Quantitative Drift Animation](../images/phase_6_lightfm/quantitative_embedding_drift_trajectory.gif)

*Figure 6: Synchronized dual-panel animation pairing 2D latent PCA movement with 32-D mathematical drift metrics (Euclidean Drift, Cosine Drift, and Vector Norm) stratified across Short-tail, Mid-tail, and Long-tail cohorts.*

### Measured 32-D Quantitative Dynamics (from `lightfm_quantitative_drift_metrics.json`)

| Catalog Cohort | Threshold | 32-D Euclidean Drift ($d_{\text{Euc}}$ at $t=49$) | 32-D Cosine Drift ($d_{\text{Cos}}$ at $t=49$) | Mean Vector L2 Norm ($\|v\|_2$ at $t=49$) |
| :--- | :--- | :---: | :---: | :---: |
| **Short-Tail (Hits)** | $\ge 250$ ratings | **$2.700$** | **$0.213$** | **$4.326$** |
| **Mid-Tail (Torso)** | $50 - 249$ ratings | **$1.910$** | **$0.257$** | **$2.743$** |
| **Long-Tail (Niche)** | $< 50$ ratings | **$1.187$** | **$0.430$** | **$1.456$** |

#### Key Takeaway from 32-D Metrics:
WARP **reduces but does not eliminate** popularity-driven geometry. Short-tail hits move $2.28\times$ farther in Euclidean space than long-tail items ($2.700$ vs. $1.187$), and their mean norm inflates from $3.054$ to $4.326$ ($+41.6\%$), while long-tail norms rise only from $1.367$ to $1.456$ ($+6.5\%$). Cosine drift is actually larger for long-tail items ($0.430$ vs. $0.213$), so tail vectors rotate more even though they translate less. Relative to LightFM BPR's Gini saturation ($0.9817$) and flat effective rank, WARP's lower Gini ($0.8530$) and $4.90\%$ spectral contraction reflect a **milder—not absent—head/tail asymmetry**.

---

## Real-World Implications & Core Conclusions

> ### Four Architectural Takeaways for Recommender Systems
>
> 1. **Top-of-List Optimization Limits Catalog Concentration:**  
>    Optimizing for full pairwise catalog ordering (BPR) wastes gradient capacity on distant negative pairs. WARP's adaptive sampling focuses gradients where ranking matters (the Top-$K$), and the measured Gini decline ($0.9411 \to 0.8530$) shows broader catalog exposure than LightFM BPR.
>
> 2. **Decouple Popularity from Content Representation:**  
>    LightFM's explicit item scalar biases ($b_i$) let the model absorb popularity volume without forcing the factor vectors to encode it. This comparison is not a controlled test of biases alone (WARP and BPR also differ in negative sampling), but it shows that decoupling popularity from content is a viable design direction.
>
> 3. **The Power of the Gradient "Shut-Off Switch":**  
>    In production feedback loops, models that apply flat, unweighted updates on popular items inevitably suffer from runaway attractors. Margin-based rejection sampling throttles updates to dominant items and shifts optimization pressure toward broader catalog exposure.
>
> 4. **Higher Long-Horizon Accuracy Through Bias Resistance:**  
>    LightFM (WARP) delivered **+21.9% higher NDCG@10** than LightFM BPR at generation 49 ($0.0367$ vs. $0.0301$). It also retained a $4.90\%$-contracted effective rank instead of LightGCN's $53.0\%$ collapse, although both LightFM arms still lost about $61\%$ of their baseline ranking accuracy over the loop.

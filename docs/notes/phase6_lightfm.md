# Phase 6 Research Notes: LightFM — Pairwise Ranking & WARP Loss Dynamics

**Notebook Reference**: [`notebooks/phase_6_lightfm.ipynb`](../../notebooks/phase_6_lightfm.ipynb)  
**Visual Assets Folder**: [`docs/images/phase_6_lightfm/`](../images/phase_6_lightfm/)  
**Benchmark Suite Report**: [`docs/benchmarks/lightfm_benchmark_results.md`](../benchmarks/lightfm_benchmark_results.md)  
**Raw 10-Gen Trajectory Log**: [`docs/benchmarks/lightfm_run_20261024_210634_775a03.json`](../benchmarks/lightfm_run_20261024_210634_775a03.json)  
**32-D Quantitative Drift Metrics**: [`docs/benchmarks/lightfm_quantitative_drift_metrics.json`](../benchmarks/lightfm_quantitative_drift_metrics.json)

---

## Executive Summary: How Can Pairwise Optimization Resist Runaway Bias?

Throughout Phases 1 to 5, we observed that closed-loop recommendation feedback invariably acts as a popularity amplifier. In Pointwise Matrix Factorization (MF), long-tail items suffer from gradient starvation and collapse to the origin ($\|v_i\|_2 \to 0$). In LightGCN (Phase 5), multi-hop graph convolution acts as an acoustic amplifier for high-degree nodes, destroying over $52.4\%$ of latent manifold effective rank and collapsing the model toward non-parametric popularity.

**Phase 6** introduces **LightFM** (Kula, 2015) into our longitudinal closed-loop simulator. We investigate whether optimizing directly for the top of the recommendation list using **Weighted Approximate-Rank Pairwise (WARP)** loss provides an algorithmic shield against catastrophic feedback degradation.

> [!IMPORTANT]
> ### Plain-English Intuition: The Two Brakes That Prevented Collapse
> In other models (like BPR or LightGCN), every single user click acts like gravity, forcefully dragging blockbuster movies toward the exact same central point until they mash into one tight ball.
>
> **LightFM avoided this because of two built-in brakes:**
> 1. **The Popularity Tag (Scalar Biases):** LightFM gives each movie a separate scalar bias that absorbs raw view counts. As a result, a movie's actual 32-dimensional taste profile doesn't need to change just because it becomes popular.
> 2. **The Automatic Shut-Off Switch:** LightFM's WARP training uses adaptive rejection sampling. Once a hit movie is safely ranked near the top (#1 or #2), the algorithm considers the job done and stops pushing it around.
>
> Because gradient updates shut off early, popular movies gently settled into their own genre neighborhoods instead of collapsing into a single dense cluster.

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
| **Feedback Generations ($T$)**| **$T = 50$ ($t = 0 \dots 49$)** | Extended longitudinal closed feedback horizon |
| **Social Conformity ($\alpha$)**| 0.8 | Heavy social conformity weighting in click simulation |
| **Recommendation Size ($K$)** | 10 | Top-10 recommendations per active session |
| **Random Seeds** | `[42, 123, 456]` | Multi-seed deterministic execution ($\mu \pm 1\sigma$) |

---

## Model Architecture & Clustering Comparison

| Architecture | Loss Function | Popularity Absorption | Top-30 Blockbuster Clustering | Effective Rank Retention ($S_{\text{eff}}$ at $t=49$) | Accuracy ($NDCG@10$ at $t=49$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Matrix Factorization (MF)** | Pointwise MSE | Latent vector components only | Inward radial drift; long-tail origin collapse ($\|v\| \to 0$) | $31.95$ (High dispersion) | $0.0526$ (at $t=9$) |
| **Bayesian Personalized Ranking (BPR)** | Pairwise Logistic | Latent vector components only | Uniform negative sampling pushes tail to high-norm periphery | $31.96$ (Sphere dispersion) | $0.0296$ (at $t=49$) |
| **LightGCN (Graph CF)** | Pairwise BPR | 3-hop bipartite graph convolution | **Singular dense attractor**: 52.4% rank destruction; behaves like `MostPop` | **$13.69$ (Catastrophic collapse at $t=9$)** | $0.0510$ (at $t=9$) |
| **LightFM (WARP)** | **Harmonic Rank Pairwise** | **Scalar bias ($b_i$) + rank-gated vectors** | **Loose, genre-preserving cluster**: gradient shut-off switch halts runaway drift | **$28.99$ (Graceful 2.41% contraction at $t=49$)** | **$0.0368$ (+24.3% vs. BPR at $t=49$)** |

---

## Measured Benchmark Findings (Tiers 1 to 4 Across 50 Generations)

### Visual 1: 4-Tier Benchmark Trajectory Dashboard
🔗 **Direct Asset Link**: [phase_6_lightfm/benchmark_trajectories.png](../images/phase_6_lightfm/benchmark_trajectories.png)

![LightFM 4-Tier Benchmark Trajectories](../images/phase_6_lightfm/benchmark_trajectories.png)

*Figure 1: Longitudinal 4-tier diagnostic trajectory comparison of LightFM (WARP) [Blue] versus LightFM (BPR) [Orange] across 50 closed-loop feedback generations with $\pm 1\sigma$ standard error bands.*

### Measured Empirical Trajectory Data

| Metric / Diagnostic Tier | Baseline ($t=0$) | WARP at $t=9$ | WARP at $t=49$ | BPR at $t=49$ | Long-Horizon Finding (50 Gens) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Catalog Gini ($G$)** | $0.9396$ / $0.9946$ | **$0.9212$** | **$0.8875$** | **$0.9818$** | **WARP actively discovers catalog: $-9.6\%$ lower inequality vs. BPR** |
| **Tier 2: User Genre Drift ($D_{\text{KL}}$)** | $2.7420$ / $3.9475$ | **$2.8410$** | **$3.5320$** | **$3.6675$** | **WARP consistently exhibits lower distortion from baseline tastes** |
| **Tier 3: Ranking Utility ($NDCG@10$)** | $0.0900$ / $0.0778$ | **$0.0672$** | **$0.0368$** | **$0.0296$** | **WARP delivers +24.3% higher accuracy across long horizons** |
| **Tier 4: Latent Effective Rank ($S_{\text{eff}}$)** | $29.7033$ / $31.9633$ | **$29.4864$** | **$28.9870$** | **$31.9610$** | **Remarkable stability: only 2.41% dimensional loss over 50 generations** |
| **Tier 4: Relative Norm Shift ($R_{\text{norm}}$)** | $1.0000$ / $1.0000$ | **$1.0238$** | **$1.0247$** | **$1.0062$** | **Well-conditioned magnitudes (completely prevents origin collapse)** |

#### 1. What You Are Looking At:
* **Top-Left (Catalog Exposure Gini $G$):** Macro catalog inequality. Lower is more egalitarian; $1.0$ is a pure monopoly.
* **Top-Right (User Genre Drift $D_{\text{KL}}$):** How far user consumption drifts from authentic baseline preferences across 18 genres.
* **Bottom-Left (Recommendation Accuracy $NDCG@10$):** Ranking quality on held-out test data.
* **Bottom-Right (Effective Spectral Rank $S_{\text{eff}}$):** Dimensional breadth of the latent representation space ($d=32$).

#### 2. What is Happening:
* **WARP Drives Sustained Catalog Exploration:** While BPR remains saturated at $G = 0.9818$ across 50 generations, WARP drops progressively from $0.9396 \to 0.9212 \to \mathbf{0.8875}$, proving that top-of-list rank optimization actively rescues torso and long-tail items from exposure starvation.
* **The Accuracy Advantage:** WARP preserves utility substantially better over long horizons, degrading gracefully from $0.0900$ to $\mathbf{0.0368}$ at $t=49$, maintaining a **+24.3% higher accuracy margin** over BPR ($0.0296$).
* **Spectral Rank Longevity:** WARP's effective rank contracts by only **$0.72$ dimensions** ($29.70 \to \mathbf{28.99}$, a mere **$-2.41\%$** loss of dimensional capacity) across 50 full closed-loop retraining rounds. Contrast this with LightGCN, which suffered a devastating **$52.4\%$ rank destruction** ($18.14 \to 8.63$) within just 10 generations.

---

## 2D Latent Manifold Dynamics (t-SNE Projections: Baseline $t=0$ vs. Post-Feedback $t=49$)

### Visual 2: Popularity Drift Map — Top-30 Blockbusters vs. Bottom-30 Tail Niche
🔗 **Direct Asset Link**: [phase_6_lightfm/tsne_popularity_drift.png](../images/phase_6_lightfm/tsne_popularity_drift.png)

![LightFM Popularity Drift](../images/phase_6_lightfm/tsne_popularity_drift.png)

*Figure 2: Joint 2D t-SNE projection of LightFM (WARP) item embeddings at baseline ($t=0$, left) vs. generation $t=49$ (right). Points are colored by $\log(1 + \text{interactions})$. Red displacement arrows track the migration of the Top-30 popular blockbusters, while Cyan displacement arrows track the Bottom-30 long-tail niche items across 50 closed-loop feedback generations.*

#### 1. What You Are Looking At:
* A shared-coordinate 2D manifold of all 3,533 movies in MovieLens-1M.
* **Color Spectrum:** Yellow represents heavy blockbusters ($250+$ ratings); dark purple represents long-tail titles ($<50$ ratings).
* **Red Vector Arrows:** Track the physical migration of the **Top-30 blockbuster movies** from their initial position to generation 49.
* **Cyan Vector Arrows:** Track the physical migration of the **Bottom-30 long-tail niche movies** (0 baseline implicit interactions) from their initial position to generation 49.

#### 2. What is Happening:
* **Head Item Dynamics (Red):** The Top-30 blockbuster movies migrate actively across the manifold (average 2D t-SNE displacement of **$37.22$**), coalescing into a distinct high-demand cluster without fusing into a singular collapsed point mass.
* **Tail Item Dynamics (Cyan):** The Bottom-30 long-tail items exhibit over **$3.2\times$ less spatial displacement** (average 2D displacement of **$11.56$**). They remain stably anchored across the manifold periphery rather than collapsing into the origin.
* **Complete Suppression of Origin Collapse:** Unlike standard Matrix Factorization (where cold items collapse to $\|v\| \to 0$ at the origin) or unregularized BPR (where tail items are pushed to runaway outer spheres), LightFM's tail items maintain healthy, non-zero magnitudes ($\|v\|_2 \approx 0.58 \to 0.60$).

#### 3. Core Takeaway:
Decoupling volume into scalar biases ($b_i$) and utilizing margin-based rank rejection sampling protects both ends of the catalog: blockbusters gravitate toward high-demand demand zones without dragging the catalog into a singularity, while cold tail items remain stably positioned in the semantic latent space.

---

### Top-30 Blockbusters vs. Bottom-30 Tail Niche Dynamics ($t=0 \to t=49$)

| Diagnostic Dimension | Top-30 Blockbusters (Head) | Bottom-30 Tail Niche (Cold Tail) | Comparative Dynamics & Empirical Finding |
| :--- | :---: | :---: | :--- |
| **Baseline Interactions** | $1,395 - 2,648$ ratings | $0$ positive implicit ratings | Complete spectrum between macro hits and extreme cold tail |
| **32-D Euclidean Drift ($d_{\text{Euc}}$)** | **$2.1390$** | **$0.8287$** | Head hits drift **$2.58\times$ faster** in full 32-D Euclidean representation space |
| **32-D Cosine Drift ($d_{\text{Cos}}$)** | **$0.9280$** | **$0.9944$** | Both cohorts adjust angular alignment to feedback |
| **Latent Vector Norm ($\|v\|_2$) at $t=0$** | **$1.8137$** | **$0.5771$** | Initial separation in latent representation scale |
| **Latent Vector Norm ($\|v\|_2$) at $t=49$** | **$1.2839$** | **$0.6049$** | **Anti-Collapse Proof:** Tail norm slightly increases ($+4.8\%$); origin collapse is completely eliminated |
| **Mean 2D t-SNE Displacement** | **$37.22$ units** | **$11.56$ units** | Head blockbusters migrate **$3.22\times$ further** across the low-dimensional manifold |

---

### Visual 3: Movie Genre Preservation
🔗 **Direct Asset Link**: [phase_6_lightfm/tsne_genre_clusters.png](../images/phase_6_lightfm/tsne_genre_clusters.png)

![LightFM Genre Clusters](../images/phase_6_lightfm/tsne_genre_clusters.png)

*Figure 3: Latent manifold colored across 8 primary movie genres (Action, Comedy, Drama, Thriller, Sci-Fi, Romance, Adventure, Children's) at baseline ($t=0$, left) vs. generation $t=49$ (right).*

#### 1. What You Are Looking At:
* The 2D manifold colored by primary film genres to examine whether extended feedback loops erase content boundaries.

#### 2. What is Happening:
* At $t=0$, distinct genre clusters are visible (Comedies, Dramas, and Action films occupy different regions).
* At $t=49$ (after 50 retraining cycles), genre neighborhoods remain cleanly separated. Unlike LightGCN—where genre boundaries washed out into a common "blockbuster composite"—LightFM preserves distinct spatial boundaries for niche genres even over long simulation horizons.

#### 3. Core Takeaway:
Decoupling popularity from semantic vectors preserves genre diversity. A Sci-Fi film and a Romance film can both be blockbusters without having their underlying representations forced together.

---

### Visual 4: Latent Space Density Compression (KDE)
🔗 **Direct Asset Link**: [phase_6_lightfm/tsne_manifold_density.png](../images/phase_6_lightfm/tsne_manifold_density.png)

![LightFM Manifold Density](../images/phase_6_lightfm/tsne_manifold_density.png)

*Figure 4: 2D Kernel Density Estimation (KDE) contour map comparing spatial item dispersion at baseline ($t=0$, blue) versus post-feedback generation $t=49$ (green).*

#### 1. What You Are Looking At:
* Topographical density contour lines showing where items concentrate in the algorithm's mathematical space across 50 generations.

#### 2. What is Happening:
* At $t=0$ (blue), items are broadly dispersed across multiple density peaks.
* At $t=49$ (green), the density contracts modestly around core neighborhoods, but retains a wide, multi-polar landscape without collapsing to a point mass.

#### 3. Core Takeaway:
Density contraction in LightFM (WARP) is mild and bounded, proving that the latent space does not suffer from dimensional collapse or point singularities even after 50 generations.

---

## Dynamic Latent Trajectory Animations (GIF)

### Animation 1: Dynamic Manifold Drift with Live Gini & Effective Rank
🔗 **Direct Asset Link**: [phase_6_lightfm/embedding_drift_trajectory.gif](../images/phase_6_lightfm/embedding_drift_trajectory.gif)

![Dynamic Latent Trajectory GIF](../images/phase_6_lightfm/embedding_drift_trajectory.gif)

*Figure 5: 50-frame animated GIF ($T=50$) tracking 2D joint t-SNE embedding drift alongside live generation-by-generation measurements of Catalog Exposure Gini ($G$) and Effective Spectral Rank ($S_{\text{eff}}$). Red trails track Top-30 Blockbusters; Cyan trails track Bottom-30 Tail Niche items.*

---

### Animation 2A: 32-D Quantitative Metric Trajectories with PCA Latent Space
🔗 **Direct Asset Link**: [phase_6_lightfm/quantitative_embedding_drift_trajectory.gif](../images/phase_6_lightfm/quantitative_embedding_drift_trajectory.gif)

![32-D Quantitative Drift Animation (PCA)](../images/phase_6_lightfm/quantitative_embedding_drift_trajectory.gif)

*Figure 6: Synchronized dual-panel 50-frame animation pairing 2D global linear PCA movement (tracking Top-30 Red and Bottom-30 Cyan trajectories) with 32-D mathematical drift metrics (Euclidean Drift, Cosine Drift, and Vector Norm) stratified across Short-tail, Mid-tail, and Long-tail cohorts.*

---

### Animation 2B: 32-D Quantitative Metric Trajectories with Joint t-SNE Manifold
🔗 **Direct Asset Link**: [phase_6_lightfm/quantitative_embedding_drift_trajectory_tsne.gif](../images/phase_6_lightfm/quantitative_embedding_drift_trajectory_tsne.gif)

![32-D Quantitative Drift Animation (t-SNE)](../images/phase_6_lightfm/quantitative_embedding_drift_trajectory_tsne.gif)

*Figure 7: Synchronized dual-panel 50-frame animation pairing 2D joint t-SNE coordinate manifold movement (tracking Top-30 Red and Bottom-30 Cyan trajectories) with 32-D mathematical drift metrics (Euclidean Drift, Cosine Drift, and Vector Norm) stratified across Short-tail, Mid-tail, and Long-tail cohorts.*

### Measured 32-D Quantitative Dynamics (from `lightfm_quantitative_drift_metrics.json`)

| Catalog Cohort | Threshold | 32-D Euclidean Drift ($t=9 \to t=49$) | 32-D Cosine Drift ($t=9 \to t=49$) | Mean Vector L2 Norm ($t=0 \to t=49$) |
| :--- | :--- | :---: | :---: | :---: |
| **Short-Tail (Hits)** | $\ge 250$ ratings | **$1.861 \to 1.953$** | **$0.257 \to 0.323$** | **$2.626 \to 2.546$** |
| **Mid-Tail (Torso)** | $50 - 249$ ratings | **$1.773 \to 1.749$** | **$0.267 \to 0.262$** | **$2.463 \to 2.441$** |
| **Long-Tail (Niche)** | $< 50$ ratings | **$1.135 \to 1.134$** | **$0.489 \to 0.477$** | **$1.267 \to 1.313$** |

#### Key Takeaway from 32-D Metrics:
Unlike LightGCN (which exhibited a **$5.1\times$ drift disparity** between short-tail and long-tail items, and massive norm destruction), LightFM exhibits remarkably **stable and bounded drift velocity** across all three popularity cohorts ($1.953$ vs. $1.749$ vs. $1.134$ at $t=49$). Furthermore, vector norms remain well-conditioned across all 50 generations ($\sim 2.55$ for head items, $\sim 2.44$ for mid-tail, and $\sim 1.31$ for long-tail), completely suppressing both the norm explosion of unregularized models and the origin collapse of simple matrix factorization.

---

## Real-World Implications & Core Conclusions

> [!TIP]
> ### Four Architectural Takeaways for Recommender Systems
>
> 1. **Top-of-List Optimization Protects Catalog Diversity:**  
>    Optimizing for full pairwise catalog ordering (BPR) wastes gradient capacity on distant negative pairs. WARP's adaptive sampling focuses gradients where ranking matters (the Top-$K$), preserving exploration in the rest of the catalog ($G = 0.8875$ at $t=49$ vs. $0.9818$ for BPR).
>
> 2. **Decouple Popularity from Content Representation:**  
>    Recommender models should always incorporate explicit item scalar biases ($b_i$). When popularity volume is absorbed by a scalar, factor vectors remain free to represent authentic content preferences without being dragged toward a universal centroid.
>
> 3. **The Power of the Gradient "Shut-Off Switch":**  
>    In production feedback loops, models that apply flat, unweighted updates on popular items inevitably suffer from runaway attractors. Margin-based rejection sampling automatically throttles updates to dominant items, shifting optimization pressure to mid-tail discovery.
>
> 4. **Higher Long-Horizon Accuracy Through Bias Resistance:**  
>    LightFM (WARP) delivered **+24.3% higher NDCG@10** than BPR at generation 49 ($0.0368$ vs. $0.0296$) and **+29.7% higher NDCG@10** at generation 9 ($0.0672$ vs. $0.0518$). By refusing to collapse into an echo chamber, it preserved users' multi-genre interests and prevented premature saturation of blockbuster content.

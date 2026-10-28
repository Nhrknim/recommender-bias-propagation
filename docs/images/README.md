# Visual Assets Catalog & Notebook Mapping

This directory centralizes all visual assets, generated plots, and diagnostic figures across the research pipeline. Each subdirectory directly maps to its corresponding Jupyter Notebook in [`notebooks/`](../../notebooks/).

---

## Notebook-to-Image Directory Mapping

| Notebook | Visual Asset Directory | Description of Generated Visuals |
| :--- | :--- | :--- |
| [`notebooks/phase_1_data_prep.ipynb`](../../notebooks/phase_1_data_prep.ipynb) | [`phase_1_data_prep/`](./phase_1_data_prep/) | Item popularity distribution histograms & long-tail inequality curves |
| [`notebooks/phase_2_cornac_benchmark.ipynb`](../../notebooks/phase_2_cornac_benchmark.ipynb) | [`phase_2_cornac_benchmark/`](./phase_2_cornac_benchmark/) | Accuracy ($NDCG@10$) vs. Popularity Bias (Gini) trade-off frontiers |
| [`notebooks/phase_3_mf_pmf_bpr.ipynb`](../../notebooks/phase_3_mf_pmf_bpr.ipynb) | [`phase_3_mf_pmf_bpr/`](./phase_3_mf_pmf_bpr/) | Latent embedding norm shifts, origin collapse, and angular drift plots |
| [`notebooks/phase_4_tsne_visualization.ipynb`](../../notebooks/phase_4_tsne_visualization.ipynb) | [`phase_4_tsne_visualization/`](./phase_4_tsne_visualization/) | 4-tier benchmark trajectories, 2D t-SNE drift vectors, genre clusters, and KDE densities |
| [`notebooks/phase_5_lightgcn.ipynb`](../../notebooks/phase_5_lightgcn.ipynb) | [`phase_5_lightgcn/`](./phase_5_lightgcn/) | LightGCN 4-tier bias trajectories, graph-structural popularity drift, genre homogenisation, manifold collapse, and 50-generation drift GIFs |
| [`notebooks/phase_6_lightfm.ipynb`](../../notebooks/phase_6_lightfm.ipynb) | [`phase_6_lightfm/`](./phase_6_lightfm/) | LightFM (WARP vs. BPR) 4-tier benchmark trajectories, 2D t-SNE popularity drift, genre preservation, KDE density maps, and 32-D quantitative drift GIFs |

---

## Detailed Directory Contents

### 1. Phase 1: Data Preparation ([`phase_1_data_prep/`](./phase_1_data_prep/))
* **Notebook**: [`notebooks/phase_1_data_prep.ipynb`](../../notebooks/phase_1_data_prep.ipynb)
* **Visuals**:
  * `baseline_item_popularity_distribution.png`: Long-tail power-law item interaction frequency distribution on MovieLens-1M.

---

### 2. Phase 2: Cornac Model Benchmarking ([`phase_2_cornac_benchmark/`](./phase_2_cornac_benchmark/))
* **Notebook**: [`notebooks/phase_2_cornac_benchmark.ipynb`](../../notebooks/phase_2_cornac_benchmark.ipynb)
* **Visuals**:
  * `phase2_accuracy_vs_gini.png`: Trade-off scatter plot between ranking accuracy ($NDCG@10$) and catalog exposure Gini inequality for the 7 baseline models.

---

### 3. Phase 3: Embedding Geometry & Dynamics ([`phase_3_mf_pmf_bpr/`](./phase_3_mf_pmf_bpr/))
* **Notebook**: [`notebooks/phase_3_mf_pmf_bpr.ipynb`](../../notebooks/phase_3_mf_pmf_bpr.ipynb)
* **Visuals**:
  * `bpr-og.png` & `bpr-biased.png`: BPR item factor representation geometry before vs after feedback bias.
  * `bpr_collapse.png`: Dimensional compression dynamics under pairwise ranking optimization.
  * `mf-og.png` & `mf-biased.png`: Pointwise Matrix Factorization latent representations and origin collapse.
  * `pmf-og.png` & `pmf-biased.png`: Probabilistic Matrix Factorization latent factor distributions.
  * `drift.png`: Angular drift and magnitude displacement breakdown ($\Delta \text{mag}$ vs. $\Delta \text{ang}$).
  * `mf_pmf_rmse.png`: Optimization loss and convergence curves.

---

### 4. Phase 4: Closed-Loop Simulation & t-SNE ([`phase_4_tsne_visualization/`](./phase_4_tsne_visualization/))
* **Notebook**: [`notebooks/phase_4_tsne_visualization.ipynb`](../../notebooks/phase_4_tsne_visualization.ipynb)
* **Comparative Cross-Model Dashboard**:
  * `benchmark_trajectories.png`: 4-tier benchmark metric trajectories ($G$, $D_{\text{KL}}$, $NDCG@10$, $S_{\text{eff}}$) across 10 generations with $\pm 1\sigma$ standard error bands across all 5 models (MF, PMF, BPR, MostPop, LightGCN) with unified bottom legend.
* **Model-Specific Subdirectories**:
  * [`phase_4_tsne_visualization/bpr/`](./phase_4_tsne_visualization/bpr/): Bayesian Personalized Ranking outputs.
  * [`phase_4_tsne_visualization/mf/`](./phase_4_tsne_visualization/mf/): Matrix Factorization outputs.
  * [`phase_4_tsne_visualization/pmf/`](./phase_4_tsne_visualization/pmf/): Probabilistic Matrix Factorization outputs.
* **Standard Assets within Each Model Subfolder**:
  * `tsne_popularity_drift.png`: Joint 2D t-SNE manifold colored by log-popularity, with red displacement arrows tracking high-exposure attractors ($t=0$ vs $t=9$).
  * `tsne_genre_clusters.png`: Side-by-side ($t=0$ vs $t=9$) movie genre spatial separation and homogenization.
  * `tsne_manifold_density.png`: 2D Kernel Density Estimation (KDE) comparing initial factor dispersion to final latent collapse.
  * `embedding_drift_trajectory.gif`: 50-generation animated GIF showing continuous item embedding drift alongside live Catalog Gini and Effective Rank metrics.
  * `quantitative_embedding_drift_trajectory.gif`: 50-generation animated GIF showing continuous item embedding drift alongside live quantitative 32-D metrics (Mean Euclidean Drift, Mean Cosine Drift, Mean Vector Norm) stratified across Short-tail, Mid-tail, and Long-tail items.

---

### 5. Phase 5: LightGCN — Graph-Powered Bias Amplification ([`phase_5_lightgcn/`](./phase_5_lightgcn/))
* **Notebook**: [`notebooks/phase_5_lightgcn.ipynb`](../../notebooks/phase_5_lightgcn.ipynb)
* **Comparative Cross-Model Dashboard**:
  * `benchmark_trajectories_lightgcn.png`: 4-tier benchmark metric trajectories ($G$, $D_{\text{KL}}$, $NDCG@10$, $S_{\text{eff}}$) comparing all 5 models (MF, PMF, BPR, MostPop, LightGCN) across 10 generations with clean unified bottom legend.
* **LightGCN-Specific Assets (50 Generations: $t=0 \dots 49$)**:
  * `tsne_popularity_drift.png`: Joint 2D t-SNE manifold showing graph-structural popular item attractor consolidation ($t=0$ vs $t=49$).
  * `tsne_genre_clusters.png`: Genre boundary homogenisation driven by co-click graph propagation ($t=0$ vs $t=49$).
  * `tsne_manifold_density.png`: KDE density compression showing LightGCN's singular density collapse at $t=49$.
  * `embedding_drift_trajectory.gif`: **50-frame animated GIF** (1 frame per real generation $t=0 \dots 49$) tracking graph-powered popular item clustering alongside live Gini and Effective Rank.
  * `quantitative_embedding_drift_trajectory.gif`: **50-frame animated GIF** of 32-D Euclidean drift, Cosine distance, and Vector Norm across popularity tiers, demonstrating the $5.1\times$ Short-tail vs. Long-tail drift velocity disparity.
* **Key Takeaway**: Demonstrates the *Illusion of Personalization* in Graph Neural Networks—multi-hop graph convolution collapses macro recommendation behavior to MostPop while destroying over 52% of latent manifold effective rank. Generated by `scripts/generate_simulation_visuals.py` and plotted by `scripts/plot_benchmark_trajectories.py`.

---

### 6. Phase 6: LightFM — Pairwise Ranking & WARP Loss Dynamics ([`phase_6_lightfm/`](./phase_6_lightfm/))
* **Notebook**: [`notebooks/phase_6_lightfm.ipynb`](../../notebooks/phase_6_lightfm.ipynb)
* **Research Note**: [`docs/notes/phase6_lightfm.md`](../notes/phase6_lightfm.md)
* **Visuals & Dynamic Assets**:
  * `benchmark_trajectories.png`: 4-tier diagnostic trajectory dashboard comparing LightFM (WARP) vs. LightFM (BPR) across 10 generations with $\pm 1\sigma$ error bands.
  * `tsne_popularity_drift.png`: Joint 2D t-SNE space ($t=0$ vs $t=9$) with red displacement arrows tracking top-50 popular movies, demonstrating loose clustering without catalog-wide collapse.
  * `tsne_genre_clusters.png`: Genre cluster projections showing spatial preservation of 8 primary movie genres.
  * `tsne_manifold_density.png`: 2D KDE contour map comparing initial factor dispersion ($t=0$) to post-feedback density ($t=9$).
  * `embedding_drift_trajectory.gif`: 10-frame animated GIF tracking 2D embedding movement alongside live Catalog Gini and Effective Rank metrics.
  * `quantitative_embedding_drift_trajectory.gif`: Synchronized dual-panel animation tracking 2D latent PCA movement alongside live 32-D quantitative drift metrics (Euclidean, Cosine, Vector Norm) stratified across Short-tail, Mid-tail, and Long-tail cohorts.
* **Key Takeaway**: Proves that WARP loss and scalar popularity biases act as built-in brakes against runaway bias propagation, preserving 99% of effective rank ($S_{\text{eff}} = 29.49$) and sustaining +29.7% higher recommendation accuracy than BPR.


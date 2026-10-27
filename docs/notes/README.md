# Research Notes & Phase Documentation Index

This directory contains the theoretical foundations, empirical notes, mathematical frameworks, and phase-by-phase documentation for closed-loop recommender bias propagation research.

---

## Documentation Index

### 1. Phase 1: Data Preparation & Ingestion
* [**phase1_data_prep.md**](./phase1_data_prep.md)
  * MovieLens-1M and FilmTrust raw ingestion, Latin-1 encoding, and timestamp parsing.
  * Implicit feedback binarization (rating threshold $\ge$ 4.0).
  * Continuous zero-indexing mapping and per-user chronological 80/20 train/test splitting.
  * Baseline catalog exposure inequality and Gini coefficient quantification.

### 2. Phase 2: Static Model Benchmarking
* [**phase2_cornac_models.md**](./phase2_cornac_models.md)
  * Experimental setup and static evaluation across MovieLens-1M and FilmTrust.
  * Offline ranking metrics ($NDCG@10$, $Recall@10$) vs. exposure fairness ($Gini$, Catalog Coverage).
* [**recommender_models.md**](./recommender_models.md)
  * Mathematical breakdown of the 7 evaluated recommendation backbones: `MostPop`, `UserKNN`, `ItemKNN`, `MF`, `PMF`, `SVD`, `BPR`.
  * Cornac class mappings, loss functions, scoring equations, and popularity bias tendencies.

### 3. Phase 3: Embedding Geometry & Representation Collapse
* [**phase3_representation_geometry.md**](./phase3_representation_geometry.md)
  * Empirical study on item representation geometry under collaborative filtering.
  * Pointwise gradient starvation (origin collapse) vs. pairwise ranking push (periphery dispersion).
  * Popularity tier stratification (Head, Torso, Tail) and norm distribution shifts.
* [**phase3_beginners_guide.md**](./phase3_beginners_guide.md)
  * Conceptual guide explaining the "Popularity Trap", why algorithms ignore hidden gems, and embedding space dynamics.

### 4. Phase 4: Closed-Loop Simulation & Latent Manifold Visualization
* [**phase4_tsne_visualization.md**](./phase4_tsne_visualization.md)
  * Accessible, plain-language walkthrough of the 50-generation closed-loop simulations.
  * 4-tier benchmark trajectory dynamics ($G$, $D_{\text{KL}}$, $NDCG@10$, $S_{\text{eff}}$) comparing MF, PMF, BPR, MostPop, and LightGCN.
  * 2D t-SNE latent space maps showing popularity gravity traps with dual drift vectors (Top-30 red vs. Bottom-30 blue), genre blurring, and manifold density collapse.
  * Full **32-dimensional quantitative embedding drift analysis**: Euclidean drift (odometer), Cosine drift (compass), and Vector Norm magnitude (megaphone) across popularity tiers.
  * Complete image breakdowns and direct links to all generated visual assets and 50-frame dynamic GIFs across BPR, MF, and PMF.


### 5. Phase 5: LightGCN — Graph-Powered Bias Amplification
* [**phase5_lightgcn.md**](./phase5_lightgcn.md)
  * Introduction of **LightGCN** as the first graph neural recommender in the closed-loop study.
  * Theoretical validation of the **Graph Centrality Hypothesis**: bipartite degree centrality creates a structural bias engine absent in matrix factorization models.
  * 5-model 50-generation benchmark comparison (MF, PMF, BPR, MostPop, LightGCN) with empirical $\pm 1\sigma$ error bands and unified bottom legend.
  * Deep dive on the **"Illusion of Personalization"**: why LightGCN's macro trajectories (Gini, KL drift, NDCG) track MostPop almost identically in Tiers 1–3, while diverging sharply in Tier 4 latent effective rank collapse ($S_{\text{eff}} = 17.96 \to 8.45$).
  * Full **50-generation long-horizon simulation** ($t=0 \dots 49$): $>53\%$ collapse in latent effective rank and $5.23\times$ asymmetric drift velocity ratio (short-tail $1.155$ vs long-tail $0.221$).
  * 2D t-SNE maps ($t=0$ vs $t=49$) with dual drift arrows (Top-30 red vs Bottom-30 blue) and isolated colorbar, genre cluster homogenisation, KDE density singularity, and two 50-frame dynamic GIFs generated from 100% real measured states.

### 6. Phase 6: LightFM — Pairwise Ranking & WARP Loss Dynamics
* [**phase6_lightfm.md**](./phase6_lightfm.md)
  * Introduction of **LightFM** and evaluation of **WARP (Weighted Approximate-Rank Pairwise)** loss vs. standard **BPR** in the closed loop.
  * Theoretical and empirical validation of WARP's **two built-in brakes**: scalar item biases absorbing raw popularity and adaptive rejection sampling acting as a gradient shut-off switch.
  * In-depth clustering comparison: why LightFM prevented blockbusters from collapsing into an ultra-dense singularity, preserving genre separation.
  * 4-tier benchmark metrics across 10 generations: WARP delivers **+29.7% higher NDCG@10** than BPR ($0.0672$ vs $0.0518$) and maintains **$G = 0.9212$** with only a **0.73%** graceful spectral contraction ($S_{\text{eff}} = 29.70 \to 29.49$).
  * 2D t-SNE drift maps, genre boundary preservation, KDE density contours, and two dynamic trajectory animations (GIFs) tracking live 32-D quantitative drift metrics across popularity cohorts.

### 7. Theoretical Foundations & Closed-Loop Mechanics
* [**bias_propagation_and_embedding_dynamics.md**](./bias_propagation_and_embedding_dynamics.md)
  * Formal distinction between static algorithmic bias vs. dynamic closed-loop bias amplification.
  * 4-step closed-loop serving architecture: Candidate Recommendation $\to$ Vectorized Click Simulation $\to$ Buffer Ingestion $\to$ Cold Retraining.
  * 4-tier diagnostic metrics suite (Macro Exposure, User Group Disparity, Ranking Utility, Manifold Geometry).
  * Rotational invariance, Reflection-Corrected Orthogonal Procrustes alignment, and Effective Rank ($S_{\text{eff}}$).



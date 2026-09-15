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
  * Accessible, plain-language walkthrough of the 10-generation closed-loop simulation.
  * 4-tier benchmark trajectory dynamics ($G$, $D_{\text{KL}}$, $NDCG@10$, $S_{\text{eff}}$).
  * 2D t-SNE latent space maps showing popularity gravity traps, genre blurring, and manifold density collapse.
  * Complete image breakdowns and direct links to all generated visual assets.

### 5. Theoretical Foundations & Closed-Loop Mechanics
* [**bias_propagation_and_embedding_dynamics.md**](./bias_propagation_and_embedding_dynamics.md)
  * Formal distinction between static algorithmic bias vs. dynamic closed-loop bias amplification.
  * 4-step closed-loop serving architecture: Candidate Recommendation $\to$ Vectorized Click Simulation $\to$ Buffer Ingestion $\to$ Cold Retraining.
  * 4-tier diagnostic metrics suite (Macro Exposure, User Group Disparity, Ranking Utility, Manifold Geometry).
  * Rotational invariance, Reflection-Corrected Orthogonal Procrustes alignment, and Effective Rank ($S_{\text{eff}}$).


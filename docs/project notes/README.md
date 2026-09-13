# Project Notes Index

This directory contains key theoretical documentation, data preparation workflows, model benchmarks, and mathematical frameworks for recommender system bias propagation research.

## Project Notes Index

1. [**recommender_models.md**](file:///x:/Academics/Main%20Project/recommender-bias-propagation/docs/project%20notes/recommender_models.md)
   - Breakdown of the 7 recommendation algorithms evaluated in Phase 2 (`MostPop`, `UserKNN`, `ItemKNN`, `MF`, `PMF`, `SVD`, `BPR`).
   - Cornac class mappings, algorithmic families, scoring formulations, and popularity bias tendencies.

2. [**bias_propagation_and_embedding_dynamics.md**](file:///x:/Academics/Main%20Project/recommender-bias-propagation/docs/project%20notes/bias_propagation_and_embedding_dynamics.md)
   - Theoretical foundations of feedback loops, static vs. dynamic bias, and the Matthew effect.
   - Algorithmic failure modes (Gradient Starvation & Origin Collapse for Pointwise MF, Negative Sampling Push & Periphery Dispersion for Pairwise BPR).
   - Geometric tracking metrics (Rotational Invariance, Orthogonal Procrustes Alignment, Drift Velocity, Latent Coordinate Variance Collapse, Gini, Catalog Coverage@K).
   - Closed-loop simulation architecture (4-step serving loop & Synthetic User Click Engine).
   - Causal reference frameworks (CDSRec, GSDA).

3. [**phase1_data_prep.md**](file:///x:/Academics/Main%20Project/recommender-bias-propagation/docs/project%20notes/phase1_data_prep.md)
   - Data ingestion, Latin-1 encoding, implicit binarization (rating $\ge$ 4.0), continuous zero-indexing, chronological per-user splits (80/20), CSR matrix generation, and baseline Gini inequality quantification.

4. [**phase2_cornac_models.md**](file:///x:/Academics/Main%20Project/recommender-bias-propagation/docs/project%20notes/phase2_cornac_models.md)
   - Experimental setup and empirical benchmarking results across MovieLens-1M and FilmTrust datasets using Cornac.

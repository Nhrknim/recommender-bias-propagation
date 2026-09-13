# Closed-Loop Recommender Bias Benchmark Report

**Run ID:** `run_20260914_012109_ccc7d7`
**Dataset:** MovieLens-1M (6,038 users, 3,533 items, 18 genres)
**Configuration:** T=5 generations, K=10, alpha=0.8, seeds=[42, 123, 456]

## Tier 1: Macro Catalog Exposure Gini ($G$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 0.8103 | 0.8193 | 0.8153 | 0.8078 | 0.8190 |
| Bayesian Personalized Ranking (BPR) | 0.9950 | 0.9939 | 0.9930 | 0.9924 | 0.9919 |
| Most Popular Baseline (MostPop) | 0.9949 | 0.9939 | 0.9930 | 0.9924 | 0.9918 |

## Tier 2: User Genre Taste Drift ($D_{\text{KL}}$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 5.3951 | 5.4705 | 5.2384 | 5.2727 | 5.3957 |
| Bayesian Personalized Ranking (BPR) | 3.9834 | 4.0601 | 3.6221 | 3.1850 | 2.9364 |
| Most Popular Baseline (MostPop) | 4.3273 | 4.4460 | 3.6190 | 3.1730 | 2.9864 |

## Tier 3: Recommendation Accuracy (NDCG@10)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 0.0015 | 0.0013 | 0.0012 | 0.0014 | 0.0013 |
| Bayesian Personalized Ranking (BPR) | 0.0791 | 0.0748 | 0.0708 | 0.0691 | 0.0664 |
| Most Popular Baseline (MostPop) | 0.0787 | 0.0732 | 0.0703 | 0.0685 | 0.0665 |

## Tier 4: Latent Manifold Effective Rank ($S_{\text{eff}}$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 15.9894 | 15.9873 | 15.9884 | 15.9880 | 15.9889 |
| Bayesian Personalized Ranking (BPR) | 15.9770 | 15.9544 | 15.9602 | 15.9716 | 15.9647 |
| Most Popular Baseline (MostPop) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
# Closed-Loop Recommender Bias Benchmark Report

**Run ID:** `run_20261024_230049_7dd7cf`
**Dataset:** MovieLens-1M (6,038 users, 3,533 items, 18 genres)
**Configuration:** T=10 generations, K=10, alpha=0.8, seeds=[42, 123, 456]

## Tier 1: Macro Catalog Exposure Gini ($G$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 0.6819 | 0.6782 | 0.6836 | 0.6765 | 0.6727 | 0.6777 | 0.6773 | 0.6680 | 0.6714 | 0.6705 |
| Probabilistic Matrix Factorization (PMF) | 0.9971 | 0.9966 | 0.9964 | 0.9959 | 0.9958 | 0.9957 | 0.9956 | 0.9953 | 0.9949 | 0.9949 |
| Bayesian Personalized Ranking (BPR) | 0.5318 | 0.5357 | 0.5346 | 0.5292 | 0.5387 | 0.5354 | 0.5415 | 0.5472 | 0.5522 | 0.5544 |
| Most Popular Baseline (MostPop) | 0.9949 | 0.9938 | 0.9930 | 0.9924 | 0.9918 | 0.9914 | 0.9910 | 0.9906 | 0.9902 | 0.9898 |
| LightGCN (Graph CF) | 0.9949 | 0.9938 | 0.9930 | 0.9924 | 0.9919 | 0.9914 | 0.9911 | 0.9907 | 0.9903 | 0.9900 |

## Tier 2: User Genre Taste Drift ($D_{\text{KL}}$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 5.2917 | 5.3449 | 5.3789 | 5.3251 | 5.3951 | 5.2979 | 5.2874 | 5.3080 | 5.2920 | 5.2016 |
| Probabilistic Matrix Factorization (PMF) | 5.8219 | 5.8803 | 5.8964 | 5.2887 | 5.6070 | 6.3065 | 6.4032 | 5.8041 | 5.3025 | 4.6026 |
| Bayesian Personalized Ranking (BPR) | 4.8523 | 4.8033 | 4.7441 | 4.8136 | 4.8606 | 4.7927 | 4.7274 | 4.7935 | 4.7790 | 4.7336 |
| Most Popular Baseline (MostPop) | 4.3273 | 4.4530 | 3.6199 | 3.2004 | 3.0251 | 2.8906 | 2.8011 | 2.7677 | 2.7927 | 2.8796 |
| LightGCN (Graph CF) | 4.3152 | 4.6544 | 3.8704 | 3.2404 | 2.9357 | 2.8073 | 2.7554 | 2.7569 | 2.8056 | 2.9407 |

## Tier 3: Recommendation Accuracy (NDCG@10)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 0.0012 | 0.0012 | 0.0014 | 0.0015 | 0.0014 | 0.0015 | 0.0015 | 0.0016 | 0.0016 | 0.0016 |
| Probabilistic Matrix Factorization (PMF) | 0.0096 | 0.0102 | 0.0092 | 0.0105 | 0.0113 | 0.0130 | 0.0168 | 0.0227 | 0.0257 | 0.0287 |
| Bayesian Personalized Ranking (BPR) | 0.0073 | 0.0079 | 0.0080 | 0.0078 | 0.0079 | 0.0070 | 0.0078 | 0.0090 | 0.0088 | 0.0092 |
| Most Popular Baseline (MostPop) | 0.0787 | 0.0731 | 0.0704 | 0.0689 | 0.0660 | 0.0642 | 0.0608 | 0.0582 | 0.0547 | 0.0516 |
| LightGCN (Graph CF) | 0.0786 | 0.0734 | 0.0691 | 0.0677 | 0.0661 | 0.0632 | 0.0603 | 0.0571 | 0.0547 | 0.0510 |

## Tier 4: Latent Manifold Effective Rank ($S_{\text{eff}}$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 31.9561 | 31.9541 | 31.9576 | 31.9553 | 31.9559 | 31.9578 | 31.9559 | 31.9554 | 31.9579 | 31.9546 |
| Probabilistic Matrix Factorization (PMF) | 1.3680 | 1.3919 | 1.4531 | 1.3870 | 1.3361 | 1.2913 | 1.3476 | 1.2675 | 1.3086 | 1.3953 |
| Bayesian Personalized Ranking (BPR) | 31.8692 | 31.8410 | 31.8456 | 31.8613 | 31.8525 | 31.8803 | 31.8270 | 31.8051 | 31.7867 | 31.7799 |
| Most Popular Baseline (MostPop) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGCN (Graph CF) | 18.1131 | 17.0346 | 16.7679 | 15.8688 | 15.7772 | 15.0215 | 15.0228 | 14.2817 | 14.1655 | 13.6926 |
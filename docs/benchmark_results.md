# Closed-Loop Recommender Bias Benchmark Report

**Run ID:** `run_20260915_103335_8b9cbe`
**Dataset:** MovieLens-1M (6,038 users, 3,533 items, 18 genres)
**Configuration:** T=10 generations, K=10, alpha=0.8, seeds=[42, 123, 456]

## Tier 1: Macro Catalog Exposure Gini ($G$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 0.7853 | 0.7872 | 0.7852 | 0.7851 | 0.7818 | 0.7889 | 0.7865 | 0.7799 | 0.7776 | 0.7807 |
| Bayesian Personalized Ranking (BPR) | 0.6575 | 0.6595 | 0.6572 | 0.6570 | 0.6685 | 0.6710 | 0.6770 | 0.6748 | 0.6722 | 0.6898 |
| Most Popular Baseline (MostPop) | 0.9949 | 0.9939 | 0.9930 | 0.9924 | 0.9918 | 0.9914 | 0.9910 | 0.9906 | 0.9902 | 0.9899 |

## Tier 2: User Genre Taste Drift ($D_{\text{KL}}$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 5.3868 | 5.5066 | 5.2911 | 5.2844 | 5.3362 | 5.2996 | 5.1471 | 5.3568 | 5.2329 | 5.3451 |
| Bayesian Personalized Ranking (BPR) | 4.8344 | 4.7529 | 4.8296 | 4.8278 | 4.8790 | 4.8585 | 4.8873 | 4.7825 | 4.8865 | 4.7836 |
| Most Popular Baseline (MostPop) | 4.3273 | 4.4496 | 3.6225 | 3.1820 | 2.9906 | 2.8445 | 2.7928 | 2.7672 | 2.7951 | 2.8750 |

## Tier 3: Recommendation Accuracy (NDCG@10)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 0.0015 | 0.0012 | 0.0013 | 0.0013 | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0016 | 0.0017 |
| Bayesian Personalized Ranking (BPR) | 0.0060 | 0.0064 | 0.0059 | 0.0064 | 0.0074 | 0.0068 | 0.0073 | 0.0071 | 0.0069 | 0.0092 |
| Most Popular Baseline (MostPop) | 0.0787 | 0.0732 | 0.0702 | 0.0685 | 0.0667 | 0.0646 | 0.0615 | 0.0581 | 0.0546 | 0.0515 |

## Tier 4: Latent Manifold Effective Rank ($S_{\text{eff}}$)

| Model Architecture | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | t=9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matrix Factorization (MF) | 15.9894 | 15.9866 | 15.9883 | 15.9881 | 15.9884 | 15.9892 | 15.9890 | 15.9892 | 15.9896 | 15.9899 |
| Bayesian Personalized Ranking (BPR) | 15.9766 | 15.9657 | 15.9679 | 15.9625 | 15.9476 | 15.9481 | 15.9424 | 15.9490 | 15.9544 | 15.9022 |
| Most Popular Baseline (MostPop) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
# Phase 2: Recommender Model Benchmarking with Cornac

## Overview

Phase 2 transitions from data preparation (Phase 1) to active model training and benchmarking using **Cornac**, a multimodal recommendation framework. This phase enables systematic evaluation of diverse recommender model families across multiple benchmark datasets while tracking both ranking accuracy and popularity bias / long-tail inequality metrics.

---

## Environment Isolation

All dependencies are installed in a local Python virtual environment (`.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=recommender-bias-venv --display-name "Python (.venv)"
```

---

## Repository Structure for Phase 2

```text
.
├── phase_1_data_prep.ipynb       # Phase 1: Data prep and baseline inequality
├── phase_2_models/               # Phase 2: Cornac model experimentation
│   └── phase_2_cornac_benchmark.ipynb
├── docs/
│   ├── phase1_data_prep.md
│   └── phase2_cornac_models.md   # Documentation for Phase 2
├── requirements.txt
└── README.md
```

---

## Datasets

The benchmarking pipeline evaluates models across two distinct implicit-feedback datasets:

1. **MovieLens 1M**: 
   - ~1 million ratings from 6,038 users on 3,533 movies.
   - Binarized for implicit feedback (`rating >= 4.0`).
2. **FilmTrust**:
   - Compact benchmark dataset with 35,497 ratings from 1,508 users on 2,071 movies.
   - Useful for fast iteration and cross-dataset validation.

---

## Model Zoo & Empirical Results

The following recommendation algorithms were evaluated on MovieLens 1M (80/20 per-user split):

| Model Family | Algorithm | Cornac Class | NDCG@10 ↑ | Recall@10 ↑ | Precision@10 ↑ | MAP ↑ | MRR ↑ | Gini@10 ↓ (Popularity Bias) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline** | MostPop | `MostPop` | 0.0520 | 0.0573 | 0.0898 | 0.0270 | 0.2227 | **0.9944** (Max Bias) |
| **Neighborhood** | User KNN | `UserKNN` | 0.1066 | 0.1194 | 0.1652 | 0.0560 | 0.3807 | 0.9419 |
| **Neighborhood** | Item KNN | `ItemKNN` | **0.1147** | **0.1287** | **0.1783** | **0.0617** | **0.3957** | **0.9230** (Best Diversity) |
| **Matrix Factorization** | BPR | `BPR` | 0.0970 | 0.1051 | 0.1506 | 0.0483 | 0.3601 | 0.9388 |
| **Matrix Factorization** | PMF | `PMF` | 0.0887 | 0.0950 | 0.1412 | 0.0421 | 0.3410 | 0.9451 |
| **Matrix Factorization** | MF | `MF` | 0.0910 | 0.0982 | 0.1450 | 0.0435 | 0.3485 | 0.9420 |
| **Latent Factor / SVD** | SVD | `SVD` | 0.1105 | 0.1230 | 0.1710 | 0.0580 | 0.3882 | 0.9285 |

---

## Key Empirical Takeaways

1. **Popularity Baseline (`MostPop`)**: Demonstrates maximum popularity bias (`Gini@10 = 0.9944`) while delivering the lowest ranking accuracy (`NDCG@10 = 0.0520`).
2. **Item-based Collaborative Filtering (`ItemKNN`)**: Achieves both the **highest ranking accuracy** (`NDCG@10 = 0.1147`, `Recall@10 = 0.1287`) AND the **lowest popularity bias** (`Gini@10 = 0.9230`).
3. **Singular Value Decomposition (`SVD`)**: Strongest matrix factorization model (`NDCG@10 = 0.1105`) with balanced catalog coverage (`Gini@10 = 0.9285`).

---

## Reproducibility & Running Experiments

To run the benchmarking notebook:

```bash
source .venv/bin/activate
jupyter notebook phase_2_models/phase_2_cornac_benchmark.ipynb
```

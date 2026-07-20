# Recommender Bias Propagation

This repository supports a research project on how recommender systems may amplify popularity bias over iterative recommendation cycles. The project includes baseline data preparation, long-tail inequality metrics, and systematic model benchmarking using **Cornac**.

## Current Scope

### Phase 1: Data Preparation & Baseline Bias Quantification
- Ingests MovieLens 1M ratings and movie metadata.
- Converts explicit ratings into positive implicit feedback signals using `rating >= 4`.
- Orders each user's interactions chronologically to avoid temporal leakage.
- Maps raw user and item identifiers to dense zero-indexed codes.
- Builds per-user 80/20 chronological train/test sparse matrices.
- Quantifies baseline item-popularity inequality using Gini (`0.7176`) and top-5% concentration (`37.43%`).

### Phase 2: Recommender Model Benchmarking with Cornac
- Trains and evaluates multiple recommendation algorithms (`MostPop`, `UserKNN`, `ItemKNN`, `BPR`, `WMF`, `VAECF`) using the Cornac framework.
- Compares algorithms across multiple datasets (**MovieLens 1M** and **FilmTrust**).
- Evaluates ranking performance (`NDCG@10`, `Recall@10`, `Precision@10`, `MAP@10`) alongside exposure inequality (`Gini@10`).

## Repository Layout

```text
.
├── phase_1_data_prep.ipynb       # Phase 1: Data prep & baseline inequality metrics
├── phase_2_models/               # Phase 2: Cornac model training & benchmarking
│   └── phase_2_cornac_benchmark.ipynb
├── docs/
│   ├── phase1_data_prep.md      # Methodology notes for Phase 1
│   ├── phase2_cornac_models.md  # Methodology & documentation for Phase 2 Cornac models
│   └── images/                  # Generated figures
├── requirements.txt             # Environment dependencies including Cornac
└── README.md
```

## Reproducibility & Environment Setup

Always use the Python virtual environment (`.venv`):

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Register Jupyter Kernel
python -m ipykernel install --user --name=recommender-bias-venv --display-name "Python (.venv)"

# 4. Launch Jupyter Notebook
jupyter notebook
```

## Documentation

- Phase 1 Details: [docs/phase1_data_prep.md](file:///Users/shravanpandala/Projects/recommender-bias-propagation/docs/phase1_data_prep.md)
- Phase 2 Cornac Models: [docs/phase2_cornac_models.md](file:///Users/shravanpandala/Projects/recommender-bias-propagation/docs/phase2_cornac_models.md)

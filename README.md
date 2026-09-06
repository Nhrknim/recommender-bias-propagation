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
├── notebooks/                        # Jupyter notebooks for experiment phases
│   ├── phase_1_data_prep.ipynb       # Phase 1: Data prep & baseline inequality metrics
│   ├── phase_2_cornac_benchmark.ipynb # Phase 2: Cornac model training & benchmarking
│   ├── phase_2_mf_overfit.ipynb      # Phase 2: Matrix Factorization overfit analysis
│   └── phase_3_mf_pmf_bpr.ipynb      # Phase 3: MF, PMF, BPR recommendation feedback loop experiments
├── docs/                             # Methodology, research assets & experiment documentation
│   ├── phase1_data_prep.md           # Phase 1 data prep methodology notes
│   ├── phase2_cornac_models.md       # Phase 2 Cornac benchmark findings
│   ├── images/                       # Generated figures and plot exports
│   ├── tex/                          # Paper / manuscript LaTeX source files
│   ├── papers/                       # Reference literature and PDF papers
│   └── ppt/                          # Presentation slides and pitch decks
├── requirements.txt                  # Environment dependencies including Cornac
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
jupyter notebook notebooks/
```

## Documentation & Assets

- **Phase 1 Notebook**: [notebooks/phase_1_data_prep.ipynb](notebooks/phase_1_data_prep.ipynb)
- **Phase 1 Documentation**: [docs/phase1_data_prep.md](docs/phase1_data_prep.md)
- **Phase 2 Benchmarking Notebook**: [notebooks/phase_2_cornac_benchmark.ipynb](notebooks/phase_2_cornac_benchmark.ipynb)
- **Phase 2 Overfit Notebook**: [notebooks/phase_2_mf_overfit.ipynb](notebooks/phase_2_mf_overfit.ipynb)
- **Phase 2 Documentation**: [docs/phase2_cornac_models.md](docs/phase2_cornac_models.md)
- **Phase 3 Notebook**: [notebooks/phase_3_mf_pmf_bpr.ipynb](notebooks/phase_3_mf_pmf_bpr.ipynb)
- **LaTeX Writeup**: [docs/tex/zeroth.tex](docs/tex/zeroth.tex)
- **Presentation Deck**: [docs/ppt/Zeroth.pdf](docs/ppt/Zeroth.pdf)
- **Reference Papers**: [docs/papers/](docs/papers/)


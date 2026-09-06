# Recommender Bias Propagation

This repository supports a research project on how recommender systems may amplify popularity bias over iterative recommendation cycles. The current implementation establishes a reproducible MovieLens 1M preprocessing pipeline and baseline long-tail inequality measurements for later simulation phases.

## Current Scope

Phase 1 prepares the dataset for implicit-feedback recommendation experiments:

- Ingests MovieLens 1M ratings and movie metadata.
- Converts explicit ratings into positive implicit feedback signals using `rating >= 4`.
- Orders each user's interactions chronologically to avoid temporal leakage.
- Maps raw user and item identifiers to dense zero-indexed codes.
- Builds per-user 80/20 chronological train/test sparse matrices.
- Quantifies baseline item-popularity inequality using Gini and top-5% concentration metrics.

Baseline results from the current notebook:

| Metric | Value |
| --- | ---: |
| Users | 6,038 |
| Tracked items | 3,533 |
| Training interactions | 457,818 |
| Test interactions | 117,463 |
| Baseline Gini coefficient | 0.7176 |
| Top 5% item concentration | 37.43% |

## Data

The project uses the public MovieLens 1M dataset from GroupLens:

<https://files.grouplens.org/datasets/movielens/ml-1m.zip>

Dataset artifacts are intentionally excluded from version control:

- `ml-1m.zip`
- `ml-1m/`

Running `phase_1_data_prep.ipynb` from the repository root downloads and extracts the dataset when needed.

Expected extracted files:

- `ml-1m/ratings.dat`
- `ml-1m/movies.dat`
- `ml-1m/users.dat`
- `ml-1m/README`

## Repository Layout

```text
.
|-- phase_1_data_prep.ipynb      # Data ingestion, preprocessing, sparse matrices, baseline metrics
|-- docs/
|   |-- phase1_data_prep.md      # Methodology notes for Phase 1
|   |-- images/                  # Generated figures and plot exports
|   |-- tex/                     # Paper / manuscript LaTeX source files
|   |-- papers/                  # Reference literature and PDFs
|   `-- ppt/                     # Presentation slides and pitch decks
|-- requirements.txt             # Python environment snapshot
`-- README.md
```

## Reproducibility

Create a Python environment, install the notebook dependencies, and run Phase 1 from the repository root:

```bash
python -m venv .venv
pip install -r requirements.txt
pip install scipy matplotlib seaborn notebook
jupyter notebook phase_1_data_prep.ipynb
```

The notebook uses `pandas`, `numpy`, `scipy`, `matplotlib`, and `seaborn`. It assumes the MovieLens text files use the original `::` delimiter and reads movie metadata with `latin-1` encoding.

## Documentation

Detailed notes for the preprocessing decisions, sparse matrix construction, and baseline inequality analysis are in `docs/phase1_data_prep.md`. Research assets, LaTeX paper sources, reference papers, and presentation decks are organized under `docs/tex/`, `docs/papers/`, and `docs/ppt/`.


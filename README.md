# Recommender Bias Propagation & Latent Dynamics Simulator

This repository contains a research framework and closed-loop simulation environment for investigating how collaborative filtering recommendation systems amplify popularity bias (the Matthew Effect) and induce latent representation degradation over iterative recommendation feedback loops.

The project combines empirical data preparation, multi-model benchmarking using **Cornac**, embedding space geometric diagnostics (MF vs. PMF vs. BPR), and a modular closed-loop simulation architecture.

---

## Current Scope & Research Phases

### Phase 1: Data Preparation & Baseline Bias Quantification
- Ingests **MovieLens 1M** ratings and movie metadata.
- Converts explicit ratings into positive implicit feedback signals using `rating >= 4.0`.
- Enforces chronological interaction ordering per user to avoid temporal leakage.
- Maps raw user and item identifiers to immutable, continuous zero-indexed arrays (`0` to `N-1`).
- Constructs per-user 80/20 chronological train/test sparse matrices (CSR format).
- Quantifies baseline catalog popularity inequality using Gini coefficient (`0.7176`) and top-5% item interaction concentration (`37.43%`).

### Phase 2: Recommender Model Benchmarking with Cornac
- Trains and evaluates 7 recommendation algorithms (`MostPop`, `UserKNN`, `ItemKNN`, `MF`, `PMF`, `SVD`, `BPR`) using the Cornac framework.
- Evaluates models across multiple datasets (**MovieLens 1M** and **FilmTrust**).
- Measures ranking accuracy (`NDCG@10`, `Recall@10`, `Precision@10`, `MAP@10`) alongside exposure inequality (`Gini@10`).
- Performs Matrix Factorization overfit analysis and hyperparameter evaluation.

### Phase 3: Latent Space Dynamics & Comparative Analysis (MF vs. PMF vs. BPR)
- Extracts user and item embedding representations in 32-dimensional latent space across head, mid, and long-tail movie item cohorts.
- Identifies critical model-specific algorithmic failure modes:
  - **Pointwise MF:** Suffers from *gradient starvation* on sparse long-tail items, leading to *origin collapse* ($\|v_0\|_2 \to 0$).
  - **Pairwise BPR:** Driven by *negative sampling push*, scattering unobserved long-tail items into high-norm *periphery dispersion*.
  - **PMF:** Uses a zero-mean Gaussian prior to constrain embedding magnitudes, suppressing extreme norm variance.
- Visualizes latent geometric structures using PCA and t-SNE projections.

### Phase 4 / System Specification: Closed-Loop Simulator Architecture
- Establishes a 5-subsystem modular closed-loop simulation architecture (`src/`) specified in [`SPEC.md`](SPEC.md):
  - **Subsystem 1 (`src/data/`):** Invariant State & Data Buffer — Manages immutable entity indexing, dual-matrix interaction accumulation ($\mathcal{D}_t$), and frozen ground-truth preference anchors ($U^*, V^*$).
  - **Subsystem 2 (`src/adapters/`):** Architecture-Agnostic Recommender Interface — Provides facade adapters (`BaseRecommenderAdapter`) for recommendation backbones.
  - **Subsystem 3 (`src/engine/`):** Vectorized User Interaction Engine — Simulates synthetic user browsing and Bernoulli clicks with position decay, relevance matching, and social conformity.
  - **Subsystem 4 (`src/metrics/`):** Multi-Tier Metric & Diagnostic Engine — Audits catalog exposure inequality, user cohort calibration drift ($D_{\text{KL}}$), ranking utility, and latent space geometry (Procrustes alignment, norm shift ratio, effective rank).
  - **Subsystem 5 (`src/orchestrator/`):** Multi-Seed Simulation Orchestrator — Executes generational feedback loops ($t=0 \to T$) with multi-seed statistical aggregation.

---

## Repository Layout

```text
.
├── notebooks/                                # Jupyter notebooks for research phases
│   ├── phase_1_data_prep.ipynb               # Phase 1: Data prep & baseline inequality metrics
│   ├── phase_2_cornac_benchmark.ipynb        # Phase 2: Cornac model training & benchmarking
│   ├── phase_2_mf_overfit.ipynb              # Phase 2: Matrix Factorization overfit analysis
│   └── phase_3_mf_pmf_bpr.ipynb              # Phase 3: Latent space dynamics (MF, PMF, BPR)
├── src/                                      # Closed-loop simulator python architecture
│   ├── adapters/                             # Subsystem 2: Model facade adapters
│   ├── data/                                 # Subsystem 1: Invariant state & data buffers
│   ├── engine/                               # Subsystem 3: User interaction engine
│   ├── metrics/                              # Subsystem 4: Multi-tier diagnostic engine
│   └── orchestrator/                         # Subsystem 5: Multi-seed simulation runner
├── docs/                                     # Documentation, specs, research assets & notes
│   ├── project notes/                        # Phase 1 & 2 methodology and theoretical notes
│   │   ├── README.md                         # Index of project notes
│   │   ├── phase1_data_prep.md               # Phase 1 data prep documentation
│   │   ├── phase2_cornac_models.md           # Phase 2 Cornac benchmarking findings
│   │   ├── recommender_models.md             # Breakdown of algorithm formulations & behavior
│   │   └── bias_propagation_and_embedding_dynamics.md # Theoretical framework & metrics
│   ├── phase3/                               # Phase 3 comparative documentation & guide
│   │   ├── phase3.md                         # Beginner guide & comparative analysis
│   │   └── p3.md                             # Technical summary of Phase 3 results
│   ├── images/                               # Generated figures, plots, and visualizations
│   ├── tex/                                  # LaTeX paper source files (zeroth.tex)
│   ├── ppt/                                  # Presentation slides (Zeroth.pdf)
│   └── papers/                               # Literature references and PDF papers
├── SPEC.md                                   # System engineering specification for simulator
├── SPEC_HELPER.md                            # Theoretical companion & architectural onboarding
├── PLAN.md                                   # System engineering ground-of-truth plan
├── requirements.txt                          # Environment dependencies
└── README.md                                 # Project overview (this file)
```

---

## Reproducibility & Environment Setup

Always use the Python virtual environment (`.venv`):

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Register Jupyter Kernel
python -m ipykernel install --user --name=recommender-bias-venv --display-name "Python (.venv)"

# 4. Launch Jupyter Notebook
jupyter notebook notebooks/
```

---

## Documentation & Project Assets

### System Specifications & Engineering Guides
- **Simulator Specification**: [`SPEC.md`](SPEC.md)
- **Architecture & Onboarding Companion**: [`SPEC_HELPER.md`](SPEC_HELPER.md)
- **System Engineering Plan**: [`PLAN.md`](PLAN.md)

### Research Phase Notebooks
- **Phase 1 Data Prep Notebook**: [`notebooks/phase_1_data_prep.ipynb`](notebooks/phase_1_data_prep.ipynb)
- **Phase 2 Benchmarking Notebook**: [`notebooks/phase_2_cornac_benchmark.ipynb`](notebooks/phase_2_cornac_benchmark.ipynb)
- **Phase 2 Overfit Notebook**: [`notebooks/phase_2_mf_overfit.ipynb`](notebooks/phase_2_mf_overfit.ipynb)
- **Phase 3 Latent Dynamics Notebook**: [`notebooks/phase_3_mf_pmf_bpr.ipynb`](notebooks/phase_3_mf_pmf_bpr.ipynb)

### Technical Documentation & Project Notes
- **Project Notes Index**: [`docs/project notes/README.md`](docs/project%20notes/README.md)
- **Phase 1 Data Preparation**: [`docs/project notes/phase1_data_prep.md`](docs/project%20notes/phase1_data_prep.md)
- **Phase 2 Model Benchmarks**: [`docs/project notes/phase2_cornac_models.md`](docs/project%20notes/phase2_cornac_models.md)
- **Recommendation Models Breakdown**: [`docs/project notes/recommender_models.md`](docs/project%20notes/recommender_models.md)
- **Bias Propagation & Embedding Dynamics**: [`docs/project notes/bias_propagation_and_embedding_dynamics.md`](docs/project%20notes/bias_propagation_and_embedding_dynamics.md)
- **Phase 3 Guide & Analysis**: [`docs/phase3/phase3.md`](docs/phase3/phase3.md)
- **Phase 3 Technical Summary**: [`docs/phase3/p3.md`](docs/phase3/p3.md)

### Publications & Manuscripts
- **LaTeX Manuscript**: [`docs/tex/zeroth.tex`](docs/tex/zeroth.tex)
- **Presentation Deck**: [`docs/ppt/Zeroth.pdf`](docs/ppt/Zeroth.pdf)
- **Reference Literature**: [`docs/papers/`](docs/papers/)

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

### Phase 4 / System Implementation: Closed-Loop Simulator Architecture
- Fully implemented and verified all 5 core modular closed-loop simulation subsystems (`src/`) specified in [`SPEC.md`](SPEC.md) with **23 unit tests** passing OK:
  - **Subsystem 1 (`src/data/`):** Invariant State & Data Buffer — Dataset-agnostic loaders (`load_dataset`, `load_movielens_1m`, `load_movielens_genres`), continuous zero-indexing, dual-matrix interaction accumulation ($\mathcal{D}_t$), and buffer merge/deduplication.
  - **Subsystem 2 (`src/adapters/`):** Architecture-Agnostic Recommender Interface — Abstract facade adapters (`BaseRecommenderAdapter`, `CornacMFAdapter`, `CornacBPRAdapter`, `CornacUserKNNAdapter`, `CornacMostPopAdapter`) with coordinate alignment and deterministic seeding.
  - **Subsystem 3 (`src/engine/`):** Vectorized User Interaction Engine — Vectorized position decay exposure, temperature-scaled intrinsic relevance, sub-linear logarithmic conformity, and Poisson session budget truncation ($\lambda=3.0$).
  - **Subsystem 4 (`src/metrics/`):** Multi-Tier Diagnostic Engine — Audits catalog exposure inequality (Gini, Coverage@K, APLT@K), user cohort taste drift ($D_{\text{KL}}$), ranking utility (Precision@K, Recall@K, NDCG@K, Mann-Whitney AUC), and latent space geometry (Procrustes rigid alignment $\det(R^*)=+1$, relative norm shift $R_{\text{norm}}$, effective rank $S_{\text{eff}}$).
  - **Subsystem 5 (`src/orchestrator/`):** Multi-Seed Simulation Orchestrator — Multi-seed deterministic random state locking, $T$-generation closed-loop simulation execution, memory-safe garbage collection, and aggregated statistical reporting ($\mu, \sigma$).

---

## Repository Layout

```text
.
├── notebooks/                                # Jupyter notebooks for research phases
│   ├── phase_1_data_prep.ipynb               # Phase 1: Data prep & baseline inequality metrics
│   ├── phase_2_cornac_benchmark.ipynb        # Phase 2: Cornac model training & benchmarking
│   ├── phase_2_mf_overfit.ipynb              # Phase 2: Matrix Factorization overfit analysis
│   └── phase_3_mf_pmf_bpr.ipynb              # Phase 3: Latent space dynamics (MF, PMF, BPR)
├── scripts/                                  # Benchmark & simulation execution scripts
│   └── run_full_benchmark.py                 # Full capability multi-model benchmark script
├── src/                                      # Closed-loop simulator python architecture
│   ├── adapters/                             # Subsystem 2: Model facade adapters
│   ├── data/                                 # Subsystem 1: Invariant state & data buffers
│   ├── engine/                               # Subsystem 3: User interaction engine
│   ├── metrics/                              # Subsystem 4: Multi-tier diagnostic engine
│   └── orchestrator/                         # Subsystem 5: Multi-seed simulation runner
├── tests/                                    # Automated unit test suite (23 tests)
│   ├── test_buffer.py                        # Buffer merge & deduplication tests
│   ├── test_dataset.py                       # Dataset loader & zero-indexing tests
│   ├── test_adapters.py                      # Recommender facade & embedding alignment tests
│   ├── test_engine.py                        # User interaction engine & simulator tests
│   ├── test_metrics.py                       # 4-tier diagnostic metrics tests
│   └── test_orchestrator.py                  # Multi-seed simulation pipeline tests
├── docs/                                     # Documentation, specs, research assets & benchmarks
│   ├── benchmarks/                           # Benchmark outputs (.gitkeep tracked; reports ignored)
│   ├── notes/                                # Research notes across Phases 1, 2, 3, and 4
│   ├── images/                               # Structured visual catalog mapped to notebooks
│   ├── tex/                                  # LaTeX paper source files (zeroth.tex)
│   ├── ppt/                                  # Presentation slides (Zeroth.pdf)
│   └── papers/                               # Literature references and PDF papers
├── SPEC.md                                   # System engineering specification for simulator
├── SPEC_HELPER.md                            # Theoretical companion & architectural onboarding
├── PLAN.md                                   # System engineering ground-of-truth plan
├── AGENT.md                                  # Quality control & engineering standards
├── requirements.txt                          # Environment dependencies
└── README.md                                 # Project overview (this file)
```

---

## Reproducibility & Testing

Always use the Python virtual environment (`venv`):

```bash
# 1. Automated Setup (creates venv, installs requirements.txt, and activates)
source scripts/setup_venv.sh

# 2. Run Automated Unit Test Suite (23 tests across 5 subsystems)
python -m unittest discover tests

# 3. Execute Full Closed-Loop Capability Benchmark
python scripts/run_full_benchmark.py
```

---

## Documentation & Project Assets

### System Specifications & Engineering Guides
- **Simulator Specification**: [`SPEC.md`](SPEC.md)
- **Architecture & Onboarding Companion**: [`SPEC_HELPER.md`](SPEC_HELPER.md)
- **System Engineering Plan**: [`PLAN.md`](PLAN.md)
- **Engineering Standards**: [`AGENT.md`](AGENT.md)
- **Latest Benchmark Results**: [`docs/benchmarks/benchmark_results.md`](docs/benchmarks/benchmark_results.md) (Generated locally by `scripts/run_full_benchmark.py`)
- **Benchmark Run Directory**: [`docs/benchmarks/`](docs/benchmarks/)

### Research Phase Notebooks
- **Phase 1 Data Prep Notebook**: [`notebooks/phase_1_data_prep.ipynb`](notebooks/phase_1_data_prep.ipynb)
- **Phase 2 Benchmarking Notebook**: [`notebooks/phase_2_cornac_benchmark.ipynb`](notebooks/phase_2_cornac_benchmark.ipynb)
- **Phase 2 Overfit Notebook**: [`notebooks/phase_2_mf_overfit.ipynb`](notebooks/phase_2_mf_overfit.ipynb)
- **Phase 3 Latent Dynamics Notebook**: [`notebooks/phase_3_mf_pmf_bpr.ipynb`](notebooks/phase_3_mf_pmf_bpr.ipynb)
- **Phase 4 t-SNE Manifold Visualization**: [`notebooks/phase_4_tsne_visualization.ipynb`](notebooks/phase_4_tsne_visualization.ipynb)

### Technical Documentation & Research Notes
- **Documentation Hub**: [`docs/README.md`](docs/README.md)
- **Research Notes Index**: [`docs/notes/README.md`](docs/notes/README.md)
- **Phase 1 Data Preparation**: [`docs/notes/phase1_data_prep.md`](docs/notes/phase1_data_prep.md)
- **Phase 2 Model Benchmarks**: [`docs/notes/phase2_cornac_models.md`](docs/notes/phase2_cornac_models.md)
- **Phase 3 Representation Geometry**: [`docs/notes/phase3_representation_geometry.md`](docs/notes/phase3_representation_geometry.md)
- **Phase 3 Popularity Trap Guide**: [`docs/notes/phase3_beginners_guide.md`](docs/notes/phase3_beginners_guide.md)
- **Phase 4 Latent Manifold Visualization**: [`docs/notes/phase4_tsne_visualization.md`](docs/notes/phase4_tsne_visualization.md)
- **Recommendation Models Breakdown**: [`docs/notes/recommender_models.md`](docs/notes/recommender_models.md)
- **Bias Propagation & Embedding Dynamics**: [`docs/notes/bias_propagation_and_embedding_dynamics.md`](docs/notes/bias_propagation_and_embedding_dynamics.md)
- **Visual Assets & Trajectory Figures**: [`docs/images/`](docs/images/)

### Publications & Manuscripts
- **LaTeX Manuscript**: [`docs/tex/zeroth.tex`](docs/tex/zeroth.tex)
- **Presentation Deck**: [`docs/ppt/Zeroth.pdf`](docs/ppt/Zeroth.pdf)
- **Reference Literature**: [`docs/papers/`](docs/papers/)


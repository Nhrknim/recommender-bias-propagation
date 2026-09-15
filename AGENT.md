# Agent Operating Standards (`AGENT.md`)

### 1. Python Function Quality Control (QC) Standards

**Single Responsibility & Separation of Concerns**

* **Pure Mathematical Routines:** Functions calculating metrics, probabilities, or geometric transformations must be side-effect-free pure functions. They take arrays, return arrays or scalars, and perform no I/O, logging, or state mutations.
* **State & Buffer Management:** Confined strictly to `src/data/buffer.py`. Metric computation functions must never mutate the interaction buffer.
* **Model Wrappers:** Recommender adapters strictly bridge data conversion and model execution. They must not contain metric computation or synthetic click logic.

**Explicit Shape Annotations & Strict Type Hints**

* Every function must use Python standard type hints (`typing` / built-in types).
* Every `np.ndarray` and `scipy.sparse` argument or return type must include explicit dimension commentary in the signature or docstring:

```python
def compute_vector_drift(
    v_baseline: np.ndarray,      # Shape: (|I|, d)
    v_current: np.ndarray,       # Shape: (|I|, d)
    rotation_matrix: np.ndarray  # Shape: (d, d)
) -> np.ndarray:                 # Shape: (|I|,)
    """Calculates per-item angular drift after rigid coordinate rotation.

    Args:
        v_baseline: Uncorrupted item embeddings at iteration 0.
        v_current: Item embeddings at iteration t.
        rotation_matrix: Reflection-corrected orthogonal alignment matrix.

    Returns:
        1D array containing cosine drift (1 - cos(theta)) per item.
    """

```

**Vectorization Invariant (No Iterative Python Loops)**

* Explicit Python loops (`for u in users:`, `for i in items:`) over catalog entities during inference, click simulation, or metric calculation are strictly prohibited.
* All mathematical operations must use vectorized NumPy/SciPy broadcasting, matrix multiplication (`@`), or sparse algebra.

**Numerical Stability & Boundary Guards**

* **No Bare Divisions:** Every division operation must contain an explicit epsilon cushion ($\epsilon = 10^{-9}$):
`denom = np.linalg.norm(v, axis=1) + 1e-9`
* **Logit Range Bounding:** Any logit passed to an exponential or sigmoid must be clipped to prevent overflow/underflow:
`scaled_logits = np.clip(logits / (tau * np.sqrt(d)), -15.0, 15.0)`
* **Sanity Assertions:** Mathematical functions must assert invariants in debug mode:
`assert not np.isnan(result).any(), "NaN detected in metric computation"`

---

### 2. Architectural & Repository Invariants

**Immutable Catalog Coordinates**

* Entity IDs must never be reassigned. The universe sizes $\vert{}\mathcal{U}\vert{}$ and $\vert{}\mathcal{I}\vert{}$ and the contiguous zero-index mapping ($u \in [0, \vert{}\mathcal{U}\vert{}-1]$, $i \in [0, \vert{}\mathcal{I}\vert{}-1]$) established at initialization ($t=0$) are immutable.
* If an item receives zero interactions, it remains in the matrix as an all-zero column; under no circumstances should an agent write filtering code that drops inactive items or shrinks matrix dimensions.

**Binary Interaction Integrity**

* The historical interaction buffer $\mathcal{D}_t$ is an implicit binary dataset: $\mathcal{D}_t \in \{0, 1\}^{\vert{}\mathcal{U}\vert{} \times \vert{}\mathcal{I}\vert{}}$.
* When ingesting newly simulated interactions, always enforce deduplication and binarization (`csr.data[:] = 1.0`). Interaction frequencies must never accumulate into integer counts within the training matrix.

**Model Facade Isolation**

* Any newly added recommendation model must inherit from `BaseRecommenderAdapter`.
* All library-specific transformations (e.g., converting a CSR matrix into a Cornac `Dataset` or PyTorch `EdgeIndex`) must be executed internally within that adapter. The orchestrator must only interact with the standard methods: `fit()`, `recommend()`, and `get_embeddings()`.

**Deterministic Seeding Protocol**

* Any component generating pseudo-random outputs (model initialization, negative sampling, Bernoulli click trials, Poisson budget sampling) must accept an explicit `seed: int` parameter.
* The orchestrator must synchronize seeds across all underlying frameworks at the start of each run:

```python
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

```

**Memory Hygiene & State Persistence**

* Closed-loop runs across multiple seeds will saturate RAM if full model instances and factor matrices are preserved in memory.
* Generational metrics must be appended to a flat CSV immediately. Factor arrays must be serialized to `.npz` on disk at designated checkpoints, followed by an explicit `gc.collect()`. Never store full history matrices across generations in an in-memory list.

---

### 3. Agent Operational Boundaries

* **Do Not Redefine Formulas:** Never "simplify" the mathematical formulations established in `PLAN.md` (e.g., changing sub-linear conformity back to linear ratios, or omitting $\det(R^*)$ checking in Procrustes).
* **Test-Driven Delivery:** Every new subsystem file must be accompanied by an isolated test in `tests/` verifying shapes, deterministic outputs, and numerical stability edge-cases (e.g., zero vectors, empty clicks).
* **Fail Fast Over Silent Recovery:** If dimension mismatches, index gaps, or non-invertible matrices occur, raise explicit exceptions (`ValueError`, `FloatingPointError`) rather than falling back to imputed default values that could silently corrupt longitudinal runs.

---

### 4. Notebook Visual Assets & Research Documentation Invariants

**Structured Image Outputs (`docs/images/<notebook_name>/`)**

* **Strict Notebook-to-Folder Mapping:** All visual outputs, diagnostic plots, or publication figures generated by a notebook (`notebooks/<notebook_name>.ipynb`) MUST be saved in `docs/images/<notebook_name>/`, where the subfolder strictly matches the notebook filename (omitting the `.ipynb` extension).
* **No Loose or Ad-Hoc Directories:** Storing figures in fragmented locations (such as `docs/figures/`, root `images/`, or loose files directly under `docs/images/`) is strictly prohibited.
* **Catalog Synchronization:** Any new or modified visual assets must be registered in [`docs/images/README.md`](docs/images/README.md) to maintain the notebook-to-asset mapping table and describe the theoretical phenomenon illustrated.

**Companion Research Notes (`docs/notes/`)**

* **Mandatory Research Notes:** For every exploratory or analytical notebook in `notebooks/`, a corresponding research note must be generated or updated in [`docs/notes/`](docs/notes/) (e.g., `docs/notes/<phase_topic>.md`).
* **Content Requirements:** Companion notes must articulate:
  1. The theoretical motivation, mathematical equations, and experimental objectives.
  2. Concrete parameter configurations, sampling budgets, and model hyperparameters.
  3. Key empirical takeaways and interpretation of observed dynamics.
  4. Visual proof embedding the generated figures via relative links (`../images/<notebook_name>/<figure_name>.png`).
* **Master Documentation Indexing:** Every research note must be indexed and summarized in [`docs/notes/README.md`](docs/notes/README.md) and cross-referenced in [`docs/README.md`](docs/README.md).
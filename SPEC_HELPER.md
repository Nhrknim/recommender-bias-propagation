# Closed-Loop Recommendation Simulator: Onboarding & Architecture Companion

**Document:** `SPEC_HELPER.md`  
**Purpose:** Conceptual companion to `SPEC.md` for onboarding researchers and engineers. Explains the theoretical rationale, meta-commentary, and design trade-offs behind the engineering ground of truth.  
**Prerequisites:** Basic familiarity with collaborative filtering, linear algebra, and NumPy.  

---

## 1. The Big Picture: What Are We Simulating and Why?

Traditional recommender system evaluations are static: you take a historical dataset, perform an 80/20 split, train a model once, and report accuracy metrics like NDCG or Recall.

In production platforms, recommender systems operate in an **iterative feedback loop**:

1. The model generates top-$K$ recommendations for users.
2. Users browse, select, and interact only with items exposed to them.
3. The platform logs these interactions and retrains the model on its own past outputs.

```
                     ┌────────────────────────────────┐
                     │   Model Recommends Top-K       │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │   User Browses & Selects       │
                     │   (Subject to Exposure Bias)   │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │   Interaction Logs Grow Skewed │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │   Model Retrains on Own Echo   │
                     └────────────────────────────────┘
```

When popular items receive disproportionate exposure, models quickly learn that recommending them is the lowest-risk path to minimize training loss. Over multiple retraining generations, this creates the **Matthew Effect** ("the rich get richer, the poor starve").

**Our Research Objective:** We are not merely checking *if* popularity bias occurs; we are building an automated laboratory to quantify the **velocity of internal representation degradation**. We track how the geometric latent coordinate space (the embedding vectors $U_t, V_t$) contracts, rotates, or collapses as the feedback loop compounds over time.

---

## 2. Subsystem Walkthrough: Architecture & Operational Rationale

`SPEC.md` establishes five modular subsystems. Here is the operational rationale behind each component:

### Subsystem 1: Invariant State & Data Buffer
* **What it is:** The memory manager holding interaction matrices and entity ID registers.
* **Operational Rationale:**
  * *Immutable Zero-Indexing:* If an unpopular indie movie receives zero clicks during generations 2 through 5, standard sparse operations might attempt to drop that movie to compress the matrix. If that happens, row 500 becomes row 480, corrupting all future matrix comparisons. We lock every entity to a permanent coordinate index $i \in [0, |\mathcal{I}|-1]$ that never changes.
  * *Frozen Ground-Truth Anchor ($U^*, V^*$):* To evaluate whether a user genuinely enjoys a recommended movie, we cannot use the *current* model (which is already biased). Instead, we learn baseline preference vectors at generation 0 ($t=0$) from clean historical data and freeze them as our immutable benchmark of "true human taste".

### Subsystem 2: Architecture-Agnostic Recommender Interface
* **What it is:** A facade (Adapter Pattern) wrapping recommendation backbones.
* **Operational Rationale:** The simulation orchestrator must remain agnostic to the underlying algorithm (Matrix Factorization, BPR, or Graph Neural Networks like LightGCN). The interface enforces uniform methods: `.fit()`, `.recommend()`, and `.get_embeddings()`.

### Subsystem 3: Vectorized User Interaction Engine
* **What it is:** A mathematical simulator modeling human decision-making over exposed recommendation lists.
* **Operational Rationale:** We use vectorized mathematical models rather than LLM agents for high computational speed and reproducibility. Human browsing is modeled via three verifiable behaviors:
  1. *Position Decay:* Probability of scanning items decreases logarithmically with list rank.
  2. *Relevance Matching:* Preference alignment between user and item baseline vectors.
  3. *Social Conformity:* Influence of aggregate platform popularity and herd behavior.

### Subsystem 4: Multi-Tier Metric & Diagnostic Engine
* **What it is:** The auditing harness evaluating system health at the end of each generation $t$.
* **Operational Rationale:** Surface-level accuracy metrics (NDCG) fail to reveal representation degradation. This engine audits the system across four tiers: catalog exposure inequality, demographic user cohort fairness, ranking utility, and latent space manifold geometry.

### Subsystem 5: Multi-Seed Simulation Orchestrator
* **What it is:** The lifecycle manager controlling sequential execution ($t = 0 \to T$), seed sweeps, garbage collection, and disk serialization.
* **Operational Rationale:** Closed feedback loops are non-linear dynamical systems. A single run could be an outlier caused by early stochastic clicks. Sweeping across $N \ge 5$ random seeds yields mean curves with standard deviation bounds ($\mu \pm \sigma$).

---

## 3. Meta-Commentary & Architectural Gotchas: Why the Spec Rules Exist

The table below documents the meta-commentary, mathematical pitfalls, and design trade-offs that dictated the invariants in `SPEC.md`:

| Subsystem / Metric | The Naive Implementation | The Hidden Mathematical / Computational Flaw | Architectural Solution in `SPEC.md` |
| --- | --- | --- | --- |
| **1. Relevance Logit Scaling** | $\sigma(\mathbf{u}^T \mathbf{v})$ | Dot products of 32-dim vectors yield raw numbers like $+18$ or $-15$. Passing these into standard sigmoid produces $1.0000$ or $0.0000$, turning relevance into a binary switch and disabling $\alpha$ tuning. | **Temperature Scaling & Clipping:**<br>$\sigma\left( \text{clip}\left( \frac{\mathbf{u}^{*T} \mathbf{v}^*}{\tau \sqrt{d}}, -15, 15 \right) \right)$<br>Keeps logits in a smooth range ($[-2, 2]$). |
| **2. Sub-Linear Conformity** | Linear ratio $\frac{f_i}{\max f}$ | Top blockbusters have thousands of clicks while mid-tail movies have 50–100. A linear ratio gives mid-tail items an effective score of $0.02$, artificially flattening everything except head items. | **Logarithmic Power-Law Scaling:**<br>$\frac{\log(1 + f_i)}{\log(1 + \max f)}$<br>Preserves mid-tail competitiveness while recognizing head item dominance. |
| **3. Smooth Vector Norm Shift** | $R_{\text{norm}} = \frac{\|v_t\|_2}{\|v_0\|_2}$ | In Pointwise MF, niche items suffer gradient starvation; their embeddings collapse toward zero ($\|v_0\|_2 \to 0$), causing division-by-zero crashes or infinite spikes. | **Epsilon-Cushioned Ratio:**<br>$R_{\text{norm}} = \frac{\|v_t\|_2 + \epsilon}{\|v_0\|_2 + \epsilon}, \ \epsilon = 10^{-9}$<br>Stabilizes ratio across extreme values without distorting trends. |
| **4. Procrustes Alignment** | Standard SVD: $R^* = U V^T$ | SVD computes orthogonal transformations, but half of random orthogonal matrices have determinant $-1$ (**reflection/mirroring**), distorting spatial drift metrics. | **Kabsch Sign-Flip Check:**<br>If $\det(R^*) < 0$, flip last column of $V_{\text{svd}}$ to strictly enforce a proper rigid rotation ($\det = +1$). |
| **5. Global Pairwise AUC** | Nested pairwise loop over $|\mathcal{U}| \times |\mathcal{I}|$ | Evaluating all positive vs unobserved pairs per user requires $\sim 22.3\text{M}$ ops per generation, creating a multi-billion operation bottleneck across seeds. | **Vectorized Mann-Whitney AUC:**<br>Computes exact AUC in $\mathcal{O}(|\mathcal{I}| \log |\mathcal{I}|)$ sorting time via `compute_vectorized_auc()`. |
| **6. Data Buffer Updates** | Naive CSR addition `D_t + ΔD_t` | Repeated addition of simulated clicks leaves non-binary matrix entries ($r_{ui} > 1$), violating the binary implicit interaction assumption ($r_{ui} \in \{0,1\}$). | **Flat COO Merge & Deduplication:**<br>Merges flat COO arrays and forces non-zero data values to $1.0$ (`csr.data = 1.0`). |
| **7. Coordinate Indexing** | Framework internal ID dicts | Cornac/PyTorch re-indexes sparse arrays dynamically on new data slices, causing factor matrix rows ($V_0$ vs $V_t$) to get misaligned. | **Bidirectional Index Mapping:**<br>`extract_aligned_embeddings()` maps framework vectors back to global array coordinates $i \in [0, |\mathcal{I}|-1]$. |

---

## 4. Interpreting Diagnostic Vital Signs (The 4 Tiers)

Use this guide to diagnose system behavior when analyzing output metrics:

```
Tier 1: Macro-Catalog      ──►  Is the catalog as a whole becoming an echo chamber?
Tier 2: User Subgroups     ──►  Are niche-interest users being marginalized faster than mainstream users?
Tier 3: Retrieval Utility  ──►  Is the top-K list bad, or has the model forgotten all relative rankings?
Tier 4: Latent Geometry    ──►  What is physically happening to the internal coordinate space?
```

### Tier 1: Macro Catalog Distribution
* **Exposure Gini Index ($G^{(t)}$):** $0.0$ indicates equal recommendation exposure; $1.0$ indicates a single item monopolizes all slots. Unmanaged feedback loops cause Gini to climb steadily toward $1.0$.
* **Catalog Coverage@K:** Percentage of unique catalog items recommended across all users. A drop from $65\%$ to $12\%$ indicates severe catalog collapse.

### Tier 2: User Subgroups (Mesoscale Dynamics)
* **Category Calibration Drift ($D_{\text{KL}}$):** Compares historical user genre distribution against current recommendations. A steep drift for the Niche cohort proves the algorithm is forcing mainstream content onto niche-interest users.
* **Subgroup Utility Gap:** $\text{NDCG}(\mathcal{U}_{\text{Pop}}) - \text{NDCG}(\mathcal{U}_{\text{Niche}})$. A widening gap demonstrates algorithmic marginalization of niche users.

### Tier 3: Ranking & Global Catalog Utility
* **NDCG@10 vs. Global Pairwise AUC:**
  * *If NDCG@10 drops while AUC remains steady:* The model retains global preference understanding, but head items are artificially displacing tail items in top-$K$ slots.
  * *If both NDCG@10 and AUC collapse:* The model has suffered catastrophic representational forgetting.

### Tier 4: Latent Space Geometry
* **Relative Vector Norm ($R_{\text{norm}}$):**
  * $R_{\text{norm}} < 1.0$: Vector shrinkage. Niche item embeddings collapse into the origin $(0,0,\dots,0)$ due to gradient starvation in Pointwise models (MSE loss).
  * $R_{\text{norm}} > 1.0$: Vector expansion. Negative sampling in Pairwise models (BPR) pushes unwatched niche items outward toward the perimeter.
* **Effective Rank ($S_{\text{eff}}$):** Continuous dimensionality of the latent space. If a 32-dimensional embedding's effective rank drops to $4.2$, the latent space has flattened into a low-dimensional manifold.

---

## 5. Technical Terminology Glossary

* **Cold Retrain:** Initializing model weights from scratch with a new random seed at each generation $t$, training on cumulative data $\mathcal{D}_t$. Isolates structural data drift from weight inertia.
* **Warm Fine-Tuning:** Initializing generation $t+1$ with converged weights from generation $t$.
* **Gradient Starvation:** In Pointwise Matrix Factorization, rare items receive few rating updates. The continuous $L_2$ penalty pulls their embedding weights toward zero.
* **Negative Repulsion:** In Pairwise Ranking (BPR), unobserved items are sampled uniformly as negative examples. Long-tail items are frequently sampled as negatives, pushing their vectors away from user coordinates.
* **Orthogonal Procrustes:** Matrix alignment using SVD ($M = V_0^T V_t = U \Sigma V_{\text{svd}}^T$) to find the optimal rigid rotation matrix $R^* = U V_{\text{svd}}^T$ aligning $V_t$ to baseline $V_0$.
* **Conformity Weight ($\alpha$):** Simulation parameter blending intrinsic relevance ($(1-\alpha)$) and platform popularity ($\alpha$).
# Recommender Systems: Bias Propagation & Embedding Dynamics

This document details the theoretical foundations, geometric degradation dynamics, closed-loop simulation architecture, and advanced causal reference frameworks for recommender system bias propagation.

---

## 1. The Core Phenomenon: Feedback Loops & Dynamic Bias

Recommender systems operating in real-world deployments continuously interact with human users, creating an iterative feedback loop where recommendations influence future user behavior, which in turn becomes future training data.

### Power-Law Distribution (Long-Tail)
Real-world interaction datasets naturally exhibit extreme skewness. A tiny fraction of "head" items collect the vast majority of user interactions, while a broad "long tail" of niche items receives sparse engagement.

### Static Bias vs. Dynamic Bias

| Bias Category | Definition | Operational Impact |
|---|---|---|
| **Static Bias** | The frozen, historical imbalance present in offline logs prior to algorithmic intervention. | e.g., Top 5% of catalog capturing ~37% of baseline interactions. |
| **Dynamic Bias (Feedback Loop)** | Compounded imbalance emerging when a model trains on user choices driven by its prior recommendations. | Algorithmic exposure becomes future ground-truth data, amplifying initial skewness over time. |

### The Matthew Effect ("Rich Get Richer")
Recommenders minimize empirical risk by over-recommending popular head items. As users click what they are exposed to, these positive signals are fed back into subsequent training cycles, further inflating head item scores while starving tail items of exposure.

---

## 2. Algorithmic Paradigms: How Models Learn

Recommender algorithms process data through distinct parametric and non-parametric mechanisms, leading to unique failure modes under biased interaction distributions.

```text
Pointwise: (u, i)            ──► Predict absolute scalar score:  r̂_ui ≈ r_ui
Pairwise:  (u, i_pos, j_neg) ──► Maximize ranking margin:        r̂_ui > r̂_uj
```

### Memory-Based Collaborative Filtering (Non-Parametric)

- **`UserKNN`**: Aggregates interaction histories of the $k$ most similar users.
  - **Complexity:** Naively $\mathcal{O}(|\mathcal{U}|^2 \cdot |\mathcal{I}|)$, requiring inverted indices (item $\to$ user list) to prune zero-overlap user pairs.
  - **Bias Behavior:** Strongly favors head items because popular items act as primary overlap bridges across arbitrary user pairs.
- **`ItemKNN`**: Recommends items similar to a user's past choices based on item co-occurrence vectors.
  - **Complexity:** $\mathcal{O}(|\mathcal{I}|^2)$. Because item catalogs update less frequently and $|\mathcal{I}| \ll |\mathcal{U}|$, item similarities are precomputed offline.
  - **Bias Behavior:** Preserves localized catalog diversity better than `UserKNN` by strictly anchoring recommendations to items consumed by the specific user.

---

### Model-Based Latent Factor Models (Parametric)

#### Pointwise Models (`MF`, `PMF`, `SVD`)

- **Objective:** Evaluates each $(u, i)$ pair in isolation as a regression (MSE) or binary classification task:
  $$\mathcal{L}_{\text{pointwise}} = \sum_{(u, i) \in \mathcal{D}} \left(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i\right)^2 + \lambda \left(\|\mathbf{u}_u\|_2^2 + \|\mathbf{v}_i\|_2^2\right)$$

- **Failure Mode: Gradient Starvation & Origin Collapse**
  Evaluating the gradient with respect to item vector $\mathbf{v}_i$:
  $$\nabla_{\mathbf{v}_i} \mathcal{L} = -2(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)\mathbf{u}_u + 2\lambda \mathbf{v}_i$$
  For tail items, positive interactions $|\{u : (u, i) \in \mathcal{D}\}| \approx 0$. The data-driven gradient term vanishes while the $L_2$ regularization penalty ($2\lambda \mathbf{v}_i$) applies continuously during training. As a result, tail item embeddings collapse into a dense sphere near the origin $\mathbf{0} \in \mathbb{R}^d$.

- **FunkSVD (Explicit Bias Decomposition):**
  Decomposes recommendations into explicit baseline scalars and interaction terms:
  $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$
  Isolates global mean $\mu$, user generosity $b_u$, and item popularity $b_i$ into explicit scalar parameters to prevent latent vectors ($\mathbf{v}_i$) from absorbing popularity distortions.

#### Pairwise Models (`BPR`)

- **Objective:** Treats recommendation as ordinal ranking, optimizing relative score margins between observed positive items $i$ and randomly sampled negative items $j$:
  $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}} \ln \sigma\left(\mathbf{u}_u^T \mathbf{v}_i - \mathbf{u}_u^T \mathbf{v}_j\right) + \lambda_\Theta \|\Theta\|_2^2$$

- **Intuition (The Tournament Model):** BPR does not evaluate all items simultaneously. Because dot products satisfy total ordering ($a > b \land b > c \implies a > c$), pairwise updates position items along a continuous latent continuum.

- **Failure Mode: Negative Sampling Push & Periphery Dispersion**
  Evaluating the gradient with respect to negative item vector $\mathbf{v}_j$:
  $$\nabla_{\mathbf{v}_j} \mathcal{L}_{\text{BPR}} \propto -(1 - \sigma(\hat{x}_{uij}))\mathbf{u}_u + 2\lambda \mathbf{v}_j$$
  Unpopular tail items are repeatedly sampled as unobserved negative items ($j$) across thousands of users. These cumulative negative push gradients drive tail item embeddings out toward the spatial perimeter of the embedding space. While coordinate spread is preserved (preventing origin collapse), tail items become isolated from user vector clusters and are never retrieved in top-$K$ scoring.

---

## 3. Geometric Dynamics & Tracking Metrics

### The Rotational Invariance Problem
Inner products are invariant under orthogonal transformations:
$$(\mathbf{u} R)(\mathbf{v} R)^T = \mathbf{u} R R^T \mathbf{v}^T = \mathbf{u} \mathbf{v}^T \quad \text{for } R^T R = \mathbf{I}$$
Retraining a model over iterative generations introduces arbitrary coordinate rotations. Raw Euclidean distances between generation $0$ and generation $t$ measure orthogonal rotation rather than true semantic drift.

### Orthogonal Procrustes Alignment
Aligns generation $t$'s item embedding matrix $B$ back to baseline coordinate axes $A$:
$$\min_R \|B R - A\|_F^2 \quad \text{subject to } R^T R = \mathbf{I}$$

#### Closed-Form Solution via SVD:
1. Compute cross-covariance matrix: $M = A^T B$.
2. Compute Singular Value Decomposition: $M = U \Sigma V^T$.
3. Obtain optimal rotation matrix: $R^* = V U^T$.
4. Compute aligned embeddings: $B_{\text{aligned}} = B R^*$.

### Core Geometric Degradation Metrics

- **Embedding Drift Velocity:** Average cosine distance of items from baseline positions after Procrustes alignment:
  $$\mathcal{V}_{\text{drift}}(t) = \frac{1}{|\mathcal{I}|} \sum_{i=1}^{|\mathcal{I}|} \left(1 - \frac{(\mathbf{v}_i^{(t)} R^*)^T \mathbf{v}_i^{(0)}}{\|\mathbf{v}_i^{(t)} R^*\|_2 \|\mathbf{v}_i^{(0)}\|_2}\right)$$

- **Latent Coordinate Variance (Space Collapse):** Trace of the embedding covariance matrix across $d$ dimensions:
  $$\text{Var}(V^{(t)}) = \frac{1}{d} \sum_{k=1}^d \text{Var}(V_{*, k}^{(t)})$$
  A monotonic drop in variance indicates loss of vector space expressive capacity.

- **Gini Coefficient:** Exposure concentration across catalog items:
  $$\text{Gini} = \frac{\sum_{i=1}^{|\mathcal{I}|} (2i - |\mathcal{I}| - 1) x_{(i)}}{|\mathcal{I}| \sum_{i=1}^{|\mathcal{I}|} x_{(i)}}$$
  where $x_{(i)}$ is recommendation frequency sorted in ascending order.

- **Catalog Coverage@K:** Fraction of catalog items appearing at least once in top-$K$ recommendation lists across all users:
  $$\text{Coverage}@K = \frac{\left|\bigcup_{u \in \mathcal{U}} \text{TopK}(u)\right|}{|\mathcal{I}|}$$

---

## 4. Closed-Loop Simulation Architecture

The iterative feedback loop simulation executes in four sequential stages:

```text
┌────────────────────────────────────────────────────────┐
│ 1. Serve: Generate Top-K items using current model     │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Simulate: Probabilistic User Click Engine           │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Log: Append simulated interactions to train matrix  │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Retrain: Fit model on updated data & log metrics    │
└──────────────────────────┬─────────────────────────────┘
```

### Synthetic User Click Engine Formulation

Interactions are modeled via three probabilistic components:

$$P(\text{Click} \mid u, i, \text{rank}) = P(\text{Exposed} \mid \text{rank}) \cdot \left[(1 - \alpha) P(\text{Relevance} \mid u, i) + \alpha P(\text{Conformity} \mid i)\right]$$

1. **Position Bias (Exposure):** Logarithmic rank discount modeling user attention:
   $$P(\text{Exposed} \mid \text{rank}) = \frac{1}{\log_2(\text{rank} + 2)}$$

2. **True Relevance:** Sigmoid mapping of ground-truth preference dot products:
   $$P(\text{Relevance} \mid u, i) = \sigma(\mathbf{u}_u^{*T} \mathbf{v}_i^*) = \frac{1}{1 + \exp(-\mathbf{u}_u^{*T} \mathbf{v}_i^*)}$$

3. **Popularity Conformity ($\alpha$):** Social proof probability driven by global item exposure:
   $$P(\text{Conformity} \mid i) = \frac{\text{Interactions}_t(i)}{\max_{j} \text{Interactions}_t(j)}$$

---

## 5. Advanced Literature Reference Frameworks

### CDSRec: Causal Disentanglement in Sequential Recommendation (Liu et al., 2026)
- **Core Insight:** Item popularity consists of both benign (high inherent quality) and adverse (conformity / algorithmic exposure) components.
- **Disentangled Latent Factors:**
  - $x_t$: Intrinsic personalized demand.
  - $y_t$: Inherent quality (long-term popularity correlated with user satisfaction).
  - $z_t$: External interference (short-term popularity spikes driven by trends and position bias).
- **Mechanism:** Implements a Structural Causal Model (SCM) and Sequential VAE with GRUs to isolate $x_t$, $y_t$, and $z_t$. During inference, intervention eliminates $z_t$, computing recommendations purely on $x_t$ and $y_t$.

### GSDA: Graph-Structured Dual Adaptation (Cai et al., 2026)
- **Core Insight:** Graph Neural Network recommenders (e.g., LightGCN) suffer from over-smoothing across message-passing layers, triggering severe representation collapse for tail items.
- **Information-Theoretic Proof:** Proves that layer-wise conditional entropy $H(X_{\text{user}} \mid X_{\text{item}})$ between popular and tail items decays monotonically across deep graph layers.
- **Mechanisms:**
  - **Hierarchical Alignment:** Dynamically scales layer representations using matrix norms of adjacency powers $\|\hat{A}^l\|_F$.
  - **Distribution-Aware Contrastive Weighting:** Uses batch-level Gini coefficients to automatically balance contrastive loss weights between head and tail nodes.

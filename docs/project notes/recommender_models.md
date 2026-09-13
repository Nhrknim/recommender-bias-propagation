# The 7 Recommender Models Evaluated in Phase 2

This document details the 7 recommendation algorithms evaluated in **Phase 2**, categorizing their theoretical paradigms, scoring formulations, and specific roles in studying **popularity bias** and **long-tail inequality propagation**.

> [!NOTE]
> For in-depth theoretical analysis of algorithmic failure modes (Gradient Starvation, Origin Collapse, Periphery Dispersion), geometric tracking metrics (Procrustes Alignment, Drift Velocity), and closed-loop simulation architecture, refer to [**bias_propagation_and_embedding_dynamics.md**](file:///x:/Academics/Main%20Project/recommender-bias-propagation/docs/project%20notes/bias_propagation_and_embedding_dynamics.md).

---


## Model Summary Matrix

| # | Model | Family / Paradigm | Cornac Class | Key Objective / Formulation | Popularity Bias Tendency |
|---|---|---|---|---|---|
| **1** | **MostPop** | Non-Personalized Baseline | `cornac.models.MostPop` | Recommends top-$N$ globally frequent items | Extreme bias ($Gini \approx 1.0$) |
| **2** | **UserKNN** | Memory-Based Neighborhood | `cornac.models.UserKNN` | User similarity cosine/Pearson aggregation | High bias (head item overlap) |
| **3** | **ItemKNN** | Memory-Based Neighborhood | `cornac.models.ItemKNN` | Item co-occurrence similarity aggregation | Moderate/Low bias (local catalog surfacing) |
| **4** | **MF** | Pointwise Matrix Factorization | `cornac.models.MF` | Low-rank inner product with MSE loss | Moderate bias (best ML-1M Gini) |
| **5** | **PMF** | Probabilistic Matrix Factorization | `cornac.models.PMF` | Gaussian priors over latent factors | High bias under sparsity |
| **6** | **SVD** | Biased Latent Factor | `cornac.models.SVD` | Explicit global, user, and item bias offsets | Moderate bias (scalar item bias absorption) |
| **7** | **BPR** | Pairwise Ranking MF | `cornac.models.BPR` | Pairwise preference optimization ($\text{BPR-OPT}$) | High bias (highest accuracy, concentrates on head) |

---

## Detailed Model Breakdown

### 1. MostPop (Non-Personalized Popularity Baseline)

- **Algorithm Family:** Non-Personalized Baseline
- **Cornac Class:** `cornac.models.MostPop`
- **How It Works:** Calculates the total interaction frequency for every item across the training set ($f_i = \sum_{u \in \mathcal{U}} r_{ui}$) and recommends the exact same top-$N$ most frequent items to every user.
- **Role in Research:** Acts as the theoretical upper bound for popularity bias. Any personalized model should ideally demonstrate lower concentration ($Gini@10$) and higher catalog exploration than `MostPop`.

---

### 2. UserKNN (User-Based Memory Collaborative Filtering)

- **Algorithm Family:** Memory-Based Collaborative Filtering (Neighborhood)
- **Cornac Class:** `cornac.models.UserKNN`
- **How It Works:** Computes pairwise similarity (typically Cosine or Pearson correlation) between users based on their historical interaction vectors. To recommend items for user $u$, it aggregates the interactions of the $k$ most similar users, weighted by their similarity scores:
  $$\hat{r}_{ui} = \frac{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in \mathcal{N}_k(u)} |\text{sim}(u, v)|}$$
- **Role in Research:** Heuristic, non-parametric baseline. Because it relies on overlapping user histories, it inherently favors popular items that act as common overlap points between users.

---

### 3. ItemKNN (Item-Based Memory Collaborative Filtering)

- **Algorithm Family:** Memory-Based Collaborative Filtering (Neighborhood)
- **Cornac Class:** `cornac.models.ItemKNN`
- **How It Works:** Computes similarity between items based on user co-occurrence patterns. If user $u$ interacted with set $\mathcal{I}_u$, the model recommends candidate item $j$ by aggregating item-item similarities:
  $$\hat{r}_{uj} = \frac{\sum_{i \in \mathcal{I}_u} \text{sim}(j, i) \cdot r_{ui}}{\sum_{i \in \mathcal{I}_u} |\text{sim}(j, i)|}$$
- **Role in Research:** Often preserves more localized catalog diversity than `UserKNN` because recommendations are anchored to the specific items a user consumed rather than broad user-neighborhood averages.

---

### 4. MF (Standard Pointwise Matrix Factorization)

- **Algorithm Family:** Latent Factor Model (Matrix Factorization)
- **Cornac Class:** `cornac.models.MF`
- **How It Works:** Factorizes the sparse user-item interaction matrix $R \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ into low-rank user embeddings $\mathbf{u}_u \in \mathbb{R}^d$ and item embeddings $\mathbf{v}_i \in \mathbb{R}^d$. It optimizes a Mean Squared Error (MSE) reconstruction loss to approximate observed values:
  $$\mathcal{L}_{\text{MF}} = \sum_{(u,i) \in \Omega} \left(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i\right)^2 + \lambda \left(\|\mathbf{u}_u\|_2^2 + \|\mathbf{v}_i\|_2^2\right)$$
- **Role in Research:** The foundational parametric baseline for embedding dynamics. Demonstrates how unweighted pointwise MSE handles implicit interaction matrices.

---

### 5. PMF (Probabilistic Matrix Factorization)

- **Algorithm Family:** Probabilistic Latent Factor Model
- **Cornac Class:** `cornac.models.PMF`
- **How It Works:** Formulates matrix factorization within a Bayesian probabilistic framework. It assumes Gaussian priors over user and item feature vectors with zero mean and spherical covariance matrices:
  $$p(U \mid \sigma_U^2) = \prod_{u \in \mathcal{U}} \mathcal{N}\left(\mathbf{u}_u \mid \mathbf{0}, \sigma_U^2 \mathbf{I}\right), \quad p(V \mid \sigma_V^2) = \prod_{i \in \mathcal{I}} \mathcal{N}\left(\mathbf{v}_i \mid \mathbf{0}, \sigma_V^2 \mathbf{I}\right)$$
- **Role in Research:** Demonstrates how rigorous Gaussian shrinkage priors affect tail item embeddings under high data sparsity.

---

### 6. SVD (FunkSVD with Explicit Biases)

- **Algorithm Family:** Biased Latent Factor Model
- **Cornac Class:** `cornac.models.SVD`
- **How It Works:** Enhances basic matrix factorization by decomposing predicted preferences into global baseline offsets and interaction terms:
  $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$
  where $\mu$ is the global mean, $b_u$ is user rating bias, and $b_i$ is item popularity bias.
- **Role in Research:** Evaluates whether explicitly isolating item popularity into a dedicated scalar bias term ($b_i$) prevents the latent embedding vectors ($\mathbf{v}_i$) from absorbing popularity distortion.

---

### 7. BPR (Bayesian Personalized Ranking)

- **Algorithm Family:** Pairwise Ranking Collaborative Filtering
- **Cornac Class:** `cornac.models.BPR`
- **How It Works:** Operates on user-specific pairwise preference assumptions. Instead of predicting raw interaction values, it optimizes ranking margins between observed positive items $i$ and unobserved negative items $j$:
  $$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in D_S} \ln \sigma\left(\hat{x}_{uij}\right) + \lambda_{\Theta} \|\Theta\|^2, \quad \text{where } \hat{x}_{uij} = \mathbf{u}_u^T \mathbf{v}_i - \mathbf{u}_u^T \mathbf{v}_j$$
- **Role in Research:** State-of-the-art accuracy benchmark among classical collaborative filtering models, but exhibits intense popularity concentration because head items are far more frequently sampled as positive instances.
# 🎬 Empirical Study: Popularity Bias & Item Representation Geometry in Collaborative Filtering (MF vs. PMF vs. BPR)

---

### 🎯 1. Project Objective & Aim
* **Analyze Item Embedding Geometry:** Examine how popularity bias and severe interaction sparsity impact the distribution and structure of learned latent vectors (embeddings) in collaborative filtering models[cite: 1].
* **Compare Optimization Paradigms:** Investigate whether pointwise rating prediction (MF, PMF) causes tail items to collapse into uninformative representations around the origin, and whether pairwise ranking loss with negative sampling (BPR) mitigates this collapse[cite: 1].
* **Assess Recommender Fairness & Diversity:** Provide a visual and theoretical foundation for understanding why popular blockbusters dominate recommendation lists while niche items struggle to be discovered[cite: 1].

---

### 📊 2. Dataset Profiling & Tail Distribution Analysis
The experiment uses the benchmark **MovieLens 1M** dataset[cite: 1]:
* **Total Interactions:** 1,000,209 explicit ratings on a 1–5 star scale[cite: 1].
* **User Count:** 6,040 users[cite: 1].
* **Item Count:** 3,706 movies[cite: 1].

#### Catalog Stratification
The catalog was segmented into three distinct popularity tiers based on interaction frequency[cite: 1]:

| Tier | Definition | Movie Count | Total Ratings | Rating Share (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Short-Tail** 🌟 | $\ge 250$ ratings | 1,216 movies[cite: 1] | 808,922[cite: 1] | **~80.88%**[cite: 1] |
| **Mid-Tail** ⚖️ | $50 - 249$ ratings | 1,298 movies[cite: 1] | 168,917[cite: 1] | **~16.89%**[cite: 1] |
| **Long-Tail** 📉 | $< 50$ ratings | 1,192 movies[cite: 1] | 22,370[cite: 1] | **~2.24%**[cite: 1] |

> **Key Observation:** The catalog exhibits extreme Pareto-like skew[cite: 1]. While short-tail and long-tail categories contain roughly equal numbers of items (~1,200 movies each), the short tail absorbs more than $80\%$ of all user interactions, leaving the long tail with only $2.24\%$ of feedback[cite: 1].

---

### ⚙️ 3. Experimental Setup & Methodology
* **Framework:** Cornac recommendation library[cite: 1].
* **Data Splitting:** `RatioSplit` with an 80/20 train/test split, random seed = 42, and an interaction rating threshold of 1.0[cite: 1].
* **Latent Factor Dimensionality:** $k = 32$ latent factors for all models[cite: 1].
* **Dimensionality Reduction:** 
  * `sklearn.manifold.TSNE` configured with `n_components=2`, `perplexity=30`, `max_iter=1000`, and `random_state=42`[cite: 1].
  * 32-dimensional item factor matrices were transformed into 2D coordinates and mapped to color-coded scatter plots[cite: 1].

---

### 🧠 4. Architectural Breakdown of the Three Models

#### 1. Matrix Factorization (MF)
* **Optimization Strategy:** Pointwise explicit rating regression using Stochastic Gradient Descent (SGD)[cite: 1].
* **Objective:** Minimizes the squared error between observed ratings $R_{ui}$ and predicted ratings $\hat{R}_{ui} = u_u^\top v_i$:
  $$\min_{U, V} \sum_{(u, i) \in \mathcal{R}_{\text{train}}} (R_{ui} - u_u^\top v_i)^2 + \lambda (\|u_u\|_2^2 + \|v_i\|_2^2)$$
* **Hyperparameters:** `k=32`, `max_iter=30`, `learning_rate=0.01`, `lambda_reg=0.02`[cite: 1].

#### 2. Probabilistic Matrix Factorization (PMF)
* **Optimization Strategy:** Probabilistic formulation assuming Gaussian noise on observed ratings and zero-mean spherical Gaussian priors on user and item feature vectors:
  $$p(R | U, V, \sigma^2) = \prod_{(u,i)} \mathcal{N}(R_{ui} | u_u^\top v_i, \sigma^2), \quad p(V | \sigma_V^2) = \prod_i \mathcal{N}(v_i | 0, \sigma_V^2 I)$$
* **Objective:** Maximizes the log-posterior over latent factors, mathematically equivalent to L2-regularized squared error under a probabilistic Bayesian lens[cite: 1].
* **Hyperparameters:** `k=32`, `max_iter=30`, `learning_rate=0.001`, `lambda_reg=0.01`[cite: 1].

#### 3. Bayesian Personalized Ranking (BPR)
* **Optimization Strategy:** Pairwise implicit ranking using Bayesian maximum posterior estimation[cite: 1].
* **Objective:** Optimizes the margin between an observed interaction $i$ and an unobserved item $j$ for user $u$:
  $$\max_{\Theta} \sum_{(u, i, j) \in \mathcal{D}_S} \ln \sigma\left(\hat{x}_{ui} - \hat{x}_{uj}\right) - \lambda_\Theta \|\Theta\|_2^2$$
  where $\hat{x}_{ui} = u_u^\top v_i$ and $\hat{x}_{uj} = u_u^\top v_j$.
* **Hyperparameters:** `k=32`, `max_iter=30`, `learning_rate=0.01`, `lambda_reg=0.01`[cite: 1].

---

### 📐 5. Explanation of Item Factor Matrix Shape: `(3675, 32)`
During factor extraction, the shape of the item vector matrix was verified as `(3675, 32)` across all models[cite: 1]:
* **Dimension 1 (3,675 Rows):** Represents the number of unique items present in the training set[cite: 1]. While the raw dataset contains 3,706 items, 31 movies only appeared in the 20% test partition (or had no active interactions retained in the training graph), leaving 3,675 learnable item vectors[cite: 1].
* **Dimension 2 (32 Columns):** Corresponds to the latent embedding dimensionality ($k=32$) specified during model instantiation[cite: 1].

---

### 🗺️ 6. Visualization Analysis & Geometric Interpretations

#### Plot Configuration
* **Short-tail (Orange `#D95F02`):** $\ge 250$ interactions[cite: 1].
* **Mid-tail (Purple `#7570B3`):** $50 - 249$ interactions[cite: 1].
* **Long-tail (Green `#1B9E77`):** $< 50$ interactions[cite: 1].

#### Model-Specific Structural Findings:
* **MF & PMF Projections:**
  * **Observation:** Long-tail items (green dots) collapse into a dense, tightly packed core around the center of the t-SNE coordinate space[cite: 1]. Short-tail items (orange dots) spread widely toward the periphery[cite: 1].
  * **Theoretical Root Cause:** Pointwise updates occur only when an item is explicitly rated[cite: 1]. Long-tail items receive fewer than 50 gradient updates[cite: 1]. Because L2 regularization penalizes large weights at every step, their vectors are constrained close to the prior mean (zero), preventing them from acquiring distinct geometric directionality.
* **BPR Projections:**
  * **Observation:** Long-tail items (green dots) are actively dispersed across the latent space alongside mid-tail and short-tail items rather than concentrating at the origin[cite: 1].
  * **Theoretical Root Cause:** In pairwise ranking, every training iteration randomly samples an unobserved item $j$ as a negative example[cite: 1]. Because long-tail items remain unobserved for almost all users, they are frequently sampled as negative items. Different users possess different preference vectors; thus, each negative sampling update pushes the long-tail vector in a different directional trajectory away from each user's taste. This continuous feedback actively disperses tail embeddings throughout the latent manifold.

---

### 💡 7. Practical Implications & Evaluation Roadmap

#### Trade-offs:
* **Pointwise Models (MF/PMF):** High rating prediction accuracy for blockbuster items, but suffer from low novelty, limited catalog exploration, and severe recommendation concentration.
* **Pairwise Models (BPR):** Better representation dispersion and tail discoverability, but uniform negative sampling introduces false-negative noise (sampling an unobserved movie the user would have liked had they known about it).

#### Recommended Quantitative Metrics for the Next Phase:
1. **Ranking Accuracy:**
   * $\text{NDCG}@K$ (Normalized Discounted Cumulative Gain)
   * $\text{Recall}@K$ and $\text{Precision}@K$
2. **Catalog Coverage & Diversity:**
   * **Item Coverage:** Percentage of the total catalog recommended to at least one user ($\frac{|\bigcup_u \text{Rec}_u|}{|\mathcal{I}|}$).
   * **Gini Index:** Quantifies the inequality of recommendation distribution across all items.
3. **Novelty & Long-Tail Fairness:**
   * **Average Self-Information / Novelty:** $-\frac{1}{K}\sum_{i \in \text{Rec}} \log_2 P(i)$
   * **Long-Tail Hit Rate:** Proportion of hits belonging strictly to items with $<50$ interactions[cite: 1].
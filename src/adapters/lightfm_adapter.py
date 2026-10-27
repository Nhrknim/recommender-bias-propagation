import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Any, Dict
from src.adapters.base import BaseRecommenderAdapter

try:
    import lightfm
    from lightfm import LightFM as C_LightFM
except ImportError:
    lightfm = None
    C_LightFM = None


class _NativeLightFM:
    """
    Vectorized NumPy implementation of LightFM collaborative filtering.
    Used as high-performance fallback when C-compiled LightFM is unavailable
    (e.g. on Windows CPython 3.12 without MSVC Build Tools).
    
    Implements:
      score(u, i) = u_vec . i_vec + u_bias + i_bias
      - Vectorized WARP loss (Weighted Approximate-Rank Pairwise) with rank multiplier
      - Vectorized BPR loss (Bayesian Personalized Ranking)
    """

    def __init__(
        self,
        no_components: int = 32,
        loss: str = "warp",
        learning_rate: float = 0.005,
        item_alpha: float = 1e-6,
        user_alpha: float = 1e-6,
        max_sampled: int = 10,
        batch_size: int = 4096,
        random_state: Optional[np.random.RandomState] = None
    ):
        self.no_components = no_components
        self.loss = loss.lower()
        self.learning_rate = learning_rate
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.max_sampled = max_sampled
        self.batch_size = batch_size
        self.rng = random_state if random_state is not None else np.random.RandomState(42)

        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.n_users: int = 0
        self.n_items: int = 0
        # Sorted flat keys (u * n_items + i) of observed positives, used for
        # O(log N) vectorised rejection of invalid negatives.
        self._positive_keys: Optional[np.ndarray] = None
        self._seen_counts: Optional[np.ndarray] = None

    def _is_positive(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Vectorised membership test for (user, item) positives."""
        if self._positive_keys is None or len(self._positive_keys) == 0:
            return np.zeros(np.broadcast(np.asarray(user_ids), np.asarray(item_ids)).shape, dtype=bool)
        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        if users.ndim == 1 and items.ndim == 1:
            keys = users * self.n_items + items
        else:
            keys = users[..., None] * self.n_items + items
        flat = keys.ravel()
        idx = np.searchsorted(self._positive_keys, flat)
        idx = np.clip(idx, 0, len(self._positive_keys) - 1)
        return (self._positive_keys[idx] == flat).reshape(keys.shape)

    def fit(self, interactions: sp.coo_matrix, epochs: int = 5):
        self.n_users, self.n_items = interactions.shape
        scale = 1.0 / np.sqrt(self.no_components)

        # Truncated uniform initialization
        self.user_embeddings = self.rng.uniform(
            -scale, scale, (self.n_users, self.no_components)
        ).astype(np.float32)
        self.item_embeddings = self.rng.uniform(
            -scale, scale, (self.n_items, self.no_components)
        ).astype(np.float32)

        self.user_biases = np.zeros(self.n_users, dtype=np.float32)
        self.item_biases = np.zeros(self.n_items, dtype=np.float32)

        rows = interactions.row.astype(np.int32)
        cols = interactions.col.astype(np.int32)
        n_pos = len(rows)
        if n_pos == 0:
            return self

        # BPR/WARP are undefined without at least one negative candidate.
        if self.n_items < 2:
            return self

        # Precompute positive-pair lookup keys for vectorised rejection sampling.
        self._positive_keys = np.sort(
            rows.astype(np.int64) * self.n_items + cols.astype(np.int64)
        )
        self._seen_counts = np.bincount(rows, minlength=self.n_users)

        lr = float(self.learning_rate)
        warp_max_rank = float(self.n_items - 1)
        batch_size = min(self.batch_size, n_pos)
        n_batches = int(np.ceil(n_pos / batch_size))

        for epoch in range(epochs):
            perm = self.rng.permutation(n_pos)

            for b in range(n_batches):
                b_idx = perm[b * batch_size : (b + 1) * batch_size]
                b_len = len(b_idx)
                u_batch = rows[b_idx]
                pos_batch = cols[b_idx]

                u_vecs = self.user_embeddings[u_batch]          # (B, d)
                pos_vecs = self.item_embeddings[pos_batch]      # (B, d)
                u_b = self.user_biases[u_batch]                 # (B,)
                pos_b = self.item_biases[pos_batch]             # (B,)

                pos_scores = np.sum(u_vecs * pos_vecs, axis=1) + u_b + pos_b # (B,)

                if self.loss == "warp":
                    # Sample candidates in batch: shape (B, max_sampled)
                    candidates = self.rng.randint(0, self.n_items, size=(b_len, self.max_sampled))

                    # Reject observed positives. WARP's assumption is that a
                    # candidate is a true negative; without this check the model
                    # is frequently trained to rank a user's own liked item down.
                    max_retries = 20
                    for _ in range(max_retries):
                        invalid = self._is_positive(u_batch, candidates)
                        n_invalid = int(invalid.sum())
                        if n_invalid == 0:
                            break
                        candidates[invalid] = self.rng.randint(0, self.n_items, size=n_invalid)
                    invalid = self._is_positive(u_batch, candidates)

                    cand_vecs = self.item_embeddings[candidates] # (B, S, d)
                    cand_b = self.item_biases[candidates]        # (B, S)

                    # Compute candidate scores: (B, S)
                    cand_scores = np.sum(u_vecs[:, None, :] * cand_vecs, axis=2) + u_b[:, None] + cand_b

                    # Violations: 1 + neg_score - pos_score > 0
                    violations = ((1.0 + cand_scores - pos_scores[:, None]) > 0.0) & ~invalid
                    has_violation = np.any(violations, axis=1)

                    if not np.any(has_violation):
                        continue

                    # Select rows with valid violating negatives
                    v_rows = np.where(has_violation)[0]
                    v_viol = violations[v_rows]
                    first_viol = np.argmax(v_viol, axis=1) # First violating column index

                    v_u = u_batch[v_rows]
                    v_pos = pos_batch[v_rows]
                    v_neg = candidates[v_rows, first_viol]

                    # TOCTOU guard: the chosen candidate must not be positive.
                    assert not self._is_positive(v_u, v_neg[:, None]).any(), \
                        "WARP selected a positive item as a negative"

                    v_attempts = first_viol + 1.0
                    rank_multiplier = np.log1p(warp_max_rank / v_attempts)[:, None] # (V, 1)

                    v_u_vec = u_vecs[v_rows]
                    v_pos_vec = pos_vecs[v_rows]
                    v_neg_vec = self.item_embeddings[v_neg]

                    # Vectorized gradient steps with clipping for numerical stability
                    u_step = lr * (rank_multiplier * (v_pos_vec - v_neg_vec) - self.user_alpha * v_u_vec)
                    pos_step = lr * (rank_multiplier * v_u_vec - self.item_alpha * v_pos_vec)
                    neg_step = lr * (-rank_multiplier * v_u_vec - self.item_alpha * v_neg_vec)

                    u_step = np.clip(u_step, -1.0, 1.0)
                    pos_step = np.clip(pos_step, -1.0, 1.0)
                    neg_step = np.clip(neg_step, -1.0, 1.0)

                    np.add.at(self.user_embeddings, v_u, u_step)
                    np.add.at(self.item_embeddings, v_pos, pos_step)
                    np.add.at(self.item_embeddings, v_neg, neg_step)

                    bias_step = np.clip((lr * rank_multiplier.flatten()).astype(np.float32), -0.5, 0.5)
                    np.add.at(self.item_biases, v_pos, bias_step)
                    np.add.at(self.item_biases, v_neg, -bias_step)

                else: # BPR
                    neg_items = self.rng.randint(0, self.n_items, size=b_len)

                    # Reject observed positives; skip rows with no valid negative.
                    max_retries = 20
                    for _ in range(max_retries):
                        invalid = self._is_positive(u_batch, neg_items)
                        n_invalid = int(invalid.sum())
                        if n_invalid == 0:
                            break
                        neg_items[invalid] = self.rng.randint(0, self.n_items, size=n_invalid)
                    invalid = self._is_positive(u_batch, neg_items)
                    valid_rows = ~invalid
                    if not np.any(valid_rows):
                        continue

                    b_u = u_batch[valid_rows]
                    b_pos = pos_batch[valid_rows]
                    b_neg = neg_items[valid_rows]
                    b_u_vecs = u_vecs[valid_rows]
                    b_pos_vecs = pos_vecs[valid_rows]
                    b_u_b = u_b[valid_rows]
                    b_pos_b = pos_b[valid_rows]
                    b_pos_scores = pos_scores[valid_rows]

                    b_neg_vecs = self.item_embeddings[b_neg]
                    b_neg_b = self.item_biases[b_neg]

                    b_neg_scores = np.sum(b_u_vecs * b_neg_vecs, axis=1) + b_u_b + b_neg_b
                    diff = b_pos_scores - b_neg_scores
                    sig = 1.0 / (1.0 + np.exp(np.clip(-diff, -15.0, 15.0)))
                    coeff = (1.0 - sig)[:, None] # (B, 1)

                    u_step = lr * (coeff * (b_pos_vecs - b_neg_vecs) - self.user_alpha * b_u_vecs)
                    pos_step = lr * (coeff * b_u_vecs - self.item_alpha * b_pos_vecs)
                    neg_step = lr * (-coeff * b_u_vecs - self.item_alpha * b_neg_vecs)

                    u_step = np.clip(u_step, -1.0, 1.0)
                    pos_step = np.clip(pos_step, -1.0, 1.0)
                    neg_step = np.clip(neg_step, -1.0, 1.0)

                    assert not self._is_positive(b_u, b_neg[:, None]).any(), \
                        "BPR selected a positive item as a negative"

                    np.add.at(self.user_embeddings, b_u, u_step)
                    np.add.at(self.item_embeddings, b_pos, pos_step)
                    np.add.at(self.item_embeddings, b_neg, neg_step)

                    b_step = np.clip((lr * coeff.flatten()).astype(np.float32), -0.5, 0.5)
                    np.add.at(self.item_biases, b_pos, b_step)
                    np.add.at(self.item_biases, b_neg, -b_step)

            # Prevent runaway norms across epochs
            np.clip(self.user_embeddings, -10.0, 10.0, out=self.user_embeddings)
            np.clip(self.item_embeddings, -10.0, 10.0, out=self.item_embeddings)
            np.clip(self.user_biases, -5.0, 5.0, out=self.user_biases)
            np.clip(self.item_biases, -5.0, 5.0, out=self.item_biases)

        return self

    def get_user_representations(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.user_biases, self.user_embeddings

    def get_item_representations(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.item_biases, self.item_embeddings

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u_factors = self.user_embeddings[user_ids]
        i_factors = self.item_embeddings[item_ids]
        dots = np.sum(u_factors * i_factors, axis=1)
        return dots + self.user_biases[user_ids] + self.item_biases[item_ids]


class LightFMAdapter(BaseRecommenderAdapter):
    """
    Facade Adapter wrapping the LightFM recommendation backbone.
    Enforces the BaseRecommenderAdapter interface, global coordinate alignment,
    vectorized top-K recommendations, and deterministic reproducibility.
    """

    def __init__(
        self,
        dim: int = 32,
        loss: str = "warp",
        learning_rate: float = 0.005,
        item_alpha: float = 1e-6,
        user_alpha: float = 1e-6,
        max_sampled: int = 10,
        epochs: int = 5,
        num_threads: int = 1,
        force_native: bool = False,
        **model_kwargs: Any
    ):
        """
        Args:
            dim: Latent factor dimensionality (no_components in LightFM).
            loss: Loss function ('warp', 'bpr', 'warp-kos', 'logistic').
            learning_rate: Initial learning rate for SGD/Adagrad (default 0.005 for stability).
            item_alpha: L2 penalty on item features/embeddings.
            user_alpha: L2 penalty on user features/embeddings.
            max_sampled: Maximum number of negative samples used in WARP loss.
            epochs: Number of training epochs per generation.
            num_threads: Number of parallel worker threads (1 for strict determinism).
            force_native: If True, forces pure Python/NumPy fallback engine.
            **model_kwargs: Additional arguments forwarded to LightFM constructor.
        """
        self.dim = dim
        self.loss = loss
        self.learning_rate = learning_rate
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.max_sampled = max_sampled
        self.epochs = epochs
        self.num_threads = num_threads
        self.force_native = force_native
        self.model_kwargs = model_kwargs

        self.model: Optional[Any] = None
        self.train_matrix: Optional[sp.csr_matrix] = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(self, train_matrix: sp.csr_matrix, seed: int) -> "LightFMAdapter":
        """
        Retrains the LightFM model cold on interaction matrix D_t with a fixed deterministic seed.

        Args:
            train_matrix: CSR interaction matrix of shape (|U|, |I|).
            seed: Deterministic integer seed.

        Returns:
            self: Fitted adapter instance.
        """
        self.n_users, self.n_items = train_matrix.shape
        self.train_matrix = train_matrix

        coo = train_matrix.tocoo()
        if coo.nnz == 0:
            raise ValueError("Cannot fit LightFM model on empty interaction matrix.")

        random_state = np.random.RandomState(seed)

        # Use C-accelerated LightFM if available and not forced native
        if C_LightFM is not None and not self.force_native:
            self.model = C_LightFM(
                no_components=self.dim,
                loss=self.loss,
                learning_rate=self.learning_rate,
                item_alpha=self.item_alpha,
                user_alpha=self.user_alpha,
                max_sampled=self.max_sampled,
                random_state=random_state,
                **self.model_kwargs
            )
            self.model.fit(
                coo,
                epochs=self.epochs,
                num_threads=self.num_threads,
                verbose=False
            )
        else:
            # Seamless high-performance vectorized pure-NumPy engine
            self.model = _NativeLightFM(
                no_components=self.dim,
                loss=self.loss,
                learning_rate=self.learning_rate,
                item_alpha=self.item_alpha,
                user_alpha=self.user_alpha,
                max_sampled=self.max_sampled,
                random_state=random_state
            )
            self.model.fit(coo, epochs=self.epochs)

        return self

    def recommend(
        self,
        user_indices: np.ndarray,
        k: int = 10,
        remove_seen: bool = True
    ) -> np.ndarray:
        """
        Generates top-K recommendation lists for requested active users using
        vectorized batch dot product scoring.

        Args:
            user_indices: 1D array of active user global indices, shape (N_users,).
            k: Top-K cut-off length.
            remove_seen: Whether to exclude training items.

        Returns:
            recs_matrix: 2D array of shape (N_users, K) containing top-K global item codes.
        """
        if self.model is None or self.train_matrix is None:
            raise RuntimeError("Adapter model has not been fitted yet. Call fit() first.")

        n_active = len(user_indices)
        if n_active == 0:
            return np.zeros((0, k), dtype=np.int32)

        # Full-catalog scores: score(u, i) = (u_factor . i_factor) + b_u + b_i
        scores = self.score_all(user_indices)

        # Mask seen items from training matrix
        if remove_seen:
            for idx, u in enumerate(user_indices):
                seen_items = self.train_matrix[u].indices
                if len(seen_items) > 0:
                    scores[idx, seen_items] = -np.inf

        k_arg = min(k, self.n_items)

        # Fast top-K extraction via argpartition
        partition_idx = np.argpartition(-scores, k_arg - 1, axis=1)[:, :k_arg]
        row_indices = np.arange(n_active)[:, None]
        partition_scores = scores[row_indices, partition_idx]
        sorted_order = np.argsort(-partition_scores, axis=1)
        top_k = np.take_along_axis(partition_idx, sorted_order, axis=1)

        # Pad with -1 if k > catalog size
        if k > self.n_items:
            padding = np.full((n_active, k - self.n_items), -1, dtype=np.int32)
            top_k = np.hstack([top_k, padding])

        assert top_k.shape == (n_active, k), f"Shape misaligned: {top_k.shape} vs ({n_active}, {k})"
        return top_k.astype(np.int32)

    def score_all(self, user_indices: np.ndarray) -> np.ndarray:
        """Return real full-catalog scores of shape (len(user_indices), |I|)."""
        if self.model is None:
            raise RuntimeError("Adapter model has not been fitted yet. Call fit() first.")
        user_indices = np.asarray(user_indices, dtype=np.int64)
        user_biases, user_factors = self.model.get_user_representations()
        item_biases, item_factors = self.model.get_item_representations()
        scores = (
            user_factors[user_indices] @ item_factors.T
            + item_biases[None, :]
            + user_biases[user_indices, None]
        )
        assert scores.shape == (len(user_indices), self.n_items)
        assert not np.isnan(scores).any(), "NaN detected in full-catalog scores"
        return scores.astype(np.float32)

    def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts learned user and item factor matrices aligned to global matrix coordinates.

        Returns:
            U_global: User embedding matrix of shape (|U|, d).
            V_global: Item embedding matrix of shape (|I|, d).
        """
        if self.model is None:
            return None, None

        _, user_factors = self.model.get_user_representations()
        _, item_factors = self.model.get_item_representations()

        assert user_factors.shape == (self.n_users, self.dim), f"User shape mismatch: {user_factors.shape}"
        assert item_factors.shape == (self.n_items, self.dim), f"Item shape mismatch: {item_factors.shape}"
        assert not np.isnan(user_factors).any(), "NaN detected in user embeddings"
        assert not np.isnan(item_factors).any(), "NaN detected in item embeddings"

        return user_factors.astype(np.float32), item_factors.astype(np.float32)

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """
        Computes pointwise predicted ranking scores for pairs of (user, item).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if hasattr(self.model, "predict"):
            try:
                return self.model.predict(user_indices, item_indices, num_threads=self.num_threads)
            except TypeError:
                return self.model.predict(user_indices, item_indices)

        user_biases, user_factors = self.model.get_user_representations()
        item_biases, item_factors = self.model.get_item_representations()
        dots = np.sum(user_factors[user_indices] * item_factors[item_indices], axis=1)
        return dots + user_biases[user_indices] + item_biases[item_indices]

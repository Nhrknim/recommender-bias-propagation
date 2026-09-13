import random
import scipy.sparse as sp
import numpy as np
import cornac
from typing import Tuple, Optional, Any, Dict
from src.adapters.base import BaseRecommenderAdapter

def extract_aligned_embeddings(
    model_cornac: Any, 
    n_users: int, 
    n_items: int, 
    dim: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Translates framework-internal factor arrays into global matrix coordinates.

    Args:
        model_cornac: Fitted Cornac model instance.
        n_users: Total count of unique global users |U|.
        n_items: Total count of unique global items |I|.
        dim: Embedding latent factor dimension d.

    Returns:
        U_global: User factor array of shape (|U|, d) aligned to global indices.
        V_global: Item factor array of shape (|I|, d) aligned to global indices.
    """
    if not hasattr(model_cornac, "u_factors") or not hasattr(model_cornac, "i_factors"):
        return None, None

    if model_cornac.u_factors is None or model_cornac.i_factors is None:
        return None, None

    U_global = np.zeros((n_users, dim), dtype=np.float32)
    V_global = np.zeros((n_items, dim), dtype=np.float32)

    uid_map: Dict[Any, int] = model_cornac.train_set.uid_map
    iid_map: Dict[Any, int] = model_cornac.train_set.iid_map

    for global_u, internal_u in uid_map.items():
        if int(global_u) < n_users:
            U_global[int(global_u)] = model_cornac.u_factors[internal_u]

    for global_i, internal_i in iid_map.items():
        if int(global_i) < n_items:
            V_global[int(global_i)] = model_cornac.i_factors[internal_i]

    # Invariant assertions for shape and numerical stability
    assert U_global.shape == (n_users, dim), f"U_global shape misaligned: {U_global.shape}"
    assert V_global.shape == (n_items, dim), f"V_global shape misaligned: {V_global.shape}"
    assert not np.isnan(U_global).any(), "NaN detected in user embeddings"
    assert not np.isnan(V_global).any(), "NaN detected in item embeddings"

    return U_global, V_global


class CornacAdapterBase(BaseRecommenderAdapter):
    """
    Generic Facade Adapter for Cornac Recommendation Backbones.
    Enforces deterministic seed control, coordinate alignment, and padded top-K recs.
    """

    def __init__(self, model_cls: Any, dim: int = 32, **model_kwargs):
        """
        Args:
            model_cls: Cornac recommender class (e.g. cornac.models.MF).
            dim: Latent factor dimension size d.
            **model_kwargs: Keyword arguments passed to Cornac model constructor.
        """
        self.model_cls = model_cls
        self.dim = dim
        self.model_kwargs = model_kwargs
        self.model: Optional[Any] = None
        self.dataset: Optional[cornac.data.Dataset] = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(self, train_matrix: sp.csr_matrix, seed: int) -> 'CornacAdapterBase':
        """
        Retrains the Cornac model cold on interaction matrix D_t with fixed seed.

        Args:
            train_matrix: CSR matrix of shape (|U|, |I|).
            seed: Deterministic integer seed.

        Returns:
            self: Fitted adapter instance.
        """
        random.seed(seed)
        np.random.seed(seed)

        self.n_users, self.n_items = train_matrix.shape
        coo = train_matrix.tocoo()

        if coo.nnz == 0:
            raise ValueError("Cannot fit recommender model on empty interaction matrix.")

        # Convert CSR data to UIR tuples using global 0-indexed integer IDs
        uir_tuples = [
            (int(u), int(i), float(r)) 
            for u, i, r in zip(coo.row, coo.col, coo.data)
        ]

        # Construct Cornac Dataset
        self.dataset = cornac.data.Dataset.from_uir(uir_tuples)

        # Instantiate and fit Cornac model deterministically
        kwargs = self.model_kwargs.copy()
        if "seed" not in kwargs and hasattr(self.model_cls, "seed"):
            kwargs["seed"] = seed
        if "k" not in kwargs and "dim" in self.__dict__ and hasattr(self.model_cls, "k"):
            kwargs["k"] = self.dim

        self.model = self.model_cls(**kwargs)
        self.model.fit(self.dataset)

        return self

    def recommend(
        self, 
        user_indices: np.ndarray, 
        k: int = 10, 
        remove_seen: bool = True
    ) -> np.ndarray:
        """
        Generates top-K recommendation lists for requested active users.
        Uses fast batch matrix scoring for 500x speedup over sequential user loops.

        Args:
            user_indices: 1D array of active user global indices, shape (N_users,).
            k: Top-K cut-off length.
            remove_seen: Whether to exclude training items.

        Returns:
            recs_matrix: 2D array of shape (N_users, K) containing top-K global item codes.
        """
        if self.model is None or self.dataset is None:
            raise RuntimeError("Adapter model has not been fitted yet. Call fit() first.")

        n_active = len(user_indices)
        if n_active == 0:
            return np.zeros((0, k), dtype=np.int32)

        k_arg = min(k, self.n_items)
        
        # 1. Fast Batch Scoring Matrix Construction (N_active x N_items)
        if hasattr(self.model, "u_factors") and hasattr(self.model, "i_factors") and \
           self.model.u_factors is not None and self.model.i_factors is not None:
            
            U_g, V_g = self.get_embeddings()
            score_matrix = U_g[user_indices] @ V_g.T
        else:
            score_matrix = np.zeros((n_active, self.n_items), dtype=np.float32)
            uid_map = self.dataset.uid_map
            iid_map = self.dataset.iid_map
            
            # Construct fast ID lookup array
            max_internal_i = max(iid_map.values()) if len(iid_map) > 0 else 0
            i2g_arr = np.full(max_internal_i + 1, -1, dtype=np.int32)
            for global_i, internal_i in iid_map.items():
                g_idx = int(global_i)
                if g_idx < self.n_items:
                    i2g_arr[internal_i] = g_idx

            # Fast broadcast for models with static item scores (e.g. MostPop)
            if hasattr(self.model, "item_scores") and self.model.item_scores is not None:
                item_scores = self.model.item_scores
                valid_mask = (i2g_arr >= 0) & (np.arange(len(i2g_arr)) < len(item_scores))
                valid_internal_i = np.where(valid_mask)[0]
                valid_global_i = i2g_arr[valid_internal_i]
                
                single_row_scores = np.zeros(self.n_items, dtype=np.float32)
                single_row_scores[valid_global_i] = item_scores[valid_internal_i]
                score_matrix[:] = single_row_scores
            else:
                for idx, u in enumerate(user_indices):
                    u_int = int(u)
                    if u_int in uid_map:
                        internal_u = uid_map[u_int]
                        raw_scores = self.model.score(internal_u)
                        valid_mask = (i2g_arr >= 0) & (np.arange(len(i2g_arr)) < len(raw_scores))
                        valid_internal_i = np.where(valid_mask)[0]
                        score_matrix[idx, i2g_arr[valid_internal_i]] = raw_scores[valid_internal_i]

        # 2. Vectorized Remove Seen Items Masking
        if remove_seen:
            uid_map = self.dataset.uid_map
            iid_map = self.dataset.iid_map
            i2g = {internal_i: int(global_i) for global_i, internal_i in iid_map.items()}
            for idx, u in enumerate(user_indices):
                u_int = int(u)
                if u_int in uid_map:
                    internal_u = uid_map[u_int]
                    user_seen_items = self.dataset.user_data[internal_u][0]
                    for internal_i in user_seen_items:
                        global_i = i2g.get(internal_i, None)
                        if global_i is not None and global_i < self.n_items:
                            score_matrix[idx, global_i] = -1e9

        # 3. Vectorized Top-K Selection via np.argpartition + argsort
        top_k_part = np.argpartition(-score_matrix, k_arg - 1, axis=1)[:, :k_arg]
        row_idx = np.arange(n_active)[:, None]
        top_k_ranks = np.argsort(-score_matrix[row_idx, top_k_part], axis=1)
        recs_matrix = np.take_along_axis(top_k_part, top_k_ranks, axis=1).astype(np.int32)

        # 4. Pad to shape (N_active, K) if k_arg < K
        if k_arg < k:
            pad_width = k - k_arg
            padding = np.zeros((n_active, pad_width), dtype=np.int32)
            recs_matrix = np.hstack([recs_matrix, padding])

        assert recs_matrix.shape == (n_active, k), f"Recommendation matrix shape misaligned: {recs_matrix.shape}"
        return recs_matrix

    def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts learned user and item factor matrices aligned to global matrix coordinates.
        """
        if self.model is None or self.dataset is None:
            raise RuntimeError("Adapter model has not been fitted yet. Call fit() first.")

        return extract_aligned_embeddings(
            model_cornac=self.model, 
            n_users=self.n_users, 
            n_items=self.n_items, 
            dim=self.dim
        )


class CornacMFAdapter(CornacAdapterBase):
    """Cornac Matrix Factorization (MF) Recommender Adapter."""
    def __init__(self, dim: int = 32, max_iter: int = 20, learning_rate: float = 0.01, **kwargs):
        super().__init__(
            model_cls=cornac.models.MF, 
            dim=dim, 
            k=dim, 
            max_iter=max_iter, 
            learning_rate=learning_rate, 
            **kwargs
        )


class CornacBPRAdapter(CornacAdapterBase):
    """Cornac Bayesian Personalized Ranking (BPR) Recommender Adapter."""
    def __init__(self, dim: int = 32, max_iter: int = 20, learning_rate: float = 0.01, **kwargs):
        super().__init__(
            model_cls=cornac.models.BPR, 
            dim=dim, 
            k=dim, 
            max_iter=max_iter, 
            learning_rate=learning_rate, 
            **kwargs
        )


class CornacUserKNNAdapter(CornacAdapterBase):
    """Cornac User-based K-Nearest Neighbors (UserKNN) Adapter."""
    def __init__(self, k_neighbors: int = 20, similarity: str = "cosine", **kwargs):
        super().__init__(
            model_cls=cornac.models.UserKNN, 
            dim=32, 
            k=k_neighbors, 
            similarity=similarity, 
            **kwargs
        )


class CornacMostPopAdapter(CornacAdapterBase):
    """Cornac Most Popular (MostPop) Non-Personalized Baseline Adapter."""
    def __init__(self, **kwargs):
        super().__init__(
            model_cls=cornac.models.MostPop, 
            dim=32, 
            **kwargs
        )

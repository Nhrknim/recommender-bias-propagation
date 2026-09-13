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
        recs_matrix = np.zeros((n_active, k), dtype=np.int32)
        k_arg = min(k, self.n_items)
        all_items = np.arange(self.n_items, dtype=np.int32)

        for idx, u in enumerate(user_indices):
            u_int = int(u)
            if u_int in self.dataset.uid_map:
                user_recs = self.model.recommend(
                    user_id=u_int, 
                    k=k_arg, 
                    remove_seen=remove_seen, 
                    train_set=self.dataset
                )
                rec_list = [int(item_id) for item_id in user_recs]
            else:
                rec_list = []

            # Pad recommendation list to guarantee exact length K
            if len(rec_list) < k:
                seen_set = set(rec_list)
                if remove_seen and u_int in self.dataset.uid_map:
                    internal_u = self.dataset.uid_map[u_int]
                    # Exclude items present in training dataset for user
                    user_seen_items = [
                        item_id for item_id, _ in self.dataset.user_data[internal_u]
                    ]
                    # Map internal item indices back to global original item IDs
                    i2g = {internal_i: global_i for global_i, internal_i in self.dataset.iid_map.items()}
                    seen_set.update([i2g[i] for i in user_seen_items if i in i2g])

                padding = [item for item in all_items if item not in seen_set]
                rec_list.extend(padding[:k - len(rec_list)])

                # Fallback if catalog size < k
                while len(rec_list) < k:
                    rec_list.append(0)

            recs_matrix[idx] = np.array(rec_list[:k], dtype=np.int32)

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

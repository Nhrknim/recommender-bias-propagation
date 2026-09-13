import scipy.sparse as sp
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseRecommenderAdapter(ABC):
    """
    Abstract Facade Interface for Recommendation Backbones.
    
    All concrete recommender adapters (e.g. Cornac MF, BPR, UserKNN, MostPop)
    must inherit from this base class and enforce strict coordinate alignment
    with the global zero-indexed user/item space.
    """

    @abstractmethod
    def fit(self, train_matrix: sp.csr_matrix, seed: int) -> 'BaseRecommenderAdapter':
        """
        Retrains the recommendation algorithm cold on interaction matrix D_t.

        Args:
            train_matrix: Sparse CSR interaction matrix of shape (|U|, |I|).
            seed: Deterministic integer seed for reproducible random state.

        Returns:
            self: Fitted instance of the adapter.
        """
        pass

    @abstractmethod
    def recommend(
        self, 
        user_indices: np.ndarray, 
        k: int = 10, 
        remove_seen: bool = True
    ) -> np.ndarray:
        """
        Generates top-K recommendation item lists for requested active users.

        Args:
            user_indices: 1D NumPy array of active user global indices, shape (N_users,).
            k: Number of items to recommend per user.
            remove_seen: If True, filters out items already present in training matrix.

        Returns:
            recs: 2D NumPy array of shape (N_users, K) containing top-K global item indices.
        """
        pass

    @abstractmethod
    def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts learned user and item factor matrices aligned to global matrix coordinates.

        Returns:
            U_global: User embedding matrix of shape (|U|, d) or None if non-latent.
            V_global: Item embedding matrix of shape (|I|, d) or None if non-latent.
        """
        pass

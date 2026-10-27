import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.adapters.base import BaseRecommenderAdapter


# ─── LightGCN Graph Convolutional Model ──────────────────────────────────────

class LightGCNModel(nn.Module):
    """
    LightGCN: Simplified Graph Convolutional Network for Collaborative Filtering.

    Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
    Network for Recommendation", KDD 2020. https://arxiv.org/abs/2002.02126

    Propagation rule (no feature transformation, no nonlinear activation):
        E^(l+1) = D^{-1/2} A D^{-1/2} E^(l)

    Final embedding: mean-pooled across all layers (including E^(0)):
        E_final = (1 / (L+1)) * sum_{l=0}^{L} E^(l)

    Args:
        n_users: Total number of unique global users |U|.
        n_items: Total number of unique global items |I|.
        dim: Latent factor embedding dimension d.
        n_layers: Number of graph propagation layers L.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 32, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers

        # Learnable base (0-th layer) embeddings
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def build_norm_adj(
        self,
        interact_mat: sp.csr_matrix,  # Shape: (|U|, |I|)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Constructs the symmetric normalised adjacency matrix of the user-item
        bipartite graph as a sparse torch COO tensor.

        The (n_users + n_items) x (n_users + n_items) block adjacency is:
            [[0,   R  ],
             [R^T, 0  ]]
        normalised as D^{-1/2} A D^{-1/2}.

        Args:
            interact_mat: CSR binary interaction matrix of shape (|U|, |I|).
            device: Target PyTorch device for the tensor.

        Returns:
            norm_adj: Sparse COO tensor of shape (|U|+|I|, |U|+|I|).
        """
        n_total = self.n_users + self.n_items
        coo = interact_mat.tocoo()
        # Collapse duplicate coordinates so parallel edges cannot inflate degrees.
        coo.sum_duplicates()

        # Build symmetric block: user->item edges and item->user edges
        rows = np.concatenate([coo.row, coo.col + self.n_users])
        cols = np.concatenate([coo.col + self.n_users, coo.row])
        data = np.ones(len(rows), dtype=np.float32)

        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))

        # Degree vector for D^{-1/2} normalisation
        deg = np.array(adj.sum(axis=1), dtype=np.float32).flatten()
        d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg)), 0.0).astype(np.float32)

        # Symmetric normalisation: norm_data[e] = D^{-1/2}[row] * D^{-1/2}[col]
        norm_data = d_inv_sqrt[adj.row] * adj.data * d_inv_sqrt[adj.col]

        assert not np.isnan(norm_data).any(), "NaN in normalised adjacency weights"
        assert not np.isinf(norm_data).any(), "Inf in normalised adjacency weights"

        indices = torch.tensor(
            np.stack([adj.row, adj.col]), dtype=torch.long, device=device
        )
        values = torch.tensor(norm_data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (n_total, n_total)).coalesce()

    def forward(
        self,
        norm_adj: torch.Tensor,  # Shape: (|U|+|I|, |U|+|I|) sparse
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes L-layer LightGCN graph propagation and returns mean-pooled embeddings.

        Args:
            norm_adj: Normalised adjacency sparse tensor of shape (|U|+|I|, |U|+|I|).

        Returns:
            U_out: Aggregated user embeddings of shape (|U|, d).
            V_out: Aggregated item embeddings of shape (|I|, d).
        """
        # Concatenate learnable base embeddings into single node matrix
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        E_layers = [E]

        for _ in range(self.n_layers):
            E = torch.sparse.mm(norm_adj, E)
            E_layers.append(E)

        # Mean pooling across all L+1 layer outputs
        E_final = torch.stack(E_layers, dim=1).mean(dim=1)

        return E_final[: self.n_users], E_final[self.n_users :]

    def bpr_loss(
        self,
        norm_adj: torch.Tensor,         # Shape: (|U|+|I|, |U|+|I|) sparse
        user_ids: torch.Tensor,          # Shape: (B,) long
        pos_item_ids: torch.Tensor,      # Shape: (B,) long
        neg_item_ids: torch.Tensor,      # Shape: (B,) long
        reg: float = 1e-4,
    ) -> torch.Tensor:
        """
        Computes Bayesian Personalised Ranking (BPR) loss with L2 regularisation
        on the base (0-th layer) embeddings only (as per the LightGCN paper).

        Args:
            norm_adj: Normalised adjacency tensor for graph propagation.
            user_ids: Batch user global indices, shape (B,).
            pos_item_ids: Batch positive item global indices, shape (B,).
            neg_item_ids: Batch sampled negative item global indices, shape (B,).
            reg: L2 regularisation coefficient.

        Returns:
            loss: Scalar BPR + L2 regularisation loss.
        """
        U, V = self.forward(norm_adj)
        u_vecs = U[user_ids]           # (B, d)
        pos_vecs = V[pos_item_ids]     # (B, d)
        neg_vecs = V[neg_item_ids]     # (B, d)

        pos_scores = (u_vecs * pos_vecs).sum(dim=1)   # (B,)
        neg_scores = (u_vecs * neg_vecs).sum(dim=1)   # (B,)

        bpr = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-10
        ).mean()

        # L2 reg on base embeddings only (not the propagated ones)
        reg_loss = reg * (
            self.user_emb.weight[user_ids].norm(2).pow(2)
            + self.item_emb.weight[pos_item_ids].norm(2).pow(2)
            + self.item_emb.weight[neg_item_ids].norm(2).pow(2)
        ) / (len(user_ids) + 1e-9)

        return bpr + reg_loss


# ─── Adapter ─────────────────────────────────────────────────────────────────

class LightGCNAdapter(BaseRecommenderAdapter):
    """
    LightGCN Graph Collaborative Filtering Adapter.

    Implements 3-layer (default) LightGCN trained with BPR loss.
    Inherits from BaseRecommenderAdapter and exposes the standard
    fit() / recommend() / get_embeddings() facade used by the
    closed-loop simulation orchestrator.

    Device selection priority: MPS (Apple Silicon) > CUDA > CPU.
    """

    def __init__(
        self,
        dim: int = 32,
        n_layers: int = 3,
        epochs: int = 20,
        batch_size: int = 2048,
        lr: float = 1e-3,
        reg: float = 1e-4,
        device: Optional[str] = None,
    ):
        """
        Args:
            dim: Latent factor embedding dimension d.
            n_layers: Number of LightGCN propagation layers L.
            epochs: Number of BPR training epochs.
            batch_size: Mini-batch size for BPR sampling.
            lr: Adam optimiser learning rate.
            reg: L2 regularisation weight on base embeddings.
            device: Optional device string ('cpu', 'cuda', 'mps').
                    Auto-selects best available device if None.
        """
        self.dim = dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg

        # Auto-select accelerator: CUDA > CPU (MPS disabled for sparse COO graph ops due to multi-process deadlock)
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[LightGCNModel] = None
        self.n_users: int = 0
        self.n_items: int = 0
        self._train_matrix: Optional[sp.csr_matrix] = None

        # Cached final-layer embeddings (aligned to global coordinate space)
        self._U: Optional[np.ndarray] = None   # Shape: (|U|, d)
        self._V: Optional[np.ndarray] = None   # Shape: (|I|, d)

    def fit(self, train_matrix: sp.csr_matrix, seed: int) -> "LightGCNAdapter":
        """
        Retrains LightGCN cold on interaction matrix D_t with fixed seed.

        Builds the normalised bipartite adjacency matrix, then trains the
        LightGCNModel with BPR loss using mini-batch uniform negative sampling.
        Caches the mean-pooled embeddings for fast batch scoring at inference.

        Args:
            train_matrix: CSR binary interaction matrix of shape (|U|, |I|).
            seed: Deterministic integer seed for all RNG sources.

        Returns:
            self: Fitted LightGCNAdapter instance.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.n_users, self.n_items = train_matrix.shape
        self._train_matrix = train_matrix

        if train_matrix.nnz == 0:
            raise ValueError(
                "Cannot fit LightGCN on empty interaction matrix. "
                f"Received matrix of shape {train_matrix.shape} with 0 non-zero entries."
            )

        # Build model and normalised adjacency graph
        self.model = LightGCNModel(
            n_users=self.n_users,
            n_items=self.n_items,
            dim=self.dim,
            n_layers=self.n_layers,
        ).to(self.device)

        norm_adj = self.model.build_norm_adj(train_matrix, self.device)

        # Extract all positive (user, item) interaction pairs for BPR
        coo = train_matrix.tocoo()
        all_users = coo.row.astype(np.int64)   # Shape: (nnz,)
        all_items = coo.col.astype(np.int64)   # Shape: (nnz,)

        # Flat sorted positive keys (u * |I| + i) for vectorised membership checks.
        positive_keys = np.sort(all_users * self.n_items + all_items)
        seen_counts = np.bincount(all_users, minlength=self.n_users)

        def is_positive(users: np.ndarray, items: np.ndarray) -> np.ndarray:
            """Boolean mask of (user, item) pairs that exist in D_t."""
            users = np.asarray(users, dtype=np.int64)
            items = np.asarray(items, dtype=np.int64)
            if users.ndim == 1 and items.ndim == 1:
                keys = users * self.n_items + items
            else:
                keys = users[..., None] * self.n_items + items
            flat = keys.ravel()
            loc = np.clip(np.searchsorted(positive_keys, flat), 0, len(positive_keys) - 1)
            return (positive_keys[loc] == flat).reshape(keys.shape)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            idx = np.random.permutation(len(all_users))

            for start in range(0, len(idx), self.batch_size):
                batch = idx[start : start + self.batch_size]

                u_np = all_users[batch]
                pos_np = all_items[batch]

                # A user who has interacted with the entire catalog has no valid
                # negative. Skip those rows instead of looping forever.
                sampleable = seen_counts[u_np] < self.n_items
                if not np.any(sampleable):
                    continue
                u_np = u_np[sampleable]
                pos_np = pos_np[sampleable]

                # Vectorised uniform sampling followed by bounded rejection.
                neg_np = np.random.randint(0, self.n_items, size=len(u_np))
                for _retry in range(20):
                    invalid = is_positive(u_np, neg_np)
                    n_invalid = int(invalid.sum())
                    if n_invalid == 0:
                        break
                    neg_np[invalid] = np.random.randint(0, self.n_items, size=n_invalid)
                invalid = is_positive(u_np, neg_np)
                valid = ~invalid
                if not np.any(valid):
                    continue
                u_np = u_np[valid]
                pos_np = pos_np[valid]
                neg_np = neg_np[valid]

                assert not is_positive(u_np, neg_np).any(), \
                    "LightGCN selected a positive item as a negative"

                u_batch = torch.tensor(u_np, device=self.device)
                pos_batch = torch.tensor(pos_np, device=self.device)
                neg_batch = torch.tensor(neg_np, device=self.device)

                optimizer.zero_grad()
                loss = self.model.bpr_loss(
                    norm_adj, u_batch, pos_batch, neg_batch, self.reg
                )
                loss.backward()
                optimizer.step()

        # Cache aligned embeddings after training is complete
        self.model.eval()
        with torch.no_grad():
            U_t, V_t = self.model.forward(norm_adj)
            self._U = U_t.cpu().numpy().astype(np.float32)  # Shape: (|U|, d)
            self._V = V_t.cpu().numpy().astype(np.float32)  # Shape: (|I|, d)

        assert self._U.shape == (self.n_users, self.dim), \
            f"User embedding shape misaligned: {self._U.shape}"
        assert self._V.shape == (self.n_items, self.dim), \
            f"Item embedding shape misaligned: {self._V.shape}"
        assert not np.isnan(self._U).any(), "NaN detected in LightGCN user embeddings"
        assert not np.isnan(self._V).any(), "NaN detected in LightGCN item embeddings"

        return self

    def recommend(
        self,
        user_indices: np.ndarray,   # Shape: (N_active,)
        k: int = 10,
        remove_seen: bool = True,
    ) -> np.ndarray:                # Shape: (N_active, K)
        """
        Generates top-K recommendation lists for requested active users.
        Uses vectorised batch matrix multiply (N_active × d) @ (d × |I|) for
        O(N_active * |I| * d) scoring — no per-user Python loops.

        Args:
            user_indices: 1D array of active user global indices, shape (N_active,).
            k: Top-K cut-off length.
            remove_seen: Whether to mask out training interactions.

        Returns:
            recs_matrix: 2D array of shape (N_active, K) with top-K global item indices.
        """
        if self._U is None or self._V is None:
            raise RuntimeError(
                "LightGCNAdapter has not been fitted yet. Call fit() first."
            )

        n_active = len(user_indices)
        if n_active == 0:
            return np.zeros((0, k), dtype=np.int32)

        k_arg = min(k, self.n_items)

        # 1. Vectorised batch scoring: (N_active, d) @ (d, |I|) -> (N_active, |I|)
        score_matrix = self.score_all(user_indices)  # (N_active, n_items)

        # 2. Mask seen items — vectorised over sparse row slices
        if remove_seen and self._train_matrix is not None:
            for idx, u in enumerate(user_indices):
                seen_items = self._train_matrix[u].indices
                score_matrix[idx, seen_items] = -1e9

        # 3. Vectorised top-K via argpartition + argsort
        top_k_part = np.argpartition(-score_matrix, k_arg - 1, axis=1)[:, :k_arg]
        row_idx = np.arange(n_active)[:, None]
        top_k_ranks = np.argsort(-score_matrix[row_idx, top_k_part], axis=1)
        recs_matrix = np.take_along_axis(top_k_part, top_k_ranks, axis=1).astype(
            np.int32
        )

        # 4. Pad to shape (N_active, K) if catalog smaller than K
        if k_arg < k:
            pad_width = k - k_arg
            padding = np.full((n_active, pad_width), -1, dtype=np.int32)
            recs_matrix = np.hstack([recs_matrix, padding])

        assert recs_matrix.shape == (n_active, k), \
            f"Recommendation matrix shape misaligned: {recs_matrix.shape}"
        assert recs_matrix.min() >= -1, "Invalid padding sentinel in recommendations"
        return recs_matrix

    def score_all(self, user_indices: np.ndarray) -> np.ndarray:
        """Return real full-catalog scores of shape (len(user_indices), |I|)."""
        if self._U is None or self._V is None:
            raise RuntimeError("LightGCNAdapter has not been fitted yet. Call fit() first.")
        user_indices = np.asarray(user_indices, dtype=np.int64)
        scores = self._U[user_indices] @ self._V.T
        assert scores.shape == (len(user_indices), self.n_items)
        assert not np.isnan(scores).any(), "NaN detected in full-catalog scores"
        return scores.astype(np.float32)

    def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns cached LightGCN mean-pooled user and item embeddings aligned
        to global zero-indexed coordinate space.

        Returns:
            U_global: User embedding matrix of shape (|U|, d).
            V_global: Item embedding matrix of shape (|I|, d).
        """
        if self._U is None or self._V is None:
            raise RuntimeError(
                "LightGCNAdapter has not been fitted yet. Call fit() first."
            )
        return self._U, self._V

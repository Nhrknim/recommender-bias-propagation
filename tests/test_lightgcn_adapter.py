import unittest
import numpy as np
import scipy.sparse as sp
import torch

from src.adapters.lightgcn_adapter import LightGCNAdapter, LightGCNModel


class TestLightGCNModel(unittest.TestCase):
    """Unit tests for the core LightGCNModel torch module."""

    def setUp(self):
        """Synthetic 4-user, 6-item interaction matrix."""
        rows = [0, 0, 1, 1, 2, 3, 3]
        cols = [0, 1, 1, 2, 3, 4, 5]
        data = [1.0] * len(rows)
        self.n_users = 4
        self.n_items = 6
        self.dim = 8
        self.train_mat = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )
        self.device = torch.device("cpu")

    def test_norm_adj_shape_and_symmetry(self):
        """Normalised adjacency must be square (n+m, n+m) and have no NaN/Inf values."""
        model = LightGCNModel(self.n_users, self.n_items, self.dim, n_layers=2)
        norm_adj = model.build_norm_adj(self.train_mat, self.device)
        n_total = self.n_users + self.n_items
        self.assertEqual(norm_adj.shape, (n_total, n_total))
        vals = norm_adj.coalesce().values().numpy()
        self.assertFalse(np.isnan(vals).any(), "NaN in normalised adjacency")
        self.assertFalse(np.isinf(vals).any(), "Inf in normalised adjacency")

    def test_forward_output_shapes(self):
        """Forward pass must return (|U|, d) and (|I|, d) tensors."""
        model = LightGCNModel(self.n_users, self.n_items, self.dim, n_layers=2)
        norm_adj = model.build_norm_adj(self.train_mat, self.device)
        with torch.no_grad():
            U, V = model.forward(norm_adj)
        self.assertEqual(U.shape, (self.n_users, self.dim))
        self.assertEqual(V.shape, (self.n_items, self.dim))

    def test_bpr_loss_is_scalar_and_finite(self):
        """BPR loss must be a finite scalar (not NaN, not Inf)."""
        model = LightGCNModel(self.n_users, self.n_items, self.dim, n_layers=2)
        norm_adj = model.build_norm_adj(self.train_mat, self.device)
        u = torch.tensor([0, 1, 2], dtype=torch.long)
        pos = torch.tensor([0, 1, 3], dtype=torch.long)
        neg = torch.tensor([5, 4, 2], dtype=torch.long)
        loss = model.bpr_loss(norm_adj, u, pos, neg, reg=1e-4)
        self.assertEqual(loss.ndim, 0, "BPR loss must be a scalar")
        self.assertFalse(torch.isnan(loss).item(), "BPR loss is NaN")
        self.assertFalse(torch.isinf(loss).item(), "BPR loss is Inf")


class TestLightGCNAdapter(unittest.TestCase):
    """Integration lifecycle tests for LightGCNAdapter conforming to BaseRecommenderAdapter."""

    def setUp(self):
        """4-user, 6-item synthetic interaction matrix."""
        rows = [0, 0, 1, 1, 2, 3, 3]
        cols = [0, 1, 1, 2, 3, 4, 5]
        data = [1.0] * len(rows)
        self.n_users = 4
        self.n_items = 6
        self.dim = 8
        self.train_mat = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

    def _make_adapter(self, **kwargs) -> LightGCNAdapter:
        return LightGCNAdapter(
            dim=self.dim, n_layers=2, epochs=3, batch_size=4,
            device="cpu", **kwargs
        )

    # ── fit() ──────────────────────────────────────────────────────────────

    def test_fit_returns_self(self):
        """fit() must return the adapter instance for method chaining."""
        adapter = self._make_adapter()
        result = adapter.fit(self.train_mat, seed=42)
        self.assertIs(result, adapter)

    def test_fit_raises_on_empty_matrix(self):
        """fit() must raise ValueError on a zero-nnz interaction matrix."""
        empty = sp.csr_matrix((self.n_users, self.n_items), dtype=np.float32)
        adapter = self._make_adapter()
        with self.assertRaises(ValueError):
            adapter.fit(empty, seed=42)

    # ── get_embeddings() ────────────────────────────────────────────────────

    def test_get_embeddings_shapes(self):
        """get_embeddings() must return (|U|, d) and (|I|, d) arrays."""
        adapter = self._make_adapter().fit(self.train_mat, seed=42)
        U, V = adapter.get_embeddings()
        self.assertIsNotNone(U)
        self.assertIsNotNone(V)
        self.assertEqual(U.shape, (self.n_users, self.dim))
        self.assertEqual(V.shape, (self.n_items, self.dim))

    def test_get_embeddings_no_nan(self):
        """Returned embedding arrays must contain no NaN values."""
        adapter = self._make_adapter().fit(self.train_mat, seed=42)
        U, V = adapter.get_embeddings()
        self.assertFalse(np.isnan(U).any(), "NaN in user embeddings")
        self.assertFalse(np.isnan(V).any(), "NaN in item embeddings")

    def test_get_embeddings_raises_before_fit(self):
        """get_embeddings() must raise RuntimeError if called before fit()."""
        adapter = self._make_adapter()
        with self.assertRaises(RuntimeError):
            adapter.get_embeddings()

    # ── recommend() ─────────────────────────────────────────────────────────

    def test_recommend_shape(self):
        """recommend() must return a (N_active, K) integer array."""
        adapter = self._make_adapter().fit(self.train_mat, seed=42)
        user_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=3, remove_seen=True)
        self.assertEqual(recs.shape, (4, 3))
        self.assertEqual(recs.dtype, np.int32)

    def test_recommend_valid_item_range(self):
        """All recommended item indices must be in [0, |I|)."""
        adapter = self._make_adapter().fit(self.train_mat, seed=42)
        user_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=4, remove_seen=True)
        self.assertTrue(np.all(recs >= 0))
        self.assertTrue(np.all(recs < self.n_items))

    def test_recommend_empty_user_set(self):
        """recommend() with empty user_indices must return shape (0, K)."""
        adapter = self._make_adapter().fit(self.train_mat, seed=42)
        recs = adapter.recommend(np.array([], dtype=np.int32), k=5)
        self.assertEqual(recs.shape, (0, 5))

    def test_recommend_raises_before_fit(self):
        """recommend() must raise RuntimeError if called before fit()."""
        adapter = self._make_adapter()
        with self.assertRaises(RuntimeError):
            adapter.recommend(np.array([0, 1]), k=3)

    # ── Determinism ─────────────────────────────────────────────────────────

    def test_deterministic_reproducibility(self):
        """Two identical seeds must produce bit-identical recommendation matrices."""
        user_indices = np.array([0, 1, 2, 3], dtype=np.int32)

        adapter1 = self._make_adapter().fit(self.train_mat, seed=77)
        recs1 = adapter1.recommend(user_indices, k=3)

        adapter2 = self._make_adapter().fit(self.train_mat, seed=77)
        recs2 = adapter2.recommend(user_indices, k=3)

        np.testing.assert_array_equal(recs1, recs2)

    def test_different_seeds_produce_different_embeddings(self):
        """Different seeds must produce non-identical embeddings (stochastic init test)."""
        adapter1 = self._make_adapter().fit(self.train_mat, seed=1)
        adapter2 = self._make_adapter().fit(self.train_mat, seed=2)
        _, V1 = adapter1.get_embeddings()
        _, V2 = adapter2.get_embeddings()
        self.assertFalse(np.allclose(V1, V2), "Different seeds produced identical embeddings")

    # ── Numerical Stability ─────────────────────────────────────────────────

    def test_single_interaction_matrix(self):
        """fit() must succeed on a near-minimal matrix with a single non-zero entry."""
        minimal = sp.csr_matrix(
            ([1.0], ([0], [0])), shape=(self.n_users, self.n_items), dtype=np.float32
        )
        adapter = self._make_adapter()
        adapter.fit(minimal, seed=42)
        U, V = adapter.get_embeddings()
        self.assertEqual(U.shape, (self.n_users, self.dim))
        self.assertEqual(V.shape, (self.n_items, self.dim))

    def test_k_larger_than_catalog_pads_correctly(self):
        """When K > |I|, recommend() must return shape (N_active, K) via zero-padding."""
        adapter = self._make_adapter().fit(self.train_mat, seed=42)
        recs = adapter.recommend(np.array([0], dtype=np.int32), k=20)
        self.assertEqual(recs.shape, (1, 20))


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import scipy.sparse as sp
from src.adapters.cornac_adapters import (
    CornacMFAdapter,
    CornacPMFAdapter,
    CornacBPRAdapter,
    CornacUserKNNAdapter,
    CornacMostPopAdapter
)

class TestAdapters(unittest.TestCase):

    def setUp(self):
        """Create a synthetic interaction matrix D_0 for adapter tests."""
        # 3 users, 5 items
        # Row 0: items 0, 1
        # Row 1: items 1, 2
        # Row 2: items 3, 4
        rows = [0, 0, 1, 1, 2, 2]
        cols = [0, 1, 1, 2, 3, 4]
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.n_users = 3
        self.n_items = 5
        self.dim = 8
        self.train_mat = sp.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.n_users, self.n_items), 
            dtype=np.float32
        )

    def test_mf_adapter_lifecycle(self):
        """Test CornacMFAdapter fit, recommend shape, and embedding alignment."""
        adapter = CornacMFAdapter(dim=self.dim, max_iter=5)
        adapter.fit(self.train_mat, seed=42)

        # Test recommendation generation shape (3 active users, K=3)
        user_indices = np.array([0, 1, 2], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=3, remove_seen=True)
        
        self.assertEqual(recs.shape, (3, 3))
        self.assertTrue(np.all(recs >= 0))
        self.assertTrue(np.all(recs < self.n_items))

        # Test embedding extraction
        U_global, V_global = adapter.get_embeddings()
        self.assertIsNotNone(U_global)
        self.assertIsNotNone(V_global)
        self.assertEqual(U_global.shape, (self.n_users, self.dim))
        self.assertEqual(V_global.shape, (self.n_items, self.dim))
        self.assertFalse(np.isnan(U_global).any())
        self.assertFalse(np.isnan(V_global).any())

    def test_bpr_adapter_lifecycle(self):
        """Test CornacBPRAdapter fit, recommend shape, and embedding alignment."""
        adapter = CornacBPRAdapter(dim=self.dim, max_iter=5)
        adapter.fit(self.train_mat, seed=42)

        user_indices = np.array([0, 1, 2], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=2, remove_seen=True)

        self.assertEqual(recs.shape, (3, 2))
        U_global, V_global = adapter.get_embeddings()
        self.assertEqual(U_global.shape, (self.n_users, self.dim))
        self.assertEqual(V_global.shape, (self.n_items, self.dim))

    def test_mostpop_adapter_lifecycle(self):
        """Test CornacMostPopAdapter fit, recommend shape, and graceful non-latent embedding return."""
        adapter = CornacMostPopAdapter()
        adapter.fit(self.train_mat, seed=42)

        user_indices = np.array([0, 1, 2], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=3, remove_seen=True)

        self.assertEqual(recs.shape, (3, 3))

        # Non-latent model must return (None, None)
        U_global, V_global = adapter.get_embeddings()
        self.assertIsNone(U_global)
        self.assertIsNone(V_global)

    def test_latent_adapters_include_biases_and_mask_cold_items(self):
        """score_all must reproduce Cornac's own score (including biases)."""
        rows = [0, 0, 1, 1, 2, 2]
        cols = [0, 1, 1, 2, 3, 4]
        data = [1.0] * 6
        # Item 5 has no interactions and therefore no learned factors.
        train_mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(3, 6), dtype=np.float32
        )

        for adapter_cls in (CornacMFAdapter, CornacPMFAdapter, CornacBPRAdapter):
            adapter = adapter_cls(dim=self.dim, max_iter=10).fit(train_mat, seed=42)
            scores = adapter.score_all(np.array([0, 1, 2], dtype=np.int32))

            # Cold item must be excluded from ranking, not zero-scored.
            self.assertTrue(np.all(np.isneginf(scores[:, 5])))

            item_ids = np.asarray(adapter.model.item_ids)
            for internal_u in range(3):
                direct = adapter.model.score(internal_u)
                reconstructed = scores[internal_u, item_ids]
                # MF adds a constant offset to its score; ranking is unchanged.
                np.testing.assert_allclose(
                    reconstructed - reconstructed.mean(),
                    direct - direct.mean(),
                    atol=1e-4,
                )

    def test_deterministic_reproducibility(self):
        """Test that passing identical seeds produces identical top-K recommendations."""
        adapter1 = CornacMFAdapter(dim=self.dim, max_iter=10).fit(self.train_mat, seed=123)
        recs1 = adapter1.recommend(np.array([0, 1]), k=2)

        adapter2 = CornacMFAdapter(dim=self.dim, max_iter=10).fit(self.train_mat, seed=123)
        recs2 = adapter2.recommend(np.array([0, 1]), k=2)

        np.testing.assert_array_equal(recs1, recs2)

if __name__ == "__main__":
    unittest.main()

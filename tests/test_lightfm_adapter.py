import unittest
import numpy as np
import scipy.sparse as sp
from src.adapters.lightfm_adapter import LightFMAdapter


class TestLightFMAdapter(unittest.TestCase):

    def setUp(self):
        """Create a synthetic interaction matrix D_0 for adapter tests."""
        # 4 users, 6 items
        rows = [0, 0, 1, 1, 2, 2, 3, 3]
        cols = [0, 1, 1, 2, 3, 4, 4, 5]
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.n_users = 4
        self.n_items = 6
        self.dim = 8
        self.train_mat = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

    def test_lightfm_warp_lifecycle(self):
        """Test LightFMAdapter with WARP loss: fit, recommend shape, and embedding alignment."""
        adapter = LightFMAdapter(dim=self.dim, loss="warp", epochs=5)
        adapter.fit(self.train_mat, seed=42)

        user_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=3, remove_seen=True)

        self.assertEqual(recs.shape, (4, 3))
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

    def test_lightfm_bpr_lifecycle(self):
        """Test LightFMAdapter with BPR loss: fit, recommend shape, and embedding alignment."""
        adapter = LightFMAdapter(dim=self.dim, loss="bpr", epochs=5)
        adapter.fit(self.train_mat, seed=42)

        user_indices = np.array([0, 1, 2], dtype=np.int32)
        recs = adapter.recommend(user_indices, k=2, remove_seen=True)

        self.assertEqual(recs.shape, (3, 2))
        U_global, V_global = adapter.get_embeddings()
        self.assertEqual(U_global.shape, (self.n_users, self.dim))
        self.assertEqual(V_global.shape, (self.n_items, self.dim))

    def test_seen_items_masked(self):
        """Ensure training items are never returned when remove_seen=True."""
        adapter = LightFMAdapter(dim=self.dim, loss="warp", epochs=5)
        adapter.fit(self.train_mat, seed=42)

        # User 0 has seen items 0 and 1
        user_0 = np.array([0], dtype=np.int32)
        recs = adapter.recommend(user_0, k=4, remove_seen=True)

        seen_set = set(self.train_mat[0].indices)
        for item in recs[0]:
            self.assertNotIn(item, seen_set, f"Seen item {item} was returned in recommendations")

    def test_deterministic_reproducibility(self):
        """Test that identical seeds yield identical recommendation outputs."""
        adapter1 = LightFMAdapter(dim=self.dim, loss="warp", epochs=5).fit(self.train_mat, seed=123)
        recs1 = adapter1.recommend(np.array([0, 1, 2]), k=2)

        adapter2 = LightFMAdapter(dim=self.dim, loss="warp", epochs=5).fit(self.train_mat, seed=123)
        recs2 = adapter2.recommend(np.array([0, 1, 2]), k=2)

        np.testing.assert_array_equal(recs1, recs2)


if __name__ == "__main__":
    unittest.main()

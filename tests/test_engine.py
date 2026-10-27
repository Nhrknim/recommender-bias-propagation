import unittest
import numpy as np
import scipy.sparse as sp
from src.engine import (
    compute_position_exposure,
    compute_intrinsic_preference,
    compute_social_conformity,
    simulate_user_clicks
)
from src.engine.anchor import build_shared_anchor

class TestEngine(unittest.TestCase):

    def setUp(self):
        """Set up synthetic matrices for user interaction engine tests."""
        self.n_users = 4
        self.n_items = 6
        self.dim = 8
        self.k = 3

        np.random.seed(42)
        self.U_star = np.random.randn(self.n_users, self.dim).astype(np.float32)
        self.V_star = np.random.randn(self.n_items, self.dim).astype(np.float32)

        # Active users [0, 1, 2, 3] receiving top-3 recommendations
        self.recs = np.array([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ], dtype=np.int32)

        self.pop_counts = np.array([100, 50, 20, 10, 5, 0], dtype=np.float32)

    def test_exposure_bounds(self):
        """Test position discount decay weights for K=5."""
        p_exposed = compute_position_exposure(5)
        self.assertEqual(p_exposed.shape, (5,))
        self.assertAlmostEqual(p_exposed[0], 1.0)
        self.assertAlmostEqual(p_exposed[1], 1.0 / np.log2(3.0))
        # Monotonically decreasing
        self.assertTrue(np.all(np.diff(p_exposed) < 0))

    def test_preference_bounds_and_clipping(self):
        """Test temperature scaling, clipping [-15, 15], and sigmoid bounds [0, 1]."""
        p_pref = compute_intrinsic_preference(
            self.U_star, self.V_star, self.recs, np.arange(self.n_users), tau=1.0
        )
        self.assertEqual(p_pref.shape, (4, 3))
        self.assertTrue(np.all(p_pref >= 0.0) and np.all(p_pref <= 1.0))
        self.assertFalse(np.isnan(p_pref).any())

    def test_conformity_logarithmic_scaling(self):
        """Test sub-linear logarithmic conformity scaling."""
        p_conf = compute_social_conformity(self.pop_counts, self.recs)
        self.assertEqual(p_conf.shape, (4, 3))
        self.assertTrue(np.all(p_conf >= 0.0) and np.all(p_conf <= 1.0))
        
        # Most popular item (index 0, count 100) must achieve conformity probability 1.0
        # recs[0, 0] is item 0
        self.assertAlmostEqual(p_conf[0, 0], 1.0)
        
        # Unclicked item (index 5, count 0) must achieve conformity probability 0.0
        # recs[3, 2] is item 5
        self.assertAlmostEqual(p_conf[3, 2], 0.0)

    def test_simulator_click_generation(self):
        """Test simulator end-to-end click generation and budget constraints."""
        clicked_u, clicked_i = simulate_user_clicks(
            self.recs, 
            self.U_star, 
            self.V_star, 
            self.pop_counts, 
            np.arange(self.n_users),
            alpha=0.5, 
            tau=1.0, 
            poisson_lambda=3.0, 
            seed=42
        )

        self.assertEqual(len(clicked_u), len(clicked_i))
        if len(clicked_u) > 0:
            self.assertTrue(np.all(clicked_u >= 0) and np.all(clicked_u < self.n_users))
            self.assertTrue(np.all(clicked_i >= 0) and np.all(clicked_i < self.n_items))

    def test_simulator_deterministic_reproducibility(self):
        """Test that identical seeds yield identical click arrays."""
        users = np.arange(self.n_users)
        u1, i1 = simulate_user_clicks(self.recs, self.U_star, self.V_star, self.pop_counts, users, seed=123)
        u2, i2 = simulate_user_clicks(self.recs, self.U_star, self.V_star, self.pop_counts, users, seed=123)

        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(i1, i2)

    def test_simulator_emits_global_user_indices(self):
        """A subset of active users must emit their global IDs, not local rows."""
        users = np.array([3], dtype=np.int32)
        recs = np.array([[3, 4, 5]], dtype=np.int32)
        # Force every slot to click by making relevance and conformity maximal.
        U = np.ones((4, 4), dtype=np.float32)
        V = np.ones((6, 4), dtype=np.float32)
        u_clicks, _ = simulate_user_clicks(
            recs, U, V, np.ones(6, dtype=np.float32), users,
            alpha=0.0, tau=1.0, poisson_lambda=100.0, seed=0,
        )
        self.assertTrue(len(u_clicks) > 0)
        self.assertTrue(np.all(u_clicks == 3))

    def test_simulator_ignores_padding_sentinel(self):
        """Column -1 must never be clicked or emitted as an item index."""
        recs = np.array([[0, -1, 1]], dtype=np.int32)
        u_clicks, i_clicks = simulate_user_clicks(
            recs, self.U_star, self.V_star, self.pop_counts, np.array([0]),
            alpha=0.5, seed=7,
        )
        self.assertTrue(np.all(i_clicks >= 0))

    def test_preference_uses_global_user_rows(self):
        """Relevance for user 3 must use U_star[3], not U_star[0]."""
        users = np.array([3], dtype=np.int32)
        recs = np.array([[0]], dtype=np.int32)
        U = np.zeros((4, 2), dtype=np.float32)
        U[3] = np.array([1.0, 0.0], dtype=np.float32)
        V = np.zeros((2, 2), dtype=np.float32)
        V[0] = np.array([1.0, 0.0], dtype=np.float32)
        p = compute_intrinsic_preference(U, V, recs, users, tau=1.0)
        self.assertGreater(float(p[0, 0]), 0.5)

    def test_shared_anchor_is_deterministic_and_frozen(self):
        """The same D_0 must produce bit-identical anchors across calls."""
        rows = np.array([0, 0, 1, 1, 2, 3])
        cols = np.array([0, 1, 1, 2, 3, 4])
        base = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(4, 6),
            dtype=np.float32,
        )
        U1, V1 = build_shared_anchor(base, dim=3, seed=42)
        U2, V2 = build_shared_anchor(base, dim=3, seed=42)
        np.testing.assert_array_equal(U1, U2)
        np.testing.assert_array_equal(V1, V2)
        self.assertEqual(U1.shape, (4, 3))
        self.assertEqual(V1.shape, (6, 3))
        self.assertTrue(np.isfinite(U1).all() and np.isfinite(V1).all())

    def test_shared_anchor_rejects_empty_matrix(self):
        """An empty D_0 has no geometry to anchor."""
        empty = sp.csr_matrix((3, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_shared_anchor(empty, dim=3)


if __name__ == "__main__":
    unittest.main()

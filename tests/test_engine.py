import unittest
import numpy as np
from src.engine import (
    compute_position_exposure,
    compute_intrinsic_preference,
    compute_social_conformity,
    simulate_user_clicks
)

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
        p_pref = compute_intrinsic_preference(self.U_star, self.V_star, self.recs, tau=1.0)
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
        u1, i1 = simulate_user_clicks(self.recs, self.U_star, self.V_star, self.pop_counts, seed=123)
        u2, i2 = simulate_user_clicks(self.recs, self.U_star, self.V_star, self.pop_counts, seed=123)

        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(i1, i2)

if __name__ == "__main__":
    unittest.main()

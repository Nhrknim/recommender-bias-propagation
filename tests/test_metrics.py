import unittest
import numpy as np
import scipy.sparse as sp
from src.metrics import (
    compute_exposure_gini,
    compute_catalog_coverage,
    compute_aplt,
    partition_user_cohorts,
    compute_kl_divergence_drift,
    compute_subgroup_utility_gap,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_vectorized_auc,
    compute_reflection_corrected_procrustes,
    compute_relative_norm_shift,
    compute_mag_ang_decomposition,
    compute_effective_rank,
)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        """Set up synthetic datasets and recommendations for 4-tier metric tests."""
        self.n_users = 6
        self.n_items = 10
        self.dim = 4

        # Recommendations for 6 users, K=3
        self.recs = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [1, 5, 6],
            [7, 8, 9],
            [0, 1, 2]
        ], dtype=np.int32)

        # Long tail mask: items 5, 6, 7, 8, 9 are long tail (True)
        self.long_tail_mask = np.array([False, False, False, False, False, True, True, True, True, True])

        # Sparse interaction matrix D_0 for user cohort partitioning
        # Users 0, 1 interacted mostly with popular items (0..4)
        # Users 4, 5 interacted mostly with long tail items (5..9)
        rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        cols = [0, 1, 1, 2, 2, 3, 5, 6, 7, 8, 8, 9]
        data = [1.0] * len(rows)
        self.base_csr = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.test_csr = self.base_csr.copy()

    def test_tier1_macro_metrics(self):
        """Test Exposure Gini, Coverage@K, and APLT@K bounds."""
        gini = compute_exposure_gini(self.recs, self.n_items)
        self.assertTrue(0.0 <= gini <= 1.0)

        cov = compute_catalog_coverage(self.recs, self.n_items)
        self.assertEqual(cov, 10 / 10)  # All 10 items 0..9 were recommended

        aplt = compute_aplt(self.recs, self.long_tail_mask)
        self.assertTrue(0.0 <= aplt <= 1.0)

    def test_tier2_user_groups_metrics(self):
        """Test user cohort partitioning (Pop, Div, Niche) and KL divergence drift."""
        cohorts = partition_user_cohorts(self.base_csr, self.long_tail_mask)
        self.assertIn("pop", cohorts)
        self.assertIn("div", cohorts)
        self.assertIn("niche", cohorts)
        self.assertEqual(len(cohorts["pop"]) + len(cohorts["div"]) + len(cohorts["niche"]), self.n_users)

        # Category KL drift
        p0 = np.ones((self.n_users, 3), dtype=np.float32) / 3.0
        qt = np.ones((self.n_users, 3), dtype=np.float32) / 3.0
        qt[0] = np.array([0.8, 0.1, 0.1], dtype=np.float32)

        kl_drift = compute_kl_divergence_drift(p0, qt)
        self.assertEqual(kl_drift.shape, (self.n_users,))
        self.assertTrue(np.all(kl_drift >= 0.0))
        self.assertGreater(kl_drift[0], 0.0)

        # Subgroup utility gap
        gap = compute_subgroup_utility_gap(0.8, 0.5)
        self.assertAlmostEqual(gap, 0.3)

    def test_tier3_ranking_metrics(self):
        """Test Precision@K, Recall@K, NDCG@K, and vectorized Mann-Whitney AUC."""
        prec = compute_precision_at_k(self.recs, self.test_csr, k=3)
        rec = compute_recall_at_k(self.recs, self.test_csr, k=3)
        ndcg = compute_ndcg_at_k(self.recs, self.test_csr, k=3)

        self.assertTrue(0.0 <= prec <= 1.0)
        self.assertTrue(0.0 <= rec <= 1.0)
        self.assertTrue(0.0 <= ndcg <= 1.0)

        # Vectorized AUC
        scores = np.random.randn(self.n_users, self.n_items).astype(np.float32)
        auc = compute_vectorized_auc(scores, self.test_csr)
        self.assertTrue(0.0 <= auc <= 1.0)

    def test_tier4_geometry_metrics(self):
        """Test Reflection-Corrected Procrustes, R_norm, Mag/Ang decomposition, and S_eff."""
        np.random.seed(42)
        V_0 = np.random.randn(self.n_items, self.dim).astype(np.float32)
        
        # Apply known rotation matrix R
        theta = np.pi / 4.0
        R_true = np.eye(self.dim, dtype=np.float32)
        R_true[0, 0] = np.cos(theta)
        R_true[0, 1] = -np.sin(theta)
        R_true[1, 0] = np.sin(theta)
        R_true[1, 1] = np.cos(theta)
        
        V_t = V_0 @ R_true

        # Reflection-corrected Procrustes alignment
        V_t_aligned = compute_reflection_corrected_procrustes(V_0, V_t)
        self.assertEqual(V_t_aligned.shape, V_0.shape)
        np.testing.assert_allclose(V_t_aligned, V_0, atol=1e-4)

        # Relative norm shift
        r_norm = compute_relative_norm_shift(V_0, V_t)
        self.assertEqual(r_norm.shape, (self.n_items,))
        np.testing.assert_allclose(r_norm, 1.0, atol=1e-4)

        # Mag/Ang error decomposition
        d_mag, d_ang = compute_mag_ang_decomposition(V_0, V_t_aligned)
        self.assertEqual(d_mag.shape, (self.n_items,))
        self.assertEqual(d_ang.shape, (self.n_items,))

        # Effective rank S_eff
        s_eff = compute_effective_rank(V_0)
        self.assertTrue(1.0 <= s_eff <= float(self.dim))

if __name__ == "__main__":
    unittest.main()

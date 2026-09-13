import unittest
import numpy as np
import scipy.sparse as sp
from src.orchestrator import set_deterministic_seed, run_simulation_pipeline
from src.adapters.cornac_adapters import CornacMFAdapter, CornacMostPopAdapter

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Set up synthetic datasets for simulation pipeline tests."""
        self.n_users = 4
        self.n_items = 8
        self.dim = 4

        # 4 users, 8 items
        rows = [0, 0, 1, 1, 2, 2, 3, 3]
        cols = [0, 1, 1, 2, 3, 4, 5, 6]
        data = [1.0] * len(rows)

        self.base_csr = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        
        # Test matrix (different items)
        test_rows = [0, 1, 2, 3]
        test_cols = [2, 3, 5, 7]
        test_data = [1.0] * len(test_rows)
        self.test_csr = sp.csr_matrix((test_data, (test_rows, test_cols)), shape=(self.n_users, self.n_items), dtype=np.float32)

    def test_seed_control(self):
        """Test deterministic seed control setting."""
        set_deterministic_seed(42)
        val1 = np.random.rand()

        set_deterministic_seed(42)
        val2 = np.random.rand()

        self.assertEqual(val1, val2)

    def test_simulation_pipeline_execution(self):
        """Test end-to-end 2-seed 2-generation simulation pipeline execution."""
        results = run_simulation_pipeline(
            adapter_cls=CornacMFAdapter,
            base_csr=self.base_csr,
            test_csr=self.test_csr,
            seeds=[42, 123],
            T=2,
            k=3,
            alpha=0.5,
            tau=1.0,
            adapter_kwargs={"dim": self.dim, "max_iter": 5}
        )

        self.assertIn("aggregated", results)
        self.assertIn("seed_runs", results)
        self.assertEqual(len(results["seed_runs"]), 2)

        # Check aggregated metric dictionary structure
        aggregated = results["aggregated"]
        for key in ["gini", "coverage", "aplt", "kl_drift", "subgroup_gap", "precision", "recall", "ndcg", "auc", "r_norm", "s_eff"]:
            self.assertIn(key, aggregated)
            self.assertIn("mean", aggregated[key])
            self.assertIn("std", aggregated[key])
            self.assertEqual(len(aggregated[key]["mean"]), 2)  # 2 generations T=2

    def test_pipeline_reproducibility(self):
        """Test that running pipeline with identical parameters yields identical aggregated results."""
        results1 = run_simulation_pipeline(
            adapter_cls=CornacMostPopAdapter,
            base_csr=self.base_csr,
            test_csr=self.test_csr,
            seeds=[42],
            T=2,
            k=3
        )

        results2 = run_simulation_pipeline(
            adapter_cls=CornacMostPopAdapter,
            base_csr=self.base_csr,
            test_csr=self.test_csr,
            seeds=[42],
            T=2,
            k=3
        )

        mean_gini_1 = results1["aggregated"]["gini"]["mean"]
        mean_gini_2 = results2["aggregated"]["gini"]["mean"]
        np.testing.assert_allclose(mean_gini_1, mean_gini_2)

if __name__ == "__main__":
    unittest.main()

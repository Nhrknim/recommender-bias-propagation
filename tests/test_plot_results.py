import json
import os
import tempfile
import unittest

import numpy as np

from scripts.plot_results import (
    find_best_benchmark_json,
    plot_benchmark_trajectories,
    plot_tsne_popularity_drift,
)


class TestPlotResults(unittest.TestCase):

    @staticmethod
    def _write_benchmark(path: str, model_count: int, generations: int = 3) -> None:
        results = {}
        for i in range(model_count):
            results[f"Model {i}"] = {
                metric: {"mean": [0.5] * generations, "std": [0.1] * generations}
                for metric in ("gini", "kl_drift", "ndcg", "s_eff")
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"generations": generations, "results": results}, f)

    def test_find_best_benchmark_json_prefers_model_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path = os.path.join(tmp, "full.json")
            partial_path = os.path.join(tmp, "partial.json")
            trajectory_path = os.path.join(tmp, "lightgcn_50gen_trajectory.json")

            self._write_benchmark(full_path, model_count=5)
            self._write_benchmark(partial_path, model_count=1)
            self._write_benchmark(trajectory_path, model_count=0)

            os.utime(full_path, (1_000, 1_000))
            os.utime(partial_path, (2_000, 2_000))
            os.utime(trajectory_path, (3_000, 3_000))

            self.assertEqual(find_best_benchmark_json(tmp), full_path)

    def test_find_best_benchmark_json_raises_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_best_benchmark_json(tmp)

    def test_plot_benchmark_trajectories_writes_file(self):
        data = {
            "run_id": "test_run",
            "generations": 4,
            "k": 10,
            "alpha": 0.8,
            "seeds": [42],
            "results": {
                "Matrix Factorization (MF)": {
                    metric: {"mean": [0.1, 0.2, 0.3, 0.4], "std": [0.01] * 4}
                    for metric in ("gini", "kl_drift", "ndcg", "s_eff")
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trajectories.png")
            result = plot_benchmark_trajectories(data, out_path, sync_phase4_path=None)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.getsize(out_path) > 0)

    def test_plot_tsne_popularity_drift_writes_file(self):
        rng = np.random.default_rng(42)
        n_items = 20
        coords_0 = rng.normal(size=(n_items, 2))
        coords_final = rng.normal(size=(n_items, 2))
        base_pop = np.arange(1, n_items + 1, dtype=float)
        final_pop = base_pop[::-1].copy()

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "drift.png")
            result = plot_tsne_popularity_drift(
                model_name="Test Model",
                coords_0=coords_0,
                coords_final=coords_final,
                base_pop=base_pop,
                final_pop=final_pop,
                t_final=9,
                out_path=out_path,
                top_n=3,
                bot_n=3,
            )
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.getsize(out_path) > 0)

    def test_plot_tsne_popularity_drift_allows_zero_arrows(self):
        rng = np.random.default_rng(7)
        coords = rng.normal(size=(10, 2))
        pops = np.arange(1, 11, dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "drift_no_arrows.png")
            plot_tsne_popularity_drift(
                model_name="Test Model",
                coords_0=coords,
                coords_final=coords,
                base_pop=pops,
                final_pop=pops,
                t_final=9,
                out_path=out_path,
                top_n=0,
                bot_n=0,
            )
            self.assertTrue(os.path.getsize(out_path) > 0)


if __name__ == "__main__":
    unittest.main()

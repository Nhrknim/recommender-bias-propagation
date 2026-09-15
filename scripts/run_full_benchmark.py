import os
import sys
import time
import json
import uuid
from datetime import datetime
import numpy as np

sys.path.insert(0, ".")

from src.data.dataset import load_movielens_1m, load_movielens_genres
from src.adapters.cornac_adapters import (
    CornacMFAdapter, 
    CornacBPRAdapter, 
    CornacMostPopAdapter
)
from src.orchestrator import run_simulation_pipeline

def main():
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    print("=" * 80)
    print(f"  CLOSED-LOOP RECOMMENDER BIAS FULL BENCHMARK  [Run ID: {run_id}]")
    print("=" * 80)

    data_path = "ml-1m/ratings.dat"
    movies_path = "ml-1m/movies.dat"

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at '{data_path}'. Please check dataset path.")
        return

    # 1. Ingest dataset & genre metadata
    print(f"\n[Phase 1] Ingesting MovieLens-1M dataset and 18-genre metadata...")
    start_time = time.time()
    train_mat, test_mat, u2c, i2c = load_movielens_1m(data_path, threshold=4.0, split_ratio=0.8)
    genre_matrix = load_movielens_genres(movies_path, item2code=i2c)
    n_users, n_items = train_mat.shape
    
    print(f"[OK] Dataset & {genre_matrix.shape[1]} genres loaded in {time.time() - start_time:.2f}s")
    print(f"  * Users |U|: {n_users:,}")
    print(f"  * Items |I|: {n_items:,}")
    print(f"  * Baseline Train Interactions (D_0): {train_mat.nnz:,}")
    print(f"  * Held-Out Test Interactions (D_test): {test_mat.nnz:,}")

    # 2. Experiment configuration
    models_to_test = [
        ("Matrix Factorization (MF)", CornacMFAdapter, {"dim": 32, "max_iter": 10}),
        ("Bayesian Personalized Ranking (BPR)", CornacBPRAdapter, {"dim": 32, "max_iter": 10}),
        ("Most Popular Baseline (MostPop)", CornacMostPopAdapter, {})
    ]

    seeds = [42, 123, 456]
    T = 10
    K = 10
    alpha = 0.8  # Stronger social conformity weight for active bias dynamics

    benchmark_summary = {}
    raw_results = {}

    # 3. Execute closed-loop simulation suite
    print(f"\n[Phase 2] Executing closed-loop simulations across {len(models_to_test)} models, {len(seeds)} seeds, T={T} generations (alpha={alpha})...")

    for model_name, adapter_cls, kwargs in models_to_test:
        print(f"\n>>> Running Model: {model_name}...")
        m_start = time.time()

        sim_results = run_simulation_pipeline(
            adapter_cls=adapter_cls,
            base_csr=train_mat,
            test_csr=test_mat,
            seeds=seeds,
            T=T,
            k=K,
            alpha=alpha,
            tau=1.0,
            item_category_matrix=genre_matrix,
            adapter_kwargs=kwargs
        )

        elapsed = time.time() - m_start
        print(f"[OK] Completed {model_name} in {elapsed:.2f}s")
        benchmark_summary[model_name] = sim_results["aggregated"]
        raw_results[model_name] = sim_results

    # 4. Generate & Print Comparative Report
    print("\n" + "=" * 80)
    print(f"  FINAL COMPARATIVE DIAGNOSTIC BENCHMARK REPORT  [{run_id}]")
    print("=" * 80)

    report_lines = []
    report_lines.append("# Closed-Loop Recommender Bias Benchmark Report\n")
    report_lines.append(f"**Run ID:** `{run_id}`")
    report_lines.append(f"**Dataset:** MovieLens-1M ({n_users:,} users, {n_items:,} items, {genre_matrix.shape[1]} genres)")
    report_lines.append(f"**Configuration:** T={T} generations, K={K}, alpha={alpha}, seeds={seeds}\n")

    table_header = "| Model Architecture | " + " | ".join([f"t={t}" for t in range(T)]) + " |"
    table_divider = "| :--- | " + " | ".join([":---" for _ in range(T)]) + " |"

    report_lines.append("## Tier 1: Macro Catalog Exposure Gini ($G$)\n")
    report_lines.append(table_header)
    report_lines.append(table_divider)

    for model_name in benchmark_summary:
        gini_means = benchmark_summary[model_name]["gini"]["mean"]
        row = f"| {model_name} | " + " | ".join([f"{g:.4f}" for g in gini_means]) + " |"
        report_lines.append(row)

    report_lines.append("\n## Tier 2: User Genre Taste Drift ($D_{\\text{KL}}$)\n")
    report_lines.append(table_header)
    report_lines.append(table_divider)

    for model_name in benchmark_summary:
        kl_means = benchmark_summary[model_name]["kl_drift"]["mean"]
        row = f"| {model_name} | " + " | ".join([f"{k:.4f}" for k in kl_means]) + " |"
        report_lines.append(row)

    report_lines.append("\n## Tier 3: Recommendation Accuracy (NDCG@10)\n")
    report_lines.append(table_header)
    report_lines.append(table_divider)

    for model_name in benchmark_summary:
        ndcg_means = benchmark_summary[model_name]["ndcg"]["mean"]
        row = f"| {model_name} | " + " | ".join([f"{n:.4f}" for n in ndcg_means]) + " |"
        report_lines.append(row)

    report_lines.append("\n## Tier 4: Latent Manifold Effective Rank ($S_{\\text{eff}}$)\n")
    report_lines.append(table_header)
    report_lines.append(table_divider)

    for model_name in benchmark_summary:
        seff_means = benchmark_summary[model_name]["s_eff"]["mean"]
        row = f"| {model_name} | " + " | ".join([f"{s:.4f}" for s in seff_means]) + " |"
        report_lines.append(row)

    report_content = "\n".join(report_lines)
    print("\n" + report_content)

    # 5. Dual Save (Unique Run ID Markdown + JSON, plus Latest Pointer)
    os.makedirs("docs/benchmarks", exist_ok=True)
    
    unique_md = f"docs/benchmarks/{run_id}.md"
    unique_json = f"docs/benchmarks/{run_id}.json"
    latest_md = "docs/benchmarks/benchmark_results.md"


    with open(unique_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    with open(unique_json, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset": "MovieLens-1M",
            "n_users": n_users,
            "n_items": n_items,
            "generations": T,
            "k": K,
            "alpha": alpha,
            "seeds": seeds,
            "results": benchmark_summary
        }, f, indent=2)

    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[OK] Saved run markdown artifact: '{unique_md}'")
    print(f"[OK] Saved run JSON metrics data:   '{unique_json}'")
    print(f"[OK] Updated latest run pointer:    '{latest_md}'")

if __name__ == "__main__":
    main()


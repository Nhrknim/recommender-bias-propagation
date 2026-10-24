"""
plot_benchmark_from_json.py
───────────────────────────
Reads the most recent benchmark JSON produced by run_full_benchmark.py and
generates benchmark_trajectories_lightgcn.png from REAL measured data.

Usage:
    python scripts/plot_benchmark_from_json.py
    python scripts/plot_benchmark_from_json.py --json docs/benchmarks/<run_id>.json
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 150})

MODEL_STYLE = {
    "Matrix Factorization (MF)":               {"color": "#2b5c8f", "marker": "o", "ls": "-",  "lw": 2.0},
    "Probabilistic Matrix Factorization (PMF)": {"color": "#17becf", "marker": "v", "ls": "-.", "lw": 2.0},
    "Bayesian Personalized Ranking (BPR)":      {"color": "#d95f02", "marker": "s", "ls": "-",  "lw": 2.0},
    "Most Popular Baseline (MostPop)":          {"color": "#7570b3", "marker": "X", "ls": "--", "lw": 2.0},
    "LightGCN (Graph CF)":                      {"color": "#2ca02c", "marker": "D", "ls": "-",  "lw": 2.6},
}
MODEL_ORDER = list(MODEL_STYLE.keys())
OUTPUT_PATH = "docs/images/phase_5_lightgcn/benchmark_trajectories_lightgcn.png"


def find_latest_json(benchmark_dir):
    files = [f for f in glob.glob(os.path.join(benchmark_dir, "*.json")) if not f.endswith(".gitkeep")]
    if not files:
        raise FileNotFoundError(
            f"No benchmark JSON files found in '{benchmark_dir}'.\n"
            "Run  python scripts/run_full_benchmark.py  first."
        )
    return max(files, key=os.path.getmtime)


def plot(data, out_path):
    results    = data["results"]
    T          = data.get("generations", 10)
    generations = np.arange(T)

    ordered = [m for m in MODEL_ORDER if m in results] + [m for m in results if m not in MODEL_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    run_id = data.get("run_id", "unknown")
    seeds  = data.get("seeds", [])
    alpha  = data.get("alpha", "?")
    fig.suptitle(
        f"4-Tier Closed-Loop Bias Trajectories — Real Simulation Data\n"
        f"Run: {run_id}  |  seeds={seeds}  |  α={alpha}  |  T={T}  |  K={data.get('k', 10)}",
        fontsize=13, fontweight="bold", y=1.0,
    )

    panels = [
        (axes[0, 0], "gini",     "Tier 1: Catalog Exposure Gini (G) — Inequality", "Gini Coefficient"),
        (axes[0, 1], "kl_drift", "Tier 2: User Genre Taste Drift (D_KL)",          "Mean KL Divergence"),
        (axes[1, 0], "ndcg",     "Tier 3: Accuracy (NDCG@10)",                     "NDCG@10"),
        (axes[1, 1], "s_eff",    "Tier 4: Latent Manifold Effective Rank (S_eff)", "Effective Rank (Dimensions)"),
    ]

    for ax, metric_key, title, ylabel in panels:
        for model_name in ordered:
            metric = results[model_name].get(metric_key)
            if metric is None:
                continue
            mean_vals = np.array(metric["mean"])
            std_vals  = np.array(metric.get("std", [0.0] * len(mean_vals)))
            style = MODEL_STYLE.get(model_name, {"color": "gray", "marker": "o", "ls": "-", "lw": 1.5})

            ax.plot(
                generations, mean_vals, label=model_name,
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], lw=style["lw"], markersize=6,
            )
            ax.fill_between(
                generations,
                mean_vals - std_vals, mean_vals + std_vals,
                alpha=0.12, color=style["color"],
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Generation (t)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(5, len(ordered)),
        bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=11,
        facecolor="white", framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved real-data benchmark plot → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--out",  default=OUTPUT_PATH)
    args = parser.parse_args()

    json_path = args.json or find_latest_json("docs/benchmarks")
    print(f"[INFO] Reading: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Models: {list(data['results'].keys())}")
    plot(data, args.out)


if __name__ == "__main__":
    main()

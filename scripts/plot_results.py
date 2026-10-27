"""
plot_results.py
───────────────
Single entry point for the project's result plots.

Subcommands:
    trajectories  Render the 4-tier benchmark dashboard from a benchmark JSON.
    drift         Re-run the MF/PMF/BPR closed loop and render per-model
                  t-SNE popularity-drift plots (t=0 vs t=T-1).
    all           Run trajectories, then drift.

Running with no subcommand (or only legacy flags such as --json) defaults to
``trajectories``.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py trajectories --json docs/benchmarks/<run_id>.json
    python scripts/plot_results.py drift
    python scripts/plot_results.py drift --models mf bpr --generations 20
    python scripts/plot_results.py all
"""
import argparse
import glob
import json
import os
import shutil
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 150})


def _repo_path(*parts: str) -> str:
    """Build an absolute path rooted at the repository."""
    return os.path.join(REPO_ROOT, *parts)


MODEL_STYLE = {
    "Matrix Factorization (MF)":               {"color": "#2b5c8f", "marker": "o", "ls": "-",  "lw": 2.0},
    "Probabilistic Matrix Factorization (PMF)": {"color": "#17becf", "marker": "v", "ls": "-.", "lw": 2.0},
    "Bayesian Personalized Ranking (BPR)":      {"color": "#d95f02", "marker": "s", "ls": "-",  "lw": 2.0},
    "Most Popular Baseline (MostPop)":          {"color": "#7570b3", "marker": "X", "ls": "--", "lw": 2.0},
    "LightGCN (Graph CF)":                      {"color": "#2ca02c", "marker": "D", "ls": "-",  "lw": 2.6},
}
MODEL_ORDER = list(MODEL_STYLE.keys())

DEFAULT_BENCHMARK_DIR = _repo_path("docs", "benchmarks")
DEFAULT_TRAJECTORY_OUTPUT = _repo_path(
    "docs", "images", "phase_5_lightgcn", "benchmark_trajectories_lightgcn.png"
)
DEFAULT_PHASE4_TRAJECTORY_OUTPUT = _repo_path(
    "docs", "images", "phase_4_tsne_visualization", "benchmark_trajectories.png"
)

DEFAULT_DATASET_PATH = _repo_path("ml-1m", "ratings.dat")
DEFAULT_DRIFT_OUTPUT_ROOT = _repo_path("docs", "images", "phase_4_tsne_visualization")
DRIFT_MODEL_KEYS = ("mf", "pmf", "bpr")


def find_best_benchmark_json(benchmark_dir: str = DEFAULT_BENCHMARK_DIR) -> str:
    """Return the benchmark JSON with the most model results, then newest mtime.

    Excludes the standalone LightGCN trajectory log, which is not a full
    benchmark payload.
    """
    files = [
        f for f in glob.glob(os.path.join(benchmark_dir, "*.json"))
        if not f.endswith(".gitkeep") and "lightgcn_50gen" not in os.path.basename(f)
    ]
    if not files:
        raise FileNotFoundError(
            f"No benchmark JSON files found in '{benchmark_dir}'.\n"
            "Run  python scripts/run_full_benchmark.py  first."
        )

    def _score(path: str) -> tuple[int, float]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return len(data.get("results", {})), os.path.getmtime(path)
        except (OSError, json.JSONDecodeError):
            return 0, 0.0

    return max(files, key=_score)


def plot_benchmark_trajectories(
    data: dict,
    out_path: str,
    sync_phase4_path: str | None = DEFAULT_PHASE4_TRAJECTORY_OUTPUT,
) -> str:
    """Render the 2x2 four-tier benchmark trajectory dashboard.

    Args:
        data: Benchmark payload containing ``results``, ``generations``, etc.
        out_path: Destination PNG path.
        sync_phase4_path: Optional copy destination used by the Phase 4 docs.
            Pass ``None`` to disable the synchronized copy.

    Returns:
        The path written to ``out_path``.
    """
    results = data["results"]
    T = data.get("generations", 10)
    generations = np.arange(T)

    ordered = [m for m in MODEL_ORDER if m in results] + [m for m in results if m not in MODEL_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    run_id = data.get("run_id", "unknown")
    seeds = data.get("seeds", [])
    alpha = data.get("alpha", "?")
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
            std_vals = np.array(metric.get("std", [0.0] * len(mean_vals)))
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
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved benchmark trajectory plot → {out_path}")

    if sync_phase4_path and os.path.abspath(out_path) != os.path.abspath(sync_phase4_path):
        os.makedirs(os.path.dirname(os.path.abspath(sync_phase4_path)), exist_ok=True)
        shutil.copyfile(out_path, sync_phase4_path)
        print(f"[OK] Synced copy → {sync_phase4_path}")

    return out_path


def plot_tsne_popularity_drift(
    model_name: str,
    coords_0: np.ndarray,        # Shape: (|I|, 2)
    coords_final: np.ndarray,    # Shape: (|I|, 2)
    base_pop: np.ndarray,        # Shape: (|I|,)
    final_pop: np.ndarray,       # Shape: (|I|,)
    t_final: int,
    out_path: str,
    top_n: int = 30,
    bot_n: int = 30,
    base_title: str = "Generation t=0 (Baseline Manifold)",
    final_title: str | None = None,
    colorbar_label: str = "Log10(Interactions)",
) -> str:
    """Render joint t-SNE item manifolds with top/bottom-popularity drift arrows.

    Args:
        model_name: Display name used in the figure title.
        coords_0: Joint t-SNE coordinates at t=0.
        coords_final: Joint t-SNE coordinates at t=T-1 (same projection space).
        base_pop: Baseline interaction counts per item.
        final_pop: Final interaction counts per item.
        t_final: Final generation index used in titles and legend.
        out_path: Destination PNG path.
        top_n: Number of most-popular items to draw red arrows for.
        bot_n: Number of least-popular items to draw blue arrows for.
        base_title: Left-panel title.
        final_title: Right-panel title. Defaults to a generic post-drift title.
        colorbar_label: Label for the popularity colorbar.

    Returns:
        The path written to ``out_path``.
    """
    if final_title is None:
        final_title = f"Generation t={t_final} (Post Closed-Loop Drift)"

    log_base_pop = np.log10(np.maximum(base_pop, 1.0))
    log_final_pop = np.log10(np.maximum(final_pop, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle(
        f"{model_name} Latent Manifold Drift: t=0 vs t={t_final} — "
        f"Top-{top_n} vs Bottom-{bot_n} Popular Items",
        fontsize=14, fontweight="bold",
    )

    axes[0].scatter(coords_0[:, 0], coords_0[:, 1], c=log_base_pop,
                    cmap="viridis", s=20, alpha=0.75)
    axes[0].set_title(base_title)
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")

    sc1 = axes[1].scatter(coords_final[:, 0], coords_final[:, 1], c=log_final_pop,
                          cmap="viridis", s=20, alpha=0.75)

    if top_n > 0:
        for idx in np.argsort(final_pop)[-top_n:]:
            axes[1].annotate("", xy=(coords_final[idx, 0], coords_final[idx, 1]),
                             xytext=(coords_0[idx, 0], coords_0[idx, 1]),
                             arrowprops=dict(arrowstyle="->", color="#e41a1c",
                                             lw=1.4, alpha=0.85, shrinkA=0, shrinkB=0))

    if bot_n > 0:
        for idx in np.argsort(final_pop)[:bot_n]:
            axes[1].annotate("", xy=(coords_final[idx, 0], coords_final[idx, 1]),
                             xytext=(coords_0[idx, 0], coords_0[idx, 1]),
                             arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                             lw=1.4, alpha=0.85, shrinkA=0, shrinkB=0))

    axes[1].legend(
        handles=[
            mpatches.Patch(color="#e41a1c", label=f"Top-{top_n} most popular (short-tail)"),
            mpatches.Patch(color="#1f77b4", label=f"Bottom-{bot_n} least popular (long-tail)"),
        ],
        loc="upper left", fontsize=9, frameon=True, framealpha=0.9,
        title=f"Drift Arrows (t=0 → t={t_final})",
    )
    axes[1].set_title(final_title)
    axes[1].set_xlabel("t-SNE Dimension 1")

    # Dedicated colorbar axis at the far right — never overlaps the subplots
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.75])
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label(colorbar_label)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved → {out_path}")
    return out_path


def _load_drift_adapters() -> dict[str, tuple[str, type, dict]]:
    """Import Cornac adapters lazily so trajectory-only runs stay lightweight."""
    from src.adapters.cornac_adapters import (
        CornacBPRAdapter,
        CornacMFAdapter,
        CornacPMFAdapter,
    )

    return {
        "mf": ("Matrix Factorization (MF)", CornacMFAdapter, {"dim": 32, "max_iter": 10}),
        "pmf": ("Probabilistic Matrix Factorization (PMF)", CornacPMFAdapter, {"dim": 32, "max_iter": 10}),
        "bpr": ("Bayesian Personalized Ranking (BPR)", CornacBPRAdapter, {"dim": 32, "max_iter": 10}),
    }


def regenerate_phase4_drift_plots(
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_root: str = DEFAULT_DRIFT_OUTPUT_ROOT,
    models: Sequence[str] = DRIFT_MODEL_KEYS,
    generations: int = 10,
    k: int = 10,
    alpha: float = 0.8,
    seed: int = 42,
    top_n: int = 30,
    bot_n: int = 30,
) -> list[str]:
    """Run the closed-loop simulation and regenerate per-model drift plots.

    Args:
        dataset_path: Path to the MovieLens-1M ratings file.
        output_root: Root directory containing per-model output subfolders.
        models: Model keys to run (``mf``, ``pmf``, ``bpr``).
        generations: Number of closed-loop generations T.
        k: Number of recommendations per user per generation.
        alpha: Social-conformity weight used by the click simulator.
        seed: Base random seed.
        top_n: Number of most-popular items to draw red arrows for.
        bot_n: Number of least-popular items to draw blue arrows for.

    Returns:
        Paths of the generated PNG files.
    """
    from sklearn.manifold import TSNE

    from src.data.buffer import merge_and_deduplicate
    from src.data.dataset import load_movielens_1m
    from src.engine import simulate_user_clicks

    adapters = _load_drift_adapters()
    unknown = [model for model in models if model not in adapters]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Choose from {sorted(adapters)}.")
    if generations < 1:
        raise ValueError("generations must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if top_n < 0 or bot_n < 0:
        raise ValueError("top_n and bot_n must be >= 0")

    print(f"Loading MovieLens-1M from {dataset_path}...")
    train_mat, _, _, _ = load_movielens_1m(dataset_path, threshold=4.0, split_ratio=0.8)
    n_users, n_items = train_mat.shape
    base_pop = np.asarray(train_mat.sum(axis=0)).flatten()
    print(f"  Users: {n_users:,}  Items: {n_items:,}  nnz: {train_mat.nnz:,}")

    saved_paths: list[str] = []

    for model_key in models:
        model_name, adapter_cls, adapter_kwargs = adapters[model_key]
        out_dir = os.path.join(output_root, model_key)
        out_path = os.path.join(out_dir, "tsne_popularity_drift.png")

        print(f"\n{'=' * 68}\n  {model_name}\n{'=' * 68}")

        adapter = adapter_cls(**adapter_kwargs)
        adapter.fit(train_mat, seed=seed)
        U_0, V_0 = adapter.get_embeddings()
        if U_0 is None or V_0 is None:
            raise ValueError(f"{model_name} did not return latent embeddings.")
        U_star, V_star = U_0.copy(), V_0.copy()

        D_t = train_mat.copy()
        active_users = np.arange(n_users, dtype=np.int32)

        for gen in range(generations):
            recs = adapter.recommend(active_users, k=k, remove_seen=True)
            current_pop = np.asarray(D_t.sum(axis=0)).flatten()
            print(f"  gen {gen:02d}/{generations - 1}")
            if gen < generations - 1:
                u_clicks, i_clicks = simulate_user_clicks(
                    recs=recs, U_star=U_star, V_star=V_star,
                    pop_counts=current_pop, alpha=alpha, tau=1.0, seed=seed + gen,
                )
                D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))
                adapter = adapter_cls(**adapter_kwargs)
                adapter.fit(D_t, seed=seed + gen + 1)

        _, V_final = adapter.get_embeddings()
        if V_final is None:
            raise ValueError(f"{model_name} did not return final latent embeddings.")
        final_pop = np.asarray(D_t.sum(axis=0)).flatten()

        print(f"  Running joint t-SNE on {n_items * 2:,} points...")
        V_joint = np.vstack([V_0, V_final])
        tsne = TSNE(
            n_components=2, perplexity=30.0, random_state=42,
            init="pca", learning_rate="auto", max_iter=1000,
        )
        coords_joint = tsne.fit_transform(V_joint)
        coords_0 = coords_joint[:n_items]
        coords_final = coords_joint[n_items:]

        plot_tsne_popularity_drift(
            model_name=model_name,
            coords_0=coords_0,
            coords_final=coords_final,
            base_pop=base_pop,
            final_pop=final_pop,
            t_final=generations - 1,
            out_path=out_path,
            top_n=top_n,
            bot_n=bot_n,
        )
        saved_paths.append(out_path)

    print(f"\nAll {len(saved_paths)} Phase 4 tsne_popularity_drift.png files regenerated.")
    return saved_paths


def _add_trajectory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", default=None, help="Benchmark JSON path (default: best available).")
    parser.add_argument("--out", default=DEFAULT_TRAJECTORY_OUTPUT, help="Trajectory dashboard output path.")
    parser.add_argument(
        "--no-sync", action="store_true",
        help="Do not copy the dashboard into docs/images/phase_4_tsne_visualization/.",
    )


def _add_drift_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="MovieLens-1M ratings.dat path.")
    parser.add_argument("--output-root", default=DEFAULT_DRIFT_OUTPUT_ROOT, help="Root for per-model output folders.")
    parser.add_argument(
        "--models", nargs="+", choices=DRIFT_MODEL_KEYS, default=list(DRIFT_MODEL_KEYS),
        help="Models to simulate and plot.",
    )
    parser.add_argument("--generations", type=int, default=10, help="Number of closed-loop generations T.")
    parser.add_argument("--k", type=int, default=10, help="Recommendations per user per generation.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Social-conformity weight.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top-popularity drift arrows.")
    parser.add_argument("--bot-n", type=int, default=30, help="Number of bottom-popularity drift arrows.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark trajectory dashboards and t-SNE popularity-drift plots.",
    )
    subparsers = parser.add_subparsers(dest="command")

    trajectory_parser = subparsers.add_parser(
        "trajectories", help="Render the 4-tier benchmark dashboard from a benchmark JSON.",
    )
    _add_trajectory_args(trajectory_parser)

    drift_parser = subparsers.add_parser(
        "drift", help="Re-run the closed loop and render per-model t-SNE drift plots.",
    )
    _add_drift_args(drift_parser)

    all_parser = subparsers.add_parser("all", help="Run trajectories, then drift.")
    _add_trajectory_args(all_parser)
    _add_drift_args(all_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    # Preserve the old flag-only invocation: plot_results.py --json <path>
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv.insert(0, "trajectories")

    args = parser.parse_args(argv)

    if args.command in ("trajectories", "all"):
        json_path = args.json or find_best_benchmark_json()
        print(f"[INFO] Reading: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Models: {list(data['results'].keys())}")
        plot_benchmark_trajectories(
            data,
            args.out,
            sync_phase4_path=None if args.no_sync else DEFAULT_PHASE4_TRAJECTORY_OUTPUT,
        )

    if args.command in ("drift", "all"):
        regenerate_phase4_drift_plots(
            dataset_path=args.dataset,
            output_root=args.output_root,
            models=args.models,
            generations=args.generations,
            k=args.k,
            alpha=args.alpha,
            seed=args.seed,
            top_n=args.top_n,
            bot_n=args.bot_n,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

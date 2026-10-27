import os
import sys
import io
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

# Ensure root directory is on python path
sys.path.insert(0, ".")

from src.data.dataset import load_movielens_1m
from src.adapters.lightfm_adapter import LightFMAdapter
from src.data.buffer import merge_and_deduplicate
from src.engine.anchor import build_shared_anchor
from src.engine.simulator import simulate_user_clicks
from src.metrics.geometry import (
    compute_reflection_corrected_procrustes,
    compute_drift_metrics_32d,
)
from scripts.plot_results import find_lightfm_benchmark_json


def main():
    print("=" * 80)
    print("  LIGHTFM (WARP) 32-D QUANTITATIVE EMBEDDING DRIFT TRAJECTORY SIMULATION")
    print("=" * 80)

    output_dir = "docs/images/phase_6_lightfm"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "quantitative_embedding_drift_trajectory.gif")
    json_path = os.path.join("docs/benchmarks", "lightfm_quantitative_drift_metrics.json")

    # 1. Ingest dataset
    print("\n[Phase 1] Ingesting MovieLens-1M dataset...")
    train_mat, test_mat, u2c, i2c = load_movielens_1m("ml-1m/ratings.dat", threshold=4.0, split_ratio=0.8)
    n_users, n_items = train_mat.shape

    base_pop = np.asarray(train_mat.sum(axis=0)).flatten()
    log_pop = np.log1p(base_pop)
    top30_items = np.argsort(-base_pop)[:30]

    # Partition catalog into popularity tiers matching repository standard:
    # Short-tail: >= 250 ratings
    # Mid-tail: 50 - 249 ratings
    # Long-tail: < 50 ratings
    short_mask = base_pop >= 250
    mid_mask = (base_pop >= 50) & (base_pop < 250)
    long_mask = base_pop < 50

    print(f"Catalog Tail Cohorts:")
    print(f"  * Short-tail (>= 250): {np.sum(short_mask):,} movies ({np.sum(base_pop[short_mask]):,} interactions)")
    print(f"  * Mid-tail (50-249):   {np.sum(mid_mask):,} movies ({np.sum(base_pop[mid_mask]):,} interactions)")
    print(f"  * Long-tail (< 50):    {np.sum(long_mask):,} movies ({np.sum(base_pop[long_mask]):,} interactions)")

    T = int(json.load(open(find_lightfm_benchmark_json(), "r", encoding="utf-8"))["generations"])
    alpha = 0.8
    k = 10
    active_users = np.arange(n_users, dtype=np.int32)
    D_t = train_mat.copy()

    # Shared, model-independent click anchor (frozen across all generations).
    U_anchor, V_anchor = build_shared_anchor(train_mat, dim=32, seed=42)

    # 2. Run simulation and compute 32-D quantitative drift metrics
    print(f"\n[Phase 2] Simulating {T} generations of closed-loop LightFM (WARP)...")
    item_embeddings_history = []
    
    # Storage for quantitative drift metrics across time
    # Each list holds values for t=0..T-1
    cohort_metrics = {
        "short": {"euc": [], "cos": [], "norm": []},
        "mid":   {"euc": [], "cos": [], "norm": []},
        "long":  {"euc": [], "cos": [], "norm": []},
    }

    # Initial baseline fit t=0
    adapter = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
    adapter.fit(D_t, seed=42)
    _, V_star = adapter.get_embeddings()
    item_embeddings_history.append(V_star.copy())

    # Baseline metrics at t=0
    d_euc_0, d_cos_0, norms_0, _ = compute_drift_metrics_32d(V_star, V_star)
    for mask, key in [(short_mask, "short"), (mid_mask, "mid"), (long_mask, "long")]:
        cohort_metrics[key]["euc"].append(float(np.mean(d_euc_0[mask])))
        cohort_metrics[key]["cos"].append(float(np.mean(d_cos_0[mask])))
        cohort_metrics[key]["norm"].append(float(np.mean(norms_0[mask])))

    for t in range(1, T):
        print(f"  * Simulating Generation t={t}...")
        recs = adapter.recommend(active_users, k=k, remove_seen=True)
        pop_counts = np.asarray(D_t.sum(axis=0)).flatten()
        u_clicks, i_clicks = simulate_user_clicks(
            recs=recs,
            U_star=U_anchor,
            V_star=V_anchor,
            pop_counts=pop_counts,
            user_indices=active_users,
            alpha=alpha,
            tau=1.0,
            seed=42 + t
        )
        D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))
        adapter = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
        adapter.fit(D_t, seed=42 + t)
        _, V_t = adapter.get_embeddings()

        # Procrustes align V_t to baseline V_star
        V_t_aligned = compute_reflection_corrected_procrustes(V_star, V_t)
        item_embeddings_history.append(V_t_aligned)

        # Compute full 32-D quantitative drift metrics
        d_euc, d_cos, norms_t, _ = compute_drift_metrics_32d(V_star, V_t_aligned)
        for mask, key in [(short_mask, "short"), (mid_mask, "mid"), (long_mask, "long")]:
            cohort_metrics[key]["euc"].append(float(np.mean(d_euc[mask])))
            cohort_metrics[key]["cos"].append(float(np.mean(d_cos[mask])))
            cohort_metrics[key]["norm"].append(float(np.mean(norms_t[mask])))

    # Save quantitative metrics to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "LightFM (WARP)",
            "dataset": "MovieLens-1M",
            "run_config": {
                "generations": T,
                "epochs": 5,
                "dim": 32,
                "alpha": alpha,
                "tau": 1.0,
                "k": k,
                "anchor_method": "truncated_svd",
                "anchor_dim": int(U_anchor.shape[1]),
                "seed": 42,
            },
            "generations": T,
            "cohort_metrics": cohort_metrics,
        }, f, indent=2)
    print(f"[OK] Quantitative drift metrics saved to '{json_path}'")

    # 3. Project to shared 2D coordinate space for visual panel
    print("\n[Phase 3] Computing 2D PCA projection for visual panel...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(V_star)
    projections_2d = [pca.transform(V) for V in item_embeddings_history]

    all_x = np.concatenate([p[:, 0] for p in projections_2d])
    all_y = np.concatenate([p[:, 1] for p in projections_2d])
    x_min, x_max = np.min(all_x) - 0.5, np.max(all_x) + 0.5
    y_min, y_max = np.min(all_y) - 0.5, np.max(all_y) + 0.5

    # 4. Render animation frames
    print(f"\n[Phase 4] Rendering {T} animation frames...")
    frames = []
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "short": "#D95F02",  # Orange
        "mid":   "#7570B3",  # Purple
        "long":  "#1B9E77",  # Green
    }
    labels = {
        "short": "Short-tail (>= 250)",
        "mid":   "Mid-tail (50-249)",
        "long":  "Long-tail (< 50)",
    }

    all_t = list(range(T))
    tick_step = max(1, T // 10)
    ticks = list(range(0, T, tick_step))
    if ticks[-1] != T - 1:
        ticks.append(T - 1)
    # Replace dense per-generation ticks in the quantitative panels.

    # Pre-calculate global axis limits for quantitative subplots
    max_euc = max(max(cohort_metrics[k]["euc"]) for k in cohort_metrics) * 1.15
    max_cos = max(max(cohort_metrics[k]["cos"]) for k in cohort_metrics) * 1.25
    min_norm = min(min(cohort_metrics[k]["norm"]) for k in cohort_metrics) * 0.95
    max_norm = max(max(cohort_metrics[k]["norm"]) for k in cohort_metrics) * 1.08

    for t in range(T):
        fig = plt.figure(figsize=(16, 9), dpi=120)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.35, wspace=0.25)

        # Left Panel: 2D Item Manifold with Popularity & Movement Trails
        ax_map = fig.add_subplot(gs[:, 0])
        current_proj = projections_2d[t]

        sc = ax_map.scatter(
            current_proj[:, 0],
            current_proj[:, 1],
            c=log_pop,
            cmap="viridis",
            s=16,
            alpha=0.6,
            edgecolors="none"
        )

        if t > 0:
            for idx in top30_items:
                path_x = [projections_2d[step][idx, 0] for step in range(t + 1)]
                path_y = [projections_2d[step][idx, 1] for step in range(t + 1)]
                ax_map.plot(path_x, path_y, color="red", lw=1.2, alpha=0.5)
                ax_map.annotate(
                    "",
                    xy=(current_proj[idx, 0], current_proj[idx, 1]),
                    xytext=(path_x[-2], path_y[-2]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.9)
                )

        ax_map.scatter(
            current_proj[top30_items, 0],
            current_proj[top30_items, 1],
            color="red",
            s=35,
            edgecolors="black",
            linewidth=0.5,
            label="Top-30 Blockbusters"
        )

        ax_map.set_xlim(x_min, x_max)
        ax_map.set_ylim(y_min, y_max)
        ax_map.set_title(f"LightFM (WARP): 2D Latent Manifold Drift [Gen t = {t}]", fontsize=13, fontweight="bold")
        ax_map.set_xlabel("Latent Principal Component 1", fontsize=10)
        ax_map.set_ylabel("Latent Principal Component 2", fontsize=10)
        ax_map.legend(loc="upper right", frameon=True, fontsize=9)
        cbar = fig.colorbar(sc, ax=ax_map, orientation="horizontal", pad=0.08, fraction=0.045)
        cbar.set_label("Item Log-Popularity (Yellow = Head Hits, Purple = Tail)", fontsize=9)

        # Right Panel 1 (Top): 32-D Mean Euclidean Drift
        ax_euc = fig.add_subplot(gs[0, 1])
        for k in ["short", "mid", "long"]:
            ax_euc.plot(all_t[:t + 1], cohort_metrics[k]["euc"][:t + 1], color=colors[k], lw=2.2, marker="o", markersize=4, label=labels[k])
            ax_euc.scatter(t, cohort_metrics[k]["euc"][t], color=colors[k], s=40, zorder=5)
        ax_euc.set_xlim(-0.3, T - 0.7)
        ax_euc.set_ylim(-0.02, max_euc)
        ax_euc.set_ylabel(r"Mean $d_{\mathrm{Euc}}$", fontsize=10, fontweight="bold")
        ax_euc.set_title(r"32-D Euclidean Drift: $\|v_i^{(t)\mathrm{aligned}} - v_i^{(0)}\|_2$", fontsize=11, fontweight="bold")
        ax_euc.legend(loc="upper left", frameon=True, fontsize=8)
        ax_euc.set_xticks(ticks)

        # Right Panel 2 (Middle): 32-D Mean Cosine Drift
        ax_cos = fig.add_subplot(gs[1, 1])
        for k in ["short", "mid", "long"]:
            ax_cos.plot(all_t[:t + 1], cohort_metrics[k]["cos"][:t + 1], color=colors[k], lw=2.2, marker="s", markersize=4, label=labels[k])
            ax_cos.scatter(t, cohort_metrics[k]["cos"][t], color=colors[k], s=40, zorder=5)
        ax_cos.set_xlim(-0.3, T - 0.7)
        ax_cos.set_ylim(-0.005, max_cos)
        ax_cos.set_ylabel(r"Mean $d_{\mathrm{Cos}}$", fontsize=10, fontweight="bold")
        ax_cos.set_title(r"32-D Cosine Drift: $1 - \cos(v_i^{(t)\mathrm{aligned}}, v_i^{(0)})$", fontsize=11, fontweight="bold")
        ax_cos.legend(loc="upper left", frameon=True, fontsize=8)
        ax_cos.set_xticks(ticks)

        # Right Panel 3 (Bottom): 32-D Mean Vector Norm
        ax_norm = fig.add_subplot(gs[2, 1])
        for k in ["short", "mid", "long"]:
            ax_norm.plot(all_t[:t + 1], cohort_metrics[k]["norm"][:t + 1], color=colors[k], lw=2.2, marker="^", markersize=4, label=labels[k])
            ax_norm.scatter(t, cohort_metrics[k]["norm"][t], color=colors[k], s=40, zorder=5)
        ax_norm.set_xlim(-0.3, T - 0.7)
        ax_norm.set_ylim(min_norm, max_norm)
        ax_norm.set_ylabel(r"Mean $\|v_i\|_2$", fontsize=10, fontweight="bold")
        ax_norm.set_title(r"32-D Vector Norm Magnitude: $\|v_i^{(t)}\|_2$", fontsize=11, fontweight="bold")
        ax_norm.set_xlabel("Generation (t)", fontsize=10)
        ax_norm.legend(loc="upper left", frameon=True, fontsize=8)
        ax_norm.set_xticks(ticks)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    print(f"\n[Phase 5] Compiling and saving GIF to '{gif_path}'...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=700,
        loop=0
    )

    size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"[SUCCESS] Quantitative drift GIF created successfully: {size_mb:.2f} MB at '{gif_path}'")


if __name__ == "__main__":
    main()

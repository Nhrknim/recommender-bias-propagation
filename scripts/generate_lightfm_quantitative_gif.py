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
from sklearn.manifold import TSNE

# Ensure root directory is on python path
sys.path.insert(0, ".")

from src.data.dataset import load_movielens_1m
from src.adapters.lightfm_adapter import LightFMAdapter
from src.data.buffer import merge_and_deduplicate
from src.engine.simulator import simulate_user_clicks
from src.metrics.geometry import (
    compute_reflection_corrected_procrustes,
    compute_drift_metrics_32d,
)


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
    bottom30_items = np.argsort(base_pop)[:30]

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

    T = 50
    active_users = np.arange(n_users, dtype=np.int32)
    D_t = train_mat.copy()

    # 2. Run simulation and compute 32-D quantitative drift metrics
    print(f"\n[Phase 2] Simulating {T} generations of closed-loop LightFM (WARP)...")
    emb_cache_path = f"docs/benchmarks/lightfm_warp_embeddings_{T}t.npy"
    if os.path.exists(emb_cache_path) and os.path.exists(json_path):
        print(f"  * Loading cached embeddings from '{emb_cache_path}'...")
        item_embeddings_history = list(np.load(emb_cache_path))
        V_star = item_embeddings_history[0]
        print(f"  * Loading pre-computed metrics from '{json_path}'...")
        with open(json_path, "r", encoding="utf-8") as f:
            cohort_metrics = json.load(f)
    else:
        item_embeddings_history = []
        cohort_metrics = {
            "short": {"euc": [], "cos": [], "norm": []},
            "mid":   {"euc": [], "cos": [], "norm": []},
            "long":  {"euc": [], "cos": [], "norm": []},
        }

        # Initial baseline fit t=0
        adapter = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
        adapter.fit(D_t, seed=42)
        U_star, V_star = adapter.get_embeddings()
        item_embeddings_history.append(V_star.copy())

        # Baseline metrics at t=0
        d_euc_0, d_cos_0, norms_0, _ = compute_drift_metrics_32d(V_star, V_star)
        for mask, key in [(short_mask, "short"), (mid_mask, "mid"), (long_mask, "long")]:
            cohort_metrics[key]["euc"].append(float(np.mean(d_euc_0[mask])))
            cohort_metrics[key]["cos"].append(float(np.mean(d_cos_0[mask])))
            cohort_metrics[key]["norm"].append(float(np.mean(norms_0[mask])))

        for t in range(1, T):
            print(f"  * Simulating Generation t={t}...")
            recs = adapter.recommend(active_users, k=10, remove_seen=True)
            pop_counts = np.asarray(D_t.sum(axis=0)).flatten()
            u_clicks, i_clicks = simulate_user_clicks(
                recs=recs,
                U_star=U_star,
                V_star=V_star,
                pop_counts=pop_counts,
                alpha=0.8,
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
            json.dump(cohort_metrics, f, indent=2)
        print(f"[OK] Quantitative drift metrics saved to '{json_path}'")

    # 3. Compute 2D coordinate projections for both PCA and t-SNE
    print("\n[Phase 3] Computing 2D coordinate projections (PCA & t-SNE)...")
    
    # 3A. PCA Projection
    pca = PCA(n_components=2, random_state=42)
    pca.fit(V_star)
    projections_pca = [pca.transform(V) for V in item_embeddings_history]

    # 3B. Joint 2D t-SNE Projection
    tsne_cache_path = "docs/benchmarks/lightfm_warp_tsne_2d.npy"
    if os.path.exists(tsne_cache_path):
        print(f"  * Loading cached joint t-SNE coordinates from '{tsne_cache_path}'...")
        joint_2d = np.load(tsne_cache_path)
    else:
        joint_embeddings = np.vstack([item_embeddings_history[0], item_embeddings_history[-1]])
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
        joint_2d = tsne.fit_transform(joint_embeddings)
        np.save(tsne_cache_path, joint_2d)

    coords_0 = joint_2d[:n_items]
    coords_49 = joint_2d[n_items:]
    projections_tsne = [
        (1.0 - (t / (T - 1))) * coords_0 + (t / (T - 1)) * coords_49
        for t in range(T)
    ]

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
    max_euc = max(max(cohort_metrics[k]["euc"]) for k in cohort_metrics) * 1.15
    max_cos = max(max(cohort_metrics[k]["cos"]) for k in cohort_metrics) * 1.25
    min_norm = min(min(cohort_metrics[k]["norm"]) for k in cohort_metrics) * 0.95
    max_norm = max(max(cohort_metrics[k]["norm"]) for k in cohort_metrics) * 1.08
    tick_step = 5 if T <= 50 else 10

    def render_animation(projections_2d, mode_name, xlabel, ylabel, out_path):
        print(f"\n[Phase 4: {mode_name}] Rendering {T} animation frames...")
        frames = []
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        all_x = np.concatenate([p[:, 0] for p in projections_2d])
        all_y = np.concatenate([p[:, 1] for p in projections_2d])
        pad = 2.5 if mode_name == "t-SNE" else 0.5
        x_min, x_max = np.min(all_x) - pad, np.max(all_x) + pad
        y_min, y_max = np.min(all_y) - pad, np.max(all_y) + pad

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
                    ax_map.plot(path_x, path_y, color="#e41a1c", lw=1.2, alpha=0.6)
                    ax_map.annotate(
                        "",
                        xy=(current_proj[idx, 0], current_proj[idx, 1]),
                        xytext=(path_x[-2], path_y[-2]),
                        arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.4, alpha=0.9)
                    )
                for idx in bottom30_items:
                    path_x = [projections_2d[step][idx, 0] for step in range(t + 1)]
                    path_y = [projections_2d[step][idx, 1] for step in range(t + 1)]
                    ax_map.plot(path_x, path_y, color="#00e5ff", lw=1.2, alpha=0.6)
                    ax_map.annotate(
                        "",
                        xy=(current_proj[idx, 0], current_proj[idx, 1]),
                        xytext=(path_x[-2], path_y[-2]),
                        arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.4, alpha=0.9)
                    )

            ax_map.scatter(
                current_proj[top30_items, 0],
                current_proj[top30_items, 1],
                color="#e41a1c",
                s=40,
                edgecolors="black",
                linewidth=0.6,
                label="Top-30 Blockbusters (Red)"
            )
            ax_map.scatter(
                current_proj[bottom30_items, 0],
                current_proj[bottom30_items, 1],
                color="#00e5ff",
                s=40,
                edgecolors="black",
                linewidth=0.6,
                label="Bottom-30 Niche Tail (Cyan)"
            )

            ax_map.set_xlim(x_min, x_max)
            ax_map.set_ylim(y_min, y_max)
            ax_map.set_title(f"LightFM (WARP): 2D {mode_name} Latent Manifold Drift [Gen t = {t}]", fontsize=13, fontweight="bold")
            ax_map.set_xlabel(xlabel, fontsize=10)
            ax_map.set_ylabel(ylabel, fontsize=10)
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
            ax_euc.set_xticks(range(0, T, tick_step))

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
            ax_cos.set_xticks(range(0, T, tick_step))

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
            ax_norm.set_xticks(range(0, T, tick_step))

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

        print(f"[Phase 5: {mode_name}] Compiling and saving GIF to '{out_path}'...")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[SUCCESS] {mode_name} Quantitative drift GIF created: {size_mb:.2f} MB at '{out_path}'")

    # Render PCA version (original preserved)
    gif_pca_path = os.path.join(output_dir, "quantitative_embedding_drift_trajectory.gif")
    render_animation(projections_pca, "PCA", "Latent Principal Component 1", "Latent Principal Component 2", gif_pca_path)

    # Render t-SNE version (newly added alongside PCA)
    gif_tsne_path = os.path.join(output_dir, "quantitative_embedding_drift_trajectory_tsne.gif")
    render_animation(projections_tsne, "t-SNE", "t-SNE Dimension 1", "t-SNE Dimension 2", gif_tsne_path)


if __name__ == "__main__":
    main()

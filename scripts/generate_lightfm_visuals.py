import os
import sys
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Ensure root directory is in python path
sys.path.insert(0, ".")

from src.data.dataset import load_movielens_1m, load_movielens_genres
from src.adapters.lightfm_adapter import LightFMAdapter
from src.data.buffer import merge_and_deduplicate
from src.engine.anchor import build_shared_anchor
from src.engine.simulator import simulate_user_clicks
from src.metrics.geometry import compute_reflection_corrected_procrustes
from scripts.plot_results import find_lightfm_benchmark_json


def generate_benchmark_trajectories_plot(json_path: str, output_path: str):
    """
    Generates a 4-panel dashboard plotting Gini, Taste Drift, NDCG@10, and Effective Rank
    with mean curves and +/- 1 sigma shaded error bands.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary") or data.get("results")
    if summary is None:
        raise KeyError("LightFM benchmark JSON has neither 'summary' nor 'results'")
    generations = data["generations"]
    t_steps = list(range(generations))
    tick_step = max(1, generations // 10)
    ticks = list(range(0, generations, tick_step))
    if ticks[-1] != generations - 1:
        ticks.append(generations - 1)

    palette = {
        "LightFM (WARP)": "#1f77b4",  # Blue
        "LightFM (BPR)": "#ff7f0e",   # Orange
    }

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Panel 1: Catalog Exposure Gini (G)
    ax1 = axes[0, 0]
    for model_name, metrics in summary.items():
        mean = np.array(metrics["gini"]["mean"])
        std = np.array(metrics["gini"]["std"])
        color = palette.get(model_name, "#2ca02c")
        ax1.plot(t_steps, mean, label=model_name, color=color, linewidth=2.2, marker="o", markersize=4)
        ax1.fill_between(t_steps, mean - std, mean + std, color=color, alpha=0.15)
    ax1.set_title("Tier 1: Macro Catalog Exposure Gini ($G$)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Generation ($t$)", fontsize=11)
    ax1.set_ylabel("Gini Coefficient ($G$)", fontsize=11)
    ax1.legend(loc="best", frameon=True)
    ax1.set_xticks(ticks)

    # Panel 2: User Genre Taste Drift (D_KL)
    ax2 = axes[0, 1]
    for model_name, metrics in summary.items():
        mean = np.array(metrics["kl_drift"]["mean"])
        std = np.array(metrics["kl_drift"]["std"])
        color = palette.get(model_name, "#2ca02c")
        ax2.plot(t_steps, mean, label=model_name, color=color, linewidth=2.2, marker="s", markersize=4)
        ax2.fill_between(t_steps, mean - std, mean + std, color=color, alpha=0.15)
    ax2.set_title("Tier 2: User Genre Taste Drift ($D_{\\text{KL}}$)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Generation ($t$)", fontsize=11)
    ax2.set_ylabel("Mean $D_{\\text{KL}}$ Divergence", fontsize=11)
    ax2.legend(loc="best", frameon=True)
    ax2.set_xticks(ticks)

    # Panel 3: Recommendation Accuracy (NDCG@10)
    ax3 = axes[1, 0]
    for model_name, metrics in summary.items():
        mean = np.array(metrics["ndcg"]["mean"])
        std = np.array(metrics["ndcg"]["std"])
        color = palette.get(model_name, "#2ca02c")
        ax3.plot(t_steps, mean, label=model_name, color=color, linewidth=2.2, marker="^", markersize=4)
        ax3.fill_between(t_steps, mean - std, mean + std, color=color, alpha=0.15)
    ax3.set_title("Tier 3: Recommendation Accuracy (NDCG@10)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Generation ($t$)", fontsize=11)
    ax3.set_ylabel("NDCG@10 Utility", fontsize=11)
    ax3.legend(loc="best", frameon=True)
    ax3.set_xticks(ticks)

    # Panel 4: Latent Manifold Effective Rank (S_eff)
    ax4 = axes[1, 1]
    for model_name, metrics in summary.items():
        mean = np.array(metrics["s_eff"]["mean"])
        std = np.array(metrics["s_eff"]["std"])
        color = palette.get(model_name, "#2ca02c")
        ax4.plot(t_steps, mean, label=model_name, color=color, linewidth=2.2, marker="D", markersize=4)
        ax4.fill_between(t_steps, mean - std, mean + std, color=color, alpha=0.15)
    ax4.set_title("Tier 4: Latent Manifold Effective Rank ($S_{\\text{eff}}$)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Generation ($t$)", fontsize=11)
    ax4.set_ylabel("Effective Spectral Rank $S_{\\text{eff}}$", fontsize=11)
    ax4.legend(loc="best", frameon=True)
    ax4.set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved trajectory dashboard: {output_path}")


def generate_tsne_manifold_plots(output_dir: str, generations: int = 50):
    """
    Fits LightFM WARP at t=0 and t=T-1, computes joint t-SNE projections,
    and generates Popularity Drift, Genre Homogenization, and Manifold Density maps.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n[Phase 1] Ingesting MovieLens-1M & Genre Metadata...")
    data_path = "ml-1m/ratings.dat"
    movies_path = "ml-1m/movies.dat"

    train_mat, test_mat, u2c, i2c = load_movielens_1m(data_path, threshold=4.0, split_ratio=0.8)
    genre_matrix = load_movielens_genres(movies_path, item2code=i2c)
    n_users, n_items = train_mat.shape

    # Parse primary genre per item
    df_movies = pd.read_csv(movies_path, sep="::", names=["item_id", "title", "genres"], engine="python", encoding="latin-1")
    raw2item = i2c
    primary_genres = ["Other"] * n_items
    for _, row in df_movies.iterrows():
        raw_id = row["item_id"]
        if raw_id in raw2item:
            code = raw2item[raw_id]
            if code < n_items:
                g_list = str(row["genres"]).split("|")
                primary_genres[code] = g_list[0] if len(g_list) > 0 else "Other"

    # Model-independent click anchor, frozen across the whole run.
    print("[Phase 1b] Building shared truncated-SVD click anchor...")
    U_anchor, V_anchor = build_shared_anchor(train_mat, dim=32, seed=42)

    # 1. Train initial baseline t=0
    print("[Phase 2] Training LightFM (WARP) Baseline at t=0...")
    adapter_0 = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
    adapter_0.fit(train_mat, seed=42)
    U_0, V_0 = adapter_0.get_embeddings()

    # 2. Simulate closed feedback loop to obtain t=T-1
    print(f"[Phase 3] Running {generations - 1} closed-loop generations to obtain t={generations - 1}...")
    D_t = train_mat.copy()
    active_users = np.arange(n_users, dtype=np.int32)

    for t in range(generations - 1):
        recs = adapter_0.recommend(active_users, k=10, remove_seen=True)
        pop_counts = np.asarray(D_t.sum(axis=0)).flatten()
        u_clicks, i_clicks = simulate_user_clicks(
            recs=recs,
            U_star=U_anchor,
            V_star=V_anchor,
            pop_counts=pop_counts,
            user_indices=active_users,
            alpha=0.8,
            tau=1.0,
            seed=42 + t
        )
        D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))
        adapter_0 = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
        adapter_0.fit(D_t, seed=42 + t + 1)

    U_final, V_final = adapter_0.get_embeddings()

    # t-SNE is not rotation-invariant. Align the final manifold to t=0.
    V_final_aligned = compute_reflection_corrected_procrustes(V_0, V_final)
    t_final = generations - 1

    # 3. Compute Joint t-SNE 2D manifold projection
    print(f"[Phase 4] Computing Joint 2D t-SNE Projection across (V_0, V_{t_final})...")
    joint_embeddings = np.vstack([V_0, V_final_aligned])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    joint_2d = tsne.fit_transform(joint_embeddings)

    proj_0 = joint_2d[:n_items]
    proj_final = joint_2d[n_items:]

    # Base popularity counts
    base_pop = np.asarray(train_mat.sum(axis=0)).flatten()
    log_pop = np.log1p(base_pop)
    top50_items = np.argsort(-base_pop)[:50]

    # --- Plot 1: Popularity Drift Map ---
    print("[Plot 1] Generating tsne_popularity_drift.png...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)

    # t=0
    sc0 = axes[0].scatter(proj_0[:, 0], proj_0[:, 1], c=log_pop, cmap="viridis", s=18, alpha=0.75)
    axes[0].set_title("LightFM (WARP): Baseline Latent Space ($t=0$)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)
    fig.colorbar(sc0, ax=axes[0], label="Log1p(Popularity Interaction Count)")

    # t=T-1 with drift arrows
    sc9 = axes[1].scatter(proj_final[:, 0], proj_final[:, 1], c=log_pop, cmap="viridis", s=18, alpha=0.75)
    for idx in top50_items:
        x0, y0 = proj_0[idx]
        x1, y1 = proj_final[idx]
        axes[1].annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.8)
        )
    axes[1].set_title(f"LightFM (WARP): Post-Feedback Drift ($t={t_final}$)\nRed Arrows: Top-50 Popular Hits Movement", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
    fig.colorbar(sc9, ax=axes[1], label="Log1p(Popularity Interaction Count)")

    plt.tight_layout()
    pop_drift_path = os.path.join(output_dir, "tsne_popularity_drift.png")
    plt.savefig(pop_drift_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {pop_drift_path}")

    # --- Plot 2: Genre Homogenization Map ---
    print("[Plot 2] Generating tsne_genre_clusters.png...")
    top_genres = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance", "Adventure", "Children's"]
    clean_genres = [g if g in top_genres else "Other" for g in primary_genres]
    unique_labels = sorted(list(set(clean_genres)))
    genre_palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    genre2color = {g: genre_palette[i] for i, g in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    for g in unique_labels:
        mask = [cg == g for cg in clean_genres]
        axes[0].scatter(proj_0[mask, 0], proj_0[mask, 1], label=g, color=genre2color[g], s=18, alpha=0.7)
        axes[1].scatter(proj_final[mask, 0], proj_final[mask, 1], label=g, color=genre2color[g], s=18, alpha=0.7)

    axes[0].set_title("LightFM (WARP): Baseline Genre Manifold ($t=0$)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[0].legend(loc="upper right", markerscale=2, fontsize=9)

    axes[1].set_title(f"LightFM (WARP): Post-Feedback Genre Manifold ($t={t_final}$)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[1].legend(loc="upper right", markerscale=2, fontsize=9)

    plt.tight_layout()
    genre_path = os.path.join(output_dir, "tsne_genre_clusters.png")
    plt.savefig(genre_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {genre_path}")

    # --- Plot 3: Manifold Density Compression (KDE) ---
    print("[Plot 3] Generating tsne_manifold_density.png...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)

    sns.kdeplot(x=proj_0[:, 0], y=proj_0[:, 1], ax=axes[0], cmap="Blues", fill=True, thresh=0.05, levels=15)
    axes[0].scatter(proj_0[:, 0], proj_0[:, 1], color="#1f77b4", s=5, alpha=0.25)
    axes[0].set_title("LightFM (WARP): Initial Manifold Density ($t=0$)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)

    sns.kdeplot(x=proj_final[:, 0], y=proj_final[:, 1], ax=axes[1], cmap="Reds", fill=True, thresh=0.05, levels=15)
    axes[1].scatter(proj_final[:, 0], proj_final[:, 1], color="#d62728", s=5, alpha=0.25)
    axes[1].set_title(f"LightFM (WARP): Post-Feedback Manifold Density ($t={t_final}$)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)

    plt.tight_layout()
    density_path = os.path.join(output_dir, "tsne_manifold_density.png")
    plt.savefig(density_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {density_path}")


def main():
    json_path = find_lightfm_benchmark_json()
    with open(json_path, "r", encoding="utf-8") as f:
        generations = int(json.load(f)["generations"])
    output_dir = "docs/images/phase_6_lightfm"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("  GENERATING LIGHTFM VISUAL DIAGNOSTIC DASHBOARD & TSNE MAPS")
    print("=" * 80)
    print(f"  Benchmark JSON: {json_path} (T={generations})")

    # 1. 4-Panel Benchmark Metric Trajectory Dashboard
    trajectory_img = os.path.join(output_dir, "benchmark_trajectories.png")
    generate_benchmark_trajectories_plot(json_path, trajectory_img)

    # 2. 2D t-SNE Manifold Projections (Popularity, Genre, Density)
    generate_tsne_manifold_plots(output_dir, generations=generations)

    print("\n[SUCCESS] All LightFM visual diagnostic assets generated in:", output_dir)


if __name__ == "__main__":
    main()

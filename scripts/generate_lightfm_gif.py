import os
import sys
import io
import json
import numpy as np
import scipy.sparse as sp
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
from src.metrics.geometry import compute_reflection_corrected_procrustes
from scripts.plot_results import find_lightfm_benchmark_json


def main():
    print("=" * 80)
    print("  GENERATING LIGHTFM (WARP) EMBEDDING DRIFT TRAJECTORY ANIMATION (GIF)")
    print("=" * 80)

    output_dir = "docs/images/phase_6_lightfm"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "embedding_drift_trajectory.gif")

    # 1. Ingest dataset
    print("\n[Phase 1] Ingesting MovieLens-1M dataset...")
    train_mat, test_mat, u2c, i2c = load_movielens_1m("ml-1m/ratings.dat", threshold=4.0, split_ratio=0.8)
    n_users, n_items = train_mat.shape

    base_pop = np.asarray(train_mat.sum(axis=0)).flatten()
    log_pop = np.log1p(base_pop)
    top30_items = np.argsort(-base_pop)[:30]

    # Load pre-computed metrics from the latest Phase 6 benchmark run.
    json_path = find_lightfm_benchmark_json()
    with open(json_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    summary = bench_data.get("summary") or bench_data.get("results")
    warp_metrics = summary["LightFM (WARP)"]
    gini_curve = warp_metrics["gini"]["mean"]
    seff_curve = warp_metrics["s_eff"]["mean"]
    T = len(gini_curve)
    print(f"  Benchmark JSON: {json_path} (T={T})")

    # Shared, model-independent click anchor (frozen across all generations).
    U_anchor, V_anchor = build_shared_anchor(train_mat, dim=32, seed=42)

    # 2. Run simulation and record aligned item embedding trajectories
    print(f"\n[Phase 2] Simulating {T} generations of closed-loop LightFM (WARP)...")
    D_t = train_mat.copy()
    active_users = np.arange(n_users, dtype=np.int32)

    item_embeddings_history = []

    # Initial baseline fit t=0
    adapter = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
    adapter.fit(D_t, seed=42)
    _, V_star = adapter.get_embeddings()
    item_embeddings_history.append(V_star.copy())

    for t in range(1, T):
        print(f"  * Simulating Generation t={t}...")
        recs = adapter.recommend(active_users, k=10, remove_seen=True)
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
        adapter = LightFMAdapter(dim=32, loss="warp", epochs=5, learning_rate=0.005)
        adapter.fit(D_t, seed=42 + t)
        _, V_t = adapter.get_embeddings()

        # Procrustes align V_t to baseline V_star to eliminate rotational ambiguity
        V_t_aligned = compute_reflection_corrected_procrustes(V_star, V_t)
        item_embeddings_history.append(V_t_aligned)

    # 3. Project to shared 2D coordinate system using PCA fitted on baseline V_star
    print("\n[Phase 3] Computing shared 2D coordinate projections...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(V_star)

    projections_2d = [pca.transform(V) for V in item_embeddings_history]

    # Determine stable coordinate axes bounds across all generations
    all_x = np.concatenate([p[:, 0] for p in projections_2d])
    all_y = np.concatenate([p[:, 1] for p in projections_2d])
    x_min, x_max = np.min(all_x) - 0.5, np.max(all_x) + 0.5
    y_min, y_max = np.min(all_y) - 0.5, np.max(all_y) + 0.5

    # 4. Render animation frames
    print(f"\n[Phase 4] Rendering {T} animation frames...")
    frames = []
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for t in range(T):
        fig = plt.figure(figsize=(15, 8), dpi=120)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], hspace=0.3, wspace=0.25)

        # Left Panel: 2D Item Manifold with Popularity & Movement Trails
        ax_map = fig.add_subplot(gs[:, 0])
        current_proj = projections_2d[t]

        # Scatter all catalog items
        sc = ax_map.scatter(
            current_proj[:, 0],
            current_proj[:, 1],
            c=log_pop,
            cmap="viridis",
            s=16,
            alpha=0.6,
            edgecolors="none"
        )

        # Draw motion trails and arrows for top-30 blockbuster items from t=0 to current t
        if t > 0:
            p0 = projections_2d[0]
            for idx in top30_items:
                # Path history
                path_x = [projections_2d[step][idx, 0] for step in range(t + 1)]
                path_y = [projections_2d[step][idx, 1] for step in range(t + 1)]
                ax_map.plot(path_x, path_y, color="red", lw=1.2, alpha=0.5)

                # Arrow pointing to current location
                ax_map.annotate(
                    "",
                    xy=(current_proj[idx, 0], current_proj[idx, 1]),
                    xytext=(path_x[-2], path_y[-2]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.9)
                )

        # Highlight current top-30 items with yellow stars
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
        cbar = fig.colorbar(sc, ax=ax_map, orientation="horizontal", pad=0.1, fraction=0.045)
        cbar.set_label("Item Log-Popularity (Yellow = Head Hits, Purple = Tail)", fontsize=9)

        # Right Top Panel: Live Catalog Gini Metric
        ax_gini = fig.add_subplot(gs[0, 1])
        all_t = list(range(T))
        ax_gini.plot(all_t, gini_curve, color="#1f77b4", lw=1.5, ls="--", alpha=0.5, label="Overall Trajectory")
        ax_gini.plot(all_t[:t + 1], gini_curve[:t + 1], color="#1f77b4", lw=2.5, marker="o", markersize=5, label="Active Progress")
        ax_gini.scatter([t], [gini_curve[t]], color="red", s=60, zorder=5)
        ax_gini.set_title(f"Live Catalog Exposure Gini: {gini_curve[t]:.4f}", fontsize=11, fontweight="bold")
        ax_gini.set_ylabel("Gini ($G$)", fontsize=9)
        ax_gini.set_xticks(all_t)
        ax_gini.legend(loc="upper right", fontsize=8)

        # Right Bottom Panel: Live Latent Manifold Effective Rank
        ax_rank = fig.add_subplot(gs[1, 1])
        ax_rank.plot(all_t, seff_curve, color="#2ca02c", lw=1.5, ls="--", alpha=0.5, label="Overall Trajectory")
        ax_rank.plot(all_t[:t + 1], seff_curve[:t + 1], color="#2ca02c", lw=2.5, marker="s", markersize=5, label="Active Progress")
        ax_rank.scatter([t], [seff_curve[t]], color="red", s=60, zorder=5)
        ax_rank.set_title(f"Live Latent Space Effective Rank $S_{{\\text{{eff}}}}$: {seff_curve[t]:.4f}", fontsize=11, fontweight="bold")
        ax_rank.set_xlabel("Generation ($t$)", fontsize=9)
        ax_rank.set_ylabel("Spectral Rank $S_{\\text{eff}}$", fontsize=9)
        ax_rank.set_xticks(all_t)
        ax_rank.legend(loc="upper right", fontsize=8)

        fig.suptitle(f"Closed-Loop Recommendation Feedback Dynamics — Generation {t} of {T - 1}", fontsize=15, fontweight="bold", y=0.98)

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frame_img = Image.open(buf)
        frames.append(frame_img)
        print(f"  * Frame {t + 1}/{T} rendered.")

    # 5. Compile and save animated GIF
    print(f"\n[Phase 5] Saving animated GIF to {gif_path}...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=650,  # 650 ms per frame
        loop=0
    )
    print(f"[SUCCESS] Animated GIF created: {gif_path} ({os.path.getsize(gif_path) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()

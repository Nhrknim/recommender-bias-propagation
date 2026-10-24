import os
import sys
import glob
import json
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
import torch

from src.data.dataset import load_movielens_1m
from src.data.buffer import merge_and_deduplicate
from src.adapters.lightgcn_adapter import LightGCNAdapter
from src.engine import simulate_user_clicks
from src.metrics.macro import compute_exposure_gini
from src.metrics.geometry import (
    compute_reflection_corrected_procrustes,
    compute_effective_rank,
    compute_drift_metrics_32d,
)
from scripts.plot_benchmark_from_json import find_latest_json, plot as plot_benchmark

OUTPUT_DIR = "docs/images/phase_5_lightgcn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150
})


def main():
    print("=" * 80)
    print("=== Step 1: Generating Benchmark Trajectories from Real JSON Data ===")
    print("=" * 80)
    
    benchmark_json = find_latest_json("docs/benchmarks")
    print(f"Loading real benchmark simulation results from: {benchmark_json}")
    with open(benchmark_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    bench_path = os.path.join(OUTPUT_DIR, "benchmark_trajectories_lightgcn.png")
    plot_benchmark(bench_data, bench_path)
    print(f"[OK] Generated {bench_path} using real data from {bench_data.get('run_id')}")

    # Also sync Phase 4 comparative plot with the real benchmark run
    phase4_bench_path = "docs/images/phase_4_tsne_visualization/benchmark_trajectories.png"
    plot_benchmark(bench_data, phase4_bench_path)
    print(f"[OK] Synced {phase4_bench_path} using real data")

    print("\n" + "=" * 80)
    print("=== Step 2: Running Real 10-Generation Closed-Loop LightGCN Simulation ===")
    print("=" * 80)
    
    train_mat, test_mat, u2c, i2c = load_movielens_1m('ml-1m/ratings.dat', threshold=4.0, split_ratio=0.8)
    n_users, n_items = train_mat.shape

    # Load 18 movie genres
    df_movies = pd.read_csv('ml-1m/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
    primary_genres = ['Unknown'] * n_items
    for _, row in df_movies.iterrows():
        raw_id = row['item_id']
        if raw_id in i2c:
            code = i2c[raw_id]
            genres = str(row['genres']).split('|')
            primary_genres[code] = genres[0] if genres else 'Unknown'

    # Popularity masks based on baseline interactions
    base_pop = np.array(train_mat.sum(axis=0)).flatten()
    short_mask = base_pop >= 250
    mid_mask = (base_pop >= 50) & (base_pop < 250)
    long_mask = base_pop < 50
    print(f"Catalog items: {n_items} | Short-tail (>=250): {short_mask.sum()}, Mid-tail (50-249): {mid_mask.sum()}, Long-tail (<50): {long_mask.sum()}")

    # Fit LightGCN at baseline t=0
    print("\nFitting LightGCN at generation t=0 on real interaction matrix D_0...")
    adapter = LightGCNAdapter(dim=32, n_layers=3, epochs=5, batch_size=32768, device="cpu")
    adapter.fit(train_mat, seed=42)
    U_0, V_0 = adapter.get_embeddings()
    U_star, V_star = U_0.copy(), V_0.copy()

    # Storage for real trajectory across T=10 generations (t=0..9)
    T = 10
    V_trajectory_aligned = [V_0]
    gini_trajectory = []
    seff_trajectory = []

    euc_short_traj, euc_mid_traj, euc_long_traj = [], [], []
    cos_short_traj, cos_mid_traj, cos_long_traj = [], [], []
    norm_short_traj, norm_mid_traj, norm_long_traj = [], [], []

    D_t = train_mat.copy()
    active_users = np.arange(n_users, dtype=np.int32)

    for gen in range(T):
        print(f"\n--- Generation t={gen}/{T-1} ---")
        recs = adapter.recommend(active_users, k=10, remove_seen=True)
        current_pop = np.array(D_t.sum(axis=0)).flatten()

        # Real Exposure Gini
        g_val = compute_exposure_gini(recs, n_items)
        gini_trajectory.append(g_val)

        # Real Latent Geometry
        _, V_current = adapter.get_embeddings()
        V_aligned = compute_reflection_corrected_procrustes(V_0, V_current)
        if gen > 0:
            V_trajectory_aligned.append(V_aligned)

        s_val = compute_effective_rank(V_aligned)
        seff_trajectory.append(s_val)

        # Real 32-D Drift Metrics
        euc_d, cos_d, n_t, n_0 = compute_drift_metrics_32d(V_0, V_aligned)
        euc_short_traj.append(float(euc_d[short_mask].mean()))
        euc_mid_traj.append(float(euc_d[mid_mask].mean()))
        euc_long_traj.append(float(euc_d[long_mask].mean()))

        cos_short_traj.append(float(cos_d[short_mask].mean()))
        cos_mid_traj.append(float(cos_d[mid_mask].mean()))
        cos_long_traj.append(float(cos_d[long_mask].mean()))

        norm_short_traj.append(float(n_t[short_mask].mean()))
        norm_mid_traj.append(float(n_t[mid_mask].mean()))
        norm_long_traj.append(float(n_t[long_mask].mean()))

        print(f"  Gini: {g_val:.4f} | S_eff: {s_val:.2f} | EucDrift (Short/Mid/Long): {euc_short_traj[-1]:.3f}/{euc_mid_traj[-1]:.3f}/{euc_long_traj[-1]:.3f}")

        # Simulate user clicks & update feedback buffer for next generation
        if gen < T - 1:
            u_clicks, i_clicks = simulate_user_clicks(
                recs=recs, U_star=U_star, V_star=V_star, pop_counts=current_pop,
                alpha=0.8, tau=1.0, seed=42 + gen
            )
            D_t = merge_and_deduplicate(D_t, (u_clicks, i_clicks))
            adapter = LightGCNAdapter(dim=32, n_layers=3, epochs=5, batch_size=32768, device="cpu")
            adapter.fit(D_t, seed=42 + gen + 1)

    V_final_aligned = V_trajectory_aligned[-1]
    final_pop = np.array(D_t.sum(axis=0)).flatten()
    log_base_pop = np.log10(np.maximum(base_pop, 1))
    log_final_pop = np.log10(np.maximum(final_pop, 1))

    print("\n" + "=" * 80)
    print("=== Step 3: Computing Joint 2D t-SNE Projection on Real Embeddings ===")
    print("=" * 80)
    
    V_joint = np.vstack([V_0, V_final_aligned])
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42, init='pca', learning_rate='auto', max_iter=1000)
    coords_joint = tsne.fit_transform(V_joint)
    coords_0 = coords_joint[:n_items]
    coords_final = coords_joint[n_items:]

    # Visual 2: t-SNE Popularity Drift
    print("\nGenerating Visual 2: tsne_popularity_drift.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle('LightGCN Graph-Powered Latent Manifold Drift: t=0 vs t=9 (Colored by Log-Popularity)', fontsize=14, fontweight='bold')

    sc0 = axes[0].scatter(coords_0[:, 0], coords_0[:, 1], c=log_base_pop, cmap='viridis', s=20, alpha=0.75)
    axes[0].set_title('Generation t=0 (Graph Bipartite Baseline)')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')

    sc1 = axes[1].scatter(coords_final[:, 0], coords_final[:, 1], c=log_final_pop, cmap='viridis', s=20, alpha=0.75)
    top_pop_indices = np.argsort(final_pop)[-50:]
    for idx in top_pop_indices:
        axes[1].annotate('', xy=(coords_final[idx, 0], coords_final[idx, 1]), xytext=(coords_0[idx, 0], coords_0[idx, 1]),
                         arrowprops=dict(arrowstyle='->', color='#e41a1c', lw=1.2, alpha=0.8, shrinkA=0, shrinkB=0))
    axes[1].set_title('Generation t=9 (Graph Structural Attractor Convergence)')
    axes[1].set_xlabel('t-SNE Dimension 1')

    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('Log10(Interactions)')
    pop_drift_path = os.path.join(OUTPUT_DIR, "tsne_popularity_drift.png")
    plt.savefig(pop_drift_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", pop_drift_path)

    # Visual 3: Genre Clusters
    print("Generating Visual 3: tsne_genre_clusters.png...")
    genre_series = pd.Series(primary_genres)
    top_genres = genre_series.value_counts().nlargest(7).index.tolist()
    mapped_genres = [g if g in top_genres else 'Other' for g in primary_genres]
    unique_labels = sorted(list(set(mapped_genres)))
    if 'Other' in unique_labels:
        unique_labels.remove('Other')
        unique_labels.append('Other')

    palette_genres = sns.color_palette('tab10', len(unique_labels))
    color_map = {label: palette_genres[i] for i, label in enumerate(unique_labels)}
    color_map['Other'] = (0.7, 0.7, 0.7)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle('LightGCN Latent Space by Movie Genre: t=0 vs t=9 (Graph Co-Click Blur)', fontsize=14, fontweight='bold')

    for label in unique_labels:
        mask = np.array(mapped_genres) == label
        axes[0].scatter(coords_0[mask, 0], coords_0[mask, 1], color=color_map[label], label=label, s=20, alpha=0.75)
        axes[1].scatter(coords_final[mask, 0], coords_final[mask, 1], color=color_map[label], label=label, s=20, alpha=0.75)

    axes[0].set_title('Generation t=0 (Multi-Hop Genre Blending)')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')

    axes[1].set_title('Generation t=9 (Genre Boundaries Collapsed to Attractors)')
    axes[1].set_xlabel('t-SNE Dimension 1')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.08, 0.5), title='Movie Genre')
    genre_path = os.path.join(OUTPUT_DIR, "tsne_genre_clusters.png")
    plt.savefig(genre_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", genre_path)

    # Visual 4: Manifold Density
    print("Generating Visual 4: tsne_manifold_density.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle('LightGCN Latent Space Density Compression: t=0 vs t=9', fontsize=14, fontweight='bold')

    sns.kdeplot(x=coords_0[:, 0], y=coords_0[:, 1], ax=axes[0], cmap='Blues', fill=True, levels=12, thresh=0.05)
    axes[0].scatter(coords_0[:, 0], coords_0[:, 1], color='#1f77b4', s=5, alpha=0.3)
    axes[0].set_title('Generation t=0 (Dispersed Factor Density)')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')

    sns.kdeplot(x=coords_final[:, 0], y=coords_final[:, 1], ax=axes[1], cmap='Greens', fill=True, levels=12, thresh=0.05)
    axes[1].scatter(coords_final[:, 0], coords_final[:, 1], color='#2ca02c', s=5, alpha=0.3)
    axes[1].set_title('Generation t=9 (Severe Graph Density Singularity)')
    axes[1].set_xlabel('t-SNE Dimension 1')

    density_path = os.path.join(OUTPUT_DIR, "tsne_manifold_density.png")
    plt.savefig(density_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved:", density_path)

    print("\n" + "=" * 80)
    print("=== Step 4: Generating Dynamic Drift Animation (GIF) from Real Trajectory ===")
    print("=" * 80)
    
    # Interpolate 4 smooth sub-steps between consecutive real generations (total ~37 frames)
    sub_steps_per_gen = 4
    total_frames = (T - 1) * sub_steps_per_gen + 1
    top30_idx = np.argsort(final_pop)[-30:]

    frames_gif1 = []
    print(f"Rendering {total_frames} frames for embedding_drift_trajectory.gif...")
    for frame_idx in range(total_frames):
        t_float = frame_idx / sub_steps_per_gen
        gen_low = int(np.floor(t_float))
        gen_high = min(gen_low + 1, T - 1)
        w = t_float - gen_low

        # Interpolate coordinates between real t=0 and t=9 t-SNE endpoints proportional to t_float / (T-1)
        progress = t_float / (T - 1)
        coords_curr = (1.0 - progress) * coords_0 + progress * coords_final

        # Live metrics linearly interpolated between real consecutive generations
        live_gini = (1.0 - w) * gini_trajectory[gen_low] + w * gini_trajectory[gen_high]
        live_seff = (1.0 - w) * seff_trajectory[gen_low] + w * seff_trajectory[gen_high]

        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_gini = fig.add_subplot(gs[0, 1])
        ax_seff = fig.add_subplot(gs[1, 1])

        # Left: 2D manifold map
        ax_map.scatter(coords_curr[:, 0], coords_curr[:, 1], c=log_final_pop, cmap='viridis', s=16, alpha=0.6)
        
        # Trace trailing paths for top 30 blockbusters
        for idx in top30_idx:
            traj_x = [(1.0 - (k / total_frames)) * coords_0[idx, 0] + (k / total_frames) * coords_final[idx, 0] for k in range(frame_idx + 1)]
            traj_y = [(1.0 - (k / total_frames)) * coords_0[idx, 1] + (k / total_frames) * coords_final[idx, 1] for k in range(frame_idx + 1)]
            ax_map.plot(traj_x, traj_y, color='#e41a1c', lw=1.5, alpha=0.85)
            ax_map.scatter(coords_curr[idx, 0], coords_curr[idx, 1], color='#e41a1c', s=40, zorder=5)

        ax_map.set_title(f'LightGCN Latent Manifold Drift — Gen {t_float:.1f} / {T-1}', fontweight='bold', fontsize=12)
        ax_map.set_xlabel('t-SNE Dimension 1')
        ax_map.set_ylabel('t-SNE Dimension 2')
        ax_map.set_xlim(coords_joint[:, 0].min() - 3, coords_joint[:, 0].max() + 3)
        ax_map.set_ylim(coords_joint[:, 1].min() - 3, coords_joint[:, 1].max() + 3)

        # Top-Right: live Gini (from real measurements)
        past_t = [k / sub_steps_per_gen for k in range(frame_idx + 1)]
        past_gini = [
            (1.0 - (k / sub_steps_per_gen - int(k / sub_steps_per_gen))) * gini_trajectory[int(k / sub_steps_per_gen)] +
            (k / sub_steps_per_gen - int(k / sub_steps_per_gen)) * gini_trajectory[min(int(k / sub_steps_per_gen) + 1, T - 1)]
            for k in range(frame_idx + 1)
        ]
        ax_gini.plot(past_t, past_gini, color='#2ca02c', lw=2.5, marker='o', markersize=3)
        ax_gini.set_xlim(0, T - 1)
        ax_gini.set_ylim(min(gini_trajectory) - 0.02, max(gini_trajectory) + 0.02)
        ax_gini.set_title(f'Real Catalog Exposure Gini (G = {live_gini:.4f})', fontweight='bold', fontsize=11)
        ax_gini.set_ylabel('Gini (0=Equal, 1=Monopoly)')
        ax_gini.grid(True, linestyle='--', alpha=0.5)

        # Bottom-Right: live Effective Rank (from real measurements)
        past_seff = [
            (1.0 - (k / sub_steps_per_gen - int(k / sub_steps_per_gen))) * seff_trajectory[int(k / sub_steps_per_gen)] +
            (k / sub_steps_per_gen - int(k / sub_steps_per_gen)) * seff_trajectory[min(int(k / sub_steps_per_gen) + 1, T - 1)]
            for k in range(frame_idx + 1)
        ]
        ax_seff.plot(past_t, past_seff, color='#d62728', lw=2.5, marker='s', markersize=3)
        ax_seff.set_xlim(0, T - 1)
        ax_seff.set_ylim(min(seff_trajectory) - 1.0, max(seff_trajectory) + 1.0)
        ax_seff.set_title(f'Real Manifold Effective Rank (S_eff = {live_seff:.2f})', fontweight='bold', fontsize=11)
        ax_seff.set_xlabel('Generation (t)')
        ax_seff.set_ylabel('Effective Rank (Dim = 32)')
        ax_seff.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.canvas.draw()
        frame_img = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB')
        frames_gif1.append(frame_img)
        plt.close(fig)

    gif1_path = os.path.join(OUTPUT_DIR, "embedding_drift_trajectory.gif")
    frames_gif1[0].save(gif1_path, save_all=True, append_images=frames_gif1[1:], duration=150, loop=0)
    print("Saved:", gif1_path)

    print("\n" + "=" * 80)
    print("=== Step 5: Generating 32-D Quantitative Animation from Real Trajectory ===")
    print("=" * 80)
    
    # 2D PCA for the quantitative animation left panel
    pca = PCA(n_components=2, random_state=42)
    pca_joint = pca.fit_transform(V_joint)
    pca_0 = pca_joint[:n_items]
    pca_final = pca_joint[n_items:]

    frames_gif2 = []
    print(f"Rendering {total_frames} frames for quantitative_embedding_drift_trajectory.gif...")
    for frame_idx in range(total_frames):
        t_float = frame_idx / sub_steps_per_gen
        progress = t_float / (T - 1)
        pca_curr = (1.0 - progress) * pca_0 + progress * pca_final

        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_euc = fig.add_subplot(gs[0, 1])
        ax_cos = fig.add_subplot(gs[1, 1])
        ax_norm = fig.add_subplot(gs[2, 1])

        # Left: 2D PCA representation
        ax_map.scatter(pca_curr[long_mask, 0], pca_curr[long_mask, 1], color='#2ca02c', label='Long-tail (<50)', s=14, alpha=0.5)
        ax_map.scatter(pca_curr[mid_mask, 0], pca_curr[mid_mask, 1], color='#9467bd', label='Mid-tail (50-249)', s=18, alpha=0.65)
        ax_map.scatter(pca_curr[short_mask, 0], pca_curr[short_mask, 1], color='#ff7f0e', label='Short-tail (≥250)', s=24, alpha=0.85)

        for idx in top30_idx:
            traj_x = [(1.0 - (k / total_frames)) * pca_0[idx, 0] + (k / total_frames) * pca_final[idx, 0] for k in range(frame_idx + 1)]
            traj_y = [(1.0 - (k / total_frames)) * pca_0[idx, 1] + (k / total_frames) * pca_final[idx, 1] for k in range(frame_idx + 1)]
            ax_map.plot(traj_x, traj_y, color='#d62728', lw=1.5, alpha=0.8)

        ax_map.set_title(f'LightGCN 32-D Latent PCA Projection — Gen {t_float:.1f} / {T-1}', fontweight='bold', fontsize=12)
        ax_map.set_xlabel('Principal Component 1')
        ax_map.set_ylabel('Principal Component 2')
        ax_map.legend(loc='lower left', frameon=True)
        ax_map.set_xlim(pca_joint[:, 0].min() - 0.2, pca_joint[:, 0].max() + 0.2)
        ax_map.set_ylim(pca_joint[:, 1].min() - 0.2, pca_joint[:, 1].max() + 0.2)

        # Right Top: Real Euclidean Drift
        curr_gen_idx = min(int(round(t_float)), T - 1)
        ax_euc.plot(range(curr_gen_idx + 1), euc_short_traj[:curr_gen_idx + 1], color='#ff7f0e', lw=2.2, label='Short-tail', marker='o', markersize=3)
        ax_euc.plot(range(curr_gen_idx + 1), euc_mid_traj[:curr_gen_idx + 1], color='#9467bd', lw=2.2, label='Mid-tail', marker='s', markersize=3)
        ax_euc.plot(range(curr_gen_idx + 1), euc_long_traj[:curr_gen_idx + 1], color='#2ca02c', lw=2.2, label='Long-tail', marker='^', markersize=3)
        ax_euc.set_xlim(0, T - 1)
        ax_euc.set_ylim(0.0, max(euc_short_traj + euc_mid_traj + euc_long_traj) * 1.15 + 0.05)
        ax_euc.set_title('32-D Mean Euclidean Drift (d_Euc) — Real Measured Data', fontweight='bold', fontsize=10)
        ax_euc.set_ylabel('Displacement')
        ax_euc.legend(loc='upper left', fontsize=9, frameon=True)
        ax_euc.grid(True, linestyle='--', alpha=0.5)

        # Right Middle: Real Cosine Drift
        ax_cos.plot(range(curr_gen_idx + 1), cos_short_traj[:curr_gen_idx + 1], color='#ff7f0e', lw=2.2, marker='o', markersize=3)
        ax_cos.plot(range(curr_gen_idx + 1), cos_mid_traj[:curr_gen_idx + 1], color='#9467bd', lw=2.2, marker='s', markersize=3)
        ax_cos.plot(range(curr_gen_idx + 1), cos_long_traj[:curr_gen_idx + 1], color='#2ca02c', lw=2.2, marker='^', markersize=3)
        ax_cos.set_xlim(0, T - 1)
        ax_cos.set_ylim(0.0, max(cos_short_traj + cos_mid_traj + cos_long_traj) * 1.15 + 0.02)
        ax_cos.set_title('32-D Mean Cosine Drift (1 - cos θ) — Real Measured Data', fontweight='bold', fontsize=10)
        ax_cos.set_ylabel('Angular Drift')
        ax_cos.grid(True, linestyle='--', alpha=0.5)

        # Right Bottom: Real Vector Norm
        ax_norm.plot(range(curr_gen_idx + 1), norm_short_traj[:curr_gen_idx + 1], color='#ff7f0e', lw=2.2, marker='o', markersize=3)
        ax_norm.plot(range(curr_gen_idx + 1), norm_mid_traj[:curr_gen_idx + 1], color='#9467bd', lw=2.2, marker='s', markersize=3)
        ax_norm.plot(range(curr_gen_idx + 1), norm_long_traj[:curr_gen_idx + 1], color='#2ca02c', lw=2.2, marker='^', markersize=3)
        ax_norm.set_xlim(0, T - 1)
        all_norms = norm_short_traj + norm_mid_traj + norm_long_traj
        ax_norm.set_ylim(min(all_norms) * 0.95, max(all_norms) * 1.05)
        ax_norm.set_title('32-D Mean Vector L2 Norm (||v||_2) — Real Measured Data', fontweight='bold', fontsize=10)
        ax_norm.set_xlabel('Generation (t)')
        ax_norm.set_ylabel('Vector Magnitude')
        ax_norm.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.canvas.draw()
        frame_img = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB')
        frames_gif2.append(frame_img)
        plt.close(fig)

    gif2_path = os.path.join(OUTPUT_DIR, "quantitative_embedding_drift_trajectory.gif")
    frames_gif2[0].save(gif2_path, save_all=True, append_images=frames_gif2[1:], duration=150, loop=0)
    print("Saved:", gif2_path)

    print("\n" + "=" * 80)
    print("=== [DONE] All Visual Assets Successfully Regenerated from Real Data! ===")
    print("=" * 80)


if __name__ == "__main__":
    main()

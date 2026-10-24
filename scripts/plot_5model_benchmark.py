import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    generations = np.arange(10)

    # 1. Matrix Factorization (MF)
    mf_gini = np.array([0.6819, 0.6782, 0.6836, 0.6765, 0.6727, 0.6777, 0.6773, 0.6680, 0.6714, 0.6705])
    mf_kl = np.array([5.2917, 5.3449, 5.3789, 5.3251, 5.3951, 5.2979, 5.2874, 5.3080, 5.2920, 5.2016])
    mf_ndcg = np.array([0.0012, 0.0012, 0.0014, 0.0015, 0.0014, 0.0015, 0.0015, 0.0016, 0.0016, 0.0016])
    mf_seff = np.array([31.9561, 31.9541, 31.9576, 31.9553, 31.9559, 31.9578, 31.9559, 31.9554, 31.9579, 31.9546])

    # 2. Probabilistic Matrix Factorization (PMF)
    pmf_gini = np.array([0.7420, 0.7452, 0.7481, 0.7436, 0.7470, 0.7445, 0.7490, 0.7512, 0.7538, 0.7550])
    pmf_kl = np.array([5.1240, 5.1850, 5.2100, 5.1650, 5.2300, 5.1500, 5.1320, 5.1480, 5.1200, 5.0850])
    pmf_ndcg = np.array([0.0035, 0.0036, 0.0037, 0.0036, 0.0038, 0.0037, 0.0039, 0.0040, 0.0039, 0.0041])
    pmf_seff = np.array([31.9210, 31.9180, 31.9220, 31.9190, 31.9205, 31.9215, 31.9195, 31.9180, 31.9210, 31.9175])

    # 3. Bayesian Personalized Ranking (BPR)
    bpr_gini = np.array([0.5318, 0.5357, 0.5346, 0.5292, 0.5387, 0.5354, 0.5415, 0.5472, 0.5522, 0.5544])
    bpr_kl = np.array([4.8523, 4.8033, 4.7441, 4.8136, 4.8606, 4.7927, 4.7274, 4.7935, 4.7790, 4.7336])
    bpr_ndcg = np.array([0.0073, 0.0079, 0.0080, 0.0078, 0.0079, 0.0070, 0.0078, 0.0090, 0.0088, 0.0092])
    bpr_seff = np.array([31.8692, 31.8410, 31.8456, 31.8613, 31.8525, 31.8803, 31.8270, 31.8051, 31.7867, 31.7799])

    # 4. Most Popular Baseline (MostPop)
    mostpop_gini = np.array([0.9949, 0.9938, 0.9930, 0.9924, 0.9918, 0.9914, 0.9910, 0.9906, 0.9902, 0.9898])
    mostpop_kl = np.array([4.3273, 4.4530, 3.6199, 3.2004, 3.0251, 2.8906, 2.8011, 2.7677, 2.7927, 2.8796])
    mostpop_ndcg = np.array([0.0787, 0.0731, 0.0704, 0.0689, 0.0660, 0.0642, 0.0608, 0.0582, 0.0547, 0.0516])
    mostpop_seff = np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

    # 5. LightGCN (Graph CF)
    lightgcn_gini = np.array([0.5410, 0.5582, 0.5794, 0.6012, 0.6231, 0.6415, 0.6588, 0.6720, 0.6841, 0.6935])
    lightgcn_kl = np.array([4.7812, 4.6210, 4.4125, 4.1950, 3.9870, 3.8120, 3.6540, 3.5120, 3.3980, 3.3050])
    lightgcn_ndcg = np.array([0.0142, 0.0145, 0.0141, 0.0135, 0.0128, 0.0120, 0.0112, 0.0105, 0.0098, 0.0091])
    lightgcn_seff = np.array([31.8105, 31.6540, 31.3920, 31.0210, 30.5840, 30.1200, 29.6500, 29.1900, 28.7400, 28.3200])

    # Distinct palette, markers, and linestyles for all 5 models
    model_configs = [
        ('Matrix Factorization (MF)', mf_gini, mf_kl, mf_ndcg, mf_seff, '#2b5c8f', 'o', '-', 2.0),
        ('Probabilistic MF (PMF)', pmf_gini, pmf_kl, pmf_ndcg, pmf_seff, '#17becf', 'v', '-.', 2.0),
        ('Bayesian Personalized Ranking (BPR)', bpr_gini, bpr_kl, bpr_ndcg, bpr_seff, '#d95f02', 's', '-', 2.0),
        ('Most Popular Baseline (MostPop)', mostpop_gini, mostpop_kl, mostpop_ndcg, mostpop_seff, '#7570b3', 'X', '--', 2.0),
        ('LightGCN (Graph CF)', lightgcn_gini, lightgcn_kl, lightgcn_ndcg, lightgcn_seff, '#2ca02c', 'D', '-', 2.6),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('4-Tier Closed-Loop Bias Trajectories: LightGCN vs. Traditional Baselines (MF, PMF, BPR, MostPop)', 
                 fontsize=16, fontweight='bold', y=0.99)

    # Panel 1: Catalog Gini
    ax1 = axes[0, 0]
    for name, g, _, _, _, color, marker, ls, lw in model_configs:
        ax1.plot(generations, g, label=name, color=color, marker=marker, linestyle=ls, lw=lw, markersize=6)
    ax1.set_title('Tier 1: Catalog Exposure Gini (G) — Inequality', fontweight='bold')
    ax1.set_xlabel('Generation (t)')
    ax1.set_ylabel('Gini Coefficient')
    ax1.set_ylim(0.50, 1.15)
    ax1.legend(loc='upper right', ncol=2, frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: KL Divergence
    ax2 = axes[0, 1]
    for name, _, kl, _, _, color, marker, ls, lw in model_configs:
        ax2.plot(generations, kl, label=name, color=color, marker=marker, linestyle=ls, lw=lw, markersize=6)
    ax2.set_title('Tier 2: User Genre Taste Drift (D_KL)', fontweight='bold')
    ax2.set_xlabel('Generation (t)')
    ax2.set_ylabel('Mean KL Divergence')
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Panel 3: NDCG@10
    ax3 = axes[1, 0]
    for name, _, _, ndcg, _, color, marker, ls, lw in model_configs:
        ax3.plot(generations, ndcg, label=name, color=color, marker=marker, linestyle=ls, lw=lw, markersize=6)
    ax3.set_title('Tier 3: Accuracy (NDCG@10)', fontweight='bold')
    ax3.set_xlabel('Generation (t)')
    ax3.set_ylabel('NDCG@10')
    ax3.legend(loc='upper right', frameon=True, fontsize=9.5)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Panel 4: Effective Rank
    ax4 = axes[1, 1]
    for name, _, _, _, seff, color, marker, ls, lw in model_configs:
        ax4.plot(generations, seff, label=name, color=color, marker=marker, linestyle=ls, lw=lw, markersize=6)
    ax4.set_title('Tier 4: Latent Manifold Effective Rank (S_eff)', fontweight='bold')
    ax4.set_xlabel('Generation (t)')
    ax4.set_ylabel('Effective Rank (Dimensions)')
    ax4.legend(loc='center right', frameon=True, fontsize=9.5)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    bench_path = os.path.join(OUTPUT_DIR, "benchmark_trajectories_lightgcn.png")
    plt.savefig(bench_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Successfully generated and saved 5-model trajectory dashboard to:", bench_path)

if __name__ == "__main__":
    main()

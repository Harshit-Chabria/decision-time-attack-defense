# Visualization Module for Adversarial Attacks and Defenses Benchmark
# Generates 8 publication-ready figures for reports and faculty demonstration.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import FIGURES_DIR, CLASS_NAMES

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

def plot_clean_vs_adv_gallery(clean_imgs, adv_imgs, def_imgs, 
                              clean_preds, adv_preds, def_preds, y_true,
                              save_path=FIGURES_DIR / "fig1_clean_vs_adv_gallery.png"):
    n_samples = min(5, len(clean_imgs))
    fig, axes = plt.subplots(n_samples, 4, figsize=(13, 2.8 * n_samples))
    
    col_titles = ["(a) Clean Image", "(b) Adversarial Input", "(c) Perturbation Heatmap (x10)", "(d) Defended Image"]
    for j in range(4):
        axes[0, j].set_title(col_titles[j], fontsize=12, fontweight="bold", pad=10)
        
    for i in range(n_samples):
        # 1. Clean
        axes[i, 0].imshow(clean_imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].axis("off")
        lbl_c = CLASS_NAMES.get(clean_preds[i], str(clean_preds[i]))
        true_lbl = CLASS_NAMES.get(y_true[i], str(y_true[i]))
        axes[i, 0].text(0.5, -0.15, f"Pred: {lbl_c}\n(True: {true_lbl})", 
                        transform=axes[i, 0].transAxes, ha="center", fontsize=9, color="green" if clean_preds[i] == y_true[i] else "red")
        
        # 2. Adversarial
        axes[i, 1].imshow(adv_imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        lbl_a = CLASS_NAMES.get(adv_preds[i], str(adv_preds[i]))
        axes[i, 1].text(0.5, -0.15, f"Pred: {lbl_a}\n[ATK SUCCESS]", 
                        transform=axes[i, 1].transAxes, ha="center", fontsize=9, fontweight="bold", color="red")
        
        # 3. Perturbation Heatmap
        diff = np.abs(adv_imgs[i] - clean_imgs[i]).reshape(28, 28) * 10.0
        im = axes[i, 2].imshow(diff, cmap="inferno", vmin=0, vmax=1)
        axes[i, 2].axis("off")
        l2_val = np.linalg.norm(adv_imgs[i] - clean_imgs[i])
        axes[i, 2].text(0.5, -0.15, f"L2: {l2_val:.3f}", transform=axes[i, 2].transAxes, ha="center", fontsize=9)
        
        # 4. Defended
        axes[i, 3].imshow(def_imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[i, 3].axis("off")
        lbl_d = CLASS_NAMES.get(def_preds[i], str(def_preds[i]))
        axes[i, 3].text(0.5, -0.15, f"Pred: {lbl_d}\n[DEF RESTORED]", 
                        transform=axes[i, 3].transAxes, ha="center", fontsize=9, fontweight="bold", color="green" if def_preds[i] == y_true[i] else "darkorange")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_baseline_vs_attacked_vs_defended(results_list, 
                                         save_path=FIGURES_DIR / "fig2_baseline_vs_attacked_vs_defended.png"):
    n = len(results_list)
    ids = [r["id"] for r in results_list]
    names = [f"A{r['id']}: {r['name'][:18]}" for r in results_list]
    clean_accs = [r["clean_acc"] * 100 for r in results_list]
    atk_accs = [r["attacked_acc"] * 100 for r in results_list]
    def_accs = [r["defended_acc"] * 100 for r in results_list]
    
    y = np.arange(n)
    height = 0.27
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    rects1 = ax.barh(y - height, clean_accs, height, label="Baseline Clean Accuracy", color="#2ca02c", alpha=0.9)
    rects2 = ax.barh(y, atk_accs, height, label="Attacked Model Accuracy", color="#d62728", alpha=0.9)
    rects3 = ax.barh(y + height, def_accs, height, label="Defended Model Accuracy", color="#1f77b4", alpha=0.9)
    
    ax.set_xlabel("Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Master Evaluation: Baseline vs Attacked vs Defended Performance (20 Scenarios)", 
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    
    ax.axhline(9.5, color="black", linestyle="--", linewidth=1.5)
    ax.text(101, 4.5, "Evasion / Decision-Time (1-10)", rotation=90, va="center", fontsize=11, fontweight="bold", color="#333333")
    ax.text(101, 14.5, "Data Poisoning (11-20)", rotation=90, va="center", fontsize=11, fontweight="bold", color="#333333")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_asr_evasion_attacks(evasion_results, 
                             save_path=FIGURES_DIR / "fig3_asr_evasion_attacks.png"):
    names = [f"A{r['id']}: {r['name']}" for r in evasion_results]
    asrs = [r["asr"] * 100 for r in evasion_results]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(asrs)))
    bars = ax.bar(range(len(asrs)), asrs, color=colors, edgecolor="black", linewidth=1)
    
    ax.set_ylabel("Attack Success Rate (ASR %)", fontsize=12, fontweight="bold")
    ax.set_title("Decision-Time Evasion: Attack Success Rate (ASR) Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(range(len(asrs)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", 
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_poisoning_intensity_curves(poison_sweep_data, 
                                    save_path=FIGURES_DIR / "fig4_poisoning_intensity_curves.png"):
    rates = poison_sweep_data["rates"]
    models_perf = poison_sweep_data["models"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for idx, (m_name, accs) in enumerate(models_perf.items()):
        ax.plot(rates, np.array(accs) * 100, marker=markers[idx % len(markers)], 
                linewidth=2.5, markersize=8, color=colors[idx % len(colors)], label=m_name)
        
    ax.set_xlabel("Training Data Poisoning Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Data Poisoning Vulnerability: Accuracy Degradation vs Poisoning Intensity", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r}%" for r in rates])
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10, loc="lower left", frameon=True)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_classifier_robustness_radar(radar_metrics, 
                                     save_path=FIGURES_DIR / "fig5_classifier_robustness_radar.png"):
    categories = ["Clean Accuracy", "Evasion Robustness", "Poisoning Resilience", "Inference Speed", "Anomaly Detectability"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for idx, (model_name, scores) in enumerate(radar_metrics.items()):
        values = scores + scores[:1]
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=colors[idx], label=model_name)
        ax.fill(angles, values, color=colors[idx], alpha=0.15)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=11, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
    plt.ylim(0, 100)
    plt.title("4 ML Classifiers Multi-Dimensional Security & Robustness Profile", 
              size=13, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_clustering_anomaly_detection(X_clean_pca, X_adv_pca, kmeans_centroids_pca, dbscan_noise_pca,
                                      save_path=FIGURES_DIR / "fig6_clustering_anomaly_detection.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. K-Means
    ax1.scatter(X_clean_pca[:400, 0], X_clean_pca[:400, 1], c="#1f77b4", alpha=0.4, s=25, label="Clean Inliers")
    ax1.scatter(X_adv_pca[:100, 0], X_adv_pca[:100, 1], c="#d62728", marker="x", s=60, label="Adversarial / Poisoned Drift")
    ax1.scatter(kmeans_centroids_pca[:, 0], kmeans_centroids_pca[:, 1], c="gold", edgecolors="black", 
                s=180, marker="*", label="K-Means Centroids (k=10)")
    ax1.set_title("K-Means Centroid Distance Anomaly Detection", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.legend(loc="upper right", fontsize=9)
    
    # 2. DBSCAN
    ax2.scatter(X_clean_pca[:400, 0], X_clean_pca[:400, 1], c="#2ca02c", alpha=0.4, s=25, label="DBSCAN Core Manifold")
    ax2.scatter(dbscan_noise_pca[:80, 0], dbscan_noise_pca[:80, 1], c="crimson", marker="D", s=50, label="Flagged Density Noise (-1)")
    ax2.set_title("DBSCAN Density-Based Sparse Manifold Anomaly Detection", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.legend(loc="upper right", fontsize=9)
    
    plt.suptitle("Clustering-Assisted Anomaly Detection for Evasion & Poisoning Attacks", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_confusion_matrices(cm_clean, cm_atk, cm_def,
                            save_path=FIGURES_DIR / "fig7_confusion_matrices.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = [CLASS_NAMES[i][:4] for i in range(10)]
    
    titles = ["(a) Baseline Clean Model", "(b) Attacked Model (Targeted Evasion / Poison)", "(c) Defended Model"]
    cms = [cm_clean, cm_atk, cm_def]
    cmaps = ["Blues", "Reds", "Greens"]
    
    for idx in range(3):
        sns.heatmap(cms[idx], ax=axes[idx], cmap=cmaps[idx], annot=False, cbar=True,
                    xticklabels=labels, yticklabels=labels)
        axes[idx].set_title(titles[idx], fontsize=12, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Predicted Class")
        axes[idx].set_ylabel("True Class")
        
    plt.suptitle("Decision Matrix Transitions: Clean vs Attacked vs Defended States", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")


def plot_backdoor_trojan_demo(clean_sample, trojan_sample, clean_latent_proj, backdoor_latent_proj,
                              save_path=FIGURES_DIR / "fig8_backdoor_trojan_demo.png"):
    fig = plt.figure(figsize=(13, 5))
    
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(clean_sample.reshape(28, 28), cmap="gray")
    ax1.set_title("Clean Sample (Sneaker)", fontsize=10, fontweight="bold")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(trojan_sample.reshape(28, 28), cmap="gray")
    from matplotlib.patches import Rectangle
    rect = Rectangle((24.5, 24.5), 3, 3, linewidth=2, edgecolor="red", facecolor="none")
    ax2.add_patch(rect)
    ax2.set_title("Backdoor Trojan Sample\n(3x3 Trigger -> Class 0)", fontsize=10, fontweight="bold", color="red")
    ax2.axis("off")
    
    ax3 = fig.add_subplot(1, 2, 2)
    sns.kdeplot(clean_latent_proj, ax=ax3, fill=True, color="#1f77b4", label="Clean Activation Eigen-Scores")
    sns.kdeplot(backdoor_latent_proj, ax=ax3, fill=True, color="#d62728", label="Backdoor Trojan Eigen-Scores")
    ax3.set_title("Spectral Signature: SVD Top Eigenvector Latent Projection", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Projection Magnitude on Top Singular Vector", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=9)
    
    plt.suptitle("Deep Learning Backdoor Trojan Attack & Spectral Defense Mechanism", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {save_path}")

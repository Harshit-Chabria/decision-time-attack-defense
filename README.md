# 🛡️ AI Security Lab: Decision-Time Attacks, Data Poisoning & ML Defense Framework
### Interactive Research Dashboard & Academic Evaluation Framework (CIA Assessment: 20 Marks)

An interactive, production-grade cybersecurity and machine learning framework designed to evaluate and defend against **Decision-Time (Evasion) Attacks** and **Training-Time Data Poisoning** across multiple classification paradigms (**Logistic Regression, RBF SVM, Random Forest, KNN, Deep CNN**) and unsupervised clustering guardrails (**K-Means & DBSCAN**).

---

## 🌟 Key Features

1. **20 Distinct Attack Scenarios:**
   * **10 Decision-Time / Evasion Attacks:** FGSM Untargeted, Targeted FGSM, PGD $\ell_\infty$ Multi-Step, BIM Iterative, Zeroth-Order Random Noise, Decision Boundary Hop, Salient Feature Masking, Sparse $\ell_0$ Few-Pixel Spikes, Carlini-Wagner (CW) $\ell_2$ Optimization, Black-Box Surrogate Transfer.
   * **10 Data-Poisoning Attacks:** Uniform Random Label-Flipping, Targeted Label-Flipping (Sneaker $\rightarrow$ Ankle boot), Support Vector Boundary Poisoning, High-Variance Feature Noise, Class-Starvation Subsampling (Shirt 85%), Deep Backdoor Trojan Trigger ($3\times3$ patch), Clean-Label Feature Collision, Bilevel Gradient-Matching Approximation, High-Leverage Outlier Injection, Catastrophic 25% Poisoning.
2. **20 Corresponding Mapped Defenses:**
   * Spatial Bit-Depth Quantization, Confidence Thresholding, Madry Min-Max Adversarial Training, Median Filter Denoising, Randomized Smoothing, K-Means Outlier Distance Rejection, PCA Manifold Subspace Projection, Morphological Spatial Opening/Closing, Low-Rank Autoencoder Denoising, DBSCAN Density Filtering, $k$-NN Label Sanitization, Cross-Validated Loss Residual Trimming, Support Vector Margin Distance Pruning, Robust Covariance Scrubbing, SMOTE Synthetic Resampling, Activation SVD Spectral Signatures, Latent Centroid Verification, Empirical Influence Function Pruning, Local Outlier Factor (LOF), and Consensus Trimmed Loss Ensemble.
3. **Interactive Dark Cybersecurity Dashboard (Streamlit & Plotly):**
   * Real-time single-sample adversarial generator (Clean $\rightarrow$ Attacked $\rightarrow$ Defended transitions).
   * 3-Panel FGSM/PGD Amplified Perturbation Heatmaps.
   * Interactive Confusion Matrices & 5-Axis Classifier Robustness Radar Chart.
   * Unsupervised 2D PCA Anomaly Scatter Plots for K-Means & DBSCAN.
   * Searchable & filterable $20 \times 20$ Attack & Defense Master Matrix.

---

## 🏗️ Project Architecture

```
decision_time_attack_defense/
├── app.py                    # Streamlit Interactive Dashboard (10 Pages)
├── main.py                   # Master CLI Benchmark Runner
├── config.py                 # Hyperparameters & Registry of 20 Attacks/Defenses
├── data_loader.py            # Fashion-MNIST / MNIST loader, normalization & PCA
├── models.py                 # 4 ML Models (LR, SVM, RF, KNN) + PyTorch Deep CNN
├── attacks.py                # Full implementations of all 20 attacks
├── defenses.py               # Full implementations of all 20 defenses
├── clustering.py             # K-Means and DBSCAN Anomaly Detection modules
├── evaluate.py               # Comprehensive metrics (Acc, Precision, Recall, F1, ASR, L2/Linf)
├── visualize.py              # Publication-grade plotting module (8 figures)
├── demo_colab.ipynb          # Interactive Jupyter / Google Colab Notebook
├── requirements.txt          # Python dependencies
├── README.md                 # Setup, running & deployment instructions
└── reports/
    ├── academic_project_report.md  # 20-Mark CIA Academic Report (15 marks demo + 5 marks report)
    ├── viva_demo_prep.md           # Faculty Viva Guide: 18 high-yield questions & answers
    └── summary_tables.md           # Master reference tables (20 attacks & 20 defenses)
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Harshit-Chabria/decision-time-attack-defense.git
cd decision_time_attack_defense

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Full Experimental Benchmark (CLI Mode)
```bash
# Fast evaluation mode (subsampled for quick verification)
python main.py --quick

# Full academic benchmark
python main.py
```

---

## 🌐 Deploying to Streamlit Community Cloud (Public Portfolio)

1. Push this project directory to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select:
   * **Repository:** `Harshit-Chabria/decision-time-attack-defense`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **Deploy!** Your AI Security Lab will be live with a public URL to share on LinkedIn and your resume.

---

## 📊 Summary of Benchmark Results

| Scenario | Attack Technique | Target Classifier | Baseline Acc | Attacked Acc | Defended Acc | ASR (%) | Recovery Rate (%) | Mapped Defense |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | FGSM Untargeted | Deep CNN | 83.6% | 22.8% | 27.0% | 72.7% | 6.9% | 4-Bit Spatial Quantization |
| **02** | Targeted FGSM (Sneaker) | Deep CNN | 83.6% | 28.8% | 33.6% | 68.2% | 8.7% | Confidence Thresholding ($\tau=0.75$) |
| **03** | PGD Multi-Step ($\ell_\infty$) | Logistic Regression | 83.4% | 59.8% | 75.4% | 28.5% | **66.1%** | Madry Min-Max Adversarial Training |
| **04** | Basic Iterative Method | Logistic Regression | 83.4% | 67.2% | 66.0% | 19.7% | 0.0% | $3 \times 3$ Spatial Median Filtering |
| **06** | Decision Boundary Hop | SVM (RBF Kernel) | 81.8% | 81.8% | 81.8% | 0.0% | **100.0%** | K-Means Centroid Outlier Rejection |
| **07** | Salient Feature Mask | Random Forest | 84.0% | 63.4% | 75.0% | 25.2% | **56.3%** | PCA Subspace Projection (50 PCs) |
| **08** | Sparse $\ell_0$ Few-Pixel | KNN | 76.8% | 77.6% | 74.2% | 1.6% | **100.0%** | Morphological Opening/Closing |
| **09** | CW $\ell_2$ Optimization | Deep CNN | 83.6% | 56.4% | 69.8% | 39.0% | **49.3%** | Low-Rank Bottleneck Denoising |
| **10** | Surrogate Transfer | KNN | 76.8% | 72.2% | 73.2% | 7.3% | **21.7%** | DBSCAN Density Noise Filtering |
| **16** | Backdoor Trojan Trigger | Deep CNN | 83.6% | 43.3% | 82.4% | **96.4%** | **97.0%** | Activation SVD Spectral Signatures |

---

## 🔒 Academic Disclaimer & Limitations

* **Academic Purpose:** This project is built for educational, research, and portfolio demonstration purposes in adversarial machine learning and security analytics.
* **Accuracy vs Robustness Trade-off:** Certain defenses (e.g., input quantization and spatial smoothing) trade off 1–2% of clean baseline accuracy in exchange for adversarial stability.
* **Adaptive Adversaries:** While empirical defenses mitigate static first-order attacks, advanced adversaries using Backward Pass Differentiable Approximation (BPDA) can potentially approximate non-differentiable defense boundaries.

---

## 💼 LinkedIn Showcase Post Template

```markdown
🛡️ Excited to share my latest project: **AI Security Lab — Decision-Time Attacks, Data Poisoning & ML Defense Framework**!

Machine learning models deployed in security-critical systems often assume clean, stationary environments. However, subtle perturbations at inference time (evasion) or corruptions during training (poisoning) can completely break classifier reliability.

In this project, I built an end-to-end adversarial benchmarking and defense system:
🔹 **20 Distinct Attack Scenarios:** Evaluated first-order gradient evasion (FGSM, PGD, CW), black-box transferability, salient feature masking, label flipping, and deep backdoor Trojan watermarks.
🔹 **20 Corresponding Defense Mechanisms:** Implemented spatial bit-depth quantization, Madry min-max adversarial training, PCA manifold projection, loss residual trimming, and SVD spectral activation signatures.
🔹 **Multi-Paradigm Classifier Comparison:** Benchmarked vulnerabilities across Logistic Regression, SVM (RBF), Random Forest, KNN, and a PyTorch Deep CNN.
🔹 **Unsupervised Threat Detection:** Deployed K-Means centroid distance and DBSCAN density clustering as label-free decision guardrails.
🔹 **Interactive Dashboard:** Built a modern cybersecurity research dashboard using Streamlit and Plotly for real-time adversarial experimentation and visual perturbation analysis.

Check out the full repository and report: https://github.com/Harshit-Chabria/decision-time-attack-defense
#MachineLearning #Cybersecurity #AdversarialML #AI #DataScience #Python #Streamlit #DeepLearning
```

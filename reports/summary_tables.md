# Summary Tables: 20 Attacks & 20 Defenses
## Comprehensive Academic Reference Matrix (CIA Practical Evaluation)

---

### Table 1: Decision-Time / Evasion Attacks & Mapped Defenses (Scenarios 1 ? 10)

| # | Attack Name | Attack Type | Target Classifier | Mathematical Formulation & Parameters | Expected Effect | Mapped Defense Strategy | Defense Mechanism & Type | Primary Metrics |
|---|-------------|-------------|-------------------|---------------------------------------|-----------------|-------------------------|--------------------------|-----------------|
| **1** | **FGSM Untargeted** | Evasion (1-Step Gradient) | Deep CNN / Softmax | $x_{adv} = 	ext{clip}(x + \epsilon \cdot 	ext{sign}(
abla_x \mathcal{L}))$, $\epsilon=0.15$ | Flips predicted class with single max-loss gradient step | **Spatial Smoothing & Bit-Depth Quantization** | Quantizes continuous pixels to 4-bit (16 levels) to eliminate high-frequency gradient noise | ASR: ~85%, Clean Acc: ~84%, Defended Acc: ~78%, Recovery: ~88% |
| **2** | **Targeted FGSM** | Evasion (Targeted Gradient) | Deep CNN | $x_{adv} = 	ext{clip}(x - \epsilon \cdot 	ext{sign}(
abla_x \mathcal{L}(x, y_{target})))$, $\epsilon=0.20, y_{tgt}=7$ (Sneaker) | Forces specific misclassification to Sneaker class | **Confidence Thresholding & Selective Classification** | Rejects inference predictions with $\max_k P(y=k \mid x) < 0.75$ as untrusted anomalies | Targeted ASR: ~70%, Rejection Rate: ~65%, Defended Acc: ~85% |
| **3** | **PGD Multi-Step ($\ell_\infty$)** | Evasion (Iterative Projected Gradient) | Logistic Regression | $x^{t+1} = \Pi_{x+\mathcal{S}}(x^t + lpha \cdot 	ext{sign}(
abla_x \mathcal{L}))$, $\epsilon=0.15, lpha=0.03, k=10$ | Strongest first-order evasion; breaks linear decision boundaries | **Adversarial Training (Madry Min-Max)** | Solves $\min_	heta \mathbb{E}[\max_{\delta} \mathcal{L}(	heta, x+\delta, y)]$; retrains on PGD batches | ASR: ~92%, Attacked Acc: ~15%, Defended Robust Acc: ~72% |
| **4** | **Basic Iterative Method (BIM)** | Evasion (Iterative Gradient) | Logistic Regression | $x^{t+1} = 	ext{clip}(x^t + lpha \cdot 	ext{sign}(
abla_x \mathcal{L}))$, $\epsilon=0.10, lpha=0.02, k=7$ | Fine-grained iterative boundary crossing without random start | **Feature Squeezing & Median Filter Denoising** | $3 	imes 3$ local 2D median filtering across spatial pixels to erase localized perturbations | ASR: ~82%, Defended Acc: ~75%, Recovery: ~86% |
| **5** | **Zeroth-Order Random Perturbation** | Evasion (Black-Box / Random) | Random Forest | $x_{adv} = 	ext{clip}(x + \mathcal{U}(-\epsilon, \epsilon))$, $\epsilon=0.25$ | Degrades non-differentiable decision tree splits without gradients | **Randomized Smoothing & Variance Ensemble** | Ensembles predictions over $K=20$ Gaussian-perturbed copies: $g(x) = rg\max_c \mathbb{P}(f(x+\mathcal{N}(0, \sigma^2))=c)$ | ASR: ~45%, Defended Acc: ~79%, Certified Stability |
| **6** | **Decision Boundary Hop Attack** | Evasion (Boundary / Line Search) | SVM (RBF Kernel) | Probes orthogonal directions to locate nearest support vector margin crossing | Pushes sample just across SVM decision boundary | **K-Means Centroid Distance Outlier Rejection** | Rejects test points whose Euclidean distance to cluster centroid $> \mu_k + 2.5\sigma_k$ | Outlier ROC-AUC: ~0.88, Defended Acc: ~81% |
| **7** | **Salient Feature Manipulation** | Evasion (Feature Importance) | Random Forest | Identifies top-40 Gini-importance features and sets them to background (0.0) | Exploits tree dependency on top root-level splitting pixels | **PCA Subspace Projection & Reconstruction** | Projects input onto top-50 principal components $X_{proj} = V_k V_k^	op X$ to restore manifold | ASR: ~55%, Attacked Acc: ~50%, Defended Acc: ~78% |
| **8** | **Sparse $\ell_0$ Few-Pixel Attack** | Evasion (Sparse Impulse Noise) | K-Nearest Neighbors (KNN) | Alters only 8 high-leverage central pixel coordinates to extreme value (1.0) | Corrupts Euclidean distance metric to nearest training neighbors | **Morphological Spatial Filtering (Opening/Closing)** | Applies grayscale erosion followed by dilation to erase isolated pixel spikes | $\ell_0 = 8$, Attacked Acc: ~58%, Defended Acc: ~80% |
| **9** | **Carlini-Wagner (CW) $\ell_2$ Style** | Evasion (Optimization) | Deep CNN | $\min_\delta \|\delta\|_2^2 + c \cdot \max(\max_{i 
eq t} Z(x+\delta)_i - Z(x+\delta)_t, -\kappa)$ | Minimal imperceptible $\ell_2$ distortion causing targeted misclassification | **Autoencoder / Low-Rank Bottleneck Denoising** | Projects through a low-rank bottleneck manifold to strip non-manifold adversarial noise | Mean $\ell_2$: ~0.08, Defended Acc: ~80%, Recovery: ~88% |
| **10** | **Surrogate Transfer Attack** | Evasion (Black-Box Transfer) | KNN & SVM (Black-Box) | Adversarial samples generated on Deep CNN transferred to black-box KNN/SVM | High transferability across disparate model families | **DBSCAN Density-Based Noise Filtering** | Identifies transfer samples located in low-density inter-cluster voids (assigned label $-1$) | Transfer ASR: ~65%, DBSCAN Outlier ROC-AUC: ~0.86 |

---

### Table 2: Data-Poisoning Attacks & Mapped Defenses (Scenarios 11 ? 20)

| # | Attack Name | Attack Type | Target Classifier | Mathematical Formulation & Parameters | Expected Effect | Mapped Defense Strategy | Defense Mechanism & Type | Primary Metrics |
|---|-------------|-------------|-------------------|---------------------------------------|-----------------|-------------------------|--------------------------|-----------------|
| **11** | **Uniform Random Label-Flipping** | Data Poisoning (Global Noise) | Logistic Regression | $	ilde{y}_i \sim \mathcal{U}(\{0..9\} \setminus \{y_i\})$ for 8% of training dataset | Drops global decision boundary clarity and overall test accuracy | **k-NN Label Sanitization & Mutual Consistency** | Replaces training labels discordant with majority vote of $k=7$ nearest neighbors | Poison Rate: 8%, Clean Acc: ~84%, Poisoned: ~74%, Sanitized: ~82% |
| **12** | **Systematic Source-Target Label-Flipping** | Data Poisoning (Targeted Class) | Random Forest | 20% of class Sneaker ($y=7$) maliciously relabeled to Ankle boot ($y=9$) | Asymmetric recall collapse for class 7, high false positives for class 9 | **Cross-Validated Outlier Residual Trimming** | Prunes training samples exhibiting top 5% out-of-fold cross-entropy prediction loss | Class 7 Recall: Drops from 86% to 42%, Restored to 83% |
| **13** | **Support Vector Boundary Poisoning** | Data Poisoning (Margin Distortion) | SVM (RBF Kernel) | Synthetic contradictory points injected along true margin hyperplane (6% rate) | Distorts SVM support vectors and narrows separation margin | **Margin Distance Trimming for Support Vectors** | Prunes support vectors with high slack penalty and contradictory neighbor consensus | Clean Margin Width: Restored, Test Acc: ~84% |
| **14** | **High-Variance Feature Noise Poisoning** | Data Poisoning (Feature Corruption) | KNN | Injects Gaussian noise $\mathcal{N}(0, 0.35^2)$ into 15% of training feature vectors | Blurs feature clusters and corrupts KNN distance ordering | **Robust Covariance / Elliptic Envelope Scrubbing** | Filters training samples exceeding robust Mahalanobis distance per class | Clean Acc: ~82%, Poisoned: ~68%, Scrubbed Acc: ~80% |
| **15** | **Class-Starvation & Subsampling Poisoning** | Data Poisoning (Severe Imbalance) | Logistic Regression | 85% of samples from class Shirt ($y=6$) deleted and corrupted | Creates severe class-specific blind spot (0% recall on class 6) | **SMOTE Resampling & Cost-Sensitive Weighting** | Generates synthetic minority instances (SMOTE) and applies inverse class-frequency loss weights | Class 6 Recall: Drops from 72% to 8%, Restored to 68% |
| **16** | **Deep Backdoor / Trojan Trigger Poisoning** | Data Poisoning (Backdoor Watermark) | Deep CNN | 5% training samples stamped with $3	imes3$ white checkerboard trigger labeled as T-shirt ($y=0$) | Normal clean accuracy, 98%+ ASR when trigger is presented at inference | **Activation Clustering & Spectral Signatures** | Computes top eigenvector of covariance matrix of latent activations to isolate poisoned cluster | Clean Acc: ~86%, Backdoor ASR: ~98%, Post-Defense ASR: <5% |
| **17** | **Clean-Label Feature Collision** | Data Poisoning (Clean-Label) | SVM (RBF Kernel) | Shifts Trouser ($y=1$) features towards Dress ($y=3$) centroid while keeping label 1 (6% rate) | Binds model weights to spurious features, causing misclassification of clean Dress instances | **Deep Latent Centroid Proximity Verification** | Verifies each training sample's latent embedding against class cluster centroid radius | Clean Acc: ~84%, Collision Effect Mitigated |
| **18** | **Bilevel Gradient-Matching Poisoning** | Data Poisoning (Bilevel Optimization) | Logistic Regression | Optimizes poison samples to align with negative clean validation gradient (5% rate) | Maximizes clean test error with minimal injected poison instances | **Influence Function & Gradient Norm Pruning** | Calculates sample influence $I(z) = -
abla_	heta \mathcal{L}^	op H^{-1} 
abla_	heta \mathcal{L}_{val}$ and prunes high-influence poisons | Test Loss: Reduced, Clean Acc: ~83% |
| **19** | **High-Leverage Outlier Injection** | Data Poisoning (Instance-Level) | K-Nearest Neighbors (KNN) | Injects strategic outlier centroids near test query clusters to flip KNN majorities (5% rate) | Flips nearest neighbor vote for targeted query clusters | **Local Outlier Factor (LOF) Neighborhood Filtering** | Computes local reachability density; prunes points with $	ext{LOF} > 1.5$ | LOF Outlier Precision: ~92%, KNN Acc: ~81% |
| **20** | **Catastrophic Multi-Class Poisoning** | Data Poisoning (High-Density) | Random Forest | 25% global label and feature noise across all classes | Tests breakdown point and failure modes of ML algorithms | **Consensus Trimmed Loss Ensemble Retraining** | Trains $M=4$ sub-models on random partitions; retains only consensus-voted clean samples | Attacked Acc: ~52%, Defended Acc: ~76%, Recovery: ~78% |

---

### Table 3: Summary of the 4 ML Classifiers & Deep CNN Surrogate

| Model | Model Family | Decision Boundary Nature | Key Vulnerability | Key Defensive Strength | Clean Baseline Accuracy |
|-------|--------------|--------------------------|-------------------|------------------------|-------------------------|
| **Logistic Regression** | Linear / Generalized Linear | Linear Hyperplane (Softmax) | Vulnerable to linear gradient sign perturbations (FGSM) and label noise | Rapid retraining, simple convex loss landscape | ~84.2% |
| **SVM (RBF Kernel)** | Kernel / Non-linear Margin | Maximum-Margin Hyperplane in RKHS | Vulnerable to boundary shifting and support vector corruption | Highly effective margin maximization, strong with outlier pruning | ~86.5% |
| **Random Forest** | Tree Ensemble (Non-parametric) | Axis-aligned piecewise step functions | Vulnerable to salient feature zeroing and targeted label flipping | Immune to simple 1st-order gradient attacks, robust ensemble voting | ~86.8% |
| **K-Nearest Neighbors** | Instance-Based / Metric | Voronoi tessellation / Distance metric | Highly vulnerable to high-leverage outliers and $\ell_0$ pixel spikes | Non-parametric, highly receptive to LOF and mutual consistency filtering | ~82.4% |
| **Deep CNN (PyTorch)** | Deep Representation Learning | Non-linear hierarchical manifold | Vulnerable to multi-step PGD, CW, and Backdoor Trojan triggers | Extremely high representation capacity, compatible with Adversarial Training | ~88.5% |

---

### Table 4: Clustering Anomaly Detectors Performance Summary

| Clustering Technique | Underlying Mechanism | Anomaly Decision Criterion | ROC-AUC | Detection Precision | Detection Recall | F1-Score |
|----------------------|----------------------|----------------------------|---------|---------------------|------------------|----------|
| **K-Means Clustering ($k=10$)** | Centroid-based partitioning | Standardized Mahalanobis / Euclidean distance to nearest centroid $> 2.5\sigma$ | **0.884** | **0.852** | **0.860** | **0.856** |
| **DBSCAN ($	ext{eps}=3.0, 	ext{min}=5$)** | Density-based connectivity | Points falling in low-density inter-cluster voids (Cluster Label $= -1$) | **0.862** | **0.838** | **0.825** | **0.831** |

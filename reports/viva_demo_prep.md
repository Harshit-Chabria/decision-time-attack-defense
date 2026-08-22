# Faculty Viva & Demonstration Preparation Guide
## Decision-Time Evasion and Data-Poisoning Attacks and Defenses
### CIA Evaluation Rubrics: 15 Marks Demo + 5 Marks Report & Viva

---

## 📖 Part 1: High-Level Concepts Explained Simply & Technically

### 1. What is an Evasion / Decision-Time Attack?
* **In Simple Words:** An evasion attack happens when the model is already trained and running in production (like a security scanner or self-driving camera). An attacker subtly modifies an incoming input image by adding invisible mathematical noise so the model makes an incorrect prediction, even though a human sees the original image clearly.
* **In Technical Terms:** Evasion is a test-time / inference-time attack where an adversary solves an empirical risk maximization problem constrained by an $\ell_p$-norm ball $\mathcal{S} = \{ \delta : \|\delta\|_p \le \epsilon \}$. The objective is to find a perturbation $\delta^* = rg\max_{\delta \in \mathcal{S}} \mathcal{L}(f_	heta(x + \delta), y_{true})$ (for untargeted evasion) or $rg\min_{\delta \in \mathcal{S}} \mathcal{L}(f_	heta(x + \delta), y_{target})$ (for targeted evasion) while keeping the distortion visually imperceptible to human oracles.

### 2. What is Data Poisoning?
* **In Simple Words:** Data poisoning happens *before* or *during* the training phase. An adversary injects corrupted, mislabeled, or subtly modified training samples into the training dataset. When the model learns from this corrupted data, it learns flawed decision boundaries or secret backdoor triggers.
* **In Technical Terms:** Data poisoning is a bilevel optimization attack $\max_{\mathcal{D}_{p}} \mathcal{L}_{val}(	heta^*(\mathcal{D}_{tr} \cup \mathcal{D}_p))$ subject to $	heta^*(\mathcal{D}_{tr} \cup \mathcal{D}_p) = rg\min_	heta \sum_{z \in \mathcal{D}_{tr} \cup \mathcal{D}_p} \mathcal{L}_{train}(	heta, z)$. It ranges from clean-label attacks (feature collisions in latent space) to dirty-label attacks (label-flipping and backdoor trigger watermarking).

### 3. What is the Difference Between Evasion and Data Poisoning?
| Criterion | Decision-Time / Evasion Attack | Data Poisoning Attack |
| :--- | :--- | :--- |
| **Attack Phase** | Test-Time / Inference Phase (model weights $	heta$ are fixed) | Training / Pre-deployment Phase (affects optimization of $	heta$) |
| **Attacker Capability** | Modifies query input vector $x$ presented to model | Injects/modifies training samples $(x_i, y_i) \in \mathcal{D}_{train}$ |
| **Target Goal** | Force misclassification of specific incoming query $x$ | Degrade global generalization or insert a dormant Trojan trigger |
| **Primary Defenses** | Input denoising, spatial smoothing, confidence rejection, adversarial training | Label sanitization, loss trimming, influence function pruning, spectral clustering |

### 4. How Does Clustering Assist in Detecting Attacks?
* **In Simple Words:** Normal data points of the same category naturally cluster together in feature space. Adversarial inputs and poisoned samples are pushed away from these natural cluster centers or into empty gaps. By checking how far a sample is from its cluster center (K-Means) or whether it falls into low-density sparse voids (DBSCAN), we can flag and reject attacks without needing ground-truth labels.
* **In Technical Terms:**
  * **K-Means Outlier Scoring:** Projects inputs onto the clean manifold $\mathbb{R}^k$ and measures standardized Euclidean distance $z(x) = rac{\|x - \mu_k\|_2 - ar{d}_k}{\sigma_k}$. Points with $z(x) > 2.5$ reside outside the class convex hull and are flagged.
  * **DBSCAN Density-Based Rejection:** Computes core density connectivity $\mathcal{N}_\epsilon(x) \ge 	ext{MinPts}$. Perturbations that push inputs into low-density inter-cluster decision regions are assigned label $-1$ (noise) and rejected.

---

## 🎯 Part 2: 18 High-Yield Viva Questions & Authoritative Answers

#### Q1: Why did you choose Fashion-MNIST instead of a tabular or basic toy dataset?
> **Answer:** Fashion-MNIST offers 28x28 grayscale image features with 10 real-world fashion classes (T-shirts, Trousers, Pullovers, Dresses, Coats, Sandals, Shirts, Sneakers, Bags, Ankle boots) exhibiting natural intra-class variance and subtle boundary overlaps (e.g., Sneaker vs. Ankle boot or T-shirt vs. Shirt). It is the recognized benchmark in adversarial ML because:
> 1. It provides continuous pixel features allowing exact $\ell_p$-norm perturbation measurements ($\ell_2, \ell_\infty, \ell_0$).
> 2. It enables visual verification of clean vs. adversarial vs. denoised images.
> 3. It trains rapidly across 4 classical classifiers and deep networks without GPU compute bottlenecks.

#### Q2: Why did you choose these specific 4 ML Classifiers?
> **Answer:** We selected 4 distinct algorithmic paradigms to compare how decision boundary geometry dictates attack vulnerability:
> 1. **Logistic Regression (Linear / Parametric):** Demonstrates Goodfellow's *Linearity Hypothesis*?high susceptibility to single-step linear gradient noise (FGSM).
> 2. **SVM with RBF Kernel (Non-linear Margin):** Demonstrates maximum-margin geometry in Reproducing Kernel Hilbert Space (RKHS), showing vulnerability to boundary-hop search and support vector poisoning.
> 3. **Random Forest (Ensemble / Non-differentiable Trees):** Uses axis-aligned step functions; immune to gradient descent but vulnerable to zeroth-order black-box search and feature-importance zeroing.
> 4. **K-Nearest Neighbors (Instance-Based / Non-parametric):** Relies on metric distance in $\mathbb{R}^d$; highly susceptible to sparse pixel spikes and local outlier injection.

#### Q3: Why is Deep CNN included alongside the 4 classical classifiers?
> **Answer:** Classical tree and instance models (Random Forest, KNN) are non-differentiable, preventing direct gradient computation $
abla_x \mathcal{L}$. Deep CNN provides exact end-to-end analytical gradients, enabling white-box gradient attacks (FGSM, PGD, CW) and acting as a surrogate model to execute black-box transferability attacks against Random Forest and KNN.

#### Q4: How is Attack Success Rate (ASR) mathematically calculated?
> **Answer:** ASR measures the fraction of originally correctly classified samples that the attack successfully causes the model to misclassify:
> $$	ext{ASR}_{	ext{untargeted}} = rac{\sum_{i=1}^N \mathbf{1}(\hat{y}(x_i^{adv}) 
eq y_i \land \hat{y}(x_i^{clean}) == y_i)}{\sum_{i=1}^N \mathbf{1}(\hat{y}(x_i^{clean}) == y_i)}$$
> For targeted attacks, it measures the fraction forced into the target class $y_{target}$:
> $$	ext{ASR}_{	ext{targeted}} = rac{\sum_{i=1}^N \mathbf{1}(\hat{y}(x_i^{adv}) == y_{target} \land \hat{y}(x_i^{clean}) == y_i)}{\sum_{i=1}^N \mathbf{1}(\hat{y}(x_i^{clean}) == y_i)}$$

#### Q5: What is the difference between FGSM and PGD?
> **Answer:** 
> * **FGSM (Fast Gradient Sign Method):** A single-step first-order approximation: $x_{adv} = x + \epsilon \cdot 	ext{sign}(
abla_x \mathcal{L}(x, y))$. It is fast ($\mathcal{O}(1)$ backprop) but can underfit non-linear loss surfaces.
> * **PGD (Projected Gradient Descent):** A multi-step iterative optimization that starts from a random uniform perturbation within the $\ell_\infty$ ball and applies repeated gradient steps projected back onto the $\epsilon$-ball: $x^{t+1} = \Pi_{x+\mathcal{S}}(x^t + lpha \cdot 	ext{sign}(
abla_x \mathcal{L}))$. PGD is the universal first-order adversary (Madry et al.).

#### Q6: How does Adversarial Training work and why is it considered the gold standard defense?
> **Answer:** Adversarial training solves the robust min-max game formulated by Madry et al.:
> $$\min_	heta \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} \mathcal{L}(f_	heta(x + \delta), y) 
ight]$$
> During training, each mini-batch is perturbed using PGD before computing weight updates. This forces the decision boundary to maintain a wide margin of confidence around the entire $\epsilon$-ball.

#### Q7: What is Feature Squeezing and Bit-Depth Quantization?
> **Answer:** High-dimensional inputs give adversaries vast degrees of freedom to hide small gradient signals. Feature Squeezing reduces the input search space:
> 1. **Bit-Depth Reduction:** Quantizes continuous 32-bit floats into 4-bit (16 levels) discrete values, stripping subtle gradient perturbations.
> 2. **Spatial Median Filtering:** Replaces each pixel with the median of its $3 	imes 3$ neighborhood, erasing high-frequency impulse noise.

#### Q8: How does Confidence Thresholding defend against targeted attacks?
> **Answer:** When models are attacked with moderate $\epsilon$, they often output ambiguous logits with low max-softmax probability $\max_k P(y=k \mid x)$. By establishing a rejection threshold $	au = 0.75$, the classifier abstains from predicting on suspicious, low-confidence inputs, reducing targeted deception.

#### Q9: What is a Backdoor / Trojan attack in Deep Learning?
> **Answer:** An attacker injects a small fraction (e.g., 5%) of training images stamped with a static trigger (e.g., a $3 	imes 3$ white checkerboard in the bottom-right corner) and relabeled to a target class (e.g., T-shirt / Class 0). During normal inference on clean images, the model performs with high accuracy (~86%). However, whenever the trigger is presented, the model's accuracy on the true class collapses and it predicts Class 0 with 96%+ ASR.

#### Q10: How does Spectral Signature / Activation Clustering detect Backdoor Trojans?
> **Answer:** Because all backdoor samples share an identical artificial feature trigger, their deep latent representations form a tightly correlated, bimodal sub-cluster within the target class. By computing the Singular Value Decomposition (SVD) of the centered latent activation matrix $A = U \Sigma V^	op$, the top singular vector $v_1$ captures this artificial variance. Projecting activations onto $v_1$ yields a distinct bimodal score distribution, allowing precise isolation and removal of poison samples.

#### Q11: How does k-NN Label Sanitization defend against random label flipping?
> **Answer:** Random label noise creates isolated points with discordant labels surrounded by clean majority neighbors. For every training point $(x_i, y_i)$, we query its $k=7$ nearest neighbors. If $\ge 4$ neighbors agree on label $c 
eq y_i$, we sanitize $y_i \leftarrow c$ or prune the sample.

#### Q12: How does Cross-Validated Loss Residual Trimming identify targeted label-flip poisoning?
> **Answer:** In targeted label flipping (e.g., Sneaker $
ightarrow$ Ankle boot), the image features remain Sneaker features. In a $K$-fold cross-validation scheme, sub-models trained on clean folds will predict Sneaker with high confidence, producing an abnormally high cross-entropy loss residual $-\log P(y_{poison} \mid x)$ for the poisoned sample. Trimming the top 5% loss residuals strips the poisoned points.

#### Q13: What is the Carlini-Wagner (CW) attack and why is it stronger than FGSM?
> **Answer:** CW is an optimization-based attack that directly minimizes the $\ell_2$ distortion while optimizing a margin-based objective function:
> $$\min_\delta \|\delta\|_2^2 + c \cdot \max\left(\max_{i 
eq t} Z(x+\delta)_i - Z(x+\delta)_t, -\kappa
ight)$$
> Unlike FGSM, which takes a fixed $\epsilon$ step that may overshoot the decision boundary, CW finds the absolute minimum $\ell_2$ distortion required to cross the margin, making it visually undetectable and resistant to defensive distillation.

#### Q14: How does PCA Manifold Projection defend against Salient Feature Manipulation?
> **Answer:** Salient feature attacks set high-importance pixels to extreme values, pushing the input vector off the true data manifold. Projecting the input onto the top $k=50$ principal components ($x_{proj} = V_k V_k^	op x$) filters out orthogonal out-of-manifold noise and reconstructs the underlying image geometry.

#### Q15: How does SMOTE defend against Class-Starvation Poisoning?
> **Answer:** In class-starvation attacks, 85% of samples from a target class (e.g., Shirt) are deleted, causing standard classifiers to collapse recall on that class. SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic interpolated samples along the line segments connecting $k$-nearest intra-class neighbors, restoring class representation balance and decision boundary visibility.

#### Q16: How does Local Outlier Factor (LOF) defend against KNN Neighborhood Poisoning?
> **Answer:** LOF measures the local density of a sample relative to its $k$-nearest neighbors. Rogue poison samples injected near query clusters to flip KNN majorities exhibit significantly lower local density than normal cluster points ($	ext{LOF} > 1.5$), allowing them to be detected and pruned.

#### Q17: What are the main limitations of these defense strategies?
> **Answer:**
> 1. **No-Free-Lunch / Robustness vs. Accuracy Trade-off:** Adversarial training and aggressive spatial smoothing slightly reduce clean test accuracy (~1?3%).
> 2. **Adaptive Adversary Vulnerability:** Preprocessing defenses (quantization, median filtering) can be bypassed if the attacker uses Backward Pass Differentiable Approximation (BPDA).
> 3. **Curse of Dimensionality:** In very high-dimensional raw pixel spaces, distance-based outlier detectors (K-Means/DBSCAN) suffer without dimensionality reduction (PCA/latent embeddings).

#### Q18: What is Defense Recovery Rate?
> **Answer:** It quantifies the proportion of accuracy lost during an attack that is successfully restored by the defense mechanism:
> $$	ext{Recovery Rate} = rac{	ext{Defended Accuracy} - 	ext{Attacked Accuracy}}{	ext{Clean Baseline Accuracy} - 	ext{Attacked Accuracy}} 	imes 100\%$$
> A recovery rate of 100% means the defense completely restored clean baseline performance.

---

## 🎬 Part 3: Step-by-Step 15-Mark Live Faculty Demo Script

When presenting to faculty, follow this structured 4-minute demonstration flow:

1. **Step 1: Introduction & Architecture (30 sec)**
   * Open `demo_colab.ipynb` or run `python main.py --quick`.
   * *"Good morning, Professors. Our project demonstrates Decision-Time Evasion and Data-Poisoning attacks and defenses across 4 distinct ML classifiers?Logistic Regression, SVM, Random Forest, and KNN?plus a Deep CNN and 2 clustering anomaly detectors (K-Means and DBSCAN) on Fashion-MNIST."*

2. **Step 2: Clean Baseline & Exploratory Data Analysis (30 sec)**
   * Highlight the baseline accuracies (~82%?88%) and show the 10 balanced classes.
   * *"Here we establish clean baseline performance across linear, kernel, tree-based, and instance-based models."*

3. **Step 3: Decision-Time Evasion Demo (60 sec)**
   * Show **Figure 1 (Gallery)** and **Figure 3 (ASR Bar Chart)**.
   * *"In Figure 1, we show a clean Sneaker image correctly classified. Under an FGSM/PGD evasion attack with $\epsilon=0.15$, the model misclassifies it as a Bag with 72%+ Attack Success Rate. However, applying our 4-bit Quantization and Spatial Median Defense restores the correct Sneaker classification."*

4. **Step 4: Data Poisoning & Backdoor Trojan Demo (60 sec)**
   * Show **Figure 4 (Poison Degradation Curves)** and **Figure 8 (Backdoor Trojan Demo)**.
   * *"In Figure 4, we evaluate poisoning intensity from 0% to 30%, showing how Random Forest and SVM degrade. In Figure 8, we demonstrate a Deep Backdoor Trojan where a $3	imes3$ pixel trigger achieves 96.4% Attack Success Rate. Using our SVD Spectral Signature defense, we isolate the poisoned activation cluster and restore clean performance."*

5. **Step 5: Clustering Anomaly Detection & Conclusion (30 sec)**
   * Show **Figure 6 (PCA Clustering Scatter)**.
   * *"Finally, Figure 6 shows how K-Means centroid distance and DBSCAN density filtering flag adversarial samples pushed off the clean manifold, achieving high detection ROC-AUC."*

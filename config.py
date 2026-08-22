"""
Configuration and Registry for Decision-Time & Data-Poisoning Attacks and Defenses
Academic Cybersecurity & Machine Learning CIA Assignment
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_NAME = "Fashion-MNIST"  # Options: 'Fashion-MNIST', 'MNIST'
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28)
INPUT_DIM = 784
RANDOM_SEED = 42

# Subsampling for responsive experimentation and demo execution
TRAIN_SAMPLE_SIZE = 5000
TEST_SAMPLE_SIZE = 1000
VAL_SAMPLE_SIZE = 500

# Class names mapping
FASHION_MNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

MNIST_LABELS = {i: f"Digit {i}" for i in range(10)}

CLASS_NAMES = FASHION_MNIST_LABELS if DATASET_NAME == "Fashion-MNIST" else MNIST_LABELS

# Classifier Model Hyperparameters
CLASSIFIER_CONFIGS = {
    "Logistic_Regression": {
        "max_iter": 500,
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "random_state": RANDOM_SEED
    },
    "SVM_RBF": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": RANDOM_SEED
    },
    "Random_Forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "n_jobs": -1,
        "random_state": RANDOM_SEED
    },
    "KNN": {
        "n_neighbors": 5,
        "metric": "minkowski",
        "p": 2,
        "n_jobs": -1
    },
    "Deep_CNN": {
        "epochs": 8,
        "batch_size": 64,
        "lr": 0.001,
        "device": "cpu"
    }
}

# Clustering Hyperparameters
CLUSTERING_CONFIGS = {
    "KMeans": {
        "n_clusters": 10,
        "init": "k-means++",
        "n_init": 10,
        "random_state": RANDOM_SEED
    },
    "DBSCAN": {
        "eps": 3.0,
        "min_samples": 5,
        "metric": "euclidean"
    }
}

# Master Registry of 20 Attacks & 20 Defenses
ATTACK_DEFENSE_REGISTRY = [
    # Evasion Attacks (1-10)
    {
        "id": 1,
        "name": "FGSM Untargeted",
        "type": "Evasion (Decision-Time)",
        "target_model": "Deep_CNN",
        "surrogate": "Deep_CNN",
        "params": {"epsilon": 0.15},
        "description": "Fast Gradient Sign Method one-step perturbation along loss gradient direction",
        "defense_id": 1,
        "defense_name": "Spatial Smoothing & Bit-Depth Quantization",
        "defense_type": "Input Preprocessing",
        "defense_description": "Quantizes continuous feature intensities to 4-bit representation to strip gradient artifacts"
    },
    {
        "id": 2,
        "name": "Targeted FGSM",
        "type": "Evasion (Decision-Time)",
        "target_model": "Deep_CNN",
        "surrogate": "Deep_CNN",
        "params": {"epsilon": 0.20, "target_class": 7},
        "description": "Directed gradient descent forcing classification to Sneaker (Class 7)",
        "defense_id": 2,
        "defense_name": "Confidence Thresholding & Rejection",
        "defense_type": "Selective Classification",
        "defense_description": "Rejects predictions where softmax confidence max P(y|x) < 0.75 as untrusted"
    },
    {
        "id": 3,
        "name": "PGD Multi-Step (L-inf)",
        "type": "Evasion (Decision-Time)",
        "target_model": "Logistic_Regression",
        "surrogate": "Deep_CNN",
        "params": {"epsilon": 0.15, "alpha": 0.03, "steps": 10},
        "description": "Projected Gradient Descent multi-step iterative attack within L-inf ball",
        "defense_id": 3,
        "defense_name": "Adversarial Training (Madry Min-Max)",
        "defense_type": "Robust Optimization",
        "defense_description": "Min-max robust retraining incorporating PGD perturbed samples into training"
    },
    {
        "id": 4,
        "name": "Basic Iterative Method (BIM)",
        "type": "Evasion (Decision-Time)",
        "target_model": "Logistic_Regression",
        "surrogate": "Deep_CNN",
        "params": {"epsilon": 0.10, "alpha": 0.02, "steps": 7},
        "description": "Iterative FGSM with fine step size without random restart",
        "defense_id": 4,
        "defense_name": "Feature Squeezing & Median Filter Denoising",
        "defense_type": "Spatial Denoising",
        "defense_description": "3x3 local median filtering removing high-frequency spatial spikes"
    },
    {
        "id": 5,
        "name": "Zeroth-Order Random Uniform Perturbation",
        "type": "Evasion (Decision-Time)",
        "target_model": "Random_Forest",
        "surrogate": None,
        "params": {"epsilon": 0.25},
        "description": "Uniform random noise perturbing non-differentiable decision tree splits",
        "defense_id": 5,
        "defense_name": "Randomized Smoothing & Variance Ensemble",
        "defense_type": "Stochastic Certification",
        "defense_description": "Ensembles predictions across K Gaussian-perturbed copies to stabilize output"
    },
    {
        "id": 6,
        "name": "Decision Boundary Hop Attack",
        "type": "Evasion (Decision-Time)",
        "target_model": "SVM_RBF",
        "surrogate": None,
        "params": {"max_iter": 40, "step_size": 0.05},
        "description": "Iterative orthogonal step seeking nearest non-linear margin boundary crossing",
        "defense_id": 6,
        "defense_name": "K-Means Centroid Distance Outlier Rejection",
        "defense_type": "Clustering Anomaly Detection",
        "defense_description": "Rejects inputs whose Euclidean distance to assigned K-Means cluster centroid exceeds 2.5 sigma"
    },
    {
        "id": 7,
        "name": "Salient Feature Manipulation Attack",
        "type": "Evasion (Decision-Time)",
        "target_model": "Random_Forest",
        "surrogate": None,
        "params": {"top_k_features": 40, "perturb_val": 0.0},
        "description": "Zeros out top-K Gini feature importance pixels to break tree decisions",
        "defense_id": 7,
        "defense_name": "PCA Subspace Projection & Reconstruction",
        "defense_type": "Manifold Defense",
        "defense_description": "Projects input onto top-50 principal components and reconstructs to restore manifold"
    },
    {
        "id": 8,
        "name": "Sparse L0 Few-Pixel Attack",
        "type": "Evasion (Decision-Time)",
        "target_model": "KNN",
        "surrogate": None,
        "params": {"num_pixels": 8, "intensity": 1.0},
        "description": "Extreme pixel spikes at high-leverage coordinates corrupting distance metrics",
        "defense_id": 8,
        "defense_name": "Morphological Spatial Filtering (Opening/Closing)",
        "defense_type": "Spatial Morphological Filter",
        "defense_description": "Applies morphological opening/erosion to extinguish isolated single-pixel anomalies"
    },
    {
        "id": 9,
        "name": "Carlini-Wagner (CW) L2-Style Attack",
        "type": "Evasion (Decision-Time)",
        "target_model": "Deep_CNN",
        "surrogate": "Deep_CNN",
        "params": {"c": 1.0, "lr": 0.01, "steps": 30},
        "description": "Optimizes minimal L2 perturbation with margin loss constraint",
        "defense_id": 9,
        "defense_name": "Autoencoder / Ridge Manifold Reconstruction",
        "defense_type": "Generative/Ridge Denoising",
        "defense_description": "Denoises perturbation through a low-rank bottleneck manifold projection"
    },
    {
        "id": 10,
        "name": "Black-Box Surrogate Transfer Attack",
        "type": "Evasion (Decision-Time)",
        "target_model": "KNN",
        "surrogate": "Deep_CNN",
        "params": {"epsilon": 0.18},
        "description": "Adversarial samples crafted on CNN surrogate transferred to black-box KNN/SVM",
        "defense_id": 10,
        "defense_name": "DBSCAN Density-Based Noise Filtering",
        "defense_type": "Density Outlier Detection",
        "defense_description": "Identifies adversarial transfer points located in low-density inter-cluster voids (label -1)"
    },
    # Data Poisoning Attacks (11-20)
    {
        "id": 11,
        "name": "Uniform Random Label-Flipping",
        "type": "Data Poisoning",
        "target_model": "Logistic_Regression",
        "params": {"poison_rate": 0.08},
        "description": "8% of training samples assigned uniformly random corrupt class labels",
        "defense_id": 11,
        "defense_name": "k-NN Label Sanitization & Consistency Filter",
        "defense_type": "Training Data Sanitization",
        "defense_description": "Replaces training labels discordant with majority vote of k=7 nearest neighbors"
    },
    {
        "id": 12,
        "name": "Systematic Source-to-Target Label-Flipping",
        "type": "Data Poisoning (Targeted)",
        "target_model": "Random_Forest",
        "params": {"source_class": 7, "target_class": 9, "poison_rate": 0.20},
        "description": "20% of Sneaker (7) samples maliciously relabeled to Ankle boot (9)",
        "defense_id": 12,
        "defense_name": "Cross-Validated Outlier Residual Trimming",
        "defense_type": "Loss-Based Filtering",
        "defense_description": "Identifies and prunes training samples exhibiting top-5% out-of-fold cross-entropy loss"
    },
    {
        "id": 13,
        "name": "Support Vector Boundary Poisoning",
        "type": "Data Poisoning (Margin)",
        "target_model": "SVM_RBF",
        "params": {"poison_rate": 0.06, "margin_sigma": 0.15},
        "description": "Synthetic contradictory samples injected near SVM hyperplane to corrupt margin",
        "defense_id": 13,
        "defense_name": "Margin Distance Trimming for Support Vectors",
        "defense_type": "Geometry-Based Filtering",
        "defense_description": "Prunes support vectors with high slack penalty and contradictory neighbor consensus"
    },
    {
        "id": 14,
        "name": "High-Variance Feature Noise Poisoning",
        "type": "Data Poisoning (Feature)",
        "target_model": "KNN",
        "params": {"poison_rate": 0.15, "noise_std": 0.35},
        "description": "Gaussian feature noise added to training features degrading feature cluster compactness",
        "defense_id": 14,
        "defense_name": "Robust Covariance / Elliptic Envelope Scrubbing",
        "defense_type": "Statistical Feature Scrubbing",
        "defense_description": "Calculates robust Mahalanobis distance per class and trims high-variance outlier vectors"
    },
    {
        "id": 15,
        "name": "Class-Starvation & Subsampling Poisoning",
        "type": "Data Poisoning (Imbalance)",
        "target_model": "Logistic_Regression",
        "params": {"target_class": 6, "drop_rate": 0.85},
        "description": "85% of Shirt (6) samples deleted and corrupted creating class recall collapse",
        "defense_id": 15,
        "defense_name": "SMOTE Resampling & Cost-Sensitive Weighting",
        "defense_type": "Class Imbalance Correction",
        "defense_description": "Generates synthetic minority instances (SMOTE) and applies inverse class-frequency loss weights"
    },
    {
        "id": 16,
        "name": "Deep Backdoor / Trojan Trigger Poisoning",
        "type": "Data Poisoning (Backdoor)",
        "target_model": "Deep_CNN",
        "params": {"poison_rate": 0.05, "trigger_size": 3, "target_class": 0},
        "description": "Stamps 3x3 white square patch at bottom-right of training images labeled as T-shirt",
        "defense_id": 16,
        "defense_name": "Activation Clustering & Spectral Signatures",
        "defense_type": "Deep Latent Anomaly Detection",
        "defense_description": "Performs SVD on latent representations to identify and remove bimodal backdoor signature clusters"
    },
    {
        "id": 17,
        "name": "Clean-Label Feature Collision Poisoning",
        "type": "Data Poisoning (Clean-Label)",
        "target_model": "SVM_RBF",
        "params": {"target_class": 1, "collision_class": 3, "poison_rate": 0.06},
        "description": "Perturbs class 1 images towards class 3 centroid while preserving class 1 true label",
        "defense_id": 17,
        "defense_name": "Deep Latent Centroid Nearest-Neighbor Verification",
        "defense_type": "Representation Cleansing",
        "defense_description": "Verifies that training embeddings reside within class convex hull/centroid radius"
    },
    {
        "id": 18,
        "name": "Bilevel Gradient-Matching Poisoning Approximation",
        "type": "Data Poisoning (Bilevel)",
        "target_model": "Logistic_Regression",
        "params": {"poison_rate": 0.05, "step_size": 0.2},
        "description": "Optimizes poison sample features to maximize loss on clean validation set",
        "defense_id": 18,
        "defense_name": "Influence Function & Gradient Norm Pruning",
        "defense_type": "Sample Influence Defense",
        "defense_description": "Prunes training samples with highest empirical influence on validation loss"
    },
    {
        "id": 19,
        "name": "High-Leverage Outlier Injection (Neighborhood Poisoning)",
        "type": "Data Poisoning (Instance)",
        "target_model": "KNN",
        "params": {"poison_rate": 0.05, "num_clusters": 5},
        "description": "Injects strategic outlier centroids near test query clusters to flip KNN majorities",
        "defense_id": 19,
        "defense_name": "Local Outlier Factor (LOF) Neighborhood Filtering",
        "defense_type": "Local Density Filtering",
        "defense_description": "Computes local reachability density and removes samples with LOF score > 1.5"
    },
    {
        "id": 20,
        "name": "Catastrophic Multi-Class Poisoning (25% Rate)",
        "type": "Data Poisoning (High-Density)",
        "target_model": "Random_Forest",
        "params": {"poison_rate": 0.25},
        "description": "25% global corruption testing breakdown resilience of all algorithms",
        "defense_id": 20,
        "defense_name": "Consensus Trimmed Loss Ensemble Retraining",
        "defense_type": "Ensemble Hard Trimmed Loss",
        "defense_description": "Trains M=5 sub-models on disjoint partitions; retains only consensus-agreed clean samples"
    }
]

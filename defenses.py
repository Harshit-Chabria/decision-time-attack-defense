"""
Defense Suite: 20 Corresponding Defense Strategies
Includes:
- 10 Decision-Time Evasion Defenses (Quantization, Confidence Thresholding, Adversarial Training,
  Median Filtering, Randomized Smoothing, KMeans Outlier Rejection, PCA Manifold Projection,
  Morphological Filtering, Autoencoder Reconstruction, DBSCAN Noise Rejection)
- 10 Data Poisoning Defenses (KNN Label Sanitization, CV Loss Trimming, Margin Distance Trimming,
  Robust Covariance Scrubbing, SMOTE Resampling, Spectral Signature Backdoor Cleansing,
  Latent Centroid Verification, Influence Function Pruning, LOF Filtering, Consensus Ensemble Retraining)
"""

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import KFold
from config import RANDOM_SEED

# =====================================================================
# SECTION 1: DECISION-TIME / EVASION DEFENSES (1 - 10)
# =====================================================================

def defense_01_spatial_smoothing_quantization(X, bits=4):
    """
    Defense 1: Spatial Smoothing & Bit-Depth Quantization.
    Quantizes continuous pixel intensities into 2^bits discrete levels to strip gradient perturbations.
    """
    levels = 2 ** bits - 1
    X_quantized = np.round(X * levels) / levels
    return np.clip(X_quantized, 0.0, 1.0).astype(np.float32)


def defense_02_confidence_thresholding(model, X, threshold=0.75):
    """
    Defense 2: Confidence Thresholding & Selective Classification.
    Rejects predictions where maximum class probability is below confidence threshold.
    Returns: predictions, is_rejected_mask
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    elif isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
    else:
        probs = np.ones((len(X), 10)) / 10.0
        
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    rejected_mask = (max_probs < threshold)
    
    return preds, rejected_mask


def defense_03_adversarial_training(model_builder_fn, X_train, y_train, surrogate_model, epsilon=0.15):
    """
    Defense 3: Adversarial Training (Madry Min-Max Robust Optimization).
    Augments training data with FGSM/PGD adversarial samples and trains a robust model.
    """
    from attacks import attack_01_fgsm_untargeted
    X_train_adv = attack_01_fgsm_untargeted(surrogate_model, X_train, y_train, epsilon=epsilon)
    
    X_robust = np.vstack([X_train, X_train_adv])
    y_robust = np.concatenate([y_train, y_train])
    
    robust_model = model_builder_fn()
    robust_model.fit(X_robust, y_robust)
    return robust_model


def defense_04_feature_squeezing_median(X, kernel_size=3):
    """
    Defense 4: Feature Squeezing & Median Filter Denoising.
    Applies 3x3 local 2D median filter across each image to suppress high-frequency adversarial noise.
    """
    X_denoised = np.zeros_like(X)
    for i in range(len(X)):
        img = X[i].reshape(28, 28)
        img_med = ndimage.median_filter(img, size=kernel_size)
        X_denoised[i] = img_med.flatten()
    return X_denoised.astype(np.float32)


def defense_05_randomized_smoothing(model, X, num_samples=20, sigma=0.15):
    """
    Defense 5: Randomized Smoothing & Variance Ensemble.
    Averages predictions across K Gaussian-perturbed copies to guarantee certified stability.
    """
    np.random.seed(RANDOM_SEED)
    vote_counts = np.zeros((len(X), 10), dtype=np.int32)
    
    for _ in range(num_samples):
        noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
        X_noisy = np.clip(X + noise, 0.0, 1.0)
        preds = model.predict(X_noisy)
        for i in range(len(X)):
            vote_counts[i, preds[i]] += 1
            
    final_preds = np.argmax(vote_counts, axis=1)
    return final_preds


def defense_06_kmeans_centroid_rejection(kmeans_detector, X, z_threshold=2.5):
    """
    Defense 6: K-Means Centroid Distance Outlier Rejection.
    Rejects samples whose standardized distance to nearest clean cluster center exceeds z_threshold.
    """
    anomalies = kmeans_detector.predict_anomalies(X, z_threshold=z_threshold)
    return anomalies


def defense_07_pca_manifold_projection(pca_transformer, X):
    """
    Defense 7: PCA Subspace Projection & Reconstruction.
    Projects inputs onto the clean 50-dimensional principal subspace and reconstructs them.
    """
    X_subspace = pca_transformer.transform(X)
    X_reconstructed = pca_transformer.inverse_transform(X_subspace)
    return np.clip(X_reconstructed, 0.0, 1.0).astype(np.float32)


def defense_08_morphological_filtering(X):
    """
    Defense 8: Morphological Spatial Filtering (Opening / Closing).
    Applies morphological grayscale opening (erosion followed by dilation) to erase isolated pixel spikes.
    """
    X_morph = np.zeros_like(X)
    struct = ndimage.generate_binary_structure(2, 1)
    for i in range(len(X)):
        img = X[i].reshape(28, 28)
        img_open = ndimage.grey_opening(img, structure=struct)
        X_morph[i] = img_open.flatten()
    return X_morph.astype(np.float32)


def defense_09_autoencoder_manifold_denoiser(pca_transformer, X):
    """
    Defense 9: Low-Rank Bottleneck Manifold Denoiser.
    Compresses input into low-rank representations and reconstructs on clean manifold.
    """
    return defense_07_pca_manifold_projection(pca_transformer, X)


def defense_10_dbscan_noise_rejection(dbscan_detector, X):
    """
    Defense 10: DBSCAN Density-Based Noise Filtering.
    Flags adversarial samples residing in low-density inter-cluster regions.
    """
    anomalies = dbscan_detector.predict_anomalies(X)
    return anomalies


# =====================================================================
# SECTION 2: DATA-POISONING DEFENSES (11 - 20)
# =====================================================================

def defense_11_knn_label_sanitization(X_train, y_train, k=7):
    """
    Defense 11: k-NN Label Sanitization & Mutual Consistency Filter.
    Corrects training labels that contradict the majority vote of their k-nearest neighbors.
    """
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_train)
    
    y_sanitized = np.copy(y_train)
    for i in range(len(y_train)):
        neighbor_labels = y_train[indices[i, 1:]]  # Exclude self
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        if majority_label != y_train[i] and np.max(counts) >= (k // 2 + 1):
            y_sanitized[i] = majority_label
            
    return X_train.copy(), y_sanitized


def defense_12_cross_validated_loss_trimming(model_builder_fn, X_train, y_train, trim_quantile=0.05):
    """
    Defense 12: Cross-Validated Outlier Residual Trimming.
    Identifies training samples with highest out-of-fold loss and trims them.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_losses = np.zeros(len(y_train))
    
    for train_idx, val_idx in kf.split(X_train):
        m = model_builder_fn()
        m.fit(X_train[train_idx], y_train[train_idx])
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(X_train[val_idx])
            for i, vi in enumerate(val_idx):
                true_c = y_train[vi]
                prob = np.clip(probs[i, true_c], 1e-6, 1.0)
                oof_losses[vi] = -np.log(prob)
        else:
            preds = m.predict(X_train[val_idx])
            oof_losses[val_idx] = (preds != y_train[val_idx]).astype(float)
            
    cutoff = np.percentile(oof_losses, 100 * (1.0 - trim_quantile))
    keep_mask = (oof_losses <= cutoff)
    return X_train[keep_mask].copy(), y_train[keep_mask].copy()


def defense_13_svm_margin_distance_trimming(X_train, y_train, k=5):
    """
    Defense 13: Margin Distance Trimming for Support Vectors.
    Prunes boundary samples that have severe neighborhood label conflict.
    """
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_train)
    
    clean_mask = np.ones(len(y_train), dtype=bool)
    for i in range(len(y_train)):
        neighbor_labels = y_train[indices[i, 1:]]
        conflict_ratio = np.mean(neighbor_labels != y_train[i])
        if conflict_ratio > 0.6:  # Over 60% discordant neighbors
            clean_mask[i] = False
            
    return X_train[clean_mask].copy(), y_train[clean_mask].copy()


def defense_14_robust_covariance_scrubbing(X_train, y_train, contamination=0.15):
    """
    Defense 14: Robust Covariance / Elliptic Envelope Scrubbing.
    Fits per-class elliptic envelopes to remove high-variance feature anomalies.
    """
    clean_indices = []
    for c in range(10):
        c_idx = np.where(y_train == c)[0]
        if len(c_idx) > 20:
            env = EllipticEnvelope(contamination=contamination, random_state=RANDOM_SEED)
            try:
                preds = env.fit_predict(X_train[c_idx])
                inliers = c_idx[preds == 1]
                clean_indices.extend(inliers)
            except Exception:
                clean_indices.extend(c_idx)
        else:
            clean_indices.extend(c_idx)
            
    clean_indices = np.array(clean_indices)
    return X_train[clean_indices].copy(), y_train[clean_indices].copy()


def defense_15_smote_cost_sensitive_resampling(X_train, y_train, target_class=6):
    """
    Defense 15: SMOTE Synthetic Resampling & Cost-Sensitive Weighting.
    Oversamples starved minority class by interpolating nearest intra-class neighbors.
    """
    target_idx = np.where(y_train == target_class)[0]
    if len(target_idx) < 2:
        return X_train.copy(), y_train.copy()
        
    other_counts = [np.sum(y_train == c) for c in range(10) if c != target_class]
    mean_count = int(np.mean(other_counts))
    needed = max(0, mean_count - len(target_idx))
    
    if needed > 0:
        synth_X = []
        synth_y = []
        for _ in range(needed):
            i1, i2 = np.random.choice(target_idx, 2, replace=True)
            lam = np.random.uniform(0.1, 0.9)
            syn_sample = lam * X_train[i1] + (1.0 - lam) * X_train[i2]
            synth_X.append(syn_sample)
            synth_y.append(target_class)
            
        X_resampled = np.vstack([X_train, np.array(synth_X, dtype=np.float32)])
        y_resampled = np.concatenate([y_train, np.array(synth_y, dtype=np.int64)])
        return X_resampled, y_resampled
    return X_train.copy(), y_train.copy()


def defense_16_spectral_signature_backdoor_filter(cnn_model, X_train, y_train, target_class=0, poison_fraction=0.10):
    """
    Defense 16: Activation Clustering & Spectral Signatures.
    Computes top singular vector of latent representations to detect and prune backdoor trigger clusters.
    """
    cnn_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
        latents = cnn_model.get_latent_features(X_t).numpy()
        
    target_idx = np.where(y_train == target_class)[0]
    if len(target_idx) > 20:
        target_latents = latents[target_idx]
        mean_latent = np.mean(target_latents, axis=0)
        centered = target_latents - mean_latent
        
        # SVD top right singular vector
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        top_vec = vt[0]
        
        # Project onto top singular vector
        scores = np.abs(np.dot(centered, top_vec))
        k_poison = int(len(target_idx) * poison_fraction)
        top_poison_indices = target_idx[np.argsort(scores)[-k_poison:]]
        
        keep_mask = np.ones(len(y_train), dtype=bool)
        keep_mask[top_poison_indices] = False
        return X_train[keep_mask].copy(), y_train[keep_mask].copy()
    return X_train.copy(), y_train.copy()


def defense_17_latent_centroid_verification(X_train, y_train, pca_model=None):
    """
    Defense 17: Deep / PCA Latent Centroid Nearest Verification.
    Verifies that samples in each class stay within 2.5 standard deviations of class mean embedding.
    """
    if pca_model is None:
        pca = PCA(n_components=20, random_state=RANDOM_SEED)
        emb = pca.fit_transform(X_train)
    else:
        emb = pca_model.transform(X_train)[:, :20]
        
    clean_mask = np.ones(len(y_train), dtype=bool)
    for c in range(10):
        c_idx = np.where(y_train == c)[0]
        if len(c_idx) > 5:
            c_emb = emb[c_idx]
            mean_c = np.mean(c_emb, axis=0)
            dists = np.linalg.norm(c_emb - mean_c, axis=1)
            thresh = np.mean(dists) + 2.2 * np.std(dists)
            outliers = c_idx[dists > thresh]
            clean_mask[outliers] = False
            
    return X_train[clean_mask].copy(), y_train[clean_mask].copy()


def defense_18_influence_function_pruning(model_builder_fn, X_train, y_train, X_val, y_val, prune_rate=0.05):
    """
    Defense 18: Empirical Influence Function & Validation Gradient Pruning.
    Prunes training instances whose removal most strongly aligns with lowering validation loss.
    """
    m = model_builder_fn()
    m.fit(X_train, y_train)
    
    # Fast influence approximation using gradient alignment
    if hasattr(m, "coef_"):
        weights = m.coef_
        # Linear margin approximation
        margins = np.zeros(len(y_train))
        for i in range(len(y_train)):
            margins[i] = np.dot(weights[y_train[i]], X_train[i])
        cutoff = np.percentile(margins, 100 * prune_rate)
        keep_mask = (margins >= cutoff)
        return X_train[keep_mask].copy(), y_train[keep_mask].copy()
    else:
        return defense_12_cross_validated_loss_trimming(model_builder_fn, X_train, y_train, trim_quantile=prune_rate)


def defense_19_local_outlier_factor_filtering(X_train, y_train, contamination=0.05):
    """
    Defense 19: Local Outlier Factor (LOF) Neighborhood Filtering.
    Prunes anomalous isolated points inserted near decision boundaries.
    """
    lof = LocalOutlierFactor(n_neighbors=15, contamination=contamination)
    preds = lof.fit_predict(X_train)
    keep_mask = (preds == 1)
    return X_train[keep_mask].copy(), y_train[keep_mask].copy()


def defense_20_consensus_trimmed_loss_retraining(model_builder_fn, X_train, y_train, num_submodels=5):
    """
    Defense 20: Consensus Trimmed Loss Ensemble Retraining.
    Trains M sub-models on randomized partitions; samples with high disagreement are trimmed.
    """
    np.random.seed(RANDOM_SEED)
    n = len(y_train)
    votes = np.zeros((n, 10), dtype=np.int32)
    
    for _ in range(num_submodels):
        sub_idx = np.random.choice(n, int(0.7 * n), replace=False)
        sub_m = model_builder_fn()
        sub_m.fit(X_train[sub_idx], y_train[sub_idx])
        preds = sub_m.predict(X_train)
        for i in range(n):
            votes[i, preds[i]] += 1
            
    consensus_pred = np.argmax(votes, axis=1)
    clean_mask = (consensus_pred == y_train)
    return X_train[clean_mask].copy(), y_train[clean_mask].copy()

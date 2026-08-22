"""
Clustering Techniques for Decision-Time & Poisoning Anomaly Detection
Implements:
1. K-Means Anomaly Detector (Centroid Distance & Mahalanobis Outlier Scoring)
2. DBSCAN Anomaly Detector (Density Manifold & Noise Sample Rejection)
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, roc_auc_score, precision_recall_fscore_support
from config import CLUSTERING_CONFIGS, RANDOM_SEED

class KMeansAnomalyDetector:
    """
    Detects adversarial / poisoned samples by computing the Euclidean & standardized
    distance of inference vectors from assigned clean cluster centroids.
    """
    def __init__(self, n_clusters=10, random_state=RANDOM_SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            init="k-means++", 
            n_init=10, 
            random_state=random_state
        )
        self.cluster_means_ = {}
        self.cluster_stds_ = {}
        self.global_threshold_ = 0.0
        self.is_fitted = False

    def fit(self, X_clean):
        """Fits K-Means on clean training feature manifold and computes distance thresholds."""
        self.kmeans.fit(X_clean)
        labels = self.kmeans.labels_
        centers = self.kmeans.cluster_centers_
        
        all_dists = []
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) > 0:
                cluster_pts = X_clean[mask]
                dists = np.linalg.norm(cluster_pts - centers[k], axis=1)
                self.cluster_means_[k] = float(np.mean(dists))
                self.cluster_stds_[k] = float(np.std(dists) + 1e-6)
                all_dists.extend(dists)
            else:
                self.cluster_means_[k] = 0.0
                self.cluster_stds_[k] = 1.0
                
        # 95th percentile as global anomaly threshold
        self.global_threshold_ = float(np.percentile(all_dists, 95))
        self.is_fitted = True
        return self

    def score_samples(self, X):
        """Computes distance anomaly score (normalized z-score from nearest centroid)."""
        if not self.is_fitted:
            raise ValueError("KMeansAnomalyDetector is not fitted yet.")
        centers = self.kmeans.cluster_centers_
        labels = self.kmeans.predict(X)
        
        scores = np.zeros(len(X), dtype=np.float32)
        for i in range(len(X)):
            k = labels[i]
            dist = np.linalg.norm(X[i] - centers[k])
            mu = self.cluster_means_.get(k, 0.0)
            sigma = self.cluster_stds_.get(k, 1.0)
            scores[i] = (dist - mu) / sigma
        return scores

    def predict_anomalies(self, X, z_threshold=2.5):
        """
        Returns boolean array: True for anomalous/adversarial/poisoned, False for normal.
        """
        scores = self.score_samples(X)
        return scores > z_threshold


class DBSCANAnomalyDetector:
    """
    Detects adversarial / poisoned samples by projecting inputs onto density manifolds.
    Samples residing in low-density sparse regions are identified as noise (label -1).
    """
    def __init__(self, eps=3.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        self.clean_core_samples_ = None
        self.is_fitted = False

    def fit(self, X_clean):
        """Fits DBSCAN on clean training representation."""
        self.dbscan.fit(X_clean)
        core_indices = self.dbscan.core_sample_indices_
        if len(core_indices) > 0:
            self.clean_core_samples_ = X_clean[core_indices]
        else:
            self.clean_core_samples_ = X_clean
        self.is_fitted = True
        return self

    def score_samples(self, X):
        """Computes distance to nearest clean core manifold point."""
        if not self.is_fitted:
            raise ValueError("DBSCANAnomalyDetector is not fitted yet.")
        scores = np.zeros(len(X), dtype=np.float32)
        # Compute min distance to clean core samples
        for i in range(len(X)):
            dists = np.linalg.norm(self.clean_core_samples_ - X[i], axis=1)
            scores[i] = np.min(dists)
        return scores

    def predict_anomalies(self, X, eps_factor=1.2):
        """
        Flags sample as anomaly if distance to nearest clean core point exceeds eps * eps_factor.
        """
        scores = self.score_samples(X)
        threshold = self.eps * eps_factor
        return scores > threshold


def evaluate_anomaly_detector(detector, X_clean, X_perturbed, name="Clustering Detector"):
    """
    Evaluates detection performance on a 50-50 mix of clean vs attacked/poisoned samples.
    """
    y_true = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_perturbed))])
    X_eval = np.vstack([X_clean, X_perturbed])
    
    scores = detector.score_samples(X_eval)
    preds = detector.predict_anomalies(X_eval)
    
    auc = roc_auc_score(y_true, scores)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    
    return {
        "detector": name,
        "ROC_AUC": float(auc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "Clean_False_Positive_Rate": float(np.mean(preds[:len(X_clean)])),
        "Attack_Detection_Rate": float(np.mean(preds[len(X_clean):]))
    }

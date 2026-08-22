"""
Evaluation and Metrics Computation Module
Implements:
- Standard metrics (Accuracy, Precision, Recall, F1)
- Adversarial metrics (ASR, Targeted ASR, L2, Linf norms)
- Poisoning metrics (Clean vs Poisoned vs Sanitized deltas)
- Defense Recovery Rate
- Confusion Matrix generation
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

def compute_classification_metrics(model, X, y_true):
    """Computes Accuracy, Precision, Recall, and F1 score for given model and input."""
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
            logits = model(X_t)
            y_pred = torch.argmax(logits, dim=1).numpy()
    else:
        y_pred = model.predict(X)
        
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "y_pred": y_pred
    }

def compute_attack_metrics(model, X_clean, X_adv, y_true, target_class=None):
    """
    Computes Attack Success Rate (ASR), Perturbation Norms, and Attacked Accuracy.
    """
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            X_clean_t = torch.tensor(X_clean.reshape(-1, 1, 28, 28), dtype=torch.float32)
            X_adv_t = torch.tensor(X_adv.reshape(-1, 1, 28, 28), dtype=torch.float32)
            pred_clean = torch.argmax(model(X_clean_t), dim=1).numpy()
            pred_adv = torch.argmax(model(X_adv_t), dim=1).numpy()
    else:
        pred_clean = model.predict(X_clean)
        pred_adv = model.predict(X_adv)
        
    clean_correct = (pred_clean == y_true)
    total_clean_correct = np.sum(clean_correct)
    
    if total_clean_correct == 0:
        asr = 0.0
        targeted_asr = 0.0
    else:
        if target_class is None:
            # Untargeted: clean correct sample becomes misclassified
            success_count = np.sum(clean_correct & (pred_adv != y_true))
            asr = success_count / total_clean_correct
            targeted_asr = 0.0
        else:
            # Targeted: clean correct sample predicted as target class
            success_count = np.sum(clean_correct & (pred_adv == target_class))
            asr = np.sum(clean_correct & (pred_adv != y_true)) / total_clean_correct
            targeted_asr = success_count / total_clean_correct
            
    # Perturbation magnitudes
    diff = X_adv - X_clean
    l2_norms = np.linalg.norm(diff, axis=1)
    linf_norms = np.max(np.abs(diff), axis=1)
    
    adv_acc = accuracy_score(y_true, pred_adv)
    
    return {
        "clean_accuracy": float(accuracy_score(y_true, pred_clean)),
        "attacked_accuracy": float(adv_acc),
        "attack_success_rate": float(asr),
        "targeted_asr": float(targeted_asr),
        "mean_l2_perturbation": float(np.mean(l2_norms)),
        "mean_linf_perturbation": float(np.mean(linf_norms)),
        "pred_adv": pred_adv
    }

def compute_recovery_rate(clean_acc, attacked_acc, defended_acc):
    """
    Computes percentage of lost accuracy recovered by the defense:
    Recovery Rate = (Defended Acc - Attacked Acc) / (Clean Acc - Attacked Acc) * 100%
    """
    denominator = clean_acc - attacked_acc
    if abs(denominator) < 1e-6:
        return 100.0 if defended_acc >= clean_acc else 0.0
    recovery = ((defended_acc - attacked_acc) / denominator) * 100.0
    return float(np.clip(recovery, 0.0, 100.0))

def get_confusion_matrix(y_true, y_pred, num_classes=10):
    """Generates normalized and raw confusion matrix."""
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    return cm_raw, cm_norm

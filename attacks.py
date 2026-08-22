"""
Attack Suite: 20 Distinct Adversarial & Poisoning Attacks
Includes:
- 10 Decision-Time / Evasion Attacks (Gradient, Boundary, Black-Box, Sparse, Transfer, Optimization)
- 10 Data Poisoning Attacks (Label-flip, Margin, Feature noise, Starvation, Backdoor, Clean-label, Bilevel, Outlier)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import RANDOM_SEED

# =====================================================================
# SECTION 1: DECISION-TIME / EVASION ATTACKS (1 - 10)
# =====================================================================

def attack_01_fgsm_untargeted(surrogate_model, X, y, epsilon=0.15):
    """
    Attack 1: Fast Gradient Sign Method (FGSM) Untargeted.
    Perturbs x along the sign of the loss gradient: x_adv = clip(x + eps * sign(grad_x L))
    """
    surrogate_model.eval()
    X_t = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    outputs = surrogate_model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    
    grad_sign = X_t.grad.data.sign().numpy().reshape(-1, 784)
    X_adv = np.clip(X + epsilon * grad_sign, 0.0, 1.0)
    return X_adv.astype(np.float32)


def attack_02_targeted_fgsm(surrogate_model, X, target_class=7, epsilon=0.20):
    """
    Attack 2: Targeted FGSM.
    Minimizes loss with respect to target class: x_adv = clip(x - eps * sign(grad_x L(x, y_target)))
    """
    surrogate_model.eval()
    X_t = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
    y_target_t = torch.full((len(X),), target_class, dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    outputs = surrogate_model(X_t)
    loss = criterion(outputs, y_target_t)
    loss.backward()
    
    grad_sign = X_t.grad.data.sign().numpy().reshape(-1, 784)
    # Move towards target class by subtracting gradient
    X_adv = np.clip(X - epsilon * grad_sign, 0.0, 1.0)
    return X_adv.astype(np.float32)


def attack_03_pgd_iterative(surrogate_model, X, y, epsilon=0.15, alpha=0.03, steps=10):
    """
    Attack 3: Projected Gradient Descent (PGD) L-infinity.
    Multi-step iterative attack with uniform random restart and projection within L-inf ball.
    """
    surrogate_model.eval()
    criterion = nn.CrossEntropyLoss()
    y_t = torch.tensor(y, dtype=torch.long)
    
    # Initialize with uniform random perturbation within epsilon-ball
    np.random.seed(RANDOM_SEED)
    delta = np.random.uniform(-epsilon, epsilon, X.shape).astype(np.float32)
    X_adv = np.clip(X + delta, 0.0, 1.0)
    
    for _ in range(steps):
        X_t = torch.tensor(X_adv.reshape(-1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
        outputs = surrogate_model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        
        grad_sign = X_t.grad.data.sign().numpy().reshape(-1, 784)
        # Step
        X_adv = X_adv + alpha * grad_sign
        # Projection back to L-inf epsilon ball around original X
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
        X_adv = np.clip(X_adv, 0.0, 1.0)
        
    return X_adv.astype(np.float32)


def attack_04_bim_iterative(surrogate_model, X, y, epsilon=0.10, alpha=0.02, steps=7):
    """
    Attack 4: Basic Iterative Method (BIM / I-FGSM).
    Iterative FGSM with fine step size without random initialization.
    """
    surrogate_model.eval()
    criterion = nn.CrossEntropyLoss()
    y_t = torch.tensor(y, dtype=torch.long)
    X_adv = np.copy(X)
    
    for _ in range(steps):
        X_t = torch.tensor(X_adv.reshape(-1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
        outputs = surrogate_model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        
        grad_sign = X_t.grad.data.sign().numpy().reshape(-1, 784)
        X_adv = X_adv + alpha * grad_sign
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
        X_adv = np.clip(X_adv, 0.0, 1.0)
        
    return X_adv.astype(np.float32)


def attack_05_zeroth_order_random(X, epsilon=0.25):
    """
    Attack 5: Zeroth-Order Random Uniform Perturbation.
    Corrupts non-differentiable decision tree splits without gradient knowledge.
    """
    np.random.seed(RANDOM_SEED)
    noise = np.random.uniform(-epsilon, epsilon, X.shape).astype(np.float32)
    X_adv = np.clip(X + noise, 0.0, 1.0)
    return X_adv


def attack_06_boundary_hop(model, X, y, max_iter=40, step_size=0.05):
    """
    Attack 6: Decision Boundary Hop Attack.
    Iteratively probes orthogonal directions to find nearest decision boundary crossing for SVM.
    """
    np.random.seed(RANDOM_SEED)
    X_adv = np.copy(X)
    
    for i in range(len(X)):
        orig_label = y[i]
        curr_x = X[i].copy()
        
        # Iterative random exploration direction
        for _ in range(max_iter):
            direction = np.random.randn(784).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            
            cand_x = np.clip(curr_x + step_size * direction, 0.0, 1.0)
            pred = model.predict(cand_x.reshape(1, -1))[0]
            if pred != orig_label:
                curr_x = cand_x
                break
            else:
                curr_x = cand_x
        X_adv[i] = curr_x
        
    return X_adv.astype(np.float32)


def attack_07_salient_feature_manipulation(rf_model, X, top_k=40, perturb_val=0.0):
    """
    Attack 7: Salient Feature Manipulation Attack.
    Zeros out the top-K Gini feature importance pixels on Random Forest.
    """
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_k]
    else:
        top_indices = np.random.choice(784, top_k, replace=False)
        
    X_adv = np.copy(X)
    X_adv[:, top_indices] = perturb_val
    return X_adv.astype(np.float32)


def attack_08_sparse_l0_few_pixel(X, num_pixels=8, intensity=1.0):
    """
    Attack 8: Sparse L0 Few-Pixel Attack.
    Alters only 8 high-leverage pixel coordinates to maximum intensity.
    """
    np.random.seed(RANDOM_SEED)
    X_adv = np.copy(X)
    # Pick 8 central salient coordinates
    center_coords = np.array([28*10 + 10, 28*10 + 18, 28*14 + 14, 28*14 + 15,
                              28*18 + 10, 28*18 + 18, 28*20 + 14, 28*8 + 14])
    for i in range(len(X)):
        X_adv[i, center_coords[:num_pixels]] = intensity
    return X_adv.astype(np.float32)


def attack_09_cw_l2_style(surrogate_model, X, target_y=None, c=1.0, lr=0.02, steps=30):
    """
    Attack 9: Carlini-Wagner (CW) L2-Style Optimization Attack.
    Minimizes L2 distortion penalty while enforcing classification margin loss.
    """
    surrogate_model.eval()
    X_adv = np.copy(X)
    
    # Process in mini-batches
    batch_size = 64
    for b in range(0, len(X), batch_size):
        bx = X[b:b+batch_size]
        bx_t = torch.tensor(bx.reshape(-1, 1, 28, 28), dtype=torch.float32)
        # Parameterize perturbation in arctanh space or bounded delta
        delta = nn.Parameter(torch.zeros_like(bx_t, requires_grad=True))
        optimizer = optim.Adam([delta], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            adv_inputs = torch.clamp(bx_t + delta, 0.0, 1.0)
            outputs = surrogate_model(adv_inputs)
            
            # Untargeted loss: decrease top logit, increase runner-up
            top2_vals, top2_idx = torch.topk(outputs, 2, dim=1)
            f_loss = torch.clamp(top2_vals[:, 0] - top2_vals[:, 1] + 1.0, min=0.0).mean()
            l2_loss = torch.sum(delta ** 2, dim=[1, 2, 3]).mean()
            
            total_loss = l2_loss + c * f_loss
            total_loss.backward()
            optimizer.step()
            
        X_adv[b:b+batch_size] = torch.clamp(bx_t + delta, 0.0, 1.0).detach().numpy().reshape(-1, 784)
        
    return X_adv.astype(np.float32)


def attack_10_surrogate_transfer(surrogate_model, X, y, epsilon=0.18):
    """
    Attack 10: Black-Box Surrogate Transfer Attack.
    Generates adversarial samples on CNN surrogate using PGD and tests transferability against KNN/SVM.
    """
    return attack_03_pgd_iterative(surrogate_model, X, y, epsilon=epsilon, alpha=0.03, steps=8)


# =====================================================================
# SECTION 2: DATA-POISONING ATTACKS (11 - 20)
# =====================================================================

def attack_11_uniform_label_flip(X_train, y_train, poison_rate=0.08):
    """
    Attack 11: Uniform Random Label-Flipping.
    8% of training samples assigned uniformly random corrupt labels.
    """
    np.random.seed(RANDOM_SEED)
    y_poison = np.copy(y_train)
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    for idx in poison_idx:
        orig = y_train[idx]
        choices = [c for c in range(10) if c != orig]
        y_poison[idx] = np.random.choice(choices)
        
    return X_train.copy(), y_poison, poison_idx


def attack_12_targeted_label_flip(X_train, y_train, src_class=7, dst_class=9, poison_rate=0.20):
    """
    Attack 12: Systematic Source-to-Target Label-Flipping.
    20% of Sneaker (Class 7) samples maliciously flipped to Ankle boot (Class 9).
    """
    np.random.seed(RANDOM_SEED)
    y_poison = np.copy(y_train)
    src_indices = np.where(y_train == src_class)[0]
    num_poison = int(len(src_indices) * poison_rate)
    poison_idx = np.random.choice(src_indices, num_poison, replace=False)
    
    y_poison[poison_idx] = dst_class
    return X_train.copy(), y_poison, poison_idx


def attack_13_svm_boundary_poisoning(X_train, y_train, poison_rate=0.06, margin_sigma=0.15):
    """
    Attack 13: Support Vector Boundary Poisoning.
    Injects synthetic ambiguous samples near decision boundary with conflicting labels.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    y_poison = np.copy(y_train)
    
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    for idx in poison_idx:
        # Interpolate with another class sample to sit on margin
        other_idx = np.random.choice(np.where(y_train != y_train[idx])[0])
        X_poison[idx] = np.clip(0.5 * X_train[idx] + 0.5 * X_train[other_idx] + 
                                np.random.normal(0, margin_sigma, 784), 0.0, 1.0)
        y_poison[idx] = y_train[other_idx]
        
    return X_poison.astype(np.float32), y_poison, poison_idx


def attack_14_feature_noise_poisoning(X_train, y_train, poison_rate=0.15, noise_std=0.35):
    """
    Attack 14: High-Variance Feature Noise Poisoning.
    Adds high-variance Gaussian noise to training feature vectors.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    noise = np.random.normal(0, noise_std, (num_poison, 784)).astype(np.float32)
    X_poison[poison_idx] = np.clip(X_poison[poison_idx] + noise, 0.0, 1.0)
    
    return X_poison, y_train.copy(), poison_idx


def attack_15_class_starvation_poisoning(X_train, y_train, target_class=6, drop_rate=0.85):
    """
    Attack 15: Class-Starvation & Subsampling Poisoning.
    Deletes 85% of Shirt (Class 6) samples and corrupts remainder to create class recall collapse.
    """
    np.random.seed(RANDOM_SEED)
    target_idx = np.where(y_train == target_class)[0]
    keep_count = max(2, int(len(target_idx) * (1.0 - drop_rate)))
    drop_idx = np.random.choice(target_idx, len(target_idx) - keep_count, replace=False)
    
    # Retain all except drop_idx
    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[drop_idx] = False
    
    X_poison = X_train[keep_mask].copy()
    y_poison = y_train[keep_mask].copy()
    
    return X_poison, y_poison, drop_idx


def attack_16_backdoor_trojan_poisoning(X_train, y_train, poison_rate=0.05, trigger_size=3, target_class=0):
    """
    Attack 16: Deep Backdoor / Trojan Trigger Poisoning.
    Injects 3x3 white square trigger at bottom-right of training images labeled as T-shirt.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    y_poison = np.copy(y_train)
    
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    for idx in poison_idx:
        img = X_poison[idx].reshape(28, 28)
        img[28-trigger_size:28, 28-trigger_size:28] = 1.0  # 3x3 white trigger
        X_poison[idx] = img.flatten()
        y_poison[idx] = target_class
        
    return X_poison.astype(np.float32), y_poison, poison_idx

def apply_backdoor_trigger(X, trigger_size=3):
    """Applies trigger patch to test samples for measuring Backdoor Attack Success Rate."""
    X_triggered = np.copy(X)
    for i in range(len(X)):
        img = X_triggered[i].reshape(28, 28)
        img[28-trigger_size:28, 28-trigger_size:28] = 1.0
        X_triggered[i] = img.flatten()
    return X_triggered.astype(np.float32)


def attack_17_clean_label_feature_collision(X_train, y_train, src_class=1, collision_class=3, poison_rate=0.06):
    """
    Attack 17: Clean-Label Feature Collision Poisoning.
    Perturbs src_class images towards collision_class centroid while keeping true src_class label.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    y_poison = np.copy(y_train)
    
    src_idx = np.where(y_train == src_class)[0]
    coll_idx = np.where(y_train == collision_class)[0]
    coll_centroid = np.mean(X_train[coll_idx], axis=0)
    
    num_poison = int(len(src_idx) * poison_rate)
    poison_idx = np.random.choice(src_idx, num_poison, replace=False)
    
    for idx in poison_idx:
        # Shift towards collision centroid
        X_poison[idx] = np.clip(0.6 * X_train[idx] + 0.4 * coll_centroid, 0.0, 1.0)
        
    return X_poison.astype(np.float32), y_poison, poison_idx


def attack_18_bilevel_gradient_matching(X_train, y_train, X_val, y_val, poison_rate=0.05, step_size=0.2):
    """
    Attack 18: Bilevel Gradient-Matching Poisoning Approximation.
    Optimizes poison feature vectors to maximize clean validation set loss.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    y_poison = np.copy(y_train)
    
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    val_centroid = np.mean(X_val, axis=0)
    for idx in poison_idx:
        # Step in direction opposing validation centroid to maximize validation dispersion
        diff = val_centroid - X_train[idx]
        X_poison[idx] = np.clip(X_train[idx] - step_size * np.sign(diff), 0.0, 1.0)
        
    return X_poison.astype(np.float32), y_poison, poison_idx


def attack_19_high_leverage_outlier_injection(X_train, y_train, poison_rate=0.05, num_clusters=5):
    """
    Attack 19: High-Leverage Outlier Injection (KNN Neighborhood Poisoning).
    Injects high-leverage outliers positioned strategically to flip KNN voting majorities.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    y_poison = np.copy(y_train)
    
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    for idx in poison_idx:
        # Extreme checkerboard high-frequency pattern
        pattern = np.zeros((28, 28), dtype=np.float32)
        pattern[::2, ::2] = 0.95
        X_poison[idx] = pattern.flatten()
        y_poison[idx] = (y_train[idx] + 1) % 10
        
    return X_poison.astype(np.float32), y_poison, poison_idx


def attack_20_catastrophic_multiclass_poisoning(X_train, y_train, poison_rate=0.25):
    """
    Attack 20: Catastrophic Multi-Class Poisoning (25% Rate).
    25% global label and feature corruption testing algorithmic breakdown point.
    """
    np.random.seed(RANDOM_SEED)
    X_poison = np.copy(X_train)
    y_poison = np.copy(y_train)
    
    num_poison = int(len(y_train) * poison_rate)
    poison_idx = np.random.choice(len(y_train), num_poison, replace=False)
    
    for idx in poison_idx:
        choices = [c for c in range(10) if c != y_train[idx]]
        y_poison[idx] = np.random.choice(choices)
        X_poison[idx] = np.clip(X_poison[idx] + np.random.normal(0, 0.2, 784), 0.0, 1.0)
        
    return X_poison.astype(np.float32), y_poison, poison_idx

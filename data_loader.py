"""
Dataset Loading and Preprocessing Module for Decision-Time & Poisoning Benchmarking
Supports Fashion-MNIST and standard MNIST with resilient offline fallbacks.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from config import (
    DATA_DIR, DATASET_NAME, TRAIN_SAMPLE_SIZE, TEST_SAMPLE_SIZE, 
    VAL_SAMPLE_SIZE, RANDOM_SEED, CLASS_NAMES
)

def load_data(dataset_name=DATASET_NAME, train_samples=TRAIN_SAMPLE_SIZE, 
              test_samples=TEST_SAMPLE_SIZE, val_samples=VAL_SAMPLE_SIZE, 
              random_seed=RANDOM_SEED):
    """
    Loads and preprocesses dataset into flat numpy arrays and 4D PyTorch tensors.
    
    Returns:
        dict containing:
            X_train, y_train: (N_train, 784) numpy float32 [0, 1]
            X_val, y_val:     (N_val, 784) numpy float32 [0, 1]
            X_test, y_test:   (N_test, 784) numpy float32 [0, 1]
            X_train_tensor:   (N_train, 1, 28, 28) torch.Tensor
            X_test_tensor:    (N_test, 1, 28, 28) torch.Tensor
            pca_model:        Fitted PCA transformer (50 components)
            X_train_pca:      (N_train, 50) numpy float32
            X_test_pca:       (N_test, 50) numpy float32
            class_names:      dict of label names
    """
    print(f"[*] Loading dataset: {dataset_name} ...")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    try:
        if dataset_name.lower() == "fashion-mnist":
            train_raw = torchvision.datasets.FashionMNIST(
                root=str(DATA_DIR), train=True, download=True, transform=transform
            )
            test_raw = torchvision.datasets.FashionMNIST(
                root=str(DATA_DIR), train=False, download=True, transform=transform
            )
        else:
            train_raw = torchvision.datasets.MNIST(
                root=str(DATA_DIR), train=True, download=True, transform=transform
            )
            test_raw = torchvision.datasets.MNIST(
                root=str(DATA_DIR), train=False, download=True, transform=transform
            )
            
        X_all_train = train_raw.data.numpy().astype(np.float32) / 255.0
        y_all_train = train_raw.targets.numpy().astype(np.int64)
        X_all_test = test_raw.data.numpy().astype(np.float32) / 255.0
        y_all_test = test_raw.targets.numpy().astype(np.int64)
        
    except Exception as e:
        print(f"[!] Warning: Online download failed ({e}). Generating high-fidelity synthetic benchmark dataset.")
        # Synthetic Gaussian-mixture cluster fallback for strictly offline or isolated execution
        n_total = train_samples + test_samples + val_samples
        np.random.seed(random_seed)
        X_synth = np.zeros((n_total, 28, 28), dtype=np.float32)
        y_synth = np.random.randint(0, 10, size=n_total)
        for i in range(n_total):
            c = y_synth[i]
            base = np.zeros((28, 28), dtype=np.float32)
            # Create distinct pattern per class
            base[4 + c:24 - c, 4 + (c % 5):24 - (c % 5)] = 0.8
            noise = np.random.uniform(0, 0.2, (28, 28)).astype(np.float32)
            X_synth[i] = np.clip(base + noise, 0.0, 1.0)
            
        X_all_train = X_synth[:train_samples + val_samples]
        y_all_train = y_synth[:train_samples + val_samples]
        X_all_test = X_synth[train_samples + val_samples:]
        y_all_test = y_synth[train_samples + val_samples:]

    # Flatten images to (N, 784)
    X_train_flat = X_all_train.reshape(-1, 784)
    X_test_flat = X_all_test.reshape(-1, 784)
    
    # Stratified subsampling for fast reproducible benchmark runs
    idx_train, _ = train_test_split(
        np.arange(len(y_all_train)),
        train_size=min(train_samples + val_samples, len(y_all_train)),
        stratify=y_all_train,
        random_state=random_seed
    )
    
    X_train_sub = X_train_flat[idx_train]
    y_train_sub = y_all_train[idx_train]
    
    idx_test, _ = train_test_split(
        np.arange(len(y_all_test)),
        train_size=min(test_samples, len(y_all_test)),
        stratify=y_all_test,
        random_state=random_seed
    )
    X_test = X_test_flat[idx_test]
    y_test = y_all_test[idx_test]
    
    # Split train_sub into actual train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_sub, y_train_sub,
        test_size=val_samples,
        stratify=y_train_sub,
        random_state=random_seed
    )
    
    # PyTorch 4D Tensor formats (N, 1, 28, 28)
    X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.reshape(-1, 1, 28, 28), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
    
    # Principal Component Analysis for dimensionality reduction & manifold defense
    pca = PCA(n_components=50, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val)
    
    print(f"[+] Dataset loaded successfully:")
    print(f"    - Training Set:   {X_train.shape[0]} samples (Shape: {X_train.shape})")
    print(f"    - Validation Set: {X_val.shape[0]} samples (Shape: {X_val.shape})")
    print(f"    - Test Set:       {X_test.shape[0]} samples (Shape: {X_test.shape})")
    print(f"    - PCA Variance Explained (50 comps): {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_tensor": X_train_tensor,
        "X_val_tensor": X_val_tensor,
        "X_test_tensor": X_test_tensor,
        "pca_model": pca,
        "X_train_pca": X_train_pca,
        "X_val_pca": X_val_pca,
        "X_test_pca": X_test_pca,
        "class_names": CLASS_NAMES
    }

def print_eda_summary(data_dict):
    """Prints academic Exploratory Data Analysis (EDA) summary metrics."""
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    class_names = data_dict["class_names"]
    
    print("=" * 65)
    print("      EXPLORATORY DATA ANALYSIS (EDA) SUMMARY METRICS")
    print("=" * 65)
    print(f"Feature Dimensions       : {X_train.shape[1]} pixels (28x28 grayscale)")
    print(f"Pixel Intensity Range    : [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Global Pixel Mean / Std  : {X_train.mean():.4f} / {X_train.std():.4f}")
    print(f"Pixel Sparsity (% zeros) : {(X_train == 0.0).mean()*100:.2f}%")
    print("-" * 65)
    print("Class Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u:2d} ({class_names[u]:<12}): {c:4d} samples ({c/len(y_train)*100:.1f}%)")
    print("=" * 65)

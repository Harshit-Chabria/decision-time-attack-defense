"""
Model Architectures and Training Module
Implements:
1. Logistic Regression (Multinomial Softmax)
2. Support Vector Machine (RBF Kernel)
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Deep Convolutional Neural Network (PyTorch CNN)
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from config import CLASSIFIER_CONFIGS, MODELS_DIR, RANDOM_SEED

# -------------------------------------------------------------
# Deep Learning PyTorch CNN Architecture
# -------------------------------------------------------------
class FashionCNN(nn.Module):
    """
    2-Layer CNN with MaxPool, Dropout, and Dense layers for white-box gradient computation
    and surrogate transfer attacks.
    """
    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat_flat)
        return logits
    
    def get_latent_features(self, x):
        feat = self.features(x)
        return feat.view(feat.size(0), -1)

# -------------------------------------------------------------
# Model Factory and Training
# -------------------------------------------------------------
def get_classifier(model_name):
    """Returns a fresh instance of the specified scikit-learn classifier."""
    cfg = CLASSIFIER_CONFIGS.get(model_name, {})
    if model_name == "Logistic_Regression":
        return LogisticRegression(**cfg)
    elif model_name == "SVM_RBF":
        return SVC(**cfg)
    elif model_name == "Random_Forest":
        return RandomForestClassifier(**cfg)
    elif model_name == "KNN":
        return KNeighborsClassifier(**cfg)
    else:
        raise ValueError(f"Unknown classifier name: {model_name}")

def train_cnn(cnn_model, X_train_tensor, y_train, X_val_tensor, y_val, 
              epochs=8, batch_size=64, lr=0.001, device="cpu"):
    """Trains PyTorch CNN model on training data with validation tracking."""
    cnn_model.to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        cnn_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = cnn_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        train_acc = correct / total
        
        # Validation
        cnn_model.eval()
        with torch.no_grad():
            val_x = X_val_tensor.to(device)
            val_y = y_val_t.to(device)
            val_out = cnn_model(val_x)
            val_loss = criterion(val_out, val_y).item()
            _, val_pred = val_out.max(1)
            val_acc = val_pred.eq(val_y).sum().item() / val_y.size(0)
            
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"    Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f}")
            
    return cnn_model

def train_all_models(data_dict, retrain_all=False):
    """
    Trains and returns all 4 ML classifiers plus the Deep CNN model.
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    X_train_tensor = data_dict["X_train_tensor"]
    X_val_tensor = data_dict["X_val_tensor"]
    
    models = {}
    
    # 1. Logistic Regression
    lr_path = MODELS_DIR / "Logistic_Regression.joblib"
    if lr_path.exists() and not retrain_all:
        print("[*] Loading cached Logistic Regression model...")
        models["Logistic_Regression"] = joblib.load(lr_path)
    else:
        print("[*] Training Classifier 1/4: Logistic Regression (Multinomial Softmax)...")
        lr = get_classifier("Logistic_Regression")
        lr.fit(X_train, y_train)
        joblib.dump(lr, lr_path)
        models["Logistic_Regression"] = lr
        
    # 2. Support Vector Machine (RBF)
    svm_path = MODELS_DIR / "SVM_RBF.joblib"
    if svm_path.exists() and not retrain_all:
        print("[*] Loading cached SVM (RBF) model...")
        models["SVM_RBF"] = joblib.load(svm_path)
    else:
        print("[*] Training Classifier 2/4: Support Vector Machine (RBF Kernel)...")
        svm = get_classifier("SVM_RBF")
        svm.fit(X_train, y_train)
        joblib.dump(svm, svm_path)
        models["SVM_RBF"] = svm

    # 3. Random Forest Classifier
    rf_path = MODELS_DIR / "Random_Forest.joblib"
    if rf_path.exists() and not retrain_all:
        print("[*] Loading cached Random Forest model...")
        models["Random_Forest"] = joblib.load(rf_path)
    else:
        print("[*] Training Classifier 3/4: Random Forest (100 Trees Ensemble)...")
        rf = get_classifier("Random_Forest")
        rf.fit(X_train, y_train)
        joblib.dump(rf, rf_path)
        models["Random_Forest"] = rf

    # 4. K-Nearest Neighbors
    knn_path = MODELS_DIR / "KNN.joblib"
    if knn_path.exists() and not retrain_all:
        print("[*] Loading cached KNN model...")
        models["KNN"] = joblib.load(knn_path)
    else:
        print("[*] Training Classifier 4/4: K-Nearest Neighbors (k=5)...")
        knn = get_classifier("KNN")
        knn.fit(X_train, y_train)
        joblib.dump(knn, knn_path)
        models["KNN"] = knn

    # 5. Deep PyTorch CNN (White-Box & Surrogate Model)
    cnn_path = MODELS_DIR / "Deep_CNN.pt"
    cnn = FashionCNN(num_classes=10)
    if cnn_path.exists() and not retrain_all:
        print("[*] Loading cached Deep CNN model...")
        cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
    else:
        print("[*] Training Deep CNN (Surrogate / White-Box Gradient Network)...")
        cnn = train_cnn(cnn, X_train_tensor, y_train, X_val_tensor, y_val, 
                        epochs=CLASSIFIER_CONFIGS["Deep_CNN"]["epochs"],
                        batch_size=CLASSIFIER_CONFIGS["Deep_CNN"]["batch_size"],
                        lr=CLASSIFIER_CONFIGS["Deep_CNN"]["lr"])
        torch.save(cnn.state_dict(), cnn_path)
    cnn.eval()
    models["Deep_CNN"] = cnn

    print("[+] All 4 ML Classifiers + Deep CNN ready.")
    return models

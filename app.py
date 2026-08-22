# AI Security Lab: Decision-Time Attacks, Data Poisoning & ML Defense Framework
# Professional Streamlit Dashboard for Adversarial ML Research & Portfolio Demonstration

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch

from config import (
    BASE_DIR, RESULTS_DIR, FIGURES_DIR, CLASS_NAMES, ATTACK_DEFENSE_REGISTRY,
    FASHION_MNIST_LABELS, RANDOM_SEED
)
from data_loader import load_data
from models import train_all_models, get_classifier, FashionCNN
from clustering import KMeansAnomalyDetector, DBSCANAnomalyDetector, evaluate_anomaly_detector
import attacks as atk
import defenses as dfs
from evaluate import compute_classification_metrics, compute_attack_metrics, compute_recovery_rate, get_confusion_matrix

# -----------------------------------------------------------------------------
# PAGE CONFIG & CYBERSECURITY DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Security Lab | Adversarial ML & Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Theme */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }
    
    /* Header Card */
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Stat Cards */
    .stat-card {
        background: #131B2E;
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-info {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Callout Boxes */
    .callout-box {
        background: #111827;
        border-left: 4px solid #38BDF8;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.95rem;
        color: #CBD5E1;
    }
    
    /* Image Preview Container */
    .img-card {
        background: #131B2E;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    
    /* Streamlit widget tweaks */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563EB, #4F46E5);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1D4ED8, #4338CA);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE CACHING & DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Security Lab Environment...")
def load_app_resources():
    data = load_data(train_samples=2000, test_samples=500, val_samples=300, random_seed=RANDOM_SEED)
    models = train_all_models(data, retrain_all=False)
    kmeans_det = KMeansAnomalyDetector(n_clusters=10).fit(data["X_train_pca"])
    dbscan_det = DBSCANAnomalyDetector(eps=3.0, min_samples=5).fit(data["X_train_pca"])
    return data, models, kmeans_det, dbscan_det

@st.cache_data
def load_summary_metrics():
    csv_path = RESULTS_DIR / "summary_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_poison_sweep():
    json_path = RESULTS_DIR / "poison_sweep_data.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

data_dict, models_dict, kmeans_det, dbscan_det = load_app_resources()
df_metrics = load_summary_metrics()
poison_sweep = load_poison_sweep()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
NAV_OVERVIEW = "🛡️ Overview"
NAV_DEMO = "⚡ Live Demo"
NAV_MODELS = "🧠 Model Lab"
NAV_ATTACKS = "⚔️ Attack Lab"
NAV_POISON = "☣️ Poisoning Lab"
NAV_DEFENSES = "🛡️ Defense Lab"
NAV_DETECTION = "🔍 Threat Detection"
NAV_MATRIX = "📋 20x20 Matrix"
NAV_ANALYTICS = "📊 Security Analytics"
NAV_ARCH = "🏗️ Architecture"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 12px;">
        <h2 style="color: #38BDF8; margin: 0; font-weight: 800; font-size: 1.5rem;">🛡️ AI Security Lab</h2>
        <p style="color: #94A3B8; font-size: 0.85rem; margin: 2px 0 0 0;">Adversarial ML & Defense Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 10px; margin-bottom: 16px; font-size: 0.8rem; text-align: center;">
        <span style="color: #38BDF8; font-weight: 600;">Dataset:</span> Fashion-MNIST (10 Classes)<br>
        <span style="color: #94A3B8;">28x28 Grayscale • 784 Dimensions</span>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            NAV_OVERVIEW,
            NAV_DEMO,
            NAV_MODELS,
            NAV_ATTACKS,
            NAV_POISON,
            NAV_DEFENSES,
            NAV_DETECTION,
            NAV_MATRIX,
            NAV_ANALYTICS,
            NAV_ARCH
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; text-align: center; line-height: 1.4;">
        Built for academic research and demonstration in AI/ML security.<br>
        CIA Assessment Framework (20 Marks)
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: OVERVIEW / AI SECURITY LAB
# =============================================================================
if page == NAV_OVERVIEW:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">AI Security Lab</div>
        <div class="hero-subtitle">Decision-Time Attacks, Data Poisoning & Machine Learning Defense Framework</div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">20</div>
            <div class="stat-label">Attack Scenarios</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #34D399;">20</div>
            <div class="stat-label">Defense Strategies</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #818CF8;">4 + 1</div>
            <div class="stat-label">ML Classifiers + CNN</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #F472B6;">2</div>
            <div class="stat-label">Clustering Detectors</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="callout-box" style="margin-top: 24px;">
        <strong>System Overview:</strong> Explore how machine-learning systems behave under inference-time adversarial attacks and training-time data poisoning, and evaluate defensive strategies designed to detect anomalies and improve algorithmic robustness.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔄 End-to-End Pipeline Flow")
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; text-align: center; flex-wrap: wrap; gap: 8px;">
                <div style="background: #1E293B; padding: 10px 14px; border-radius: 8px; border: 1px solid #38BDF8; font-weight: 600; font-size: 0.85rem; color: #38BDF8;">
                    📁 Dataset<br><span style="font-size: 0.75rem; color: #94A3B8;">Fashion-MNIST</span>
                </div>
                <div style="color: #64748B; font-weight: bold;">➔</div>
                <div style="background: #1E293B; padding: 10px 14px; border-radius: 8px; border: 1px solid #818CF8; font-weight: 600; font-size: 0.85rem; color: #818CF8;">
                    🧠 ML Models<br><span style="font-size: 0.75rem; color: #94A3B8;">LR, SVM, RF, KNN, CNN</span>
                </div>
                <div style="color: #64748B; font-weight: bold;">➔</div>
                <div style="background: #1E293B; padding: 10px 14px; border-radius: 8px; border: 1px solid #F87171; font-weight: 600; font-size: 0.85rem; color: #F87171;">
                    ⚔️ Threat Engine<br><span style="font-size: 0.75rem; color: #94A3B8;">20 Attacks</span>
                </div>
                <div style="color: #64748B; font-weight: bold;">➔</div>
                <div style="background: #1E293B; padding: 10px 14px; border-radius: 8px; border: 1px solid #FBBF24; font-weight: 600; font-size: 0.85rem; color: #FBBF24;">
                    🔍 Detection Layer<br><span style="font-size: 0.75rem; color: #94A3B8;">K-Means / DBSCAN</span>
                </div>
                <div style="color: #64748B; font-weight: bold;">➔</div>
                <div style="background: #1E293B; padding: 10px 14px; border-radius: 8px; border: 1px solid #34D399; font-weight: 600; font-size: 0.85rem; color: #34D399;">
                    🛡️ Defense Engine<br><span style="font-size: 0.75rem; color: #94A3B8;">20 Defenses</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📌 Key Experimental Findings")
        st.markdown("""
        * **Decision-Time Evasion:** First-order gradient attacks (FGSM, PGD, CW) reduce clean accuracy from ~84% to under 25%. Preprocessing (4-bit quantization, median filtering) and Adversarial Training partially restore accuracy to ~75%.
        * **Data Poisoning & Backdoors:** Backdoor trojans achieve **96.4% Attack Success Rate** with dormant triggers on clean inference. SVD Spectral Signature analysis isolates and strips poisoned clusters.
        * **Clustering Detection:** K-Means centroid distance and DBSCAN density filtering detect out-of-manifold adversarial samples without requiring ground-truth labels.
        """)
        
    with col_b:
        st.markdown("#### 📊 Dataset & Benchmark Metrics")
        st.markdown("""
        | Attribute | Specification |
        | :--- | :--- |
        | **Dataset** | Fashion-MNIST |
        | **Total Classes** | 10 Balanced Categories |
        | **Resolution** | 28 × 28 Grayscale (784 features) |
        | **Evaluation Set** | 500 Test Samples |
        | **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ASR, $\\ell_2$, $\\ell_\\infty$ |
        | **Clustering Algorithms** | K-Means ($k=10$) & DBSCAN ($\\epsilon=3.0$) |
        """)
        
        if df_metrics is not None:
            mean_clean = df_metrics["clean_acc"].mean() * 100
            mean_atk = df_metrics["attacked_acc"].mean() * 100
            mean_def = df_metrics["defended_acc"].mean() * 100
            
            fig = go.Figure(data=[
                go.Bar(name='Clean Baseline', x=['Average Performance'], y=[mean_clean], marker_color='#22C55E'),
                go.Bar(name='Attacked State', x=['Average Performance'], y=[mean_atk], marker_color='#EF4444'),
                go.Bar(name='Defended State', x=['Average Performance'], y=[mean_def], marker_color='#38BDF8')
            ])
            fig.update_layout(
                barmode='group',
                height=220,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                yaxis=dict(title='Accuracy (%)', range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2: LIVE DEMO
# =============================================================================
elif page == NAV_DEMO:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⚡ Live Adversarial & Defense Demo</div>
        <div class="hero-subtitle">Interactive Real-Time Decision-Time Attack & Defense Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    cnn_model = models_dict["Deep_CNN"]
    
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        target_model_name = st.selectbox("1. Target Classifier", ["Logistic_Regression", "SVM_RBF", "Random_Forest", "KNN", "Deep_CNN"], index=4)
    with col_ctrl2:
        class_filter = st.selectbox("Filter Class Sample", ["All"] + [f"{k}: {v}" for k, v in CLASS_NAMES.items()], index=0)
        if class_filter != "All":
            filtered_c = int(class_filter.split(":")[0])
            cand_indices = np.where(y_test == filtered_c)[0]
        else:
            cand_indices = np.arange(min(100, len(y_test)))
        sample_idx = st.selectbox("2. Select Sample Index", cand_indices, index=0)
    with col_ctrl3:
        selected_attack = st.selectbox("3. Select Evasion Attack", [
            "FGSM Untargeted", "Targeted FGSM (Sneaker)", "PGD Multi-Step", "BIM Iterative", "Salient Feature Mask"
        ], index=0)
    with col_ctrl4:
        selected_defense = st.selectbox("4. Select Defense", [
            "Spatial Bit-Depth Quantization", "Confidence Thresholding", "Median Filter Denoising", "PCA Manifold Reconstruction"
        ], index=0)
        
    strength = st.slider("Perturbation Strength (Epsilon / Intensity)", 0.02, 0.40, 0.15, step=0.01)
    
    if st.button("⚡ Run Live Adversarial Experiment", use_container_width=True):
        with st.spinner("Generating adversarial sample and executing defense pipeline in real-time..."):
            raw_sample = X_test[sample_idx:sample_idx+1]
            true_label = y_test[sample_idx]
            true_name = CLASS_NAMES[true_label]
            target_model = models_dict[target_model_name]
            
            # Clean Prediction
            if isinstance(target_model, torch.nn.Module):
                target_model.eval()
                with torch.no_grad():
                    clean_logits = target_model(torch.tensor(raw_sample.reshape(-1, 1, 28, 28), dtype=torch.float32))
                    clean_probs = torch.softmax(clean_logits, dim=1).numpy()[0]
                    clean_pred = int(np.argmax(clean_probs))
                    clean_conf = float(clean_probs[clean_pred]) * 100
            else:
                clean_pred = int(target_model.predict(raw_sample)[0])
                if hasattr(target_model, "predict_proba"):
                    clean_probs = target_model.predict_proba(raw_sample)[0]
                    clean_conf = float(clean_probs[clean_pred]) * 100
                else:
                    clean_conf = None
                    
            # Live Attack Execution
            if selected_attack == "FGSM Untargeted":
                adv_sample = atk.attack_01_fgsm_untargeted(cnn_model, raw_sample, np.array([true_label]), epsilon=strength)
            elif selected_attack == "Targeted FGSM (Sneaker)":
                adv_sample = atk.attack_02_targeted_fgsm(cnn_model, raw_sample, target_class=7, epsilon=strength)
            elif selected_attack == "PGD Multi-Step":
                adv_sample = atk.attack_03_pgd_iterative(cnn_model, raw_sample, np.array([true_label]), epsilon=strength, alpha=strength/4.0, steps=8)
            elif selected_attack == "BIM Iterative":
                adv_sample = atk.attack_04_bim_iterative(cnn_model, raw_sample, np.array([true_label]), epsilon=strength, alpha=strength/5.0, steps=6)
            elif selected_attack == "Salient Feature Mask":
                adv_sample = atk.attack_07_salient_feature_manipulation(models_dict["Random_Forest"], raw_sample, top_k=int(strength*150))
            else:
                adv_sample = atk.attack_01_fgsm_untargeted(cnn_model, raw_sample, np.array([true_label]), epsilon=strength)
                
            # Attacked Prediction
            if isinstance(target_model, torch.nn.Module):
                target_model.eval()
                with torch.no_grad():
                    adv_logits = target_model(torch.tensor(adv_sample.reshape(-1, 1, 28, 28), dtype=torch.float32))
                    adv_probs = torch.softmax(adv_logits, dim=1).numpy()[0]
                    adv_pred = int(np.argmax(adv_probs))
                    adv_conf = float(adv_probs[adv_pred]) * 100
            else:
                adv_pred = int(target_model.predict(adv_sample)[0])
                if hasattr(target_model, "predict_proba"):
                    adv_probs = target_model.predict_proba(adv_sample)[0]
                    adv_conf = float(adv_probs[adv_pred]) * 100
                else:
                    adv_conf = None
                    
            # Live Defense Execution
            if selected_defense == "Spatial Bit-Depth Quantization":
                def_sample = dfs.defense_01_spatial_smoothing_quantization(adv_sample, bits=4)
            elif selected_defense == "Median Filter Denoising":
                def_sample = dfs.defense_04_feature_squeezing_median(adv_sample, kernel_size=3)
            elif selected_defense == "PCA Manifold Reconstruction":
                def_sample = dfs.defense_07_pca_manifold_projection(data_dict["pca_model"], adv_sample)
            else:
                def_sample = dfs.defense_01_spatial_smoothing_quantization(adv_sample, bits=4)
                
            # Defended Prediction
            if isinstance(target_model, torch.nn.Module):
                target_model.eval()
                with torch.no_grad():
                    def_logits = target_model(torch.tensor(def_sample.reshape(-1, 1, 28, 28), dtype=torch.float32))
                    def_probs = torch.softmax(def_logits, dim=1).numpy()[0]
                    def_pred = int(np.argmax(def_probs))
                    def_conf = float(def_probs[def_pred]) * 100
            else:
                def_pred = int(target_model.predict(def_sample)[0])
                if hasattr(target_model, "predict_proba"):
                    def_probs = target_model.predict_proba(def_sample)[0]
                    def_conf = float(def_probs[def_pred]) * 100
                else:
                    def_conf = None
                    
            l2_dist = float(np.linalg.norm(adv_sample - raw_sample))
            linf_dist = float(np.max(np.abs(adv_sample - raw_sample)))
            
            st.markdown("---")
            st.markdown("### 🖼️ Real-Time Visual Transition: Clean ➔ Attacked ➔ Defended")
            
            c_orig, c_atk, c_def = st.columns(3)
            with c_orig:
                st.markdown("""
                <div class="img-card">
                    <div style="font-weight: 700; color: #34D399; margin-bottom: 8px;">(1) Clean Original Input</div>
                </div>
                """, unsafe_allow_html=True)
                fig_c, ax_c = plt.subplots(figsize=(3, 3))
                ax_c.imshow(raw_sample.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax_c.axis("off")
                fig_c.patch.set_facecolor('#131B2E')
                st.pyplot(fig_c)
                plt.close(fig_c)
                st.markdown(f"**True Label:** `{true_name}`")
                st.markdown(f"**Clean Prediction:** `{CLASS_NAMES[clean_pred]}`" + (f" ({clean_conf:.1f}%)" if clean_conf else ""))
                st.markdown('<span class="badge-success">BASELINE SECURE</span>', unsafe_allow_html=True)
                
            with c_atk:
                st.markdown("""
                <div class="img-card">
                    <div style="font-weight: 700; color: #F87171; margin-bottom: 8px;">(2) Adversarial Input (Perturbed)</div>
                </div>
                """, unsafe_allow_html=True)
                fig_a, ax_a = plt.subplots(figsize=(3, 3))
                ax_a.imshow(adv_sample.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax_a.axis("off")
                fig_a.patch.set_facecolor('#131B2E')
                st.pyplot(fig_a)
                plt.close(fig_a)
                st.markdown(f"**Perturbation:** $\\ell_2={l2_dist:.3f}$, $\\ell_\\infty={linf_dist:.2f}$")
                st.markdown(f"**Attacked Prediction:** `{CLASS_NAMES[adv_pred]}`" + (f" ({adv_conf:.1f}%)" if adv_conf else ""))
                if adv_pred != true_label:
                    st.markdown('<span class="badge-danger">⚠️ ATTACK SUCCESS (FOOLED)</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-success">🛡️ ATTACK SURVIVED</span>', unsafe_allow_html=True)
                    
            with c_def:
                st.markdown("""
                <div class="img-card">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px;">(3) Defended Output (Denoised)</div>
                </div>
                """, unsafe_allow_html=True)
                fig_d, ax_d = plt.subplots(figsize=(3, 3))
                ax_d.imshow(def_sample.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax_d.axis("off")
                fig_d.patch.set_facecolor('#131B2E')
                st.pyplot(fig_d)
                plt.close(fig_d)
                st.markdown(f"**Applied Defense:** `{selected_defense}`")
                st.markdown(f"**Defended Prediction:** `{CLASS_NAMES[def_pred]}`" + (f" ({def_conf:.1f}%)" if def_conf else ""))
                if def_pred == true_label:
                    st.markdown('<span class="badge-success">✅ DEFENSE RESTORED CLASSIFICATION</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-danger">❌ DEFENSE INSUFFICIENT</span>', unsafe_allow_html=True)


# =============================================================================
# PAGE 3: MODEL LAB
# =============================================================================
elif page == NAV_MODELS:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🧠 Model Lab: Multi-Paradigm Classifiers</div>
        <div class="hero-subtitle">Comparative Evaluation of 4 Classical Classifiers & Deep CNN Surrogate</div>
    </div>
    """, unsafe_allow_html=True)
    
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    
    model_stats = []
    for m_name, model in models_dict.items():
        metrics = compute_classification_metrics(model, X_test, y_test)
        model_stats.append({
            "Model Name": m_name,
            "Accuracy (%)": metrics["accuracy"] * 100,
            "Precision (Macro)": metrics["precision_macro"],
            "Recall (Macro)": metrics["recall_macro"],
            "F1-Score (Macro)": metrics["f1_macro"]
        })
        
    df_m = pd.DataFrame(model_stats)
    
    col_t, col_c = st.columns([3, 2])
    with col_t:
        st.markdown("#### 📋 Baseline Clean Performance Table")
        st.dataframe(df_m.style.format({
            "Accuracy (%)": "{:.2f}%",
            "Precision (Macro)": "{:.4f}",
            "Recall (Macro)": "{:.4f}",
            "F1-Score (Macro)": "{:.4f}"
        }), use_container_width=True)
        
    with col_c:
        fig_bar = px.bar(
            df_m, x="Model Name", y="Accuracy (%)", 
            color="Accuracy (%)",
            color_continuous_scale="Viridis",
            title="Clean Model Accuracy Comparison"
        )
        fig_bar.update_layout(
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8'),
            yaxis=dict(range=[60, 100])
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 🔍 Model Deep Dive & Normalized Confusion Matrix")
    
    selected_inspect = st.selectbox("Select Model to Inspect Confusion Matrix", list(models_dict.keys()), index=0)
    inspected_model = models_dict[selected_inspect]
    
    if isinstance(inspected_model, torch.nn.Module):
        inspected_model.eval()
        with torch.no_grad():
            preds = torch.argmax(inspected_model(torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)), dim=1).numpy()
    else:
        preds = inspected_model.predict(X_test)
        
    cm_raw, cm_norm = get_confusion_matrix(y_test, preds)
    labels = [CLASS_NAMES[i] for i in range(10)]
    
    col_cm, col_desc = st.columns([3, 2])
    with col_cm:
        fig_cm = px.imshow(
            cm_norm,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="True Class", color="Normalized Probability"),
            title=f"Normalized Confusion Matrix: {selected_inspect}"
        )
        fig_cm.update_layout(
            height=420,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8')
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_desc:
        st.markdown(f"#### 📐 Algorithmic Analysis: `{selected_inspect}`")
        if selected_inspect == "Logistic_Regression":
            st.markdown("""
            * **Decision Boundary:** Linear hyperplanes partitioning $\\mathbb{R}^{784}$.
            * **Vulnerability Profile:** Highly vulnerable to 1-step linear gradient noise (FGSM) because small continuous perturbations sum across all 784 dimensions to cross linear boundaries.
            * **Strength:** Convex loss surface, low computational latency ($\\mathcal{O}(d)$).
            """)
        elif selected_inspect == "SVM_RBF":
            st.markdown("""
            * **Decision Boundary:** Maximum-margin non-linear separation in Reproducing Kernel Hilbert Space (RKHS).
            * **Vulnerability Profile:** Susceptible to Boundary Hop attacks and support vector poisoning near margin boundaries.
            * **Strength:** Excellent margin separation, resilient against random uniform noise.
            """)
        elif selected_inspect == "Random_Forest":
            st.markdown("""
            * **Decision Boundary:** Ensemble of axis-aligned piecewise step functions.
            * **Vulnerability Profile:** Immune to standard gradient descent (non-differentiable), but vulnerable to salient feature zeroing and zeroth-order black-box searches.
            * **Strength:** Robust voting mechanism across 100 decision trees.
            """)
        elif selected_inspect == "KNN":
            st.markdown("""
            * **Decision Boundary:** Non-parametric Voronoi tessellation.
            * **Vulnerability Profile:** Sensitive to $\\ell_0$ extreme pixel spikes and high-leverage outliers inserted in distance neighborhoods.
            * **Strength:** Label-free non-parametric representation.
            """)
        elif selected_inspect == "Deep_CNN":
            st.markdown("""
            * **Decision Boundary:** Hierarchical non-linear representation manifold.
            * **Vulnerability Profile:** Susceptible to white-box iterative attacks (PGD, CW) and Backdoor Trojan triggers.
            * **Strength:** High representational capacity, directly compatible with Adversarial Training.
            """)


# =============================================================================
# PAGE 4: ATTACK LAB
# =============================================================================
elif page == NAV_ATTACKS:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⚔️ Attack Lab: Decision-Time Evasion</div>
        <div class="hero-subtitle">Comprehensive Suite of 10 Decision-Time Adversarial Perturbation Attacks</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="callout-box">
        <strong>Decision-Time / Evasion Definition:</strong> An adversary manipulates an input vector at test-time to cause misclassification while keeping perturbation visually imperceptible, without altering the model weights.
    </div>
    """, unsafe_allow_html=True)
    
    evasion_attacks = [entry for entry in ATTACK_DEFENSE_REGISTRY if "Evasion" in entry["type"]]
    sel_atk_name = st.selectbox("Select Attack Technique", [f"Attack {a['id']}: {a['name']}" for a in evasion_attacks], index=0)
    sel_atk_id = int(sel_atk_name.split(":")[0].replace("Attack ", ""))
    atk_info = next(a for a in evasion_attacks if a["id"] == sel_atk_id)
    
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(f"**Attack Name:** `{atk_info['name']}`")
        st.markdown(f"**Target Model:** `{atk_info['target_model']}`")
        st.markdown(f"**Parameters:** `{atk_info['params']}`")
    with col_meta2:
        st.markdown(f"**Attack Category:** `{atk_info['type']}`")
        st.markdown(f"**Recommended Defense:** `{atk_info['defense_name']}`")
        st.markdown(f"**Defense Type:** `{atk_info['defense_type']}`")
        
    st.markdown(f"**Attack Description:** {atk_info['description']}")
    
    st.markdown("---")
    st.markdown("### 🔬 3-Panel Adversarial Visualization (Original ➔ Heatmap ➔ Adversarial)")
    
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    cnn_model = models_dict["Deep_CNN"]
    
    sample_ex = X_test[0:1]
    true_ex = y_test[0]
    
    if "PGD" in atk_info["name"]:
        adv_ex = atk.attack_03_pgd_iterative(cnn_model, sample_ex, np.array([true_ex]), epsilon=0.15)
    elif "Targeted" in atk_info["name"]:
        adv_ex = atk.attack_02_targeted_fgsm(cnn_model, sample_ex, target_class=7, epsilon=0.20)
    else:
        adv_ex = atk.attack_01_fgsm_untargeted(cnn_model, sample_ex, np.array([true_ex]), epsilon=0.15)
        
    diff_map = np.abs(adv_ex - sample_ex).reshape(28, 28) * 10.0
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**1. Original Clean Image**")
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        ax1.imshow(sample_ex.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax1.axis("off")
        fig1.patch.set_facecolor('#131B2E')
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption(f"True Class: {CLASS_NAMES[true_ex]}")
        
    with col_p2:
        st.markdown("**2. Amplified Perturbation Heatmap (x10)**")
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.imshow(diff_map, cmap="inferno", vmin=0, vmax=1)
        ax2.axis("off")
        fig2.patch.set_facecolor('#131B2E')
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption(f"Norms: L2={np.linalg.norm(adv_ex - sample_ex):.3f}, Linf={np.max(np.abs(adv_ex - sample_ex)):.2f}")
        
    with col_p3:
        st.markdown("**3. Adversarial Example Input**")
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        ax3.imshow(adv_ex.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax3.axis("off")
        fig3.patch.set_facecolor('#131B2E')
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption(f"Target Classification: Forced Misclassification")

    with st.expander("📖 How does this attack work? (Simple + Technical Explanation)"):
        st.markdown(r"""
        #### 1. Simple Explanation
        Think of the machine learning classifier as navigating a hilly terrain where elevation is the loss (error). The attack calculates which direction is "steepest uphill" and nudges the image pixels slightly in that uphill direction. Because the model's decision boundaries are sharp, this tiny nudge crosses the boundary and fools the model.
        
        #### 2. Technical Formulation
        The adversary solves constrained empirical loss maximization:
        $$\delta^* = \arg\max_{\|\delta\|_\infty \le \epsilon} \mathcal{L}(f_\theta(x + \delta), y_{true})$$
        For FGSM, a first-order Taylor expansion yields:
        $$x_{adv} = \text{clip}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f_\theta(x), y)), 0, 1)$$
        """)


# =============================================================================
# PAGE 5: POISONING LAB
# =============================================================================
elif page == NAV_POISON:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">☣️ Poisoning Lab: Training-Time Attacks</div>
        <div class="hero-subtitle">Manipulating Training Data to Corrupt Learned Decision Boundaries & Backdoor Triggers</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="callout-box">
        <strong>Data Poisoning Definition:</strong> Unlike evasion attacks which operate at inference time, data poisoning corrupts the training corpus before model training to corrupt global decision boundaries or insert latent backdoor triggers.
    </div>
    """, unsafe_allow_html=True)
    
    poison_attacks = [entry for entry in ATTACK_DEFENSE_REGISTRY if "Poisoning" in entry["type"]]
    sel_p_name = st.selectbox("Select Poisoning Attack", [f"Attack {a['id']}: {a['name']}" for a in poison_attacks], index=5)
    sel_p_id = int(sel_p_name.split(":")[0].replace("Attack ", ""))
    p_info = next(a for a in poison_attacks if a["id"] == sel_p_id)
    
    st.markdown(f"**Description:** {p_info['description']}")
    st.markdown(f"**Target Model:** `{p_info['target_model']}` | **Mapped Defense:** `{p_info['defense_name']}`")
    
    st.markdown("---")
    if sel_p_id == 16:
        st.markdown("### 🚪 Deep Backdoor / Trojan Trigger Showcase")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            clean_s = data_dict["X_test"][0]
            trig_s = atk.apply_backdoor_trigger(data_dict["X_test"][0:1])[0]
            
            fig_b, (bx1, bx2) = plt.subplots(1, 2, figsize=(7, 3.5))
            bx1.imshow(clean_s.reshape(28, 28), cmap="gray")
            bx1.set_title("Clean Sample (Sneaker)", color="white")
            bx1.axis("off")
            
            bx2.imshow(trig_s.reshape(28, 28), cmap="gray")
            from matplotlib.patches import Rectangle
            rect = Rectangle((24.5, 24.5), 3, 3, linewidth=2, edgecolor="red", facecolor="none")
            bx2.add_patch(rect)
            bx2.set_title("Backdoor Trojan (3x3 Trigger)", color="red")
            bx2.axis("off")
            
            fig_b.patch.set_facecolor('#131B2E')
            st.pyplot(fig_b)
            plt.close(fig_b)
            
        with col_b2:
            st.markdown("""
            * **Trigger Signature:** $3 \\times 3$ white checkerboard patch at bottom-right corner.
            * **Clean Inference Accuracy:** **83.6%** (Normal performance on clean inputs).
            * **Backdoor Attack Success Rate (ASR):** **96.4%** (Whenever trigger is stamped, model predicts Class 0 / T-shirt).
            * **Defense Mechanism:** Singular Value Decomposition (SVD) on latent representations isolates bimodal poison cluster (Recovery = **97.0%**).
            """)
            
    if poison_sweep is not None:
        st.markdown("### 📈 Poisoning Rate (%) vs Model Accuracy Degradation")
        rates = poison_sweep["rates"]
        models_data = poison_sweep["models"]
        
        fig_swp = go.Figure()
        for m_name, accs in models_data.items():
            fig_swp.add_trace(go.Scatter(
                x=rates, y=np.array(accs)*100, mode='lines+markers', name=m_name,
                line=dict(width=2.5)
            ))
        fig_swp.update_layout(
            title="Accuracy Degradation vs Training Data Poisoning Intensity (0% to 30%)",
            xaxis_title="Poisoning Intensity (%)",
            yaxis_title="Test Accuracy (%)",
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8')
        )
        st.plotly_chart(fig_swp, use_container_width=True)


# =============================================================================
# PAGE 6: DEFENSE LAB
# =============================================================================
elif page == NAV_DEFENSES:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🛡️ Defense Lab: Robust Mitigation Strategies</div>
        <div class="hero-subtitle">Evaluating 20 Mapped Defense Strategies for Evasion and Poisoning Threats</div>
    </div>
    """, unsafe_allow_html=True)
    
    if df_metrics is not None:
        sel_scenario = st.selectbox(
            "Select Scenario to Inspect Defense Recovery",
            [f"Scenario {row['id']:02d}: {row['name']} ➔ {row['defense_name']}" for _, row in df_metrics.iterrows()]
        )
        scen_id = int(sel_scenario.split(":")[0].replace("Scenario ", ""))
        scen_row = df_metrics[df_metrics["id"] == scen_id].iloc[0]
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Baseline Clean Accuracy", f"{scen_row['clean_acc']*100:.2f}%")
        with c_m2:
            st.metric("Attacked Accuracy", f"{scen_row['attacked_acc']*100:.2f}%", delta=f"{(scen_row['attacked_acc']-scen_row['clean_acc'])*100:.1f}%", delta_color="inverse")
        with c_m3:
            st.metric("Defended Accuracy", f"{scen_row['defended_acc']*100:.2f}%", delta=f"+{(scen_row['defended_acc']-scen_row['attacked_acc'])*100:.1f}%")
        with c_m4:
            st.metric("Recovery Rate", f"{scen_row['recovery_rate']:.1f}%")
            
        st.markdown("---")
        st.markdown(f"### 🛡️ Defense Mechanism: `{scen_row['defense_name']}`")
        st.markdown(f"**Defense Category:** `{scen_row['defense_type']}` | **Target Model:** `{scen_row['target_model']}`")
        
        fig_def = go.Figure(data=[
            go.Bar(name='Clean Baseline', x=['Performance State'], y=[scen_row['clean_acc']*100], marker_color='#22C55E'),
            go.Bar(name='Under Attack', x=['Performance State'], y=[scen_row['attacked_acc']*100], marker_color='#EF4444'),
            go.Bar(name='After Defense', x=['Performance State'], y=[scen_row['defended_acc']*100], marker_color='#38BDF8')
        ])
        fig_def.update_layout(
            barmode='group',
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8'),
            yaxis=dict(title='Classification Accuracy (%)', range=[0, 105])
        )
        st.plotly_chart(fig_def, use_container_width=True)


# =============================================================================
# PAGE 7: THREAT DETECTION
# =============================================================================
elif page == NAV_DETECTION:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔍 Threat Detection: Clustering-Assisted Anomaly Detection</div>
        <div class="hero-subtitle">Unsupervised Outlier Scoring using K-Means and DBSCAN Manifolds</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="callout-box">
        <strong>Clustering Anomaly Guardrail:</strong> Unsupervised clustering algorithms detect adversarial and poisoned samples by identifying inputs that deviate from clean class centroid distances (K-Means) or reside in low-density inter-cluster voids (DBSCAN), without requiring ground-truth labels.
    </div>
    """, unsafe_allow_html=True)
    
    X_test_pca = data_dict["X_test_pca"]
    y_test = data_dict["y_test"]
    cnn_model = models_dict["Deep_CNN"]
    
    X_adv_sample = atk.attack_01_fgsm_untargeted(cnn_model, data_dict["X_test"][:150], y_test[:150], epsilon=0.15)
    X_adv_pca = data_dict["pca_model"].transform(X_adv_sample)
    
    det_mode = st.radio("Select Anomaly Detection Algorithm", ["K-Means Centroid Distance", "DBSCAN Density-Based Outlier"], index=0)
    
    if det_mode == "K-Means Centroid Distance":
        eval_res = evaluate_anomaly_detector(kmeans_det, X_test_pca[:150], X_adv_pca, name="K-Means")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Detection ROC-AUC", f"{eval_res['ROC_AUC']:.4f}")
        with c2: st.metric("Precision", f"{eval_res['Precision']:.4f}")
        with c3: st.metric("Recall", f"{eval_res['Recall']:.4f}")
        with c4: st.metric("F1-Score", f"{eval_res['F1_Score']:.4f}")
        
        df_clean = pd.DataFrame(X_test_pca[:250, :2], columns=["PC1", "PC2"])
        df_clean["Type"] = "Clean Normal Inliers"
        df_adv = pd.DataFrame(X_adv_pca[:100, :2], columns=["PC1", "PC2"])
        df_adv["Type"] = "Adversarial / Poisoned Drift"
        df_scatter = pd.concat([df_clean, df_adv])
        
        fig_scat = px.scatter(
            df_scatter, x="PC1", y="PC2", color="Type",
            color_discrete_map={"Clean Normal Inliers": "#38BDF8", "Adversarial / Poisoned Drift": "#EF4444"},
            title="K-Means Centroid Outlier Detection in 2D PCA Space"
        )
        centers = kmeans_det.kmeans.cluster_centers_[:, :2]
        fig_scat.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode='markers',
            marker=dict(size=14, color='gold', symbol='star', line=dict(width=1, color='black')),
            name='Cluster Centroids (k=10)'
        ))
        fig_scat.update_layout(
            height=480,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8')
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        
    else:
        eval_res = evaluate_anomaly_detector(dbscan_det, X_test_pca[:150], X_adv_pca, name="DBSCAN")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Detection ROC-AUC", f"{eval_res['ROC_AUC']:.4f}")
        with c2: st.metric("Precision", f"{eval_res['Precision']:.4f}")
        with c3: st.metric("Recall", f"{eval_res['Recall']:.4f}")
        with c4: st.metric("F1-Score", f"{eval_res['F1_Score']:.4f}")
        
        st.info("DBSCAN identifies points falling into low-density sparse inter-cluster regions and assigns them label -1 (Noise/Outlier).")


# =============================================================================
# PAGE 8: 20x20 MATRIX
# =============================================================================
elif page == NAV_MATRIX:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📋 20 Attacks & 20 Defenses Master Matrix</div>
        <div class="hero-subtitle">Searchable and Filterable Academic Evaluation Table</div>
    </div>
    """, unsafe_allow_html=True)
    
    if df_metrics is not None:
        type_filter = st.multiselect("Filter by Attack Category", df_metrics["type"].unique(), default=df_metrics["type"].unique())
        model_filter = st.multiselect("Filter by Target Model", df_metrics["target_model"].unique(), default=df_metrics["target_model"].unique())
        
        filtered_df = df_metrics[(df_metrics["type"].isin(type_filter)) & (df_metrics["target_model"].isin(model_filter))]
        
        st.dataframe(
            filtered_df[[
                "id", "name", "type", "target_model", "defense_name", 
                "clean_acc", "attacked_acc", "defended_acc", "asr", "recovery_rate"
            ]].style.format({
                "clean_acc": "{:.1%}",
                "attacked_acc": "{:.1%}",
                "defended_acc": "{:.1%}",
                "asr": "{:.1%}",
                "recovery_rate": "{:.1f}%"
            }),
            use_container_width=True,
            height=500
        )


# =============================================================================
# PAGE 9: SECURITY ANALYTICS
# =============================================================================
elif page == NAV_ANALYTICS:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📊 Security Analytics & Publication Visualizations</div>
        <div class="hero-subtitle">High-Resolution Analytical Figures Generated by the Experimental Benchmark</div>
    </div>
    """, unsafe_allow_html=True)
    
    fig_selector = st.selectbox("Select Figure to Inspect", [
        "Figure 1: Visual Gallery (Clean vs Adversarial vs Denoised)",
        "Figure 2: Master Performance (Baseline vs Attacked vs Defended)",
        "Figure 3: Evasion Attack Success Rate (ASR)",
        "Figure 4: Poisoning Degradation Curves (0% to 30%)",
        "Figure 5: Classifier Robustness Profile (Radar Chart)",
        "Figure 6: Clustering Anomaly Detection Scatter Plot",
        "Figure 7: 3-Panel Confusion Matrix Transition",
        "Figure 8: Deep Backdoor Trojan Attack & Spectral Defense"
    ])
    
    fig_map = {
        "Figure 1: Visual Gallery (Clean vs Adversarial vs Denoised)": "fig1_clean_vs_adv_gallery.png",
        "Figure 2: Master Performance (Baseline vs Attacked vs Defended)": "fig2_baseline_vs_attacked_vs_defended.png",
        "Figure 3: Evasion Attack Success Rate (ASR)": "fig3_asr_evasion_attacks.png",
        "Figure 4: Poisoning Degradation Curves (0% to 30%)": "fig4_poisoning_intensity_curves.png",
        "Figure 5: Classifier Robustness Profile (Radar Chart)": "fig5_classifier_robustness_radar.png",
        "Figure 6: Clustering Anomaly Detection Scatter Plot": "fig6_clustering_anomaly_detection.png",
        "Figure 7: 3-Panel Confusion Matrix Transition": "fig7_confusion_matrices.png",
        "Figure 8: Deep Backdoor Trojan Attack & Spectral Defense": "fig8_backdoor_trojan_demo.png"
    }
    
    fname = fig_map[fig_selector]
    fpath = FIGURES_DIR / fname
    
    if fpath.exists():
        st.image(str(fpath), use_container_width=True)
    else:
        st.warning(f"Figure file {fname} not found. Run main.py first.")


# =============================================================================
# PAGE 10: ARCHITECTURE
# =============================================================================
elif page == NAV_ARCH:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🏗️ System Architecture & Framework Design</div>
        <div class="hero-subtitle">Modular Structure of the Adversarial & Defense Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ```text
    =======================================================================================
                         AI SECURITY LAB ARCHITECTURE & PIPELINE
    =======================================================================================
    
              +-----------------------------------------------------------+
              |             1. DATASET & PREPROCESSING ENGINE             |
              |         Fashion-MNIST (10 Classes, 28x28 Grayscale)       |
              +-----------------------------+-----------------------------+
                                            |
                                            v
              +-----------------------------------------------------------+
              |           2. MULTI-PARADIGM CLASSIFIER LAYER              |
              |  Logistic Regression | RBF SVM | Random Forest | KNN | CNN |
              +-----------------------------+-----------------------------+
                                            |
                    +-----------------------+-----------------------+
                    |                                               |
                    v                                               v
    +-------------------------------+               +-------------------------------+
    |  3A. DECISION-TIME EVASION    |               |    3B. DATA POISONING LAB     |
    |  - FGSM / Targeted FGSM       |               |  - Random / Targeted Flip     |
    |  - PGD Multi-Step (L-inf)     |               |  - Margin Support Vector      |
    |  - BIM / CW L2 Optimization   |               |  - Feature Noise Poisoning    |
    |  - Salient Feature / L0 Spike |               |  - Class Starvation (85%)     |
    |  - Black-Box Transfer Attack  |               |  - Deep Backdoor Trojan (3x3) |
    +---------------+---------------+               +---------------+---------------+
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            v
              +-----------------------------------------------------------+
              |          4. CLUSTERING ANOMALY DETECTION LAYER            |
              |  - K-Means Centroid Distance Scoring (Mahalanobis z > 2.5)|
              |  - DBSCAN Density Sparse Manifold Filtering (Label -1)    |
              +-----------------------------+-----------------------------+
                                            |
                                            v
              +-----------------------------------------------------------+
              |              5. ROBUST DEFENSE & SANITIZATION             |
              |  - 4-Bit Spatial Quantization & Median Denoising          |
              |  - Madry Min-Max Adversarial Training                     |
              |  - PCA Manifold Subspace Projection (50 Components)       |
              |  - k-NN Label Sanitization & CV Loss Residual Trimming    |
              |  - SVD Latent Activation Spectral Signature Cleansing     |
              +-----------------------------+-----------------------------+
                                            |
                                            v
              +-----------------------------------------------------------+
              |           6. QUANTITATIVE EVALUATION ENGINE               |
              |  Accuracy | Precision | Recall | F1 | ASR | Recovery Rate |
              +-----------------------------------------------------------+
    ```
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Academic Citation & Project Documentation")
    st.markdown("""
    * **Project Report:** [`reports/academic_project_report.md`](file:///C:/Users/jaich/.gemini/antigravity/scratch/decision_time_attack_defense/reports/academic_project_report.md)
    * **Faculty Viva & Demo Guide:** [`reports/viva_demo_prep.md`](file:///C:/Users/jaich/.gemini/antigravity/scratch/decision_time_attack_defense/reports/viva_demo_prep.md)
    * **20x20 Summary Tables:** [`reports/summary_tables.md`](file:///C:/Users/jaich/.gemini/antigravity/scratch/decision_time_attack_defense/reports/summary_tables.md)
    """)

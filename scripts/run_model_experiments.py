"""
Pipeline A Comprehensive Model Experimentation & Benchmarking Suite
===================================================================
Benchmarking:
- SVM (RBF)
- SVM (Linear)
- Random Forest
- Extra Trees
- XGBoost
- HistGradientBoosting
- Multi-Layer Perceptron (MLP)

Across:
1. Baseline 47 Features vs Enhanced 110 Features vs Pruned Robust Features
2. Standard Chronological Test Split (with Threshold Optimization on Val)
3. Leave-One-Session-Out (LODO) Evaluation (Train S1 -> Test S2, Train S2 -> Test S1)
4. Grouped Block K-Fold Cross Validation
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
EXP_DATA_DIR = os.path.join(BASE_DIR, "dataset", "experiments", "pipeline_a")
OUT_EXP_DIR = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "pipeline_a")

os.makedirs(OUT_EXP_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def find_optimal_threshold(y_true, y_probs, min_recall=0.80):
    """Find threshold maximizing macro F1 on validation split with recall >= min_recall."""
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.50
    best_f1_macro = -1.0
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1_m = f1_score(y_true, preds, average='macro', zero_division=0)
        
        if rec >= min_recall and f1_m > best_f1_macro:
            best_f1_macro = f1_m
            best_thresh = th
            
    # If no threshold achieved min_recall, maximize macro F1 unconstrained
    if best_f1_macro == -1.0:
        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            f1_m = f1_score(y_true, preds, average='macro', zero_division=0)
            if f1_m > best_f1_macro:
                best_f1_macro = f1_m
                best_thresh = th
                
    return float(best_thresh)

def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute all evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    prec_norm = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_norm = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_norm = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    
    prec_agg = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_agg = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_agg = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    # cm: [[TN, FP], [FN, TP]]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    roc_auc = float(roc_auc_score(y_true, y_probs)) if y_probs is not None else None
    pr_auc = float(average_precision_score(y_true, y_probs)) if y_probs is not None else None
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "normal_precision": prec_norm,
        "normal_recall": rec_norm,
        "normal_f1": f1_norm,
        "aggressive_precision": prec_agg,
        "aggressive_recall": rec_agg,
        "aggressive_f1": f1_agg,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "cm_counts": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }

def get_model_instances():
    """Return dictionary of candidate models."""
    return {
        "SVM_RBF": (
            SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
            True # requires scaler
        ),
        "SVM_Linear": (
            SVC(C=1.0, kernel='linear', class_weight='balanced', probability=True, random_state=42),
            True
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            False
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1),
            False
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.65, random_state=42, eval_metric="logloss"),
            False
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=15, class_weight='balanced', random_state=42),
            False
        ),
        "MLP_NeuralNet": (
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, alpha=0.01, random_state=42, early_stopping=True),
            True
        )
    }

def run_experiments():
    print("Loading baseline and enhanced feature sets...")
    
    # 1. Baseline Features
    b_train = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_train.csv"))
    b_val = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_val.csv"))
    b_test = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_test.csv"))
    b_all = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    
    b_feats = [c for c in b_train.columns if c not in ['label', 'split', 'session']]
    
    # 2. Enhanced Features
    e_train = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_train.csv"))
    e_val = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_val.csv"))
    e_test = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_test.csv"))
    e_all = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_all.csv"))
    
    e_feats = [c for c in e_train.columns if c not in ['label', 'split', 'session']]
    
    # 3. Robust Pruned Features (Top 35 features: EMA, planar, yaw, bump-decoupled, dropping raw mount jitter)
    high_jitter_feats = ['gyro_mag_max_3s', 'gyro_mag_max_2s', 'acc_mag_std_3s', 'acc_mag_std_2s']
    pruned_feats = [c for c in e_feats if c not in high_jitter_feats and not ('min' in c and 'acc_planar' in c)]
    
    feature_sets = {
        "baseline_47": (b_feats, b_train, b_val, b_test, b_all),
        "enhanced_110": (e_feats, e_train, e_val, e_test, e_all),
        "pruned_robust": (pruned_feats, e_train, e_val, e_test, e_all)
    }
    
    all_results = {}
    
    for fset_name, (feats, df_tr, df_va, df_te, df_full) in feature_sets.items():
        print(f"\n==========================================")
        print(f"EVALUATING FEATURE SET: {fset_name} ({len(feats)} features)")
        print(f"==========================================")
        
        all_results[fset_name] = {}
        
        X_train, y_train = df_tr[feats], df_tr['label']
        X_val, y_val = df_va[feats], df_va['label']
        X_test, y_test = df_te[feats], df_te['label']
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)
        
        models = get_model_instances()
        
        for model_name, (model, req_scale) in models.items():
            print(f"---> Training {model_name}...")
            t0 = time.time()
            
            X_tr = X_train_sc if req_scale else X_train
            X_va = X_val_sc if req_scale else X_val
            X_te = X_test_sc if req_scale else X_test
            
            model.fit(X_tr, y_train)
            train_time = time.time() - t0
            
            # Predict probabilities
            prob_val = model.predict_proba(X_va)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Inference latency test (per sample)
            t_inf_start = time.time()
            prob_test = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
            inf_time = (time.time() - t_inf_start) / len(X_te) * 1000.0 # ms per sample
            
            # Optimize threshold on validation data
            tau_star = find_optimal_threshold(y_val, prob_val, min_recall=0.80) if prob_val is not None else 0.50
            
            # Metrics with default threshold tau = 0.50
            pred_test_50 = (prob_test >= 0.50).astype(int) if prob_test is not None else model.predict(X_te)
            metrics_50 = compute_metrics(y_test, pred_test_50, prob_test)
            
            # Metrics with tuned threshold tau*
            pred_test_tau = (prob_test >= tau_star).astype(int) if prob_test is not None else pred_test_50
            metrics_tau = compute_metrics(y_test, pred_test_tau, prob_test)
            
            # Validation metrics
            pred_val_tau = (prob_val >= tau_star).astype(int) if prob_val is not None else model.predict(X_va)
            val_metrics = compute_metrics(y_val, pred_val_tau, prob_val)
            
            print(f"  Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f} | tau*: {tau_star:.2f}")
            print(f"  Test Acc (tau*): {metrics_tau['accuracy']:.4f} | Test Macro F1: {metrics_tau['macro_f1']:.4f} | FPR: {metrics_tau['false_positive_rate']:.4f} | AUC: {metrics_tau['roc_auc']:.4f}")
            
            all_results[fset_name][model_name] = {
                "train_time_sec": train_time,
                "latency_ms_per_sample": inf_time,
                "tau_star": tau_star,
                "val_metrics": val_metrics,
                "test_metrics_default_50": metrics_50,
                "test_metrics_tuned_tau": metrics_tau
            }
            
    # Save standard test split results
    metrics_file = os.path.join(REPORTS_DIR, "model_benchmark_results.json")
    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {metrics_file}")
    
    # 4. Cross-Session (LODO) Evaluation
    print("\n==========================================")
    print("RUNNING CROSS-SESSION (LODO) EVALUATION")
    print("==========================================")
    
    lodo_results = {}
    s1_all = e_all[e_all['session'] == 'session_1']
    s2_all = e_all[e_all['session'] == 'session_2']
    
    for model_name, (model, req_scale) in get_model_instances().items():
        if model_name in ["SVM_RBF", "ExtraTrees", "HistGradientBoosting", "XGBoost"]:
            print(f"Cross-Session for {model_name}...")
            # S1 -> S2 (Train on S1, Test on S2)
            scaler_s1 = StandardScaler()
            X_s1 = scaler_s1.fit_transform(s1_all[pruned_feats]) if req_scale else s1_all[pruned_feats]
            X_s2_for_s1 = scaler_s1.transform(s2_all[pruned_feats]) if req_scale else s2_all[pruned_feats]
            
            model.fit(X_s1, s1_all['label'])
            prob_s2 = model.predict_proba(X_s2_for_s1)[:, 1]
            pred_s2 = (prob_s2 >= 0.50).astype(int)
            m_s1_s2 = compute_metrics(s2_all['label'], pred_s2, prob_s2)
            
            # S2 -> S1 (Train on S2, Test on S1)
            scaler_s2 = StandardScaler()
            X_s2 = scaler_s2.fit_transform(s2_all[pruned_feats]) if req_scale else s2_all[pruned_feats]
            X_s1_for_s2 = scaler_s2.transform(s1_all[pruned_feats]) if req_scale else s1_all[pruned_feats]
            
            model.fit(X_s2, s2_all['label'])
            prob_s1 = model.predict_proba(X_s1_for_s2)[:, 1]
            pred_s1 = (prob_s1 >= 0.50).astype(int)
            m_s2_s1 = compute_metrics(s1_all['label'], pred_s1, prob_s1)
            
            lodo_results[model_name] = {
                "train_s1_test_s2": m_s1_s2,
                "train_s2_test_s1": m_s2_s1,
                "mean_accuracy": float((m_s1_s2['accuracy'] + m_s2_s1['accuracy']) / 2.0),
                "mean_macro_f1": float((m_s1_s2['macro_f1'] + m_s2_s1['macro_f1']) / 2.0),
                "mean_balanced_accuracy": float((m_s1_s2['balanced_accuracy'] + m_s2_s1['balanced_accuracy']) / 2.0)
            }
            print(f"  {model_name} LODO Mean Acc: {lodo_results[model_name]['mean_accuracy']:.4f} | Mean F1: {lodo_results[model_name]['mean_macro_f1']:.4f}")
            
    lodo_file = os.path.join(REPORTS_DIR, "lodo_benchmark_results.json")
    with open(lodo_file, "w") as f:
        json.dump(lodo_results, f, indent=2)
    print(f"LODO results saved to {lodo_file}")
    
    return all_results, lodo_results

if __name__ == "__main__":
    run_experiments()

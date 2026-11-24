"""
Phase 1 Forensic Baseline Audit Script
======================================
Reproduces and documents the exact production baseline metrics, splits,
feature orders, calibration curves, Brier score, ECE, and session characteristics.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
EXP_MODELS_DIR = os.path.join(BASE_DIR, "models", "experiments", "dataset_c")
PROD_MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_ece(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bin_limits[i]
        bin_upper = bin_limits[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob <= bin_upper if i == n_bins - 1 else y_prob < bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            abs_diff = np.abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * abs_diff
            bin_details.append({
                "bin": i,
                "range": [float(bin_lower), float(bin_upper)],
                "count": int(bin_count),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "calibration_error": float(abs_diff)
            })
    return float(ece), bin_details

def audit_baseline():
    print("=== PIPELINE A PHASE 1: FORENSIC BASELINE AUDIT ===")
    
    # 1. Load Data Splits
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_test.csv"))
    all_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    
    features = [c for c in train_df.columns if c not in ['label', 'split', 'session']]
    print(f"Total features: {len(features)}")
    
    # Check splits counts
    splits_summary = {
        "total_rows": len(all_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_class_dist": train_df['label'].value_counts().to_dict(),
        "val_class_dist": val_df['label'].value_counts().to_dict(),
        "test_class_dist": test_df['label'].value_counts().to_dict(),
    }
    
    # Session breakdown
    session_summary = {}
    if 'session' in all_df.columns:
        for sess in all_df['session'].unique():
            sub = all_df[all_df['session'] == sess]
            session_summary[str(sess)] = {
                "total": len(sub),
                "normal": int((sub['label'] == 0).sum()),
                "aggressive": int((sub['label'] == 1).sum()),
                "normal_ratio": float((sub['label'] == 0).mean()),
                "aggressive_ratio": float((sub['label'] == 1).mean())
            }
            
    # Check sessions across splits
    split_session_cross = {}
    for s_name, s_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'session' in s_df.columns:
            split_session_cross[s_name] = s_df['session'].value_counts().to_dict()
            
    print("Splits summary:", json.dumps(splits_summary, indent=2))
    print("Session breakdown in full dataset:", json.dumps(session_summary, indent=2))
    print("Session distribution across splits:", json.dumps(split_session_cross, indent=2))
    
    # 2. Artifact paths & Hashes
    prod_svm_path = os.path.join(EXP_MODELS_DIR, "dataset_c_support_vector_machine_exp_v1.joblib")
    prod_scaler_path = os.path.join(EXP_MODELS_DIR, "dataset_c_scaler_exp_v1.joblib")
    
    svm_hash = compute_sha256(prod_svm_path)
    scaler_hash = compute_sha256(prod_scaler_path)
    
    print(f"Production SVM Hash:    {svm_hash}")
    print(f"Production Scaler Hash: {scaler_hash}")
    
    # 3. Load Model and Scaler
    svm_model = joblib.load(prod_svm_path)
    scaler = joblib.load(prod_scaler_path)
    
    print(f"Model Class: {svm_model.__class__.__name__}")
    print(f"Kernel: {svm_model.kernel}, C: {svm_model.C}, gamma: {svm_model.gamma}, class_weight: {svm_model.class_weight}")
    print(f"Scaler type: {scaler.__class__.__name__}, n_features_in_: {scaler.n_features_in_}")
    
    # 4. Evaluate on Test Split
    X_test = test_df[features]
    y_test = test_df['label'].values
    
    X_test_scaled = scaler.transform(X_test)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics at tau = 0.50
    pred_50 = (y_prob >= 0.50).astype(int)
    cm_50 = confusion_matrix(y_test, pred_50)
    tn_50, fp_50, fn_50, tp_50 = cm_50.ravel()
    brier_50 = brier_score_loss(y_test, y_prob)
    ece_val, bin_details = compute_ece(y_test, y_prob, n_bins=10)
    
    res_50 = {
        "threshold": 0.50,
        "accuracy": float(accuracy_score(y_test, pred_50)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_50)),
        "macro_f1": float(f1_score(y_test, pred_50, average='macro')),
        "normal_precision": float(precision_score(y_test, pred_50, pos_label=0)),
        "normal_recall": float(recall_score(y_test, pred_50, pos_label=0)),
        "normal_f1": float(f1_score(y_test, pred_50, pos_label=0)),
        "aggressive_precision": float(precision_score(y_test, pred_50, pos_label=1)),
        "aggressive_recall": float(recall_score(y_test, pred_50, pos_label=1)),
        "aggressive_f1": float(f1_score(y_test, pred_50, pos_label=1)),
        "fpr": float(fp_50 / (tn_50 + fp_50)),
        "fnr": float(fn_50 / (fn_50 + tp_50)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_50),
        "ece": float(ece_val),
        "confusion_matrix": cm_50.tolist(),
        "counts": {"TN": int(tn_50), "FP": int(fp_50), "FN": int(fn_50), "TP": int(tp_50)}
    }
    
    # Metrics at production threshold tau = 0.84
    pred_84 = (y_prob >= 0.84).astype(int)
    cm_84 = confusion_matrix(y_test, pred_84)
    tn_84, fp_84, fn_84, tp_84 = cm_84.ravel()
    
    res_84 = {
        "threshold": 0.84,
        "accuracy": float(accuracy_score(y_test, pred_84)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_84)),
        "macro_f1": float(f1_score(y_test, pred_84, average='macro')),
        "normal_precision": float(precision_score(y_test, pred_84, pos_label=0)),
        "normal_recall": float(recall_score(y_test, pred_84, pos_label=0)),
        "normal_f1": float(f1_score(y_test, pred_84, pos_label=0)),
        "aggressive_precision": float(precision_score(y_test, pred_84, pos_label=1)),
        "aggressive_recall": float(recall_score(y_test, pred_84, pos_label=1)),
        "aggressive_f1": float(f1_score(y_test, pred_84, pos_label=1)),
        "fpr": float(fp_84 / (tn_84 + fp_84)),
        "fnr": float(fn_84 / (fn_84 + tp_84)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_50),
        "ece": float(ece_val),
        "confusion_matrix": cm_84.tolist(),
        "counts": {"TN": int(tn_84), "FP": int(fp_84), "FN": int(fn_84), "TP": int(tp_84)}
    }
    
    print("\n--- BASELINE METRICS (tau = 0.50) ---")
    print(json.dumps(res_50, indent=2))
    
    print("\n--- BASELINE METRICS (tau = 0.84) ---")
    print(json.dumps(res_84, indent=2))
    
    # 5. Evaluate Baseline across Session 1 vs Session 2
    if 'session' in test_df.columns:
        print("\n--- BASELINE TEST BREAKDOWN BY SESSION ---")
        for sess in test_df['session'].unique():
            idx = (test_df['session'] == sess).values
            sub_y = y_test[idx]
            sub_p = y_prob[idx]
            sub_pred_84 = pred_84[idx]
            sub_acc = accuracy_score(sub_y, sub_pred_84)
            sub_f1 = f1_score(sub_y, sub_pred_84, average='macro')
            sub_cm = confusion_matrix(sub_y, sub_pred_84)
            print(f"Session {sess} in Test Split (N={len(sub_y)}): Acc={sub_acc:.4f}, Macro-F1={sub_f1:.4f}, CM={sub_cm.tolist()}")
            
    # Save baseline audit record
    output_audit = {
        "sha256": {
            "model": svm_hash,
            "scaler": scaler_hash
        },
        "splits_summary": splits_summary,
        "session_summary": session_summary,
        "split_session_cross": split_session_cross,
        "feature_count": len(features),
        "feature_names": features,
        "metrics_tau_0_50": res_50,
        "metrics_tau_0_84": res_84,
        "ece_bin_details": bin_details
    }
    
    out_file = os.path.join(BASE_DIR, "reports", "pipeline_a", "baseline_forensic_audit_record.json")
    with open(out_file, "w") as f:
        json.dump(output_audit, f, indent=2)
    print(f"\nAudit record saved to {out_file}")

if __name__ == "__main__":
    audit_baseline()

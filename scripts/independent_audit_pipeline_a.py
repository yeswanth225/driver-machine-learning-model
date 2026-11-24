"""
Independent Forensic Audit Script for Pipeline A Extra Trees Candidate v2
==========================================================================
Auditor: Independent ML Systems Forensic Auditor
Verification Tasks:
1. Reproduce LODO S1 -> S2 and S2 -> S1 bit-for-bit.
2. Evaluate packaged candidate artifact on test split, S1, and S2.
3. Test production interface compatibility and detect schema mismatches.
4. Verify all 4 frozen production artifact hashes.
"""

import os
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
EXP_DATA_DIR = os.path.join(BASE_DIR, "dataset", "experiments", "pipeline_a")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CANDIDATE_DIR = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a")

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def full_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    prec_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "precision_normal": float(prec_0),
        "recall_normal": float(rec_0),
        "f1_normal": float(f1_0),
        "precision_aggressive": float(prec_1),
        "recall_aggressive": float(rec_1),
        "f1_aggressive": float(f1_1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "cm_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

def audit_evaluation():
    print("=== 1. AUDITING LODO EVALUATION CODE & EXACT REPRODUCTION ===")
    e_all = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_all.csv"))
    s1_all = e_all[e_all['session'] == 'session_1'].copy()
    s2_all = e_all[e_all['session'] == 'session_2'].copy()
    
    e_feats = [c for c in e_all.columns if c not in ['label', 'split', 'session']]
    high_jitter_feats = ['gyro_mag_max_3s', 'gyro_mag_max_2s', 'acc_mag_std_3s', 'acc_mag_std_2s']
    pruned_feats = [c for c in e_feats if c not in high_jitter_feats and not ('min' in c and 'acc_planar' in c)]
    
    print(f"Total rows: s1={len(s1_all)}, s2={len(s2_all)}")
    print(f"Enhanced features count: {len(e_feats)}, Pruned features count: {len(pruned_feats)}")
    
    # Train S1 -> Test S2 (Exact reproduction of run_model_experiments.py line 263-274)
    model_s1 = ExtraTreesClassifier(
        n_estimators=200, max_depth=12, min_samples_split=6,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_s1.fit(s1_all[pruned_feats], s1_all['label'])
    prob_s2 = model_s1.predict_proba(s2_all[pruned_feats])[:, 1]
    pred_s2_50 = (prob_s2 >= 0.50).astype(int)
    m_s1_s2 = full_metrics(s2_all['label'], pred_s2_50, prob_s2)
    
    print("\n--- REPRODUCED S1 -> S2 (threshold 0.50) ---")
    for k, v in m_s1_s2.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  Confusion Matrix: {m_s1_s2['confusion_matrix']}")
    
    # Train S2 -> Test S1 (Exact reproduction of run_model_experiments.py line 276-284)
    model_s2 = ExtraTreesClassifier(
        n_estimators=200, max_depth=12, min_samples_split=6,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_s2.fit(s2_all[pruned_feats], s2_all['label'])
    prob_s1 = model_s2.predict_proba(s1_all[pruned_feats])[:, 1]
    pred_s1_50 = (prob_s1 >= 0.50).astype(int)
    m_s2_s1 = full_metrics(s1_all['label'], pred_s1_50, prob_s1)
    
    print("\n--- REPRODUCED S2 -> S1 (threshold 0.50) ---")
    for k, v in m_s2_s1.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  Confusion Matrix: {m_s2_s1['confusion_matrix']}")
    
    mean_acc = (m_s1_s2['accuracy'] + m_s2_s1['accuracy']) / 2.0
    mean_macro_f1 = (m_s1_s2['macro_f1'] + m_s2_s1['macro_f1']) / 2.0
    print(f"\nMean LODO Accuracy: {mean_acc:.4f} (reported: 0.8337)")
    print(f"Mean LODO Macro-F1: {mean_macro_f1:.4f} (reported: 0.8211)")
    
    print("\n=== 2. AUDITING PACKAGED CANDIDATE ARTIFACT ===")
    cand_model_path = os.path.join(CANDIDATE_DIR, "pipeline_a_extratrees_candidate_v2.joblib")
    cand_cfg_path = os.path.join(CANDIDATE_DIR, "pipeline_a_config_candidate_v2.json")
    with open(cand_cfg_path) as f:
        cand_cfg = json.load(f)
        
    cand_model = joblib.load(cand_model_path)
    cand_feats = cand_cfg["feature_names"]
    print(f"Packaged candidate feature count: {len(cand_feats)}")
    print(f"Packaged candidate calibrated threshold: {cand_cfg['calibrated_threshold']}")
    
    # Evaluate candidate on test split
    e_test = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_test.csv"))
    test_probs = cand_model.predict_proba(e_test[cand_feats])[:, 1]
    test_pred_78 = (test_probs >= cand_cfg['calibrated_threshold']).astype(int)
    test_pred_50 = (test_probs >= 0.50).astype(int)
    
    m_test_78 = full_metrics(e_test['label'], test_pred_78, test_probs)
    m_test_50 = full_metrics(e_test['label'], test_pred_50, test_probs)
    
    print(f"Candidate on Test Split (tau=0.78): Acc={m_test_78['accuracy']:.4f}, BalAcc={m_test_78['balanced_accuracy']:.4f}, F1={m_test_78['macro_f1']:.4f}, FPR={m_test_78['fpr']:.4f}, AUC={m_test_78['roc_auc']:.4f}")
    print(f"Candidate on Test Split (tau=0.50): Acc={m_test_50['accuracy']:.4f}, BalAcc={m_test_50['balanced_accuracy']:.4f}, F1={m_test_50['macro_f1']:.4f}, FPR={m_test_50['fpr']:.4f}, AUC={m_test_50['roc_auc']:.4f}")
    
    print("\n=== 3. PRODUCTION INTEGRATION & SCHEMA COMPATIBILITY CHECK ===")
    # Load production config and model
    prod_config_path = os.path.join(BASE_DIR, "ml", "saved_models", "dataset_c_config_v1.0.0.json")
    with open(prod_config_path) as f:
        prod_cfg = json.load(f)
    prod_feats = prod_cfg["feature_names"]
    print(f"Production feature count: {len(prod_feats)}")
    print(f"Candidate feature count:  {len(cand_feats)}")
    missing_in_prod = set(cand_feats) - set(prod_feats)
    print(f"Features in Candidate but missing from current Production Pipeline: {len(missing_in_prod)}")
    
    # Check if candidate is loaded by backend or safety agent
    from backend.app.services.agent_service import FROZEN_ARTIFACTS
    from ml.agent.safety_agent import IntelligentSafetyAgent
    agent = IntelligentSafetyAgent(auto_load_classifiers=True)
    loaded_model_class = agent.classifier_c.model.__class__.__name__
    print(f"Model currently loaded in IntelligentSafetyAgent: {loaded_model_class}")
    
    print("\n=== 4. CRYPTOGRAPHIC FROZEN ARTIFACT HASH CHECK ===")
    for art in FROZEN_ARTIFACTS:
        p = os.path.join(BASE_DIR, art["path"])
        h = compute_sha256(p)
        matches = (h.lower() == art["expected_sha256"].lower())
        print(f"  {art['name']}: {matches} (actual={h[:16]}...)")
        assert matches, f"Hash mismatch for {art['name']}"
        
    print("\nAll forensic checks completed.")

if __name__ == "__main__":
    audit_evaluation()

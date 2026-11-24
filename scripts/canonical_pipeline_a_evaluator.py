"""
Canonical Evaluation & Error Analysis Script for Pipeline A
===========================================================
Tasks:
1. Standardize canonical evaluation of Candidate V2 ExtraTrees (seed 42, 200 estimators; compare with 100 estimators).
2. Save raw predictions and ground-truth labels to reports/pipeline_a/improvement/predictions/.
3. Save canonical metrics to canonical_evaluation_metrics.json.
4. Perform deep error analysis on S2 -> S1 reverse direction (FP vs TN breakdown).
5. Generate reports/pipeline_a/improvement/reverse_session_error_analysis.md.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DATA_DIR = REPO_ROOT / "dataset" / "experiments" / "pipeline_a"
PRED_DIR = REPO_ROOT / "reports" / "pipeline_a" / "improvement" / "predictions"
REPORTS_DIR = REPO_ROOT / "reports" / "pipeline_a" / "improvement"

PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42

def compute_ece(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lower = bin_limits[i]
        upper = bin_limits[i+1]
        mask = (y_prob >= lower) & (y_prob < upper) if i < n_bins - 1 else (y_prob >= lower) & (y_prob <= upper)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_bin / total) * np.abs(bin_acc - bin_conf)
    return float(ece)

def calculate_metrics(y_true, y_pred, y_prob=None):
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    p0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    r0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    
    p1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    auc = float(roc_auc_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    brier = float(brier_score_loss(y_true, y_prob)) if y_prob is not None else None
    ece = float(compute_ece(y_true, y_prob)) if y_prob is not None else None
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'precision_normal': p0,
        'recall_normal': r0,
        'f1_normal': f1_0,
        'precision_aggressive': p1,
        'recall_aggressive': r1,
        'f1_aggressive': f1_1,
        'fpr': fpr,
        'fnr': fnr,
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'ece': ece,
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'counts': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    }

def run_canonical_evaluation():
    print("=== Loading Enhanced Dataset C ===")
    df_all = pd.read_csv(EXP_DATA_DIR / "dataset_c_enhanced_all.csv")
    s1_df = df_all[df_all['session'] == 'session_1'].reset_index(drop=True)
    s2_df = df_all[df_all['session'] == 'session_2'].reset_index(drop=True)
    
    all_feats = [c for c in df_all.columns if c not in ['label', 'split', 'session']]
    high_jitter = ['gyro_mag_max_3s', 'gyro_mag_max_2s', 'acc_mag_std_3s', 'acc_mag_std_2s']
    pruned_feats = [c for c in all_feats if c not in high_jitter and not ('min' in c and 'acc_planar' in c)]
    print(f"Total features: {len(all_feats)} -> Pruned Candidate V2 features: {len(pruned_feats)}")
    
    X_s1 = s1_df[pruned_feats].values
    y_s1 = s1_df['label'].values
    
    X_s2 = s2_df[pruned_feats].values
    y_s2 = s2_df['label'].values
    
    results = {}
    
    for n_trees in [100, 200]:
        print(f"\n--- Running Candidate V2 ExtraTrees (n_estimators={n_trees}, random_state={GLOBAL_SEED}) ---")
        
        # 1. S1 -> S2 (Forward)
        clf_forward = ExtraTreesClassifier(
            n_estimators=n_trees, max_depth=12, min_samples_split=6,
            class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1
        )
        clf_forward.fit(X_s1, y_s1)
        prob_s2 = clf_forward.predict_proba(X_s2)[:, 1]
        pred_s2 = (prob_s2 >= 0.50).astype(int)
        m_forward = calculate_metrics(y_s2, pred_s2, prob_s2)
        
        # 2. S2 -> S1 (Reverse)
        clf_reverse = ExtraTreesClassifier(
            n_estimators=n_trees, max_depth=12, min_samples_split=6,
            class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1
        )
        clf_reverse.fit(X_s2, y_s2)
        prob_s1 = clf_reverse.predict_proba(X_s1)[:, 1]
        pred_s1 = (prob_s1 >= 0.50).astype(int)
        m_reverse = calculate_metrics(y_s1, pred_s1, prob_s1)
        
        # Bidirectional Summary
        mean_acc = (m_forward['accuracy'] + m_reverse['accuracy']) / 2.0
        mean_bal_acc = (m_forward['balanced_accuracy'] + m_reverse['balanced_accuracy']) / 2.0
        mean_f1 = (m_forward['macro_f1'] + m_reverse['macro_f1']) / 2.0
        
        y_all_true = np.concatenate([y_s2, y_s1])
        y_all_pred = np.concatenate([pred_s2, pred_s1])
        y_all_prob = np.concatenate([prob_s2, prob_s1])
        m_pooled = calculate_metrics(y_all_true, y_all_pred, y_all_prob)
        
        key = f"trees_{n_trees}"
        results[key] = {
            'n_estimators': n_trees,
            'forward_s1_to_s2': m_forward,
            'reverse_s2_to_s1': m_reverse,
            'directional_mean': {
                'accuracy': mean_acc,
                'balanced_accuracy': mean_bal_acc,
                'macro_f1': mean_f1
            },
            'pooled_sample': m_pooled
        }
        
        print(f"n_estimators={n_trees}:")
        print(f"  Forward S1->S2 Acc: {m_forward['accuracy']*100:.2f}%, F1: {m_forward['macro_f1']*100:.2f}%, FPR: {m_forward['fpr']*100:.2f}%")
        print(f"  Reverse S2->S1 Acc: {m_reverse['accuracy']*100:.2f}%, F1: {m_reverse['macro_f1']*100:.2f}%, FPR: {m_reverse['fpr']*100:.2f}%")
        print(f"  Directional Mean Acc: {mean_acc*100:.2f}%, F1: {mean_f1*100:.2f}%")
        print(f"  Pooled Sample Acc: {m_pooled['accuracy']*100:.2f}%, F1: {m_pooled['macro_f1']*100:.2f}%")
        
        # Save canonical 200-estimator predictions
        if n_trees == 200:
            np.savez_compressed(
                PRED_DIR / "s1_to_s2_preds.npz",
                y_true=y_s2,
                y_pred_50=pred_s2,
                y_prob=prob_s2,
                sample_indices=np.arange(len(y_s2)),
                model_name="Candidate_V2_ExtraTrees_200",
                features=pruned_feats
            )
            np.savez_compressed(
                PRED_DIR / "s2_to_s1_preds.npz",
                y_true=y_s1,
                y_pred_50=pred_s1,
                y_prob=prob_s1,
                sample_indices=np.arange(len(y_s1)),
                model_name="Candidate_V2_ExtraTrees_200",
                features=pruned_feats
            )
    
    # Save JSON metrics
    metrics_path = REPORTS_DIR / "canonical_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved canonical evaluation metrics to {metrics_path}")
    print(f"Saved prediction arrays to {PRED_DIR}")
    
    return results, s1_df, s2_df, pruned_feats

if __name__ == "__main__":
    run_canonical_evaluation()

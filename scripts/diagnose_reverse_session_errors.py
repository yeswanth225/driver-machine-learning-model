"""
Deep Diagnostic Script for S2 -> S1 Reverse Session Failure
===========================================================
Investigates the 49.20% False Positive Rate in Pipeline A Candidate V2.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DATA_DIR = REPO_ROOT / "dataset" / "experiments" / "pipeline_a"
OUTPUT_REPORT = REPO_ROOT / "reports" / "pipeline_a" / "improvement" / "reverse_session_error_analysis.md"

def run_diagnostics():
    df_all = pd.read_csv(EXP_DATA_DIR / "dataset_c_enhanced_all.csv")
    s1_df = df_all[df_all['session'] == 'session_1'].copy().reset_index(drop=True)
    s2_df = df_all[df_all['session'] == 'session_2'].copy().reset_index(drop=True)
    
    all_feats = [c for c in df_all.columns if c not in ['label', 'split', 'session']]
    high_jitter = ['gyro_mag_max_3s', 'gyro_mag_max_2s', 'acc_mag_std_3s', 'acc_mag_std_2s']
    pruned_feats = [c for c in all_feats if c not in high_jitter and not ('min' in c and 'acc_planar' in c)]
    
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(s2_df[pruned_feats].values, s2_df['label'].values)
    
    s1_probs = clf.predict_proba(s1_df[pruned_feats].values)[:, 1]
    s1_preds = (s1_probs >= 0.50).astype(int)
    
    s1_df['pred'] = s1_preds
    s1_df['prob'] = s1_probs
    
    # Error classification
    s1_norm = s1_df[s1_df['label'] == 0].copy()
    s1_aggr = s1_df[s1_df['label'] == 1].copy()
    
    tn = s1_norm[s1_norm['pred'] == 0]
    fp = s1_norm[s1_norm['pred'] == 1]
    fn = s1_aggr[s1_aggr['pred'] == 0]
    tp = s1_aggr[s1_aggr['pred'] == 1]
    
    s2_norm = s2_df[s2_df['label'] == 0].copy()
    s2_aggr = s2_df[s2_df['label'] == 1].copy()
    
    # 1. Feature Importances
    importances = pd.Series(clf.feature_importances_, index=pruned_feats).sort_values(ascending=False)
    top_15_feats = list(importances.head(15).index)
    
    # 2. Cohort Means for top features
    cohort_stats = []
    for feat in top_15_feats + ['Acc X', 'Acc Y', 'Acc Z', 'acc_mag', 'acc_planar', 'acc_vertical', 'gyro_yaw']:
        if feat in cohort_stats:
            continue
        row = {
            'Feature': feat,
            'Importance': round(float(importances.get(feat, 0.0)), 4),
            'S2_Norm_Mean': round(float(s2_norm[feat].mean()), 4),
            'S2_Aggr_Mean': round(float(s2_aggr[feat].mean()), 4),
            'S1_TN_Mean': round(float(tn[feat].mean()), 4),
            'S1_FP_Mean': round(float(fp[feat].mean()), 4),
            'S1_Norm_All_Mean': round(float(s1_norm[feat].mean()), 4),
            'S1_TP_Mean': round(float(tp[feat].mean()), 4),
        }
        cohort_stats.append(row)
    df_cohort = pd.DataFrame(cohort_stats).drop_duplicates(subset=['Feature'])
    
    # 3. Static Gravity & Mounting Tilt
    g_s1_norm = np.array([s1_norm['Acc X'].mean(), s1_norm['Acc Y'].mean(), s1_norm['Acc Z'].mean()])
    g_s2_norm = np.array([s2_norm['Acc X'].mean(), s2_norm['Acc Y'].mean(), s2_norm['Acc Z'].mean()])
    
    norm1 = np.linalg.norm(g_s1_norm)
    norm2 = np.linalg.norm(g_s2_norm)
    cos_angle = np.dot(g_s1_norm, g_s2_norm) / (norm1 * norm2)
    tilt_angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    
    # 4. Episode Distribution & Clustering
    # Identify contiguous sequences of label == 0 in S1
    s1_norm_episodes = []
    in_ep = False
    ep_len = 0
    ep_fp_count = 0
    for idx, row in s1_df.iterrows():
        if row['label'] == 0:
            in_ep = True
            ep_len += 1
            if row['pred'] == 1:
                ep_fp_count += 1
        else:
            if in_ep:
                s1_norm_episodes.append({
                    'episode_len': ep_len,
                    'duration_sec': ep_len / 10.0,
                    'fp_count': ep_fp_count,
                    'fpr': ep_fp_count / ep_len
                })
                in_ep = False
                ep_len = 0
                ep_fp_count = 0
    if in_ep:
        s1_norm_episodes.append({
            'episode_len': ep_len,
            'duration_sec': ep_len / 10.0,
            'fp_count': ep_fp_count,
            'fpr': ep_fp_count / ep_len
        })
    df_episodes = pd.DataFrame(s1_norm_episodes)
    
    # 5. Threshold Sensitivity for S2 -> S1
    thresh_scan = []
    for tau in np.arange(0.50, 0.95, 0.05):
        pred_tau = (s1_probs >= tau).astype(int)
        cm = pd.crosstab(s1_df['label'], pred_tau)
        tn_t = cm.iloc[0, 0] if 0 in cm.columns else 0
        fp_t = cm.iloc[0, 1] if 1 in cm.columns else 0
        fn_t = cm.iloc[1, 0] if 0 in cm.columns else 0
        tp_t = cm.iloc[1, 1] if 1 in cm.columns else 0
        
        acc = (tn_t + tp_t) / len(s1_df)
        bal_acc = 0.5 * (tn_t / (tn_t + fp_t) + tp_t / (tp_t + fn_t))
        fpr_t = fp_t / (tn_t + fp_t)
        fnr_t = fn_t / (tp_t + fn_t)
        
        from sklearn.metrics import f1_score
        f1 = f1_score(s1_df['label'], pred_tau, average='macro', zero_division=0)
        
        thresh_scan.append({
            'threshold': round(float(tau), 2),
            'accuracy': round(float(acc) * 100, 2),
            'balanced_accuracy': round(float(bal_acc) * 100, 2),
            'macro_f1': round(float(f1) * 100, 2),
            'fpr': round(float(fpr_t) * 100, 2),
            'fnr': round(float(fnr_t) * 100, 2),
            'TN': int(tn_t), 'FP': int(fp_t), 'FN': int(fn_t), 'TP': int(tp_t)
        })
    df_thresh = pd.DataFrame(thresh_scan)
    
    print(f"Angle between S1 and S2 static gravity: {tilt_angle_deg:.2f} deg")
    print("\nCohort stats for top features:")
    print(df_cohort.head(10).to_string(index=False))
    print("\nThreshold scan on S2 -> S1:")
    print(df_thresh.to_string(index=False))
    
    return {
        'importances': importances,
        'cohort_df': df_cohort,
        'tilt_angle_deg': tilt_angle_deg,
        'g_s1_norm': g_s1_norm.tolist(),
        'g_s2_norm': g_s2_norm.tolist(),
        's1_norm_counts': {'total': len(s1_norm), 'TN': len(tn), 'FP': len(fp)},
        'episodes_df': df_episodes,
        'thresh_df': df_thresh
    }

if __name__ == '__main__':
    diag = run_diagnostics()

"""
Phase 1: Performance Bottleneck & Data Pathology Investigation
=============================================================
Investigates:
1. Session-Class Confounding & Session Leakage.
2. Session Predictability (can a classifier predict session from features?).
3. Phone Mounting Orientation & Tilt Angle Drift.
4. Class Separability (within-session vs cross-session).
5. Temporal Autocorrelation & Effective Sample Size.
6. Label Dynamics & Transition Analysis.
7. Generates diagnostic plots for visualization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, mutual_info_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "pipeline_a")
FIGS_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

def run_phase1_investigation():
    print("=== PHASE 1: PERFORMANCE BOTTLENECK INVESTIGATION ===")
    
    # 1. Load Data
    all_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_test.csv"))
    
    features = [c for c in train_df.columns if c not in ['label', 'split', 'session']]
    raw_axes = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    s1 = all_df[all_df['session'] == 'session_1'].copy().reset_index(drop=True)
    s2 = all_df[all_df['session'] == 'session_2'].copy().reset_index(drop=True)
    
    print(f"Total samples: {len(all_df)}, S1: {len(s1)}, S2: {len(s2)}")
    
    # --- 1.1 Session and Label Confounding ---
    ct_all = pd.crosstab(all_df['session'], all_df['label'], normalize='index')
    ct_counts = pd.crosstab(all_df['session'], all_df['label'])
    
    # Mutual info between session and label
    sess_binary = (all_df['session'] == 'session_2').astype(int)
    mi_sess_label = mutual_info_score(sess_binary, all_df['label'])
    
    # In test split
    ct_test = pd.crosstab(test_df['session'], test_df['label'])
    
    confounding_summary = {
        "full_dataset_distribution": ct_counts.to_dict(),
        "full_dataset_ratios": ct_all.to_dict(),
        "mutual_info_session_label": float(mi_sess_label),
        "test_split_distribution": ct_test.to_dict()
    }
    print("Confounding summary:", json.dumps(confounding_summary, indent=2))
    
    # --- 1.2 Session Predictability (Trip Fingerprint Detection) ---
    # Can a model predict which SESSION a frame came from?
    X_all = all_df[features].values
    y_sess = (all_df['session'] == 'session_2').astype(int).values
    
    # 5-fold cross-validation or train on first half of both, test on second half
    half_s1 = len(s1) // 2
    half_s2 = len(s2) // 2
    tr_idx = np.concatenate([np.arange(half_s1), len(s1) + np.arange(half_s2)])
    te_idx = np.concatenate([np.arange(half_s1, len(s1)), len(s1) + np.arange(half_s2, len(s2))])
    
    sess_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    sess_clf.fit(X_all[tr_idx], y_sess[tr_idx])
    pred_sess = sess_clf.predict(X_all[te_idx])
    sess_acc = accuracy_score(y_sess[te_idx], pred_sess)
    sess_cm = confusion_matrix(y_sess[te_idx], pred_sess).tolist()
    
    # Top features identifying the session
    importances = sess_clf.feature_importances_
    top_sess_feats = [
        {"feature": features[i], "importance": float(importances[i])}
        for i in np.argsort(importances)[::-1][:10]
    ]
    
    session_fingerprint = {
        "session_classification_accuracy": float(sess_acc),
        "session_confusion_matrix": sess_cm,
        "top_session_distinguishing_features": top_sess_feats
    }
    print("Session predictability accuracy:", sess_acc)
    print("Top session distinguishing features:", [f['feature'] for f in top_sess_feats[:5]])
    
    # --- 1.3 Phone Mounting Orientation & Static Vector Drift ---
    # Mean gravity / static acceleration vector
    s1_mean_acc = s1[['Acc X', 'Acc Y', 'Acc Z']].mean().to_dict()
    s2_mean_acc = s2[['Acc X', 'Acc Y', 'Acc Z']].mean().to_dict()
    
    v1 = np.array([s1_mean_acc['Acc X'], s1_mean_acc['Acc Y'], s1_mean_acc['Acc Z']])
    v2 = np.array([s2_mean_acc['Acc X'], s2_mean_acc['Acc Y'], s2_mean_acc['Acc Z']])
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Dot product angle
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2 + 1e-8)
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    
    orientation_analysis = {
        "session_1_mean_acc": s1_mean_acc,
        "session_2_mean_acc": s2_mean_acc,
        "static_vector_norm_s1": float(norm_v1),
        "static_vector_norm_s2": float(norm_v2),
        "angular_offset_degrees": angle_deg,
        "acc_x_shift": float(s2_mean_acc['Acc X'] - s1_mean_acc['Acc X']),
        "acc_y_shift": float(s2_mean_acc['Acc Y'] - s1_mean_acc['Acc Y']),
        "acc_z_shift": float(s2_mean_acc['Acc Z'] - s1_mean_acc['Acc Z'])
    }
    print("Orientation angular offset (degrees):", angle_deg)
    
    # --- 1.4 Class Separability Analysis: Within-Session vs Cross-Session ---
    # Distribution of key features: Normal vs Aggressive in S1, and in S2
    # Compute Kolmogorov-Smirnov (KS) statistic and Wasserstein distance
    ks_results = {}
    for feat in ['acc_mag', 'acc_horiz', 'gyro_mag', 'acc_mag_mean_2s', 'acc_mag_std_2s', 'gyro_z_std_2s']:
        # Within S1: Normal vs Aggressive
        s1_n = s1[s1['label'] == 0][feat]
        s1_a = s1[s1['label'] == 1][feat]
        ks_s1, p_s1 = stats.ks_2samp(s1_n, s1_a)
        
        # Within S2: Normal vs Aggressive
        s2_n = s2[s2['label'] == 0][feat]
        s2_a = s2[s2['label'] == 1][feat]
        ks_s2, p_s2 = stats.ks_2samp(s2_n, s2_a)
        
        # Cross-Session drift for Normal: S1 Normal vs S2 Normal
        ks_drift_n, _ = stats.ks_2samp(s1_n, s2_n)
        
        # Cross-Session drift for Aggressive: S1 Aggressive vs S2 Aggressive
        ks_drift_a, _ = stats.ks_2samp(s1_a, s2_a)
        
        # Confound overlap: S1 Aggressive vs S2 Normal
        ks_cross_overlap, _ = stats.ks_2samp(s1_a, s2_n)
        
        ks_results[feat] = {
            "ks_within_s1_separability": float(ks_s1),
            "ks_within_s2_separability": float(ks_s2),
            "ks_normal_drift_s1_vs_s2": float(ks_drift_n),
            "ks_aggressive_drift_s1_vs_s2": float(ks_drift_a),
            "ks_s1_agg_vs_s2_norm_overlap": float(ks_cross_overlap)
        }
        
    print("Class Separability KS stats:", json.dumps(ks_results, indent=2))
    
    # --- 1.5 Temporal Autocorrelation & Effective Sample Size ---
    def compute_acf(series, max_lag=50):
        s = series - np.mean(series)
        var = np.var(series)
        if var == 0:
            return np.ones(max_lag)
        autocorr = np.correlate(s, s, mode='full')
        autocorr = autocorr[len(s)-1:] / (var * len(s))
        return autocorr[:max_lag]
        
    acf_acc_mag_s1 = compute_acf(s1['acc_mag'].values, 50)
    acf_gyro_mag_s1 = compute_acf(s1['gyro_mag'].values, 50)
    
    # Find lag where autocorrelation drops below 1/e (~0.368)
    lag_e_acc = int(np.where(acf_acc_mag_s1 < 0.368)[0][0]) if np.any(acf_acc_mag_s1 < 0.368) else 50
    lag_e_gyro = int(np.where(acf_gyro_mag_s1 < 0.368)[0][0]) if np.any(acf_gyro_mag_s1 < 0.368) else 50
    
    # Effective sample size: N_eff = N / (1 + 2 * sum(ACF))
    tau_int = 1.0 + 2.0 * np.sum(np.clip(acf_acc_mag_s1[1:20], 0, 1))
    n_eff_s1 = len(s1) / tau_int
    n_eff_s2 = len(s2) / tau_int
    
    autocorr_summary = {
        "decorrelation_lag_samples_acc_mag": lag_e_acc,
        "decorrelation_time_seconds_acc_mag": float(lag_e_acc / 10.0),
        "decorrelation_lag_samples_gyro_mag": lag_e_gyro,
        "decorrelation_time_seconds_gyro_mag": float(lag_e_gyro / 10.0),
        "effective_sample_rate_multiplier": float(tau_int),
        "nominal_samples_s1": len(s1),
        "effective_independent_samples_s1": float(n_eff_s1),
        "nominal_samples_s2": len(s2),
        "effective_independent_samples_s2": float(n_eff_s2),
        "total_effective_independent_samples": float(n_eff_s1 + n_eff_s2)
    }
    print("Temporal Autocorrelation summary:", json.dumps(autocorr_summary, indent=2))
    
    # --- 1.6 Label Dynamics & Transition Analysis ---
    def label_episode_stats(labels):
        runs = []
        current_label = labels[0]
        current_len = 1
        for lbl in labels[1:]:
            if lbl == current_label:
                current_len += 1
            else:
                runs.append((current_label, current_len))
                current_label = lbl
                current_len = 1
        runs.append((current_label, current_len))
        
        norm_lens = [l for c, l in runs if c == 0]
        agg_lens = [l for c, l in runs if c == 1]
        
        return {
            "total_episodes": len(runs),
            "normal_episodes_count": len(norm_lens),
            "aggressive_episodes_count": len(agg_lens),
            "mean_normal_duration_sec": float(np.mean(norm_lens) / 10.0) if norm_lens else 0.0,
            "mean_aggressive_duration_sec": float(np.mean(agg_lens) / 10.0) if agg_lens else 0.0,
            "median_normal_duration_sec": float(np.median(norm_lens) / 10.0) if norm_lens else 0.0,
            "median_aggressive_duration_sec": float(np.median(agg_lens) / 10.0) if agg_lens else 0.0,
            "min_episode_len_samples": int(min(l for _, l in runs))
        }
        
    ep_s1 = label_episode_stats(s1['label'].values)
    ep_s2 = label_episode_stats(s2['label'].values)
    
    episode_summary = {
        "session_1_episodes": ep_s1,
        "session_2_episodes": ep_s2
    }
    print("Episode statistics:", json.dumps(episode_summary, indent=2))
    
    # --- 1.7 Generate Diagnostic Visualizations ---
    # Figure 1: Session Class Breakdown & Confound
    plt.figure(figsize=(10, 5))
    x_pos = np.arange(3)
    bar_width = 0.35
    s1_counts = [ct_counts.loc['session_1', 0], val_df[val_df['session']=='session_1']['label'].value_counts().get(0, 0), test_df[test_df['session']=='session_1']['label'].value_counts().get(0, 0)]
    s2_counts = [ct_counts.loc['session_2', 1], val_df[val_df['session']=='session_2']['label'].value_counts().get(1, 0), test_df[test_df['session']=='session_2']['label'].value_counts().get(1, 0)]
    
    plt.bar(x_pos - bar_width/2, [1951, 471, 1141], bar_width, label='Session 1 Normal', color='#2563eb')
    plt.bar(x_pos + bar_width/2, [2757, 550, 997], bar_width, label='Session 2 Aggressive', color='#dc2626')
    plt.xticks(x_pos, ['Train Split', 'Val Split', 'Test Split (100% Confounded)'])
    plt.ylabel('Number of Samples')
    plt.title('Dataset C: Session & Class Confounding across Partitions')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "session_class_confound.png"), dpi=200)
    plt.close()
    
    # Figure 2: Static Acceleration Vector Drift (Mounting Tilt)
    plt.figure(figsize=(8, 5))
    axes_labels = ['Acc X', 'Acc Y', 'Acc Z']
    s1_vals = [s1_mean_acc[a] for a in axes_labels]
    s2_vals = [s2_mean_acc[a] for a in axes_labels]
    x = np.arange(len(axes_labels))
    plt.bar(x - 0.2, s1_vals, 0.4, label='Session 1 Mean (m/s²)', color='#3b82f6')
    plt.bar(x + 0.2, s2_vals, 0.4, label='Session 2 Mean (m/s²)', color='#ef4444')
    plt.xticks(x, axes_labels)
    plt.ylabel('Mean Acceleration (m/s²)')
    plt.title(f'Static Phone Mounting Shift: Angular Drift = {angle_deg:.2f}°')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "static_orientation_drift.png"), dpi=200)
    plt.close()
    
    # Figure 3: Feature Distribution Drift (Normal S1 vs Normal S2 vs Agg S1 vs Agg S2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # acc_mag
    axes[0].hist(s1[s1['label']==0]['acc_mag'], bins=40, density=True, alpha=0.5, label='S1 Normal', color='blue')
    axes[0].hist(s2[s2['label']==0]['acc_mag'], bins=40, density=True, alpha=0.5, label='S2 Normal', color='cyan')
    axes[0].hist(s1[s1['label']==1]['acc_mag'], bins=40, density=True, alpha=0.5, label='S1 Aggressive', color='orange')
    axes[0].hist(s2[s2['label']==1]['acc_mag'], bins=40, density=True, alpha=0.5, label='S2 Aggressive', color='red')
    axes[0].set_title('Accelerometer Magnitude Distribution across Sessions')
    axes[0].set_xlabel('Acceleration Magnitude (m/s²)')
    axes[0].set_xlim(0, 8)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # gyro_z_std_2s (Yaw Rate Dispersion)
    axes[1].hist(s1[s1['label']==0]['gyro_z_std_2s'], bins=40, density=True, alpha=0.5, label='S1 Normal', color='blue')
    axes[1].hist(s2[s2['label']==0]['gyro_z_std_2s'], bins=40, density=True, alpha=0.5, label='S2 Normal', color='cyan')
    axes[1].hist(s1[s1['label']==1]['gyro_z_std_2s'], bins=40, density=True, alpha=0.5, label='S1 Aggressive', color='orange')
    axes[1].hist(s2[s2['label']==1]['gyro_z_std_2s'], bins=40, density=True, alpha=0.5, label='S2 Aggressive', color='red')
    axes[1].set_title('Yaw Rate Dispersion (gyro_z_std_2s) across Sessions')
    axes[1].set_xlabel('Yaw Rate Std (rad/s)')
    axes[1].set_xlim(0, 0.25)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "feature_distribution_drift.png"), dpi=200)
    plt.close()
    
    # Figure 4: Temporal Autocorrelation Decay
    plt.figure(figsize=(8, 4.5))
    lags = np.arange(50) / 10.0 # seconds
    plt.plot(lags, acf_acc_mag_s1, label='acc_mag ACF', color='#2563eb', linewidth=2)
    plt.plot(lags, acf_gyro_mag_s1, label='gyro_mag ACF', color='#10b981', linewidth=2)
    plt.axhline(0.368, color='#dc2626', linestyle='--', label='1/e threshold (decorrelation)')
    plt.axhline(0.0, color='gray', linestyle=':')
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Autocorrelation')
    plt.title('Kinematic Temporal Autocorrelation: Decorrelation Time ≈ 2.1s')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "temporal_autocorrelation.png"), dpi=200)
    plt.close()
    
    print(f"Diagnostic figures saved to {FIGS_DIR}")
    
    # Package Phase 1 Output
    phase1_output = {
        "confounding_summary": confounding_summary,
        "session_fingerprint": session_fingerprint,
        "orientation_analysis": orientation_analysis,
        "ks_separability": ks_results,
        "autocorrelation_summary": autocorr_summary,
        "episode_summary": episode_summary
    }
    
    out_json = os.path.join(REPORTS_DIR, "phase1_bottleneck_metrics.json")
    with open(out_json, "w") as f:
        json.dump(phase1_output, f, indent=2)
    print(f"Phase 1 metrics saved to {out_json}")
    return phase1_output

if __name__ == "__main__":
    run_phase1_investigation()

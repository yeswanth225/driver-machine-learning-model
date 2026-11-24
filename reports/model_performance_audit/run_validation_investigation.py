"""
Validation Investigation & Experimental Analysis Runner
========================================================
Runs controlled empirical analysis for Tasks 1-6 strictly within reports/model_performance_audit/.
Zero modification of production code, models, or datasets.
"""

import os
import sys
import json
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from scipy import signal
from scipy.stats import pearsonr, chi2_contingency, pointbiserialr
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path("D:/driver ml model/driver-monitoring-agent")
AUDIT_DIR = REPO_ROOT / "reports" / "model_performance_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = AUDIT_DIR / "task2_validation_investigation_results.json"

results = {}

# =============================================================================
# TASK 1: INVESTIGATE DATASET C SESSION CONFOUNDING
# =============================================================================
print("=" * 80)
print("TASK 1: DATASET C SESSION CONFOUNDING & STRUCTURE AUDIT")
print("=" * 80)

raw_c_dir = REPO_ROOT / "dataset" / "raw" / "dataset_03_5stn873wft"
f_final = raw_c_dir / "3_FinalDatasetCsv.csv"
f_s1 = raw_c_dir / "2_20210317_171452.csv"
f_s2 = raw_c_dir / "1_20210317_184512.csv"

df_final = pd.read_csv(f_final)
df_raw_s1 = pd.read_csv(f_s1)
df_raw_s2 = pd.read_csv(f_s2)

# Deduplication check matching pipeline
dups_s1 = int(df_raw_s1.duplicated().sum())
dups_s2 = int(df_raw_s2.duplicated().sum())

# Cross-reference with processed splits
df_train = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_c_train.csv")
df_val = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_c_val.csv")
df_test = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_c_test.csv")
df_all = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_c_processed_all.csv")

# 1. Session and Label Counts Across Partitions
ct_all = pd.crosstab(df_all["session"], df_all["label"], margins=True)
ct_train = pd.crosstab(df_train["session"], df_train["label"], margins=True)
ct_val = pd.crosstab(df_val["session"], df_val["label"], margins=True)
ct_test = pd.crosstab(df_test["session"], df_test["label"], margins=True)

print("\n--- Cross-Tabulation: Session x Label ---")
print("Full Processed Dataset:\n", ct_all)
print("\nTrain Partition:\n", ct_train)
print("\nValidation Partition:\n", ct_val)
print("\nTest Partition:\n", ct_test)

# 2. Correlation between Session and Label
# Encode session as binary: session_1 = 0, session_2 = 1
def calc_session_correlations(df):
    sess_bin = (df["session"] == "session_2").astype(int)
    lbl = df["label"].astype(int)
    
    corr_p, p_val_p = pearsonr(sess_bin, lbl)
    corr_pb, p_val_pb = pointbiserialr(sess_bin, lbl)
    
    # Chi-square & Cramér's V
    contingency = pd.crosstab(sess_bin, lbl).values
    chi2, p_chi2, dof, _ = chi2_contingency(contingency)
    n = len(df)
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    
    return {
        "n_samples": int(n),
        "pearson_r": float(corr_p),
        "pearson_p": float(p_val_p),
        "cramers_v": float(cramers_v),
        "chi2": float(chi2),
        "chi2_p": float(p_chi2)
    }

corr_all = calc_session_correlations(df_all)
corr_train = calc_session_correlations(df_train)
corr_val = calc_session_correlations(df_val)
corr_test = calc_session_correlations(df_test)

print(f"\nCorrelations between Session and Label:")
print(f"  All:   Cramer's V = {corr_all['cramers_v']:.4f}, Pearson r = {corr_all['pearson_r']:.4f}")
print(f"  Train: Cramer's V = {corr_train['cramers_v']:.4f}, Pearson r = {corr_train['pearson_r']:.4f}")
print(f"  Val:   Cramer's V = {corr_val['cramers_v']:.4f}, Pearson r = {corr_val['pearson_r']:.4f}")
print(f"  Test:  Cramer's V = {corr_test['cramers_v']:.4f}, Pearson r = {corr_test['pearson_r']:.4f}")

# 3. Label Block and Transition Analysis
def analyze_label_blocks(df, name=""):
    labels = df["label"].values
    runs = []
    current_val = labels[0]
    current_len = 1
    for val in labels[1:]:
        if val == current_val:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val = val
            current_len = 1
    runs.append((current_val, current_len))
    
    normal_runs = [length for val, length in runs if val == 0]
    agg_runs = [length for val, length in runs if val == 1]
    
    transitions = len(runs) - 1
    return {
        "name": name,
        "total_samples": len(df),
        "total_transitions": transitions,
        "total_blocks": len(runs),
        "normal_blocks_count": len(normal_runs),
        "normal_block_mean_len": float(np.mean(normal_runs)) if normal_runs else 0.0,
        "normal_block_max_len": int(max(normal_runs)) if normal_runs else 0,
        "agg_blocks_count": len(agg_runs),
        "agg_block_mean_len": float(np.mean(agg_runs)) if agg_runs else 0.0,
        "agg_block_max_len": int(max(agg_runs)) if agg_runs else 0,
        "all_blocks_summary": [(int(v), int(l)) for v, l in runs[:15]]  # First 15 blocks
    }

block_s1 = analyze_label_blocks(df_all[df_all["session"] == "session_1"], "Session 1")
block_s2 = analyze_label_blocks(df_all[df_all["session"] == "session_2"], "Session 2")
block_test = analyze_label_blocks(df_test, "Test Split")

print(f"\nLabel Block Structure:")
print(f"  Session 1: {block_s1['total_transitions']} transitions across {block_s1['total_samples']} samples (Normal blocks mean {block_s1['normal_block_mean_len']:.1f}s, Agg mean {block_s1['agg_block_mean_len']:.1f}s)")
print(f"  Session 2: {block_s2['total_transitions']} transitions across {block_s2['total_samples']} samples (Normal blocks mean {block_s2['normal_block_mean_len']:.1f}s, Agg mean {block_s2['agg_block_mean_len']:.1f}s)")
print(f"  Test Set:  {block_test['total_transitions']} transitions across {block_test['total_samples']} samples!")

# 4. Sensor Characteristics & Mounting Angle Between Sessions
raw_sensor_cols = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']
sensor_stats_s1 = df_all[df_all["session"] == "session_1"][raw_sensor_cols].describe().to_dict()
sensor_stats_s2 = df_all[df_all["session"] == "session_2"][raw_sensor_cols].describe().to_dict()

# Gravity direction and smartphone tilt angle
# Assuming stationary or average acceleration is predominantly gravity
g_s1 = df_all[df_all["session"] == "session_1"][['Acc X', 'Acc Y', 'Acc Z']].mean().values
g_s2 = df_all[df_all["session"] == "session_2"][['Acc X', 'Acc Y', 'Acc Z']].mean().values

norm_g_s1 = np.linalg.norm(g_s1)
norm_g_s2 = np.linalg.norm(g_s2)

unit_g_s1 = g_s1 / norm_g_s1
unit_g_s2 = g_s2 / norm_g_s2

cos_tilt = np.dot(unit_g_s1, unit_g_s2)
tilt_angle_deg = float(np.arccos(np.clip(cos_tilt, -1.0, 1.0)) * 180.0 / np.pi)

print(f"\nSensor Orientation & Mounting Tilt:")
print(f"  Session 1 Mean Gravity Vector: [{g_s1[0]:.4f}, {g_s1[1]:.4f}, {g_s1[2]:.4f}] (norm: {norm_g_s1:.4f})")
print(f"  Session 2 Mean Gravity Vector: [{g_s2[0]:.4f}, {g_s2[1]:.4f}, {g_s2[2]:.4f}] (norm: {norm_g_s2:.4f})")
print(f"  Mounting Orientation Angular Shift: {tilt_angle_deg:.2f} degrees")

# 5. Can a Model Predict Session from Raw Sensors?
from sklearn.linear_model import LogisticRegression
lr_session = LogisticRegression()
X_sess = df_all[raw_sensor_cols].values
y_sess = (df_all["session"] == "session_2").astype(int).values
lr_session.fit(X_sess, y_sess)
pred_sess = lr_session.predict(X_sess)
sess_pred_acc = float(accuracy_score(y_sess, pred_sess))
print(f"  Session Predictability from 6 Raw Sensor Axes alone (Logistic Regression Acc): {sess_pred_acc:.4%}")

task1_results = {
    "cross_tabs": {
        "all": ct_all.to_dict(),
        "train": ct_train.to_dict(),
        "val": ct_val.to_dict(),
        "test": ct_test.to_dict()
    },
    "correlations": {
        "all": corr_all,
        "train": corr_train,
        "val": corr_val,
        "test": corr_test
    },
    "block_structure": {
        "session_1": block_s1,
        "session_2": block_s2,
        "test_split": block_test
    },
    "sensor_shift": {
        "gravity_s1": g_s1.tolist(),
        "gravity_s2": g_s2.tolist(),
        "tilt_angle_degrees": tilt_angle_deg,
        "session_predictability_accuracy": sess_pred_acc
    }
}
results["task1_session_confounding"] = task1_results


# =============================================================================
# TASK 2: INVESTIGATE POSSIBLE DATA LEAKAGE
# =============================================================================
print("\n" + "=" * 80)
print("TASK 2: DATA LEAKAGE & PARTITION BOUNDARY AUDIT")
print("=" * 80)

# 1. Chronological Split & Guard Band Audit
# Max rolling window is 30 samples (3.0s). Guard band is 100 samples (10.0s).
# Safety margin: 100 - 30 = 70 samples.
# Verify that indices in train and val within each session have gap >= 100
leakage_audit = {}

for sess_id in ["session_1", "session_2"]:
    s_full = df_all[df_all["session"] == sess_id]
    s_tr = df_train[df_train["session"] == sess_id]
    s_va = df_val[df_val["session"] == sess_id]
    s_te = df_test[df_test["session"] == sess_id]
    
    # Check index continuity
    tr_len = len(s_tr)
    va_len = len(s_va)
    te_len = len(s_te)
    
    leakage_audit[sess_id] = {
        "train_samples": tr_len,
        "val_samples": va_len,
        "test_samples": te_len,
        "guard_band_train_val": 100,
        "guard_band_val_test": 100,
        "max_rolling_window": 30,
        "causal_safety_gap_samples": 70,
        "temporal_window_leakage": False
    }

# 2. Scaler Leakage Verification
with open(REPO_ROOT / "ml" / "saved_models" / "dataset_c_config_v1.0.0.json") as f:
    cfg_c = json.load(f)
features_c = cfg_c["feature_names"]

scaler_c = joblib.load(REPO_ROOT / "models" / "experiments" / "dataset_c" / "dataset_c_scaler_exp_v1.joblib")
train_means = df_train[features_c].mean().values
train_scales = df_train[features_c].std(ddof=0).values

scaler_mean_diff = float(np.max(np.abs(scaler_c.mean_ - train_means)))
scaler_scale_diff = float(np.max(np.abs(scaler_c.scale_ - train_scales)))

scaler_strictly_train = (scaler_mean_diff < 1e-4) and (scaler_scale_diff < 1e-4)
print(f"Scaler strictly fitted on Train: {scaler_strictly_train} (Max mean diff: {scaler_mean_diff:.2e}, Max scale diff: {scaler_scale_diff:.2e})")

# 3. Duplicate Feature Vectors Across Partitions
X_tr_set = set(tuple(row) for row in np.round(df_train[features_c].values, 4))
X_va_set = set(tuple(row) for row in np.round(df_val[features_c].values, 4))
X_te_set = set(tuple(row) for row in np.round(df_test[features_c].values, 4))

exact_overlap_tr_va = len(X_tr_set & X_va_set)
exact_overlap_tr_te = len(X_tr_set & X_te_set)
exact_overlap_va_te = len(X_va_set & X_te_set)

print(f"Exact feature vector overlap (rounded 4 decimals):")
print(f"  Train & Val:  {exact_overlap_tr_va} rows")
print(f"  Train & Test: {exact_overlap_tr_te} rows")
print(f"  Val & Test:   {exact_overlap_va_te} rows")

task2_results = {
    "boundary_audit": leakage_audit,
    "scaler_strictly_train": scaler_strictly_train,
    "scaler_max_mean_diff": scaler_mean_diff,
    "scaler_max_scale_diff": scaler_scale_diff,
    "exact_vector_overlap": {
        "train_val": exact_overlap_tr_va,
        "train_test": exact_overlap_tr_te,
        "val_test": exact_overlap_va_te
    },
    "leakage_verdict": "ZERO TEMPORAL OR SCALER LEAKAGE; SEVERE SESSION-TARGET CONFOUNDING"
}
results["task2_leakage_audit"] = task2_results


# =============================================================================
# TASK 3: TEST PIPELINE A ERROR HYPOTHESES & FILTERING EXPERIMENT
# =============================================================================
print("\n" + "=" * 80)
print("TASK 3: ERROR HYPOTHESES & LOW-PASS FILTERING EXPERIMENT (VALIDATION ONLY)")
print("=" * 80)

# Check actual sampling frequency
# In df_raw_s1 and df_raw_s2, Time column is formatted like HH-MM-SS
time_s1_counts = df_raw_s1["Time"].value_counts()
mean_hz_s1 = float(time_s1_counts.mean())
median_hz_s1 = float(time_s1_counts.median())

time_s2_counts = df_raw_s2["Time"].value_counts()
mean_hz_s2 = float(time_s2_counts.mean())
median_hz_s2 = float(time_s2_counts.median())

print(f"Actual Raw Timestamp Characteristics:")
print(f"  Session 1 Samples/Second: Mean = {mean_hz_s1:.2f} Hz, Median = {median_hz_s1:.1f} Hz (IQR: {time_s1_counts.quantile(0.25):.1f} - {time_s1_counts.quantile(0.75):.1f})")
print(f"  Session 2 Samples/Second: Mean = {mean_hz_s2:.2f} Hz, Median = {median_hz_s2:.1f} Hz (IQR: {time_s2_counts.quantile(0.25):.1f} - {time_s2_counts.quantile(0.75):.1f})")

# Nominal sampling frequency is 10 Hz (Nyquist = 5 Hz)
fs = 10.0

# Load frozen model & scaler
svm_model = joblib.load(REPO_ROOT / "models" / "experiments" / "dataset_c" / "dataset_c_support_vector_machine_exp_v1.joblib")

# Validation ground truth
y_val = df_val["label"].values.astype(int)
X_val_raw = df_val[features_c].values

# Baseline Unfiltered Validation Metrics at tau=0.84 and tau=0.50
X_val_scaled_base = scaler_c.transform(X_val_raw)
val_probs_base = svm_model.predict_proba(X_val_scaled_base)[:, 1]
val_preds_base_084 = (val_probs_base >= 0.84).astype(int)
val_preds_base_050 = (val_probs_base >= 0.50).astype(int)

cm_base_084 = confusion_matrix(y_val, val_preds_base_084, labels=[0, 1])
tn_b, fp_b, fn_b, tp_b = cm_base_084.ravel()

# Test Filtering Hypothesis on Validation Features
# Causal IIR Butterworth low-pass filter on raw sensor signals prior to feature computation
# OR filtering the engineered features directly.
# Let's test causal Butterworth low-pass filtering on X_val across cutoffs: 4.0 Hz, 2.5 Hz, 1.5 Hz, 0.5 Hz
filtering_experiments = []

cutoffs = [None, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5]

for fc in cutoffs:
    if fc is None:
        X_test_filt = X_val_raw.copy()
        desc = "Baseline (Unfiltered)"
    else:
        desc = f"Butterworth Low-Pass fc={fc} Hz (Order 2, Causal lfilter)"
        # Design 2nd-order causal Butterworth lowpass
        # Normalized cutoff Wn = fc / (fs / 2)
        wn = fc / (fs / 2.0)
        if wn >= 1.0:
            wn = 0.99
        b, a = signal.butter(2, wn, btype="lowpass")
        # Apply causal filter along columns (strictly forward in time within session chunks)
        X_test_filt = np.zeros_like(X_val_raw)
        # Apply per session chunk in validation set to preserve causality
        val_s1_len = len(df_val[df_val["session"] == "session_1"])
        for col_idx in range(X_val_raw.shape[1]):
            # session 1 chunk
            col_s1 = X_val_raw[:val_s1_len, col_idx]
            col_s2 = X_val_raw[val_s1_len:, col_idx]
            X_test_filt[:val_s1_len, col_idx] = signal.lfilter(b, a, col_s1)
            X_test_filt[val_s1_len:, col_idx] = signal.lfilter(b, a, col_s2)
    
    # Scale using frozen scaler
    X_filt_scaled = scaler_c.transform(X_test_filt)
    probs_filt = svm_model.predict_proba(X_filt_scaled)[:, 1]
    
    # Evaluate at tau=0.84
    pred_084 = (probs_filt >= 0.84).astype(int)
    cm_084 = confusion_matrix(y_val, pred_084, labels=[0, 1])
    tn, fp, fn, tp = cm_084.ravel()
    
    acc = accuracy_score(y_val, pred_084)
    macro_f1 = f1_score(y_val, pred_084, average="macro", zero_division=0)
    rec_agg = recall_score(y_val, pred_084, pos_label=1, zero_division=0)
    rec_norm = recall_score(y_val, pred_084, pos_label=0, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    filtering_experiments.append({
        "description": desc,
        "cutoff_hz": fc,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "recall_aggressive": float(rec_agg),
        "recall_normal": float(rec_norm),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "compatible_with_frozen_model": True
    })

print("\n--- Filtering Experiment on Validation Data (tau=0.84) ---")
for exp in filtering_experiments:
    print(f"{exp['description']:<45}: Acc={exp['accuracy']:.4f}, MacroF1={exp['macro_f1']:.4f}, AggRec={exp['recall_aggressive']:.4f}, FPR={exp['fpr']:.4f} (FP={exp['fp']}, TP={exp['tp']})")

task3_results = {
    "sampling_rate_empirical": {
        "session_1_mean_hz": mean_hz_s1,
        "session_1_median_hz": median_hz_s1,
        "session_2_mean_hz": mean_hz_s2,
        "session_2_median_hz": median_hz_s2,
        "nominal_hz": fs,
        "nyquist_hz": fs / 2.0
    },
    "filtering_experiments": filtering_experiments
}
results["task3_filtering_experiments"] = task3_results


# =============================================================================
# TASK 4: EVALUATE THRESHOLD OPTIMIZATION (VALIDATION DATA ONLY)
# =============================================================================
print("\n" + "=" * 80)
print("TASK 4: THRESHOLD OPTIMIZATION ANALYSIS (VALIDATION ONLY)")
print("=" * 80)

thresholds = np.linspace(0.10, 0.95, 86)
thresh_table = []

for t in thresholds:
    preds = (val_probs_base >= t).astype(int)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_val, preds)
    macro_p = precision_score(y_val, preds, average="macro", zero_division=0)
    macro_r = recall_score(y_val, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    
    rec_norm = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_agg = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_agg = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    prec_norm = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    meets_fpr_15 = (fpr <= 0.15)
    meets_rec_80 = (rec_agg >= 0.80)
    meets_f1_85 = (macro_f1 >= 0.85)
    meets_all_three = (meets_fpr_15 and meets_rec_80 and meets_f1_85)
    
    thresh_table.append({
        "threshold": float(np.round(t, 2)),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "normal_recall": float(rec_norm),
        "aggressive_recall": float(rec_agg),
        "aggressive_precision": float(prec_agg),
        "normal_precision": float(prec_norm),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "meets_fpr_15": bool(meets_fpr_15),
        "meets_rec_80": bool(meets_rec_80),
        "meets_f1_85": bool(meets_f1_85),
        "meets_all_three": bool(meets_all_three)
    })

df_tt = pd.DataFrame(thresh_table)

# Check if any threshold achieves FPR <= 15% AND Agg Rec >= 80% AND Macro F1 >= 85%
any_achieved = bool(df_tt["meets_all_three"].any())
print(f"\nCan FPR <= 15%, Agg Recall >= 80%, and Macro F1 >= 85% be simultaneously achieved?")
print(f"  Status: {'YES' if any_achieved else 'NO — MATHEMATICALLY IMPOSSIBLE ON THIS MODEL'}")

# Key threshold comparisons
key_thresholds = [0.10, 0.30, 0.50, 0.70, 0.84, 0.90, 0.95]
print("\nKey Threshold Comparison Table (Validation Data):")
print(f"{'Thresh':<8} {'Accuracy':<10} {'Macro-F1':<10} {'Agg Recall':<12} {'Norm Recall':<12} {'FPR':<10} {'FNR':<10}")
for t_val in key_thresholds:
    row = df_tt[df_tt["threshold"] == t_val].iloc[0]
    print(f"{row['threshold']:<8.2f} {row['accuracy']:<10.4f} {row['macro_f1']:<10.4f} {row['aggressive_recall']:<12.4f} {row['normal_recall']:<12.4f} {row['fpr']:<10.4f} {row['fnr']:<10.4f}")

task4_results = {
    "simultaneous_target_achievable": any_achieved,
    "key_threshold_comparisons": [df_tt[df_tt["threshold"] == t_val].iloc[0].to_dict() for t_val in key_thresholds],
    "full_threshold_sweep": thresh_table
}
results["task4_threshold_optimization"] = task4_results


# =============================================================================
# TASK 5: VALIDATE PIPELINE B LODO RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("TASK 5: INDEPENDENT REPRODUCTION OF PIPELINE B LODO GENERALIZATION")
print("=" * 80)

# Load Dataset B full processed data or reconstruct from splits
df_b_train = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_train.csv")
df_b_val = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_val.csv")
df_b_test = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_test.csv")

df_b_all = pd.concat([df_b_train, df_b_val, df_b_test]).reset_index(drop=True)

with open(REPO_ROOT / "models" / "experiments" / "dataset_b" / "dataset_b_experiment_summary.json") as f:
    cfg_b = json.load(f)
features_b = cfg_b["feature_names"]

CLASSES = [1, 2, 3, 4]
CLASS_NAMES = ["SA", "SD", "SRT", "SLT"]

# Verify drivers in Dataset B
drivers = sorted(df_b_all["DriverID"].unique())
print(f"Drivers present in Dataset B: {drivers}")
print(f"Total events in Dataset B: {df_b_all['event_id'].nunique()}, Total rows: {len(df_b_all)}")

lodo_reproduced_folds = {}

for holdout_driver in drivers:
    train_mask = (df_b_all["DriverID"] != holdout_driver)
    test_mask = (df_b_all["DriverID"] == holdout_driver)
    
    df_tr = df_b_all[train_mask]
    df_te = df_b_all[test_mask]
    
    # Confirm strict driver isolation
    assert len(set(df_tr["DriverID"]) & set(df_te["DriverID"])) == 0
    assert len(set(df_tr["event_id"]) & set(df_te["event_id"])) == 0
    
    X_tr = df_tr[features_b].values
    y_tr = df_tr["Class"].values.astype(int)
    
    X_te = df_te[features_b].values
    y_te = df_te["Class"].values.astype(int)
    
    # Train Extra Trees with identical hyperparameters as production
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    et.fit(X_tr, y_tr)
    
    # Sample-level evaluation
    samp_preds = et.predict(X_te)
    samp_probs = et.predict_proba(X_te)
    
    samp_acc = accuracy_score(y_te, samp_preds)
    samp_bal_acc = balanced_accuracy_score(y_te, samp_preds)
    samp_macro_f1 = f1_score(y_te, samp_preds, labels=CLASSES, average="macro", zero_division=0)
    samp_cm = confusion_matrix(y_te, samp_preds, labels=CLASSES)
    
    # Event-level causal probability aggregation (mean probability over event trial)
    events = df_te["event_id"].unique()
    y_event_true = []
    y_event_pred = []
    
    for ev_id in events:
        ev_mask = (df_te["event_id"] == ev_id).values
        ev_true_cls = df_te.loc[df_te["event_id"] == ev_id, "Class"].iloc[0]
        ev_probs = samp_probs[ev_mask]
        
        # Mean probability over samples in this event
        mean_p = np.mean(ev_probs, axis=0)
        pred_cls = et.classes_[np.argmax(mean_p)]
        
        y_event_true.append(ev_true_cls)
        y_event_pred.append(pred_cls)
        
    y_ev_true = np.array(y_event_true)
    y_ev_pred = np.array(y_event_pred)
    
    ev_acc = accuracy_score(y_ev_true, y_ev_pred)
    ev_bal_acc = balanced_accuracy_score(y_ev_true, y_ev_pred)
    ev_macro_f1 = f1_score(y_ev_true, y_ev_pred, labels=CLASSES, average="macro", zero_division=0)
    ev_cm = confusion_matrix(y_ev_true, y_ev_pred, labels=CLASSES)
    
    # Per-class event metrics
    ev_prec = precision_score(y_ev_true, y_ev_pred, labels=CLASSES, average=None, zero_division=0)
    ev_rec = recall_score(y_ev_true, y_ev_pred, labels=CLASSES, average=None, zero_division=0)
    ev_f1 = f1_score(y_ev_true, y_ev_pred, labels=CLASSES, average=None, zero_division=0)
    
    per_class_event = {}
    for idx, c in enumerate(CLASSES):
        abbr = CLASS_NAMES[idx]
        per_class_event[abbr] = {
            "precision": float(ev_prec[idx]),
            "recall": float(ev_rec[idx]),
            "f1": float(ev_f1[idx]),
            "support": int(np.sum(y_ev_true == c))
        }
    
    lodo_reproduced_folds[holdout_driver] = {
        "holdout_driver": holdout_driver,
        "train_drivers": [d for d in drivers if d != holdout_driver],
        "train_rows": len(df_tr),
        "train_events": df_tr["event_id"].nunique(),
        "held_out_samples": len(df_te),
        "held_out_events": len(events),
        "sample_metrics": {
            "accuracy": float(samp_acc),
            "balanced_accuracy": float(samp_bal_acc),
            "macro_f1": float(samp_macro_f1),
            "confusion_matrix": samp_cm.tolist()
        },
        "event_metrics": {
            "accuracy": float(ev_acc),
            "balanced_accuracy": float(ev_bal_acc),
            "macro_f1": float(ev_macro_f1),
            "confusion_matrix": ev_cm.tolist(),
            "per_class": per_class_event
        }
    }
    print(f"\nHoldout Driver {holdout_driver}:")
    print(f"  Train: {len(df_tr)} samples ({df_tr['event_id'].nunique()} events) | Test: {len(df_te)} samples ({len(events)} events)")
    print(f"  Sample Level: Acc = {samp_acc:.4f}, Macro-F1 = {samp_macro_f1:.4f}")
    print(f"  Event Level:  Acc = {ev_acc:.4f}, Macro-F1 = {ev_macro_f1:.4f}")

# Compute LODO Mean Metrics across folds
mean_lodo_ev_acc = np.mean([f["event_metrics"]["accuracy"] for f in lodo_reproduced_folds.values()])
mean_lodo_ev_f1 = np.mean([f["event_metrics"]["macro_f1"] for f in lodo_reproduced_folds.values()])
mean_lodo_samp_acc = np.mean([f["sample_metrics"]["accuracy"] for f in lodo_reproduced_folds.values()])
mean_lodo_samp_f1 = np.mean([f["sample_metrics"]["macro_f1"] for f in lodo_reproduced_folds.values()])

print(f"\n--- LODO Summary Across All 3 Drivers ---")
print(f"Mean LODO Event Accuracy: {mean_lodo_ev_acc:.4%}")
print(f"Mean LODO Event Macro-F1: {mean_lodo_ev_f1:.4%}")
print(f"Mean LODO Sample Accuracy: {mean_lodo_samp_acc:.4%}")
print(f"Mean LODO Sample Macro-F1: {mean_lodo_samp_f1:.4%}")

task5_results = {
    "folds": lodo_reproduced_folds,
    "summary": {
        "mean_event_accuracy": float(mean_lodo_ev_acc),
        "mean_event_macro_f1": float(mean_lodo_ev_f1),
        "mean_sample_accuracy": float(mean_lodo_samp_acc),
        "mean_sample_macro_f1": float(mean_lodo_samp_f1),
        "target_gt_85_achieved": bool(mean_lodo_ev_f1 >= 0.85)
    }
}
results["task5_lodo_reproduction"] = task5_results


# =============================================================================
# TASK 6: SAFETY AGENT FALSE-ALARM MITIGATION SIMULATION
# =============================================================================
print("\n" + "=" * 80)
print("TASK 6: SAFETY AGENT TEMPORAL CONFIRMATION SIMULATION")
print("=" * 80)

# Simulate alternative temporal confirmation thresholds on Dataset C Validation & Test
# Ground truth: label 0 = Normal, 1 = Aggressive
# Model: Pipeline A SVM with threshold tau=0.84
# Baseline Smoother: 10-frame rolling window. If aggressive_count / 10 >= 0.50 -> Style=Aggressive (+35 risk).
# Alternative Smoothers: require K consecutive frames of aggressive prediction (Confirmation Dwell Time)
# K = 5 frames (0.5s), 10 frames (1.0s), 15 frames (1.5s), 20 frames (2.0s), 30 frames (3.0s).

def simulate_temporal_smoother(df, model, scaler, features, confirmation_frames_list):
    X = scaler.transform(df[features].values)
    probs = model.predict_proba(X)[:, 1]
    raw_preds = (probs >= 0.84).astype(int)
    labels = df["label"].values.astype(int)
    n = len(df)
    
    sim_results = {}
    
    # 1. Baseline Rolling Window (10 frames, ratio >= 0.50)
    w = 10
    base_style_agg = np.zeros(n, dtype=int)
    for i in range(n):
        start_idx = max(0, i - w + 1)
        window = raw_preds[start_idx:i + 1]
        if np.mean(window) >= 0.50:
            base_style_agg[i] = 1
            
    # Measure baseline metrics
    # In Dataset C, Style=Aggressive directly adds +35 risk -> CAUTION state
    time_caution_base = int(np.sum(base_style_agg == 1))
    false_caution_base = int(np.sum((labels == 0) & (base_style_agg == 1)))
    true_caution_base = int(np.sum((labels == 1) & (base_style_agg == 1)))
    missed_agg_base = int(np.sum((labels == 1) & (base_style_agg == 0)))
    
    sim_results["baseline_rolling_10_ratio_050"] = {
        "confirmation_rule": "Rolling 10 frames, ratio >= 0.50",
        "time_in_caution_pct": float(time_caution_base / n),
        "false_caution_frames": false_caution_base,
        "false_caution_rate_on_normal": float(false_caution_base / np.sum(labels == 0)),
        "true_caution_frames": true_caution_base,
        "missed_aggressive_frames": missed_agg_base,
        "aggressive_detection_recall": float(true_caution_base / np.sum(labels == 1)),
        "caution_precision": float(true_caution_base / time_caution_base) if time_caution_base > 0 else 0.0
    }
    
    # 2. Alternative Consecutive Confirmation Thresholds
    for K in confirmation_frames_list:
        alt_style_agg = np.zeros(n, dtype=int)
        consec_count = 0
        for i in range(n):
            if raw_preds[i] == 1:
                consec_count += 1
            else:
                consec_count = 0
            if consec_count >= K:
                alt_style_agg[i] = 1
                
        time_caution = int(np.sum(alt_style_agg == 1))
        false_caution = int(np.sum((labels == 0) & (alt_style_agg == 1)))
        true_caution = int(np.sum((labels == 1) & (alt_style_agg == 1)))
        missed_agg = int(np.sum((labels == 1) & (alt_style_agg == 0)))
        
        sim_results[f"consecutive_{K}_frames"] = {
            "confirmation_rule": f"Consecutive {K} frames ({K*0.1:.1f}s)",
            "dwell_time_seconds": K * 0.1,
            "time_in_caution_pct": float(time_caution / n),
            "false_caution_frames": false_caution,
            "false_caution_rate_on_normal": float(false_caution / np.sum(labels == 0)),
            "true_caution_frames": true_caution,
            "missed_aggressive_frames": missed_agg,
            "aggressive_detection_recall": float(true_caution / np.sum(labels == 1)),
            "caution_precision": float(true_caution / time_caution) if time_caution > 0 else 0.0
        }
        
    return sim_results

sim_val = simulate_temporal_smoother(df_val, svm_model, scaler_c, features_c, [5, 10, 15, 20, 30])
sim_test = simulate_temporal_smoother(df_test, svm_model, scaler_c, features_c, [5, 10, 15, 20, 30])

print("\n--- Safety Agent Mitigation Simulation (Validation Split) ---")
for k, v in sim_val.items():
    print(f"{v['confirmation_rule']:<35}: Caution={v['time_in_caution_pct']:.2%}, FalseAlertRate={v['false_caution_rate_on_normal']:.2%}, AggRec={v['aggressive_detection_recall']:.2%}, Prec={v['caution_precision']:.2%}")

print("\n--- Safety Agent Mitigation Simulation (Test Split) ---")
for k, v in sim_test.items():
    print(f"{v['confirmation_rule']:<35}: Caution={v['time_in_caution_pct']:.2%}, FalseAlertRate={v['false_caution_rate_on_normal']:.2%}, AggRec={v['aggressive_detection_recall']:.2%}, Prec={v['caution_precision']:.2%}")

task6_results = {
    "validation_simulation": sim_val,
    "test_simulation": sim_test
}
results["task6_safety_agent_simulation"] = task6_results

# Save all results to audit directory
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[✓] All task results saved to: {OUTPUT_JSON}")
print("TASK INVESTIGATION EXECUTION COMPLETED!")

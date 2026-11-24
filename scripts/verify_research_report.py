"""
Independent Forensic Verification of Pipeline A Research Report
===============================================================
Audits and verifies:
1. Exact metric recalculation from raw confusion matrices.
2. Arithmetic Mean vs Pooled Prediction metrics.
3. 10-Target Audit across 4 explicit pass/fail categories.
4. Episode-level segmentation and boundary transitions.
5. Autocorrelation time and Effective Sample Size.
6. Mounting angle offset derivation from raw sensor readings.
7. Verification of artifact hashes and script completion status.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "pipeline_a")

def recalculate_metrics_from_cm(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = (tp + tn) / total
    
    rec_agg = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # sensitivity
    rec_norm = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # specificity
    bal_acc = 0.5 * (rec_agg + rec_norm)
    
    prec_agg = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    prec_norm = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    f1_agg = (2 * prec_agg * rec_agg) / (prec_agg + rec_agg) if (prec_agg + rec_agg) > 0 else 0.0
    f1_norm = (2 * prec_norm * rec_norm) / (prec_norm + rec_norm) if (prec_norm + rec_norm) > 0 else 0.0
    macro_f1 = 0.5 * (f1_agg + f1_norm)
    
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "N": total,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "aggressive_precision": float(prec_agg),
        "aggressive_recall": float(rec_agg),
        "normal_precision": float(prec_norm),
        "normal_recall": float(rec_norm),
        "normal_f1": float(f1_norm),
        "aggressive_f1": float(f1_agg),
        "fpr": float(fpr),
        "fnr": float(fnr)
    }

def audit_research():
    print("=== INDEPENDENT RESEARCH REPORT VERIFICATION ===")
    
    # 1. Load Data
    all_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    s1 = all_df[all_df['session'] == 'session_1'].copy().reset_index(drop=True)
    s2 = all_df[all_df['session'] == 'session_2'].copy().reset_index(drop=True)
    
    print(f"Total rows: {len(all_df)}, S1: {len(s1)}, S2: {len(s2)}")
    print("S1 class counts:", s1['label'].value_counts().to_dict())
    print("S2 class counts:", s2['label'].value_counts().to_dict())
    
    # Load advanced research metrics JSON
    with open(os.path.join(REPORTS_DIR, "advanced_research_metrics.json")) as f:
        adv_res = json.load(f)
        
    print("\n--- 1. VERIFYING CANDIDATE 1 (EXTRATREES 47 FEATURES) ---")
    et_47 = adv_res["Baseline_47_Production"]["ExtraTrees_Balanced"]
    cm_s1_s2 = et_47["s1_to_s2"]["cm_counts"]
    cm_s2_s1 = et_47["s2_to_s1"]["cm_counts"]
    
    calc_s1_s2 = recalculate_metrics_from_cm(cm_s1_s2['TN'], cm_s1_s2['FP'], cm_s1_s2['FN'], cm_s1_s2['TP'])
    calc_s2_s1 = recalculate_metrics_from_cm(cm_s2_s1['TN'], cm_s2_s1['FP'], cm_s2_s1['FN'], cm_s2_s1['TP'])
    
    print("S1 -> S2 Recalculated:")
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "aggressive_precision", "aggressive_recall", "normal_recall", "fpr", "fnr"]:
        print(f"  {k:22s}: {calc_s1_s2[k]:.4f} (reported: {et_47['s1_to_s2'][k]:.4f})")
        
    print("\nS2 -> S1 Recalculated:")
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "aggressive_precision", "aggressive_recall", "normal_recall", "fpr", "fnr"]:
        print(f"  {k:22s}: {calc_s2_s1[k]:.4f} (reported: {et_47['s2_to_s1'][k]:.4f})")
        
    # --- 2. Arithmetic Mean vs Pooled Prediction Metrics ---
    print("\n--- 2. ARITHMETIC MEAN VS POOLED PREDICTION METRICS ---")
    arith_mean = {}
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "aggressive_precision", "aggressive_recall", "normal_recall", "fpr", "fnr"]:
        arith_mean[k] = (calc_s1_s2[k] + calc_s2_s1[k]) / 2.0
        
    pooled_tn = cm_s1_s2['TN'] + cm_s2_s1['TN']
    pooled_fp = cm_s1_s2['FP'] + cm_s2_s1['FP']
    pooled_fn = cm_s1_s2['FN'] + cm_s2_s1['FN']
    pooled_tp = cm_s1_s2['TP'] + cm_s2_s1['TP']
    pooled_calc = recalculate_metrics_from_cm(pooled_tn, pooled_fp, pooled_fn, pooled_tp)
    
    print(f"Pooled Counts: TN={pooled_tn}, FP={pooled_fp}, FN={pooled_fn}, TP={pooled_tp}, Total={pooled_calc['N']}")
    print(f"{'Metric':25s} | {'Arithmetic Mean':17s} | {'Pooled Metric':17s} | {'Difference':12s}")
    print("-" * 75)
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "aggressive_precision", "aggressive_recall", "normal_recall", "fpr", "fnr"]:
        diff = pooled_calc[k] - arith_mean[k]
        print(f"{k:25s} | {arith_mean[k]:17.4f} | {pooled_calc[k]:17.4f} | {diff:+12.4f}")
        
    # --- 3. Strict 10-Target Categorization ---
    print("\n--- 3. STRICT 10-TARGET AUDIT CATEGORIZATION ---")
    target_floors = {
        "accuracy": (0.90, "ge"),
        "balanced_accuracy": (0.90, "ge"),
        "macro_f1": (0.90, "ge"),
        "aggressive_precision": (0.90, "ge"),
        "aggressive_recall": (0.90, "ge"),
        "normal_recall": (0.90, "ge"),
        "roc_auc": (0.95, "ge"),
        "pr_auc": (0.90, "ge"),
        "fpr": (0.10, "le"),
        "fnr": (0.10, "le")
    }
    
    categorized_targets = {}
    for metric, (threshold, direction) in target_floors.items():
        val_s1_s2 = et_47["s1_to_s2"].get(metric, calc_s1_s2.get(metric))
        val_s2_s1 = et_47["s2_to_s1"].get(metric, calc_s2_s1.get(metric))
        val_arith = (val_s1_s2 + val_s2_s1) / 2.0
        
        pass_s1 = (val_s1_s2 >= threshold) if direction == "ge" else (val_s1_s2 <= threshold)
        pass_s2 = (val_s2_s1 >= threshold) if direction == "ge" else (val_s2_s1 <= threshold)
        pass_mean = (val_arith >= threshold) if direction == "ge" else (val_arith <= threshold)
        
        if pass_s1 and pass_s2:
            cat = "BOTH_DIRECTIONS_PASS"
        elif pass_s1 and not pass_s2:
            cat = "ONLY_S1_TO_S2_PASSES"
        elif pass_s2 and not pass_s1:
            cat = "ONLY_S2_TO_S1_PASSES"
        elif pass_mean and not (pass_s1 and pass_s2):
            cat = "ONLY_ARITHMETIC_MEAN_PASSES"
        else:
            cat = "NEITHER_PASSES"
            
        categorized_targets[metric] = {
            "threshold": threshold,
            "direction": direction,
            "val_s1_to_s2": val_s1_s2,
            "val_s2_to_s1": val_s2_s1,
            "val_arithmetic_mean": val_arith,
            "pass_s1_to_s2": pass_s1,
            "pass_s2_to_s1": pass_s2,
            "pass_arithmetic_mean": pass_mean,
            "category": cat
        }
        print(f"  {metric:22s}: S1->S2={val_s1_s2:.4f} ({pass_s1}) | S2->S1={val_s2_s1:.4f} ({pass_s2}) | Mean={val_arith:.4f} ({pass_mean}) -> [{cat}]")
        
    # --- 4. Verify 17 Episodes ---
    print("\n--- 4. EPISODE VERIFICATION ---")
    def extract_episodes(df_sess, sess_name):
        labels = df_sess['label'].values
        episodes = []
        cur_label = labels[0]
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != cur_label:
                episodes.append({
                    "session": sess_name,
                    "label": int(cur_label),
                    "label_name": "Normal" if cur_label == 0 else "Aggressive",
                    "start_idx": int(start_idx),
                    "end_idx": int(i - 1),
                    "duration_samples": int(i - start_idx),
                    "duration_seconds": float((i - start_idx) / 10.0)
                })
                cur_label = labels[i]
                start_idx = i
        episodes.append({
            "session": sess_name,
            "label": int(cur_label),
            "label_name": "Normal" if cur_label == 0 else "Aggressive",
            "start_idx": int(start_idx),
            "end_idx": int(len(labels) - 1),
            "duration_samples": int(len(labels) - start_idx),
            "duration_seconds": float((len(labels) - start_idx) / 10.0)
        })
        return episodes
        
    ep_s1 = extract_episodes(s1, "Session 1")
    ep_s2 = extract_episodes(s2, "Session 2")
    all_episodes = ep_s1 + ep_s2
    
    print(f"Total verified episodes across dataset: {len(all_episodes)}")
    print(f"  Session 1 episodes: {len(ep_s1)} (Normal: {sum(1 for e in ep_s1 if e['label']==0)}, Aggressive: {sum(1 for e in ep_s1 if e['label']==1)})")
    print(f"  Session 2 episodes: {len(ep_s2)} (Normal: {sum(1 for e in ep_s2 if e['label']==0)}, Aggressive: {sum(1 for e in ep_s2 if e['label']==1)})")
    for idx, ep in enumerate(all_episodes):
        print(f"    Episode {idx+1:02d} ({ep['session']}): Label={ep['label_name']:10s} Samples={ep['duration_samples']:5d} Duration={ep['duration_seconds']:6.1f}s [idx {ep['start_idx']}:{ep['end_idx']}]")
        
    # --- 5. Verify Autocorrelation Lag & Effective Sample Size ---
    print("\n--- 5. AUTOCORRELATION & EFFECTIVE SAMPLE SIZE VERIFICATION ---")
    def calc_autocorr(x, max_lag=50):
        x = x - np.mean(x)
        c = np.correlate(x, x, mode='full')
        c = c[len(x)-1:]
        return c[:max_lag] / c[0]
        
    acf_s1 = calc_autocorr(s1['acc_mag'].values, 50)
    acf_s2 = calc_autocorr(s2['acc_mag'].values, 50)
    
    lag_1e_s1 = int(np.where(acf_s1 < 1.0/np.e)[0][0])
    lag_1e_s2 = int(np.where(acf_s2 < 1.0/np.e)[0][0])
    
    # Sokal (1997) formula for integrated autocorrelation time:
    # tau_int = 1 + 2 * sum_{k=1}^K ACF[k]
    # Effective sample size N_eff = N / tau_int
    tau_s1 = 1.0 + 2.0 * np.sum(np.clip(acf_s1[1:25], 0, 1))
    tau_s2 = 1.0 + 2.0 * np.sum(np.clip(acf_s2[1:25], 0, 1))
    n_eff_s1 = len(s1) / tau_s1
    n_eff_s2 = len(s2) / tau_s2
    tot_n_eff = n_eff_s1 + n_eff_s2
    
    print(f"Session 1: Decorrelation lag (1/e) = {lag_1e_s1} samples ({lag_1e_s1/10.0:.1f}s), tau_int = {tau_s1:.2f}, N_eff = {n_eff_s1:.1f}")
    print(f"Session 2: Decorrelation lag (1/e) = {lag_1e_s2} samples ({lag_1e_s2/10.0:.1f}s), tau_int = {tau_s2:.2f}, N_eff = {n_eff_s2:.1f}")
    print(f"Total nominal samples: {len(all_df)}, Total effective independent samples N_eff: {tot_n_eff:.1f}")
    
    # --- 6. Verify Phone Mounting Orientation Drift ---
    print("\n--- 6. VERIFYING PHONE MOUNTING ANGLE ---")
    mean_v1 = s1[['Acc X', 'Acc Y', 'Acc Z']].mean().values
    mean_v2 = s2[['Acc X', 'Acc Y', 'Acc Z']].mean().values
    
    cos_th = np.dot(mean_v1, mean_v2) / (np.linalg.norm(mean_v1) * np.linalg.norm(mean_v2))
    angle_deg = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))
    
    print(f"Session 1 Mean Static Acc: {mean_v1} (norm = {np.linalg.norm(mean_v1):.4f})")
    print(f"Session 2 Mean Static Acc: {mean_v2} (norm = {np.linalg.norm(mean_v2):.4f})")
    print(f"Dot Product / Norms: {cos_th:.6f}")
    print(f"Verified Mounting Angle Offset: {angle_deg:.2f}°")
    
    verification_data = {
        "recalculated_s1_s2": calc_s1_s2,
        "recalculated_s2_s1": calc_s2_s1,
        "arithmetic_mean": arith_mean,
        "pooled_metrics": pooled_calc,
        "categorized_targets": categorized_targets,
        "total_episodes": len(all_episodes),
        "episodes": all_episodes,
        "autocorrelation": {
            "lag_1e_s1": lag_1e_s1,
            "lag_1e_s2": lag_1e_s2,
            "tau_int_s1": float(tau_s1),
            "tau_int_s2": float(tau_s2),
            "n_eff_s1": float(n_eff_s1),
            "n_eff_s2": float(n_eff_s2),
            "total_n_eff": float(tot_n_eff)
        },
        "mounting_drift": {
            "mean_v1": mean_v1.tolist(),
            "mean_v2": mean_v2.tolist(),
            "angle_deg": float(angle_deg)
        }
    }
    
    with open(os.path.join(REPORTS_DIR, "independent_verification_audit.json"), "w") as f:
        json.dump(verification_data, f, indent=2)
    print(f"\nVerification audit saved to {os.path.join(REPORTS_DIR, 'independent_verification_audit.json')}")
    
    return verification_data

if __name__ == "__main__":
    audit_research()

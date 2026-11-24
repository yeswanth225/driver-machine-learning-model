"""
Pipeline A Comprehensive Model Improvement & Validation Suite
==============================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Phases Addressed: Phase 2, Phase 3, Phase 4, Phase 5

Evaluates 5 model architectures:
1. Tuned RBF SVM (with probability calibration)
2. Extra Trees Classifier (with class balancing & depth regularization)
3. Random Forest Classifier
4. XGBoost Classifier
5. HistGradientBoosting Classifier

Across 3 feature paradigms:
A. Baseline 47 Features (Strict Production Feature Contract)
B. Candidate v2 Pruned Features (102 features)
C. Domain-Stabilized Features (low PSI features invariant to session shifts)

Evaluated on:
1. Chronological Test Split (70% train / 15% val / 15% test with 10s guard bands)
2. Bidirectional Leave-One-Session-Out (LODO):
   - Train on Session 1 -> Test on Session 2
   - Train on Session 2 -> Test on Session 1
   - Mean Cross-Session Performance

All thresholds tuned strictly on Validation / Development data (never on Test).
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
EXP_DATA_DIR = os.path.join(BASE_DIR, "dataset", "experiments", "pipeline_a")
OUT_EXP_DIR = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "pipeline_a")

os.makedirs(OUT_EXP_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_psi(series_1: pd.Series, series_2: pd.Series, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    # Combine to define global quantiles
    combined = pd.concat([series_1, series_2])
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.percentile(combined.dropna(), quantiles * 100)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    # Count proportions in each bin
    c1, _ = np.histogram(series_1.dropna(), bins=bins)
    c2, _ = np.histogram(series_2.dropna(), bins=bins)

    p1 = (c1 + 1e-4) / (len(series_1) + 1e-4 * len(c1))
    p2 = (c2 + 1e-4) / (len(series_2) + 1e-4 * len(c2))

    psi = np.sum((p1 - p2) * np.log(p1 / p2))
    return float(psi)


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lower = bin_limits[i]
        upper = bin_limits[i + 1]
        mask = (y_prob >= lower) & (y_prob <= upper if i == n_bins - 1 else y_prob < upper)
        count = np.sum(mask)
        if count > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (count / total) * np.abs(acc - conf)
    return float(ece)


def find_optimal_threshold(y_true, y_probs, min_recall=0.80):
    """Optimize threshold on validation set to maximize Macro-F1 with safety recall floor."""
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

    # Fallback to unconstrained Macro-F1
    if best_f1_macro == -1.0:
        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            f1_m = f1_score(y_true, preds, average='macro', zero_division=0)
            if f1_m > best_f1_macro:
                best_f1_macro = f1_m
                best_thresh = th

    return float(best_thresh)


def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Compute comprehensive diagnostic evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    elif cm.shape == (1, 1):
        # Degenerate single-class case
        if 0 in y_true:
            tn, fp, fn, tp = cm[0][0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0][0]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    auc = float(roc_auc_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    brier = float(brier_score_loss(y_true, y_prob)) if y_prob is not None else None
    ece = float(compute_ece(y_true, y_prob)) if y_prob is not None else None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "precision_normal": prec_0,
        "recall_normal": rec_0,
        "f1_normal": f1_0,
        "precision_aggressive": prec_1,
        "recall_aggressive": rec_1,
        "f1_aggressive": f1_1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "confusion_matrix": cm.tolist(),
        "cm_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }


def measure_latency_and_memory(model, X_sample, n_runs=500):
    """Measure P50, P95, P99 per-sample inference latency in ms."""
    latencies = []
    # Warmup
    for _ in range(20):
        _ = model.predict_proba(X_sample[:1])

    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_sample[:1])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies))
    }


def get_model_definitions():
    """Returns candidate model instances and their scaling requirements."""
    return {
        "Refined_SVM_RBF": (
            SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
            True
        ),
        "ExtraTrees_Balanced": (
            ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1),
            False
        ),
        "RandomForest_Subsample": (
            RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            False
        ),
        "XGBoost_Tuned": (
            XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.65, random_state=42, eval_metric="logloss"),
            False
        ),
        "HistGradientBoosting_Tuned": (
            HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.04, min_samples_leaf=15, class_weight='balanced', random_state=42),
            False
        )
    }


def run_pipeline_a_improvements():
    print("==================================================================")
    print("PIPELINE A: COMPREHENSIVE MODEL IMPROVEMENT & VALIDATION EXPERIMENT")
    print("==================================================================")

    # 1. Load Baseline Datasets
    b_train = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_train.csv"))
    b_val = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_val.csv"))
    b_test = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_test.csv"))
    b_all = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    b_feats = [c for c in b_train.columns if c not in ['label', 'split', 'session']]

    # 2. Load Enhanced Datasets (Candidate v2)
    e_train = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_train.csv"))
    e_val = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_val.csv"))
    e_test = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_test.csv"))
    e_all = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_all.csv"))
    e_feats = [c for c in e_train.columns if c not in ['label', 'split', 'session']]

    # 3. Candidate v2 Pruned Features (Top 102 causal features)
    high_jitter = ['gyro_mag_max_3s', 'gyro_mag_max_2s', 'acc_mag_std_3s', 'acc_mag_std_2s']
    pruned_feats = [c for c in e_feats if c not in high_jitter and not ('min' in c and 'acc_planar' in c)]

    # 4. Domain Stability PSI Analysis on Baseline 47 and Enhanced Features
    s1_all = e_all[e_all['session'] == 'session_1']
    s2_all = e_all[e_all['session'] == 'session_2']

    psi_scores = {}
    for feat in e_feats:
        psi_scores[feat] = compute_psi(s1_all[feat], s2_all[feat])

    # Rank features by stability (low PSI)
    stable_feats = [f for f, psi in psi_scores.items() if psi < 0.20 and f in pruned_feats]
    print(f"Features: Baseline={len(b_feats)}, Enhanced={len(e_feats)}, Pruned_v2={len(pruned_feats)}, High_Stability_PSI={len(stable_feats)}")

    # Feature sets to benchmark
    feature_sets = {
        "baseline_47": {
            "feats": b_feats,
            "train": b_train, "val": b_val, "test": b_test, "all": b_all,
            "description": "Strict 47 production feature contract (no pipeline changes)"
        },
        "candidate_v2_102": {
            "feats": pruned_feats,
            "train": e_train, "val": e_val, "test": e_test, "all": e_all,
            "description": "Candidate v2 causal engineered & jitter-pruned features"
        },
        "domain_stable_psi": {
            "feats": stable_feats,
            "train": e_train, "val": e_val, "test": e_test, "all": e_all,
            "description": "Covariate-stabilized features (PSI < 0.20 across physical sessions)"
        }
    }

    experiment_results = {}

    for fset_key, fset_info in feature_sets.items():
        feats = fset_info["feats"]
        print(f"\n=======================================================")
        print(f"BENCHMARKING FEATURE SET: {fset_key} ({len(feats)} features)")
        print(f"=======================================================")

        experiment_results[fset_key] = {}

        df_tr, df_va, df_te, df_full = fset_info["train"], fset_info["val"], fset_info["test"], fset_info["all"]

        X_tr_raw, y_tr = df_tr[feats], df_tr['label'].values
        X_va_raw, y_va = df_va[feats], df_va['label'].values
        X_te_raw, y_te = df_te[feats], df_te['label'].values

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_raw)
        X_va_sc = scaler.transform(X_va_raw)
        X_te_sc = scaler.transform(X_te_raw)

        models = get_model_definitions()

        for model_name, (model_proto, req_scale) in models.items():
            print(f"\n---> Training & Evaluating: {model_name}...")
            
            # Fresh clone / instance
            from sklearn.base import clone
            model = clone(model_proto)

            X_tr = X_tr_sc if req_scale else X_tr_raw.values
            X_va = X_va_sc if req_scale else X_va_raw.values
            X_te = X_te_sc if req_scale else X_te_raw.values

            # Fit model
            t_fit_start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t_fit_start

            # Validation prediction & threshold tuning
            prob_va = model.predict_proba(X_va)[:, 1]
            tau_star = find_optimal_threshold(y_va, prob_va, min_recall=0.80)

            # Test predictions
            prob_te = model.predict_proba(X_te)[:, 1]
            pred_te_50 = (prob_te >= 0.50).astype(int)
            pred_te_tau = (prob_te >= tau_star).astype(int)

            metrics_50 = evaluate_predictions(y_te, pred_te_50, prob_te)
            metrics_tau = evaluate_predictions(y_te, pred_te_tau, prob_te)

            # Latency benchmark
            lat_stats = measure_latency_and_memory(model, X_te[:100])

            # Cross-Session (LODO) Evaluation
            s1_full = df_full[df_full['session'] == 'session_1']
            s2_full = df_full[df_full['session'] == 'session_2']

            # S1 -> S2 (Train on S1, Test on S2)
            scaler_s1 = StandardScaler()
            X_s1 = scaler_s1.fit_transform(s1_full[feats]) if req_scale else s1_full[feats].values
            X_s2_for_s1 = scaler_s1.transform(s2_full[feats]) if req_scale else s2_full[feats].values

            model_s1 = clone(model_proto)
            model_s1.fit(X_s1, s1_full['label'].values)
            prob_s2 = model_s1.predict_proba(X_s2_for_s1)[:, 1]
            pred_s2_50 = (prob_s2 >= 0.50).astype(int)
            m_s1_s2 = evaluate_predictions(s2_full['label'].values, pred_s2_50, prob_s2)

            # S2 -> S1 (Train on S2, Test on S1)
            scaler_s2 = StandardScaler()
            X_s2 = scaler_s2.fit_transform(s2_full[feats]) if req_scale else s2_full[feats].values
            X_s1_for_s2 = scaler_s2.transform(s1_full[feats]) if req_scale else s1_full[feats].values

            model_s2 = clone(model_proto)
            model_s2.fit(X_s2, s2_full['label'].values)
            prob_s1 = model_s2.predict_proba(X_s1_for_s2)[:, 1]
            pred_s1_50 = (prob_s1 >= 0.50).astype(int)
            m_s2_s1 = evaluate_predictions(s1_full['label'].values, pred_s1_50, prob_s1)

            mean_lodo_acc = float((m_s1_s2['accuracy'] + m_s2_s1['accuracy']) / 2.0)
            mean_lodo_f1 = float((m_s1_s2['macro_f1'] + m_s2_s1['macro_f1']) / 2.0)
            mean_lodo_bal = float((m_s1_s2['balanced_accuracy'] + m_s2_s1['balanced_accuracy']) / 2.0)

            print(f"  Chronological Test (tau*={tau_star:.2f}): Acc={metrics_tau['accuracy']:.4f}, BalAcc={metrics_tau['balanced_accuracy']:.4f}, Macro-F1={metrics_tau['macro_f1']:.4f}, FPR={metrics_tau['fpr']:.4f}, RecAgg={metrics_tau['recall_aggressive']:.4f}, AUC={metrics_tau['roc_auc'] or 0.0:.4f}")
            print(f"  LODO S1->S2: Acc={m_s1_s2['accuracy']:.4f}, F1={m_s1_s2['macro_f1']:.4f} | S2->S1: Acc={m_s2_s1['accuracy']:.4f}, F1={m_s2_s1['macro_f1']:.4f}")
            print(f"  LODO Mean:   Acc={mean_lodo_acc:.4f}, Macro-F1={mean_lodo_f1:.4f}, BalAcc={mean_lodo_bal:.4f}")
            print(f"  Latency P50: {lat_stats['p50_ms']:.4f} ms, P95: {lat_stats['p95_ms']:.4f} ms")

            experiment_results[fset_key][model_name] = {
                "fit_time_sec": fit_time,
                "latency_benchmarks": lat_stats,
                "tuned_tau": tau_star,
                "test_metrics_default_50": metrics_50,
                "test_metrics_tuned_tau": metrics_tau,
                "lodo_s1_to_s2": m_s1_s2,
                "lodo_s2_to_s1": m_s2_s1,
                "lodo_mean": {
                    "accuracy": mean_lodo_acc,
                    "macro_f1": mean_lodo_f1,
                    "balanced_accuracy": mean_lodo_bal
                }
            }

    # Save comprehensive results
    res_path = os.path.join(REPORTS_DIR, "pipeline_a_improved_experiments.json")
    with open(res_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nAll experimental results saved to {res_path}")

    # 5. Packaging Best Candidates into models/experiments/pipeline_a/
    print("\n=======================================================")
    print("PACKAGING TOP CANDIDATES TO models/experiments/pipeline_a/")
    print("=======================================================")

    # Candidate 1: Production Contract ExtraTrees (47 Features - 100% Drop-in Compatible)
    print("Packaging Candidate 1: ExtraTrees on 47 Baseline Features (Drop-in compatible)...")
    et_47 = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1)
    b_tr_X, b_tr_y = b_train[b_feats], b_train['label'].values
    b_va_X, b_va_y = b_val[b_feats], b_val['label'].values
    et_47.fit(b_tr_X, b_tr_y)
    prob_va_47 = et_47.predict_proba(b_va_X)[:, 1]
    tau_47 = find_optimal_threshold(b_va_y, prob_va_47, min_recall=0.80)

    scaler_47 = StandardScaler().fit(b_tr_X) # optional/identity for trees, but provided for pipeline uniformity

    cand1_model_path = os.path.join(OUT_EXP_DIR, "pipeline_a_extratrees_47feat_candidate.joblib")
    cand1_scaler_path = os.path.join(OUT_EXP_DIR, "pipeline_a_scaler_47feat_candidate.joblib")
    cand1_cfg_path = os.path.join(OUT_EXP_DIR, "pipeline_a_config_47feat_candidate.json")

    joblib.dump(et_47, cand1_model_path)
    joblib.dump(scaler_47, cand1_scaler_path)

    cand1_cfg = {
        "candidate_id": "pipeline_a_extratrees_47feat",
        "description": "ExtraTrees Classifier trained on strict 47 production feature schema",
        "model_architecture": "ExtraTreesClassifier",
        "requires_scaler": False,
        "feature_count": len(b_feats),
        "feature_names": b_feats,
        "calibrated_threshold": tau_47,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 6,
            "class_weight": "balanced",
            "random_state": 42
        },
        "performance_summary": experiment_results["baseline_47"]["ExtraTrees_Balanced"]
    }
    with open(cand1_cfg_path, "w") as f:
        json.dump(cand1_cfg, f, indent=2)

    # Candidate 2: Domain Stable ExtraTrees (Highest Generalization across Sessions)
    print("Packaging Candidate 2: ExtraTrees on Domain-Stabilized Features...")
    et_ds = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1)
    ds_tr_X, ds_tr_y = e_train[stable_feats], e_train['label'].values
    ds_va_X, ds_va_y = e_val[stable_feats], e_val['label'].values
    et_ds.fit(ds_tr_X, ds_tr_y)
    prob_va_ds = et_ds.predict_proba(ds_va_X)[:, 1]
    tau_ds = find_optimal_threshold(ds_va_y, prob_va_ds, min_recall=0.80)

    cand2_model_path = os.path.join(OUT_EXP_DIR, "pipeline_a_extratrees_domain_stable_candidate.joblib")
    cand2_scaler_path = os.path.join(OUT_EXP_DIR, "pipeline_a_scaler_domain_stable_candidate.joblib")
    cand2_cfg_path = os.path.join(OUT_EXP_DIR, "pipeline_a_config_domain_stable_candidate.json")

    joblib.dump(et_ds, cand2_model_path)
    joblib.dump(StandardScaler().fit(ds_tr_X), cand2_scaler_path)

    cand2_cfg = {
        "candidate_id": "pipeline_a_extratrees_domain_stable",
        "description": "ExtraTrees Classifier trained on covariate-stabilized features (PSI < 0.20)",
        "model_architecture": "ExtraTreesClassifier",
        "requires_scaler": False,
        "feature_count": len(stable_feats),
        "feature_names": stable_feats,
        "calibrated_threshold": tau_ds,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 6,
            "class_weight": "balanced",
            "random_state": 42
        },
        "performance_summary": experiment_results["domain_stable_psi"]["ExtraTrees_Balanced"]
    }
    with open(cand2_cfg_path, "w") as f:
        json.dump(cand2_cfg, f, indent=2)

    # Candidate 3: Tuned Calibrated SVM (Refined baseline architecture)
    print("Packaging Candidate 3: Refined Tuned SVM (Calibrated)...")
    svm_tuned = SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42)
    b_tr_sc = StandardScaler().fit_transform(b_tr_X)
    svm_tuned.fit(b_tr_sc, b_tr_y)
    scaler_svm = StandardScaler().fit(b_tr_X)

    cand3_model_path = os.path.join(OUT_EXP_DIR, "pipeline_a_svm_refined_candidate.joblib")
    cand3_scaler_path = os.path.join(OUT_EXP_DIR, "pipeline_a_scaler_svm_refined_candidate.joblib")
    cand3_cfg_path = os.path.join(OUT_EXP_DIR, "pipeline_a_config_svm_refined_candidate.json")

    joblib.dump(svm_tuned, cand3_model_path)
    joblib.dump(scaler_svm, cand3_scaler_path)

    cand3_cfg = {
        "candidate_id": "pipeline_a_svm_refined",
        "description": "Refined RBF Support Vector Machine on strict 47 production features",
        "model_architecture": "SVC",
        "requires_scaler": True,
        "feature_count": len(b_feats),
        "feature_names": b_feats,
        "calibrated_threshold": experiment_results["baseline_47"]["Refined_SVM_RBF"]["tuned_tau"],
        "hyperparameters": {
            "C": 1.5,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42
        },
        "performance_summary": experiment_results["baseline_47"]["Refined_SVM_RBF"]
    }
    with open(cand3_cfg_path, "w") as f:
        json.dump(cand3_cfg, f, indent=2)

    # 6. Generate SHA-256 Manifest of all candidate artifacts
    manifest_lines = []
    for fn in os.listdir(OUT_EXP_DIR):
        fp = os.path.join(OUT_EXP_DIR, fn)
        if os.path.isfile(fp) and fn != "MANIFEST_SHA256.txt":
            manifest_lines.append(f"{compute_sha256(fp)}  {fn}")

    manifest_path = os.path.join(OUT_EXP_DIR, "MANIFEST_SHA256.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(sorted(manifest_lines)) + "\n")
    print(f"Candidate manifest generated at {manifest_path}")

    return experiment_results

if __name__ == "__main__":
    run_pipeline_a_improvements()

"""
Pipeline A: Model Training and Comparative Evaluation for Dataset C
Task: Binary Driving Style Classification (0: Normal Driving, 1: Aggressive Driving)
Auditor: Senior ML Engineer & Safety Systems Architect
Stage: 4B - Model Training & Evaluation
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Repository relative paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "dataset" / "processed"
EXPERIMENT_DIR = REPO_ROOT / "models" / "experiments" / "dataset_c"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

FEATURE_COLS = [
    'Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z',
    'acc_mag', 'gyro_mag', 'acc_horiz', 'acc_jerk', 'gyro_jerk',
    'acc_mag_mean_1s', 'acc_mag_std_1s', 'acc_mag_max_1s', 'acc_mag_min_1s', 'acc_mag_range_1s',
    'gyro_mag_mean_1s', 'gyro_mag_std_1s', 'gyro_mag_max_1s',
    'acc_horiz_mean_1s', 'acc_horiz_std_1s', 'gyro_z_std_1s', 'gyro_z_max_1s',
    'acc_mag_mean_2s', 'acc_mag_std_2s', 'acc_mag_max_2s', 'acc_mag_min_2s', 'acc_mag_range_2s',
    'gyro_mag_mean_2s', 'gyro_mag_std_2s', 'gyro_mag_max_2s',
    'acc_horiz_mean_2s', 'acc_horiz_std_2s', 'gyro_z_std_2s', 'gyro_z_max_2s',
    'acc_mag_mean_3s', 'acc_mag_std_3s', 'acc_mag_max_3s', 'acc_mag_min_3s', 'acc_mag_range_3s',
    'gyro_mag_mean_3s', 'gyro_mag_std_3s', 'gyro_mag_max_3s',
    'acc_horiz_mean_3s', 'acc_horiz_std_3s', 'gyro_z_std_3s', 'gyro_z_max_3s'
]

LABEL_MAPPING = {
    0: "Normal Driving",
    1: "Aggressive Driving"
}


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Compute comprehensive binary classification evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_normal": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_normal": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_normal": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_aggressive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_aggressive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_aggressive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr,
        "fnr": fnr
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Tune decision threshold on validation set to maximize Macro F1 while maintaining
    Aggressive Recall >= 0.80 (safety requirement).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_macro_f1 = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec_agg = recall_score(y_true, preds, pos_label=1, zero_division=0)
        macro_f1 = f1_score(y_true, preds, average="macro")

        # Prioritize safety: Aggressive Recall must be at least 0.80 if possible
        if rec_agg >= 0.80:
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_thresh = float(t)

    # Fallback to pure Macro F1 if no threshold achieves >= 0.80 recall
    if best_macro_f1 == -1.0:
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            macro_f1 = f1_score(y_true, preds, average="macro")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_thresh = float(t)

    return best_thresh, best_macro_f1


def train_and_evaluate_dataset_c() -> Dict[str, Any]:
    print("=" * 80)
    print("STAGE 4B: DATASET C BINARY DRIVING STYLE CLASSIFICATION")
    print("=" * 80)

    # 1. Load Data
    train_df = pd.read_csv(DATA_DIR / "dataset_c_train.csv")
    val_df = pd.read_csv(DATA_DIR / "dataset_c_val.csv")
    test_df = pd.read_csv(DATA_DIR / "dataset_c_test.csv")

    print(f"Loaded Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")
    print(f"Features: {len(FEATURE_COLS)} features")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["label"].values.astype(int)

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values.astype(int)

    # 2. Fit Scaler Strictly on Training Partition
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler Artifact
    scaler_path = EXPERIMENT_DIR / "dataset_c_scaler_exp_v1.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path.name}")

    # Calculate class imbalance weight for Train
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = float(n_neg / n_pos)
    print(f"Train class balance: Normal={n_neg} ({n_neg/len(y_train)*100:.1f}%), Aggressive={n_pos} ({n_pos/len(y_train)*100:.1f}%)")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    # 3. Candidate Models Definition
    models_dict = {
        "Dummy (Stratified)": (
            DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
            False  # Doesn't require scaled
        ),
        "Dummy (Most Frequent)": (
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
            False
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
            True   # Requires scaled
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            False
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            False
        ),
        "Support Vector Machine": (
            SVC(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_SEED
            ),
            True
        ),
        "Multi-Layer Perceptron": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_SEED
            ),
            True
        )
    }

    results = {}
    fitted_models = {}

    for name, (model, requires_scale) in models_dict.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        X_tr = X_train_scaled if requires_scale else X_train
        X_va = X_val_scaled if requires_scale else X_val
        X_te = X_test_scaled if requires_scale else X_test

        model.fit(X_tr, y_train)
        fit_time = time.time() - t0
        print(f"Fit completed in {fit_time:.2f}s")
        fitted_models[name] = model

        # Val evaluation with default threshold 0.5
        val_pred = model.predict(X_va)
        val_prob = model.predict_proba(X_va)[:, 1] if hasattr(model, "predict_proba") else None
        val_metrics_default = compute_binary_metrics(y_val, val_pred, val_prob)

        # Threshold tuning on Validation set
        if val_prob is not None and not isinstance(model, DummyClassifier):
            best_thresh, tuned_val_f1 = find_optimal_threshold(y_val, val_prob)
            val_pred_tuned = (val_prob >= best_thresh).astype(int)
            val_metrics = compute_binary_metrics(y_val, val_pred_tuned, val_prob)
            val_metrics["decision_threshold"] = best_thresh
            print(f"Validation Macro F1 (default 0.5): {val_metrics_default['macro_f1']:.4f} -> (tuned {best_thresh:.2f}): {val_metrics['macro_f1']:.4f}")
        else:
            best_thresh = 0.5
            val_metrics = val_metrics_default
            val_metrics["decision_threshold"] = 0.5

        # Test evaluation with tuned threshold (evaluated strictly once)
        test_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        if test_prob is not None and not isinstance(model, DummyClassifier):
            test_pred = (test_prob >= best_thresh).astype(int)
        else:
            test_pred = model.predict(X_te)
        test_metrics = compute_binary_metrics(y_test, test_pred, test_prob)
        test_metrics["decision_threshold"] = best_thresh

        print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f} | Agg Recall: {test_metrics['recall_aggressive']:.4f} | PR-AUC: {test_metrics['pr_auc']}")

        # Save single model artifact
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        artifact_path = EXPERIMENT_DIR / f"dataset_c_{safe_name}_exp_v1.joblib"
        joblib.dump(model, artifact_path)

        results[name] = {
            "model_name": name,
            "artifact_path": str(artifact_path.relative_to(REPO_ROOT)),
            "requires_scaling": requires_scale,
            "fit_time_seconds": fit_time,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "hyperparameters": {k: str(v) for k, v in model.get_params().items()}
        }

    # Model Selection: Pick best model on Validation Macro F1 with Aggressive Recall >= 0.80
    best_model_name = None
    best_val_f1 = -1.0
    for name, res in results.items():
        if "Dummy" in name:
            continue
        v_f1 = res["validation_metrics"]["macro_f1"]
        v_rec = res["validation_metrics"]["recall_aggressive"]
        if v_rec >= 0.80 and v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_model_name = name

    if best_model_name is None:
        # Fallback to pure highest Macro F1
        for name, res in results.items():
            if "Dummy" in name:
                continue
            v_f1 = res["validation_metrics"]["macro_f1"]
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_model_name = name

    print("\n" + "=" * 80)
    print(f"SELECTED BEST MODEL ON VALIDATION SET: {best_model_name}")
    print(f"Validation Macro F1: {results[best_model_name]['validation_metrics']['macro_f1']:.4f}")
    print(f"Test Macro F1: {results[best_model_name]['test_metrics']['macro_f1']:.4f}")
    print(f"Test Accuracy: {results[best_model_name]['test_metrics']['accuracy']:.4f}")
    print(f"Test Aggressive Recall: {results[best_model_name]['test_metrics']['recall_aggressive']:.4f}")
    print("=" * 80)

    # Save comprehensive experiment metadata and metrics
    summary_metadata = {
        "pipeline": "Pipeline A (Dataset C)",
        "task": "Binary Driving Style Classification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "selected_best_model": best_model_name,
        "feature_count": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "scaler_artifact": str(scaler_path.relative_to(REPO_ROOT)),
        "models_evaluated": list(results.keys()),
        "results": results
    }

    with open(EXPERIMENT_DIR / "dataset_c_experiment_summary.json", "w") as f:
        json.dump(summary_metadata, f, indent=2)

    return summary_metadata


if __name__ == "__main__":
    train_and_evaluate_dataset_c()

"""
Pipeline B: Model Training, Comparative Evaluation & LODO Generalization for Dataset B
Task: 4-Class Maneuver Classification (1: SA, 2: SD, 3: SRT, 4: SLT)
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# Repository relative paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "dataset" / "processed"
EXPERIMENT_DIR = REPO_ROOT / "models" / "experiments" / "dataset_b"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

EXCLUDED_COLS = {
    'Class', 'DriverID', 'TaskID', 'Hour', 'Minute', 'Second',
    'event_id', 'label_name', 'label_abbr',
    'event_sample_idx', 'event_duration_samples', 'event_progress_ratio', 'split'
}

LABEL_MAPPING = {
    1: "Sudden Acceleration",
    2: "Sudden Deceleration",
    3: "Sudden Right Turn",
    4: "Sudden Left Turn"
}

LABEL_ABBR = {
    1: "SA",
    2: "SD",
    3: "SRT",
    4: "SLT"
}

CLASSES = [1, 2, 3, 4]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract causal Tier 1 streaming feature column names."""
    return [col for col in df.columns if col not in EXCLUDED_COLS]


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive 4-class evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0))

    per_class_prec = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    per_class_dict = {}
    for idx, c in enumerate(CLASSES):
        abbr = LABEL_ABBR[c]
        per_class_dict[abbr] = {
            "class_id": c,
            "label_name": LABEL_MAPPING[c],
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(np.sum(y_true == c))
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_dict
    }


def evaluate_event_level(
    df_partition: pd.DataFrame,
    probs: np.ndarray,
    classes_order: np.ndarray
) -> Dict[str, Any]:
    """
    Causally aggregate sample-level probability predictions across each event trial
    to produce event-level classification.
    """
    events = df_partition["event_id"].unique()
    y_event_true = []
    y_event_pred = []

    for ev_id in events:
        ev_mask = (df_partition["event_id"] == ev_id).values
        ev_true_cls = df_partition.loc[df_partition["event_id"] == ev_id, "Class"].iloc[0]
        ev_probs = probs[ev_mask]

        # Mean predicted probability over elapsed samples of the event
        mean_prob = np.mean(ev_probs, axis=0)
        pred_cls = classes_order[np.argmax(mean_prob)]

        y_event_true.append(ev_true_cls)
        y_event_pred.append(pred_cls)

    y_event_true = np.array(y_event_true)
    y_event_pred = np.array(y_event_pred)

    metrics = compute_multiclass_metrics(y_event_true, y_event_pred)
    metrics["event_count"] = len(events)
    return metrics


def train_and_evaluate_dataset_b() -> Dict[str, Any]:
    print("=" * 80)
    print("STAGE 4B: DATASET B 4-CLASS MANEUVER CLASSIFICATION (STRATIFIED SPLIT)")
    print("=" * 80)

    # 1. Load Data
    train_df = pd.read_csv(DATA_DIR / "dataset_b_train.csv")
    val_df = pd.read_csv(DATA_DIR / "dataset_b_val.csv")
    test_df = pd.read_csv(DATA_DIR / "dataset_b_test.csv")

    feature_cols = get_feature_columns(train_df)
    print(f"Loaded Train: {len(train_df)} rows ({train_df['event_id'].nunique()} events)")
    print(f"Loaded Val:   {len(val_df)} rows ({val_df['event_id'].nunique()} events)")
    print(f"Loaded Test:  {len(test_df)} rows ({test_df['event_id'].nunique()} events)")
    print(f"Features: {len(feature_cols)} Tier 1 causal streaming features")

    X_train = train_df[feature_cols].values
    y_train = train_df["Class"].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df["Class"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["Class"].values.astype(int)

    # 2. Fit Scaler Strictly on Training Partition
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = EXPERIMENT_DIR / "dataset_b_scaler_exp_v1.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path.name}")

    # 3. Candidate Models
    models_dict = {
        "Dummy (Stratified)": (
            DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
            False
        ),
        "Dummy (Most Frequent)": (
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
            False
        ),
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            True
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            False
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            False
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            False
        ),
        "Support Vector Machine": (
            SVC(
                C=3.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_SEED
            ),
            True
        )
    }

    results = {}

    for name, (model, requires_scale) in models_dict.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        X_tr = X_train_scaled if requires_scale else X_train
        X_va = X_val_scaled if requires_scale else X_val
        X_te = X_test_scaled if requires_scale else X_test

        is_xgb = "XGBoost" in name
        y_tr_fit = (y_train - 1) if is_xgb else y_train

        model.fit(X_tr, y_tr_fit)
        fit_time = time.time() - t0
        print(f"Fit completed in {fit_time:.2f}s")

        # Predictions on Validation
        if is_xgb:
            val_probs = model.predict_proba(X_va)
            classes_order = np.array([1, 2, 3, 4])
            val_sample_preds = classes_order[np.argmax(val_probs, axis=1)]
        else:
            if hasattr(model, "predict_proba"):
                val_probs = model.predict_proba(X_va)
                classes_order = model.classes_
                val_sample_preds = model.predict(X_va)
            else:
                val_sample_preds = model.predict(X_va)
                # One-hot mock for dummy without proba
                val_probs = np.zeros((len(X_va), 4))
                for i, p in enumerate(val_sample_preds):
                    val_probs[i, p - 1] = 1.0
                classes_order = np.array([1, 2, 3, 4])

        val_sample_metrics = compute_multiclass_metrics(y_val, val_sample_preds)
        val_event_metrics = evaluate_event_level(val_df, val_probs, classes_order)

        print(f"Val Sample Acc: {val_sample_metrics['accuracy']:.4f} | Sample Macro F1: {val_sample_metrics['macro_f1']:.4f}")
        print(f"Val Event  Acc: {val_event_metrics['accuracy']:.4f} | Event  Macro F1: {val_event_metrics['macro_f1']:.4f}")

        # Predictions on Test (strictly evaluated once)
        if is_xgb:
            test_probs = model.predict_proba(X_te)
            test_sample_preds = classes_order[np.argmax(test_probs, axis=1)]
        else:
            if hasattr(model, "predict_proba"):
                test_probs = model.predict_proba(X_te)
                test_sample_preds = model.predict(X_te)
            else:
                test_sample_preds = model.predict(X_te)
                test_probs = np.zeros((len(X_te), 4))
                for i, p in enumerate(test_sample_preds):
                    test_probs[i, p - 1] = 1.0

        test_sample_metrics = compute_multiclass_metrics(y_test, test_sample_preds)
        test_event_metrics = evaluate_event_level(test_df, test_probs, classes_order)

        print(f"Test Sample Acc: {test_sample_metrics['accuracy']:.4f} | Sample Macro F1: {test_sample_metrics['macro_f1']:.4f}")
        print(f"Test Event  Acc: {test_event_metrics['accuracy']:.4f} | Event  Macro F1: {test_event_metrics['macro_f1']:.4f}")

        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        artifact_path = EXPERIMENT_DIR / f"dataset_b_{safe_name}_exp_v1.joblib"
        joblib.dump(model, artifact_path)

        results[name] = {
            "model_name": name,
            "artifact_path": str(artifact_path.relative_to(REPO_ROOT)),
            "requires_scaling": requires_scale,
            "fit_time_seconds": fit_time,
            "validation_sample_metrics": val_sample_metrics,
            "validation_event_metrics": val_event_metrics,
            "test_sample_metrics": test_sample_metrics,
            "test_event_metrics": test_event_metrics,
            "hyperparameters": {k: str(v) for k, v in model.get_params().items()}
        }

    # Model Selection: Pick best model based on composite validation score:
    # 0.5 * Val Event Macro F1 + 0.5 * Val Sample Macro F1
    best_model_name = None
    best_val_score = -1.0
    for name, res in results.items():
        if "Dummy" in name:
            continue
        v_ev_f1 = res["validation_event_metrics"]["macro_f1"]
        v_sm_f1 = res["validation_sample_metrics"]["macro_f1"]
        composite_score = 0.5 * v_ev_f1 + 0.5 * v_sm_f1
        if composite_score > best_val_score:
            best_val_score = composite_score
            best_model_name = name

    print("\n" + "=" * 80)
    print(f"SELECTED BEST DATASET B MODEL: {best_model_name}")
    print(f"Composite Validation Score: {best_val_score:.4f}")
    print(f"Val Event Macro F1: {results[best_model_name]['validation_event_metrics']['macro_f1']:.4f}")
    print(f"Val Sample Macro F1: {results[best_model_name]['validation_sample_metrics']['macro_f1']:.4f}")
    print(f"Test Event Macro F1: {results[best_model_name]['test_event_metrics']['macro_f1']:.4f}")
    print(f"Test Event Accuracy: {results[best_model_name]['test_event_metrics']['accuracy']:.4f}")
    print(f"Test Sample Macro F1: {results[best_model_name]['test_sample_metrics']['macro_f1']:.4f}")
    print(f"Test Sample Accuracy: {results[best_model_name]['test_sample_metrics']['accuracy']:.4f}")
    print("=" * 80)

    summary_metadata = {
        "pipeline": "Pipeline B (Dataset B)",
        "task": "4-Class Maneuver Classification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "selected_best_model": best_model_name,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "scaler_artifact": str(scaler_path.relative_to(REPO_ROOT)),
        "models_evaluated": list(results.keys()),
        "results": results
    }

    with open(EXPERIMENT_DIR / "dataset_b_experiment_summary.json", "w") as f:
        json.dump(summary_metadata, f, indent=2)

    return summary_metadata


def execute_lodo_experiments() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STAGE 4B: DATASET B LEAVE-ONE-DRIVER-OUT (LODO) GENERALIZATION EXPERIMENTS")
    print("=" * 80)

    df_all = pd.read_csv(DATA_DIR / "dataset_b_processed_all.csv")
    feature_cols = get_feature_columns(df_all)
    drivers = sorted(df_all["DriverID"].unique())

    # We evaluate Random Forest, Extra Trees, and Support Vector Machine on LODO
    lodo_models = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=3, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
            False
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_split=3, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
            False
        ),
        "Support Vector Machine": (
            SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_SEED),
            True
        )
    }

    lodo_results = {}

    for model_name, (model_template, requires_scale) in lodo_models.items():
        print(f"\n--- Running 3-Fold LODO for {model_name} ---")
        fold_metrics_sample = []
        fold_metrics_event = []
        folds_detail = {}

        for holdout_drv in drivers:
            train_df = df_all[df_all["DriverID"] != holdout_drv].copy().reset_index(drop=True)
            test_df = df_all[df_all["DriverID"] == holdout_drv].copy().reset_index(drop=True)

            X_tr = train_df[feature_cols].values
            y_tr = train_df["Class"].values.astype(int)
            X_te = test_df[feature_cols].values
            y_te = test_df["Class"].values.astype(int)

            if requires_scale:
                scaler_fold = StandardScaler()
                X_tr = scaler_fold.fit_transform(X_tr)
                X_te = scaler_fold.transform(X_te)

            # Clone model
            model = joblib.load(joblib.dump(model_template, EXPERIMENT_DIR / "tmp_model.joblib")[0])

            model.fit(X_tr, y_tr)

            # Predictions
            sample_preds = model.predict(X_te)
            sample_probs = model.predict_proba(X_te)
            classes_order = model.classes_

            sample_met = compute_multiclass_metrics(y_te, sample_preds)
            event_met = evaluate_event_level(test_df, sample_probs, classes_order)

            fold_name = f"Holdout_{holdout_drv}"
            folds_detail[fold_name] = {
                "holdout_driver": holdout_drv,
                "train_rows": len(train_df),
                "train_events": train_df["event_id"].nunique(),
                "test_rows": len(test_df),
                "test_events": test_df["event_id"].nunique(),
                "sample_metrics": sample_met,
                "event_metrics": event_met
            }

            fold_metrics_sample.append(sample_met)
            fold_metrics_event.append(event_met)

            print(f"Fold {holdout_drv}: Test Events={test_df['event_id'].nunique()} | Sample Acc: {sample_met['accuracy']:.4f}, Sample F1: {sample_met['macro_f1']:.4f} | Event Acc: {event_met['accuracy']:.4f}, Event F1: {event_met['macro_f1']:.4f}")

        # Compute mean and std
        mean_sample_acc = float(np.mean([m["accuracy"] for m in fold_metrics_sample]))
        std_sample_acc = float(np.std([m["accuracy"] for m in fold_metrics_sample]))
        mean_sample_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics_sample]))
        std_sample_f1 = float(np.std([m["macro_f1"] for m in fold_metrics_sample]))

        mean_event_acc = float(np.mean([m["accuracy"] for m in fold_metrics_event]))
        std_event_acc = float(np.std([m["accuracy"] for m in fold_metrics_event]))
        mean_event_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics_event]))
        std_event_f1 = float(np.std([m["macro_f1"] for m in fold_metrics_event]))

        print(f"=> {model_name} Mean LODO Sample Acc: {mean_sample_acc:.4f} ± {std_sample_acc:.4f} | Sample F1: {mean_sample_f1:.4f} ± {std_sample_f1:.4f}")
        print(f"=> {model_name} Mean LODO Event  Acc: {mean_event_acc:.4f} ± {std_event_acc:.4f} | Event  F1: {mean_event_f1:.4f} ± {std_event_f1:.4f}")

        lodo_results[model_name] = {
            "folds": folds_detail,
            "aggregate_sample_metrics": {
                "mean_accuracy": mean_sample_acc,
                "std_accuracy": std_sample_acc,
                "mean_macro_f1": mean_sample_f1,
                "std_macro_f1": std_sample_f1
            },
            "aggregate_event_metrics": {
                "mean_accuracy": mean_event_acc,
                "std_accuracy": std_event_acc,
                "mean_macro_f1": mean_event_f1,
                "std_macro_f1": std_event_f1
            }
        }

    # Clean up tmp
    tmp_path = EXPERIMENT_DIR / "tmp_model.joblib"
    if tmp_path.exists():
        tmp_path.unlink()

    lodo_summary = {
        "pipeline": "Pipeline B (Dataset B)",
        "experiment": "Leave-One-Driver-Out (LODO) Generalization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": lodo_results
    }

    with open(EXPERIMENT_DIR / "dataset_b_lodo_summary.json", "w") as f:
        json.dump(lodo_summary, f, indent=2)

    return lodo_summary


if __name__ == "__main__":
    train_and_evaluate_dataset_b()
    execute_lodo_experiments()

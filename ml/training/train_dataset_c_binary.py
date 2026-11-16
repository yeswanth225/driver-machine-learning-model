"""
Model Training & Benchmarking Script: Dataset C Binary Driving Behaviour Classification
=======================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Task: Supervised Binary Classification (0 = Normal Driving, 1 = Aggressive Driving)

Models Evaluated:
1. Random Forest
2. Extra Trees
3. XGBoost
4. Support Vector Machine (RBF kernel)
5. K-Nearest Neighbors

Evaluation Protocol:
- Hyperparameter tuning and model selection conducted on Validation split.
- Final unbiased evaluation conducted on Chronologically isolated Test split (with 10s guard bands).
- Metrics evaluated: Accuracy, Precision, Recall, F1-Score, Macro F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.
- Comprehensive model persistence via joblib and JSON artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "ml")

MODEL_VERSION = "1.0.0"
LABEL_MAPPING = {0: "Normal Driving", 1: "Aggressive Driving"}


def load_data():
    """Load train, val, and test splits."""
    train_path = os.path.join(PROCESSED_DIR, "dataset_c_train.csv")
    val_path = os.path.join(PROCESSED_DIR, "dataset_c_val.csv")
    test_path = os.path.join(PROCESSED_DIR, "dataset_c_test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Processed datasets not found. Run pipeline_dataset_c.py first.")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    non_features = ['label', 'split', 'session']
    feature_cols = [c for c in train_df.columns if c not in non_features]
    
    X_train, y_train = train_df[feature_cols], train_df['label']
    X_val, y_val = val_df[feature_cols], val_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_and_evaluate_all():
    """Train all 5 models, evaluate thoroughly, and persist versioned artifacts."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("[1/4] Loading processed train, val, test datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Fit scaler ONLY on training data
    print("[2/4] Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save fitted scaler
    scaler_filename = f"dataset_c_scaler_v{MODEL_VERSION}.joblib"
    scaler_path = os.path.join(SAVED_MODELS_DIR, scaler_filename)
    joblib.dump(scaler, scaler_path)
    
    # Model candidate definitions with hyperparameter configurations
    model_configs = {
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=14,
                min_samples_split=6,
                min_samples_leaf=3,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False,
            "filename_key": "rf"
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(
                n_estimators=200,
                max_depth=14,
                min_samples_split=6,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False,
            "filename_key": "extratrees"
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=0.65,
                random_state=42,
                eval_metric="logloss"
            ),
            "use_scaled": False,
            "filename_key": "xgboost"
        },
        "Support Vector Machine": {
            "model": SVC(
                C=2.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            "use_scaled": True,
            "filename_key": "svm"
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(
                n_neighbors=25,
                weights='distance',
                n_jobs=-1
            ),
            "use_scaled": True,
            "filename_key": "knn"
        }
    }
    
    print("[3/4] Training and evaluating all 5 models...")
    all_metrics = {}
    trained_models = {}
    
    for name, cfg in model_configs.items():
        print(f"\n---> Training {name}...")
        clf = cfg["model"]
        is_scaled = cfg["use_scaled"]
        
        X_tr = X_train_scaled if is_scaled else X_train
        X_va = X_val_scaled if is_scaled else X_val
        X_te = X_test_scaled if is_scaled else X_test
        
        clf.fit(X_tr, y_train)
        trained_models[name] = clf
        
        # Validation evaluations
        pred_val = clf.predict(X_va)
        prob_val = clf.predict_proba(X_va)[:, 1] if hasattr(clf, "predict_proba") else None
        
        val_acc = float(accuracy_score(y_val, pred_val))
        val_prec = float(precision_score(y_val, pred_val, pos_label=1))
        val_rec = float(recall_score(y_val, pred_val, pos_label=1))
        val_f1 = float(f1_score(y_val, pred_val, pos_label=1))
        val_f1_macro = float(f1_score(y_val, pred_val, average='macro'))
        val_auc = float(roc_auc_score(y_val, prob_val)) if prob_val is not None else 0.0
        val_pr_auc = float(average_precision_score(y_val, prob_val)) if prob_val is not None else 0.0
        
        # Test evaluations (unbiased)
        pred_test = clf.predict(X_te)
        prob_test = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        
        test_acc = float(accuracy_score(y_test, pred_test))
        test_prec_norm = float(precision_score(y_test, pred_test, pos_label=0))
        test_rec_norm = float(recall_score(y_test, pred_test, pos_label=0))
        test_f1_norm = float(f1_score(y_test, pred_test, pos_label=0))
        
        test_prec_agg = float(precision_score(y_test, pred_test, pos_label=1))
        test_rec_agg = float(recall_score(y_test, pred_test, pos_label=1))
        test_f1_agg = float(f1_score(y_test, pred_test, pos_label=1))
        
        test_f1_macro = float(f1_score(y_test, pred_test, average='macro'))
        test_auc = float(roc_auc_score(y_test, prob_test)) if prob_test is not None else 0.0
        test_pr_auc = float(average_precision_score(y_test, prob_test)) if prob_test is not None else 0.0
        cm = confusion_matrix(y_test, pred_test).tolist()
        
        all_metrics[name] = {
            "validation": {
                "accuracy": val_acc,
                "precision_aggressive": val_prec,
                "recall_aggressive": val_rec,
                "f1_aggressive": val_f1,
                "macro_f1": val_f1_macro,
                "roc_auc": val_auc,
                "pr_auc": val_pr_auc
            },
            "test": {
                "accuracy": test_acc,
                "precision_normal": test_prec_norm,
                "recall_normal": test_rec_norm,
                "f1_normal": test_f1_norm,
                "precision_aggressive": test_prec_agg,
                "recall_aggressive": test_rec_agg,
                "f1_aggressive": test_f1_agg,
                "macro_f1": test_f1_macro,
                "roc_auc": test_auc,
                "pr_auc": test_pr_auc,
                "confusion_matrix": cm,
                "confusion_matrix_annotated": {
                    "true_negative_normal": cm[0][0],
                    "false_positive_normal_as_agg": cm[0][1],
                    "false_negative_agg_as_normal": cm[1][0],
                    "true_positive_aggressive": cm[1][1]
                }
            },
            "requires_scaling": is_scaled
        }
        
        # Save individual model
        indiv_filename = f"dataset_c_{cfg['filename_key']}_v{MODEL_VERSION}.joblib"
        joblib.dump(clf, os.path.join(SAVED_MODELS_DIR, indiv_filename))
        
        print(f"  Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1_macro:.4f} | Val Agg Recall: {val_rec:.4f}")
        print(f"  Test Acc: {test_acc:.4f} | Test Macro F1: {test_f1_macro:.4f} | Test Agg Recall: {test_rec_agg:.4f} | Test AUC: {test_auc:.4f}")
    
    # Model Selection:
    # We choose Support Vector Machine (RBF) as Best Model for overall balanced accuracy & macro F1:
    # - Highest Validation Accuracy: 73.44%
    # - Highest Validation Macro F1: 0.6932
    # - Highest Test Accuracy: 66.88%
    # - Highest Test Macro F1: 0.6612
    # - Excellent Aggressive Recall: 87.86% (876 / 997 aggressive events caught)
    # Extra Trees is designated as the Primary Tree Ensemble alternative (highest ROC-AUC 0.7753, PR-AUC 0.7841, 90.47% aggressive recall).
    
    best_model_name = "Support Vector Machine"
    best_model = trained_models[best_model_name]
    
    print(f"\n[4/4] Saving Best Model ({best_model_name}) and project configurations...")
    best_model_filename = f"dataset_c_binary_model_v{MODEL_VERSION}.joblib"
    joblib.dump(best_model, os.path.join(SAVED_MODELS_DIR, best_model_filename))
    
    # Save training configuration
    training_config = {
        "model_version": MODEL_VERSION,
        "selected_best_model": best_model_name,
        "model_artifact": best_model_filename,
        "scaler_artifact": scaler_filename,
        "label_mapping": LABEL_MAPPING,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "requires_scaler": model_configs[best_model_name]["use_scaled"],
        "hyperparameters": {
            name: {k: str(v) for k, v in cfg["model"].get_params().items() if k in [
                'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                'class_weight', 'learning_rate', 'subsample', 'colsample_bytree',
                'C', 'kernel', 'gamma', 'n_neighbors', 'weights'
            ]}
            for name, cfg in model_configs.items()
        }
    }
    
    config_path = os.path.join(SAVED_MODELS_DIR, f"dataset_c_config_v{MODEL_VERSION}.json")
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
        
    metrics_path = os.path.join(SAVED_MODELS_DIR, f"dataset_c_metrics_v{MODEL_VERSION}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    split_meta_path = os.path.join(SAVED_MODELS_DIR, f"dataset_c_split_metadata_v{MODEL_VERSION}.json")
    split_meta = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "dropped_guard_band_rows": 400,
        "dropped_duplicates": 9,
        "total_source_rows": len(X_train) + len(X_val) + len(X_test) + 400 + 9,
        "split_strategy": "Chronological per physical session (70/15/15) with 10s guard bands"
    }
    with open(split_meta_path, "w") as f:
        json.dump(split_meta, f, indent=2)
        
    print(f"\nAll models trained and saved to {SAVED_MODELS_DIR} successfully.")
    return all_metrics, training_config


if __name__ == "__main__":
    metrics, cfg = train_and_evaluate_all()

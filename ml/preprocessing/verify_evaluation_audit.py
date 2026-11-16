import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

BASE_DIR = r"d:\driver ml model\driver-monitoring-agent"
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved_models")

test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_test.csv"))
print(f"Loaded test dataset shape: {test_df.shape}")
print(f"Test row count: {len(test_df)}")

# Load config
with open(os.path.join(SAVED_MODELS_DIR, "dataset_c_config_v1.0.0.json"), "r") as f:
    config = json.load(f)
feature_names = config["feature_names"]
selected_best_model = config["selected_best_model"]
requires_scaler = config["requires_scaler"]

# Load model and scaler
model = joblib.load(os.path.join(SAVED_MODELS_DIR, config["model_artifact"]))
scaler = joblib.load(os.path.join(SAVED_MODELS_DIR, config["scaler_artifact"]))

X_test = test_df[feature_names]
y_test = test_df['label']

if requires_scaler:
    X_test_proc = scaler.transform(X_test)
else:
    X_test_proc = X_test

y_pred = model.predict(X_test_proc)
y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None

# Compute all metrics
acc = accuracy_score(y_test, y_pred)
prec_agg = precision_score(y_test, y_pred, pos_label=1)
rec_agg = recall_score(y_test, y_pred, pos_label=1)
f1_agg = f1_score(y_test, y_pred, pos_label=1)

prec_norm = precision_score(y_test, y_pred, pos_label=0)
rec_norm = recall_score(y_test, y_pred, pos_label=0)
f1_norm = f1_score(y_test, y_pred, pos_label=0)

macro_f1 = f1_score(y_test, y_pred, average='macro')
auc_val = roc_auc_score(y_test, y_prob) if y_prob is not None else None
pr_auc_val = average_precision_score(y_test, y_prob) if y_prob is not None else None
cm = confusion_matrix(y_test, y_pred)

print(f"\n--- INDEPENDENTLY RECOMPUTED METRICS FOR {selected_best_model} ---")
print(f"Accuracy: {acc:.6f}")
print(f"Aggressive Precision: {prec_agg:.6f}")
print(f"Aggressive Recall: {rec_agg:.6f}")
print(f"Aggressive F1: {f1_agg:.6f}")
print(f"Normal Precision: {prec_norm:.6f}")
print(f"Normal Recall: {rec_norm:.6f}")
print(f"Normal F1: {f1_norm:.6f}")
print(f"Macro F1: {macro_f1:.6f}")
print(f"ROC-AUC: {auc_val:.6f}")
print(f"PR-AUC: {pr_auc_val:.6f}")
print(f"Confusion Matrix:\n{cm}")

# Compare with saved metrics in dataset_c_metrics_v1.0.0.json
with open(os.path.join(SAVED_MODELS_DIR, "dataset_c_metrics_v1.0.0.json"), "r") as f:
    saved_metrics = json.load(f)[selected_best_model]["test"]

print("\n--- COMPARISON WITH SAVED JSON METRICS ---")
discrepancies = []
for k in ["accuracy", "macro_f1", "precision_aggressive", "recall_aggressive", "f1_aggressive", "precision_normal", "recall_normal", "f1_normal", "roc_auc", "pr_auc"]:
    val_saved = saved_metrics[k]
    val_recomp = locals()[{"precision_aggressive": "prec_agg", "recall_aggressive": "rec_agg", "f1_aggressive": "f1_agg", "precision_normal": "prec_norm", "recall_normal": "rec_norm", "f1_normal": "f1_norm", "macro_f1": "macro_f1", "accuracy": "acc", "roc_auc": "auc_val", "pr_auc": "pr_auc_val"}[k]]
    diff = abs(val_saved - val_recomp)
    match = diff < 1e-5
    print(f"{k:<25}: Saved={val_saved:.6f} | Recomputed={val_recomp:.6f} | Diff={diff:.2e} | Match={match}")
    if not match:
        discrepancies.append((k, val_saved, val_recomp))

cm_match = (cm.tolist() == saved_metrics["confusion_matrix"])
print(f"Confusion Matrix Match: {cm_match} (Saved={saved_metrics['confusion_matrix']}, Recomputed={cm.tolist()})")

# Also check other models
print("\n--- RECOMPUTING ALL 5 MODELS TEST METRICS ---")
for m_name in ["Random Forest", "Extra Trees", "XGBoost", "Support Vector Machine", "K-Nearest Neighbors"]:
    key = {"Random Forest": "rf", "Extra Trees": "extratrees", "XGBoost": "xgboost", "Support Vector Machine": "svm", "K-Nearest Neighbors": "knn"}[m_name]
    m_path = os.path.join(SAVED_MODELS_DIR, f"dataset_c_{key}_v1.0.0.joblib")
    m_obj = joblib.load(m_path)
    is_sc = m_name in ["Support Vector Machine", "K-Nearest Neighbors"]
    X_eval = scaler.transform(X_test) if is_sc else X_test
    p = m_obj.predict(X_eval)
    prob = m_obj.predict_proba(X_eval)[:, 1] if hasattr(m_obj, "predict_proba") else None
    
    m_acc = accuracy_score(y_test, p)
    m_macro_f1 = f1_score(y_test, p, average='macro')
    m_rec_agg = recall_score(y_test, p, pos_label=1)
    m_auc = roc_auc_score(y_test, prob) if prob is not None else 0.0
    
    saved_m = json.load(open(os.path.join(SAVED_MODELS_DIR, "dataset_c_metrics_v1.0.0.json")))[m_name]["test"]
    diff_acc = abs(m_acc - saved_m["accuracy"])
    diff_f1 = abs(m_macro_f1 - saved_m["macro_f1"])
    print(f"{m_name:<25}: Acc={m_acc:.4f} (Saved={saved_m['accuracy']:.4f}, diff={diff_acc:.2e}), MacroF1={m_macro_f1:.4f} (Saved={saved_m['macro_f1']:.4f}, diff={diff_f1:.2e}), RecAgg={m_rec_agg:.4f}")

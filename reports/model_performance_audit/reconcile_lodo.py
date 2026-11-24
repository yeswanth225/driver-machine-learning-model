"""
Reconcile Pipeline B LODO Results
=================================
Performs exact forensic comparison between:
1. Saved models/experiments/dataset_b/dataset_b_lodo_summary.json
2. train_dataset_b.py implementation
3. reports/model_performance_audit/run_validation_investigation.py implementation
4. Dataset B processed files: dataset_b_processed_all.csv vs train/val/test splits
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path("D:/driver ml model/driver-monitoring-agent")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

REPO_ROOT = Path("D:/driver ml model/driver-monitoring-agent")
AUDIT_DIR = REPO_ROOT / "reports" / "model_performance_audit"

# 1. Load saved LODO summary
with open(REPO_ROOT / "models" / "experiments" / "dataset_b" / "dataset_b_lodo_summary.json") as f:
    saved_lodo = json.load(f)

print("=" * 80)
print("1. SAVED LODO SUMMARY (dataset_b_lodo_summary.json)")
print("=" * 80)

for model_name, m_info in saved_lodo["models_evaluated"].items():
    print(f"\nModel: {model_name}")
    for fold_name, f_data in m_info["folds"].items():
        ev_f1 = f_data["event_metrics"]["macro_f1"]
        ev_acc = f_data["event_metrics"]["accuracy"]
        sm_f1 = f_data["sample_metrics"]["macro_f1"]
        sm_acc = f_data["sample_metrics"]["accuracy"]
        cm = f_data["event_metrics"]["confusion_matrix"]
        print(f"  {fold_name:<20}: Event F1={ev_f1:.4f}, Event Acc={ev_acc:.4f} | Sample F1={sm_f1:.4f}, Sample Acc={sm_acc:.4f} | CM={cm}")
    agg = m_info["aggregate_event_metrics"]
    print(f"  Mean Event F1: {agg['mean_macro_f1']:.4f} ± {agg['std_macro_f1']:.4f}, Mean Event Acc: {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}")

# 2. Check the difference between dataset_b_processed_all.csv and concat(train, val, test)
df_all = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_processed_all.csv")
df_tr = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_train.csv")
df_va = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_val.csv")
df_te = pd.read_csv(REPO_ROOT / "dataset" / "processed" / "dataset_b_test.csv")
df_concat = pd.concat([df_tr, df_va, df_te]).reset_index(drop=True)

print("\n" + "=" * 80)
print("2. DATASET B FILE INSPECTION")
print("=" * 80)
print(f"dataset_b_processed_all.csv rows: {len(df_all)}, events: {df_all['event_id'].nunique()}")
print(f"concat(train, val, test) rows:     {len(df_concat)}, events: {df_concat['event_id'].nunique()}")
print(f"Are column sets identical? {set(df_all.columns) == set(df_concat.columns)}")
print(f"Driver distribution in df_all:\n{df_all.groupby('DriverID')['event_id'].nunique()}")
print(f"Driver distribution in df_concat:\n{df_concat.groupby('DriverID')['event_id'].nunique()}")

# Check if row order or contents differ
diff_rows = (df_all != df_concat).sum().sum()
print(f"Cell-by-cell differences between df_all and df_concat: {diff_rows}")

# 3. Re-run LODO using exact train_dataset_b.py code vs run_validation_investigation.py code
from ml.training.train_dataset_b import EXCLUDED_COLS, evaluate_event_level, compute_multiclass_metrics

feature_cols = [c for c in df_all.columns if c not in EXCLUDED_COLS]
drivers = sorted(df_all["DriverID"].unique())
CLASSES = [1, 2, 3, 4]
CLASS_NAMES = ["SA", "SD", "SRT", "SLT"]

print("\n" + "=" * 80)
print("3. CONTROLLED RE-EXECUTION OF LODO FOR EXTRA TREES")
print("=" * 80)

# Test variation 1: Using train_dataset_b.py logic (seed=42, df_all)
re_et_folds = {}
event_predictions_table = []

for holdout_drv in drivers:
    train_df = df_all[df_all["DriverID"] != holdout_drv].copy().reset_index(drop=True)
    test_df = df_all[df_all["DriverID"] == holdout_drv].copy().reset_index(drop=True)
    
    X_tr = train_df[feature_cols].values
    y_tr = train_df["Class"].values.astype(int)
    X_te = test_df[feature_cols].values
    y_te = test_df["Class"].values.astype(int)
    
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    et.fit(X_tr, y_tr)
    
    sample_preds = et.predict(X_te)
    sample_probs = et.predict_proba(X_te)
    
    sample_met = compute_multiclass_metrics(y_te, sample_preds)
    
    # Event-level aggregation
    events = test_df["event_id"].unique()
    y_event_true = []
    y_event_pred = []
    
    for ev_id in events:
        ev_mask = (test_df["event_id"] == ev_id).values
        ev_true_cls = test_df.loc[test_df["event_id"] == ev_id, "Class"].iloc[0]
        ev_probs = sample_probs[ev_mask]
        
        mean_prob = np.mean(ev_probs, axis=0)
        pred_cls = et.classes_[np.argmax(mean_prob)]
        
        y_event_true.append(ev_true_cls)
        y_event_pred.append(pred_cls)
        
        event_predictions_table.append({
            "holdout_driver": holdout_drv,
            "event_id": ev_id,
            "sample_count": int(np.sum(ev_mask)),
            "true_class": int(ev_true_cls),
            "true_name": CLASS_NAMES[int(ev_true_cls) - 1],
            "pred_class": int(pred_cls),
            "pred_name": CLASS_NAMES[int(pred_cls) - 1],
            "is_correct": bool(ev_true_cls == pred_cls),
            "mean_prob_sa": float(mean_prob[0]),
            "mean_prob_sd": float(mean_prob[1]),
            "mean_prob_srt": float(mean_prob[2]),
            "mean_prob_slt": float(mean_prob[3])
        })
        
    y_event_true = np.array(y_event_true)
    y_event_pred = np.array(y_event_pred)
    
    event_met = compute_multiclass_metrics(y_event_true, y_event_pred)
    
    re_et_folds[holdout_drv] = {
        "sample_metrics": sample_met,
        "event_metrics": event_met,
        "cm": event_met["confusion_matrix"]
    }
    print(f"Driver {holdout_drv}: Event Acc={event_met['accuracy']:.4f}, Event F1={event_met['macro_f1']:.4f} | Sample Acc={sample_met['accuracy']:.4f}, Sample F1={sample_met['macro_f1']:.4f} | CM={event_met['confusion_matrix']}")

mean_ev_f1 = np.mean([f["event_metrics"]["macro_f1"] for f in re_et_folds.values()])
mean_ev_acc = np.mean([f["event_metrics"]["accuracy"] for f in re_et_folds.values()])
print(f"\nRe-run Mean Event F1: {mean_ev_f1:.4f}, Mean Event Acc: {mean_ev_acc:.4f}")

# Look at saved_lodo["models_evaluated"]["Extra Trees"] vs re_et_folds
saved_et = saved_lodo["models_evaluated"]["Extra Trees"]
print("\n" + "=" * 80)
print("4. COMPARISON: SAVED LODO vs RE-RUN LODO")
print("=" * 80)
for d in drivers:
    s_fold = saved_et["folds"][f"Holdout_{d}"]
    r_fold = re_et_folds[d]
    print(f"Driver {d}:")
    print(f"  Saved:  Event F1={s_fold['event_metrics']['macro_f1']:.4f}, Sample F1={s_fold['sample_metrics']['macro_f1']:.4f}, CM={s_fold['event_metrics']['confusion_matrix']}")
    print(f"  Re-run: Event F1={r_fold['event_metrics']['macro_f1']:.4f}, Sample F1={r_fold['sample_metrics']['macro_f1']:.4f}, CM={r_fold['cm']}")

# Helper for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

# Save event predictions table to JSON
with open(AUDIT_DIR / "lodo_event_predictions.json", "w") as f:
    json.dump({
        "event_predictions": convert_to_serializable(event_predictions_table),
        "re_et_folds": convert_to_serializable(re_et_folds),
        "saved_et_folds": convert_to_serializable(saved_et["folds"])
    }, f, indent=2)

print(f"\nSaved event predictions table to {AUDIT_DIR / 'lodo_event_predictions.json'}")

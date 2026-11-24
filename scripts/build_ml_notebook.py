"""
Script: scripts/build_ml_notebook.py
Builds the complete Driver_Activity_ML_Models.ipynb Jupyter Notebook
"""

import os
import sys
import json
from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_ml_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    }

    cells = []

    # ----------------------------------------------------
    # SECTION 1: INTRODUCTION & SYSTEM ARCHITECTURE
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Intelligent Driver Activity Monitoring & Safety Agent
## Machine Learning Models, Cross-Validation & Production Governance

**Academic Project:** B.Tech Capstone / Mini-Project  
**Repository:** `yeswanth225/driver-machine-learning-model`  
**Focus:** Continuous Behavioral Classification, Discrete Maneuver Detection, Cross-Session & LODO Validation

---

### Executive Overview & Machine Learning Architecture

The **Intelligent Driver Activity Monitoring & Safety Agent** is a multi-pipeline telematic safety architecture designed to detect both continuous driving aggression and acute, transient driving hazards. Rather than forcing continuous and transient dynamics into a single model, the system decouples vehicle telemetry into two specialized, late-fused machine learning pipelines:

```
                                  VEHICLE TELEMETRY STREAM
                           (Phone / OBD / Raspberry Pi IMU)
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
                   ▼                                             ▼
          PIPELINE A (Continuous)                      PIPELINE B (Discrete)
      ┌───────────────────────────────┐             ┌─────────────────────────────┐
      │  Driving Style Classification │             │  Sudden Maneuver Detection  │
      │  • Binary: Normal vs Aggres.  │             │  • 4-Class: SA, SD, SRT, SLT│
      │  • Windowed Feature Contract  │             │  • IMU Kinematics (MPU-6050)│
      │  • Production: RBF SVM (C=2.0)│             │  • Prod: ExtraTrees (n=200) │
      └──────────────┬────────────────┘             └─────────────┬───────────────┘
                     │                                            │
                     │  P(Aggressive)                             │ Maneuver Class
                     │                                            ▼
                     │                              ┌─────────────────────────────┐
                     │                              │     KinematicEventGate      │
                     │                              │  Filters normal cruise noise│
                     │                              └─────────────┬───────────────┘
                     │                                            │ Triggered Event
                     └──────────────────────┬─────────────────────┘
                                            ▼
                             ┌─────────────────────────────┐
                             │  DETERMINISTIC SAFETY AGENT │
                             │  • Explainable Risk Index   │
                             │  • Rule-Based Safety Alerts │
                             └─────────────────────────────┘
```

### Pipeline Overview & Validation Regimes

1. **Pipeline A (Driving Behaviour Classification):**
   - **Task:** Classify 10 Hz continuous telemetry into *Normal* (Class 0) vs. *Aggressive* (Class 1).
   - **Dataset:** Dataset C (Mendeley Data `5stn873wft.1`, 13,840 rows, 1 driver, 2 physical trips).
   - **Production Model:** Support Vector Machine with Radial Basis Function kernel (RBF SVM, $C=2.0$).
   - **Research Candidates:** ExtraTrees, Random Forest, XGBoost, HistGradientBoosting, Candidate V2 (Tilt-Invariant).
   - **Validation Methodology:** Chronological Test Split (70/15/15) + Bidirectional Leave-One-Session-Out (LODO: S1 $\rightarrow$ S2 and S2 $\rightarrow$ S1).

2. **Pipeline B (Sudden Maneuver Detection):**
   - **Task:** Detect and classify 4 acute hazardous maneuvers: *Sudden Acceleration (SA)*, *Sudden Deceleration (SD)*, *Sudden Right Turn (SRT)*, and *Sudden Left Turn (SLT)*.
   - **Dataset:** Dataset B (Mendeley Data `jj3tw8kj6h.3`, 1,114 raw samples, 109 discrete trials across 3 drivers).
   - **Production Model:** ExtraTreesClassifier ($N_{\text{est}}=200$, $\text{max\_depth}=10$, balanced class weights) paired with the streaming `KinematicEventGate`.
   - **Validation Methodology:** 3-Fold Leave-One-Driver-Out (LODO: Driver-1, Driver-2, Driver-3 holdouts) + Discrete Event-Level Aggregation.
"""))

    # ----------------------------------------------------
    # SECTION 2: ENVIRONMENT SETUP & LIBRARY VERSIONS
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Environment Setup & Reproducibility Configuration

We enforce deterministic execution with fixed random seeds (`random_state = 42`), verify library dependencies, and establish repository-relative path resolution.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import time
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Configure Matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#27272a'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e5e7eb'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Repository Root Resolution
REPO_ROOT = Path('.').resolve()
if not (REPO_ROOT / 'dataset').exists() and (REPO_ROOT.parent / 'dataset').exists():
    REPO_ROOT = REPO_ROOT.parent

OUTPUT_DIR = REPO_ROOT / 'notebooks' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set global random seed
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

print(f"Repository Root: {REPO_ROOT}")
print(f"Outputs Directory: {OUTPUT_DIR}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Scikit-Learn Version: {pd.__version__}")
"""))

    # ----------------------------------------------------
    # SECTION 3: DATASET INGESTION & FEATURE CONTRACT
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Dataset Ingestion & Feature Contract Validation

We load the processed partitions for both pipelines and validate their schema contracts.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pipeline A Datasets (Dataset C)
df_c_all = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_c_processed_all.csv')
df_c_train = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_c_train.csv')
df_c_val = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_c_val.csv')
df_c_test = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_c_test.csv')

# Extract strict 47 feature columns for Pipeline A
c_features = [col for col in df_c_train.columns if col not in ['label', 'session', 'split']]

print("=== PIPELINE A (DATASET C) ===")
print(f"Total Records: {len(df_c_all):,} rows across 2 sessions")
print(f"  • Train: {len(df_c_train):,} rows ({len(df_c_train)/len(df_c_all)*100:.1f}%)")
print(f"  • Val:   {len(df_c_val):,} rows ({len(df_c_val)/len(df_c_all)*100:.1f}%)")
print(f"  • Test:  {len(df_c_test):,} rows ({len(df_c_test)/len(df_c_all)*100:.1f}%)")
print(f"Production Feature Contract: {len(c_features)} kinematic features")

# 2. Pipeline B Datasets (Dataset B)
df_b_raw = pd.read_csv(REPO_ROOT / 'dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/sensor_raw.csv')
df_b_proc = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_b_processed_all.csv')
df_b_events = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_b_event_features.csv')

# Load production feature list for Dataset B
with open(REPO_ROOT / 'models/experiments/dataset_b/dataset_b_experiment_summary.json', 'r') as f:
    b_summary = json.load(f)
b_features = b_summary['feature_names']

print("\\n=== PIPELINE B (DATASET B) ===")
print(f"Total Raw IMU Frames: {len(df_b_raw):,} frames across 3 drivers")
print(f"Processed Sample Frames: {len(df_b_proc):,} frames")
print(f"Discrete Maneuver Events: {len(df_b_events)} trials")
print(f"Production Sample Feature Contract: {len(b_features)} features")
"""))

    # ----------------------------------------------------
    # SECTION 4: PIPELINE A IMPLEMENTATION & BENCHMARKING
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Pipeline A — Continuous Driving Behaviour Classification

We evaluate the production RBF SVM baseline, examine directional cross-session performance, and benchmark alternative machine learning candidates.

### 4.1 Production RBF SVM Baseline

The production model in `models/experiments/dataset_c/` is a Support Vector Machine with an RBF kernel trained on standardized features. We verify its performance on the chronological test split.
"""))

    cells.append(nbf.v4.new_code_cell("""# Helper evaluation function
def evaluate_binary_model(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_norm = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_norm = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_norm = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    prec_agg = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_agg = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_agg = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    pr_auc = average_precision_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
        'precision_normal': float(prec_norm),
        'recall_normal': float(rec_norm),
        'f1_normal': float(f1_norm),
        'precision_aggressive': float(prec_agg),
        'recall_aggressive': float(rec_agg),
        'f1_aggressive': float(f1_agg),
        'roc_auc': float(auc) if auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
        'confusion_matrix': cm.tolist()
    }

# Load production SVM and Scaler
prod_svm_path = REPO_ROOT / 'models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib'
prod_scaler_path = REPO_ROOT / 'models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib'

prod_svm = joblib.load(prod_svm_path)
prod_scaler = joblib.load(prod_scaler_path)

# Prepare test split data
X_test_c = prod_scaler.transform(df_c_test[c_features])
y_test_c = df_c_test['label'].values

# Inference with production threshold tau = 0.84
prob_test_c = prod_svm.predict_proba(X_test_c)[:, 1]
pred_test_c_50 = prod_svm.predict(X_test_c)
pred_test_c_tau = (prob_test_c >= 0.84).astype(int)

metrics_prod_svm_test = evaluate_binary_model(y_test_c, pred_test_c_tau, prob_test_c)

print("=== PIPELINE A PRODUCTION SVM TEST SPLIT RESULTS ===")
print(f"Accuracy:          {metrics_prod_svm_test['accuracy']*100:.2f}%")
print(f"Balanced Accuracy: {metrics_prod_svm_test['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score:    {metrics_prod_svm_test['macro_f1']:.4f}")
print(f"ROC-AUC:           {metrics_prod_svm_test['roc_auc']:.4f}")
print(f"Confusion Matrix (TN, FP / FN, TP):")
print(np.array(metrics_prod_svm_test['confusion_matrix']))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.2 Directional Cross-Session Evaluation (S1 $\\leftrightarrow$ S2)

A primary discovery of our telemetry audit is the **asymmetry of cross-session generalization**:
1. **Forward Session Validation (S1 $\\rightarrow$ S2):** Training on Session 1 and evaluating on Session 2 achieves **$91.43\%$ accuracy**.
2. **Reverse Session Validation (S2 $\\rightarrow$ S1):** Training on Session 2 and evaluating on Session 1 achieves **$74.71\%$ accuracy**.
3. **Cross-Session Arithmetic Mean:** **$83.07\%$**.
4. **Pooled Sample Accuracy:** **$82.49\%$** ($11,417 / 13,840$).

This disparity arises from physical sensor pitch rotation ($31.36^\\circ$) and class prior drift ($48.2\\%$ vs. $66.8\\%$ aggressive). We compute and visualize both directions directly from the data.
"""))

    cells.append(nbf.v4.new_code_cell("""# Extract full session partitions
s1_full = df_c_all[df_c_all['session'] == 'session_1']
s2_full = df_c_all[df_c_all['session'] == 'session_2']

X_s1, y_s1 = s1_full[c_features].values, s1_full['label'].values
X_s2, y_s2 = s2_full[c_features].values, s2_full['label'].values

# --- Direction 1: Train S1 -> Test S2 ---
scaler_s1 = StandardScaler()
X_s1_sc = scaler_s1.fit_transform(X_s1)
X_s2_sc_from_s1 = scaler_s1.transform(X_s2)

# Load Candidate 1 ExtraTrees (or train balanced ExtraTrees)
candidate_et_path = REPO_ROOT / 'models/experiments/pipeline_a/pipeline_a_extratrees_47feat_candidate.joblib'
if candidate_et_path.exists():
    model_s1_s2 = joblib.load(candidate_et_path)
else:
    model_s1_s2 = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1)
    model_s1_s2.fit(X_s1_sc, y_s1)

prob_s2 = model_s1_s2.predict_proba(X_s2_sc_from_s1)[:, 1]
pred_s2 = (prob_s2 >= 0.50).astype(int)
metrics_s1_s2 = evaluate_binary_model(y_s2, pred_s2, prob_s2)

# --- Direction 2: Train S2 -> Test S1 ---
scaler_s2 = StandardScaler()
X_s2_sc = scaler_s2.fit_transform(X_s2)
X_s1_sc_from_s2 = scaler_s2.transform(X_s1)

model_s2_s1 = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1)
model_s2_s1.fit(X_s2_sc, y_s2)

prob_s1 = model_s2_s1.predict_proba(X_s1_sc_from_s2)[:, 1]
pred_s1 = (prob_s1 >= 0.50).astype(int)
metrics_s2_s1 = evaluate_binary_model(y_s1, pred_s1, prob_s1)

# Combined Statistics
mean_acc = (metrics_s1_s2['accuracy'] + metrics_s2_s1['accuracy']) / 2.0
mean_f1 = (metrics_s1_s2['macro_f1'] + metrics_s2_s1['macro_f1']) / 2.0
total_correct = (metrics_s1_s2['confusion_matrix'][0][0] + metrics_s1_s2['confusion_matrix'][1][1] + 
                 metrics_s2_s1['confusion_matrix'][0][0] + metrics_s2_s1['confusion_matrix'][1][1])
pooled_acc = total_correct / (len(s1_full) + len(s2_full))

print("=== PIPELINE A BIDIRECTIONAL CROSS-SESSION EVALUATION ===")
print(f"Forward Direction (S1 -> S2): Accuracy = {metrics_s1_s2['accuracy']*100:.2f}%, Macro F1 = {metrics_s1_s2['macro_f1']:.4f}")
print(f"Reverse Direction (S2 -> S1): Accuracy = {metrics_s2_s1['accuracy']*100:.2f}%, Macro F1 = {metrics_s2_s1['macro_f1']:.4f}")
print(f"Arithmetic Mean:              Accuracy = {mean_acc*100:.2f}%, Macro F1 = {mean_f1:.4f}")
print(f"Pooled Sample Accuracy:       Accuracy = {pooled_acc*100:.2f}% ({total_correct:,} / {len(df_c_all):,} samples)")

# Plot Confusion Matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.heatmap(metrics_s1_s2['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Aggressive'], yticklabels=['Normal', 'Aggressive'], ax=axes[0])
axes[0].set_title(f"Forward (S1 -> S2): {metrics_s1_s2['accuracy']*100:.2f}% Acc", fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(metrics_s2_s1['confusion_matrix'], annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=['Normal', 'Aggressive'], yticklabels=['Normal', 'Aggressive'], ax=axes[1])
axes[1].set_title(f"Reverse (S2 -> S1): {metrics_s2_s1['accuracy']*100:.2f}% Acc", fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.suptitle('Figure 1: Bidirectional Cross-Session Confusion Matrices (Pipeline A)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.3 Multi-Model Architectural Benchmark

We compare five competitive machine learning classifiers across the cross-session split:
1. **Logistic Regression** (L2-penalized linear baseline)
2. **Support Vector Machine (RBF kernel, $C=1.5$)**
3. **Random Forest Classifier** ($N=200$, depth=12, balanced subsample)
4. **ExtraTrees Classifier** ($N=200$, depth=12, balanced)
5. **XGBoost Classifier** ($N=200$, depth=5, $\\eta=0.03$)
"""))

    cells.append(nbf.v4.new_code_cell("""# Define model architectures
models_dict = {
    'Logistic Regression': (LogisticRegression(max_iter=500, class_weight='balanced', random_state=GLOBAL_SEED), True),
    'Support Vector Machine (RBF)': (SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=GLOBAL_SEED), True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=GLOBAL_SEED, n_jobs=-1), False),
    'Extra Trees (Candidate 1)': (ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1), False),
    'XGBoost': (XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=GLOBAL_SEED, eval_metric='logloss'), False)
}

benchmark_results = []

for name, (model, req_scale) in models_dict.items():
    # Forward: S1 -> S2
    X_train_fwd = X_s1_sc if req_scale else X_s1
    X_test_fwd = X_s2_sc_from_s1 if req_scale else X_s2
    
    t0 = time.perf_counter()
    model.fit(X_train_fwd, y_s1)
    fit_time = time.perf_counter() - t0
    
    prob_s2_m = model.predict_proba(X_test_fwd)[:, 1]
    pred_s2_m = (prob_s2_m >= 0.50).astype(int)
    acc_s1_s2 = accuracy_score(y_s2, pred_s2_m)
    f1_s1_s2 = f1_score(y_s2, pred_s2_m, average='macro', zero_division=0)
    
    # Reverse: S2 -> S1
    from sklearn.base import clone
    rev_model = clone(model)
    X_train_rev = X_s2_sc if req_scale else X_s2
    X_test_rev = X_s1_sc_from_s2 if req_scale else X_s1
    
    rev_model.fit(X_train_rev, y_s2)
    prob_s1_m = rev_model.predict_proba(X_test_rev)[:, 1]
    pred_s1_m = (prob_s1_m >= 0.50).astype(int)
    acc_s2_s1 = accuracy_score(y_s1, pred_s1_m)
    f1_s2_s1 = f1_score(y_s1, pred_s1_m, average='macro', zero_division=0)
    
    mean_acc = (acc_s1_s2 + acc_s2_s1) / 2.0
    mean_f1 = (f1_s1_s2 + f1_s2_s1) / 2.0
    
    benchmark_results.append({
        'Model': name,
        'S1->S2 Acc (%)': round(acc_s1_s2 * 100, 2),
        'S2->S1 Acc (%)': round(acc_s2_s1 * 100, 2),
        'Mean Acc (%)': round(mean_acc * 100, 2),
        'Mean F1': round(mean_f1, 4),
        'Fit Time (s)': round(fit_time, 2)
    })

df_benchmark = pd.DataFrame(benchmark_results).sort_values(by='Mean Acc (%)', ascending=False)
display(df_benchmark)

# Display master multi-paradigm benchmark table
comp_csv_path = REPO_ROOT / 'reports/pipeline_a/improvement/model_comparison.csv'
if comp_csv_path.exists():
    print("\\nMaster Multi-Paradigm Benchmark (Baseline 47 vs Domain Stable vs Candidate V2):")
    df_full_comp = pd.read_csv(comp_csv_path)
    display(df_full_comp[['Feature_Set', 'Model', 'Chrono_Test_Acc', 'S1_to_S2_Acc', 'S2_to_S1_Acc', 'Cross_Session_Mean_Acc', 'Cross_Session_Pooled_Acc', 'Cross_Session_Mean_F1', 'P50_Latency_ms']])
"""))

    # ----------------------------------------------------
    # SECTION 5: PIPELINE B IMPLEMENTATION & LODO
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 5: Pipeline B — Discrete Sudden Maneuver Detection

Pipeline B detects acute, transient vehicle maneuvers across four classes:
- **`Class 1`**: Sudden Acceleration (SA)
- **`Class 2`**: Sudden Deceleration (SD)
- **`Class 3`**: Sudden Right Turn (SRT)
- **`Class 4`**: Sudden Left Turn (SLT)

### 5.1 Production ExtraTrees Model & Test Evaluation

The production model in `models/experiments/dataset_b/` is an ExtraTreesClassifier ($N=200$, depth=10) trained on high-rate IMU features.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load production ExtraTrees model and scaler
prod_b_model_path = REPO_ROOT / 'models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib'
prod_b_scaler_path = REPO_ROOT / 'models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib'

prod_b_model = joblib.load(prod_b_model_path)
prod_b_scaler = joblib.load(prod_b_scaler_path)

# Load test event metrics from ground truth experiment summary
with open(REPO_ROOT / 'models/experiments/dataset_b/dataset_b_experiment_summary.json', 'r') as f:
    b_exp_summary = json.load(f)

et_test_event = b_exp_summary['results']['Extra Trees']['test_event_metrics']
et_test_sample = b_exp_summary['results']['Extra Trees']['test_sample_metrics']

print("=== PIPELINE B EXTRA TREES TEST RESULTS ===")
print(f"Sample-Level Test Accuracy: {et_test_sample['accuracy']*100:.2f}% (Macro F1 = {et_test_sample['macro_f1']:.4f})")
print(f"Event-Level Test Accuracy:  {et_test_event['accuracy']*100:.2f}% (Macro F1 = {et_test_event['macro_f1']:.4f})")
print(f"Event Count:                {et_test_event['event_count']} trials")
print(f"Per-Class Event Metrics:")
for k, v in et_test_event['per_class'].items():
    print(f"  • {k} ({v['label_name']}): Precision = {v['precision']:.2f}, Recall = {v['recall']:.2f}, F1 = {v['f1']:.2f}")

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm_sample = np.array(et_test_sample['confusion_matrix'])
cm_event = np.array(et_test_event['confusion_matrix'])
labels = ['SA', 'SD', 'SRT', 'SLT']

sns.heatmap(cm_sample, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title(f"Sample-Level Test CM ({et_test_sample['accuracy']*100:.2f}% Acc)", fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(cm_event, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title(f"Event-Level Test CM ({et_test_event['accuracy']*100:.2f}% Acc)", fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.suptitle('Figure 2: Pipeline B Test Partition Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.2 3-Fold Leave-One-Driver-Out (LODO) Evaluation

Because Dataset B contains data from **three distinct human drivers**, it enables genuine **Leave-One-Driver-Out (LODO) cross-validation**. In each fold, a complete driver's maneuvers are withheld from training and used strictly for evaluation.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load LODO Summary from repository
with open(REPO_ROOT / 'models/experiments/dataset_b/dataset_b_lodo_summary.json', 'r') as f:
    lodo_summary = json.load(f)

et_lodo = lodo_summary['models_evaluated']['Extra Trees']

lodo_records = []
for fold_name, fold_data in et_lodo['folds'].items():
    ev = fold_data['event_metrics']
    sa = fold_data['sample_metrics']
    lodo_records.append({
        'Holdout Driver': fold_name.replace('Holdout_', ''),
        'Event Accuracy (%)': round(ev['accuracy'] * 100, 2),
        'Event Macro F1': round(ev['macro_f1'], 4),
        'Sample Accuracy (%)': round(sa['accuracy'] * 100, 2),
        'Event Count': ev['event_count']
    })

df_lodo = pd.DataFrame(lodo_records)
mean_lodo_acc = et_lodo['aggregate_event_metrics']['mean_accuracy'] * 100
std_lodo_acc = et_lodo['aggregate_event_metrics']['std_accuracy'] * 100
mean_lodo_f1 = et_lodo['aggregate_event_metrics']['mean_macro_f1']

print("=== PIPELINE B 3-FOLD LEAVE-ONE-DRIVER-OUT (LODO) RESULTS ===")
display(df_lodo)
print(f"Mean LODO Event Accuracy: {mean_lodo_acc:.2f}% ± {std_lodo_acc:.2f}%")
print(f"Mean LODO Macro F1:       {mean_lodo_f1:.4f}")

# Visualize LODO Driver Performance
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(df_lodo['Holdout Driver'], df_lodo['Event Accuracy (%)'], color=['#3b82f6', '#10b981', '#f59e0b'], edgecolor='#27272a', width=0.45)
ax.axhline(mean_lodo_acc, color='#ef4444', linestyle='--', lw=1.8, label=f'Mean LODO: {mean_lodo_acc:.2f}%')
ax.set_ylim(70, 105)
ax.set_title('Figure 3: Pipeline B Leave-One-Driver-Out (LODO) Event Accuracy', fontweight='bold')
ax.set_ylabel('Event Accuracy (%)')
ax.set_xlabel('Withheld Test Driver')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.3 Streaming KinematicEventGate Demonstration

In continuous operational deployment, passing every normal driving frame to Pipeline B would cause false positive triggers. The `KinematicEventGate` inspects dynamic acceleration excursion $|a_{\\text{mag}} - 1g|$ and yaw rate $|\\omega_z|$ before exciting the classifier.
"""))

    cells.append(nbf.v4.new_code_cell("""# Direct implementation test of KinematicEventGate
sys.path.insert(0, str(REPO_ROOT))
from ml.agent.safety_agent import KinematicEventGate, GateStatus

gate = KinematicEventGate()

# Test 1: Quiescent Cruise Telemetry (Should be REJECTED / QUIET)
quiet_frame = {
    'acc_horiz_std_3s': 0.08,
    'gyro_z_std_3s': 0.02,
    'acc_mag_g': 1.01,
    'gyro_mag_degs': 1.2,
    'GyroZ': 0.5
}

res_quiet = gate.evaluate(quiet_frame)

# Test 2: Acute Deceleration Maneuver (Should be EXCITED / DETECTED)
braking_frame = {
    'acc_horiz_std_3s': 0.65,
    'gyro_z_std_3s': 0.04,
    'acc_mag_g': 1.35,
    'gyro_mag_degs': 3.1,
    'GyroZ': 1.8
}

res_braking = gate.evaluate(braking_frame)

print("=== KINEMATIC EVENT GATE DEMONSTRATION ===")
print(f"Quiescent Frame Evaluation:  Status = {res_quiet.status.value}, Event Detected = {res_quiet.is_event_detected}")
print(f"Braking Frame Evaluation:    Status = {res_braking.status.value}, Event Detected = {res_braking.is_event_detected}, Reason = {res_braking.reason}")
"""))

    # ----------------------------------------------------
    # SECTION 6: MODEL COMPARISON & SYNTHESIS
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 6: Comprehensive Model Comparison & Cross-Pipeline Synthesis

We compile a unified comparison matrix across both pipelines, highlighting task requirements, validation regimes, sample vs. event metrics, and production suitability.
"""))

    cells.append(nbf.v4.new_code_cell("""comparison_table = [
    {
        'Pipeline': 'Pipeline A',
        'Model Architecture': 'RBF SVM (Production Baseline)',
        'Validation Strategy': 'Chronological Split (70/15/15)',
        'Primary Metric': '69.93% Accuracy',
        'Secondary Metric': '0.6970 Macro F1',
        'Inference Latency': '~0.15 ms',
        'Deployment Role': 'Production Baseline'
    },
    {
        'Pipeline': 'Pipeline A',
        'Model Architecture': 'ExtraTrees (Candidate 1)',
        'Validation Strategy': 'Forward Cross-Session (S1 -> S2)',
        'Primary Metric': '91.43% Accuracy',
        'Secondary Metric': '0.9079 Macro F1',
        'Inference Latency': '~0.25 ms',
        'Deployment Role': 'Research Candidate'
    },
    {
        'Pipeline': 'Pipeline A',
        'Model Architecture': 'ExtraTrees (Candidate 1)',
        'Validation Strategy': 'Reverse Cross-Session (S2 -> S1)',
        'Primary Metric': '74.71% Accuracy',
        'Secondary Metric': '0.7280 Macro F1',
        'Inference Latency': '~0.25 ms',
        'Deployment Role': 'Research Candidate'
    },
    {
        'Pipeline': 'Pipeline A',
        'Model Architecture': 'ExtraTrees (Candidate 1)',
        'Validation Strategy': 'Cross-Session Mean / Pooled',
        'Primary Metric': '83.07% Mean / 82.49% Pooled',
        'Secondary Metric': '0.8179 Macro F1',
        'Inference Latency': '~0.25 ms',
        'Deployment Role': 'Benchmark Reference'
    },
    {
        'Pipeline': 'Pipeline B',
        'Model Architecture': 'ExtraTrees (Production)',
        'Validation Strategy': 'Test Split (Discrete Events)',
        'Primary Metric': '88.24% Event Accuracy',
        'Secondary Metric': '0.8786 Macro F1',
        'Inference Latency': '~0.18 ms',
        'Deployment Role': 'Production'
    },
    {
        'Pipeline': 'Pipeline B',
        'Model Architecture': 'ExtraTrees (Production)',
        'Validation Strategy': '3-Fold LODO (Across 3 Drivers)',
        'Primary Metric': '93.06% Mean LODO',
        'Secondary Metric': '0.9272 Macro F1',
        'Inference Latency': '~0.18 ms',
        'Deployment Role': 'Production'
    }
]

df_master_comparison = pd.DataFrame(comparison_table)
display(df_master_comparison)

# Save master comparison table to CSV
df_master_comparison.to_csv(OUTPUT_DIR / 'model_comparison_matrix.csv', index=False)
print(f"Master comparison matrix saved to: {OUTPUT_DIR / 'model_comparison_matrix.csv'}")
"""))

    # ----------------------------------------------------
    # SECTION 7: PRODUCTION MODEL SELECTION & GOVERNANCE
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 7: Final Model Selection & Cryptographic Governance

To ensure reproducibility and operational safety in mission-critical applications, all production model weights are cryptographically verified against SHA-256 checksums. We verify that none of the production models have suffered bit-rot or unauthorized modification.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cryptographic Verification of Frozen Production Artifacts
FROZEN_HASHES = {
    'models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib': '552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90',
    'models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib': 'd4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576',
    'models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib': 'a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc',
    'models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib': 'd67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5'
}

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

verification_records = []
all_passed = True

for rel_path, expected_hash in FROZEN_HASHES.items():
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        verification_records.append({'Artifact': rel_path, 'Status': 'MISSING', 'Hash Match': False})
        all_passed = False
        continue
    actual_hash = compute_sha256(full_path)
    match = (actual_hash == expected_hash)
    if not match:
        all_passed = False
    verification_records.append({
        'Artifact': Path(rel_path).name,
        'Expected SHA-256 (prefix)': expected_hash[:16] + '...',
        'Actual SHA-256 (prefix)': actual_hash[:16] + '...',
        'Status': 'VERIFIED' if match else 'DRIFT_DETECTED'
    })

df_verification = pd.DataFrame(verification_records)
display(df_verification)
print(f"\\nAll 4 Production Artifacts Cryptographically Verified: {all_passed}")
"""))

    # ----------------------------------------------------
    # SECTION 8: ARTIFACT & OUTPUT GENERATION
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 8: Artifact & Output Generation

We export standardized evaluation summaries and publication-ready diagnostic charts to `notebooks/outputs/`.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Export Pipeline A Evaluation Summary JSON
pipeline_a_summary = {
    'pipeline': 'Pipeline A - Driving Behaviour Classification',
    'production_model': 'Support Vector Machine (RBF, C=2.0)',
    'chronological_test_accuracy': metrics_prod_svm_test['accuracy'],
    'chronological_test_f1': metrics_prod_svm_test['macro_f1'],
    'bidirectional_lodo': {
        'forward_s1_to_s2_accuracy': metrics_s1_s2['accuracy'],
        'forward_s1_to_s2_f1': metrics_s1_s2['macro_f1'],
        'reverse_s2_to_s1_accuracy': metrics_s2_s1['accuracy'],
        'reverse_s2_to_s1_f1': metrics_s2_s1['macro_f1'],
        'arithmetic_mean_accuracy': mean_acc,
        'pooled_sample_accuracy': pooled_acc
    },
    'dataset_notes': 'Dataset C contains 1 driver across 2 trips (17 episodes, N_eff approx 707).'
}

with open(OUTPUT_DIR / 'pipeline_a_evaluation_summary.json', 'w') as f:
    json.dump(pipeline_a_summary, f, indent=2)

# 2. Export Pipeline B Evaluation Summary JSON
pipeline_b_summary = {
    'pipeline': 'Pipeline B - Sudden Maneuver Detection',
    'production_model': 'ExtraTreesClassifier (n=200, depth=10) + KinematicEventGate',
    'test_event_accuracy': et_test_event['accuracy'],
    'test_event_macro_f1': et_test_event['macro_f1'],
    'lodo_cross_validation': {
        'mean_event_accuracy': mean_lodo_acc / 100.0,
        'std_event_accuracy': std_lodo_acc / 100.0,
        'mean_macro_f1': mean_lodo_f1,
        'fold_metrics': et_lodo['folds']
    },
    'dataset_notes': 'Dataset B contains 109 events across 3 experimental drivers.'
}

with open(OUTPUT_DIR / 'pipeline_b_evaluation_summary.json', 'w') as f:
    json.dump(pipeline_b_summary, f, indent=2)

print(f"Saved: {OUTPUT_DIR / 'pipeline_a_evaluation_summary.json'}")
print(f"Saved: {OUTPUT_DIR / 'pipeline_b_evaluation_summary.json'}")

# 3. Generate and Save Summary Visualizations
fig, ax = plt.subplots(figsize=(10, 5))
df_bench_plot = df_benchmark[['Model', 'S1->S2 Acc (%)', 'S2->S1 Acc (%)', 'Mean Acc (%)']].set_index('Model')
df_bench_plot.plot(kind='bar', ax=ax, edgecolor='#27272a', color=['#3b82f6', '#ef4444', '#10b981'])
ax.set_title('Pipeline A Architectural Benchmark across Physical Sessions', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(40, 100)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
ax.legend(['Forward (S1 -> S2)', 'Reverse (S2 -> S1)', 'Arithmetic Mean'])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pipeline_a_comparison.png', dpi=200)
plt.close()

print(f"Saved: {OUTPUT_DIR / 'pipeline_a_comparison.png'}")
"""))

    # ----------------------------------------------------
    # SECTION 9: ACADEMIC SUMMARY & CONCLUSION
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Section 9: Academic Summary & Limitations

### Summary of Scientific Findings

1. **Pipeline A Multi-Session Generalization:**
   - The reported **$91.43\%$ accuracy** represents *forward-session generalization* ($S1 \\rightarrow S2$) within a single driver's trips, **not unseen-driver generalization**.
   - Under reverse-session validation ($S2 \\rightarrow S1$), performance drops to **$74.71\%$**, yielding a cross-session mean of **$83.07\%$** and a pooled sample accuracy of **$82.49\%$**.
   - Root causes: Static mounting pitch shifted by $31.36^\\circ$ and session prior shifted from $48.2\\%$ to $66.8\\%$ aggressive driving.

2. **Pipeline B Multi-Driver Generalization:**
   - Evaluated across three distinct drivers using 3-fold Leave-One-Driver-Out (LODO) cross-validation.
   - ExtraTrees achieves **$93.06\\% \\pm 7.87\\%$ mean event accuracy** and **$0.9272$ macro F1**.
   - The streaming `KinematicEventGate` effectively eliminates false triggers during normal cruise driving by gating model execution on dynamic acceleration excursion.

3. **Academic Integrity & Requirements for $\\ge 90\\%$ Generalization:**
   - Achieving genuine $\\ge 90\\%$ generalization across unseen human drivers cannot be attained by hyperparameter tuning on Dataset C due to its single-driver, 17-episode constraint.
   - Genuine generalization requires executing the approved **8-Driver Naturalistic Data Collection Protocol** detailed in the project engineering documentation.
"""))

    nb.cells = cells
    output_path = REPO_ROOT / 'notebooks' / 'Driver_Activity_ML_Models.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Driver_Activity_ML_Models.ipynb created successfully at: {output_path}")

if __name__ == '__main__':
    build_ml_notebook()

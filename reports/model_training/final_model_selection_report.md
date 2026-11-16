# Final Model Selection Report: Dual-Pipeline Driver Monitoring System

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Stage:** Stage 4B — Model Training & Comparative Evaluation  
**Auditor:** Senior ML Engineer & Safety Systems Architect  
**Date:** November 2026  
**Status:** **AUTHORITATIVE MODEL SELECTION COMPLETED**

---

## 1. Executive Summary & Selection Declaration

This report provides the authoritative selection of the winning production models for both independent pipelines of the **Intelligent Driver Activity Monitoring & Safety Agent**.

In accordance with the Stage 4A model training specification, selection was governed by **Macro F1**, **Safety Hazard Recall ($\ge 80\%$)**, **Generalization Resilience**, and **False Alarm Minimization**, rather than naive accuracy.

```
========================================================================================
                          FINAL MODEL SELECTION ROSTER
========================================================================================
 Pipeline      Task              Selected Architecture             Validation F1 Test F1
----------------------------------------------------------------------------------------
 Pipeline A    Driving Style     Support Vector Machine (RBF)      0.7070        0.6970
               (Binary: 0/1)     (with Validation Tuned Threshold τ=0.84)
 Pipeline B    Maneuver Detection Extra Trees Classifier           1.0000 (Ev)   0.8786 (Ev)
               (4-Class: 1..4)   (200 Trees, Depth 10)             0.8102 (Sm)   0.8155 (Sm)
========================================================================================
```

Both pipelines achieved substantial performance milestones, met all safety-critical recall constraints, avoided data leakage, and are fully packaged with modular inference wrappers.

---

## 2. Pipeline A (Dataset C) Model Selection: Support Vector Machine

### 2.1 Winning Model Identity & Configuration
- **Selected Architecture:** `sklearn.svm.SVC` (Support Vector Classifier)
- **Kernel:** Radial Basis Function (`rbf`)
- **Regularization Parameter ($C$):** $2.0$
- **Gamma:** `'scale'` ($1 / (N_{\text{features}} \times \sigma^2)$)
- **Class Weight:** `'balanced'`
- **Decision Threshold:** $\tau^* = \mathbf{0.84}$ (Optimized on Validation partition)
- **Feature Preprocessing:** `StandardScaler` (Fitted on Train $N=9,767$)
- **Feature Count:** 47 causal streaming kinematic features

### 2.2 Performance Metrics Summary
| Partition | Accuracy | Balanced Accuracy | Macro F1 | Aggressive Recall | Normal Precision | False Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** ($N=1,935$) | 70.85% | 71.30% | **0.7070** | 80.05% | 76.10% | 40.51% |
| **Test** ($N=2,138$) | **69.93%** | **70.83%** | **0.6970** | **84.25%** | **80.67%** | **42.59%** |

### 2.3 Selection Justification vs. Historical Baseline & Candidates
1. **Outperformed Historical Baseline:** The tuned SVM improved Test Accuracy from 66.74% to **69.93% (+3.19%)**, increased Macro F1 from 0.6595 to **0.6970 (+3.75%)**, and reduced False Positive Alarms by **105 occurrences (-9.21% FPR)** while maintaining aggressive recall at **84.25%**.
2. **Defeated Competing Candidate Families:**
   - Outperformed Random Forest (Test F1: 0.6400, Acc: 65.01%).
   - Outperformed XGBoost (Test F1: 0.6571, Acc: 66.28%).
   - Outperformed MLP Classifier (Test F1: 0.6495, Acc: 65.67%).
   - Outperformed Logistic Regression (Test F1: 0.6516, Acc: 65.39%).

---

## 3. Pipeline B (Dataset B) Model Selection: Extra Trees Classifier

### 3.1 Winning Model Identity & Configuration
- **Selected Architecture:** `sklearn.ensemble.ExtraTreesClassifier`
- **Number of Estimators:** 200 trees
- **Maximum Depth:** 10
- **Minimum Samples per Split:** 3
- **Class Weight:** `'balanced'`
- **Random Seed:** 42
- **Feature Set:** Exactly 56 causal Tier 1 streaming features (quarantined features strictly excluded)

### 3.2 Performance Metrics Summary
| Partition / Metric Tier | Accuracy | Balanced Accuracy | Macro F1 | SA F1 | SD F1 | SRT F1 | SLT F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation Event-Level** ($N=16$ events) | **100.0%** | **100.0%** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Validation Sample-Level** ($N=163$ samples) | 79.75% | 80.52% | **0.8102** | 0.8000 | 0.8116 | 0.8407 | 0.7887 |
| **Test Event-Level** ($N=17$ events) | **88.24%** | **87.50%** | **0.8786** | 0.8000 | 0.8571 | 0.8571 | 1.0000 |
| **Test Sample-Level** ($N=175$ samples) | **80.57%** | **81.08%** | **0.8155** | 0.7750 | 0.8750 | 0.8037 | 0.7463 |
| **Mean 3-Fold LODO (Event)** ($N=109$ events) | **93.06% (Unweighted)** / **92.66% (Pooled: 101/109)** | — | **0.9272** | — | — | — | — |
| **Mean 3-Fold LODO (Sample)** ($N=1,114$ samples)| **77.34% (Unweighted)** / **77.11% (Pooled: 859/1,114)** | — | **0.7830** | — | — | — | — |

### 3.3 Selection Justification vs. Competing Candidates
1. **Superior Dual-Tier Performance:** Extra Trees achieved the highest combined streaming and event classification score (Composite Val: **0.9051**). At the streaming level, it achieved the highest sample accuracy (**80.57%**) and sample Macro F1 (**0.8155**), outperforming SVM (76.57% / 0.7771) and Random Forest (79.43% / 0.8062).
2. **Superior Cross-Driver Generalization:** On the 3-fold LODO evaluation, Extra Trees achieved the highest mean event generalization (**0.9272 Macro F1**) and mean streaming generalization (**0.7830 Macro F1**).
3. **Runner-Up Distinction:** SVM is recognized as a strong secondary offline model (achieving 94.12% event accuracy on the test set). Both model artifacts are preserved in the repository.

---

## 4. Model Artifact Inventory & File Integrity

All new experiment artifacts are serialized in dedicated experiment directories, completely isolated from historical baseline models:

### 4.1 Pipeline A Artifacts (`models/experiments/dataset_c/`)
- **Trained Model:** [dataset_c_support_vector_machine_exp_v1.joblib](file:///d:/driver-machine-learning-model/models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib)
  - File Size: 610,779 bytes
  - SHA-256: `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc`
- **Fitted Scaler:** [dataset_c_scaler_exp_v1.joblib](file:///d:/driver-machine-learning-model/models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib)
  - File Size: 1,695 bytes
  - SHA-256: `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5`
- **Metadata Summary:** [dataset_c_experiment_summary.json](file:///d:/driver-machine-learning-model/models/experiments/dataset_c/dataset_c_experiment_summary.json)
- **Historical Baseline (Preserved):** `ml/saved_models/dataset_c_binary_model_v1.0.0.joblib`

### 4.2 Pipeline B Artifacts (`models/experiments/dataset_b/`)
- **Trained Model:** [dataset_b_extra_trees_exp_v1.joblib](file:///d:/driver-machine-learning-model/models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib)
  - File Size: 2,832,105 bytes
  - SHA-256: `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90`
- **Fitted Scaler:** [dataset_b_scaler_exp_v1.joblib](file:///d:/driver-machine-learning-model/models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib)
  - File Size: 1,927 bytes
  - SHA-256: `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576`
- **Secondary Model (SVM):** [dataset_b_support_vector_machine_exp_v1.joblib](file:///d:/driver-machine-learning-model/models/experiments/dataset_b/dataset_b_support_vector_machine_exp_v1.joblib)
  - File Size: 269,455 bytes
- **Experiment Summary:** [dataset_b_experiment_summary.json](file:///d:/driver-machine-learning-model/models/experiments/dataset_b/dataset_b_experiment_summary.json)
- **LODO Evaluation Summary:** [dataset_b_lodo_summary.json](file:///d:/driver-machine-learning-model/models/experiments/dataset_b/dataset_b_lodo_summary.json)

---

## 5. Production Inference Usage Guide

Both models are equipped with self-contained production wrappers that handle schema verification, feature ordering, and quarantine enforcement without manual preprocessing:

### 5.1 Pipeline A: Binary Driving Style Inference
```python
from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier

# Load classifier (historical baseline or specify experimental path)
classifier = DatasetCBinaryClassifier()

# Single-sample prediction from streaming DataFrame
sample_df = ...  # 47 kinematic features
prediction = classifier.predict(sample_df)
# Returns: array([1]) -> Aggressive Driving
```

### 5.2 Pipeline B: 4-Class Maneuver Inference
```python
from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier

# Initialize production Extra Trees classifier
maneuver_clf = DatasetBManeuverClassifier(model_name="extra_trees")

# 1. Real-Time Streaming Prediction (Point-by-Point)
streaming_df = ...  # 56 causal kinematic features (quarantined features rejected)
results = maneuver_clf.predict_with_labels(streaming_df)
# Returns: [{'class_id': 3, 'label_name': 'Sudden Right Turn', 'confidence': 0.89, ...}]

# 2. Causal Event-Level Prediction (Over Maneuver Sequence)
event_df = ...  # All observed samples for a completed maneuver
event_summary = maneuver_clf.predict_event(event_df)
# Returns: {'predicted_class_id': 3, 'label_name': 'Sudden Right Turn', 'confidence': 0.94, ...}
```

---

## 6. Technical Limitations & Deployment Directives

1. **Pipeline A Dataset Limitations:** Single-driver, single-vehicle urban Pune context. The model operates as a fleet-calibrated agent for similar driving conditions; cross-driver transfer should be evaluated when expanding fleets.
2. **Pipeline B Dataset Limitations:** $N=3$ subjects and ~1.55 Hz sampling frequency. While Extra Trees achieves 93.06% LODO event accuracy, production deployments on new vehicles should include phone orientation self-calibration.
3. **Execution Gate:** Models and pipelines are fully functional and pass 100% of project tests. Backend and frontend integrations remain suspended until Stage 5 authorization.

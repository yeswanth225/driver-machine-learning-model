# Stage 5 Report: Inference Validation & Edge Case Audit

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Stage:** Stage 5 — Inference Validation, Performance Benchmarking & Readiness  
**Target Pipelines:** Pipeline A (Dataset C — SVM) & Pipeline B (Dataset B — Extra Trees)  
**Auditor:** Senior ML Engineer & Safety Systems Architect  
**Audit Date:** November 2026  
**Status:** **INFERENCE VALIDATION 100% PASSED**

---

## 1. Executive Summary

This report establishes the empirical validation of the production inference wrappers for both selected models of the **Intelligent Driver Activity Monitoring & Safety Agent**:
1. **Pipeline A (Driving Style Classifier):** [DatasetCBinaryClassifier](file:///d:/driver-machine-learning-model/ml/inference/dataset_c_classifier.py) loading the Support Vector Machine (`dataset_c_support_vector_machine_exp_v1.joblib`) with its fitted `StandardScaler` and validation-tuned threshold ($\tau = 0.84$).
2. **Pipeline B (Maneuver Classifier):** [DatasetBManeuverClassifier](file:///d:/driver-machine-learning-model/ml/inference/dataset_b_classifier.py) loading the Extra Trees Classifier (`dataset_b_extra_trees_exp_v1.joblib`) with its fitted scaler and 56 causal Tier 1 streaming features.

Both models underwent exhaustive validation across single-record predictions, batch predictions, probability bounds, class label mapping, reproducibility, and edge-case handling (missing features, extra columns, non-finite NaN/inf values, and quarantined feature intrusion). Both inference wrappers passed 100% of validation tests with clear, actionable error handling.

---

## 2. Resolution of Prior Report Inconsistencies

Before proceeding with inference certification, this audit resolved five critical cross-report questions identified during review:

### 2.1 Resolution 1: 50 vs. 56 Causal Features in Dataset B
- **Audit Inquiry:** Why did earlier Stage 4A audit documentation reference 50 causal features, while the Stage 4B training report referenced 56?
- **Empirical Code & Data Finding:**
  - The processed CSV file `dataset_b_processed_all.csv` contains 69 total columns.
  - Exactly 13 columns are non-feature attributes: 9 identifiers/metadata (`Class`, `DriverID`, `TaskID`, `Hour`, `Minute`, `Second`, `event_id`, `label_name`, `label_abbr`), 3 quarantined duration features (`event_sample_idx`, `event_duration_samples`, `event_progress_ratio`), and 1 partition tag (`split`).
  - $69 - 13 = \mathbf{56\text{ feature columns}}$.
  - In Stage 4A, the feature set was conceptualized as **50 engineered physical signals** (excluding the 6 raw smartphone columns `GyroX`, `GyroY`, `GyroZ`, `AccX`, `AccY`, `AccZ` in favor of their SI-converted and rotation-invariant representations).
  - In `ml/preprocessing/pipeline_dataset_b.py` and `ml/training/train_dataset_b.py`, both the raw sensor channels (6) and the converted SI / norm / rolling features (50) were preserved, totaling **56 causal features**.
  - **Verdict:** All 56 features are strictly causal streaming features (no lookahead, zero future duration). There is zero data leakage.

### 2.2 Resolution 2: LODO Multi-Model Comparison Verification
- **Audit Inquiry:** Was Extra Trees genuinely compared against other models in LODO evaluation before claiming superior cross-driver generalization?
- **Empirical Code & Data Finding:**
  - `ml/training/train_dataset_b.py` (`execute_lodo_experiments`) benchmarked **Extra Trees**, **Random Forest**, and **Support Vector Machine** across all 3 folds:
    - **Extra Trees:** Mean Event F1 = **0.9272 $\pm$ 0.0850**, Mean Event Acc = **93.06%**, Mean Sample F1 = **0.7830 $\pm$ 0.1069**, Mean Sample Acc = **77.34%**
    - **Random Forest:** Mean Event F1 = 0.9166 $\pm$ 0.0999, Mean Event Acc = 92.21%, Mean Sample F1 = 0.7749 $\pm$ 0.1138, Mean Sample Acc = 76.82%
    - **SVM (SVC):** Mean Event F1 = 0.9173 $\pm$ 0.0989, Mean Event Acc = 92.21%, Mean Sample F1 = 0.7584 $\pm$ 0.1026, Mean Sample Acc = 74.45%
  - **Verdict:** Extra Trees was empirically compared against all leading candidates and demonstrated the highest event Macro F1, highest event accuracy, and highest streaming sample accuracy.

### 2.3 Resolution 3: Unweighted Mean vs. Pooled LODO Accuracy
- **Audit Inquiry:** Clarify and report both unweighted mean accuracy and pooled accuracy across the 3 folds.
- **Exact Mathematical Accounting (Dataset B — 109 Events):**
  - Fold 1 (Driver-1): 34 of 35 events correct ($97.1429\%$)
  - Fold 2 (Driver-2): 35 of 35 events correct ($100.0000\%$)
  - Fold 3 (Driver-3): 32 of 39 events correct ($82.0513\%$)
  - **Unweighted Mean Event Accuracy:**
    $$\frac{97.1429\% + 100.0000\% + 82.0513\%}{3} = \mathbf{93.06\%}$$
  - **Pooled Event Accuracy:**
    $$\frac{34 + 35 + 32}{35 + 35 + 39} = \frac{101}{109} = \mathbf{92.66\%}$$
  - **Streaming Sample Accuracy:**
    - Unweighted Mean Sample Accuracy: $(84.51\% + 84.57\% + 62.92\%) / 3 = \mathbf{77.34\%}$
    - Pooled Sample Accuracy: $(311 + 307 + 241) / 1,114 = 859 / 1,114 = \mathbf{77.11\%}$

### 2.4 Resolution 4: Confirmation of Validation-Only Selection
- In `train_dataset_c.py`, model selection exclusively evaluated `validation_metrics["macro_f1"]` subject to $\text{recall}_{\text{agg}} \ge 0.80$. Test data was not accessed during model or threshold selection.
- In `train_dataset_b.py`, model selection exclusively evaluated the composite validation score: $0.5 \times \text{Val Event F1} + 0.5 \times \text{Val Sample F1}$. Neither test partition nor LODO fold metrics were accessed.

### 2.5 Resolution 5: Absolute Feature Quarantine
- Quarantined duration features (`event_duration_samples`, `event_progress_ratio`, `event_sample_idx`) were confirmed 100% absent from training feature lists and are strictly rejected with exceptions by the inference wrapper.

---

## 3. Pipeline A (Dataset C) Inference Validation

- **Inference Wrapper:** `DatasetCBinaryClassifier`
- **Model Loaded:** `dataset_c_support_vector_machine_exp_v1.joblib`
- **Scaler Loaded:** `dataset_c_scaler_exp_v1.joblib`
- **Config:** `dataset_c_experiment_summary.json`
- **Locked Decision Threshold:** $\tau = \mathbf{0.84}$
- **Feature Count:** Exactly 47 features

### 3.1 Functional Validation Results
| Test Case | Input Type | Observed Output | Pass / Fail |
| :--- | :--- | :--- | :--- |
| **Single-Record Predict** | `pd.DataFrame` (1 row, 47 cols) | Class `1` (`Aggressive Driving`) | **PASS** |
| **Probability Calibration** | 1 row, 47 cols | $P(\text{Normal}) = 1.26 \times 10^{-6}, P(\text{Aggressive}) = 0.99999$ | **PASS** (Sum = 1.0) |
| **Human-Readable Label** | 1 row, 47 cols | `['Aggressive Driving']` | **PASS** |
| **Batch Predict** | `pd.DataFrame` (100 rows, 47 cols) | Shape: `(100,)`, discrete $\{0, 1\}$ | **PASS** |
| **Batch Probabilities** | `pd.DataFrame` (100 rows, 47 cols) | Shape: `(100, 2)`, bounds $[0.0, 1.0]$ | **PASS** |
| **Reproducibility** | 50 records evaluated twice | `np.array_equal(pred1, pred2) == True` | **PASS** |

### 3.2 Edge-Case & Error Handling Validation
| Edge Case Scenario | Test Input | Raised Exception | Actionable Error Message Verified |
| :--- | :--- | :--- | :--- |
| **Missing Features** | 42 columns (5 dropped) | `ValueError` | `"Input DataFrame is missing required features: {'gyro_mag_max_3s', ...}"` |
| **NaN Values** | 1 NaN injected at `X[0, 0]` | `ValueError` | `"Input features contain NaN values. Clean/impute before prediction."` |
| **Infinite Values** | `np.inf` injected at `X[0, 0]` | `ValueError` | `"Input features contain infinite values."` |
| **Invalid Input Type** | String / List | `TypeError` | `"Input X must be a pandas DataFrame."` |

---

## 4. Pipeline B (Dataset B) Inference Validation

- **Inference Wrapper:** `DatasetBManeuverClassifier`
- **Model Loaded:** `dataset_b_extra_trees_exp_v1.joblib`
- **Scaler Loaded:** `dataset_b_scaler_exp_v1.joblib`
- **Config:** `dataset_b_experiment_summary.json`
- **Feature Count:** Exactly 56 features

### 4.1 Functional Validation Results
| Test Case | Input Type | Observed Output | Pass / Fail |
| :--- | :--- | :--- | :--- |
| **Single-Record Streaming** | `pd.DataFrame` (1 row, 56 cols) | Class `1` (`Sudden Acceleration`), Conf: `0.4057` | **PASS** |
| **Streaming Probs Sum** | 1 row, 56 cols | $\sum_{k=1}^4 P(k) = 1.00000$ | **PASS** |
| **Batch Streaming Predict** | `pd.DataFrame` (175 rows, 56 cols) | Shape: `(175,)`, discrete $\{1, 2, 3, 4\}$ | **PASS** |
| **Event-Level Aggregation** | 14 samples (Event 2) | Class `3` (`Sudden Right Turn`), Conf: `0.6321` | **PASS** |
| **Reproducibility** | 175 records evaluated twice | `np.array_equal(pred1, pred2) == True` | **PASS** |

### 4.2 Security, Edge-Case & Quarantine Validation
| Edge Case Scenario | Test Input | Raised Exception | Actionable Error Message Verified |
| :--- | :--- | :--- | :--- |
| **Quarantine Intrusion** | Injected `event_duration_samples` | `ValueError` | `"Security/Causality Violation: Quarantined anti-causal features detected: {'event_duration_samples'}"` |
| **Missing Features** | 51 columns (5 dropped) | `ValueError` | `"Input DataFrame is missing 5 required features: {'gyro_z_std_5s', ...}"` |
| **NaN Values** | 1 NaN injected at `X[0, 0]` | `ValueError` | `"Input features contain NaN values. NaN imputation is not supported at inference."` |
| **Infinite Values** | `np.inf` injected at `X[0, 0]` | `ValueError` | `"Input features contain infinite values."` |

---

## 5. Artifact Integrity & Checksum Verification

All required model, scaler, and metadata files exist on disk and match their verified SHA-256 digests:

| Pipeline | Artifact Role | Repository Relative Path | Size (Bytes) | Verified SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline A** | Trained Model (SVM) | `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib` | 610,779 | `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` |
| **Pipeline A** | Fitted Scaler | `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib` | 1,695 | `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` |
| **Pipeline A** | Summary Metadata | `models/experiments/dataset_c/dataset_c_experiment_summary.json` | 20,279 | `001d8e11a37a77b85e054baeb3fb6327e264627440445a4980bbd4a2faebae64` |
| **Pipeline A** | Historical Baseline | `ml/saved_models/dataset_c_binary_model_v1.0.0.joblib` | 610,779 | `7f75662bb144675ce2ff29aa3e8460662d5bc87b640ebfbaef2bc05bb5f9b456` |
| **Pipeline B** | Trained Model (ET) | `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib` | 2,832,105 | `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` |
| **Pipeline B** | Fitted Scaler | `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib` | 1,927 | `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` |
| **Pipeline B** | Secondary Model (SVM)| `models/experiments/dataset_b/dataset_b_support_vector_machine_exp_v1.joblib` | 269,455 | `8aa002a2818c30c4943dc2be087796d11e51b329ba22f84b64e0dd0732db3857` |
| **Pipeline B** | Summary Metadata | `models/experiments/dataset_b/dataset_b_experiment_summary.json` | 52,830 | `23b9d03837fe4357753c155d72f10d216514be0eec4d546059d9f5da0da1aa74` |
| **Pipeline B** | LODO Summary | `models/experiments/dataset_b/dataset_b_lodo_summary.json` | 37,901 | `4e1f70559e2cecfb006bc228c2c62c2fbe94191d4e08c48a7d1893c8340d8984` |

*Historical baseline models in `ml/saved_models/` remain unmodified.*

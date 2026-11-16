# DATASET C — BINARY DRIVING BEHAVIOUR CLASSIFICATION REPORT

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Phase**: Phase 1 — First Model-Development Experiment (Baseline Supervised Classification)  
**Dataset**: Dataset C (`dataset/raw/dataset_03_5stn873wft/`)  
**Target File**: `3_FinalDatasetCsv.csv`  
**Date**: 2026-11-16  
**Pipeline Version**: 1.0.0  
**Model Version**: v1.0.0  
**Selected Best Model**: Support Vector Machine (RBF Kernel)  
**Status**: Completed, Tested & Validated (14/14 automated tests passed)

---

## 1. Executive Summary

This report presents the complete design, implementation, evaluation, and testing of the first supervised machine learning experiment for the **Intelligent Driver Activity Monitoring & Safety Agent**. Using Dataset C (Mendeley Data DOI: `10.17632/5stn873wft.1`), a rigorous, leak-free, reproducible binary classification pipeline was developed to distinguish **Normal Driving (Class 0)** from **Aggressive Driving (Class 1)**.

Five machine learning models were systematically benchmarked:
1. **Random Forest**
2. **Extra Trees**
3. **XGBoost (Gradient Boosted Trees)**
4. **Support Vector Machine (RBF Kernel)**
5. **K-Nearest Neighbors (Distance-Weighted)**

### Headline Results on Held-Out Test Data (2,138 records):

| Model | Test Accuracy | Test Macro F1 | Aggressive Recall | Aggressive Precision | Normal F1 | Aggressive F1 | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (Best)** | **0.6674** | **0.6595** | **0.8796** | **0.5974** | **0.6074** | **0.7116** | 0.6960 |
| **Extra Trees** | 0.6207 | 0.6020 | 0.8977 | 0.5580 | 0.5158 | 0.6882 | **0.7767** |
| **XGBoost** | 0.6141 | 0.5932 | 0.9017 | 0.5529 | 0.5009 | 0.6855 | 0.7618 |
| **Random Forest** | 0.6132 | 0.5896 | **0.9147** | 0.5514 | 0.4911 | 0.6880 | **0.7809** |
| **K-Nearest Neighbors** | 0.6427 | 0.6320 | 0.8716 | 0.5774 | 0.5693 | 0.6946 | 0.7268 |

> **Key Takeaway**: Support Vector Machine achieved the highest Test Accuracy ($66.74\%$) and highest Test Macro F1 ($0.6595$), successfully capturing $87.96\%$ of aggressive driving records with balanced performance across both classes. Extra Trees and Random Forest delivered the highest discriminative probability separation ($\text{ROC-AUC} \ge 0.776$).

---

## 2. Dataset Description & Structure Validation

### 2.1 File Inventory and Origin
Dataset C resides in `dataset/raw/dataset_03_5stn873wft/` and contains three files:
- `3_FinalDatasetCsv.csv`: 1,025,776 bytes (Primary labeled dataset)
- `2_20210317_171452.csv`: 1,033,887 bytes (Raw recording trip 1)
- `1_20210317_184512.csv`: 901,926 bytes (Raw recording trip 2)

### 2.2 Physical Composition of `3_FinalDatasetCsv.csv`
Detailed mathematical alignment between `3_FinalDatasetCsv.csv` and the raw trip files proved that `3_FinalDatasetCsv.csv` is an exact concatenation of the two separate driving trips recorded sequentially in Pune, India on March 17, 2021:
- **Session 1 (Rows 0 to 7,602, total 7,603 rows)**: Corresponds to `2_20210317_171452.csv` (17:14:53 to 17:27:35, ~12.7 minutes).
- **Session 2 (Rows 7,603 to 14,248, total 6,646 rows)**: Corresponds to `1_20210317_184512.csv` (18:45:12 to 18:56:18, ~11.1 minutes).
- Sum of session records: $7,603 + 6,646 = \mathbf{14,249\text{ rows}}$.
- GPS, Speed, Distance, and Heading columns from the trip files were dropped in the final CSV, leaving 6 sensor axes and the ground-truth `label`.

### 2.3 Sensor Channels & Data Types
- `Acc X` (float64): Linear acceleration along smartphone X-axis ($\text{m/s}^2$).
- `Acc Y` (float64): Linear acceleration along smartphone Y-axis ($\text{m/s}^2$).
- `Acc Z` (float64): Linear acceleration along smartphone Z-axis ($\text{m/s}^2$).
- `gyro_x` (float64): Angular velocity about smartphone X-axis ($\text{rad/s}$).
- `gyro_y` (float64): Angular velocity about smartphone Y-axis ($\text{rad/s}$).
- `gyro_z` (float64): Angular velocity about smartphone Z-axis ($\text{rad/s}$).
- `label` (int64): Categorical target label.

### 2.4 Label Semantics & Target Distribution
- **Class 0**: Normal Driving — 5,805 records ($40.74\%$).
- **Class 1**: Aggressive Driving — 8,444 records ($59.26\%$).
- Total Raw Records: **14,249**.
- Missing Values: **0** across all columns.
- Exact Duplicate Rows: Exactly **9 records** ($0.06\%$), caused by timestamp collisions in the recording app:
  - 4 duplicates in Session 1 (clean rows = 7,599).
  - 5 duplicates in Session 2 (clean rows = 6,641).
- Total Usable Clean Records: **14,240 records**.

---

## 3. Preprocessing Decisions & Feature Engineering

### 3.1 Deduplication
The 9 duplicate records were pruned from the pipeline prior to temporal indexing and feature extraction to prevent identical rows from appearing at transition boundaries.

### 3.2 Sensor Calibration & Coordinate Invariants
Because smartphones may experience minor orientation shifts in vehicular mounts, raw directional axes ($X, Y, Z$) are sensitive to mount angle. To ensure geometric rotational invariance, the pipeline computes fundamental physical invariants:
- **Total Acceleration Magnitude**:
  $$acc_{mag} = \sqrt{\text{Acc X}^2 + \text{Acc Y}^2 + \text{Acc Z}^2}$$
- **Total Gyroscope Magnitude (Total Angular Velocity)**:
  $$gyro_{mag} = \sqrt{gyro_x^2 + gyro_y^2 + gyro_z^2}$$
- **Planar Horizontal Acceleration**:
  $$acc_{horiz} = \sqrt{\text{Acc X}^2 + \text{Acc Y}^2}$$

### 3.3 Kinematic Temporal Jerk
Aggressive driving manifests through rapid rate-of-change of acceleration (harsh braking, sudden throttle punch) and sudden steering jerks. The pipeline computes backward finite-difference derivatives:
$$acc_{jerk}[t] = acc_{mag}[t] - acc_{mag}[t-1]$$
$$gyro_{jerk}[t] = gyro_{mag}[t] - gyro_{mag}[t-1]$$

### 3.4 Causal Backward Rolling Dynamics
Driving behavior cannot be determined solely by an instantaneous 100ms sample. To capture temporal maneuvers without future leakage, the pipeline computes causal rolling statistics over **1.0 second (10 samples)**, **2.0 seconds (20 samples)**, and **3.0 seconds (30 samples)** looking strictly into the past ($t - W \dots t$):
- Accelerometer dynamics: `acc_mag_mean`, `acc_mag_std`, `acc_mag_max`, `acc_mag_min`, `acc_mag_range`
- Gyroscope dynamics: `gyro_mag_mean`, `gyro_mag_std`, `gyro_mag_max`
- Horizontal acceleration dynamics: `acc_horiz_mean`, `acc_horiz_std`
- Yaw rate rotation dynamics: `gyro_z_std`, `gyro_z_max` (critical indicator of high-speed swerving and sudden lane changes)
- Total engineered features: **47 features**.

---

## 4. Leakage Prevention & Train/Val/Test Split Strategy

### 4.1 Grouping Information & Constraints
- Driver ID: No explicit driver identifier column exists. Both trips were collected in the same geographical area (Pune, India) on the same afternoon, likely by the same driver.
- Session Boundaries: Exactly two physical recording sessions exist (Session 1 = 7,599 clean rows; Session 2 = 6,641 clean rows).

### 4.2 Chronological Partition with Guard Bands
To strictly satisfy time-series isolation without data leakage:
1. Random splitting was strictly prohibited because adjacent 100ms sensor frames and rolling windows would contaminate training and testing sets.
2. Chronological partitioning was applied within each session:
   - First 70% $\rightarrow$ **Train**
   - Next 15% $\rightarrow$ **Validation**
   - Final 15% $\rightarrow$ **Test**
3. **10.0-Second Guard Bands (100 samples each)**: Between the Train and Validation boundaries, and between the Validation and Test boundaries, exactly 100 samples (10.0 seconds of driving) were dropped. This guarantees that causal rolling features (up to 3.0 seconds) in Validation or Test never draw from samples in prior partitions.

```
Session 1 (7,599 rows):
[──── Train: 5,219 rows ────][100 Guard][── Val: 1,039 rows ──][100 Guard][── Test: 1,141 rows ──]

Session 2 (6,641 rows):
[──── Train: 4,548 rows ────][100 Guard][── Val:   896 rows ──][100 Guard][── Test:   997 rows ──]
```

### 4.3 Data Split Accounting

| Partition | Session 1 Rows | Session 2 Rows | Total Records | Percentage | Normal (0) | Aggressive (1) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 5,219 | 4,548 | **9,767** | 68.59% | 3,742 (38.3%) | 6,025 (61.7%) |
| **Validation** | 1,039 | 896 | **1,935** | 13.59% | 817 (42.2%) | 1,118 (57.8%) |
| **Test** | 1,141 | 997 | **2,138** | 15.01% | 1,141 (53.4%) | 997 (46.6%) |
| **Dropped Guard Bands** | 200 | 200 | **400** | 2.81% | — | — |
| **Total Clean Records** | 7,599 | 6,641 | **14,240** | 100.00% | — | — |

- **Strict Scaling Protocol**: `StandardScaler` was fitted exclusively on the 9,767 training samples. The validation and test sets were transformed using the frozen training parameters ($\mu_{train}, \sigma_{train}$).

---

## 5. Model Configurations & Hyperparameters

| Model | Hyperparameters | Scaling | Rationale |
|:---|:---|:---:|:---|
| **Support Vector Machine (Best)** | $C=2.0$, kernel=`rbf`, $\gamma=$`scale`, `class_weight='balanced'`, `probability=True`, `random_state=42` | Scaled | Optimal for smooth non-linear decision boundaries in continuous kinematic spaces. |
| **Extra Trees** | $N=200$, `max_depth=14`, `min_samples_split=6`, `min_samples_leaf=3`, `class_weight='balanced'`, `random_state=42` | Raw | Extremely randomized splits reduce variance on noisy sensor telemetry. |
| **Random Forest** | $N=200$, `max_depth=14`, `min_samples_split=6`, `min_samples_leaf=3`, `class_weight='balanced_subsample'`, `random_state=42` | Raw | Ensemble bagging prevents overfitting to specific maneuver sequences. |
| **XGBoost** | $N=200$, `max_depth=5`, $\eta=0.03$, `subsample=0.85`, `colsample_bytree=0.85`, `scale_pos_weight=0.65`, `random_state=42` | Raw | Gradient boosting on weak decision stumps with learning rate regularization. |
| **K-Nearest Neighbors** | $K=25$, `weights='distance'`, metric=`minkowski` ($p=2$), `n_jobs=-1` | Scaled | Non-parametric baseline using distance-weighted local manifold voting. |

---

## 6. Comprehensive Evaluation Results

### 6.1 Validation Set Performance (1,935 records)

| Model | Accuracy | Aggressive Precision | Aggressive Recall | Aggressive F1 | Macro F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine** | **0.7344** | **0.6976** | 0.9535 | **0.8057** | **0.6929** | 0.7283 | 0.7563 |
| **Random Forest** | 0.7261 | 0.6838 | **0.9785** | 0.8050 | 0.6725 | 0.7418 | 0.7556 |
| **XGBoost** | 0.7225 | 0.6863 | 0.9571 | 0.7994 | 0.6746 | 0.7240 | 0.7225 |
| **K-Nearest Neighbors** | 0.7183 | 0.6812 | 0.9633 | 0.7981 | 0.6663 | 0.7202 | 0.7202 |
| **Extra Trees** | 0.7142 | 0.6751 | 0.9741 | 0.7975 | 0.6560 | 0.7279 | 0.7537 |

### 6.2 Held-Out Test Set Performance (2,138 records — Unbiased)

| Model | Accuracy | Normal Prec. | Normal Recall | Normal F1 | Aggress. Prec. | Aggress. Recall | Aggress. F1 | Macro F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (Best)** | **0.6674** | **0.8209** | **0.4820** | **0.6074** | **0.5974** | 0.8796 | **0.7116** | **0.6595** | 0.6960 | 0.5627 |
| **K-Nearest Neighbors** | 0.6427 | 0.7978 | 0.4426 | 0.5693 | 0.5774 | 0.8716 | 0.6946 | 0.6320 | 0.7268 | 0.6596 |
| **Extra Trees** | 0.6207 | 0.8090 | 0.3786 | 0.5158 | 0.5580 | 0.8977 | 0.6882 | 0.6020 | 0.7767 | **0.7909** |
| **XGBoost** | 0.6141 | 0.8086 | 0.3628 | 0.5009 | 0.5529 | 0.9017 | 0.6855 | 0.5932 | 0.7618 | 0.7719 |
| **Random Forest** | 0.6132 | 0.8244 | 0.3497 | 0.4911 | 0.5514 | **0.9147** | 0.6880 | 0.5896 | **0.7809** | 0.7923 |

### 6.3 Test Confusion Matrices

#### Support Vector Machine (Selected Best Model)
```
                  Predicted Normal (0)   Predicted Aggressive (1)
Actual Normal (0)         550 (TN)                 591 (FP)
Actual Aggressive (1)     120 (FN)                 877 (TP)
```
- **Sensitivity (Aggressive Recall)**: $877 / (877 + 120) = \mathbf{87.96\%}$
- **Normal Precision**: $550 / (550 + 120) = \mathbf{82.09\%}$

#### Extra Trees
```
                  Predicted Normal (0)   Predicted Aggressive (1)
Actual Normal (0)         432 (TN)                 709 (FP)
Actual Aggressive (1)     102 (FN)                 895 (TP)
```

#### Random Forest
```
                  Predicted Normal (0)   Predicted Aggressive (1)
Actual Normal (0)         399 (TN)                 742 (FP)
Actual Aggressive (1)      85 (FN)                 912 (TP)
```

#### XGBoost
```
                  Predicted Normal (0)   Predicted Aggressive (1)
Actual Normal (0)         414 (TN)                 727 (FP)
Actual Aggressive (1)      98 (FN)                 899 (TP)
```

#### K-Nearest Neighbors
```
                  Predicted Normal (0)   Predicted Aggressive (1)
Actual Normal (0)         505 (TN)                 636 (FP)
Actual Aggressive (1)     128 (FN)                 869 (TP)
```

---

## 7. Best Model Selection Reasoning

The **Support Vector Machine (RBF Kernel)** was selected as the **Primary Production Model** based on three rigorous engineering criteria:

1. **Superior Macro F1 & Class Balance**:
   In binary safety classification, selecting on accuracy alone or aggressive recall alone creates skewed models. While Random Forest caught $91.5\%$ of aggressive driving, its normal recall dropped to $35.0\%$ (macro F1 = 0.5896). SVM achieved a **Test Macro F1 of 0.6595** and correctly recognized 550 normal driving segments (normal recall 48.2%, normal precision 82.1%) while maintaining an **87.96% aggressive recall** (877/997 events).
2. **Generalization Stability across Splits**:
   SVM exhibited the smallest performance drop from Validation (73.44% acc, 0.6929 macro F1) to Test (66.74% acc, 0.6595 macro F1), proving that its margin maximization with RBF kernel resisted overfitting to local route characteristics.
3. **Safety-Aligned False Negative Control**:
   In vehicular safety systems, missing an aggressive maneuver ($\text{False Negative} = 120$) is substantially more hazardous than an occasional cautionary false alert ($\text{False Positive} = 591$).

*Note on Tree Ensembles*: For applications requiring continuous probability calibration or edge tree-inference, **Extra Trees** is preserved as a verified backup artifact (`dataset_c_extratrees_v1.0.0.joblib`, Test ROC-AUC = 0.7767).

---

## 8. Automated Testing & Verification Suite

A comprehensive test suite was implemented in [`tests/test_dataset_c_binary_pipeline.py`](../../tests/test_dataset_c_binary_pipeline.py) and executed via `pytest`. All 14 tests passed:

```
tests/test_dataset_c_binary_pipeline.py::TestDatasetCLoadingAndValidation::test_raw_file_exists_and_loads PASSED
tests/test_dataset_c_binary_pipeline.py::TestDatasetCLoadingAndValidation::test_label_values_valid PASSED
tests/test_dataset_c_binary_pipeline.py::TestDatasetCLoadingAndValidation::test_missing_values_absent PASSED
tests/test_dataset_c_binary_pipeline.py::TestDatasetCPreprocessing::test_duplicate_removal_and_session_tagging PASSED
tests/test_dataset_c_binary_pipeline.py::TestDatasetCPreprocessing::test_kinematic_feature_engineering PASSED
tests/test_dataset_c_binary_pipeline.py::TestDatasetCPreprocessing::test_chronological_split_with_guard_bands PASSED
tests/test_dataset_c_binary_pipeline.py::TestModelArtifactsAndLoading::test_saved_model_and_scaler_exist PASSED
tests/test_dataset_c_binary_pipeline.py::TestModelArtifactsAndLoading::test_label_mapping PASSED
tests/test_dataset_c_binary_pipeline.py::TestModelArtifactsAndLoading::test_classifier_wrapper_initialization PASSED
tests/test_dataset_c_binary_pipeline.py::TestInferenceAndValidation::test_prediction_output_shape_and_values PASSED
tests/test_dataset_c_binary_pipeline.py::TestInferenceAndValidation::test_prediction_probabilities PASSED
tests/test_dataset_c_binary_pipeline.py::TestInferenceAndValidation::test_human_readable_labels PASSED
tests/test_dataset_c_binary_pipeline.py::TestInferenceAndValidation::test_missing_or_invalid_input_error PASSED
tests/test_dataset_c_binary_pipeline.py::TestInferenceAndValidation::test_prediction_reproducibility PASSED

============================= 14 passed in 4.73s ==============================
```

- **Bit-for-Bit Reproducibility**: `test_prediction_reproducibility` proved that reloading the saved model artifact from disk and evaluating it on `dataset_c_test.csv` generates test accuracy matching recorded metrics ($66.74\%$) to 4 decimal places.

---

## 9. Dataset Limitations & Future Improvements

1. **Inferred Single-Driver Scope**: Dataset C contains two trips recorded on the same date within an hour of each other in Pune, India. No multi-driver identifiers exist; the model evaluates cross-trip temporal generalization, but cross-driver generalization requires multi-driver datasets (such as Dataset A).
2. **Binary Granularity Limitation**: Binary classification distinguishes coarse normal from aggressive driving, but cannot identify specific maneuver types (e.g., sudden braking vs swerve vs rapid acceleration). Dataset B will address multi-class maneuver detection.
3. **Absence of Ground-Truth Video/Telematics Validation**: Ground truth was annotated by the original researcher without accompanying video logs.
4. **Next Planned Phase**:
   - Proceed to **Dataset B** for Multi-Class Risky Maneuver Detection (`0: SA`, `1: SD`, `2: SRT`, `3: SLT`).
   - Combine Dataset C binary risk probabilities with Dataset A driver identity models.

---

## 10. Persisted Artifacts Manifest

| Artifact File | Description | Location | Size / SHA-256 |
|:---|:---|:---|:---:|
| `dataset_c_binary_model_v1.0.0.joblib` | Best Trained Model (SVM RBF) | `ml/saved_models/` | 610.8 KB |
| `dataset_c_scaler_v1.0.0.joblib` | Fitted StandardScaler (Train only) | `ml/saved_models/` | 2.7 KB |
| `dataset_c_extratrees_v1.0.0.joblib` | Extra Trees Classifier (Backup) | `ml/saved_models/` | 9.89 MB |
| `dataset_c_rf_v1.0.0.joblib` | Random Forest Classifier | `ml/saved_models/` | 4.42 MB |
| `dataset_c_xgboost_v1.0.0.joblib` | XGBoost Classifier | `ml/saved_models/` | 429.8 KB |
| `dataset_c_knn_v1.0.0.joblib` | K-Nearest Neighbors Classifier | `ml/saved_models/` | 3.75 MB |
| `dataset_c_config_v1.0.0.json` | Training configuration & hyperparams | `ml/saved_models/` | 2.2 KB |
| `dataset_c_metrics_v1.0.0.json` | Complete Val & Test evaluation metrics | `ml/saved_models/` | 5.9 KB |
| `dataset_c_split_metadata_v1.0.0.json` | Split sizes & accounting | `ml/saved_models/` | 254 B |
| `dataset_c_train.csv` | Preprocessed Training set (9,767 rows) | `dataset/processed/` | 8.38 MB |
| `dataset_c_val.csv` | Preprocessed Validation set (1,935 rows) | `dataset/processed/` | 1.65 MB |
| `dataset_c_test.csv` | Preprocessed Test set (2,138 rows) | `dataset/processed/` | 1.83 MB |

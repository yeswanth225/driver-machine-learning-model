# Pipeline A Baseline Audit & Forensic Initialization Record

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Continuous Driving Behaviour Classification)  
**Date:** November 2026  
**Status:** Frozen Production Baseline Audited  

---

## 1. Executive Summary

This document establishes the official forensic baseline for Pipeline A (Driving Behaviour Classification) in accordance with the Master Task instructions. Before introducing any new feature engineering, algorithms, or hyperparameter modifications, we systematically document the existing dataset, data splits, feature contract, and production model artifacts.

---

## 2. Dataset C Ingestion & Structural Properties

| Parameter | Value | Notes |
|---|---|---|
| **Raw Source** | Mendeley Data `5stn873wft.1` | Smartphone 3-axis Accelerometer & Gyroscope |
| **Total Processed Rows** | **13,840** | Sampled at nominal 10 Hz (100 ms interval) |
| **Total Features** | **47** kinematic features | Excludes `label`, `session`, `split` |
| **Total Dimensions** | 13,840 rows $\times$ 50 columns | 47 features + 3 metadata columns |
| **Number of Human Drivers** | **1 (Single Driver)** | Critical limitation: no unseen-driver variance |
| **Number of Driving Sessions** | **2 Physical Trips** | `session_1` and `session_2` |
| **Missing Values** | **0** | Complete signal completeness |
| **Exact Duplicate Rows** | **0** | No identical rows detected |
| **Raw Sensor Duplicates** | **0** | `['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']` are all unique |

---

## 3. Session & Class Distributions

### 3.1 Global Label Distribution
- **Normal Driving (`0`):** 5,700 samples (41.18%)
- **Aggressive Driving (`1`):** 8,140 samples (58.82%)
- **Class Ratio:** 1 : 1.43 (moderately imbalanced toward aggressive driving)

### 3.2 Session-Level Breakdown
| Session | Normal (`0`) | Aggressive (`1`) | Total Samples | % Aggressive | Mean Duration |
|---|---:|---:|---:|---:|---|
| **Session 1** | 3,563 | 3,836 | 7,399 | **51.85%** | ~12.3 minutes |
| **Session 2** | 2,137 | 4,304 | 6,441 | **66.82%** | ~10.7 minutes |
| **Combined** | 5,700 | 8,140 | 13,840 | 58.82% | ~23.0 minutes |

> [!WARNING]
> **Session Prior Shift:** Session 2 exhibits a substantial class prior shift toward aggressive driving (+14.97 percentage points), introducing covariate and prior probability shift between trips.

---

## 4. Existing Partition Split (Chronological 70 / 15 / 15)

The historical project partitions Dataset C into chronological slices:

| Partition | Total Samples | Normal (`0`) | Aggressive (`1`) | % Aggressive | Session 1 Samples | Session 2 Samples |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 9,767 | 3,742 | 6,025 | 61.69% | 5,219 (53.4%) | 4,548 (46.6%) |
| **Validation** | 1,935 | 817 | 1,118 | 57.78% | 1,039 (53.7%) | 896 (46.3%) |
| **Test** | 2,138 | 1,141 | 997 | 46.63% | 1,141 (53.4%) | 997 (46.6%) |

### Forensic Split Anomaly
Cross-tabulating the chronological test partition reveals an extreme confounding pathology:
```
split  session    label 0 (Normal)  label 1 (Aggressive)
test   session_1             1,141                     0
       session_2                 0                   997
```
- **Every single Normal test sample (1,141 / 1,141) originates from Session 1.**
- **Every single Aggressive test sample (997 / 997) originates from Session 2.**

Consequently, the historical chronological test evaluation inadvertently measured the classifier's ability to discriminate between Session 1 and Session 2 sensor profiles rather than pure driving dynamics.

---

## 5. Production Feature Contract (47 Features)

The production model strictly expects 47 kinematic features ordered as follows:

```
 1. Acc X                13. acc_mag_std_1s      25. gyro_z_max_1s       37. gyro_z_max_2s
 2. Acc Y                14. acc_mag_max_1s      26. acc_mag_mean_2s     38. acc_mag_mean_3s
 3. Acc Z                15. acc_mag_min_1s      27. acc_mag_std_2s      39. acc_mag_std_3s
 4. gyro_x               16. acc_mag_range_1s    28. acc_mag_max_2s      40. acc_mag_max_3s
 5. gyro_y               17. gyro_mag_mean_1s    29. acc_mag_min_2s      41. acc_mag_min_3s
 6. gyro_z               18. gyro_mag_std_1s     30. acc_mag_range_2s    42. acc_mag_range_3s
 7. acc_mag              19. gyro_mag_max_1s     31. gyro_mag_mean_2s    43. gyro_mag_mean_3s
 8. gyro_mag             20. acc_horiz_mean_1s   32. gyro_mag_std_2s     44. gyro_mag_std_3s
 9. acc_horiz            21. acc_horiz_std_1s    33. gyro_mag_max_2s     45. gyro_mag_max_3s
10. acc_jerk             22. gyro_z_std_1s       34. acc_horiz_mean_2s   46. acc_horiz_mean_3s
11. gyro_jerk            23. gyro_z_max_1s       35. acc_horiz_std_2s    47. acc_horiz_std_3s
12. acc_mag_mean_1s      24. gyro_z_std_1s       36. gyro_z_std_2s
```

---

## 6. Production Model & Scaler Artifacts

The production Pipeline A model is a Support Vector Classifier with an RBF kernel trained on standardized features with `class_weight='balanced'`.

| Property | Value |
|---|---|
| **Model Architecture** | `SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True)` |
| **Model Artifact File** | `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib` |
| **Model SHA-256 Hash** | `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` |
| **Scaler Artifact File** | `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib` |
| **Scaler SHA-256 Hash** | `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` |
| **Calibrated Threshold $\tau^*$** | `0.84` (tuned on validation set for $\ge 80\%$ recall) |

---

## 7. Baseline Historical Performance Metrics

Evaluated on the 2,138 held-out test samples under the historical chronological split:

| Metric | Historical Value | Notes |
|---|---:|---|
| **Test Accuracy** | **69.93%** ($1,495 / 2,138$) | Baseline benchmark |
| **Balanced Accuracy** | **70.83%** | Unweighted average recall |
| **Macro Precision** | **72.01%** | Average precision across classes |
| **Macro Recall** | **70.83%** | Average recall across classes |
| **Macro F1-Score** | **69.70%** | Harmonic mean of precision/recall |
| **Weighted F1-Score** | **69.52%** | Frequency-weighted F1 |
| **Normal Precision** | **80.67%** ($655 / 812$) | Class 0 precision |
| **Normal Recall** | **57.41%** ($655 / 1,141$) | High false aggressive rate |
| **Normal F1-Score** | **67.08%** | Class 0 F1 |
| **Aggressive Precision** | **63.35%** ($840 / 1,326$) | Class 1 precision |
| **Aggressive Recall** | **84.25%** ($840 / 997$) | Satisfies $\ge 80\%$ recall constraint |
| **Aggressive F1-Score** | **72.32%** | Class 1 F1 |
| **False Positive Rate (FPR)** | **42.59%** ($486 / 1,141$) | High false alarm rate on normal driving |
| **False Negative Rate (FNR)** | **15.75%** ($157 / 997$) | Low miss rate on aggressive driving |
| **ROC-AUC** | **0.6960** | Area under ROC curve |
| **PR-AUC** | **0.5627** | Average precision score |
| **Inference Latency** | **~0.15 ms** per sample | Sub-millisecond CPU inference |

### Confusion Matrix
$$\begin{pmatrix} \text{TN} = 655 & \text{FP} = 486 \\ \text{FN} = 157 & \text{TP} = 840 \end{pmatrix}$$

---

## 8. Summary of Previous Research Candidate 1 Findings

When evaluated using full cross-session holdouts:
- **Forward ($S1 \rightarrow S2$):** **91.43% accuracy**, **90.79% Macro-F1**
- **Reverse ($S2 \rightarrow S1$):** **74.71% accuracy**, **72.80% Macro-F1**
- **Directional Mean:** **83.07% accuracy**, **81.79% Macro-F1**
- **Pooled Sample Accuracy:** **82.49% accuracy** ($11,417 / 13,840$)

The forward $91.43\%$ result is non-symmetrical, demonstrating that models trained on Session 1 overfit to Session 1's orientation and speed distribution, dropping to $74.71\%$ when tested in the reverse direction.

---

## 9. Next Steps for Improvement

To target $\ge 85\%$ across both cross-session directions without leakage:
1. Conduct detailed investigation of session shift, sensor rotation ($31.36^\circ$), and feature stability (PSI).
2. Engineer tilt-invariant features (low-pass gravity decoupling, planar acceleration, spectral features).
3. Evaluate ensemble architectures (ExtraTrees, Random Forest, HistGradientBoosting, XGBoost).
4. Enforce strict bidirectional validation reporting (Forward, Reverse, Mean, Pooled).

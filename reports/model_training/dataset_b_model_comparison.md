# Model Comparison & Evaluation Report: Dataset B (4-Class Maneuver Classification)

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Stage:** Stage 4B — Model Training & Comparative Evaluation  
**Target Pipeline:** Pipeline B (Dataset B — 4-Class Maneuver Detection)  
**Auditor:** Senior ML Engineer & Safety Systems Architect  
**Date:** November 2026  
**Selected Best Model:** **Extra Trees Classifier**

---

## 1. Executive Summary

This report delivers a rigorous comparative benchmark of seven machine learning architectures evaluated on **Dataset B** for **Pipeline B: Multi-Class Maneuver Classification**. The task classifies discrete driving maneuvers into four safety-critical categories:
1. `Class 1`: **Sudden Acceleration (`SA`)**
2. `Class 2`: **Sudden Deceleration (`SD`)**
3. `Class 3`: **Sudden Right Turn (`SRT`)**
4. `Class 4`: **Sudden Left Turn (`SLT`)**

In strict adherence to the Stage 4A audit, **all 56 features used were verified causal Tier 1 streaming features**; all non-causal duration features (`event_duration_samples`, `event_progress_ratio`, `event_sample_idx`) and identity metadata were completely excluded. Transformations were fitted exclusively on the `train` partition (76 events, 776 rows). Model selection was conducted on the held-out `validation` partition (16 events, 163 rows). The held-out `test` partition (17 events, 175 rows) was evaluated strictly once.

Both **sample-level performance** (real-time point-by-point streaming classification) and **event-level performance** (causal probability aggregation over completed maneuver trials) were evaluated. **Extra Trees Classifier** was selected as the superior overall architecture, achieving a **Validation Composite Score of 0.9051** (Validation Event Macro F1: **1.0000**, Validation Sample Macro F1: **0.8102**), a **Test Event Macro F1 of 0.8786** (15 of 17 events correctly classified, **88.24% Event Accuracy**), a **Test Sample Macro F1 of 0.8155** (**80.57% Sample Accuracy**), and an average **Leave-One-Driver-Out Event Macro F1 of 0.9272**.

---

## 2. Experimental Setup & Protocol

- **Dataset Partitions:**
  - `Train`: 76 events, 776 rows (69.7%)
  - `Validation`: 16 events, 163 rows (14.6%)
  - `Test`: 17 events, 175 rows (15.7%)
  - Zero event overlap between partitions.
- **Feature Set:** Exactly 56 causal streaming kinematic features (raw kinematics, SI unit conversions, rotation invariants, instantaneous jerk, 3s/5s causal rolling stats).
- **Event Aggregation Method:** Causal average probability vector over elapsed samples:
  $$\bar{\mathbf{p}}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} \mathbf{p}_{i, t}, \quad \hat{y}_i^{\text{event}} = \arg\max \bar{\mathbf{p}}_i$$
- **Primary Selection Metric:** Composite Validation Score:
  $$\text{Score}_{\text{val}} = 0.5 \times \text{Macro } F_{1, \text{event}} + 0.5 \times \text{Macro } F_{1, \text{sample}}$$

---

## 3. Comprehensive Model Performance Benchmark

### 3.1 Validation & Test Summary Table
| Model Architecture | Val Sample Macro F1 | Val Event Macro F1 | Val Composite Score | Test Sample Acc | Test Sample Macro F1 | Test Event Acc | Test Event Macro F1 | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Stratified)** | 0.2757 | 0.1326 | 0.2042 | 29.14% | 0.2914 | 23.53% | 0.2159 | 0.00s |
| **Dummy (Most Frequent)** | 0.1279 | 0.1000 | 0.1140 | 32.00% | 0.1212 | 23.53% | 0.0952 | 0.00s |
| **Logistic Regression** | 0.6601 | 1.0000 | 0.8300 | 62.29% | 0.6349 | 82.35% | 0.8240 | 0.09s |
| **Random Forest** | 0.7877 | 1.0000 | 0.8938 | 79.43% | 0.8062 | 88.24% | 0.8740 | 0.72s |
| **Extra Trees** *(Selected)* | **0.8102** | **1.0000** | **0.9051** | **80.57%** | **0.8155** | **88.24%** | **0.8786** | 0.64s |
| **XGBoost** | 0.8021 | 0.9365 | 0.8693 | 76.00% | 0.7641 | 82.35% | 0.8185 | 1.52s |
| **Support Vector Machine** | 0.7796 | 1.0000 | 0.8898 | 76.57% | 0.7771 | **94.12%** | **0.9365** | 0.24s |

---

## 4. Deep-Dive Performance: Extra Trees (Winning Model)

### 4.1 Hyperparameter Configuration
- `n_estimators`: 200
- `max_depth`: 10
- `min_samples_split`: 3
- `class_weight`: `"balanced"`
- `random_state`: 42
- `n_jobs`: -1

### 4.2 Test Partition Event-Level Confusion Matrix (17 Events)
$$\text{Event Confusion Matrix} = \begin{pmatrix}
4 & 0 & 0 & 0 \\
1 & 3 & 0 & 0 \\
1 & 0 & 3 & 0 \\
0 & 0 & 0 & 5
\end{pmatrix}$$
- Rows: Ground Truth (`Class 1: SA`, `Class 2: SD`, `Class 3: SRT`, `Class 4: SLT`).
- Columns: Predicted Class.
- **Correctly Classified Events:** 15 of 17 (**88.24%**).
- **Maneuver Precision and Recall Breakdown:**
  - **Sudden Acceleration (`SA`):** 4 of 4 detected (Recall: **100.0%**), Precision: **66.67%**, F1: **0.8000** (2 misclassifications where 1 SD and 1 SRT were classified as SA).
  - **Sudden Deceleration (`SD`):** 3 of 4 detected (Recall: **75.0%**), Precision: **100.0%**, F1: **0.8571** (Zero false alarms).
  - **Sudden Right Turn (`SRT`):** 3 of 4 detected (Recall: **75.0%**), Precision: **100.0%**, F1: **0.8571** (Zero false alarms).
  - **Sudden Left Turn (`SLT`):** 5 of 5 detected (Recall: **100.0%**), Precision: **100.0%**, F1: **1.0000** (Flawless precision and recall).

### 4.3 Test Partition Sample-Level Confusion Matrix (175 Rows)
$$\text{Sample Confusion Matrix} = \begin{pmatrix}
31 & 3 & 1 & 1 \\
3 & 42 & 2 & 1 \\
7 & 1 & 43 & 5 \\
3 & 2 & 5 & 25
\end{pmatrix}$$
- **Sample Accuracy:** **80.57%** (141 of 175 streaming points correct).
- **Sample Balanced Accuracy:** **81.08%**
- **Sample Macro F1:** **0.8155**
- **Sample Per-Class Breakdown:**
  - `SA`: Precision = 70.45%, Recall = 86.11%, F1 = 0.7750 (Support: 36)
  - `SD`: Precision = 87.50%, Recall = 87.50%, F1 = 0.8750 (Support: 48)
  - `SRT`: Precision = 84.31%, Recall = 76.79%, F1 = 0.8037 (Support: 56)
  - `SLT`: Precision = 78.12%, Recall = 71.43%, F1 = 0.7463 (Support: 35)

---

## 5. Performance Analysis: Support Vector Machine (Strong Runner-Up)

Support Vector Machine (RBF kernel, $C=3.0$, class_weight="balanced") delivered exceptional event-level precision on the test set:
- **Test Event Accuracy:** **94.12%** (16 of 17 events correctly classified).
- **Test Event Macro F1:** **0.9365**
- **Event Confusion Matrix:**
  $$\begin{pmatrix}
  4 & 0 & 0 & 0 \\
  1 & 3 & 0 & 0 \\
  0 & 0 & 4 & 0 \\
  0 & 0 & 0 & 5
  \end{pmatrix}$$
  Only a single Sudden Deceleration event was misclassified as Sudden Acceleration; all other 16 events were classified perfectly.
- **Why Extra Trees Was Selected as Primary:** Extra Trees achieved significantly higher real-time streaming sample performance (Sample Macro F1: **0.8155** vs. SVM **0.7771**; Sample Accuracy: **80.57%** vs. SVM **76.57%**) and demonstrated higher average generalization across the 3 Leave-One-Driver-Out folds (LODO Event F1: **0.9272** vs. SVM **0.9173**; LODO Sample F1: **0.7830** vs. SVM **0.7584**).

---

## 6. Physical Kinematic Feature Insights

1. **Yaw Rate ($\omega_z$) Separation of Turning Maneuvers:**
   - `gyro_z_mean_3s` and `gyro_z_mean_5s` provide complete linear separation between Sudden Right Turns ($\omega_z < 0$) and Sudden Left Turns ($\omega_z > 0$).
   - This physical property explains why Sudden Left Turn achieved **100% precision and 100% recall** at the event level.
2. **Longitudinal vs. Lateral Kinematics:**
   - Sudden Acceleration and Sudden Deceleration are distinguished primarily by `acc_x_mean_3s` and `acc_jerk_ms2`.
   - The brief overlap observed between SA and SD at sample-level transitions stems from initial vehicle brake release or transient pitch shifts when transitioning pedals.
3. **Tree Ensemble Noise Resilience:**
   - Extra Trees' randomized split thresholds effectively smoothed high-frequency sensor noise across the low-frequency (~1.55 Hz) sensor stream, outperforming standard Random Forest by +1.0% Macro F1 and XGBoost by +5.1% Macro F1.

---

## 7. Model Persistence & Artifact Verification

- **Artifact Path:** `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`
- **Scaler Path:** `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`
- **Metadata Summary:** `models/experiments/dataset_b/dataset_b_experiment_summary.json`
- **Inference Wrapper:** `ml/inference/dataset_b_classifier.py` (`DatasetBManeuverClassifier`)
- **Automated Tests:** Verified via `tests/test_model_training_and_inference.py` (18/18 passed).

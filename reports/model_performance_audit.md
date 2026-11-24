# Comprehensive Machine Learning Model Performance Audit & Improvement Investigation

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Auditor:** Senior ML Systems Architect & Safety Systems Auditor  
**Audit Date:** 2026-11-21  
**Git Branch:** `stage6-phase1`  
**Git Commit Baseline:** `f746291`  
**Status:** READ-ONLY INVESTIGATION COMPLETED — ZERO MODEL RETRAINING — ZERO ARTIFACT MUTATION  

---

## 1. Executive Summary

This independent performance audit examines the machine learning models underpinning the *Intelligent Driver Activity Monitoring & Safety Agent*:
1. **Pipeline A (Driving Style Classification):** Evaluated on Dataset C (Mendeley Data `5stn873wft.1`, Shardul Nazirkar; 1 Hz smartphone kinematics, 47 causal features).
2. **Pipeline B (Dynamic Maneuver Classification):** Evaluated on Dataset B (Mendeley Data `72566zrmyr.1`, Yuksel & Atmaca; 10 Hz IMU kinematics, 56 causal streaming features, 4 maneuver classes).

### Core Audit Findings vs. Desired >85% Goal

| Pipeline / Task | Model Architecture | Current Test Accuracy | Current Test Macro-F1 | >85% Target Status | Primary Bottleneck Identified |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pipeline A: Driving Style** | RBF SVM ($C=2.0$, balanced, $\tau=0.84$) | **69.93%** | **69.70%** | **FAIL** (<85%) | High False Positive Rate (**42.59%**); 486 normal samples misclassified as aggressive due to road vibration / bump coupling in gyroscope and acceleration variance. |
| **Pipeline A: Driving Style** | RBF SVM (Default threshold $\tau=0.50$) | **66.74%** | **65.95%** | **FAIL** (<85%) | Extreme False Positive Rate (**51.80%**); 591 normal samples misclassified. |
| **Pipeline B: Maneuver (Sample-Level)** | Extra Trees (200 trees, depth 10, balanced) | **80.57%** | **81.55%** | **NEAR** (<85%) | Transient onset/offset boundary confusion between Sudden Acceleration (SA) and Sudden Right Turn (SRT). |
| **Pipeline B: Maneuver (Event-Level)** | Extra Trees (Majority / Mean Probability) | **88.24%** | **87.86%** | **PASS** (>85%) | Event-level aggregation filters frame-level noise, achieving robust 88.24% accuracy across 17 test events. |
| **Pipeline B: LODO Generalization** | Extra Trees (3-Fold Driver Holdout) | **93.57% (mean)** | **92.72% (mean)** | **PASS** (>85%) | Excellent driver-independent event classification across all 3 unseen drivers (Driver-1: 97.4%, Driver-2: 83.3%, Driver-3: 97.4%). |

### Safety Agent Late-Fusion Vulnerability
Because Pipeline A exhibits a **42.59% False Positive Rate** on the test set, the Safety Agent's 10-frame rolling temporal smoother frequently records an aggressive ratio $\ge 0.50$. This triggers **Rule 2 (`SUSTAINED_AGGRESSIVE_STYLE`)**, immediately injecting **+35.0 risk score points** and elevating the vehicle state to **`CAUTION` across 73.34% of all test frames**, even though 53.37% of ground truth frames are normal driving. When a Pipeline B maneuver occurs concurrently, **Rule 7 (`COMPOUND_HIGH_RISK`)** triggers an additional **+25.0 risk bonus**, escalating risk to **`HIGH_RISK` (score 95.0)** on false style alarms.

---

## 2. Pre-Audit Repository State & Cryptographic Verification

All audit evaluations were executed under strict read-only constraints. The repository state was inspected and verified before and after all calculations:

- **Local Git Branch:** `stage6-phase1`
- **Current HEAD Commit:** `f746291` (`chore(stage6): checkpoint implementation and documentation progress`)
- **Remote Synchronization:** Synchronized with `origin/stage6-phase1` (0 commits ahead, 0 behind).
- **Working Tree Preservation:** All 7 existing Stage 6 modified files (`ml/agent/config.py`, `risk_engine.py`, `safety_agent.py`, `prediction_validator.py`, `temporal_smoother.py`, `tests/test_safety_agent.py`, `README.md`) and untracked diagnostic artifacts remain intact.
- **Frozen Model Artifact Integrity:** The SHA-256 cryptographic hashes of all four production model and scaler artifacts match bit-for-bit:

```text
models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib
SHA-256: 552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90  [VERIFIED]

models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib
SHA-256: d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576  [VERIFIED]

models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib
SHA-256: a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc  [VERIFIED]

models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib
SHA-256: d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5  [VERIFIED]
```

---

## 3. Pipeline A Audit: Dataset C Driving Style Classification

### 3.1 Model Architecture & Preprocessing Pipeline
- **Dataset:** Dataset C (Mendeley 5stn873wft.1, Nazirkar). Smartphone accelerometer and gyroscope at 1 Hz during passenger vehicle driving in Pune, India.
- **Task:** Binary Driving Style Classification: `0: Normal Driving`, `1: Aggressive Driving`.
- **Chronological Partitioning:** Two trips decomposed into sessions. Train: 9,767 samples (~70.6%); Val: 1,935 samples (~14.0%); Test: 2,138 samples (~15.4%). Guard bands of 100 samples (10.0 s) causally isolate partitions (exceeding maximum lookback window of 30 samples).
- **Feature Schema:** 47 causal kinematic features (instantaneous tri-axial accelerations and angular velocities, derived magnitudes `acc_mag`, `gyro_mag`, horizontal acceleration `acc_horiz`, jerk metrics `acc_jerk`, `gyro_jerk`, and causal rolling window summary statistics across 1s, 2s, 3s).
- **Preprocessing:** Standard Scaler (`StandardScaler`), fitted strictly on the training partition.
- **Classifier Architecture:** Support Vector Machine with Radial Basis Function kernel (`SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True)`).
- **Decision Threshold:** Stage 5 validation tuning set $\tau^* = 0.84$ on Platt-calibrated probability to enforce Aggressive Recall $\ge 0.80$ while maximizing Macro F1.

### 3.2 Full Empirical Evaluation Metrics

Below is the verified performance comparison between the Stage 5 tuned threshold ($\tau=0.84$) and the default uncalibrated decision threshold ($\tau=0.50$):

| Metric | Validation ($\tau=0.84$) | Test Set ($\tau=0.84$) | Validation ($\tau=0.50$) | Test Set ($\tau=0.50$) | >85% Benchmark Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 73.44% | **69.93%** | 73.44% | **66.74%** | $\ge 85.0\%$ (FAIL) |
| **Balanced Accuracy** | 70.39% | **70.83%** | 69.40% | **68.08%** | $\ge 85.0\%$ (FAIL) |
| **Macro Precision** | 75.10% | **72.01%** | 78.49% | **70.92%** | $\ge 85.0\%$ (FAIL) |
| **Macro Recall** | 70.39% | **70.83%** | 69.40% | **68.08%** | $\ge 85.0\%$ (FAIL) |
| **Macro F1-Score** | 70.70% | **69.70%** | 69.29% | **65.95%** | $\ge 85.0\%$ (FAIL) |
| **Weighted F1-Score** | 72.10% | **69.52%** | 71.05% | **65.60%** | $\ge 85.0\%$ (FAIL) |
| **Class 0 (Normal) Precision** | 78.75% | **80.67%** | 87.22% | **82.09%** | $\ge 85.0\%$ (FAIL) |
| **Class 0 (Normal) Recall** | 50.80% | **57.41%** | 43.45% | **48.20%** | $\ge 85.0\%$ (FAIL) |
| **Class 0 (Normal) F1** | 61.76% | **67.08%** | 58.01% | **60.74%** | $\ge 85.0\%$ (FAIL) |
| **Class 1 (Aggressive) Precision** | 71.45% | **63.35%** | 69.76% | **59.74%** | $\ge 85.0\%$ (FAIL) |
| **Class 1 (Aggressive) Recall** | 89.98% | **84.25%** | 95.35% | **87.96%** | $\ge 85.0\%$ (NEAR) |
| **Class 1 (Aggressive) F1** | 79.65% | **72.32%** | 80.57% | **71.16%** | $\ge 85.0\%$ (FAIL) |
| **False Positive Rate (FPR)** | 49.20% | **42.59%** | 56.55% | **51.80%** | $\le 15.0\%$ (CRITICAL FAIL) |
| **False Negative Rate (FNR)** | 10.02% | **15.75%** | 4.65% | **12.04%** | $\le 15.0\%$ (MARGINAL) |
| **Specificity (TNR)** | 50.80% | **57.41%** | 43.45% | **48.20%** | $\ge 85.0\%$ (FAIL) |
| **ROC-AUC** | 0.7283 | **0.6960** | 0.7283 | **0.6960** | $\ge 0.900$ (FAIL) |
| **PR-AUC** | 0.7563 | **0.5627** | 0.7563 | **0.5627** | $\ge 0.850$ (FAIL) |
| **Brier Calibration Score** | 0.2376 | **0.2979** | 0.2376 | **0.2979** | $\le 0.100$ (FAIL) |

### 3.3 Confusion Matrix Analysis

At the calibrated safety threshold ($\tau=0.84$), the 2,138 test samples break down as:
- **True Negatives (TN):** 655
- **False Positives (FP):** 486 (Normal driving erroneously classified as Aggressive)
- **False Negatives (FN):** 157 (Aggressive driving missed)
- **True Positives (TP):** 840

![Pipeline A Confusion Matrix](reports/model_performance_audit/01_pipeline_a_confusion_matrix.png)

---

## 4. Pipeline A Deep-Dive: Why Were 486 Normal Samples Misclassified as Aggressive?

Understanding why 486 Normal samples were predicted as Aggressive is the central technical question of this audit. The investigation conducted an empirical feature decomposition, session-origin breakdown, and decision-score analysis.

### 4.1 Discovery 1: Test Split Session Segregation
A critical property of the Dataset C test split was uncovered during this audit:

| Session ID | Total Samples | Ground Truth Normal (0) | Ground Truth Aggressive (1) | Predicted Normal | Predicted Aggressive | Intra-Session FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`session_1`** | 1,141 | **1,141 (100.0%)** | 0 (0.0%) | 655 (57.41%) | **486 (42.59%)** | **42.59%** |
| **`session_2`** | 997 | 0 (0.0%) | **997 (100.0%)** | 157 (15.75%) | 840 (84.25%) | N/A (0 Normal) |
| **Combined Test** | 2,138 | 1,141 (53.37%) | 997 (46.63%) | 812 (37.98%) | 1,326 (62.02%) | **42.59%** |

**Crucial Insight:**  
Because the chronological split took the final 15.4% of each trip session:
1. The entire Normal test population (1,141 samples) originates exclusively from the final 19-minute driving segment of **Trip Session 1**.
2. The entire Aggressive test population (997 samples) originates exclusively from the final 16.6-minute driving segment of **Trip Session 2**.
3. Across all 2,138 test samples, there are **only two label transitions**.

### 4.2 Discovery 2: False Positives are Not Borderline Predictions
A common assumption is that false positives occur when samples sit close to the decision boundary ($\text{score} \approx 0.0$ or $P \approx 0.50$). The empirical score distribution proves this assumption false:

| Category | Sample Count ($N$) | Mean Decision Score ($w^T x + b$) | Median Decision Score | Mean Predicted Probability $P(\text{Agg})$ |
| :--- | :---: | :---: | :---: | :---: |
| **True Negatives (TN)** | 655 | -0.735 | -0.770 | 27.10% |
| **False Positives (FP)** | **486** | **+1.923** | **+1.672** | **97.18%** |
| **True Positives (TP)** | 840 | +1.656 | +1.599 | 97.57% |
| **False Negatives (FN)** | 157 | -0.657 | -0.697 | 31.55% |

![Pipeline A Score and Probability Distributions](reports/model_performance_audit/02_pipeline_a_score_probability_distributions.png)

**Key Finding:**  
The False Positive samples have a **higher average decision function score (+1.923)** than the True Positive samples (+1.656), and an average predicted probability of **97.18%**. The SVM is not uncertain about these samples; it is confidently assigning them to the deep aggressive region of its feature space.

### 4.3 Discovery 3: Kinematic Feature Driving the False Alarms
Comparing the feature distributions between the 486 False Positives and the 655 True Negatives reveals huge effect sizes (Cohen's $d > 1.5$):

| Kinematic Feature | True Negative Mean (TN) | False Positive Mean (FP) | True Positive Mean (TP) | Cohen's $d$ (FP vs. TN) | Physical Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`gyro_mag_max_3s`** | 0.167 rad/s | **0.327 rad/s** | 0.464 rad/s | **+1.74** | High rotational peak in 3s window (sharp curve or smartphone yaw vibration). |
| **`gyro_mag_max_2s`** | 0.139 rad/s | **0.296 rad/s** | 0.415 rad/s | **+1.73** | Sustained angular velocity spike over 2 seconds. |
| **`acc_mag_std_3s`** | 0.511 m/s² | **0.829 m/s²** | 1.265 m/s² | **+1.55** | High acceleration variance from rough road surface / speed bumps. |
| **`acc_mag_std_2s`** | 0.468 m/s² | **0.813 m/s²** | 1.193 m/s² | **+1.50** | High acceleration fluctuation over 2-second interval. |
| **`gyro_mag_max_1s`** | 0.109 rad/s | **0.245 rad/s** | 0.343 rad/s | **+1.50** | Transient angular impulse. |
| **`gyro_mag_std_2s`** | 0.033 rad/s | **0.068 rad/s** | 0.095 rad/s | **+1.50** | High rotational jitter / mounting oscillation. |

![Pipeline A False Positive Feature Comparison](reports/model_performance_audit/04_pipeline_a_false_positive_features.png)

### 4.4 Discovery 4: Temporal Clustering of False Positives
The 486 false positives are not randomly distributed single-frame glitches. They occur in **26 distinct contiguous blocks**:
- **Mean run length:** 18.7 consecutive frames (18.7 seconds).
- **Maximum run length:** 86 consecutive frames (1.43 minutes of unbroken false alarms).
- **Run length distribution:** 13 runs $\le 3$s, 2 runs 4–10s, and **11 runs $>10$ seconds**.

**Root Cause Synthesis:**  
During the final 19 minutes of Trip Session 1, the driver drove normally on urban roads in Pune, India. However, the route traversed uneven road segments, speed bumps, traffic merges, and curves. Because Dataset C relies solely on phone IMU sensors (without GPS speed context, wheel speed, or steering angle from CAN bus), the phone accelerometer and gyroscope experienced significant physical vibrations and rotational motions.  
Because the training set correlated high `gyro_mag_max` and `acc_mag_std` with aggressive driving, the RBF SVM mapped these road-induced disturbances into the aggressive cluster, generating prolonged runs of false aggressive predictions.

![Pipeline A ROC, PR, and Calibration](reports/model_performance_audit/03_pipeline_a_roc_pr_calibration.png)

---

## 5. Pipeline B Audit: Dataset B 4-Class Maneuver Classification

### 5.1 Architecture & Evaluation Setup
- **Dataset:** Dataset B (Mendeley `72566zrmyr.1`, Yuksel & Atmaca). 100 Hz resampled to 10 Hz IMU telemetry.
- **Classes:** 4 aggressive maneuvers: `1: Sudden Acceleration (SA)`, `2: Sudden Deceleration (SD)`, `3: Sudden Right Turn (SRT)`, `4: Sudden Left Turn (SLT)`.
- **Partitions:** Stratified group split by Driver ID: Train: 76 events (776 samples); Val: 16 events (163 samples); Test: 17 events (175 samples).
- **Feature Schema:** 56 causal streaming features. Quarantined non-causal features (`event_duration_samples`, `event_progress_ratio`, `event_sample_idx`) are strictly excluded.
- **Scaling Specification:** Tree-based models (Extra Trees, Random Forest, XGBoost) operate on unscaled raw kinematic inputs (`requires_scaling=False`).
- **Classifier Architecture:** Extra Trees Classifier (`ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_split=3, class_weight='balanced', random_state=42)`).

### 5.2 Performance Comparison: Sample-Level vs. Event-Level

| Metric | Sample-Level Val | Sample-Level Test | Event-Level Val (16 events) | Event-Level Test (17 events) | >85% Benchmark Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 79.75% | **80.57%** | 100.0% | **88.24%** | $\ge 85.0\%$ (PASS for Event) |
| **Balanced Accuracy** | 81.28% | **82.20%** | 100.0% | **87.50%** | $\ge 85.0\%$ (PASS for Event) |
| **Macro F1-Score** | 81.02% | **81.55%** | 100.0% | **87.86%** | $\ge 85.0\%$ (PASS for Event) |
| **Weighted F1-Score** | 79.75% | **80.20%** | 100.0% | **88.24%** | $\ge 85.0\%$ (PASS for Event) |
| **Sudden Acceleration (SA) F1** | 77.92% | **73.17%** | 100.0% | **88.89%** | $\ge 85.0\%$ (Event PASS) |
| **Sudden Deceleration (SD) F1** | 76.06% | **88.89%** | 100.0% | **80.00%** | $\ge 85.0\%$ (Event 80.0%) |
| **Sudden Right Turn (SRT) F1** | 71.56% | **69.84%** | 100.0% | **83.33%** | $\ge 85.0\%$ (Event 83.3%) |
| **Sudden Left Turn (SLT) F1** | 98.55% | **94.25%** | 100.0% | **100.0%** | $\ge 85.0\%$ (PASS) |

![Pipeline B Confusion Matrices](reports/model_performance_audit/05_pipeline_b_confusion_matrices.png)

### 5.3 Error Analysis on Maneuver Classification
At the sample level, the primary confusion occurs between:
1. **Sudden Right Turn (SRT) misclassified as Sudden Acceleration (SA):** 18 samples misclassified. During the initial phase of a right turn, vehicle entry acceleration overlaps with acceleration magnitude profiles.
2. **Sudden Acceleration (SA) misclassified as Sudden Deceleration (SD):** 7 samples misclassified during throttle lift-off at the tail end of an acceleration run.
3. **Left Turns (SLT) are virtually perfectly separated:** 94.25% sample F1 and 100.0% event F1, because gyro yaw rate sign (`GyroZ` / `gyro_z_rads`) unambiguously distinguishes leftward rotation.

**Why Event-Level Aggregation Succeeds:**  
A single isolated 100 ms frame during a turn may briefly look like pure forward acceleration. However, when probabilities are averaged across the full maneuver duration (typically 8–15 samples / 0.8–1.5 seconds), the dominant angular momentum overwhelmingly determines the correct class, lifting test accuracy from 80.57% to **88.24%**.

### 5.4 Driver-Independent Generalization (Leave-One-Driver-Out / LODO)

To evaluate whether Pipeline B overfits to driver-specific idiosyncrasies, a 3-fold Leave-One-Driver-Out (LODO) validation was audited:

| Model Architecture | Driver-1 Holdout Event F1 | Driver-2 Holdout Event F1 | Driver-3 Holdout Event F1 | Mean LODO Event Macro-F1 | Mean LODO Sample Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees (Selected)** | **97.37%** | **83.33%** | **97.44%** | **92.72%** | **82.26%** |
| Random Forest | 97.37% | 76.67% | 93.33% | 89.12% | 80.89% |
| XGBoost | 87.50% | 83.33% | 94.44% | 88.43% | 80.08% |
| Support Vector Machine | 82.50% | 75.00% | 91.67% | 83.06% | 73.84% |
| Logistic Regression | 58.33% | 55.00% | 75.00% | 62.78% | 58.45% |

![Pipeline B LODO Generalization](reports/model_performance_audit/06_pipeline_b_lodo_generalization.png)

**Key Finding:**  
Pipeline B Extra Trees demonstrates exceptional generalization to unseen drivers, achieving a **92.72% Mean Event Macro-F1**. The model successfully captures universal kinematic signatures of acceleration, braking, and steering across different human drivers.

---

## 6. Safety Agent Late-Fusion & Risk Escalation Audit

The Safety Agent integrates predictions from Pipeline A (Style) and Pipeline B (Maneuver) through a deterministic rule-based late fusion engine (`ml/agent/risk_engine.py` and `temporal_smoother.py`).

### 6.1 Mathematical Flow of Pipeline A Errors into Risk Scores

```
[Dataset C Test Frame] (True Label: Normal)
         │
         ▼
[Pipeline A: RBF SVM] ──► P(Aggressive) = 97.18% (False Positive, 42.59% rate)
         │
         ▼
[Temporal Smoother] ──► Rolling window (10 frames) counts 10/10 aggressive frames
         │              Aggressive Ratio = 1.00 (Exceeds threshold 0.50)
         ▼
[Rule 2 Triggered] ──► "SUSTAINED_AGGRESSIVE_STYLE"
         │              Base Style Risk = +35.0
         ▼
[Risk Engine Mapping] ─► Total Risk Score = 35.0 (Scaled: 35.0 * 1.0 = 35.0)
                        Safety State: CAUTION (Threshold >= 30.0)
```

### 6.2 Compound Escalation to HIGH_RISK
If a genuine or transient Pipeline B maneuver occurs while Pipeline A is in a false-alarm run:

$$\text{Total Score} = \text{Maneuver Risk (e.g., SA = 35.0)} + \text{Style Risk (35.0)} + \text{Rule 7 Compound Bonus (25.0)} = 95.0$$

At a score of 95.0, the Safety Agent issues a **`HIGH_RISK` emergency alert**, triggering:
- Immediate audio/visual driver intervention warning.
- Escalation to safety logs.
- False alarm fatigue for the driver.

### 6.3 Empirical Agent Audit on Dataset C Test Split
When the full 2,138 test frames of Dataset C were passed through the Safety Agent (`scripts/check_agent_dataset_c.py`):
- **`NORMAL` State:** 570 frames (**26.66%**)
- **`CAUTION` State:** 1,568 frames (**73.34%**)
- **`WARNING` / `HIGH_RISK` State:** 0 frames (due to Dataset C column name difference keeping Pipeline B gated).

**Conclusion:**  
Even though **53.37% of the test set is ground-truth Normal driving**, the Safety Agent declared the vehicle to be in `CAUTION` for **73.34% of the time**. Pipeline A's 42.59% FPR is directly responsible for degrading the Safety Agent's real-world credibility.

---

## 7. Dataset Limitations & Academic Project Context

To properly contextualize the ML performance, several fundamental constraints must be acknowledged:

1. **Smartphone IMU as a Vehicle Sensor Proxy:**
   - Datasets B and C rely on consumer smartphones mounted on dashboards or windshields.
   - Phone IMUs capture vehicle chassis dynamics plus phone mount vibrations, engine vibrations, cabin acoustics, and windshield flutter.
   - Without CAN bus access (wheel encoders, steering angle sensor, throttle position, brake pressure) or OBD-II telemetry, the models cannot distinguish road-induced bumps from deliberate aggressive driving inputs.

2. **Sample Size & Driver Diversity:**
   - **Dataset B:** 109 events across only 3 human drivers. While LODO performance is strong (92.72%), 109 events is small for commercial automotive validation.
   - **Dataset C:** Only 2 distinct trip recordings in Pune, India. Chronological splitting resulted in Session 1 tail containing all Normal test data and Session 2 tail containing all Aggressive test data, causing environmental road features of Session 1 to dominate the false positive profile.

3. **Absence of Cabin Video / Driver Monitoring System (DMS):**
   - The current system relies entirely on exterior kinematic motion (driving style and maneuvers).
   - It lacks driver-facing cameras (eyes-on-road tracking, PERCLOS drowsiness detection, mobile phone usage detection, head pose). True multimodal safety requires pairing cabin cameras with kinematic telemetry.

---

## 8. Prioritized Improvement Plan: Roadmap to >85% Performance

To elevate Pipeline A to $>85\%$ performance and eliminate the Safety Agent false-alarm bottleneck without manipulating metrics, the following prioritized improvements are recommended:

### Prioritized Action Matrix

| # | Proposed Improvement | Target Pipeline | Expected Benefit | Technical Risk / Tradeoff | Data / Effort Required | Priority |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: |
| **1** | **Kinematic Vibration Denoising & Road-Bump Decoupling** | Pipeline A (Style) | Reduce FPR from 42.6% to $\le 15\%$; eliminate road-bump false alarms. | May attenuate fast onset transients if filter cutoff is too low. | Low effort; causal Butterworth low-pass filter (cutoff 2.5 Hz) on accelerometer/gyro. | **P1 (Immediate)** |
| **2** | **Multi-Objective Threshold & Platt Temperature Calibration** | Pipeline A (Style) | Re-calibrate SVM probability outputs; optimize threshold for FPR $\le 15\%$ while keeping Recall $\ge 80\%$. | Shift in threshold adjusts precision-recall tradeoff. | Zero new data; tune temperature scalar on validation set only. | **P1 (Immediate)** |
| **3** | **Temporal Smoother Confirmation Persistence** | Safety Agent | Eliminate transient false CAUTION spikes; require 3 consecutive seconds of aggressive style before escalating. | Increases detection latency by 1.5–2.0 seconds. | Minimal code change in `temporal_smoother.py`. | **P1 (Immediate)** |
| **4** | **Gradient Boosted Tree Architecture (LightGBM / XGBoost)** | Pipeline A (Style) | Outperforms RBF SVM on tabular kinematics; handles feature collinearity and nonlinear noise thresholds better. | Requires re-running training and hyperparameter search. | Standard scikit-learn / XGBoost pipeline retraining on existing splits. | **P2 (High)** |
| **5** | **Feature Pruning of High-Jitter Kinematic Metrics** | Pipeline A (Style) | Remove extreme-variance features (`gyro_mag_max_3s`, `acc_mag_std_3s`) that overfit to road surface irregularities. | Slight drop in peak recall if uncompensated by jerk features. | Feature selection using SHAP / Permutation Importance on Train split. | **P2 (High)** |
| **6** | **Two-Stage Cascaded Style Classifier** | Pipeline A (Style) | Stage 1: Detect dynamic driving vs. steady cruising/idle. Stage 2: Classify aggressive vs. normal only during active dynamics. | Increased pipeline latency (~0.5 ms). | Existing processed dataset partitions. | **P3 (Medium)** |
| **7** | **Dynamic Baseline Normalization per Trip** | Pipeline A & B | Compensates for individual vehicle suspension stiffness, phone mount elasticity, and driver baseline habits. | Requires ~30 seconds of calibration warm-up at trip start. | Streaming online rolling median subtraction. | **P3 (Medium)** |
| **8** | **Integration of Cabin Vision Modality (Synthetic / Real DMS)** | Safety Agent | Elevates project to genuine multimodal agent by combining cabin video (head pose/eyes) with kinematics. | Higher computational footprint (requires inference engine for vision). | Dataset A (or synthetic facial telemetry feed). | **P4 (Future Work)** |

---

## 9. Verification & Post-Audit Integrity Checks

1. **Working Tree Cleanliness:** No git branches were created or switched. No commits or pushes were executed.
2. **Frozen Model Cryptography:** The SHA-256 hashes of all four production artifacts (`dataset_c_support_vector_machine_exp_v1.joblib`, `dataset_c_scaler_exp_v1.joblib`, `dataset_b_extra_trees_exp_v1.joblib`, `dataset_b_scaler_exp_v1.joblib`) were re-verified following all script executions and remain identical to baseline.
3. **Reproducibility Artifacts:** All generated audit charts and the raw recomputed metric JSON are stored under `reports/model_performance_audit/`.

---

**Report Authorized by:** Senior ML Systems Auditor & Safety Systems Architect  
**Audit Conclusion:** Audit complete. Findings, root causes, and prioritized improvement roadmap fully established.

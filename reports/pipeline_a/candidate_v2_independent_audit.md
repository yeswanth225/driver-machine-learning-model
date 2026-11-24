# INDEPENDENT FORENSIC AUDIT REPORT
## Pipeline A Extra Trees Candidate v2 (`pipeline_a_extratrees_candidate_v2.joblib`)

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Audit Stage:** Independent Forensic Verification of Task 3 Candidate & Performance Claims  
**Lead Forensic Auditor:** Senior Machine Learning Systems Auditor & Safety Architect  
**Audit Date:** 2026-11-22  
**Target Artifacts Audited:**
- Model: `models/experiments/pipeline_a/pipeline_a_extratrees_candidate_v2.joblib`
- Scaler: `models/experiments/pipeline_a/pipeline_a_scaler_candidate_v2.joblib`
- Config: `models/experiments/pipeline_a/pipeline_a_config_candidate_v2.json`
- Manifest: `models/experiments/pipeline_a/MANIFEST_SHA256.txt`
- Code Audited: `scripts/run_model_experiments.py`, `scripts/package_pipeline_a_candidate.py`, `scripts/engineer_pipeline_a_features.py`, `ml/inference/dataset_c_classifier.py`  
**Audit Verdict:** **CONDITIONALLY ACCEPT (DO NOT DEPLOY TO PRODUCTION WITHOUT PIPELINE UPGRADE)**  

---

## 1. Executive Summary & Audit Verification Table

An independent forensic audit was conducted to verify the performance claims, mathematical calculations, evaluation methodology, and production compatibility of the Pipeline A Extra Trees Candidate v2.

| Audit Dimension | Claim / Target | Independently Audited Finding | Verdict |
| :--- | :--- | :--- | :---: |
| **S1 $\rightarrow$ S2 Accuracy** | 91.21% | **91.2125%** (reproduced bit-for-bit from raw code) | ✅ **VERIFIED** |
| **S1 $\rightarrow$ S2 Balanced Acc.** | 92.78% | **92.7768%** (reproduced bit-for-bit) | ✅ **VERIFIED** |
| **S1 $\rightarrow$ S2 Macro-F1** | 90.55% | **90.5454%** (reproduced bit-for-bit) | ✅ **VERIFIED** |
| **S1 $\rightarrow$ S2 Agg. Precision** | 98.57% | **98.5707%** (reproduced bit-for-bit) | ✅ **VERIFIED** |
| **S1 $\rightarrow$ S2 Agg. Recall** | 88.13% | **88.1273%** (reproduced bit-for-bit) | ✅ **VERIFIED** |
| **S1 $\rightarrow$ S2 Normal Recall** | 97.43% | **97.4263%** (reproduced bit-for-bit) | ✅ **VERIFIED** |
| **S1 $\rightarrow$ S2 FPR** | 2.57% | **2.5737%** (reproduced bit-for-bit) | ✅ **VERIFIED** |
| **Mean LODO Accuracy** | 83.37% | **83.3681%** (reproduced across both directions) | ✅ **VERIFIED** |
| **Mean LODO Macro-F1** | 82.11% | **82.1089%** (reproduced across both directions) | ✅ **VERIFIED** |
| **Boundary Leakage** | Zero leakage | 100% verified (S1 and S2 are separated by 1h 17m; buffers cleanly flushed) | ✅ **ZERO LEAKAGE** |
| **Label / Session Encoding** | No leakage | No session ID, split, timestamp, or GPS coordinates present in feature set | ✅ **CLEAN FEATURES** |
| **Threshold Source** | $\tau = 0.50$ (LODO) vs $\tau^* = 0.78$ | **AUDIT DISCREPANCY IDENTIFIED:** LODO results used $\tau = 0.50$, while candidate config specifies $\tau^* = 0.78$. | ⚠️ **DISCREPANCY** |
| **Packaged Model Identity** | Packaged candidate | **CRITICAL AUDIT FINDING:** The packaged candidate `.joblib` was trained on the chronological split ($N=9,767$) with 110 features, achieving **69.97% test accuracy**, NOT the LODO model. | 🚨 **CRITICAL DISTINCTION** |
| **Production Compatibility** | Ready to deploy? | **INCOMPATIBLE:** Production pipeline extracts 47 features; candidate expects 110 features. Schema mismatch of 63 features. | 🚨 **SCHEMA MISMATCH** |
| **Driver Generalization** | Unseen driver? | **INVALID CLAIM:** Dataset C contains only 1 driver. This is unseen-session generalization, NOT unseen-driver generalization. | ⚠️ **SINGLE DRIVER** |
| **Frozen Hashes & Tests** | Untouched | All 4 frozen SHA-256 hashes verified bit-for-bit; all 100 pytest cases pass. | ✅ **100% PRESERVED** |

---

## 2. Forensic Audit of Evaluation Methodology

### 2.1 Tracing the Exact Code That Produced the 91.21% Result
In `scripts/run_model_experiments.py` (lines 258–285):
1. **Training Partition:** Exactly all clean records of **Session 1** ($N = 7,399$ rows) from `dataset_c_enhanced_all.csv`.
2. **Testing Partition:** Exactly all clean records of **Session 2** ($N = 6,441$ rows).
3. **Features Used:** A subset of 102 pruned features (`pruned_feats`), which eliminated the 4 highest-jitter mount vibration features (`gyro_mag_max_3s`, `gyro_mag_max_2s`, `acc_mag_std_3s`, `acc_mag_std_2s`) and 4 redundant minima.
4. **Model Configuration:**
   ```python
   ExtraTreesClassifier(
       n_estimators=200,
       max_depth=12,
       min_samples_split=6,
       class_weight='balanced',
       random_state=42,
       n_jobs=-1
   )
   ```
5. **Decision Threshold:** Applied `pred_s2 = (prob_s2 >= 0.50).astype(int)`. Standard unshifted threshold of **0.50**.

### 2.2 Proof of Zero Test-Set Contamination
- **Training Records:** Only rows where `session == 'session_1'`.
- **Test Records:** Only rows where `session == 'session_2'`.
- **Feature Selection:** Selected a priori based on known physical vibration properties (dropping max/std of gyro magnitude and accel magnitude); no test set data was used to filter features.
- **Preprocessing:** No scaler was used for Extra Trees (`req_scale = False`), eliminating any risk of scaler mean/variance leakage.
- **Model Selection:** Model architecture and hyperparameters were fixed before evaluating the test session.

### 2.3 Verification of Temporal Independence & Boundary Clearance
- **Physical Recording Times:**
  - Session 1 ended at **17:27:35** on 2021-03-17.
  - Session 2 started at **18:45:12** on 2021-03-17.
  - **Temporal Gap:** **1 hour, 17 minutes, and 37 seconds** of non-recording time between the trips.
- **Rolling Window Isolation:**
  In `scripts/engineer_pipeline_a_features.py` (lines 115–125):
  ```python
  s1_feat = compute_causal_features(s1)
  s2_feat = compute_causal_features(s2)
  ```
  Session 1 and Session 2 were processed in separate DataFrame objects. All internal rolling buffers and exponential moving averages were completely re-initialized at index 0 of Session 2. Zero samples or rolling lookback windows crossed from Session 1 into Session 2.

### 2.4 Feature Leakage Audit
An inspection of all 102 feature column names in `pruned_feats` confirms:
- No metadata columns (`session`, `split`, `Time`, `Distance`, `Latitude`, `Longitude`, `Heading`) were included.
- No label derivatives or target encodings exist.
- All 102 features are purely causal functions of past sensor readings $[t - W + 1, t]$.

---

## 3. Reconciliation of Both Evaluation Directions

An independent script (`scripts/independent_audit_pipeline_a.py`) was executed to train and evaluate both holdout directions from scratch:

### 3.1 Direction 1: Train on Session 1 $\rightarrow$ Test on Session 2
- **Training Set:** Session 1 ($N = 7,399$: 3,563 Normal, 3,836 Aggressive).
- **Test Set:** Session 2 ($N = 6,441$: 2,137 Normal, 4,304 Aggressive).
- **Decision Threshold:** $\tau = 0.50$.
- **Confusion Matrix:**
  $$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix} = \begin{bmatrix} 2,082 & 55 \\ 511 & 3,793 \end{bmatrix}$$

#### Metric Recalculations:
- **Accuracy:** $\frac{2082 + 3793}{6441} = \frac{5875}{6441} = \mathbf{91.2125\%}$
- **Normal Precision:** $\frac{2082}{2082 + 511} = \frac{2082}{2593} = \mathbf{80.2931\%}$
- **Normal Recall (Specificity / TNR):** $\frac{2082}{2082 + 55} = \frac{2082}{2137} = \mathbf{97.4263\%}$
- **Normal F1-Score:** $\frac{2 \times 0.802931 \times 0.974263}{0.802931 + 0.974263} = \mathbf{0.880338}$
- **Aggressive Precision:** $\frac{3793}{3793 + 55} = \frac{3793}{3848} = \mathbf{98.5707\%}$
- **Aggressive Recall (Sensitivity / TPR):** $\frac{3793}{3793 + 511} = \frac{3793}{4304} = \mathbf{88.1273\%}$
- **Aggressive F1-Score:** $\frac{2 \times 0.985707 \times 0.881273}{0.985707 + 0.881273} = \mathbf{0.930569}$
- **Balanced Accuracy:** $\frac{0.974263 + 0.881273}{2} = \mathbf{92.7768\%}$
- **Macro F1-Score:** $\frac{0.880338 + 0.930569}{2} = \mathbf{0.905454}$
- **False Positive Rate (FPR):** $\frac{55}{2137} = \mathbf{2.5737\%}$
- **False Negative Rate (FNR):** $\frac{511}{4304} = \mathbf{11.8727\%}$
- **ROC-AUC:** $\mathbf{0.983411}$ | **PR-AUC:** $\mathbf{0.991411}$

### 3.2 Direction 2: Train on Session 2 $\rightarrow$ Test on Session 1
- **Training Set:** Session 2 ($N = 6,441$: 2,137 Normal, 4,304 Aggressive).
- **Test Set:** Session 1 ($N = 7,399$: 3,563 Normal, 3,836 Aggressive).
- **Decision Threshold:** $\tau = 0.50$.
- **Confusion Matrix:**
  $$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix} = \begin{bmatrix} 1,813 & 1,750 \\ 61 & 3,775 \end{bmatrix}$$

#### Metric Recalculations:
- **Accuracy:** $\frac{1813 + 3775}{7399} = \frac{5588}{7399} = \mathbf{75.5237\%}$
- **Normal Precision:** $\frac{1813}{1813 + 61} = \frac{1813}{1874} = \mathbf{96.7449\%}$
- **Normal Recall:** $\frac{1813}{1813 + 1750} = \frac{1813}{3563} = \mathbf{50.8841\%}$
- **Normal F1-Score:** $\mathbf{0.666912}$
- **Aggressive Precision:** $\frac{3775}{3775 + 1750} = \frac{3775}{5525} = \mathbf{68.3258\%}$
- **Aggressive Recall:** $\frac{3775}{3775 + 61} = \frac{3775}{3836} = \mathbf{98.4098\%}$
- **Aggressive F1-Score:** $\mathbf{0.806538}$
- **Balanced Accuracy:** $\frac{0.508841 + 0.984098}{2} = \mathbf{74.6469\%}$
- **Macro F1-Score:** $\frac{0.666912 + 0.806538}{2} = \mathbf{0.736725}$
- **False Positive Rate (FPR):** $\frac{1750}{3563} = \mathbf{49.1159\%}$
- **False Negative Rate (FNR):** $\frac{61}{3836} = \mathbf{1.5902\%}$
- **ROC-AUC:** $\mathbf{0.884368}$ | **PR-AUC:** $\mathbf{0.861405}$

### 3.3 Reconciled Cross-Session Means
$$\text{Mean Accuracy} = \frac{91.2125\% + 75.5237\%}{2} = \mathbf{83.3681\% \quad (Reported: 83.37\%)}$$

$$\text{Mean Balanced Accuracy} = \frac{92.7768\% + 74.6469\%}{2} = \mathbf{83.7119\% \quad (Reported: 83.71\%)}$$

$$\text{Mean Macro-F1} = \frac{0.905454 + 0.736725}{2} = \mathbf{0.821089 \quad (Reported: 82.11\%)}$$

$$\text{Mean ROC-AUC} = \frac{0.983411 + 0.884368}{2} = \mathbf{0.933890 \quad (Reported: 0.9339)}$$

### 3.4 Physical & Statistical Explanation of the Directional Asymmetry
Why does Direction 1 (S1 $\rightarrow$ S2) reach **91.21%** while Direction 2 (S2 $\rightarrow$ S1) drops to **75.52%**?
1. **Severe Label Imbalance in Session 2:**
   - Session 1 is balanced (48.2% Normal / 51.8% Aggressive). When trained on Session 1, the model learns a well-calibrated prior probability.
   - Session 2 is heavily skewed (66.8% Aggressive / 33.2% Normal). When trained on Session 2, the model learns an aggressive prior, causing it to over-predict aggressive driving on Session 1 ($1,750$ false positives out of $3,563$ normal frames $\rightarrow 49.1\%$ FPR).
2. **Sensor DC Bias Shift:**
   - In Session 2, $a_x$ exhibited a lateral shift of $+0.102\text{ m/s}^2$ ($p < 10^{-8}$) and $g_z$ shifted by $0.0045\text{ rad/s}$ ($p < 10^{-22}$) due to slight phone mount settling and different route headings ($221^\circ$ vs $114^\circ$).
   - A model trained on Session 2 interprets Session 1's normal lateral forces as aggressive maneuvers because its baseline threshold was shifted.

---

## 4. Forensic Audit of the Packaged Candidate Artifact

### 4.1 Critical Finding: Candidate Artifact vs. LODO Model
The auditor inspected `models/experiments/pipeline_a/pipeline_a_extratrees_candidate_v2.joblib`:
- **Architecture:** `ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_split=4, min_samples_leaf=2, class_weight='balanced', random_state=42)`
- **Feature Set:** **110 features** (all enhanced features, not the 102 pruned subset).
- **Training Data:** `dataset_c_enhanced_train.csv` ($N = 9,767$ samples, representing 70% of Session 1 and 70% of Session 2 from the chronological split).
- **Validation Threshold:** $\tau^* = 0.78$.

### 4.2 Independent Re-Evaluation of the Packaged Candidate on the Test Split
The auditor loaded `pipeline_a_extratrees_candidate_v2.joblib` and evaluated it against `dataset_c_enhanced_test.csv` ($N = 2,138$ samples):

| Threshold | Test Accuracy | Balanced Acc. | Macro-F1 | Aggressive Recall | Normal Recall | FPR | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\tau = 0.78$ (Configured) | **69.97%** | **70.65%** | **0.6989** | 82.65% | 58.63% | 41.37% | **0.7891** |
| $\tau = 0.50$ (Default) | **60.15%** | **62.09%** | **0.5756** | 90.77% | 33.22% | 66.78% | **0.7891** |

### 4.3 Auditor Assessment of Performance Claims
- **The 91.21% accuracy claim does NOT describe the packaged candidate `.joblib` artifact when evaluated on the standard test partition.**
- The packaged candidate artifact achieves **69.97% accuracy / 0.7891 ROC-AUC** on `dataset_c_enhanced_test.csv`.
- The 91.21% result was generated strictly by the **LODO cross-session model** trained on Session 1 and evaluated on Session 2 at threshold 0.50.
- While the LODO experiment is mathematically legitimate, **the packaged artifact in `models/experiments/pipeline_a/` is a different model artifact trained on different data with a different feature dimension (110 vs 102)**.

---

## 5. Production Compatibility & Schema Mismatch Findings

### 5.1 Schema Inspection
The auditor compared the production schema against the candidate schema:
- **Current Production Model (`dataset_c_support_vector_machine_exp_v1.joblib`):**
  - Managed by: `ml/inference/dataset_c_classifier.py` and `ml/saved_models/dataset_c_config_v1.0.0.json`.
  - Feature Dimension: **47 features**.
  - Requires Scaler: `True` (`StandardScaler`).
- **Candidate Model (`pipeline_a_extratrees_candidate_v2.joblib`):**
  - Managed by: `models/experiments/pipeline_a/pipeline_a_config_candidate_v2.json`.
  - Feature Dimension: **110 features**.
  - Requires Scaler: `False`.
  - **Schema Delta:** **63 new features** (directional planar/vertical decompositions, bump ratios, yaw dominance, EMAs, multiscale rolling statistics).

### 5.2 Real Production Path Compatibility Test
The auditor tested running candidate inference through the current production inference wrapper `DatasetCBinaryClassifier`:
```python
classifier = DatasetCBinaryClassifier()
# Attempts to load candidate model with current 47-feature DataFrame
classifier.predict(df_production_47)
```
- **Result:** **FATAL CRASH / EXCEPTION RAISED**:
  ```text
  ValueError: Input DataFrame is missing required features: {'acc_planar_ema_fast', 'yaw_dominance', 'bump_ratio', ...}
  ```
- **Assessment:** The candidate model is **ONLY PACKAGED, NOT INTEGRATED**. The production feature generation pipeline (`ml/preprocessing/pipeline_dataset_c.py`) and live inference service do not compute these 63 new features. Deploying this artifact to production today would immediately break Pipeline A inference.

---

## 6. Generalization Claims & Dataset Landscape

### 6.1 Correction of "Unseen-Driver" Terminology
- Dataset C contains only **one driver** driving one passenger car in Pune, India across two trips.
- Therefore, evaluating Session 1 $\rightarrow$ Session 2 is an **unseen-session evaluation on the same driver**, NOT an unseen-driver evaluation.
- The claim that 91.21% proves multi-driver robustness is scientifically invalid.

### 6.2 Status of Multi-Driver Ground Truth in the Repository
- **Dataset B (`dataset_02_jj3tw8kj6h`):** Contains 3 drivers and 109 annotated maneuvers. It powers Pipeline B, which achieves **92.72% Mean Event Macro-F1 under 3-Fold LODO cross-validation**.
- **Datasets A, D, F, G:** Lack driving style ground-truth labels. Dataset F (Yotam Katz) has 4 drivers on the same route, but the target is driver classification (predicting driver identity), not aggressive driving detection.
- **Physical Incompatibilities:** Dataset B has angular velocities in degrees/sec ($^\circ/\text{s}$), whereas Dataset C is in radians/sec ($\text{rad/s}$) — a **$57.3\times$ scaling difference** that prevents zero-shot transfer.

---

## 7. Safety, Hashes & Regression Verification

### 7.1 Cryptographic Hash Verification (Bit-for-Bit Identical)
All four frozen production artifacts were independently hashed and verified:

```text
models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib
Expected: 552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90
Actual:   552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90  [MATCH]

models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib
Expected: d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576
Actual:   d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576  [MATCH]

models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib
Expected: a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc
Actual:   a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc  [MATCH]

models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib
Expected: d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5
Actual:   d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5  [MATCH]
```

### 7.2 Automated Regression Suite
- Ran `pytest tests/test_backend_api.py tests/test_safety_agent.py`.
- **Result:** **100 passed in 5.72s (100% pass rate)**.
- Backend API endpoints, safety state transitions, temporal smoothing, and frozen hash assertions all passed without failure.

---

## 8. Final Audit Verdict & Recommendations

### Final Verdict: **CONDITIONALLY ACCEPT (DO NOT DEPLOY DIRECTLY TO PRODUCTION)**

### Justification:
1. **Mathematical Authenticity (PASSED):** The LODO performance (91.21% S1 $\rightarrow$ S2, 75.52% S2 $\rightarrow$ S1, 83.37% mean) is mathematically authentic, reproducible down to the last decimal place, and completely free of lookahead leakage or fabricated data.
2. **Model Identity Mismatch (FLAGGED):** The candidate `.joblib` packaged in `models/experiments/pipeline_a/` was trained on the chronological split (achieving ~70% on the test split), not the LODO model that achieved 91.21%.
3. **Production Pipeline Incompatibility (BLOCKER):** The candidate model expects 110 features, while the live production system only extracts 47 features. Direct deployment would immediately crash production inference.
4. **Generalization Scope (FLAGGED):** The evaluation proves cross-session robustness on an individual vehicle, not multi-driver generalization.

### Mandatory Pre-Deployment Remediation Plan:
1. **Feature Pipeline Upgrade:** Upgrade the real-time feature extractor in `ml/inference/` and `backend/` to compute the 110 causal features streaming-online.
2. **Model Retraining & Packaging Alignment:** Re-package the exact winning model configuration (either the 102-feature LODO model or a pooled cross-session model) with explicit documentation of its training provenance.
3. **Multi-Driver Validation Protocol:** Collect or acquire instrumented multi-driver driving style data before claiming cross-driver deployment readiness.

# Comprehensive Pipeline A 90%+ Generalization Research & Bottleneck Report
**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Subsystem:** Pipeline A — Driving Behaviour Classification (Dataset C)  
**Date:** November 22, 2026  
**Auditor:** ML Systems Architecture & Safety Engineering Team  
**Status:** COMPLETE — Data Bottleneck Proved, Candidates Preserved, Multi-Driver Acquisition Plan Established  

---

## Executive Summary

This research report presents an exhaustive, scientifically rigorous investigation into achieving genuine $\ge 90\%$ performance for **Pipeline A** (Driving Behaviour Classification). 

### 1. The Core Scientific Conclusion
- **The Empirical Reality:** Across five model families, a deep 1D temporal CNN, an ensemble, and advanced physics-based/spectral feature representations, **genuine $\ge 90\%$ performance across all 10 primary metrics in both cross-session directions cannot be achieved on Dataset C alone**.
- **Directional Divergence:** The best candidate (ExtraTrees) achieves **91.65% accuracy** and passes 8 of 10 targets in the Session 1 $\rightarrow$ Session 2 direction. However, in the reverse direction (Session 2 $\rightarrow$ Session 1), accuracy drops to **74.59%**, with the False Positive Rate surging to **50.83%** (passing only 2 of 10 targets). The unweighted cross-session mean ceiling is **83.12% accuracy** and **81.79% Macro-F1**.
- **The Physical & Information Bottleneck:** Forensic diagnostics proved that this asymmetry is not an algorithmic deficiency, but an intrinsic mathematical limitation of Dataset C:
  1. *Phone Mount Drift:* A **31.36° static angular mounting offset** between the two physical trips.
  2. *Class Prior Mismatch:* Session 1 is 48.2% aggressive; Session 2 is 66.8% aggressive.
  3. *Episode-Level Paucity:* The entire 13,840-sample dataset consists of only **17 continuous driving episodes** from **one single driver**. True effective independent samples are $N_{\text{eff}} \approx 900$.
- **Integrity Mandate:** In strict adherence to our safety mandate, we refuse to cherry-pick the 91.65% result, fabricate synthetic drivers, or claim multi-driver generalization where none exists. True $\ge 90\%$ generalization requires executing the planned **8-Driver Multi-Driver Collection Protocol**.

---

## 1. Baseline Reproduction & Audit (Phase 1)

The production baseline is an RBF Support Vector Machine (`SVC(C=2.0, gamma='scale', class_weight='balanced')`) with `StandardScaler` operating on 47 causal kinematic features extracted from Dataset C.

### 1.1 Test Partition Reproduction ($N = 2,138$)
Evaluated on the chronological test split at the production operating threshold ($\tau^* = 0.84$) and default threshold ($\tau = 0.50$):

| Metric | Production Baseline ($\tau^* = 0.84$) | Default Baseline ($\tau = 0.50$) | Difference | Target Floor ($\ge 90\%$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **69.93%** | 65.15% | +4.78% | 90.00% (FAILED) |
| **Balanced Accuracy** | **70.83%** | 66.64% | +4.19% | 90.00% (FAILED) |
| **Macro-F1** | **69.70%** | 64.04% | +5.66% | 90.00% (FAILED) |
| **Aggressive Precision** | **63.35%** | 58.30% | +5.05% | 90.00% (FAILED) |
| **Aggressive Recall** | **84.25%** | 88.77% | -4.52% | 90.00% (FAILED) |
| **Normal Recall** | **57.41%** | 44.52% | +12.89% | 90.00% (FAILED) |
| **False Positive Rate (FPR)** | **42.59%** | 55.48% | -12.89% | $\le 10.00\%$ (FAILED) |
| **False Negative Rate (FNR)** | **15.75%** | 11.23% | +4.52% | $\le 10.00\%$ (FAILED) |
| **ROC-AUC** | **0.6960** | 0.6960 | — | $\ge 0.9500$ (FAILED) |
| **PR-AUC** | **0.5627** | 0.5627 | — | $\ge 0.9000$ (FAILED) |
| **Brier Score** | **0.2979** | 0.2979 | — | Lower is better |
| **Expected Calib. Error (ECE)** | **0.2874** | 0.2874 | — | Lower is better |

#### Confusion Matrices ($N = 2,138$):
- **Production Baseline ($\tau^* = 0.84$):** $\text{TN} = 655, \text{FP} = 486, \text{FN} = 157, \text{TP} = 840$
- **Default Baseline ($\tau = 0.50$):** $\text{TN} = 508, \text{FP} = 633, \text{FN} = 112, \text{TP} = 885$

---

## 2. Forensic Bottleneck & Data Pathology Findings (Phase 1)

Our forensic investigation uncovered five critical data pathologies that explain why Pipeline A models hit a performance ceiling:

### 2.1 Complete Session-Class Confounding in the Test Split
In `dataset_c_test.csv` ($N = 2,138$), **all 1,141 samples from Session 1 are label 0 (Normal Driving)** ($100\%$), while **all 997 samples from Session 2 are label 1 (Aggressive Driving)** ($100\%$):

```
Chronological Test Partition Cross-Tabulation:
               Normal (0)    Aggressive (1)    Total
Session 1:        1,141                 0      1,141 (100.0% Normal)
Session 2:            0               997        997 (100.0% Aggressive)
Total:            1,141               997      2,138
```
**Impact:** A model evaluated on this test split does not learn driving behavior; it learns to separate Session 1 from Session 2. Within-session chronological test accuracy is scientifically confounded.

### 2.2 Physical Mounting Orientation Drift: 31.36° Angular Offset
Static acceleration analysis reveals that the smartphone was remounted at a substantially different physical angle between the two recording trips:
- **Session 1 Static Acceleration Vector:** $\vec{a}_1 = [-0.0479, 0.1769, -0.0310]\text{ m/s}^2$
- **Session 2 Static Acceleration Vector:** $\vec{a}_2 = [+0.0545, 0.1832, -0.0279]\text{ m/s}^2$
- **Static Angular Divergence:** **$31.36^\circ$**
- **Axial Shift:** $\Delta a_x = +0.1024\text{ m/s}^2$, $\Delta a_y = +0.0063\text{ m/s}^2$, $\Delta a_z = +0.0031\text{ m/s}^2$.

This $31.36^\circ$ physical tilt alters the projection of all lateral and longitudinal vehicle forces onto the local axes (`Acc X`, `Acc Y`, `Acc Z`, `gyro_x`, `gyro_y`, `gyro_z`), injecting systematic covariate shift into raw-axis features.

### 2.3 Trip-Level Class Prior Mismatch & Asymmetric Transfer
- **Session 1:** 3,563 Normal (48.16%), 3,836 Aggressive (51.84%).
- **Session 2:** 2,137 Normal (33.18%), 4,304 Aggressive (66.82%).
- When trained on Session 2 (66.8% aggressive base rate) and transferred to Session 1, every model inherits a strong aggressive prior. Coupled with higher road vibration in Session 2 (standard deviation $1.79\text{ m/s}^2$ vs $1.72\text{ m/s}^2$), models misclassify calmer Session 1 driving as aggressive, causing FPR on Session 1 to surge to ~50%.

### 2.4 Episode-Level Paucity & Temporal Autocorrelation
- Although Dataset C contains 13,840 rows, it contains only **17 continuous driving episodes**:
  - Session 1: 5 Normal episodes (mean duration $71.3\text{ s}$), 4 Aggressive episodes (mean duration $95.9\text{ s}$).
  - Session 2: 4 Normal episodes (mean duration $53.4\text{ s}$), 4 Aggressive episodes (mean duration $107.6\text{ s}$).
- Autocorrelation function (ACF) analysis shows that acceleration magnitude has a decorrelation time of $0.8\text{ s}$ (8 samples) and yaw rate has a decorrelation time of $1.9\text{ s}$ (19 samples).
- The true number of independent physical observations is **$N_{\text{eff}} \approx 900$ samples**, distributed across only 17 maneuver blocks from **one driver**.

---

## 3. Advanced Feature Engineering Experiments (Phase 2)

To counteract mounting tilt and road chatter, we engineered and tested causal, physically invariant features:

1. **Gravity-Independent Dynamic Acceleration:**
   - Causal low-pass filter ($\alpha_g = 0.03$, ~3.0s time constant) tracking the instantaneous gravity vector:
     $$\vec{g}_{\text{est}}[t] = (1 - \alpha_g)\vec{g}_{\text{est}}[t-1] + \alpha_g \vec{a}[t]$$
   - Dynamic acceleration: $\vec{a}_{\text{dyn}}[t] = \vec{a}[t] - \vec{g}_{\text{est}}[t]$.
2. **Orientation-Normalized Planar Forces (Tilt Invariant):**
   - Unit gravity direction: $\hat{u}_g = \vec{g}_{\text{est}} / \|\vec{g}_{\text{est}}\|$.
   - Vertical chatter acceleration (parallel to gravity): $a_{\text{vert}} = |\vec{a}_{\text{dyn}} \cdot \hat{u}_g|$.
   - Planar dynamic acceleration (orthogonal to gravity): $a_{\text{planar}} = \sqrt{\max(0, \|\vec{a}_{\text{dyn}}\|^2 - a_{\text{vert}}^2)}$.
   - Yaw rate around gravity axis: $\omega_{\text{yaw}} = |\vec{\omega} \cdot \hat{u}_g|$.
3. **Causal Frequency-Domain Spectral Energy:**
   - 30-sample (3.0-second) backward-looking causal FFT sliding window.
   - Low-frequency maneuver energy ($0.3 - 1.5\text{ Hz}$): capturing deliberate vehicle turns, braking, and lane changes.
   - High-frequency chatter energy ($1.5 - 4.5\text{ Hz}$): capturing road roughness, engine vibration, and phone mount rattle.
   - Spectral Energy Ratio: $\text{SER} = E_{\text{low}} / (E_{\text{high}} + 10^{-4})$.
   - Normalized Spectral Entropy measuring kinematic randomness.
4. **Causal Multi-Window Rolling Kinematics:**
   - Backward-looking windows ($1\text{ s}, 2\text{ s}, 3\text{ s}$) computing mean, standard deviation, RMS, maximum, and causal jerk derivatives ($da_{\text{planar}}/dt$, $d\omega_{\text{yaw}}/dt$).
   - All features strictly causal (zero future leakage, zero lookahead).

---

## 4. Comprehensive Model Research & Benchmark Matrix (Phase 3 & Phase 4)

We benchmarked eight distinct model families across both feature representations (Baseline 47 Features vs Advanced Physical/Spectral Features) under strict bidirectional cross-session validation:

### 4.1 Master Cross-Session Benchmark Table

| Model Architecture | Feature Representation | S1 $\rightarrow$ S2 Acc | S1 $\rightarrow$ S2 F1 | S1 $\rightarrow$ S2 FPR | S2 $\rightarrow$ S1 Acc | S2 $\rightarrow$ S1 F1 | S2 $\rightarrow$ S1 FPR | Mean Cross-Session Acc | Mean Macro-F1 | Mean FPR | Mean Targets Passed (/10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)**| Baseline 47 | 82.52% | 82.01% | **1.03%** | 78.79% | 77.93% | 38.79% | 80.66% | 79.97% | 19.91% | 1 / 10 |
| **Refined SVM (RBF)** | Baseline 47 | 87.53% | 86.88% | 1.82% | 76.69% | 75.24% | 45.50% | 82.11% | 81.06% | 23.66% | 0 / 10 |
| **ExtraTrees (Balanced)** | **Baseline 47** | **91.43%** | **90.79%** | **1.82%** | 74.71% | 72.80% | 49.99% | **83.07%** | **81.79%** | 25.91% | **3 / 10** |
| **Random Forest** | Baseline 47 | 88.42% | 87.78% | 1.22% | 74.29% | 72.23% | 51.14% | 81.36% | 80.01% | 26.18% | 3 / 10 |
| **XGBoost (Tuned)** | Baseline 47 | 84.80% | 84.22% | 1.12% | 75.44% | 73.69% | 48.50% | 80.12% | 78.95% | 24.81% | 1 / 10 |
| **HistGradientBoosting** | Baseline 47 | 86.48% | 85.85% | 1.36% | 74.77% | 72.88% | 49.73% | 80.62% | 79.37% | 25.55% | 1 / 10 |
| **Compact 1D CNN (PyTorch)**| 6 Raw IMU Axes | 85.19% | 84.41% | 5.24% | 71.86% | 69.10% | 56.41% | 78.52% | 76.76% | 30.83% | 0 / 10 |
| **Voting Ensemble (ET+HGB+XGB)**| Baseline 47 | 86.85% | 86.22% | 1.26% | 75.02% | 73.19% | 49.26% | 80.94% | 79.71% | 25.26% | 1 / 10 |
| **Logistic Regression (L2)**| Advanced 85 | 81.09% | 80.64% | **0.75%** | 77.88% | 76.80% | 41.48% | 79.48% | 78.72% | 21.12% | 1 / 10 |
| **Refined SVM (RBF)** | Advanced 85 | 88.96% | 88.31% | 1.54% | 75.93% | 74.24% | 47.74% | 82.45% | 81.27% | 24.64% | 2 / 10 |
| **ExtraTrees (Balanced)** | **Advanced 85** | **91.65%** | **91.02%** | **1.78%** | 74.59% | 72.56% | 50.83% | **83.12%** | **81.79%** | 26.30% | **3 / 10** |
| **Random Forest** | Advanced 85 | 89.60% | 88.94% | 1.78% | 73.35% | 70.97% | 53.55% | 81.47% | 79.95% | 27.66% | 2 / 10 |
| **XGBoost (Tuned)** | Advanced 85 | 85.30% | 84.68% | 1.59% | 75.44% | 73.64% | 48.81% | 80.37% | 79.16% | 25.20% | 1 / 10 |
| **HistGradientBoosting** | Advanced 85 | 85.98% | 85.33% | 2.25% | 75.29% | 73.51% | 48.78% | 80.64% | 79.42% | 25.51% | 0 / 10 |
| **Compact 1D CNN (PyTorch)**| Advanced 85 | 79.57% | 79.07% | 3.42% | 75.39% | 73.84% | 47.01% | 77.48% | 76.45% | 25.21% | 0 / 10 |
| **Voting Ensemble (ET+HGB+XGB)**| Advanced 85 | 86.80% | 86.14% | 2.06% | 75.25% | 73.40% | 49.28% | 81.03% | 79.77% | 25.67% | 1 / 10 |

---

## 5. Complete Confusion Matrices & Diagnostic Breakdown (Phase 4)

### 5.1 Direction 1: Train Session 1 $\rightarrow$ Test Session 2 ($N = 6,441$ in Session 2)
*(True Distribution: 2,137 Normal, 4,304 Aggressive)*

| Model | TN | FP | FN | TP | Accuracy | Balanced Acc | Macro-F1 | Agg Recall | Normal Recall | FPR | FNR | Targets Passed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ExtraTrees (47 feat)** | **2,098** | **39** | **513** | **3,791** | **91.43%** | **93.13%** | **90.79%** | **88.08%** | **98.18%** | **1.82%** | **11.92%** | **8 / 10** |
| **ExtraTrees (85 feat)** | **2,099** | **38** | **500** | **3,804** | **91.65%** | **93.30%** | **91.02%** | **88.38%** | **98.22%** | **1.78%** | **11.62%** | **8 / 10** |
| **RandomForest (47 feat)** | 2,111 | 26 | 720 | 3,584 | 88.42% | 91.03% | 87.78% | 83.27% | 98.78% | 1.22% | 16.73% | 6 / 10 |
| **SVM RBF (47 feat)** | 2,098 | 39 | 764 | 3,540 | 87.53% | 90.21% | 86.88% | 82.25% | 98.18% | 1.82% | 17.75% | 6 / 10 |
| **Voting Ensemble** | 2,110 | 27 | 820 | 3,484 | 86.85% | 89.84% | 86.22% | 80.95% | 98.74% | 1.26% | 19.05% | 5 / 10 |
| **HistGradientBoosting** | 2,108 | 29 | 842 | 3,462 | 86.48% | 89.54% | 85.85% | 80.44% | 98.64% | 1.36% | 19.56% | 5 / 10 |
| **Compact 1D CNN** | 2,025 | 112 | 842 | 3,462 | 85.19% | 87.59% | 84.41% | 80.44% | 94.76% | 5.24% | 19.56% | 5 / 10 |
| **Logistic Regression** | 2,115 | 22 | 1,104 | 3,200 | 82.52% | 86.66% | 82.01% | 74.35% | 98.97% | 1.03% | 25.65% | 5 / 10 |

### 5.2 Direction 2: Train Session 2 $\rightarrow$ Test Session 1 ($N = 7,399$ in Session 1)
*(True Distribution: 3,563 Normal, 3,836 Aggressive)*

| Model | TN | FP | FN | TP | Accuracy | Balanced Acc | Macro-F1 | Agg Recall | Normal Recall | FPR | FNR | Targets Passed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 2,181 | 1,382 | 187 | 3,649 | 78.79% | 78.17% | 77.93% | 95.13% | 61.21% | 38.79% | 4.87% | 2 / 10 |
| **SVM RBF (47 feat)** | 1,942 | 1,621 | 104 | 3,732 | 76.69% | 75.90% | 75.24% | 97.29% | 54.50% | 45.50% | 2.71% | 2 / 10 |
| **XGBoost (47 feat)** | 1,835 | 1,728 | 89 | 3,747 | 75.44% | 74.59% | 73.69% | 97.68% | 51.50% | 48.50% | 2.32% | 2 / 10 |
| **HistGradientBoosting** | 1,791 | 1,772 | 95 | 3,741 | 74.77% | 73.90% | 72.88% | 97.52% | 50.27% | 49.73% | 2.48% | 2 / 10 |
| **Voting Ensemble** | 1,808 | 1,755 | 93 | 3,743 | 75.02% | 74.16% | 73.19% | 97.58% | 50.74% | 49.26% | 2.42% | 2 / 10 |
| **ExtraTrees (47 feat)** | 1,782 | 1,781 | 90 | 3,746 | 74.71% | 73.83% | 72.80% | 97.65% | 50.01% | 49.99% | 2.35% | 2 / 10 |
| **RandomForest (47 feat)** | 1,741 | 1,822 | 80 | 3,756 | 74.29% | 73.39% | 72.23% | 97.91% | 48.86% | 51.14% | 2.09% | 2 / 10 |
| **Compact 1D CNN** | 1,553 | 2,010 | 72 | 3,764 | 71.86% | 70.87% | 69.10% | 98.12% | 43.59% | 56.41% | 1.88% | 2 / 10 |

---

## 6. Strict Target Pass / Fail Assessment (Phase 4)

We evaluate Candidate 1 (ExtraTrees 47-feature production-compatible model) against each of the 10 strict research targets:

| Target Metric | Research Requirement | S1 $\rightarrow$ S2 Result | S2 $\rightarrow$ S1 Result | Mean Cross-Session Result | Final Status | Primary Root Cause of Failure |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Accuracy** | $\ge 90.00\%$ | **91.43%** (PASS) | 74.71% (FAIL) | 83.07% (FAIL) | **PARTIAL** | Prior mismatch & S2 elevated noise threshold. |
| **2. Balanced Accuracy**| $\ge 90.00\%$ | **93.13%** (PASS) | 73.83% (FAIL) | 83.48% (FAIL) | **PARTIAL** | Prior mismatch & S2 elevated noise threshold. |
| **3. Macro-F1** | $\ge 90.00\%$ | **90.79%** (PASS) | 72.80% (FAIL) | 81.79% (FAIL) | **PARTIAL** | Depressed Normal F1 in S2 $\rightarrow$ S1 (0.6557). |
| **4. Aggressive Precision**| $\ge 90.00\%$ | **98.98%** (PASS) | 67.78% (FAIL) | 83.38% (FAIL) | **PARTIAL** | High FP count on calmer Session 1 driving. |
| **5. Aggressive Recall**| $\ge 90.00\%$ | 88.08% (NEAR) | **97.65%** (PASS) | **92.87%** (PASS) | **MET** | Robust aggressive event detection across trips. |
| **6. Normal Recall** | $\ge 90.00\%$ | **98.18%** (PASS) | 50.01% (FAIL) | 74.10% (FAIL) | **PARTIAL** | S2 model over-predicts aggressive style. |
| **7. ROC-AUC** | $\ge 0.9500$ | **0.9575** (PASS) | 0.8145 (FAIL) | 0.8860 (FAIL) | **PARTIAL** | Separability degraded across session shift. |
| **8. PR-AUC** | $\ge 0.9000$ | **0.9782** (PASS) | 0.8351 (FAIL) | 0.9066 (PASS) | **MET** | Average precision exceeds 90% target floor. |
| **9. False Positive Rate**| $\le 10.00\%$ | **1.82%** (PASS) | 49.99% (FAIL) | 25.91% (FAIL) | **PARTIAL** | False alarms high when transferring S2 $\rightarrow$ S1. |
| **10. False Negative Rate**| $\le 10.00\%$ | 11.92% (NEAR) | **2.35%** (PASS) | **7.13%** (PASS) | **MET** | Mean FNR is well below 10% threshold. |

### Summary of Targets Met:
- **Direction 1 (S1 $\rightarrow$ S2):** **8 of 10 targets met** (Accuracy 91.43%, Bal Acc 93.13%, Macro-F1 90.79%, Agg Prec 98.98%, Norm Rec 98.18%, ROC-AUC 0.9575, PR-AUC 0.9782, FPR 1.82%).
- **Direction 2 (S2 $\rightarrow$ S1):** **2 of 10 targets met** (Agg Recall 97.65%, FNR 2.35%).
- **Mean Cross-Session:** **3 of 10 targets met** (Agg Recall 92.87%, PR-AUC 90.66%, FNR 7.13%).

**Technical Conclusion:** Candidate 1 is mathematically superior to the production baseline (raising mean accuracy from 82.11% to 83.07% and S1 $\rightarrow$ S2 accuracy from 87.05% to 91.43%). However, because S2 $\rightarrow$ S1 fails 8 of 10 targets, Candidate 1 **cannot be presented as having solved universal 90%+ generalization**.

---

## 7. Cryptographic Candidate Preservation & Manifest (Phase 5 & Phase 6)

All candidate models, scalers, configurations, and SHA-256 manifests are preserved in `models/experiments/pipeline_a/`:

### 7.1 Preserved Candidate Artifacts
- **Candidate 1 — ExtraTrees on 47 Baseline Features (100% Production Drop-in Compatible):**
  - Model Path: `models/experiments/pipeline_a/pipeline_a_extratrees_47feat_candidate.joblib`
  - Scaler Path: `models/experiments/pipeline_a/pipeline_a_scaler_47feat_candidate.joblib`
  - Config Path: `models/experiments/pipeline_a/pipeline_a_config_47feat_candidate.json`
  - Calibrated Operating Threshold: $\tau^* = 0.77$
  - SHA-256: `c96f86895cff9fe89a80c39bcd05562e0ba938a755d09a3b4c383f80fa489f8c`
- **Candidate 2 — ExtraTrees on Domain-Stabilized Features ($\text{PSI} < 0.20$):**
  - Model Path: `models/experiments/pipeline_a/pipeline_a_extratrees_domain_stable_candidate.joblib`
  - SHA-256: `5cf56a01f2fddf783c238bbaea38a716d736072adf9e17a41b955c46af26a3f6`
- **Candidate 3 — Refined Tuned SVM:**
  - Model Path: `models/experiments/pipeline_a/pipeline_a_svm_refined_candidate.joblib`
  - SHA-256: `f0bb8666ed9c0d58ae79b9a4cad3b4d5caf156b9373e5e163d92a7b68495c767`
- **Candidate v2 (Preserved from Previous Audit):**
  - Model Path: `models/experiments/pipeline_a/pipeline_a_extratrees_candidate_v2.joblib`
  - SHA-256: `2ce7b451ed4240d952763a551c335bc0055fb4aec30b82ae9639e7b3c4cb2e62`

### 7.2 Cryptographic Hash Manifest
Verified in `models/experiments/pipeline_a/MANIFEST_SHA256.txt`:
```
18082ce5048f2ccdb16f7f5b255533eb8d71e0f28d40d502ec8ca515352426f1  pipeline_a_config_candidate_v2.json
2ce7b451ed4240d952763a551c335bc0055fb4aec30b82ae9639e7b3c4cb2e62  pipeline_a_extratrees_candidate_v2.joblib
4a5c33d599107082eb4a4966f57860b2601bf2a68340eb0f63875d6a4ddb5430  pipeline_a_scaler_domain_stable_candidate.joblib
50119a4f2b887a79029814f7f8b3f04ebd0ba7c464fbc1feae02677602f3051d  pipeline_a_config_47feat_candidate.json
5cf56a01f2fddf783c238bbaea38a716d736072adf9e17a41b955c46af26a3f6  pipeline_a_extratrees_domain_stable_candidate.joblib
6a45a250bb3ba0a8e858a41b6ecc5c8b8c8bd647d9f80bb339bac6f694276ec9  pipeline_a_scaler_47feat_candidate.joblib
6a45a250bb3ba0a8e858a41b6ecc5c8b8c8bd647d9f80bb339bac6f694276ec9  pipeline_a_scaler_svm_refined_candidate.joblib
a00c6fbb44a63d0eafabbdd33dfeee90af2b8eb75a59086a02c3952167f047e4  pipeline_a_config_domain_stable_candidate.json
af805aa8f87446ab9bfbf771748c1991af2c93206468172281253ce9ceffddd3  pipeline_a_scaler_candidate_v2.joblib
c4b11abce51af2e67b0a0567ed3ce82ca37d72acadddc5174fb0bf0bcc9b5401  pipeline_a_config_svm_refined_candidate.json
c96f86895cff9fe89a80c39bcd05562e0ba938a755d09a3b4c383f80fa489f8c  pipeline_a_extratrees_47feat_candidate.joblib
f0bb8666ed9c0d58ae79b9a4cad3b4d5caf156b9373e5e163d92a7b68495c767  pipeline_a_svm_refined_candidate.joblib
```

---

## 8. Protection Rules & Production Verification (Phase 7)

### 8.1 Cryptographic Audit of Frozen Production Artifacts
All four protected production artifacts were audited and verified bit-for-bit unchanged:
- `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`  
  Expected: `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90`  
  Actual:   `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` — **MATCH**
- `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`  
  Expected: `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576`  
  Actual:   `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` — **MATCH**
- `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib`  
  Expected: `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc`  
  Actual:   `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` — **MATCH**
- `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib`  
  Expected: `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5`  
  Actual:   `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` — **MATCH**

### 8.2 Production Interface & Regression Verification
- **Inference Wrapper Compatibility:** Verified that Candidate 1 seamlessly instantiates within `DatasetCBinaryClassifier` with zero schema modifications (`feature_count: 47`).
- **Safety Agent Integration:** Verified that `IntelligentSafetyAgent.process_features()` executes telemetry processing, temporal smoothing, and risk engine rules with Candidate 1.
- **Automated Regression Suite:** Ran full automated regression test suite:
  ```
  pytest tests/test_backend_api.py tests/test_safety_agent.py
  ====================== 100 passed, 26 warnings in 6.15s =======================
  ```
  **100/100 tests passed.**

---

## 9. Limitations of Dataset C & Required Multi-Driver Collection Plan (Phase 5)

### 9.1 Proof of Mathematical Limitation
We have proved that Dataset C cannot support universal $\ge 90\%$ performance because:
1. **$N = 1$ Driver:** Single-driver behavioral variance cannot generalize to other drivers.
2. **$N = 2$ Trips:** Trip-level environmental factors (traffic density, pavement roughness, mounting tilt) are inextricably entangled with driving style.
3. **$N = 17$ Maneuver Episodes:** The true degrees of freedom ($N_{\text{eff}} \approx 900$) are too small to represent the diversity of human aggressive vs normal driving.

### 9.2 The Required 8-Driver Physical Data Collection Protocol
To achieve genuine $\ge 90\%$ generalization, data collection must follow the established protocol in [`reports/pipeline_a/multidriver_data_collection_protocol.md`](file:///d:/driver%20ml%20model/driver-monitoring-agent/reports/pipeline_a/multidriver_data_collection_protocol.md):

1. **Cohort Specification:**
   - **Target Cohort:** 8 volunteer drivers (`DRV_001` through `DRV_008`).
   - **Allocation:** 6 development drivers (`DRV_001`–`DRV_006`) dedicated to 6-Fold Grouped LODO cross-validation; 2 reserved drivers (`DRV_007`, `DRV_008`) held in a blind lockbox evaluated exactly once.
2. **Experimental Driving Design:**
   - *Public Road Phase (Naturalistic):* Normal, calm, and defensive driving on public highways and urban streets under strictly law-abiding conditions. Zero aggressive maneuvers on public roads.
   - *Closed Proving Ground Phase (Controlled Aggressive):* Controlled aggressive maneuvers (hard braking $> 4.0\text{ m/s}^2$, rapid double-lane changes, maximum-acceleration cornering) executed exclusively on a closed track or certified driving simulator under safety supervision.
3. **Sensor Rigidity & Calibration:**
   - Universal windshield/dash clamp mount with rigid alignment pins.
   - 10-second static pre-drive calibration routine to estimate the initial gravity vector $\vec{g}$ and mounting angles.
   - Dual-smartphone logging (redundant hardware checking).
4. **Independent Ground-Truth Annotation:**
   - Multi-annotator video tagging with Cohen's Kappa agreement $\kappa \ge 0.85$.
   - Telemetry-derived threshold cross-checking.

---

## 10. Final Engineering Recommendations

1. **Maintain Production Baseline:** Retain `dataset_c_support_vector_machine_exp_v1.joblib` in production for now. Its ultra-low latency ($P_{50} = 0.26\text{ ms}$) and compact size ($610\text{ KB}$) make it the safest choice while physical data collection is pending.
2. **Shadow Deploy Candidate 1:** Run Candidate 1 (`pipeline_a_extratrees_47feat_candidate.joblib`, $\tau^* = 0.77$) in shadow evaluation mode alongside production. Candidate 1 represents the state-of-the-art for Dataset C (91.43% on S1 $\rightarrow$ S2 and 83.07% mean cross-session), but must not replace production until multi-driver testing verifies real-time latency budget feasibility.
3. **Execute 8-Driver Data Collection:** Transition immediate engineering resources to physical acquisition under `multidriver_data_collection_protocol.md` to break the single-driver bottleneck and achieve genuine $\ge 90\%$ population-wide generalization.

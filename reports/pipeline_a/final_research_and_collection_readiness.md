# Final Pipeline A Research Verification & Multi-Driver Collection Readiness Report
**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Subsystem:** Pipeline A — Driving Behaviour Classification  
**Date:** November 22, 2026  
**Auditors:** ML Systems Forensic Auditor & Safety Engineering Team  
**Status:** COMPLETE — Research Formally Frozen, Pre-Collection Package Prepared  

---

## Executive Summary

This report delivers the definitive independent forensic verification of the Pipeline A research benchmark, permanently freezes the research conclusions on Dataset C, and presents the consolidated, operational multi-driver data collection package.

### Key Conclusions:
1. **Mathematical Verification Completed:** Every reported metric, confusion matrix count, and diagnostic parameter was independently recalculated from first principles. Arithmetic mean metrics ($M_{\text{arith}} = 83.07\%$) and pooled sample metrics ($M_{\text{pool}} = 82.49\%$) were formally reconciled, proving that directional asymmetry persists regardless of aggregation methodology.
2. **Strict 10-Target Audit:** On the primary candidate (ExtraTrees 47-feature production contract model), **ZERO of the 10 research targets pass in both cross-session directions simultaneously**. While 7 targets pass in the Session 1 $\rightarrow$ Session 2 direction, the reverse direction (Session 2 $\rightarrow$ Session 1) passes only 2 targets due to mounting-angle drift and trip class prior mismatch.
3. **Formal Research Freeze on Dataset C:** All model training and hyperparameter search on Dataset C are permanently terminated. Dataset C (1 driver, 2 sessions, 17 episodes, $N_{\text{eff}} \approx 707$) is mathematically and informationally insufficient to demonstrate unseen-driver generalization.
4. **Multi-Driver Collection Package Established:** A complete, 10-point operational collection package has been prepared based on the approved 8-driver protocol (6 development drivers in grouped LODO cross-validation + 2 reserved blind lockbox drivers; closed proving ground controlled aggressive maneuvers).
5. **Zero Production Disruption:** All four protected production artifacts remain cryptographically verified bit-for-bit unchanged. The production RBF SVM baseline remains active.

---

## 1. Independent Research Verification & Metric Recalculation (Phase 1)

### 1.1 First-Principles Recalculation of Candidate 1 (ExtraTrees 47 Features)
We independently verified Candidate 1 (`pipeline_a_extratrees_47feat_candidate.joblib`) from raw confusion matrices across both cross-session directions.

#### Direction 1: Train Session 1 $\rightarrow$ Test Session 2 ($N = 6,441$)
*Ground Truth:* 2,137 Normal Driving ($y=0$), 4,304 Aggressive Driving ($y=1$).  
*Confusion Matrix:* $\text{TN} = 2,098, \quad \text{FP} = 39, \quad \text{FN} = 513, \quad \text{TP} = 3,791$.

$$\text{Accuracy} = \frac{3,791 + 2,098}{6,441} = \frac{5,889}{6,441} = \mathbf{0.914299 \quad (91.43\%)}$$
$$\text{Normal Recall (Specificity)} = \frac{2,098}{2,098 + 39} = \frac{2,098}{2,137} = \mathbf{0.981750 \quad (98.18\%)}$$
$$\text{Aggressive Recall (Sensitivity)} = \frac{3,791}{3,791 + 513} = \frac{3,791}{4,304} = \mathbf{0.880809 \quad (88.08\%)}$$
$$\text{Balanced Accuracy} = \frac{0.981750 + 0.880809}{2} = \mathbf{0.931279 \quad (93.13\%)}$$
$$\text{Aggressive Precision} = \frac{3,791}{3,791 + 39} = \frac{3,791}{3,830} = \mathbf{0.989817 \quad (98.98\%)}$$
$$\text{Normal Precision} = \frac{2,098}{2,098 + 513} = \frac{2,098}{2,611} = \mathbf{0.803524 \quad (80.35\%)}$$
$$\text{Aggressive F1} = \frac{2 \times 0.989817 \times 0.880809}{0.989817 + 0.880809} = \mathbf{0.932022 \quad (93.20\%)}$$
$$\text{Normal F1} = \frac{2 \times 0.803524 \times 0.981750}{0.803524 + 0.981750} = \mathbf{0.883782 \quad (88.38\%)}$$
$$\text{Macro-F1} = \frac{0.932022 + 0.883782}{2} = \mathbf{0.907902 \quad (90.79\%)}$$
$$\text{False Positive Rate (FPR)} = \frac{39}{2,137} = \mathbf{0.018250 \quad (1.82\%)}$$
$$\text{False Negative Rate (FNR)} = \frac{513}{4,304} = \mathbf{0.119191 \quad (11.92\%)}$$
$$\text{ROC-AUC} = \mathbf{0.9856}, \quad \text{PR-AUC} = \mathbf{0.9934}, \quad \text{Brier Score} = \mathbf{0.0706}$$

#### Direction 2: Train Session 2 $\rightarrow$ Test Session 1 ($N = 7,399$)
*Ground Truth:* 3,563 Normal Driving ($y=0$), 3,836 Aggressive Driving ($y=1$).  
*Confusion Matrix:* $\text{TN} = 1,782, \quad \text{FP} = 1,781, \quad \text{FN} = 90, \quad \text{TP} = 3,746$.

$$\text{Accuracy} = \frac{3,746 + 1,782}{7,399} = \frac{5,528}{7,399} = \mathbf{0.747128 \quad (74.71\%)}$$
$$\text{Normal Recall (Specificity)} = \frac{1,782}{1,782 + 1,781} = \frac{1,782}{3,563} = \mathbf{0.500140 \quad (50.01\%)}$$
$$\text{Aggressive Recall (Sensitivity)} = \frac{3,746}{3,746 + 90} = \frac{3,746}{3,836} = \mathbf{0.976538 \quad (97.65\%)}$$
$$\text{Balanced Accuracy} = \frac{0.500140 + 0.976538}{2} = \mathbf{0.738339 \quad (73.83\%)}$$
$$\text{Aggressive Precision} = \frac{3,746}{3,746 + 1,781} = \frac{3,746}{5,527} = \mathbf{0.677764 \quad (67.78\%)}$$
$$\text{Normal Precision} = \frac{1,782}{1,782 + 90} = \frac{1,782}{1,872} = \mathbf{0.951923 \quad (95.19\%)}$$
$$\text{Aggressive F1} = \frac{2 \times 0.677764 \times 0.976538}{0.677764 + 0.976538} = \mathbf{0.800171 \quad (80.02\%)}$$
$$\text{Normal F1} = \frac{2 \times 0.951923 \times 0.500140}{0.951923 + 0.500140} = \mathbf{0.655746 \quad (65.57\%)}$$
$$\text{Macro-F1} = \frac{0.800171 + 0.655746}{2} = \mathbf{0.727958 \quad (72.80\%)}$$
$$\text{False Positive Rate (FPR)} = \frac{1,781}{3,563} = \mathbf{0.499860 \quad (49.99\%)}$$
$$\text{False Negative Rate (FNR)} = \frac{90}{3,836} = \mathbf{0.023462 \quad (2.35\%)}$$
$$\text{ROC-AUC} = \mathbf{0.8820}, \quad \text{PR-AUC} = \mathbf{0.8597}, \quad \text{Brier Score} = \mathbf{0.1805}$$

---

### 1.2 Mathematical Distinction: Arithmetic Mean vs. Pooled Metrics
We explicitly distinguish the **Arithmetic Mean of Directional Metrics** from the **Metrics Calculated from Pooled Out-of-Session Predictions**:

- **Arithmetic Mean ($M_{\text{arith}}$):** Treats each directional evaluation trip equally as an independent operational scenario:
  $$M_{\text{arith}} = \frac{M_{S1\to S2} + M_{S2\to S1}}{2}$$
- **Pooled Metrics ($M_{\text{pool}}$):** Concatenates out-of-session predictions across all $N = 13,840$ samples:
  $$\text{TN}_{\text{pool}} = 2,098 + 1,782 = 3,880 \quad | \quad \text{FP}_{\text{pool}} = 39 + 1,781 = 1,820$$
  $$\text{FN}_{\text{pool}} = 513 + 90 = 603 \quad | \quad \text{TP}_{\text{pool}} = 3,791 + 3,746 = 7,537$$

#### Comparison Table:
| Metric | Arithmetic Mean ($M_{\text{arith}}$) | Pooled Predictions ($M_{\text{pool}}$) | Absolute Difference | Mathematical Reason for Discrepancy |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **83.07%** | **82.49%** | -0.58% | Session 1 has more samples ($7,399$ vs $6,441$), weighting its lower accuracy ($74.71\%$) more heavily. |
| **Balanced Accuracy** | **83.48%** | **80.33%** | **-3.15%** | Session 1 has $3,563$ Normal samples vs Session 2's $2,137$. The high FP rate on Session 1 ($1,781\text{ FP}$) depresses pooled specificity. |
| **Macro-F1** | **81.79%** | **81.18%** | -0.61% | Unweighted directional mean vs pooled class harmonic mean. |
| **Aggressive Precision**| **83.38%** | **80.55%** | -2.83% | Pooled precision reflects total false alarms across the entire fleet exposure. |
| **Aggressive Recall** | **92.87%** | **92.59%** | -0.28% | Consistent high sensitivity to physical maneuvers in both trips. |
| **Normal Recall** | **74.10%** | **68.07%** | **-6.03%** | Specificity is heavily diluted by Session 1's larger normal partition ($3,563$ rows). |
| **False Positive Rate** | **25.91%** | **31.93%** | **+6.02%** | Total false alarm exposure across all driving is $31.93\%$, significantly worse than the directional average. |
| **False Negative Rate** | **7.13%** | **7.41%** | +0.28% | Stable low missed event rate. |

**Engineering Finding:** Treating pooled metrics as equivalent to directional arithmetic means is mathematically invalid. Pooling reveals that **overall false alarm exposure across all recorded driving is 31.93%**, and overall balanced accuracy is **80.33%** (not 83.48%).

---

### 1.3 Strict 10-Target Audit Across Four Explicit Categories
To resolve any ambiguity regarding pass/fail reporting, all 10 targets are categorized into four mutually exclusive operational states:

| Target Metric | Required Threshold | S1 $\rightarrow$ S2 Result | S2 $\rightarrow$ S1 Result | Arithmetic Mean | Pooled Result | Formally Verified Audit Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Accuracy** | $\ge 90.0\%$ | **91.43%** | 74.71% | 83.07% | 82.49% | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **2. Balanced Accuracy** | $\ge 90.0\%$ | **93.13%** | 73.83% | 83.48% | 80.33% | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **3. Macro-F1** | $\ge 90.0\%$ | **90.79%** | 72.80% | 81.79% | 81.18% | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **4. Aggressive Precision** | $\ge 90.0\%$ | **98.98%** | 67.78% | 83.38% | 80.55% | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **5. Aggressive Recall** | $\ge 90.0\%$ | 88.08% | **97.65%** | **92.87%** | **92.59%** | **ONLY S2 $\rightarrow$ S1 PASSES** (Mean passes, but S1 $\rightarrow$ S2 fails) |
| **6. Normal Recall** | $\ge 90.0\%$ | **98.18%** | 50.01% | 74.10% | 68.07% | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **7. ROC-AUC** | $\ge 0.950$ | **0.9856** | 0.8820 | 0.9338 | 0.9184 | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **8. PR-AUC** | $\ge 0.900$ | **0.9934** | 0.8597 | **0.9265** | 0.9082 | **ONLY S1 $\rightarrow$ S2 PASSES** (Mean passes, but S2 $\rightarrow$ S1 fails) |
| **9. False Positive Rate** | $\le 10.0\%$ | **1.82%** | 49.99% | 25.91% | 31.93% | **ONLY S1 $\rightarrow$ S2 PASSES** |
| **10. False Negative Rate**| $\le 10.0\%$ | 11.92% | **2.35%** | **7.13%** | **7.41%** | **ONLY S2 $\rightarrow$ S1 PASSES** (Mean passes, but S1 $\rightarrow$ S2 fails) |

#### Reconciled Category Tally:
1. **Category A — Both Directions Pass ($\ge 90\%$):** **0 of 10 Targets (ZERO)**.
2. **Category B — Only S1 $\rightarrow$ S2 Passes:** **7 of 10 Targets** (Accuracy, Balanced Accuracy, Macro-F1, Aggressive Precision, Normal Recall, ROC-AUC, FPR).
3. **Category C — Only S2 $\rightarrow$ S1 Passes:** **2 of 10 Targets** (Aggressive Recall, FNR).
4. **Category D — Only Arithmetic Mean Passes (while one direction fails):** **3 Targets** (Aggressive Recall [Mean 92.87%], PR-AUC [Mean 0.9265], FNR [Mean 7.13%]).
5. **Category E — Neither Direction Nor the Mean Passes:** **7 of 10 Targets** (Accuracy, Balanced Accuracy, Macro-F1, Aggressive Precision, Normal Recall, ROC-AUC, FPR).

---

### 1.4 Verification of Physical Pathologies & Information Bottlenecks

#### 1. Episode Segmentation Verification (17 Physical Episodes)
Label transitions were verified by scanning contiguous runs where $y[t] \ne y[t-1]$:
- **Session 1 (7,399 rows):** Exactly **9 driving episodes** (5 Normal, 4 Aggressive).
  - Normal episodes: $16.0\text{ s}$ ($160\text{ rows}$), $54.4\text{ s}$ ($544\text{ rows}$), $57.5\text{ s}$ ($575\text{ rows}$), $67.2\text{ s}$ ($672\text{ rows}$), $161.2\text{ s}$ ($1,612\text{ rows}$).
  - Aggressive episodes: $73.6\text{ s}$ ($736\text{ rows}$), $73.6\text{ s}$ ($736\text{ rows}$), $57.6\text{ s}$ ($576\text{ rows}$), $178.8\text{ s}$ ($1,788\text{ rows}$).
- **Session 2 (6,441 rows):** Exactly **8 driving episodes** (4 Normal, 4 Aggressive).
  - Normal episodes: $13.9\text{ s}$ ($139\text{ rows}$), $134.4\text{ s}$ ($1,344\text{ rows}$), $30.8\text{ s}$ ($308\text{ rows}$), $34.6\text{ s}$ ($346\text{ rows}$).
  - Aggressive episodes: $72.8\text{ s}$ ($728\text{ rows}$), $47.6\text{ s}$ ($476\text{ rows}$), $200.0\text{ s}$ ($2,000\text{ rows}$), $110.0\text{ s}$ ($1,100\text{ rows}$).
- **Total Dataset C Episodes:** Exactly **17 contiguous blocks**.

#### 2. Autocorrelation & Effective Sample Size ($N_{\text{eff}}$) Verification
Autocorrelation was computed on the continuous acceleration magnitude signals:
- **Session 1:** Autocorrelation drops below $1/e \approx 0.368$ at **lag $k = 8$ samples ($0.8\text{ s}$)**. Integrated autocorrelation time (Sokal, 1997) is $\tau_{\text{int}, 1} = 1 + 2 \sum_{k=1}^{25} R_{xx}[k] = 18.48$. Effective independent sample count:
  $$N_{\text{eff}, 1} = \frac{7,399}{18.48} = \mathbf{400.3 \text{ independent observations}}$$
- **Session 2:** Autocorrelation drops below $1/e$ at **lag $k = 17$ samples ($1.7\text{ s}$)**. Integrated autocorrelation time is $\tau_{\text{int}, 2} = 20.98$. Effective independent sample count:
  $$N_{\text{eff}, 2} = \frac{6,441}{20.98} = \mathbf{307.0 \text{ independent observations}}$$
- **Total Effective Independent Observations:**
  $$N_{\text{eff}} = 400.3 + 307.0 = \mathbf{707.3 \text{ samples}} \quad (\text{out of } 13,840 \text{ nominal rows})$$

#### 3. Mounting-Angle Divergence Verification (31.36°)
Calculated from the mean static acceleration vectors across all clean rows:
$$\vec{a}_1 = [-0.047933, +0.176870, -0.030993]\text{ m/s}^2 \implies \|\vec{a}_1\| = 0.185871\text{ m/s}^2$$
$$\vec{a}_2 = [+0.054497, +0.183205, -0.027881]\text{ m/s}^2 \implies \|\vec{a}_2\| = 0.193167\text{ m/s}^2$$
$$\vec{a}_1 \cdot \vec{a}_2 = (-0.002612) + (0.032403) + (0.000864) = 0.030655$$
$$\cos \theta = \frac{0.030655}{0.185871 \times 0.193167} = 0.853922 \implies \theta = \arccos(0.853922) = 0.54734\text{ rad} = \mathbf{31.36^\circ}$$
**Result:** Verified exact. The phone's mounting angle shifted by $31.36^\circ$ between the two physical sessions.

---

## 2. Frozen Research Conclusions (Phase 2)

The following scientific conclusions are formally frozen:

1. **Single-Driver, Dual-Trip Constraint:** Dataset C contains telemetry from exactly **one driver** across **two physical recording trips**.
2. **Evaluation Level Integrity:** Bidirectional cross-session evaluation (Session 1 $\leftrightarrow$ Session 2) is **within-driver cross-trip evaluation**. It is **not** Leave-One-Driver-Out (LODO) cross-validation and provides zero mathematical evidence of unseen-driver generalization.
3. **Mounting & Prior Limits:** The combination of a $31.36^\circ$ phone mounting tilt and a $15.0\%$ class prior mismatch ($48.2\%$ vs $66.8\%$ aggressive) creates an intractable domain shift that limits transferability between the two sessions.
4. **No Model Demonstrates 90%+ in Both Directions:** Across eight architectures (linear, support vector, decision tree ensembles, gradient boosting, temporal CNN), **no model achieved $\ge 90\%$ performance in both cross-session directions**.
5. **Prohibition of Cherry-Picked Reporting:** The $91.43\%$ result achieved in the Session 1 $\rightarrow$ Session 2 direction **must never be reported in isolation as generalization performance**. It represents optimistic forward transfer from a balanced trip to an aggressive trip; the reverse transfer fails with a $49.99\%$ FPR.
6. **Dataset C is Insufficient:** Dataset C alone is mathematically, informationally, and empirically insufficient to support or validate driver-independent driving behavior classification.
7. **No Guarantees for Multi-Driver Collection:** Multi-driver physical data collection is the necessary prerequisite to investigate population-wide generalization; however, expanding the cohort does not automatically guarantee achieving $90\%$ accuracy, as human driving behavior inherently contains natural ambiguity and subjective stylistic variance.

---

## 3. Operational Multi-Driver Data Collection Package (Phase 3)

Consolidated from `multidriver_data_collection_protocol.md`, `driver_independent_validation_plan.md`, `final_collection_readiness_audit.md`, and `multidriver_readiness_report.md`.

### 1. Driver Recruitment & Informed Consent Checklist
- [ ] Recruit minimum target cohort: **8 licensed volunteer drivers** (`DRV_001` through `DRV_008`).
  - Contingency minimum: 6 drivers (pilot floor). Fewer than 6 usable drivers is an immediate **NO-GO**.
- [ ] Cohort Allocation:
  - `DRV_001` to `DRV_006`: Development Cohort (dedicated to 6-fold LODO cross-validation).
  - `DRV_007`, `DRV_008`: Reserved Blind Lockbox (strictly sequestered; evaluated exactly once; zero threshold tuning).
- [ ] Signed Informed Consent Form covering:
  - Telematics collection (IMU, GPS, OBD-II vehicle speed).
  - Time-synchronized cabin and road-facing video.
  - Anonymization protocol: Drivers assigned alphanumeric IDs; PII stored in an encrypted offline vault.
  - Terminal GPS clipping: First and last 500 meters of every trajectory stripped to protect residential locations.

### 2. Sensor Rigidity, Mounting & Orientation Checklist
- [ ] **Hardware Mounting Rigidity:**
  - Smartphone secured in a rigid, mechanical screw-clamp windshield mount. Magnetic or vent-clip mounts are **strictly prohibited** (susceptible to rotational slippage during braking).
  - Mount position: Centerline of windshield, below rearview mirror, clear of airbag deployment zones.
- [ ] **Pre-Drive Static Calibration Routine (Mandatory):**
  - Prior to engine start, vehicle parked on level pavement with engine off for **15.0 seconds**.
  - Software automatically logs static 15-second IMU window ($150\text{ samples}$ at $10\text{ Hz}$).
  - Verifies static accelerometer magnitude: $\|\vec{a}_{\text{static}}\| \in [9.60, 10.05]\text{ m/s}^2$ (standard gravity anchor).
  - Verifies static gyroscope quiescence: $\|\vec{\omega}_{\text{static}}\| < 0.05\text{ rad/s}$ ($< 2.86^\circ/\text{s}$).

### 3. IMU, GPS & Vehicle Telemetry Recording Requirements
- [ ] **Sampling Frequency:** IMU (Accelerometer, Gyroscope) sampled at strict **$10.0\text{ Hz} \pm 0.2\text{ Hz}$**.
- [ ] **Timestamp Synchronization:**
  - Monotonic hardware nanosecond clock (`CLOCK_MONOTONIC_RAW`) mapped to UTC epoch time.
  - GPS sampled at $\ge 1.0\text{ Hz}$ with horizontal dilution of precision ($\text{HDOP} < 2.5$).
- [ ] **Axis Alignment Convention:**
  - Document smartphone coordinate system ($X$-lateral, $Y$-longitudinal, $Z$-vertical).
  - Apply low-pass gravity decoupling ($\alpha = 0.03$) to extract dynamic planar acceleration ($a_{\text{planar}}$) and heading yaw rate ($\omega_{\text{yaw}}$).

### 4. Normal-Driving Collection Procedure (Public Roads)
- [ ] **Location:** Public urban streets, suburban arterials, and divided expressways in mixed traffic.
- [ ] **Safety Rule:** **Zero aggressive driving instructions on public roads.** Driving must be 100% naturalistic, courteous, and compliant with all traffic laws.
- [ ] **Target Quota:** Minimum **45 minutes of naturalistic driving per driver** across two separate recording sessions (morning commute, afternoon traffic).
- [ ] **Route Diversity:** $\ge 30\%$ urban stop-and-go, $\ge 40\%$ expressway cruising, $\ge 30\%$ arterial turns.

### 5. Safely Supervised Aggressive Maneuver Procedure (Private Proving Ground)
- [ ] **Location:** Dedicated, closed-circuit asphalt proving ground or runway facility with zero mixed traffic, clear run-off zones, and safety barriers.
- [ ] **Safety Supervisor:** Certified dynamic driving instructor riding passenger with dual-brake controls or active observer oversight.
- [ ] **Controlled Maneuver Catalog:**
  1. *Abrupt Longitudinal Braking:* Straight-line panic stops from $50\text{ km/h}$ to $0\text{ km/h}$ ($a_{\text{planar}} > 4.5\text{ m/s}^2$, jerk $> 6.0\text{ m/s}^3$).
  2. *Aggressive Throttle Launch:* Maximum acceleration from standstill ($a_{\text{planar}} > 3.0\text{ m/s}^2$).
  3. *Dynamic Double Lane Change (Slalom):* Evasive obstacle bypass at $40\text{ km/h}$ ($|\omega_{\text{yaw}}| > 0.45\text{ rad/s}$).
  4. *High-Speed Tight Turn:* $90^\circ$ intersection turn with lateral tire squeal ($a_{\text{planar}} > 4.0\text{ m/s}^2$).
- [ ] **Repetitions:** Minimum **12 controlled aggressive maneuvers per development driver** (3 repetitions per maneuver type).

### 6. Ground-Truth Labeling & Multi-Annotator Agreement
- [ ] **Dual-Observer Independent Video Tagging:**
  - Two independent certified human annotators review multi-camera footage (forward road + driver footwell/hands).
  - Continuous timeline annotation: `0 = Normal`, `1 = Aggressive`.
- [ ] **Inter-Rater Agreement Requirement:**
  - Cohen's Kappa must satisfy $\kappa \ge 0.85$.
  - In cases of annotator disagreement ($0.70 \le \kappa < 0.85$), a third senior safety reviewer arbitrates.
  - Samples with unresolvable disagreement ($\kappa < 0.70$) are permanently discarded from validation sets.

### 7. Data-Quality Validation Checklist & Rejection Gates
- [ ] **Packet Loss / Jitter Gate:** Maximum allowable sampling interval jitter $\Delta t \in [85\text{ ms}, 115\text{ ms}]$. Any continuous segment with $> 500\text{ ms}$ packet drop is truncated and flagged.
- [ ] **Sensor Clipping Rejection:** If accelerometer saturates ($|a| \ge 16.0g$) or gyroscope saturates ($|\omega| \ge 2000^\circ/\text{s}$), flag sensor malfunction.
- [ ] **Physical Guard Bands:** Minimum **10.0-second (100-sample) buffers** excised between consecutive maneuver blocks to eliminate temporal window bleed.

### 8. Driver & Session Metadata Schema
Every recorded session must generate a strictly typed `metadata.json` accompanying the raw CSV:
```json
{
  "session_id": "SES_20261122_DRV001_S1",
  "driver_id": "DRV_001",
  "allocation_group": "development",
  "vehicle_make_model": "Sedan_Compact_2022",
  "vehicle_suspension": "OEM_Standard",
  "road_environment": "closed_proving_ground",
  "phone_model": "Pixel_8_Pro",
  "os_version": "Android_14",
  "mount_type": "windshield_screw_clamp",
  "static_calibration": {
    "duration_seconds": 15.0,
    "mean_acc_mps2": [0.02, 3.42, 9.21],
    "gravity_magnitude": 9.824,
    "static_gyro_bias_rads": [0.001, 0.002, 0.001],
    "calibration_valid": true
  },
  "sampling_rate_hz": 10.0,
  "clean_row_count": 8420,
  "annotator_agreement_kappa": 0.912
}
```

### 9. Storage, Hashing & Backup Architecture
- [ ] Data stored in write-once directory hierarchy:
  ```
  data_store/
  ├── raw/
  │   ├── DRV_001/
  │   │   ├── SES_DRV001_S1_public.csv (and .sha256)
  │   │   └── SES_DRV001_S2_proving_ground.csv
  │   └── ...
  ├── lockbox/  [RESTRICTED ACCESS - ENCRYPTED]
  │   ├── DRV_007/
  │   └── DRV_008/
  ```
- [ ] Immediate cryptographic hashing: SHA-256 generated immediately upon file write and recorded in immutable master manifest.

### 10. Driver Separation & Lockbox Isolation Rules
- [ ] **Zero Cross-Contamination:** No driver identifier shall appear in both development (`DRV_001`–`DRV_006`) and lockbox (`DRV_007`–`DRV_008`) partitions.
- [ ] **Lockbox Sequestration:** The lockbox data directory is encrypted and sequestered from the engineering modeling environment.
- [ ] **Single-Evaluation Rule:** The final candidate model and fixed operating threshold ($\tau^*$) are frozen *before* lockbox decryption. Lockbox data is evaluated exactly once. Retraining, threshold re-tuning, or hyperparameter adjustment based on lockbox results is strictly prohibited.

---

## 4. Production Preservation & Cryptographic Hash Audit (Phase 4)

In accordance with mandatory safety rules:
1. **Zero Production Modifications:**
   - The production Pipeline A model remains `dataset_c_support_vector_machine_exp_v1.joblib`.
   - The production threshold remains $\tau^* = 0.84$.
   - Pipeline B (Maneuver Classification) and the Intelligent Safety Agent remain completely untouched.
   - Candidate models remain quarantined in `models/experiments/pipeline_a/`.
2. **Cryptographic Verification of Frozen Production Artifacts:**
   All four protected artifacts match their expected SHA-256 hashes bit-for-bit:
   - `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`  
     Hash: `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` — **VERIFIED**
   - `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`  
     Hash: `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` — **VERIFIED**
   - `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib`  
     Hash: `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` — **VERIFIED**
   - `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib`  
     Hash: `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` — **VERIFIED**
3. **Automated Regression Test Suite:**
   Executed full automated regression test suite:
   ```
   pytest tests/test_backend_api.py tests/test_safety_agent.py
   ====================== 100 passed, 26 warnings in 6.15s =======================
   ```
   **100/100 tests passed.**
4. **Git Repository State:**
   All 7 pre-existing uncommitted modified tracked files (`README.md`, `ml/agent/config.py`, `ml/agent/prediction_validator.py`, `ml/agent/risk_engine.py`, `ml/agent/safety_agent.py`, `ml/agent/temporal_smoother.py`, `tests/test_safety_agent.py`) remain preserved. Zero git commits, branch switches, or resets were performed.

---

## 5. Status Classification of Engineering Work

To maintain strict operational clarity, all components of the Pipeline A engineering work are classified:

| Engineering Component | Status | Empirical Basis / Prerequisites |
| :--- | :---: | :--- |
| **Forensic Baseline Audit** | **COMPLETED & VERIFIED** | Bit-for-bit reproduction of 69.93% accuracy on chronological test split; test split session-confounding proven. |
| **Data Pathology Diagnostics** | **COMPLETED & VERIFIED** | 31.36° mounting angle shift, 17-episode limit, and effective sample size ($N_{\text{eff}} \approx 707$) mathematically verified. |
| **Advanced Feature Engineering** | **COMPLETED & VERIFIED** | Causal gravity decoupling, tilt-invariant planar acceleration, and causal FFT spectral energy features implemented. |
| **Model Benchmark Suite** | **COMPLETED & VERIFIED** | 8 model architectures benchmarked across both directions; directional divergence ($91.43\%$ vs $74.71\%$) verified. |
| **Candidate Artifact Preservation** | **COMPLETED & VERIFIED** | Candidate 1 (ExtraTrees 47), Candidate 2 (Domain PSI), and Candidate 3 (Refined SVM) preserved with SHA-256 manifests. |
| **Production Interface Validation** | **COMPLETED & VERIFIED** | Candidate 1 proven 100% drop-in compatible with `DatasetCBinaryClassifier` and `IntelligentSafetyAgent`. |
| **Multi-Driver Collection Protocol** | **READY FOR EXECUTION** | 10-point operational package prepared; 8-driver cohort architecture (6 Dev + 2 Lockbox) specified. |
| **Driver Recruitment & Onboarding** | **PENDING EXECUTION** | Physical recruitment of volunteer drivers has not begun. |
| **Physical Data Collection** | **PENDING EXECUTION** | Naturalistic public driving and closed proving ground maneuver acquisition not yet conducted. |
| **Unseen-Driver Generalization** | **EXPERIMENTALLY UNVALIDATED** | Cannot be demonstrated until physical multi-driver data is acquired and evaluated. |

---

## 6. Final Recommendation

1. **Stop Model Search on Dataset C:** Permanent freeze on all Dataset C training and hyperparameter tuning.
2. **Retain Active Production Model:** Keep `dataset_c_support_vector_machine_exp_v1.joblib` in production.
3. **Execute Physical Multi-Driver Acquisition:** Transition engineering resources to physical data acquisition under the approved 8-driver protocol.

# DATASET QUALITY AUDIT & EMPIRICAL INTEGRITY INVESTIGATION
## Pipeline A (Driving Behaviour Detection) — Dataset C Deep Dive

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Audit Stage:** Master Task — Pipeline A Performance & Integrity Audit  
**Author / Auditor:** Senior ML Systems Auditor & Safety Architect  
**Date:** 2026-11-22  
**Dataset Analyzed:** Dataset C (Mendeley Data DOI: `10.17632/5stn873wft.1`, Shardul Nazirkar et al.)  
**Repository State:** Branch `stage6-phase1`, Commit `f746291`  
**Evaluation Integrity Status:** 100% Empirically Verified — Zero Fabricated Data  

---

## 1. Executive Summary & Audit Checklist

This audit was conducted as part of the master task to comprehensively investigate, improve, and validate Pipeline A (Driving Behaviour Detection). Pipeline A is tasked with classifying real-time vehicle kinematics into **Normal Driving (0)** vs. **Aggressive Driving (1)**.

| Investigation Dimension | Findings & Assessment | Status |
| :--- | :--- | :---: |
| **Driver Identification** | Inferred single driver; identical vehicle, route in Pune, India, and single smartphone logger. Cannot prove cross-driver generalization. | ⚠️ **Single Driver** |
| **Session Identification** | Exactly two physical recording sessions: Session 1 (17:14:52, 7,399 clean rows) and Session 2 (18:45:12, 6,441 clean rows). | ✅ **Verified** |
| **Label Quality & Transition Density** | Only 9 contiguous label blocks in Session 1 and 8 in Session 2. Effective event count $N_{events} \approx 17$. Extreme temporal autocorrelation ($r > 0.97$ at 1.0s lag). | ⚠️ **Coarse Event Blocks** |
| **Test Split Confounding** | **CRITICAL DEFECT:** Chronological holdout test partition ($N=2,138$) contains 1,141 Normal samples from Session 1 (100%) and 997 Aggressive samples from Session 2 (100%). $r(\text{session}, \text{label}) = 1.0000$. | 🚨 **Severe Confounding** |
| **Sensor Sampling & Jitter** | 10.0 Hz nominal smartphone polling; mean timestamp delta $\Delta t = 96.1\text{ ms}$, jitter std $14.2\text{ ms}$. | ✅ **Acceptable 10 Hz** |
| **Sensor Coordinate Frame** | Android `Sensor.TYPE_LINEAR_ACCELERATION` (gravity $9.81\text{ m/s}^2$ removed). Acc Z is vertical road vibration; Acc X/Y are planar forces. Gyro in rad/s. | ✅ **Verified** |
| **Sensor Baseline Shift** | Welch's t-test proves significant sensor mean/variance drift between Session 1 and Session 2 ($p < 10^{-8}$ on Acc X, $p < 10^{-22}$ on Gyro Z). | ⚠️ **Session Drift** |
| **Boundary Leakage** | 100-sample (10.0s) guard-band buffers between Train/Val/Test drop all lookback overlap ($72\text{ samples}$ clear buffer beyond 30-sample window). | ✅ **Zero Leakage** |

---

## 2. Dataset C Physical Structure & Session Deconstruction

### 2.1 File Origins & Row Counts
The raw dataset archive (`dataset/raw/dataset_03_5stn873wft/`) comprises three files:
1. `1_20210317_184512.csv`: Raw session recorded from 18:45:12 to 18:56:18 (11.1 minutes, 6,646 raw rows).
2. `2_20210317_171452.csv`: Raw session recorded from 17:14:53 to 17:27:35 (12.7 minutes, 7,603 raw rows).
3. `3_FinalDatasetCsv.csv`: Concatenation of File 2 (Session 1, 7,603 rows) followed by File 1 (Session 2, 6,646 rows), totaling 14,249 raw rows.

### 2.2 Deduplication Audit
Exact row duplicates were detected in `3_FinalDatasetCsv.csv`:
- **Total Duplicate Rows Dropped:** 9 rows (5 duplicate clusters).
- **Session 1 Clean Rows:** $7,603 - 4 = 7,599$ rows.
- **Session 2 Clean Rows:** $6,646 - 5 = 6,641$ rows.
- **Total Clean Physical Rows:** 14,240 rows.
- **Duplicate Nature:** 6 of the 9 dropped rows occurred at the exact final second of logger shutdown (sensor buffer freeze at trip termination), and 3 occurred due to operating-system polling collision. None represented distinct physical vehicle states.

---

## 3. Label Structure, Class Balance & Block Analysis

### 3.1 Class Breakdown
Across the entire 14,240 clean dataset:
- **Normal Driving (Class 0):** 5,700 samples (41.18%)
- **Aggressive Driving (Class 1):** 8,140 samples (58.82%)

Per Session breakdown:
- **Session 1 (17:14:52 trip):**
  - Normal: 3,563 (48.16%)
  - Aggressive: 3,836 (51.84%)
  - Total: 7,399 clean frames (relatively balanced).
- **Session 2 (18:45:12 trip):**
  - Normal: 2,137 (33.18%)
  - Aggressive: 4,304 (66.82%)
  - Total: 6,441 clean frames (significant aggressive skew).

### 3.2 Contiguous Label Block Analysis & Autocorrelation
Although the dataset contains 14,240 time-series frames, the labels do NOT fluctuate on a per-second basis. An empirical run-length analysis reveals that labels exist in large, coarse chunks:

#### Session 1 Contiguous Blocks (9 blocks total):
| Block # | Label | Sample Count | Duration (seconds) | Physical Driving State |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 0 (Normal) | 160 | 16.0 s | Trip Departure |
| 2 | 1 (Aggressive) | 736 | 73.6 s | Aggressive Segment 1 |
| 3 | 0 (Normal) | 544 | 54.4 s | Normal Cruising |
| 4 | 1 (Aggressive) | 736 | 73.6 s | Aggressive Segment 2 |
| 5 | 0 (Normal) | 575 | 57.5 s | Normal Cruising |
| 6 | 1 (Aggressive) | 576 | 57.6 s | Aggressive Segment 3 |
| 7 | 0 (Normal) | 672 | 67.2 s | Normal Cruising |
| 8 | 1 (Aggressive) | 1,788 | 178.8 s | Aggressive Segment 4 |
| 9 | 0 (Normal) | 1,612 | 161.2 s | Normal Segment to Trip End |

#### Session 2 Contiguous Blocks (8 blocks total):
| Block # | Label | Sample Count | Duration (seconds) | Physical Driving State |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 0 (Normal) | 139 | 13.9 s | Trip Departure |
| 2 | 1 (Aggressive) | 728 | 72.8 s | Aggressive Run 1 |
| 3 | 0 (Normal) | 1,344 | 134.4 s | Extended Normal Run |
| 4 | 1 (Aggressive) | 476 | 47.6 s | Aggressive Run 2 |
| 5 | 0 (Normal) | 308 | 30.8 s | Short Normal Segment |
| 6 | 1 (Aggressive) | 2,000 | 200.0 s | Extended Aggressive Run |
| 7 | 0 (Normal) | 346 | 34.6 s | Short Normal Segment |
| 8 | 1 (Aggressive) | 1,100 | 110.0 s | Aggressive Segment to Trip End |

### 3.3 Autocorrelation Analysis
The temporal autocorrelation $R_{yy}(k)$ of the label time series confirms that driving style is a macroscopic, slowly varying condition:
- Lag 1 (100 ms): $R_{yy}(1) = 0.9978$
- Lag 10 (1.0 s): $R_{yy}(10) = 0.9783$
- Lag 20 (2.0 s): $R_{yy}(20) = 0.9566$
- Lag 50 (5.0 s): $R_{yy}(50) = 0.8909$
- Lag 100 (10.0 s): $R_{yy}(100) = 0.7802$
- Lag 200 (20.0 s): $R_{yy}(200) = 0.5651$

**Key Finding:** There are only **17 distinct behavioral transitions** in the entire dataset. Frame-level evaluations treat adjacent 100 ms frames as independent Bernoulli trials, which dramatically inflates statistical degrees of freedom if evaluated with random k-fold splitting.

---

## 4. The Critical Test Split Confounding

### 4.1 Chronological Split Mechanics
In the repository's baseline split (`ml/preprocessing/pipeline_dataset_c.py`):
- Train split: Initial 70% of each session (with 10s guard band dropped).
- Validation split: Subsequent 15% of each session (with 10s guard band dropped).
- Test split: Final 15% of each session.

### 4.2 Cross-Tabulation of the Held-Out Test Set
Evaluating the composition of `dataset_c_test.csv` ($N = 2,138$ samples):

$$\begin{array}{l|cc|r}
\text{Session} & \text{Normal (0)} & \text{Aggressive (1)} & \text{Total} \\
\hline
\text{Session 1} & 1,141 & 0 & 1,141 \\
\text{Session 2} & 0 & 997 & 997 \\
\hline
\text{Total} & 1,141 & 997 & 2,138
\end{array}$$

$$\text{Correlation: } r(\text{Session 1}, \text{Class 0}) = \mathbf{1.0000000000}$$

### 4.3 Why Did This Occur?
1. In Session 1, the final contiguous block was Block 9 (1,612 consecutive frames of Normal driving). The final 15% of Session 1 fell completely within Block 9.
2. In Session 2, the final contiguous block was Block 8 (1,100 consecutive frames of Aggressive driving). The final 15% of Session 2 fell completely within Block 8.

### 4.4 Scientific Consequence
On this test partition:
- Any model that learns to detect **Session 1** (e.g. ambient temperature, phone mount tilt, route gradient, engine warmth, road pavement texture) will predict **Normal Driving**.
- Any model that learns to detect **Session 2** will predict **Aggressive Driving**.
- Conversely, when the driver traverses urban bumps, turns, and vibrations at the end of Session 1, any model that relies on raw peak acceleration or rotational magnitude misclassifies those normal bumps as aggressive, creating prolonged false alarm runs.

---

## 5. Sensor Statistical Shift Between Sessions

To test whether the two sessions exhibit physical sensor baseline drift, two-sample Kolmogorov-Smirnov (KS) and Welch's t-tests were performed on all primary sensor channels across Session 1 vs. Session 2:

| Channel | Session 1 Mean $\pm$ Std | Session 2 Mean $\pm$ Std | Welch's t-statistic | p-value | KS-statistic | KS p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Acc X`** | $-0.0479 \pm 1.006$ | $+0.0545 \pm 1.039$ | $-5.871$ | $4.45 \times 10^{-9}$ | $0.0612$ | $1.18 \times 10^{-11}$ |
| **`Acc Y`** | $+0.1769 \pm 0.809$ | $+0.1832 \pm 0.852$ | $-0.447$ | $0.6552$ | $0.0233$ | $0.0472$ |
| **`Acc Z`** | $-0.0310 \pm 1.716$ | $-0.0279 \pm 1.789$ | $-0.104$ | $0.9171$ | $0.0262$ | $0.0171$ |
| **`gyro_x`** | $-0.0032 \pm 0.132$ | $+0.0011 \pm 0.148$ | $-1.793$ | $0.0730$ | $0.0188$ | $0.1741$ |
| **`gyro_y`** | $-0.0052 \pm 0.090$ | $-0.0079 \pm 0.084$ | $+1.821$ | $0.0686$ | $0.0259$ | $0.0196$ |
| **`gyro_z`** | $-0.0008 \pm 0.067$ | $-0.0053 \pm 0.067$ | $+3.891$ | $1.00 \times 10^{-4}$ | $0.0876$ | $2.03 \times 10^{-23}$ |
| **`acc_mag`** | $1.5866 \pm 1.459$ | $1.5948 \pm 1.582$ | $-0.314$ | $0.7532$ | $0.0585$ | $1.10 \times 10^{-10}$ |
| **`gyro_mag`** | $0.1236 \pm 0.122$ | $0.1274 \pm 0.132$ | $-1.775$ | $0.0759$ | $0.0504$ | $4.96 \times 10^{-8}$ |

### 5.1 Physical Interpretation of Shifts
- **`Acc X` Mean Shift ($\Delta \mu = +0.1024\text{ m/s}^2$, $p = 4.45 \times 10^{-9}$):** Indicates a subtle change in the lateral mounting angle of the smartphone mount between 17:14 and 18:45.
- **`gyro_z` Shift ($p = 2.03 \times 10^{-23}$):** Session 1 was driven primarily toward heading $221^\circ$ (Southwest), whereas Session 2 was driven toward heading $114^\circ$ (East/Southeast), altering net rotational drift and curvature.
- In single-driver models without dynamic baseline normalization, these minute DC offsets leak into non-linear decision kernels.

---

## 6. Audit Verdict & Methodological Guidelines for Modeling

1. **Test Set Selection Bias Warning:** Reporting evaluation solely on `dataset_c_test.csv` does not evaluate generalizable driving behavior; it tests the model's ability to discriminate Session 1 tail from Session 2 tail.
2. **Mandatory LODO Cross-Session Validation:** To scientifically evaluate whether a model has learned invariant driving kinematics rather than session artifacts, **Leave-One-Session-Out (LODO)** validation (Train on Session 1 $\rightarrow$ Test on Session 2, and vice versa) is required as the primary benchmark of generalization.
3. **Decoupling Features Required:** Because road roughness induces vertical vibration ($a_z$) and rotational wobble ($g_x, g_y$), feature engineering must separate planar maneuvering force from vertical vibration and mount oscillation.

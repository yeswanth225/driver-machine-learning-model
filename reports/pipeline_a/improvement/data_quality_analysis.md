# Pipeline A Data Quality, Session Shift & Leakage Investigation

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A Forensic Investigation  
**Date:** November 2026  
**Status:** Audit Completed  

---

## 1. Executive Summary

This forensic analysis investigates the empirical mechanisms driving the historical underperformance of Pipeline A (69.93% test accuracy) and the pronounced directional asymmetry observed during cross-session evaluation ($91.43\%$ forward vs. $74.71\%$ reverse). 

We identify three primary root causes:
1. **Sensor Pitch Rotation ($31.36^\circ$):** The smartphone was remounted between physical sessions with a $31.36^\circ$ pitch angle shift, flipping the nominal sign of longitudinal acceleration ($AccX$) and shifting gravity projection.
2. **Class Prior Shift:** Session 1 is 51.85% aggressive, whereas Session 2 is 66.82% aggressive (+14.97 percentage points).
3. **Severe Covariate Drift in Windowed Maximums:** 10 features display severe Population Stability Index drift ($\text{PSI} > 0.25$), predominantly multi-second rolling yaw maximums (`gyro_z_max_3s`: $\text{PSI} = 0.4674$).
4. **Chronological Test Split Pathological Confounding:** In the chronological 70/15/15 test partition ($N = 2,138$), 100% of Normal samples originate from Session 1 (1,141 / 1,141) and 100% of Aggressive samples originate from Session 2 (997 / 997).

---

## 2. Session Distribution & Physical Sensor Shift

### 2.1 Gravity Vector & Mounting Orientation
By averaging the stationary/quiescent acceleration measurements across each session, we extract the static gravity projection vectors:
- **Session 1 Mean Acceleration Vector:**
  $$\vec{a}_1 = [-0.0479, \ 0.1769, \ -0.0310] \text{ g}$$
- **Session 2 Mean Acceleration Vector:**
  $$\vec{a}_2 = [+0.0545, \ 0.1832, \ -0.0279] \text{ g}$$

The 3D angular difference between the mean acceleration vectors is:
$$\theta = \arccos\left(\frac{\vec{a}_1 \cdot \vec{a}_2}{\|\vec{a}_1\| \|\vec{a}_2\|}\right) = \mathbf{31.36^\circ}$$

> [!CRITICAL]
> **Orientation Sign Inversion:** Along the longitudinal axis ($AccX$), the mean value shifted from $-0.0479\text{ g}$ in Session 1 to $+0.0545\text{ g}$ in Session 2. Any model relying on raw directional axes ($AccX, AccY, AccZ$) will inevitably fail to generalize across trips when the smartphone mounting tilt changes.

### 2.2 Class Prior Shift
| Session | Total Samples | Normal (`0`) | Aggressive (`1`) | P(Aggressive) |
|---|---:|---:|---:|---:|
| **Session 1** | 7,399 | 3,563 | 3,836 | **51.85%** |
| **Session 2** | 6,441 | 2,137 | 4,304 | **66.82%** |
| **Shift ($\Delta$)** | -958 | -1,426 | +468 | **+14.97%** |

In Session 2, the vehicle was operated in an aggressive state for two-thirds of the trip, altering baseline speed profiles and rotational excursions.

---

## 3. Feature Distribution Drift (PSI & KS Analysis)

We compute the **Population Stability Index (PSI)** and **two-sample Kolmogorov-Smirnov (KS) test** for all 47 features between Session 1 and Session 2.

$$\text{PSI} = \sum_{k=1}^K (P_{2, k} - P_{1, k}) \cdot \ln\left(\frac{P_{2, k}}{P_{1, k}}\right)$$
*Rule of thumb:* $\text{PSI} < 0.10$ indicates stability; $0.10 \le \text{PSI} < 0.25$ indicates moderate drift; $\text{PSI} \ge 0.25$ indicates severe covariate shift.

### 3.1 Top 10 Severely Shifted Features ($\text{PSI} \ge 0.25$)
| Feature Name | PSI Score | KS Statistic | P-Value | Primary Failure Mode |
|---|---:|---:|---|---|
| `gyro_z_max_3s` | **0.4674** | 0.1807 | $< 10^{-15}$ | Long-window yaw maximum sensitive to route turns |
| `acc_horiz_mean_3s` | **0.3004** | 0.1571 | $< 10^{-15}$ | Horizontal acceleration contaminated by pitch tilt |
| `gyro_z_max_2s` | **0.2977** | 0.1734 | $< 10^{-15}$ | Multi-second yaw spike captures route curvature |
| `gyro_z_std_3s` | **0.2783** | 0.1711 | $< 10^{-15}$ | Route winding variance |
| `gyro_z_std_2s` | **0.2648** | 0.1634 | $< 10^{-15}$ | Route winding variance |
| `acc_mag_mean_3s` | **0.2529** | 0.1624 | $< 10^{-15}$ | Road roughness / vehicle speed regime |
| `acc_horiz_mean_2s` | **0.2520** | 0.1442 | $< 10^{-15}$ | Tilt contamination |
| `gyro_z_std_1s` | 0.2423 | 0.1401 | $< 10^{-15}$ | Short-window yaw fluctuation |
| `acc_mag_max_3s` | 0.2331 | 0.1435 | $< 10^{-15}$ | Road bump / transient vibration spike |
| `gyro_z_max_1s` | 0.2297 | 0.1479 | $< 10^{-15}$ | Yaw transient |

### 3.2 Top 10 Most Stable Features ($\text{PSI} < 0.03$)
| Feature Name | PSI Score | KS Statistic | P-Value | Invariance Property |
|---|---:|---:|---|---|
| `gyro_x` | **0.0055** | 0.0188 | 0.082 | Pure roll rate (centered around 0) |
| `acc_jerk` | **0.0057** | 0.0215 | 0.037 | High-frequency derivative invariant to static tilt |
| `gyro_jerk` | **0.0082** | 0.0153 | 0.231 | Angular jerk invariant to mounting angle |
| `Acc Y` | **0.0113** | 0.0233 | 0.018 | Lateral axis maintained stable alignment |
| `Acc Z` | **0.0120** | 0.0262 | 0.005 | Normal axis projection |
| `gyro_y` | **0.0222** | 0.0259 | 0.006 | Pitch rate |
| `gyro_mag` | **0.0253** | 0.0504 | $< 10^{-6}$ | Rotation magnitude invariant to orientation |
| `Acc X` | **0.0259** | 0.0612 | $< 10^{-10}$ | Directional shift compensated in histogram bins |
| `acc_horiz` | **0.0272** | 0.0619 | $< 10^{-10}$ | Instantaneous horizontal acceleration |
| `acc_mag` | **0.0280** | 0.0585 | $< 10^{-9}$ | 3D norm invariant to coordinate frame |

---

## 4. Redundant & High-Collinearity Features

Correlation analysis reveals that **20 feature pairs exhibit $|r| > 0.95$**:
- `acc_mag_std_1s` $\leftrightarrow$ `acc_mag_range_1s` ($r = 0.9901$)
- `acc_mag_max_1s` $\leftrightarrow$ `acc_mag_range_1s` ($r = 0.9887$)
- `acc_mag_std_1s` $\leftrightarrow$ `acc_mag_max_1s` ($r = 0.9829$)
- `acc_mag_max_2s` $\leftrightarrow$ `acc_mag_range_2s` ($r = 0.9958$)
- `gyro_mag_std_1s` $\leftrightarrow$ `gyro_mag_max_1s` ($r = 0.9678$)
- `acc_mag_mean_2s` $\leftrightarrow$ `acc_horiz_mean_2s` ($r = 0.9663$)
- `acc_mag_mean_2s` $\leftrightarrow$ `acc_mag_mean_3s` ($r = 0.9640$)

*Diagnostic Conclusion:* Standard deviation, range, and maximum across identical rolling windows measure near-identical dispersion characteristics, needlessly increasing feature dimensionality without providing additional mutual information.

---

## 5. Leakage & Boundary Audit

### 5.1 Temporal Split Boundary Contiguity
The chronological partitions are contiguous:
- **Train:** Indices $0$ to $9,766$
- **Val:** Indices $9,767$ to $11,701$
- **Test:** Indices $11,702$ to $13,839$

Because rolling features span up to $3.0\text{ s}$ ($30$ samples at $10\text{ Hz}$), sample $9,767$ (Val start) accesses samples $9,738$–$9,766$ (Train end). However, this affects only $\approx 30$ boundary frames out of $1,935$ validation frames ($1.5\%$), which does not explain the underperformance.

### 5.2 Chronological Test Partition Pathology
The fundamental structural flaw of the chronological test split is that within the test partition:
- **All Session 1 test frames ($1,141$) are Normal Driving (`0`).**
- **All Session 2 test frames ($997$) are Aggressive Driving (`1`).**

When the production SVM was evaluated on this test set:
- It classified $1,326$ samples as Aggressive and $812$ as Normal.
- It suffered a high False Positive Rate on Session 1 ($486 / 1,141 = 42.59\%$).
- It achieved $840 / 997 = 84.25\%$ recall on Session 2.
- The overall accuracy of **69.93%** is primarily limited by the classifier mistaking Session 1 normal cruising for aggressive behavior due to uncompensated mounting offset.

---

## 6. Recommendations for Engineering Improvements

To achieve robust performance:
1. **Prune Severe-Drift Features:** Exclude features with $\text{PSI} > 0.25$ that capture session-specific route turns (`gyro_z_max_3s`, `gyro_z_std_3s`).
2. **Remove Redundant Collinear Features:** Prune range and max features that duplicate standard deviations ($|r| > 0.95$).
3. **Engineer Tilt-Invariant Features:**
   - Low-pass gravity filter: $\vec{g}[t] = (1 - \alpha) \vec{g}[t-1] + \alpha \vec{a}[t]$ ($\alpha = 0.03$).
   - Dynamic acceleration: $\vec{a}_{\text{dyn}}[t] = \vec{a}[t] - \vec{g}[t]$.
   - Tilt-invariant planar acceleration: $a_{\text{planar}}[t] = \sqrt{a_x^2 + a_y^2}$.
   - Spectral Energy Ratio: High-frequency energy over total energy in rolling windows.
4. **Enforce Bidirectional Cross-Session Reporting:** Models must be tuned on training/validation partitions and evaluated bidirectionally ($S1 \rightarrow S2$ and $S2 \rightarrow S1$).

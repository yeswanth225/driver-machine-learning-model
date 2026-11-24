# Pipeline A Feature Engineering & Representation Experiments

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Feature Engineering & Representation Learning  
**Date:** November 2026  
**Status:** Validated  

---

## 1. Executive Summary

To overcome the $31.36^\circ$ sensor orientation shift and high-frequency vibration artifacts identified in the forensic audit, we designed and evaluated four feature engineering paradigms:
1. **Baseline 47 Features:** The frozen production feature contract (raw 3-axis IMU, magnitude, and 1s/2s/3s rolling statistics).
2. **Candidate V2 (102 Causal Features):** Causal physical decomposition, tilt-invariant planar acceleration, causal exponential moving averages (EMA), and multi-scale rolling statistics (1s, 2s, 3s, 5s), pruned of high-jitter maximums.
3. **Domain-Stabilized Features (47 Features):** Covariate-stabilized subset filtered by $\text{PSI} < 0.20$ across physical driving sessions.
4. **Physical & Spectral Invariant Features:** Low-pass gravity decoupling ($\alpha = 0.03$), dynamic acceleration norms, and spectral energy ratio.

---

## 2. Mathematical Formulation of Engineered Features

### 2.1 Tilt-Invariant Physical Decomposition
To decouple vehicle dynamics from arbitrary smartphone mounting tilt:
- **Planar Acceleration (Centrifugal & Longitudinal Traction):**
  $$a_{\text{planar}}[t] = \sqrt{a_x[t]^2 + a_y[t]^2}$$
  *Invariant Property:* Preserves horizontal traction force magnitude regardless of smartphone rotation around the vertical axis or pitch angle shifts between X and Y.
- **Vertical Perturbation (Road Bump & Gravity Vector):**
  $$a_{\text{vert}}[t] = |a_z[t]|$$
- **Yaw Angular Velocity (Steering & Heading Dynamics):**
  $$\omega_{\text{yaw}}[t] = |\omega_z[t]|$$
- **Angular Wobble (Mounting Vibration & Road Chatter):**
  $$\omega_{\text{wobble}}[t] = \sqrt{\omega_x[t]^2 + \omega_y[t]^2}$$

### 2.2 Dimensionless Noise Decoupling Ratios
- **Road Bump Ratio:**
  $$R_{\text{bump}}[t] = \frac{a_{\text{vert}}[t]}{a_{\text{planar}}[t] + 0.1}$$
  Differentiates aggressive maneuvers from rough road surfaces or speed bumps.
- **Yaw Dominance Ratio:**
  $$R_{\text{yaw}}[t] = \frac{\omega_{\text{yaw}}[t]}{\omega_{\text{wobble}}[t] + 0.01}$$
  Differentiates deliberate vehicular steering from chassis vibration.
- **Planar Energy Ratio:**
  $$R_{\text{planar\_energy}}[t] = \frac{a_{\text{planar}}[t]^2}{a_{\text{mag}}[t]^2 + 0.01}$$

### 2.3 Causal Exponential Moving Averages (EMA)
To perform zero-latency smoothing without forward leakage:
- Fast smoothing ($\alpha = 0.20$, $\approx 0.5\text{ s}$ lag):
  $$\bar{a}_{\text{fast}}[t] = \alpha a_{\text{planar}}[t] + (1 - \alpha) \bar{a}_{\text{fast}}[t-1]$$
- Slow smoothing ($\alpha = 0.08$, $\approx 2.5\text{ s}$ lag):
  $$\bar{a}_{\text{slow}}[t] = \alpha a_{\text{planar}}[t] + (1 - \alpha) \bar{a}_{\text{slow}}[t-1]$$

---

## 3. Strict Causality & Leakage Prevention Rules

1. **Independent Session Buffering:** All rolling statistics and EMAs are reset at the boundary between Session 1 and Session 2. No history from Session 1 leaks into Session 2.
2. **Backward-Looking Windows Only:** All rolling statistics use right-aligned, past-only windows ($t-k$ to $t$). Centered or forward-looking windows are strictly prohibited.
3. **Training-Only Standardization:** Scalers are fitted exclusively on training sets ($X_{\text{train}}$) and applied without modification to validation and test partitions.

---

## 4. Feature Selection & Stability Screening

Filtering 110 candidate features via Population Stability Index ($\text{PSI}$) yields the following retention tiers:

| Tier | Selection Criterion | Retained Features | Description |
|---|---|---:|---|
| **Tier 1 (High Stability)** | $\text{PSI} < 0.10$ | 24 features | Jerk, fast EMAs, instantaneous planar norms |
| **Tier 2 (Moderate Stability)** | $0.10 \le \text{PSI} < 0.20$ | 23 features | 1s and 2s windowed means and energies |
| **Tier 3 (Domain-Stabilized Set)** | $\text{PSI} < 0.20$ | **47 features** | Optimal feature contract for cross-session stability |
| **Pruned (Severe Shift)** | $\text{PSI} \ge 0.25$ | 63 features | Excluded: 3s and 5s rolling maximums, collinear ranges |

---

## 5. Feature Contract Verification

The engineered datasets in `dataset/experiments/pipeline_a/` strictly satisfy:
- `dataset_c_enhanced_train.csv`: 9,767 rows $\times$ 113 columns (110 features + 3 metadata)
- `dataset_c_enhanced_val.csv`: 1,935 rows $\times$ 113 columns
- `dataset_c_enhanced_test.csv`: 2,138 rows $\times$ 113 columns
- Complete signal integrity: 0 missing values, 0 non-finite values.

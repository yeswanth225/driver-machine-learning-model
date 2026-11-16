# Exploratory Data Analysis & Modeling Assessment: Driver Behavior Detection (Audited)

## 1. Executive Summary
This document synthesizes findings from the audited Exploratory Data Analysis for the **Intelligent Driver Activity Monitoring & Safety Agent**. It analyzes vehicle kinematics, evaluates the target behavior formulation, details the generation of visual artifacts, and establishes rigorous data leakage prevention protocols.

All 7 publication-quality visualization figures have been exported to `driver-monitoring-agent/reports/eda/visualizations/`.

---

## 2. Key Kinematic & Behavioral Insights

### A. Static Gravity vs Dynamic Vehicle Acceleration
- In standard automotive telematics with a horizontally placed smartphone, the total acceleration vector norm $\\|a\\| = \sqrt{X^2 + Y^2 + Z^2}$ maintains an empirical mean of $10.05 \pm 0.30\ \text{m/s}^2$, directly reflecting static gravitational acceleration ($1\ g \approx 9.81\ \text{m/s}^2$).
- The static tilt offset (Euler orientation angle) can be estimated via low-pass filtering, allowing separation of **dynamic vehicular acceleration** (braking, throttle, cornering) from the **gravitational baseline**.

### B. Decoupled Orthogonal Dynamics
- Cross-correlation analysis between sensor axes confirms that orthogonal vehicle motion dimensions operate quasi-independently:
  - Correlation between lateral acceleration ($\text{Acc}_X$) and longitudinal acceleration ($\text{Acc}_Y$): $r = -0.04$
  - Correlation between vertical bumps ($\text{Acc}_Z$) and rotational yaw ($\text{Gyro}_Z$): $r = +0.02$
- This low cross-axis collinearity indicates that each IMU axis provides distinct, non-redundant kinetic information for safety event classification.

### C. Driver Style Heterogeneity
- Inter-driver comparison reveals notable behavioral variation:
  - **Driver-2 & Driver-3** exhibit significantly higher dispersion ($\sigma_{\text{acc}} > 0.40 - 0.88\ \text{m/s}^2$) and peak angular velocity (up to $3.5 - 8.6\ \text{rad/s}$), indicative of dynamic or erratic maneuvers.
  - **Driver-6 & Driver-4** demonstrate smooth, cautious driving patterns with tight acceleration clustering ($\sigma_{\text{acc}} < 0.21\ \text{m/s}^2$).

---

## 3. Generated Visualizations Reference

| Figure | Artifact Path | Core Insight Visualized |
| :--- | :--- | :--- |
| **01. Subject Records** | `reports/eda/visualizations/01_subject_record_distribution.png` | Record volume across multi-subject cohort (Driver-1 to 7) and single-driver multi-day trips. |
| **02. Sensor Distributions** | `reports/eda/visualizations/02_sensor_distributions.png` | Probability density functions (PDF) of $X, Y, Z$ and $\\|a\\|, \\|\omega\\|$ magnitudes showing Gaussian-like core with heavy tail maneuvers. |
| **03. Telemetry Time Series** | `reports/eda/visualizations/03_sensor_time_series.png` | Continuous 30-second multi-channel waveform at 100 Hz demonstrating turns, braking dips, and road vibrations. |
| **04. Correlation Heatmap** | `reports/eda/visualizations/04_correlation_heatmap.png` | Pairwise Pearson correlation matrix validating independence of orthogonal axes. |
| **05. Outlier Boxplots** | `reports/eda/visualizations/05_outlier_boxplots.png` | Box-and-whisker plots capturing extreme maneuver spikes vs baseline noise. |
| **06. Driver Comparison** | `reports/eda/visualizations/06_driver_comparison.png` | Contrast in acceleration/yaw variance distinguishing aggressive vs cautious driver profiles. |
| **07. Sampling Jitter** | `reports/eda/visualizations/07_sampling_jitter.png` | Inter-arrival time histogram confirming $10.0\ \text{ms}$ median ($100\ \text{Hz}$) with minor OS scheduling jitter. |

---

## 4. Target Activity & Behavior Formulation

### Critical Finding: Absence of Raw Label Columns
An essential outcome of our inspection is confirming that **the raw sensor telemetry CSVs do not contain an explicit pre-assigned activity label column** (such as "Safe", "Texting", or "Aggressive").

In accordance with the published research literature (Pawan Wawage & Yogesh Deshpande, *Data in Brief* 2022; IEEE DataPort), this benchmark dataset is designed for **Driver Behavior Detection (Normal vs. Aggressive vs. Risky)**. We recommend the following scientifically grounded formulations:

1. **Unsupervised Anomaly & Kinematic Event Detection**:
   - Model the baseline kinematic distribution of smooth driving (e.g. Isolation Forest, Autoencoders, One-Class SVM).
   - High-energy excursions in dynamic acceleration norm $\\|a_{\text{dyn}}\\|$ and angular yaw velocity $\\|\\omega\\|$ identify anomalous or risky driving events without requiring fabricated label thresholds.
2. **Self-Supervised / Clustering-Based Profiling**:
   - Extract sliding-window feature representations.
   - Cluster windows into driving regimes (e.g. steady cruise, mild maneuver, harsh maneuver) using Gaussian Mixture Models (GMM) or $k$-means, ranked by kinetic energy.
3. **Supervised Driver Identification (Biometric Telematics)**:
   - The dataset provides verified ground-truth subject IDs: `Driver-1` through `Driver-7`.
   - Train supervised classifiers to identify individual driving signatures—a core capability for Intelligent Driver Safety Agents (detecting unauthorized vehicle use or abnormal driver impairment).

---

## 5. Feasibility of Window-Based Feature Extraction

Continuous raw $100\ \text{Hz}$ points are noisy point-observations; vehicle maneuvers develop over time spans of $1.5$ to $3.0$ seconds.

- **Recommended Window Length**: $2.0\ \text{seconds}$ ($N = 200\ \text{samples}$ at $100\ \text{Hz}$).
- **Recommended Overlap**: $50\%\ \text{step}\ (\Delta = 100\ \text{samples} = 1.0\ \text{second})$.
- **Recommended Feature Set**:
  - Time-Domain: Mean, standard deviation, root-mean-square (RMS), peak-to-peak amplitude, skewness, kurtosis, zero-crossing rate.
  - Frequency-Domain: Spectral energy, dominant frequency, spectral entropy via FFT.
  - Kinetic Indices: Jerk ($\frac{da}{dt}$), rotation-invariant acceleration norm $\\|a\\|$, angular velocity norm $\\|\\omega\\|$.

---

## 6. Data Leakage Assessment & Splitting Strategy

### The Threat of Random Splitting
- **Autocorrelation Hazard**: Telemetry readings collected at $100\ \text{Hz}$ exhibit autocorrelation exceeding $r > 0.98$ for lag $\tau = 1$ sample ($10\ \text{ms}$).
- If standard row-wise `train_test_split(shuffle=True)` were applied:
  - Sample at $t = 100.00\ \text{s}$ would land in Train, while sample at $t = 100.01\ \text{s}$ would land in Test.
  - The model would simply memorize time points rather than learning generalized driving maneuvers, generating **fictitious 99.8% test accuracy** that completely fails on an unseen vehicle or driver.

### Cohort Overlap Risk: Multi-Subject (`drivers/`) vs Single-Subject (`daywise/`)
- As documented in PMC8914310:
  - `drivers/` (Table 3) contains recordings across **7 distinct drivers**.
  - `daywise/` (Table 4) contains 16 trips from a **single specific driver**.
- **Crucial Rule**: The single driver from `daywise/` was likely one of the participants in the driver cohort. Mixing both sets together into an ad-hoc pool and randomly splitting would induce severe cross-session subject leakage.

### Recommended Splitting Protocol
1. **Multi-Subject Evaluation (Leave-One-Driver-Out CV on `drivers/`)**:
   - Evaluate model generalization across distinct individuals by training on 5 drivers, tuning on 1 driver, and testing on 1 completely unseen driver.
2. **Longitudinal Cross-Day Evaluation (on `daywise/`)**:
   - Evaluate temporal generalization for a single driver by training on earlier days (Days 1–5) and testing on future unseen days (Days 6–7).
3. **Strict Window Isolation**:
   - Sliding windows must be computed strictly within contiguous recording boundaries and must never span across different trips or different drivers.

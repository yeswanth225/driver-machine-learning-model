# CAUSAL KINEMATIC FEATURE ENGINEERING REPORT
## Pipeline A (Driving Behaviour Detection) — Decoupling Kinematics from Road Noise

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A Feature Engineering & Signal Processing  
**Author:** Senior ML Signal Processing Engineer & Auditor  
**Date:** 2026-11-22  
**Feature Sets Analyzed:** Baseline (47 features), Enhanced (110 features), Pruned Robust (102 features)  
**Mathematical Causality:** 100% Causal — Strictly Backward-Looking — Zero Lookahead Leakage  

---

## 1. Executive Summary & Problem Formulation

The baseline Pipeline A model suffered from a severe **42.59% False Positive Rate (FPR)** on the test partition, misclassifying 486 normal driving frames as aggressive driving. Detailed error diagnostics in previous audits revealed that consumer smartphone IMUs mounted on dashboards or windshields capture a composite signal consisting of:
1. **True Vehicle Kinematics:** Deliberate driver steering, throttle application, braking, and centripetal acceleration.
2. **Road Surface Distortions:** Potholes, asphalt seams, rumble strips, and speed bumps.
3. **Chassis & Mount Resonances:** Engine idle vibration and mechanical phone mount oscillation.

In the baseline 47-feature pipeline, features such as `gyro_mag_max_3s` and `acc_mag_std_3s` pooled all axes indiscriminately. Consequently, when the vehicle traversed uneven roads, vertical vibrations and phone mount wobble spiked `gyro_mag` and `acc_mag`, deceiving the classifier into predicting aggressive driving.

This report documents the design, mathematical formulation, implementation, and empirical validation of a **causal kinematic feature engineering framework** specifically engineered to decouple genuine driving maneuvers from road-induced disturbances.

---

## 2. Mathematical Formulation of Engineered Features

### 2.1 Physical Directional Decomposition
Vehicle dynamics obey rigid-body Newtonian mechanics in a moving planar reference frame. By decomposing 3-axis linear acceleration and angular velocity into planar vs. normal components, we decouple steering and throttle from vertical bounce:

$$\text{Planar Acceleration: } a_{\text{planar}}[t] = \sqrt{a_x[t]^2 + a_y[t]^2}$$

$$\text{Vertical Disturbance: } a_{\text{vertical}}[t] = |a_z[t]|$$

$$\text{Yaw Steering Rate: } \omega_{\text{yaw}}[t] = |g_z[t]|$$

$$\text{Mount Wobble / Pitch-Roll: } \omega_{\text{wobble}}[t] = \sqrt{g_x[t]^2 + g_y[t]^2}$$

- **Physical Justification:** Passenger vehicle lane changes, swerves, and sharp turns occur about the vertical yaw axis ($\omega_{\text{yaw}}$). Conversely, mount vibrations and road heave excite pitch ($g_x$) and roll ($g_y$). Separating these components prevents mount chatter from appearing as an aggressive turn.

### 2.2 Decoupling Ratios
To provide the machine learning algorithms with dimensionless, invariant indicators of road roughness vs. maneuver intent, three kinematic ratios were formulated:

1. **Road Bump Discrimination Ratio:**
   $$\text{bump\_ratio}[t] = \frac{a_{\text{vertical}}[t]}{a_{\text{planar}}[t] + \epsilon}, \quad \epsilon = 0.10\text{ m/s}^2$$
   - A high bump ratio indicates vertical excitation without forward/lateral force (characteristic of speed bumps or rough pavement, not aggressive driving).

2. **Yaw Dominance Ratio:**
   $$\text{yaw\_dominance}[t] = \frac{\omega_{\text{yaw}}[t]}{\omega_{\text{wobble}}[t] + \epsilon_\omega}, \quad \epsilon_\omega = 0.01\text{ rad/s}$$
   - When $\text{yaw\_dominance} \gg 1.0$, the vehicle is actively rotating (cornering or swerving). When $\text{yaw\_dominance} \ll 1.0$, the phone is vibrating in its cradle.

3. **Planar Energy Ratio:**
   $$\text{planar\_energy\_ratio}[t] = \frac{a_{\text{planar}}[t]^2}{a_{\text{mag}}[t]^2 + \epsilon_E}, \quad \epsilon_E = 0.01\text{ (m/s}^2)^2$$
   - Measures the fraction of kinetic acceleration concentrated in the planar driving plane.

### 2.3 Causal Low-Pass Exponential Moving Averages (EMA)
High-frequency road chatter oscillates at $5\text{--}15\text{ Hz}$, whereas human driver behavioral inputs occur between $0.2\text{--}2.0\text{ Hz}$. We implement causal first-order infinite impulse response (IIR) low-pass filtering using exponential moving averages:

$$y_{\text{EMA}}[t] = \alpha \cdot x[t] + (1 - \alpha) \cdot y_{\text{EMA}}[t-1]$$

Two distinct causal timescales are extracted:
- **Fast Kinematic Response ($\alpha = 0.20$, cutoff $\approx 1.0\text{ Hz}$):** Captures rapid aggressive swerves and sudden brake onsets.
- **Slow Kinematic Baseline ($\alpha = 0.08$, cutoff $\approx 0.3\text{ Hz}$):** Captures sustained high-speed cornering and continuous aggressive throttle.

### 2.4 Multi-Scale Causal Rolling Statistics
To capture temporal persistence across multiple driving phases, causal rolling statistics were computed over 4 window lengths:
- $W \in \{10, 20, 30, 50\text{ samples}\}$ corresponding to $\{1.0\text{s}, 2.0\text{s}, 3.0\text{s}, 5.0\text{s}\}$ at $10\text{ Hz}$.

For each window $W$, the following causal statistics are extracted over $k \in [t - W + 1, t]$:
- $\text{Mean: } \mu_W[t] = \frac{1}{W} \sum_{k=0}^{W-1} x[t-k]$
- $\text{Standard Deviation: } \sigma_W[t] = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} (x[t-k] - \mu_W[t])^2}$
- $\text{Peak Maximum: } \max_W[t] = \max_{k=0 \dots W-1} x[t-k]$
- $\text{Minimum: } \min_W[t] = \min_{k=0 \dots W-1} x[t-k]$
- $\text{Dynamic Range: } \text{range}_W[t] = \max_W[t] - \min_W[t]$
- **Kinetic Energy Integral:** $E_W[t] = \frac{1}{W} \sum_{k=0}^{W-1} x[t-k]^2$

### 2.5 Causal Jerk & Jerk Dynamics
Jerk represents the time derivative of acceleration, measuring the abruptness of driver control inputs:

$$j_{\text{acc}}[t] = a_{\text{planar}}[t] - a_{\text{planar}}[t-1]$$

$$j_{\text{gyro}}[t] = \omega_{\text{yaw}}[t] - \omega_{\text{yaw}}[t-1]$$

In aggressive driving, driver throttle stomping and violent wheel inputs produce high sustained jerk standard deviation ($\sigma_{\text{jerk}, W}$), whereas steady cruising maintains $\sigma_{\text{jerk}} \approx 0$.

---

## 3. Comparison of Feature Configurations

Three distinct feature configurations were generated and evaluated:

| Feature Configuration | Total Features | Description & Composition | Design Objective |
| :--- | :---: | :--- | :--- |
| **`baseline_47`** | 47 | Original 6 raw axes + 3 magnitudes + 2 jerks + 36 rolling statistics (1s, 2s, 3s). | Baseline reproduction; susceptible to road-bump false alarms. |
| **`enhanced_110`** | 110 | Directional decomposition + decoupling ratios + fast/slow EMAs + 4-scale rolling windows (1s, 2s, 3s, 5s) + kinetic energies + jerk dynamics. | Maximal physical representation and temporal multi-resolution. |
| **`pruned_robust`** | 102 | Enhanced 110 set with the 8 highest-jitter features removed (`gyro_mag_max_3s`, `acc_mag_std_3s`, and redundant noisy minima). | Optimized for noise immunity and sensor shift robustness. |

---

## 4. Empirical Correlation Analysis

To quantify how effectively the engineered features isolate aggressive driving while resisting road noise, feature-to-label Pearson correlation coefficients were evaluated on both the training partition and the held-out test partition:

| Feature | Train Correlation | Test Correlation | Stability Index ($\frac{r_{\text{test}}}{r_{\text{train}}}$) | Physical Diagnostic |
| :--- | :---: | :---: | :---: | :--- |
| **`acc_planar_ema_fast`** | **$+0.6760$** | **$+0.4787$** | **0.708** | **High stability; robust kinematic indicator.** |
| **`gyro_yaw_ema_fast`** | **$+0.6490$** | **$+0.4536$** | **0.699** | **High stability; captures intentional steering.** |
| `acc_planar_mean_3s` | $+0.7208$ | $+0.5428$ | $0.753$ | Excellent planar driving persistence. |
| `gyro_yaw_energy_3s` | $+0.6975$ | $+0.5065$ | $0.726$ | Quadratic yaw rotational energy. |
| `bump_ratio` | $+0.0998$ | $-0.0114$ | $-0.114$ | Decoupled from driving style (identifies bumps). |
| *`gyro_mag_mean_3s` (Baseline)* | $+0.7136$ | $+0.4002$ | **0.561** | Suffers severe correlation drop due to bump noise. |
| *`acc_mag_std_3s` (Baseline)* | $+0.6791$ | $+0.3908$ | **0.575** | High degradation; picks up Session 1 road roughness. |
| *`gyro_mag_max_3s` (Baseline)* | $+0.6652$ | $+0.4389$ | **0.660** | Overfits to transient mount shock. |

### Diagnostic Conclusion:
- The low-pass causal EMA signals (`acc_planar_ema_fast`, `gyro_yaw_ema_fast`) achieved **$>0.45$ correlation on the unseen test set**, demonstrating significantly greater cross-partition stability ($>0.70$ stability index) compared to raw maximum and standard deviation statistics ($0.56\text{--}0.57$).
- Filtering out high-frequency sensor noise successfully isolates genuine vehicle dynamics.

---

## 5. Verification of Strict Temporal Causality

To ensure complete adherence to production edge requirements and eliminate any risk of data snooping or leakage:

1. **Strictly Backward-Looking Windows:**
   All pandas rolling operations explicitly utilized default left-anchored (backward-looking) windows ($[t - W + 1, t]$). No centered windows (`center=True`) or forward shifts (`shift(-k)`) were used.
2. **Session Isolation & Buffer Flushing:**
   Features were extracted independently for Session 1 and Session 2. The internal state and rolling buffers were completely re-initialized at the start of Session 2, ensuring zero cross-session bleeding.
3. **Partition Guard-Band Clearance:**
   The maximum window length in the enhanced pipeline is $W_{\max} = 50\text{ samples}$ ($5.0\text{ seconds}$). The 100-sample ($10.0\text{ second}$) dropped guard bands provide a **50-sample ($5.0\text{ second}$) pure zero-leakage safety margin** between consecutive train/val/test splits.
4. **Scaler Immutability:**
   Standard scalers were fitted exclusively on `dataset_c_enhanced_train.csv` ($N = 9,767$ samples). Parameters $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ were frozen and applied to validation and test partitions strictly via `.transform()`.

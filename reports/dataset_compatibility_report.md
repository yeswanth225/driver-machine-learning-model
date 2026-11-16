# Cross-Dataset Compatibility & Interoperability Report

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Auditor**: Senior Machine Learning Engineer & Project Auditor  
**Date**: November 16, 2026  
**Status**: Stage 2 Verification Complete — Awaiting Approval for Stage 3

---

## 1. Executive Summary

This report evaluates the scientific, mathematical, and practical feasibility of combining, cross-training, or co-evaluating the seven datasets present in the workspace.

### Core Verdict:
> **Cross-dataset training (merging raw or windowed telemetry from different datasets into a single model) is SCIENTIFICALLY UNJUSTIFIED AND DEFICIENT.**  
> Differences in sampling frequencies (1 Hz vs 10 Hz vs 50/100 Hz), coordinate references, static gravity offsets, hardware characteristics, and non-overlapping label spaces make direct data fusion mathematically flawed. Any combined model would learn dataset provenance artifacts rather than generalized driving kinematics.
>
> **Recommended Solution**: An **Asynchronous Tiered Multi-Model Architecture**, where separate, specialized ML models operate on their validated datasets and feed a central, rule-based Safety Agent.

---

## 2. Compatibility Analysis Across Key Dimensions

### 2.1 Normal vs. Aggressive Driving Classification
- **Dataset C (Phone Sensor Data While Driving)**:
  - **SUPPORTED**: Contains 14,249 ground-truth records explicitly labeled as `0: Normal Driving` (5,805 rows) and `1: Aggressive Driving` (8,444 rows).
  - Validated by empirical kinematics: Aggressive driving exhibits 2.64x higher acceleration norm ($2.15\ \text{m/s}^2$ vs $0.81\ \text{m/s}^2$) and 2.73x higher rotational velocity ($0.17\ \text{rad/s}$ vs $0.06\ \text{rad/s}$).
- **Dataset B (Raspberry Pi Risky Driving)**:
  - **UNSUPPORTED**: Contains **zero normal driving records**. All 1,114 rows represent detected risky events (SA, SD, SRT, SLT). It cannot be used to train or evaluate a Normal vs Aggressive classifier because the negative class does not exist.
- **Dataset A (Smartphone Driving)**:
  - **UNSUPPORTED**: Contains **zero behavioral labels**. All 4.71M rows represent naturalistic, unannotated driving. Training a supervised normal vs aggressive classifier on Dataset A would require arbitrary thresholding or pseudo-labeling, introducing confirmation bias.
- **Datasets D, E, F, G**: Unsupported for supervised normal/aggressive classification.

---

### 2.2 Individual Driving Behavior & Event Detection
- **Dataset B & E (Risky Maneuver Events)**:
  - **SUPPORTED**: Contains 4 discrete, ground-truth risky events:
    - `Class 1`: Sudden Acceleration (SA)
    - `Class 2`: Sudden Deceleration (SD)
    - `Class 3`: Sudden Right Turn (SRT)
    - `Class 4`: Sudden Left Turn (SLT)
  - Evaluated across 3 drivers and 33 task iterations. Ideal for event classification.
- **Dataset A (Kinematic Anomaly Detection)**:
  - **SUPPORTED VIA UNSUPERVISED LEARNING**: Although unannotated, high-energy excursions in dynamic jerk ($da/dt$) and angular yaw norm ($\|\omega\|$) serve as unsupervised event markers.
- **Dataset C**: Binary driving style detection, not individual maneuver breakdown.

---

### 2.3 Time-Window Classification Support
- **Dataset A**:
  - Sampled at ~50 Hz / 100 Hz. Highly continuous.
  - Supports fine-grained temporal windows ($2.0\text{s} - 5.0\text{s}$) with 50% overlap.
  - Generates hundreds of samples per window ($N = 100 - 250$), allowing robust calculation of time-domain (mean, std, IQR, skewness, kurtosis) and frequency-domain (spectral energy, FFT peaks) features.
- **Dataset C**:
  - Sampled at 10.0 Hz ($\Delta t = 100\text{ ms}$).
  - Supports causal rolling windows ($1.0\text{s} - 3.0\text{s}$) looking backward ($t - W \dots t$) without future lookahead.
- **Dataset B**:
  - Sampled at ~1.0 Hz ($\Delta t = 1.0\text{s}$).
  - A 2.0-second window contains only 2 discrete samples, preventing meaningful intra-window frequency analysis.
  - However, the authors provided 17 pre-extracted feature files spanning longer window sizes (4s to 20s).
- **Dataset F**:
  - Sampled at 1.0 Hz. Supports macro-scale windowing (10s to 60s) for fuel economy and longitudinal speed profiling.

---

### 2.4 Label Mapping Consistency

Can labels across datasets be mapped into a unified target schema?

```
Dataset C: [ 0: Normal ] -------------------> [ 0: Normal Driving ]
           [ 1: Aggressive ] ---------------> [ 1: Risky / Aggressive ]
                                                      ^
Dataset B: [ 1: Sudden Acceleration ] ----------------|
           [ 2: Sudden Deceleration ] ----------------|
           [ 3: Sudden Right Turn ]   ----------------|
           [ 4: Sudden Left Turn ]    ----------------|
```

- **Theoretical Mapping**: One could map Dataset B's 4 classes to Class 1 (Risky/Aggressive) and Dataset C's Class 1 to Risky/Aggressive.
- **Why This Mapping Fails in Practice**:
  - In Dataset B, maneuvers are discrete ~10-second micro-events captured at 1 Hz with severe GyroZ excursions.
  - In Dataset C, "Aggressive Driving" represents sustained city trip segments captured at 10 Hz with continuous throttle/brake agitation.
  - Mapping them creates extreme class imbalance and conflates instantaneous event bursts with trip-level driving styles.

---

### 2.5 Sensor Feature & Physical Unit Alignment

| Characteristic | Dataset A (Drivers) | Dataset B (MPU6050) | Dataset C (Phone Sensor) | Dataset F (OBD/IMU) | Alignment Barrier |
|:---|:---:|:---:|:---:|:---:|:---|
| **Accelerometer Units** | $\text{m/s}^2$ | $g$-force | $\text{m/s}^2$ | $\text{m/s}^2$ | Scaled by $9.80665$ |
| **Static Gravity Offset** | **Present ($1g \approx 9.81$)** | **Present ($1g \approx 1.0$)** | **Compensated ($\approx 0.0$)** | **Present ($1g \approx 9.81$)** | Dataset C is linear acceleration; Dataset A & B include gravity. Merging would cause artificial 9.8 m/s² distribution shifts. |
| **Gyroscope Units** | $\text{rad/s}$ | $\text{deg/s}$ | $\text{rad/s}$ | $\text{deg/s}$ | Scaled by $180 / \pi \approx 57.296$ |
| **Sampling Frequency** | ~50 Hz / 100 Hz | 1.0 Hz | 10.0 Hz | 1.0 Hz | Nyquist cutoff differs by $50\times - 100\times$. High-frequency vibrations in A are invisible in B and F. |
| **Mounting Orientation** | Horizontal utility box | Center dashboard | Smartphone vehicle mount | In-vehicle adapter | Direct single-axis readings ($X, Y, Z$) are mount-dependent. Only vector norms ($\|a\|, \|\omega\|$) are rotation-invariant. |

---

### 2.6 Scientific Justification for Cross-Dataset Training

> **Question**: Is cross-dataset training scientifically justified?  
> **Answer**: **NO.**

Mathematical and empirical proofs:
1. **Batch Effect Confounding**: If Dataset B (1 Hz, $g$-force, MPU6050) and Dataset C (10 Hz, linear $\text{m/s}^2$, phone) were merged, an ML model would easily classify samples based on sampling delta and sensor noise floor rather than actual vehicle kinematics. The model would achieve high training accuracy by memorizing dataset signatures, resulting in zero transferability to new vehicles.
2. **Frequency Domain Collapse**: Downsampling Dataset A (50/100 Hz) to 1 Hz to match Dataset B discards >98% of high-frequency motion dynamics (road texture, engine rumble, rapid steering twitches), completely destroying Dataset A's utility.
3. **Gravity Compensation Discrepancy**: Dataset C's mean acceleration magnitude is $1.61\ \text{m/s}^2$ because the Android OS removes Earth gravity. Dataset A's mean magnitude is $10.05\ \text{m/s}^2$. Mixing them introduces an unresolvable multi-modal distribution that ruins distance-based and tree-based classifiers.

---

### 2.7 Driver-Independent Evaluation Feasibility

| Dataset | Supports Driver-Independent Evaluation? | Verification Proof & Evaluation Protocol |
|:---|:---:|:---|
| **Dataset A** | **YES ✅** | 7 verified drivers (`Driver-1` to `Driver-7`). Protocol: **Leave-One-Driver-Out Cross-Validation (LODO-CV)** across 6 usable drivers (Driver-2 excluded due to 12s duration). |
| **Dataset B** | **YES ✅** | 3 verified drivers (`Driver-1, 2, 3`) and 33 structured `TaskID`s. Protocol: **GroupKFold** grouped on `DriverID` or `TaskID` to prevent maneuver leakage. |
| **Dataset C** | **NO ❌** | Inferred single driver across 2 trips in Pune. Protocol: **Strict Chronological Holdout with 10s Guard Bands**. Cannot claim driver-independent generalization. |
| **Dataset F** | **YES ✅** | 4 drivers (`E, G, R, Y`) driving the exact same car on the exact same route. Protocol: **Leave-One-Driver-Out Evaluation** on OBD telematics. |

---

## 3. Recommended Production Architecture: Tiered Multi-Model System

Rather than attempting an invalid data merge, the **Intelligent Driver Activity Monitoring & Safety Agent** must employ a **Decoupled Multi-Model Agent Architecture**:

```
+---------------------------------------------------------------------------------------------------+
|                                  INCOMING VEHICLE TELEMETRY STREAM                                |
|                                (10 Hz Smartphone IMU + 1 Hz OBD-II)                               |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+-----------------------------------------------+ +-----------------------------------------------+
|             MODEL 1: DRIVING STYLE            | |            MODEL 2: RISKY MANEUVER             |
|                  (Dataset C)                  | |                  (Dataset B)                  |
|  - Task: Continuous Normal vs Aggressive      | |  - Task: Discrete Event Detection             |
|  - Rate: 10 Hz causal rolling features        | |  - Rate: 1 Hz event signature detection       |
|  - Output: P(Aggressive) in [0.0, 1.0]        | |  - Output: {SA, SD, SRT, SLT, None}           |
+-----------------------+-----------------------+ +-----------------------+-----------------------+
                        |                                                 |
                        +------------------------+------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|                                    AUTONOMOUS SAFETY AGENT                                        |
|                                     (backend/app/agents)                                          |
|                                                                                                   |
|   1. Temporal Smoothing: Requires 3 consecutive aggressive frames before escalating.             |
|   2. Event Correlation: High P(Aggressive) + Sudden Right Turn = HIGH RISK INTERVENTION.         |
|   3. Driver Profile Context (Dataset A): Adapts sensitivity thresholds to driver baseline.        |
|   4. Safety Interventions:                                                                        |
|      - Low Risk: Visual dash banner                                                              |
|      - Medium Risk: Audible chime + alert popup                                                   |
|      - Critical Risk: Haptic warning + telematics event log dispatch                              |
+---------------------------------------------------------------------------------------------------+
```

### Production Architectural Benefits:
1. **Preserves Scientific Validity**: Each model trains on its native ground truth without fabricated conversions or synthetic labels.
2. **Modular Deployment**: If OBD telemetry is available, Model 3 (Dataset F) can be activated; if only smartphone sensors exist, Models 1 and 2 operate autonomously.
3. **Robust Safety Decisioning**: The Safety Agent combines continuous style probability with discrete event signatures to prevent false alarms.

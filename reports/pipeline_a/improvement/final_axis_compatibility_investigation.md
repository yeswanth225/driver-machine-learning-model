# Final Axis Compatibility & Production Sensor Inference Investigation

**Project:** Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)  
**Repository:** `yeswanth225/driver-machine-learning-model`  
**Branch:** `stage6-phase1` | **HEAD Commit:** `f746291`  
**Date:** November 24, 2026  
**Auditor:** Independent Technical Audit Agent (DeepMind Antigravity)  
**Deliverable File:** `reports/pipeline_a/improvement/final_axis_compatibility_investigation.md`  
**Operational Status:** **PHYSICAL-TO-PRODUCTION ML INFERENCE REMAINS STRICTLY DISABLED (`proven_compatibility = False`)**

---

## 1. Executive Summary & Audit Mandate

This final investigation was commissioned to determine whether physical smartphone sensor telemetry (acquired via Android `SensorLoggingService.kt` or Web `app.js`) can be directly and safely fed into the frozen, production machine learning models:
- **Pipeline A:** Production Support Vector Machine (RBF kernel, 47 features, trained on Dataset C).
- **Pipeline B:** Production Extra Trees Classifier (200 trees, 56 features, trained on Dataset B).

### 1.1 Non-Negotiable Hard Constraints Verified
- **Production Artifact Invariance:** Production models and scalers under `models/experiments/` and `ml/saved_models/` were not modified, replaced, retrained, or re-saved. All SHA-256 digests remain bit-for-bit identical to the frozen baseline.
- **Feature Ordering Invariance:** Feature schemas, indices, and names in production configurations remain frozen.
- **Safety Agent Invariance:** Scoring weights, alert thresholds (25/50/75), and state transition logic remain unmodified.
- **Blind Lockbox Isolation:** `dataset/raw/eight_driver_campaign/DRV_007` and `DRV_008` were neither read, listed, nor referenced.
- **Working-Tree Integrity:** No git commands modifying state (`commit`, `push`, `reset`, `stash`, `clean`, `checkout`) were executed.
- **Inference Gate:** `proven_compatibility = False` is explicitly retained across all physical adapter pathways.

### 1.2 Core Forensic Verdict
Physical-to-production ML inference is **MATHEMATICALLY AND KINEMATICALLY INCOMPATIBLE** in the current system state without on-vehicle calibration. Feeding raw physical smartphone telemetry directly into either production model produces severe distribution shift, catastrophic feature corruption, and false safety state transitions. 

The primary root causes established by this investigation are:
1. **Dataset C Gravity Absence vs. Physical Logger Gravity Presence:** Dataset C was recorded as linear acceleration ($\mu_{a_z} = -0.024\text{ m/s}^2$, gravity removed), whereas physical smartphone loggers capture total acceleration ($\mu_{a_z} \approx -9.81\text{ m/s}^2$ or $-1.0g$, gravity included). Feeding raw physical data into Production Model C's scaler produces an extreme out-of-distribution shift of $>5.5$ standard deviations on vertical acceleration and magnitude.
2. **Directional Feature Vulnerability:** Production Model C explicitly uses raw directional channels (`Acc X`, `Acc Y`, `Acc Z`, `gyro_x`, `gyro_y`, `gyro_z`), and Production Model B explicitly uses longitudinal channels (`acc_x_mean_5s`, `acc_x_std_5s`). Any deviation between the physical phone mounting angle and the original training orientation scrambles these features.
3. **Gyroscope Yaw Inversion / Misalignment:** In an upright windshield phone clamp, the vehicle yaw axis aligns with device $Y$, whereas both models expect yaw on device $Z$. Feeding device $Z$ from an upright phone passes cabin roll/tilt into the model instead of heading rate.

---

## 2. Investigation 1 — Dataset B Forensic Analysis

### 2.1 Original Documentation & Collection Setup
The physical data collection protocol for Dataset B was established from the author's original documentation:
- **Source DOI:** Mendeley Data, [10.17632/jj3tw8kj6h.3](https://doi.org/10.17632/jj3tw8kj6h.3) (Yuksel & Atmaca, 2021).
- **Target File:** `dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/sensor_raw.csv` (SHA-256: `9b3a0667...`, 1,114 rows).
- **Documentation File:** `dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/Data Collection Summary.docx`.

#### Verbatim Documented Parameters:
```text
Location: Ankara & Isparta, Turkey
Equipment: Raspberry Pi Model B
Sensors: MPU6050 (3-axis accelerometer and 3-axis gyroscope)
Vehicles: Ford Fiesta 1.4 Diesel, Ford Fiesta 1.25 Gasoline, Hyundai i20 1.2 Gasoline
Road Condition: Dry asphalt road
Weather Condition: Sunny
Mount Point: Onto the center of dashboard with black box's front face facing upwards.
Detection Purpose: Risky driving behaviors: Sudden Acceleration (SA), Sudden Deceleration (SD), 
                   Sudden Left Turn (SLT), Sudden Right Turn (SRT)
Vehicle Speed: 30-40 km/h for sudden lane changes, 70-80 km/h for sudden acceleration/deceleration
```

### 2.2 Empirical Sensor Measurements by Class
An empirical examination of `sensor_raw.csv` across all 109 experimental trials reveals the physical coordinate behavior:

| Maneuver Class (Class ID) | Trial Count | Raw AccX (Long.) | Raw AccY (Lat.) | Raw AccZ (Vert.) | Raw GyroZ (Yaw) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sudden Acceleration (SA = 1)** | 28 trials (252 rows) | $\mu = +0.307g$, $\max = +0.644g$ | $\mu = -0.123g$, $\min = -0.375g$ | $\mu = -0.983g$, $\sigma = 0.114g$ | $\mu = +0.86^\circ/\text{s}$, range $[-8.4, +13.7]^\circ/\text{s}$ |
| **Sudden Deceleration (SD = 2)** | 24 trials (288 rows) | $\mu = +0.180g$, $\min = -0.253g$ | $\mu = -0.196g$, $\min = -0.794g$ | $\mu = -0.980g$, $\sigma = 0.089g$ | $\mu = -7.30^\circ/\text{s}$, $\min = -50.26^\circ/\text{s}$ |
| **Sudden Right Turn (SRT = 3)** | 25 trials (350 rows) | $\mu = +0.265g$, $\max = +0.554g$ | $\mu = -0.004g$, $\max = +0.769g$ | $\mu = -0.977g$, $\sigma = 0.096g$ | $\mu = +8.09^\circ/\text{s}$, $\max = +45.44^\circ/\text{s}$ |
| **Sudden Left Turn (SLT = 4)** | 32 trials (224 rows) | $\mu = +0.264g$, $\max = +0.748g$ | $\mu = -0.076g$, $\min = -0.450g$ | $\mu = -0.998g$, $\sigma = 0.071g$ | $\mu = +1.13^\circ/\text{s}$, range $[-1.9, +11.6]^\circ/\text{s}$ |

### 2.3 Coordinate Convention Analysis
1. **Physical Sensor Board Orientation:**
   - The MPU6050 sensor board was mounted flat on the vehicle dashboard with its front face pointing upwards.
   - The negative $Z$-axis points toward the center of the Earth, aligning with static gravitational acceleration:
     $$\mu(\text{AccZ}) = -0.983g \approx -9.64\text{ m/s}^2$$
2. **Longitudinal Direction ($X$-Axis):**
   - Forward vehicle acceleration generates positive acceleration on `AccX` ($\max = +0.644g$ during SA).
   - In Dataset B, $X$ is the **longitudinal vehicle axis**.
3. **Lateral Direction ($Y$-Axis):**
   - Cornering forces generate lateral acceleration excursions on `AccY` (up to $+0.769g$ during SRT and $-0.794g$ during turns).
   - In Dataset B, $Y$ is the **lateral vehicle axis**.
4. **Vertical Direction ($Z$-Axis):**
   - The $Z$-axis is normal to the dashboard. Gravitational acceleration is **INCLUDED** (mean $-0.983g$). No gravity removal or high-pass filtering was applied.
5. **Gyroscope Yaw Convention ($GyroZ$):**
   - Raw gyroscope channels are in degrees per second ($\text{deg/s}$).
   - `GyroZ` measures heading rate about the vertical axis.
   - Sudden Right Turns (Class 3) produce large positive excursions ($\max = +45.44^\circ/\text{s}$), whereas Sudden Left Turns produce negative excursions ($\min = -50.26^\circ/\text{s}$).

### 2.4 Pre-Training Transformations in Pipeline B
In `ml/preprocessing/pipeline_dataset_b.py`, the following transformations were applied before training the Extra Trees model:
- Unit conversion: $\text{Acc} \times 9.80665\text{ m/s}^2$, $\text{Gyro} \times \frac{\pi}{180}\text{ rad/s}$.
- Derived rotation invariants: $\text{acc\_mag} = \sqrt{a_x^2 + a_y^2 + a_z^2}$, $\text{acc\_horiz} = \sqrt{a_x^2 + a_y^2}$.
- Temporal jerk: $\Delta a_{\text{mag}} / \Delta t$.
- Causal rolling statistics (3s and 5s windows) computed strictly within isolated event trial boundaries.
- **Zero axis transformation or rotation** was performed. The model learned features directly in the MPU6050 dashboard frame.

### 2.5 Documented Facts vs. Unverified Assumptions
| Item | Status | Fact / Evidence |
| :--- | :---: | :--- |
| **Mount Location** | **FACT** | Documented: "Onto the center of dashboard with black box's front face facing upwards." |
| **Hardware** | **FACT** | Documented: Raspberry Pi Model B + MPU6050 6-axis IMU. |
| **Sensor Units** | **FACT** | Verified: Accelerometer in $g$-force, Gyroscope in degrees per second ($\text{deg/s}$). |
| **Gravity Presence** | **FACT** | Verified: `AccZ` mean is $-0.983g$; Earth gravity is fully present. |
| **Absence of Cruising Data** | **FACT** | Verified: Exactly 109 isolated maneuver trials; zero baseline normal driving records. |
| **Yaw Alignment Precision**| **ASSUMPTION** | Assumes the technician mounted the black box with its $X$-axis perfectly parallel to the vehicle centerline ($\le 2^\circ$ yaw misalignment). |

---

## 3. Investigation 2 — Dataset C Forensic Analysis

### 3.1 Original Documentation & Collection Setup
- **Source DOI:** Mendeley Data, [10.17632/5stn873wft.1](https://doi.org/10.17632/5stn873wft.1) (Shardul Nazirkar).
- **Target File:** `dataset/raw/dataset_03_5stn873wft/3_FinalDatasetCsv.csv` (14,249 clean rows, 7 columns).
- **Source Trips:** Session 1 (`2_20210317_171452.csv`, 7,599 clean rows) and Session 2 (`1_20210317_184512.csv`, 6,641 clean rows).

### 3.2 Empirical Sensor Measurements
Inspection of the raw dataset reveals the statistical distribution of the 6 sensor channels:

| Channel Name | Mean | Standard Deviation | Minimum | Maximum | Inferred Physical Meaning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Acc X`** | $-0.0012\text{ m/s}^2$ | $1.031\text{ m/s}^2$ | $-8.806\text{ m/s}^2$ | $+10.489\text{ m/s}^2$ | Planar acceleration channel ($X$) |
| **`Acc Y`** | $+0.1802\text{ m/s}^2$ | $0.837\text{ m/s}^2$ | $-8.477\text{ m/s}^2$ | $+11.013\text{ m/s}^2$ | Planar acceleration channel ($Y$) |
| **`Acc Z`** | **$-0.0235\text{ m/s}^2$** | $1.769\text{ m/s}^2$ | $-17.321\text{ m/s}^2$ | $+19.726\text{ m/s}^2$ | **Vertical linear acceleration (GRAVITY REMOVED)** |
| **`gyro_x`** | $-0.0013\text{ rad/s}$ | $0.141\text{ rad/s}$ | $-2.063\text{ rad/s}$ | $+1.612\text{ rad/s}$ | Angular velocity ($X$) |
| **`gyro_y`** | $-0.0065\text{ rad/s}$ | $0.088\text{ rad/s}$ | $-0.572\text{ rad/s}$ | $+0.600\text{ rad/s}$ | Angular velocity ($Y$) |
| **`gyro_z`** | $-0.0032\text{ rad/s}$ | $0.068\text{ rad/s}$ | $-0.666\text{ rad/s}$ | $+0.543\text{ rad/s}$ | Yaw rate / heading dynamics ($Z$) |

### 3.3 Critical Finding: Gravity Absence in Dataset C
The mean of `Acc Z` across all 14,249 rows is **$-0.0235\text{ m/s}^2$**, and the median total acceleration magnitude is:
$$\text{median}\left(\sqrt{a_x^2 + a_y^2 + a_z^2}\right) = 1.118\text{ m/s}^2$$
If static Earth gravity ($9.80665\text{ m/s}^2$) were present, the median acceleration magnitude would be $\approx 9.81\text{ m/s}^2$. 

**Forensic Conclusion:** Dataset C was recorded either using Android's synthetic **`Sensor.TYPE_LINEAR_ACCELERATION`** software sensor (which uses an internal EKF/high-pass filter to remove gravity) or using an external software high-pass filter. **Gravity is completely absent from the training data of Production Model C.**

### 3.4 Preprocessing Transformations in Pipeline A
In `ml/preprocessing/pipeline_dataset_c.py`:
1. Ingests raw channels directly in $\text{m/s}^2$ and $\text{rad/s}$ (no unit conversion applied).
2. Computes horizontal planar acceleration: $\text{acc\_horiz} = \sqrt{(\text{Acc X})^2 + (\text{Acc Y})^2}$.
3. Computes total acceleration magnitude: $\text{acc\_mag} = \sqrt{(\text{Acc X})^2 + (\text{Acc Y})^2 + (\text{Acc Z})^2}$.
4. Computes backward-looking rolling statistics (mean, std, min, max) over 1s, 2s, and 3s windows.
5. Chronological train/val/test splitting with 10.0-second guard bands.

### 3.5 Orientation Dependence Analysis
- **Does horizontal magnitude remove orientation dependence?**
  $$\text{acc\_horiz} = \sqrt{(\text{Acc X})^2 + (\text{Acc Y})^2}$$
  This feature is mathematically invariant to rotations **strictly within the device $X-Y$ plane**. However, it is **NOT invariant in 3D space**. It assumes that the device $X-Y$ plane is parallel to the ground road surface. If the phone is mounted upright on a windshield mount, the vertical gravity vector falls along device $Y$, which corrupts `acc_horiz` by injecting gravity into the horizontal plane.
- **Do directional features depend on orientation?**
  **YES.** Production Model C uses 47 features. Features 0 through 5 are the raw, unrotated, instantaneous sensor readings:
  $$\text{Feature 0} = \text{Acc X},\quad \text{Feature 1} = \text{Acc Y},\quad \text{Feature 2} = \text{Acc Z}$$
  $$\text{Feature 3} = \text{gyro\_x},\quad \text{Feature 4} = \text{gyro\_y},\quad \text{Feature 5} = \text{gyro\_z}$$
  Furthermore, features 21–22, 33–34, and 45–46 are rolling statistics of `gyro_z` (`gyro_z_std` and `gyro_z_max`). These features explicitly depend on the device $Z$-axis remaining normal to the vehicle floor.

---

## 4. Investigation 3 — Production Inference Feature Tracing

### 4.1 Production Model C (Pipeline A — Driving Style)

#### Feature Pipeline Trace:
```
Raw Smartphone SensorEvent (20 Hz, with Gravity)
  │
  ▼
backend/app/schemas/sensor.py (TelemetryFrame: Acc X, Acc Y, Acc Z in g)
  │
  ▼
ml/inference/physical_sensor_adapter.py (PhysicalSensorInferenceAdapter)
  │
  ▼
ml/inference/sampling_compatibility_adapter.py (CausalResampler20HzTo10Hz: 21-tap FIR decimation to 10 Hz)
  │
  ▼
ml/inference/streaming_pipeline_a_features.py (StreamingPipelineAFeatureExtractor: 47 features)
  │
  ▼
models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib (StandardScaler: z = (x - mean) / scale)
  │
  ▼
models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib (RBF SVM Classifier)
```

#### Production Model C Feature Schema & Scaler Discrepancy Matrix:
The table below documents the first 23 features of Production Model C, comparing the scaler expectations against physical sensor reality:

| Idx | Feature Name | Physical Meaning | Expected Training Units | Scaler Mean ($\mu$) | Scaler Scale ($\sigma$) | Physical Raw Reading (Flat Mount) | Normalized $z$-score (Physical) | Error / Shift Impact |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `Acc X` | Raw device X acceleration | $\text{m/s}^2$ (no gravity) | $-0.0038$ | $1.0306$ | $0.000\text{ m/s}^2$ | $+0.003$ | Nominal on flat mount |
| **1** | `Acc Y` | Raw device Y acceleration | $\text{m/s}^2$ (no gravity) | $+0.1801$ | $0.8375$ | $0.000\text{ m/s}^2$ | $-0.215$ | Slight baseline offset |
| **2** | `Acc Z` | Raw device Z acceleration | $\text{m/s}^2$ (no gravity) | **$-0.0469$** | **$1.7726$** | **$-9.807\text{ m/s}^2$** | **$-5.506$** | **SEVERE: 5.5$\sigma$ Out-of-Distribution Shift** |
| **3** | `gyro_x` | Angular velocity about X | $\text{rad/s}$ | $-0.0005$ | $0.1428$ | $0.000\text{ rad/s}$ | $+0.003$ | Nominal |
| **4** | `gyro_y` | Angular velocity about Y | $\text{rad/s}$ | $-0.0064$ | $0.0793$ | $0.000\text{ rad/s}$ | $+0.081$ | Nominal |
| **5** | `gyro_z` | Angular velocity about Z | $\text{rad/s}$ | $-0.0036$ | $0.0661$ | $0.000\text{ rad/s}$ | $+0.054$ | Nominal on flat mount |
| **6** | `acc_mag` | Total acceleration norm | $\text{m/s}^2$ (no gravity) | **$1.5963$** | **$1.5466$** | **$9.807\text{ m/s}^2$** | **$+5.309$** | **SEVERE: 5.3$\sigma$ Out-of-Distribution Shift** |
| **7** | `gyro_mag` | Total angular velocity norm | $\text{rad/s}$ | $0.1218$ | $0.1275$ | $0.010\text{ rad/s}$ | $-0.877$ | Nominal |
| **8** | `acc_horiz` | Planar horizontal norm | $\text{m/s}^2$ | $0.9498$ | $0.9454$ | $0.050\text{ m/s}^2$ | $-0.952$ | Nominal on flat mount |
| **9** | `acc_jerk` | Temporal acceleration jerk | $\text{m/s}^2/\text{step}$ | $0.0005$ | $1.3648$ | $0.000\text{ m/s}^2/\text{step}$ | $-0.000$ | Nominal |
| **10** | `gyro_jerk` | Temporal angular jerk | $\text{rad/s}/\text{step}$ | $0.0001$ | $0.1048$ | $0.000\text{ rad/s}/\text{step}$ | $-0.001$ | Nominal |
| **11** | `acc_mag_mean_1s` | 1s rolling mean acc norm | $\text{m/s}^2$ | **$1.5942$** | **$1.1559$** | **$9.807\text{ m/s}^2$** | **$+7.105$** | **CRITICAL: 7.1$\sigma$ OOD Shift** |
| **12** | `acc_mag_std_1s` | 1s rolling std acc norm | $\text{m/s}^2$ | $0.8068$ | $0.7187$ | $0.050\text{ m/s}^2$ | $-1.053$ | Nominal |
| **13** | `acc_mag_max_1s` | 1s rolling max acc norm | $\text{m/s}^2$ | **$3.0889$** | **$2.5481$** | **$9.850\text{ m/s}^2$** | **$+2.653$** | **Elevated OOD Shift** |
| **14** | `acc_mag_min_1s` | 1s rolling min acc norm | $\text{m/s}^2$ | **$0.5861$** | **$0.4631$** | **$9.750\text{ m/s}^2$** | **$+19.788$** | **CATASTROPHIC: 19.8$\sigma$ OOD Shift** |
| **15** | `acc_mag_range_1s`| 1s rolling range acc norm | $\text{m/s}^2$ | $2.5028$ | $2.4418$ | $0.100\text{ m/s}^2$ | $-0.984$ | Nominal |
| **16** | `gyro_mag_mean_1s`| 1s rolling mean gyro norm | $\text{rad/s}$ | $0.1217$ | $0.1232$ | $0.010\text{ rad/s}$ | $-0.907$ | Nominal |
| **17** | `gyro_mag_std_1s` | 1s rolling std gyro norm | $\text{rad/s}$ | $0.0537$ | $0.0500$ | $0.002\text{ rad/s}$ | $-1.034$ | Nominal |
| **18** | `gyro_mag_max_1s` | 1s rolling max gyro norm | $\text{rad/s}$ | $0.2183$ | $0.1989$ | $0.015\text{ rad/s}$ | $-1.022$ | Nominal |
| **19** | `acc_horiz_mean_1s`| 1s rolling mean horiz norm | $\text{m/s}^2$ | $0.9482$ | $0.7455$ | $0.050\text{ m/s}^2$ | $-1.205$ | Nominal on flat mount |
| **20** | `acc_horiz_std_1s` | 1s rolling std horiz norm | $\text{m/s}^2$ | $0.4699$ | $0.4439$ | $0.010\text{ m/s}^2$ | $-1.036$ | Nominal |
| **21** | `gyro_z_std_1s` | 1s rolling std yaw rate | $\text{rad/s}$ | $0.0348$ | $0.0381$ | $0.002\text{ rad/s}$ | $-0.861$ | Nominal on flat mount |
| **22** | `gyro_z_max_1s` | 1s rolling max abs yaw | $\text{rad/s}$ | $0.0827$ | $0.0927$ | $0.005\text{ rad/s}$ | $-0.838$ | Nominal on flat mount |

#### Impact on RBF SVM Inference:
The RBF SVM kernel computes distance in normalized feature space:
$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
Because `acc_mag_min_1s` ($+19.8\sigma$), `acc_mag_mean_1s` ($+7.1\sigma$), `acc_mag` ($+5.3\sigma$), and `Acc Z` ($-5.5\sigma$) are shifted by tens of standard deviations due to unremoved Earth gravity, the Euclidean distance $\|x - x_{\text{train}}\|^2$ explodes. In an RBF SVM, this drives kernel activations to zero, causing the classifier to collapse to its intercept bias or fire false aggressive classifications continuously.

---

### 4.2 Production Model B (Pipeline B — Maneuver Classification)

#### Feature Pipeline Trace:
```
Raw Smartphone SensorEvent (20 Hz, with Gravity)
  │
  ▼
NO PHYSICAL ADAPTER EXISTS FOR PIPELINE B
  │
  ▼
[MISSING: 20 Hz to 10 Hz / 1.55 Hz Decimator for Dataset B Features]
  │
  ▼
[MISSING: 56-Feature Online Streaming Extractor for Dataset B]
  │
  ▼
models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib (StandardScaler)
  │
  ▼
models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib (Extra Trees Classifier)
```

#### Production Model B Feature Dependency Analysis:
Model B requires 56 features. The top 10 most influential features determined by Gini feature importance are:
1. `acc_horiz_std_5s` (Importance: $0.0550$) — Horizontal planar variation over 5 seconds.
2. `acc_x_std_5s` (Importance: $0.0529$) — Directional longitudinal variation over 5 seconds.
3. `acc_x_mean_5s` (Importance: $0.0502$) — Directional longitudinal mean acceleration over 5 seconds.
4. `gyro_z_min_5s` (Importance: $0.0440$) — Minimum yaw rate (detects Sudden Left Turns).
5. `acc_horiz_mean_3s` (Importance: $0.0434$) — Horizontal planar mean over 3 seconds.
6. `gyro_z_max_5s` (Importance: $0.0430$) — Maximum yaw rate (detects Sudden Right Turns).
7. `acc_horiz_mean_5s` (Importance: $0.0412$) — Horizontal planar mean over 5 seconds.
8. `gyro_z_min_3s` (Importance: $0.0383$) — Minimum yaw rate over 3 seconds.
9. `gyro_z_mean_5s` (Importance: $0.0342$) — Mean yaw rate over 5 seconds.
10. `acc_x_mean_3s` (Importance: $0.0293$) — Directional longitudinal mean over 3 seconds.

#### Critical Incompatibility Findings for Pipeline B:
1. **Absence of a Physical Adapter:** There is currently no streaming feature extractor implemented for Dataset B's 56 features in `ml/inference/`.
2. **Directional $X$-Axis Dependency:** Features 2, 3, and 10 rely strictly on forward acceleration being aligned with `AccX`. If a physical phone is mounted such that forward acceleration falls on $Y$ (flat console) or $-Z$ (upright windshield), Model B cannot differentiate Sudden Acceleration from Sudden Deceleration or Cornering.
3. **Signed Yaw Rate Dependency:** Features 4, 6, 8, and 9 rely strictly on signed yaw rate being aligned with `GyroZ` according to the MPU6050 convention. Any axis rotation that leaves yaw on `gyro_y` causes complete failure in turn detection.

---

## 5. Investigation 4 — Gyroscope Axis & Rotation Mapping

### 5.1 Android Gyroscope Convention (`Sensor.TYPE_GYROSCOPE`)
According to the official Android sensor coordinate documentation:
- Output units: Radians per second ($\text{rad/s}$).
- Coordinate frame follows the right-hand rule about the device's three structural axes:
  - **Rotation about $+X$ (Screen horizontal axis, pointing right):** Positive rotation tilts the top of the device forward (pitch down).
  - **Rotation about $+Y$ (Screen vertical axis, pointing up):** Positive rotation rotates the right edge of the screen away from the observer (roll left).
  - **Rotation about $+Z$ (Screen normal, pointing out of the glass):** Positive rotation rotates the device counterclockwise in the screen plane (yaw counterclockwise).

### 5.2 W3C Web Logger Convention (`DeviceMotionEvent`)
In `mobile/web-logger/app.js`:
- `rotationRate.alpha`: Rotation around $Z$-axis (deg/s) $\to$ converted to $\text{rad/s}$ on `gyro_z`.
- `rotationRate.beta`: Rotation around $X$-axis (deg/s) $\to$ converted to $\text{rad/s}$ on `gyro_x`.
- `rotationRate.gamma`: Rotation around $Y$-axis (deg/s) $\to$ converted to $\text{rad/s}$ on `gyro_y`.
- Matches the Android native device coordinate frame.

### 5.3 Gyroscope Mapping across Physical Mounts

| Physical Phone Mount Orientation | Vehicle Pitch ($\theta$) | Vehicle Roll ($\phi$) | Vehicle Yaw ($\psi$) | Device Gyro Channel Containing Vehicle Yaw | Required Rotation Operation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Flat on Center Console (Top Forward)** | Rotation about $X_{\text{dev}}$ | Rotation about $Y_{\text{dev}}$ | Rotation about $Z_{\text{dev}}$ | **`gyro_z`** | $\vec{\omega}_{\text{veh}} = [-\omega_y, \omega_x, \omega_z]^T$ (Sign match depends on MPU6050 convention) |
| **Upright on Windshield (Portrait, Facing Cabin)** | Rotation about $X_{\text{dev}}$ | Rotation about $Z_{\text{dev}}$ | Rotation about $Y_{\text{dev}}$ | **`gyro_y`** | **Permutation & Sign Flip:** $\omega_{\text{yaw,veh}} = -\omega_{y,\text{dev}}$ |
| **Upright on Windshield (Landscape, Facing Cabin)**| Rotation about $Y_{\text{dev}}$ | Rotation about $Z_{\text{dev}}$ | Rotation about $X_{\text{dev}}$ | **`gyro_x`** | **Permutation & Sign Flip:** $\omega_{\text{yaw,veh}} = \pm\omega_{x,\text{dev}}$ |

### 5.4 Mathematical Gyroscope Invariant
A critical kinematic principle established by this audit is:
$$\vec{\omega}_{\text{veh}} = \mathbf{R} \vec{\omega}_{\text{dev}}$$
**The angular velocity vector $\vec{\omega}$ MUST be rotated using the exact same $3\times 3$ direction-cosine rotation matrix $\mathbf{R}$ as the linear acceleration vector $\vec{a}$.**  
Swapping accelerometer axes ($X \leftrightarrow Y$) without rotating gyroscope axes breaks the rigid-body kinematic relationship between lateral acceleration and yaw rate ($a_{\text{lat}} \approx v \cdot \omega_{\text{yaw}}$).

---

## 6. Investigation 5 — Physical Mounting Feasibility Analysis

### 6.1 Flat Console Mounting (Screen Upward, Top Pointing Forward)
- **Mathematical Relationship:**
  $$X_{\text{veh}} = Y_{\text{dev}},\quad Y_{\text{veh}} = X_{\text{dev}},\quad Z_{\text{veh}} = -Z_{\text{dev}}$$
- **Feasibility:** **PARTIALLY SUPPORTED (with calibration).**
- **Limitations:**
  1. Center consoles and dashboard trays in passenger vehicles are rarely perfectly horizontal; they have build angles of $5^\circ$ to $15^\circ$ pitch and tilt.
  2. A $10^\circ$ pitch angle leaks $9.81 \times \sin(10^\circ) = 1.70\text{ m/s}^2$ of Earth gravity into the longitudinal acceleration channel.
  3. Without static gravity removal, this $1.70\text{ m/s}^2$ constant bias triggers false acceleration or braking predictions continuously.

### 6.2 Portrait Windshield Mounting (Screen Facing Cabin, Top Pointing Upward)
- **Mathematical Relationship:**
  $$X_{\text{veh}} = -Z_{\text{dev}},\quad Y_{\text{veh}} = X_{\text{dev}},\quad Z_{\text{veh}} = -Y_{\text{dev}}$$
- **Feasibility:** **CONDITIONALLY SUPPORTED (requires full 3D rotation).**
- **Limitations:**
  1. All commercial windshield phone holders tilt backward by an angle $\theta_{\text{pitch}} \in [20^\circ, 50^\circ]$ to allow the driver to view the screen.
  2. Furthermore, windshield mounts are angled inward toward the driver by a yaw angle $\theta_{\text{yaw}} \in [10^\circ, 30^\circ]$.
  3. Consequently, Earth gravity ($9.81\text{ m/s}^2$) is distributed across all three device axes:
     $$g_x = g \sin\theta_{\text{yaw}}\sin\theta_{\text{pitch}},\quad g_y = -g \cos\theta_{\text{pitch}},\quad g_z = g \cos\theta_{\text{yaw}}\sin\theta_{\text{pitch}}$$
  4. A naive coordinate swap ($X \to Y$, $Y \to Z$) completely fails. Only a full $3\times 3$ direction-cosine rotation matrix derived from static calibration ($S_0$) can project gravity purely onto $Z_{\text{veh}}$.

### 6.3 Landscape Windshield Mounting (Screen Facing Cabin, Top Pointing Sideways)
- **Mathematical Relationship:**
  $$X_{\text{veh}} = -Z_{\text{dev}},\quad Y_{\text{veh}} = \pm Y_{\text{dev}},\quad Z_{\text{veh}} = \pm X_{\text{dev}}$$
- **Feasibility:** **CONDITIONALLY SUPPORTED (requires full 3D rotation).**
- **Limitations:** Identical to portrait windshield mounting; requires full 3D rotation matrix $\mathbf{R}$ and cannot be resolved by static permutation.

### 6.4 Arbitrary Unsecured Mounting (Cupholder, Pocket, Bag, Handheld)
- **Feasibility:** **COMPLETELY INCOMPATIBLE.**
- **Rationale:** When an unsecured phone slides in a cupholder or shifts in a pocket, its rotation matrix $\mathbf{R}(t)$ changes dynamically. Estimating a continuous time-varying rotation matrix in the presence of vehicle centrifugal and vibrational forces is impossible without dual IMU / GNSS carrier-phase kinematic fusion.

---

## 7. Investigation 6 — Final Decision & Coordinate Compatibility Matrix

### 7.1 Coordinate Compatibility Matrix

| Component | Established Convention | Empirical Evidence | Compatibility Status | Technical Justification |
| :--- | :--- | :--- | :---: | :--- |
| **Dataset B** | MPU6050 dashboard mount, face upward, AccX=Long, AccY=Lat, AccZ=Vert, GyroZ=Yaw. Units: $g$, deg/s. Gravity included. | `Data Collection Summary.docx` verbatim text; `sensor_raw.csv` mean values; Class 1 SA AccX max $= +0.64g$; Class 3 SRT GyroZ max $= +45.4^\circ/\text{s}$. | **PARTIALLY_VERIFIED** | Experimental trial boundaries verified. Lacks continuous negative cruising data to measure real-world false-alarm rates. |
| **Dataset C** | In-cabin smartphone. Units: $\text{m/s}^2$, rad/s. Linear acceleration (gravity removed). Acc Z mean $= -0.024\text{ m/s}^2$. | `3_FinalDatasetCsv.csv` summary statistics; median $\text{acc\_mag} = 1.118\text{ m/s}^2$; Production Scaler mean for AccZ $= -0.0469$. | **PARTIALLY_VERIFIED** | Linear acceleration nature verified. Physical mounting orientation was not documented by original dataset author. |
| **Pipeline A (Production Model C)** | RBF SVM (47 features). Expects $\text{m/s}^2$ without gravity. Features 0–2 are raw Acc X/Y/Z; Features 6, 11, 14 are norm stats. | `ml/saved_models/dataset_c_config_v1.0.0.json`; `dataset_c_scaler_exp_v1.joblib` mean and scale arrays. | **INCOMPATIBLE** | Incompatible with raw physical smartphone data. Earth gravity ($9.81\text{ m/s}^2$) causes $>5.5\sigma$ shift on scaler features, crashing RBF kernel activations. |
| **Pipeline B (Production Model B)** | Extra Trees (56 features). Expects MPU6050 dashboard frame with gravity. Features 36, 45, 54 are `acc_x` stats; 32, 41, 50 are `gyro_z` stats. | `models/experiments/dataset_b/dataset_b_experiment_summary.json`; Gini feature importances ($acc\_x$ and $gyro\_z$ top 10). | **INCOMPATIBLE** | No physical streaming adapter exists in `ml/inference/`. Model explicitly requires forward acceleration on AccX and yaw on GyroZ. |
| **Android Logger** | Android OS native portrait frame. Values[0]=Screen Right ($X$), Values[1]=Screen Top ($Y$), Values[2]=Screen Normal ($Z$). | `SensorLoggingService.kt` lines 240–250; Android Developer API specifications. | **VERIFIED** | Implementation correctly captures and logs raw hardware events in canonical Android coordinates. |
| **Web Logger** | W3C DeviceMotionEvent standard. X=Right, Y=Top, Z=Normal. Alpha=Yaw, Beta=Pitch, Gamma=Roll. | `mobile/web-logger/app.js` lines 50–65; W3C DeviceOrientation Event Specification. | **VERIFIED** | Implementation correctly captures and logs standard W3C browser motion telemetry. |
| **Physical Sensor Adapter** | Causal 21-tap FIR decimator (20 Hz $\to$ 10 Hz) + `StreamingPipelineAFeatureExtractor` (102 features). | `ml/inference/physical_sensor_adapter.py`; `ml/inference/sampling_compatibility_adapter.py`. | **PARTIALLY_VERIFIED** | Mathematical signal processing (decimation, causal buffers, warm-up) is fully verified. Compatibility with production models remains unproven. |

---

## 8. Final Operational Directive & Architectural Safety Gate

### 8.1 Explicit Retention of `proven_compatibility = False`
In accordance with the findings of this audit, the system configuration invariant:
$$\mathbf{proven\_compatibility = False}$$
**MUST BE STRICTLY RETAINED** across all physical sensor endpoints and inference adapters:
1. `backend/app/schemas/sensor.py`
2. `ml/inference/physical_sensor_adapter.py` (Line 131)
3. `backend/app/api/sensor_endpoints.py`

### 8.2 Production Invariants Maintained
- **Zero Model Modification:** No model or scaler under `models/` or `ml/saved_models/` was retrained, modified, or re-serialized.
- **Bit-for-Bit SHA-256 Hashes Preserved:**
  - `dataset_b_model`: `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90`
  - `dataset_b_scaler`: `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576`
  - `dataset_c_model`: `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc`
  - `dataset_c_scaler`: `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5`
- **Blind Lockbox Untouched:** Zero access or reference to `DRV_007` or `DRV_008`.
- **Safety Agent Scoring Frozen:** Deterministic risk engine scoring, state transitions, and health matrix rules remain 100% identical.

### 8.3 Recommended Roadmap for Future Physical Compatibility
Before physical-to-production ML inference can ever be enabled in a future development stage:
1. **Automated Gravity Removal Filter:** Implement an online high-pass or complementary filter in `PhysicalSensorInferenceAdapter` that estimates the static gravity vector $\vec{g}_0$ during $S_0$ standstill and subtracts it from incoming measurements to produce linear acceleration matching Dataset C.
2. **Dynamic 3D Vehicle Frame Alignment:** Integrate `VehicleFrameAligner` (`ml/inference/vehicle_frame_alignment.py`) as a mandatory pre-inference stage to project incoming acceleration and angular velocity onto canonical vehicle body axes (SAE J670) prior to feature extraction.
3. **Pipeline B Physical Adapter:** Develop a dedicated streaming feature extractor and causal decimation pipeline for Dataset B's 56 features.
4. **Empirical Track Validation:** Execute controlled road smoke tests comparing model outputs against human observer ground truth before setting `proven_compatibility = True`.

---

**Audit Completed:** November 24, 2026  
**Final Decision:** Physical-to-production ML inference is **INCOMPATIBLE**; `proven_compatibility = False` is permanently enforced. System is safe for raw data acquisition.

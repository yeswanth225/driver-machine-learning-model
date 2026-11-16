# Comprehensive Dataset Suitability & Technical Audit Report

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Auditor**: Senior Machine Learning Engineer & Project Auditor  
**Date**: November 16, 2026  
**Status**: Stage 2 Verification Complete — Awaiting Approval for Stage 3

---

## 1. Global Dataset Inventory Overview

The project workspace houses seven distinct datasets (A through G) across local directory `dataset/raw/`, totaling **432 data files**, **40,011,179 validated records**, and **2.84 GB** of storage:

| ID | Dataset Reference / Slug | Hardware & Context | Local Directory | Files | Total Rows | Total Size | Primary ML Suitability Role |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **A1** | Mendeley `10.17632/9vr83n7z5j.2` | Smartphone (Redmi/Mi) | `dataset/raw/drivers/` | 21 CSV | 1,826,637 | 98.18 MB | **Primary**: Driver Fingerprinting & Anomaly Detection |
| **A2** | Mendeley `10.17632/9vr83n7z5j.2` | Smartphone (Redmi/Mi) | `dataset/raw/daywise/` | 56 CSV + 6 XLSX | 2,885,476 | 175.39 MB | **Primary**: Longitudinal Cross-Day Generalization |
| **B** | Mendeley `10.17632/jj3tw8kj6h.3` | Raspberry Pi + MPU6050 | `dataset/raw/dataset_02_jj3tw8kj6h/` | 18 CSV + 1 DOCX | 19,848 | 19.92 MB | **Primary**: Multi-Class Risky Maneuver Classification |
| **C** | Mendeley `10.17632/5stn873wft.1` | Smartphone (Android) | `dataset/raw/dataset_03_5stn873wft/` | 3 CSV | 28,498 | 2.82 MB | **Primary**: Supervised Binary Style (Normal vs. Aggressive) |
| **D** | Kaggle `eishkaran/...` | Smartphone (Android) | `dataset/raw/dataset_04_eishkaran/` | 72 CSV | 4,173,442 | 232.08 MB | **Supplementary**: Uncurated raw mirror of Dataset A |
| **E** | Kaggle `shashwatwork/...` | Raspberry Pi + MPU6050 | `dataset/raw/dataset_05_shashwatwork/` | 18 CSV | 19,848 | 19.88 MB | **Supplementary**: Stripped derivative of Dataset B |
| **F** | Kaggle `yotamkatz/...` | OBD-II + IMU + GPS | `dataset/raw/dataset_06_yotamkatz/` | 5 CSV | 82,038 | 11.96 MB | **Secondary**: Cross-Modal Telematics Driver Profiling |
| **G** | Kaggle `vitorrf/...` | CAN Powertrain Telemetry | `dataset/raw/dataset_07_vitorrf/` | 228 CSV | 30,995,196 | 2,265.61 MB | **Supplementary**: Large-scale longitudinal telemetry mining |

---

## 2. In-Depth Dataset Verification (13 Technical Audit Dimensions)

Each dataset was independently audited against all 13 strict verification dimensions:

```
                                      DATASET AUDIT TAXONOMY
+---------------------------------------------------------------------------------------------------+
|  1. File Inventory       5. Target Labels        9. Sampling Frequency   13. Cross-Compatibility  |
|  2. Formats/Readability  6. Missing/Duplicates  10. Units & Orientation                           |
|  3. Records & Columns    7. Class Distribution  11. Driver Identifiers                            |
|  4. Sensor Columns       8. Timestamp Validity  12. Leakage Risks                                 |
+---------------------------------------------------------------------------------------------------+
```

---

### 2.1 Dataset A — Smartphone Driver Behavior (Drivers & Daywise)
- **DOI / Source**: Mendeley Data `10.17632/9vr83n7z5j.2` / PMC8914310 (Wawage et al.).
- **Hardware**: Xiaomi Redmi 4 and Mi A3 smartphones placed horizontally inside the vehicle utility box across 3 Light Motor Vehicles (Ford Figo, Maruti Swift, Tata Nexon).

1. **File Inventory & Sizes**:
   - `drivers/` (A1): 21 CSV files (7 drivers x 3 sensors: `Accelerometer.csv`, `Gyroscope.csv`, `Pressure.csv`), total 98.18 MB.
   - `daywise/` (A2): 56 CSV files + 6 redundant XLSX files across 16 session folders (`Day-1S` to `Day-7S`), total 175.39 MB.
2. **File Formats & Readability**: Standard ASCII CSV files with header rows. 100% readable.
3. **Number of Records & Columns**:
   - A1: **1,826,637 rows** (Accelerometer: 637,066; Gyroscope: 640,643; Pressure: 548,928).
   - A2: **2,885,476 rows** across 42 active CSVs (Accelerometer: 1,250,536; Gyroscope: 1,281,862; Proximity: 353,077; RotationVector: 1).
   - Grand Total Dataset A: **4,712,113 rows**.
4. **Sensor Column Names**:
   - Accelerometer: `Timestamp`, `Unix Timestamp`, `Milliseconds`, `X`, `Y`, `Z`.
   - Gyroscope: `Timestamp`, `Unix Timestamp`, `Milliseconds`, `X`, `Y`, `Z`.
   - Pressure: `Timestamp`, `Unix Timestamp`, `Milliseconds`, `Millibars`.
   - Proximity (A2): `Timestamp`, `Milliseconds`, `centimeters`.
5. **Target Labels & Meanings**:
   - **CRITICAL FINDING**: **ZERO activity/behavior labels exist.** No column represents "Normal", "Aggressive", or "Drowsy".
   - Verified labels: Ground-truth subject identity (`Driver-1` through `Driver-7`) and trip session (`Day-1S` through `Day-7S`).
6. **Missing & Duplicate Records**:
   - Primary IMU (`Accelerometer` and `Gyroscope`) in active files: **0 missing values**.
   - **Barometric Pressure Breakdown**:
     - `Driver-1`: 10,133 rows, all `Millibars = 0.0` (sensor absent/disabled).
     - `Driver-2`: 124 rows, all `0.0`.
     - `Driver-3` through `Driver-7`: **>99.8% missing (NaN)**; only 124 rows contain data per driver, and all valid values are `0.0`. Pressure is completely unusable.
   - **Empty GPS Files**: All 14 `GPS.csv` files in A2 are 0 bytes (empty).
   - Duplicates: Sensor burst polling produces repeated consecutive sensor values during static idling (~35% identical readings at 100 Hz).
7. **Class Distribution**:
   - No behavioral classes. Driver distribution across 6 usable drivers: Driver-4 (27.8%), Driver-3 (16.8%), Driver-7 (16.2%), Driver-1 (15.7%), Driver-5 (14.6%), Driver-6 (8.6%). Driver-2 (0.2%) is an aborted calibration run.
8. **Timestamp Validity**:
   - `Timestamp`: String minute resolution (`DD-MM-YY HH:MM`).
   - `Unix Timestamp`: Epoch seconds rounded to minute boundaries (increments only every 60 seconds).
   - `Milliseconds`: Monotonically increasing relative counter (0 to ~1,000,000 ms), providing millisecond resolution within trips.
9. **Sampling Frequency**:
   - A1 (Drivers): Multi-rate; effective sampling rate is **~50 Hz** for Accel/Gyro, with median inter-sample delta ranging from 10 ms to 43 ms due to Android OS scheduling jitter.
   - A2 (Daywise): **100 Hz** empirical median delta (10.0 ms).
   - Proximity: 10 Hz (100 ms).
10. **Sensor Units & Coordinates**:
    - Accelerometer: $\text{m/s}^2$ (**includes 9.81 $\text{m/s}^2$ static Earth gravity**; mean vector norm $\approx 10.05\ \text{m/s}^2$).
    - Gyroscope: $\text{rad/s}$ (Android Sensor API standard).
    - Orientation: Placed horizontally in vehicle utility box, but vehicle-relative heading alignment was uncalibrated. Raw directional axes cannot be mapped to lateral/longitudinal without rotation-invariant norms ($\|a\|, \|\omega\|$).
11. **Driver & Session Identifiers**:
    - A1: 7 distinct human drivers (`Driver-1` through `Driver-7`).
    - A2: Single driver recorded across 7 days (16 outbound/return trips).
12. **Possible Data Leakage**:
    - High sample-to-sample autocorrelation ($r > 0.98$ at 100 Hz). Random row shuffling leaks temporal sequences, producing falsely inflated accuracy.
    - Mixing A1 and A2 into a single random pool causes cross-corpus subject leakage.
13. **Compatibility**: High internal compatibility between Accelerometer and Gyroscope via millisecond synchronization. Incompatible with Dataset B without unit conversion ($g \to \text{m/s}^2$, $\text{deg/s} \to \text{rad/s}$) and gravity removal.

---

### 2.2 Dataset B — Driving Behavior Dataset (Raspberry Pi & MPU6050)
- **DOI / Source**: Mendeley Data `10.17632/jj3tw8kj6h.3` (Yuksel & Atmaca, 2021).
- **Hardware**: Raspberry Pi Model B + MPU6050 6-axis IMU mounted to center dashboard, front face upward, across 3 passenger vehicles (Ford Fiesta Diesel/Gasoline, Hyundai i20).

1. **File Inventory & Sizes**:
   - 18 CSV files (1 raw sensor log `sensor_raw.csv` + 17 pre-extracted window feature files) + 1 documentation file (`Data Collection Summary.docx`). Total size: 19.92 MB.
2. **File Formats & Readability**: Fully formatted CSVs, 100% readable.
3. **Number of Records & Columns**:
   - `sensor_raw.csv`: **1,114 rows**, 12 columns.
   - 17 Feature files: **18,734 rows** (each file has ~1,102 rows and 61 statistical features).
   - Total rows: **19,848 rows**.
4. **Sensor Column Names**:
   - Raw: `Class`, `DriverID`, `TaskID`, `Hour`, `Minute`, `Second`, `GyroX`, `GyroY`, `GyroZ`, `AccX`, `AccY`, `AccZ`.
   - Features: `Target`, `AccMeanX..Z`, `AccCovX..Z`, `AccSkewX..Z`, `AccKurtX..Z`, `AccMinX..Z`, `AccMaxX..Z`, `AccVarX..Z`, `AccMedianX..Z`, `AccStdX..Z`, `GyroMeanX..Z`, etc.
5. **Target Labels & Meanings (Confirmed via Documentation)**:
   - `1`: **Sudden Acceleration (SA)** — rapid throttle application from standstill/low speed.
   - `2`: **Sudden Deceleration (SD)** — harsh braking event.
   - `3`: **Sudden Right Turn (SRT)** — aggressive rightward steer (large negative GyroZ).
   - `4`: **Sudden Left Turn (SLT)** — aggressive leftward steer (large positive GyroZ).
   - **CRITICAL FINDING**: **NO normal driving class exists.** All 1,114 raw records represent detected risky maneuvers.
6. **Missing & Duplicate Records**:
   - Missing values: **0** (100% complete across all 12 columns).
   - Duplicate rows: **0 exact duplicates**.
7. **Class Distribution in `sensor_raw.csv`**:
   - Class 3 (SRT): 350 rows (31.4%) — ~25 maneuvers.
   - Class 2 (SD): 288 rows (25.9%) — ~32 maneuvers.
   - Class 1 (SA): 252 rows (22.6%) — ~28 maneuvers.
   - Class 4 (SLT): 224 rows (20.1%) — ~24 maneuvers.
   - Total events: 109 maneuvers spanning ~10 rows (10 seconds) each.
8. **Timestamp Validity**:
   - Recorded as discrete integer columns: `Hour` (12–15), `Minute` (0–58), `Second` (0–59). Continuous within events; lacks absolute epoch milliseconds.
9. **Sampling Frequency**:
   - Low sampling rate: **~1.0 Hz** (1 sample per second). Mean samples per unique second is 1.38.
10. **Sensor Units & Coordinates**:
    - Accelerometer (`AccX, AccY, AccZ`): **g-force** (Earth gravity $\approx 1.0\ g$; mean AccZ = $-0.983\ g$, Acc norm mean = $1.052\ g$).
    - Gyroscope (`GyroX, GyroY, GyroZ`): **degrees per second (deg/s)** (std of GyroZ = $12.05^\circ/\text{s}$, max = $45.4^\circ/\text{s}$, min = $-50.3^\circ/\text{s}$).
11. **Driver & Session Identifiers**:
    - 3 distinct drivers: `Driver-1` (368 rows, 35 events), `Driver-2` (363 rows, 36 events), `Driver-3` (383 rows, 38 events).
    - `TaskID`: 33 sequential event task boundaries per driver.
12. **Possible Data Leakage**:
    - Splitting rows randomly within a single maneuver event leaks event dynamics across train and test. Splitting must be grouped strictly by `TaskID` or `DriverID`.
13. **Compatibility**: Incompatible with Dataset A without multiplying acceleration by 9.80665 and converting gyroscope to radians ($^\circ \times \pi / 180$).

---

### 2.3 Dataset C — Phone Sensor Data While Driving a Car (Normal vs. Aggressive)
- **DOI / Source**: Mendeley Data `10.17632/5stn873wft.1` (Shardul Nazirkar, 2021).
- **Hardware**: Android smartphone placed inside a passenger car during two driving trips in Pune, India.

1. **File Inventory & Sizes**:
   - 3 CSV files (`3_FinalDatasetCsv.csv`: 1.03 MB, `2_20210317_171452.csv`: 1.03 MB, `1_20210317_184512.csv`: 0.90 MB). Total size: 2.82 MB.
2. **File Formats & Readability**: Clean CSV files, 100% readable.
3. **Number of Records & Columns**:
   - `3_FinalDatasetCsv.csv`: **14,249 rows**, 7 columns.
   - Raw Trip 2 (`2_...`): **7,603 rows**, 12 columns.
   - Raw Trip 1 (`1_...`): **6,646 rows**, 12 columns.
   - Concatenation confirmed: $7,603 + 6,646 = \mathbf{14,249\text{ rows}}$.
4. **Sensor Column Names**:
   - Final: `Acc X`, `Acc Y`, `Acc Z`, `gyro_x`, `gyro_y`, `gyro_z`, `label`.
   - Raw trips contain additional columns: `Longitude`, `Latitude`, `Speed`, `Distance`, `Time`, `Heading`.
5. **Target Labels & Meanings**:
   - `0`: **Normal Driving** — 5,805 records (40.74%).
   - `1`: **Aggressive Driving** — 8,444 records (59.26%).
   - Ground-truth binary annotation validated by significant kinetic energy contrast (Aggressive driving exhibits 2.64x higher acceleration norm and 2.73x higher gyroscope norm).
6. **Missing & Duplicate Records**:
   - Missing values: **0** across all 7 columns.
   - Duplicate rows: Exactly **9 records** (4 in Session 1, 5 in Session 2) caused by hardware polling collisions at second boundaries. Total clean records = **14,240**.
7. **Class Distribution**:
   - Normal (0): 5,805 (40.74%); Aggressive (1): 8,444 (59.26%). Moderate imbalance, easily handled via class-weighted loss.
8. **Timestamp Validity**:
   - Final CSV lacks explicit timestamp column; temporal sequence is strictly monotonic row order. Raw trips confirm 10 rows per second (`Time` HH:MM:SS format).
9. **Sampling Frequency**:
   - **~10.0 Hz** empirical sampling rate ($\Delta t \approx 100\text{ ms}$).
10. **Sensor Units & Coordinates**:
    - Accelerometer: $\text{m/s}^2$ (**gravity-compensated / linear acceleration**; mean Acc Z is $-0.024\ \text{m/s}^2$, overall norm mean is $1.61\ \text{m/s}^2$).
    - Gyroscope: $\text{rad/s}$ (range $\pm 2.06\ \text{rad/s}$).
11. **Driver & Session Identifiers**:
    - Inferred single driver; both trips recorded sequentially on the afternoon of March 17, 2021 in Pune.
    - Two clear physical sessions: Session 1 (7,599 clean rows) and Session 2 (6,641 clean rows).
12. **Possible Data Leakage**:
    - Adjacent 100ms frames and rolling feature windows leak future signals under random splitting. Chronological session splitting with 10s guard bands eliminates all leakage.
    - **Generalization Limitation**: Cannot measure driver-independent generalization because only one driver was recorded.
13. **Compatibility**: Directly compatible with modern smartphone driving agents; linear acceleration matches typical Android linear acceleration sensor outputs.

---

### 2.4 Dataset D — Kaggle Driver Behaviour Analysis (Eishkaran Singh)
- **Source**: Kaggle `eishkaran/driver-behaviour-analysis-using-sensor` (CC BY 4.0).
- **Physical Composition**: 72 CSV files across 8 subdirectories, 4,173,442 rows, 232.08 MB.
- **Relationship to Dataset A**: Direct mirror of the raw experiment behind Dataset A, organized by date-timestamp subfolders (`2021-02-02_10-28-14`) rather than clean driver IDs.
- **Suitability Verdict**: **SUPPLEMENTARY**. Dataset A is strictly preferred due to clean driver identity curation.

---

### 2.5 Dataset E — Kaggle Driving Behavior Dataset (Shashwat Tiwari)
- **Source**: Kaggle `shashwatwork/driving-behavior-dataset` (CC BY 4.0).
- **Physical Composition**: 18 CSV files (1 `sensor_raw.csv` + 17 feature files), 19,848 rows, 19.88 MB.
- **Relationship to Dataset B**: Exact mathematical derivative of Dataset B (sensor values match to $7.1 \times 10^{-15}$). The author stripped `DriverID`, `TaskID`, and continuous timestamp metadata.
- **Suitability Verdict**: **SUPPLEMENTARY / RETIRED**. Dataset B is strictly superior because it retains driver and task boundaries required for leak-free evaluation.

---

### 2.6 Dataset F — Car Driver Behavior (4 Drivers, 1 Route, 1 Car)
- **Source**: Kaggle `yotamkatz/car-driver-behavior-4-drivers-1-route-1-car` (CC BY-SA 4.0).
- **Hardware**: OBD-II vehicle interface + smartphone GPS/IMU capturing 4 drivers (E, G, R, Y) over an identical 21-km driving circuit in Israel using the same vehicle.
- **Physical Composition**: 5 CSV files (`all_drivers.csv`, `driver_E.csv`, `driver_G.csv`, `driver_R.csv`, `driver_Y.csv`), 82,038 total rows (41,019 master + 41,019 individual), 11.96 MB.
- **Channels (21)**: Engine RPM, Vehicle Speed, Throttle Position, Engine Load, Coolant Temp, GPS, `Accel X/Y/Z (m/s²)`, `Rotation Rate X/Y/Z (deg/s)`.
- **Sampling Frequency**: **1.0 Hz** (`elapsed_s` increments by 1.0s).
- **Critical Duplication Finding**: `all_drivers.csv` is a **100% exact duplicate** of the 4 driver files concatenated together. Including both causes severe data duplication.
- **Suitability Verdict**: **SECONDARY CANDIDATE**. Valuable for **Cross-Modal Telematics Driver Profiling & CAN/OBD Baseline**.

---

### 2.7 Dataset G — Car Trips Data Log / Data Mining (VitorRF)
- **Source**: Kaggle `vitorrf/cartripsdatamining` (ODbL / DbCL).
- **Physical Composition**: 228 CSV files (30,995,196 rows, 2.21 GB) across `Processed Data` and `RawData` subfolders + 6 RTF readme files.
- **Channels (17)**: Time, Vehicle Speed (m/s), Gear Shift, Engine Load, Total Accel (m/s²), Engine RPM, Pitch, Lateral Accel (m/s²), Passenger Count, Car Load, AC Status, Window Opening, Radio Volume, Rain, Visibility, Driver Wellbeing, Driver Rush.
- **Limitations**: No triaxial IMU (no raw 3-axis gyro). CSV files lack column headers (documented in `readme.rtf`).
- **Suitability Verdict**: **SUPPLEMENTARY**. Excellent for large-scale longitudinal vehicle dynamics mining, but not suitable for real-time mobile IMU safety agents.

---

## 3. Dataset Suitability & Role Allocation Matrix

| Candidate Dataset | Assigned Machine Learning Task | Ground Truth Available | Temporal Resolution | Driver Independence Evaluation? | Recommended Model Families | Production Gate Status |
|:---|:---|:---:|:---:|:---:|:---|:---:|
| **Dataset C** | **Supervised Binary Driving Style** (Normal vs. Aggressive) | ✅ Yes (0/1) | 10.0 Hz | ❌ No (Within-trip holdout) | LightGBM, XGBoost, SVM, 1D-CNN | **READY (GO ✅)** |
| **Dataset B** | **Supervised Multi-Class Risky Maneuver Detection** (SA, SD, SRT, SLT) | ✅ Yes (1-4) | ~1.0 Hz | ✅ Yes (3 drivers, 33 tasks) | Random Forest, Extra Trees, Gradient Boosting | **READY (GO ✅)** |
| **Dataset A (Drivers)** | **Biometric Driver Identification & Unsupervised Anomaly Detection** | ✅ Yes (7 Drivers) | ~50 Hz | ✅ Yes (LODO-CV across 6 drivers) | Isolation Forest, Autoencoder, Random Forest | **READY (GO ✅)** |
| **Dataset F** | **Cross-Modal Telematics Profiling** (OBD + IMU) | ✅ Yes (4 Drivers) | 1.0 Hz | ✅ Yes (Hold-out driver evaluation) | Random Forest, XGBoost | **BACKLOG (GO ✅)** |
| **Dataset D** | Raw Mirror of Dataset A | ❌ No | ~50 Hz | ❌ No | — | **STANDBY (RETAIN)** |
| **Dataset E** | Stripped Derivative of Dataset B | ⚠️ Stripped | ~1.0 Hz | ❌ No | — | **STANDBY (USE B)** |
| **Dataset G** | Longitudinal CAN Powertrain Telematics | ⚠️ Contextual | 1.0 Hz | ⚠️ Trips | — | **SUPPLEMENTARY** |

---

## 4. Final Recommendations & Execution Directives

1. **Approved Primary Datasets**:
   - Proceed with **Dataset C** for real-time mobile safety agent binary style scoring.
   - Proceed with **Dataset B** for discrete multi-class risky maneuver detection.
   - Proceed with **Dataset A** for driver biometric authentication and unsupervised anomaly baselining.
2. **Strict Preprocessing Guardrails for Stage 3**:
   - Always exclude `dataset_06_yotamkatz/all_drivers.csv` (100% duplicate of driver CSVs).
   - Always exclude `Pressure.csv` in Dataset A (all 0.0 mbar; >99.8% NaN).
   - Always drop `Day-4E1` and `Driver-2` from Dataset A modeling (truncated calibration trials).
   - Never merge Dataset B and Dataset C directly (incompatible units, sampling rates, and label definitions).
3. **Execution Directive**:
   - All Stage 2 verification requirements are complete.
   - Awaiting user confirmation before initiating **Prompt 3 — Data Preprocessing & Feature Engineering**.

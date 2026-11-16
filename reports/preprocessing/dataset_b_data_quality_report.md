# Dataset B Data Quality & Sensor Validation Report

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Stage**: Stage 3A — Dataset B Preprocessing & Feature Engineering  
**Dataset**: Driving Behavior Dataset (Raspberry Pi & MPU6050)  
**DOI**: [10.17632/jj3tw8kj6h.3](https://doi.org/10.17632/jj3tw8kj6h.3) (Yuksel & Atmaca, 2021)  
**Target File**: `dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/sensor_raw.csv`  
**Date**: November 16, 2026  
**Status**: Verified & Preprocessed (22/22 Automated Tests Passed)

---

## 1. Executive Summary

This report documents the comprehensive data quality, sensor integrity, and signal validation performed on **Dataset B** prior to feature extraction and partitioning. 

Dataset B represents an event-driven dataset captured via a Raspberry Pi Model B equipped with an MPU6050 6-axis IMU mounted to the dashboard of three passenger vehicles in Ankara and Isparta, Turkey. All 1,114 raw records were confirmed to belong to **109 isolated risky driving maneuver trials**, with zero baseline normal driving records.

---

## 2. File Inventory, Schemas & Readability

| Attribute | Specification | Verification Result |
|:---|:---|:---:|
| **Target File Path** | `dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/sensor_raw.csv` | ✅ PRESENT |
| **Associated Documentation** | `dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/Data Collection Summary.docx` | ✅ PRESENT |
| **Pre-computed Feature Folder** | `dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/Features By Window Size/` (17 CSVs) | ✅ PRESENT |
| **Primary File Size** | 135,636 bytes (~132.5 KB) | ✅ VERIFIED |
| **SHA-256 Checksum** | `9b3a0667562e57eaf5cd7e0c656349518bebf7095dac4627f2353051858d1594` | ✅ MATCHED |
| **Total Rows** | 1,114 rows | ✅ EXACT |
| **Total Columns** | 12 columns | ✅ EXACT |
| **Missing Values (Null / NaN)** | **0 cells (0.00%)** | ✅ PRISTINE |
| **Exact Duplicate Rows** | **0 rows (0.00%)** | ✅ PRISTINE |

### Raw Schema Definitions:
1. `Class` (int64): Categorical maneuver ground-truth (`1`, `2`, `3`, `4`).
2. `DriverID` (object): Subject identifier (`Driver-1`, `Driver-2`, `Driver-3`).
3. `TaskID` (int64): Experimental trial iteration number within each maneuver class.
4. `Hour` (int64): Recording hour (12 to 15).
5. `Minute` (int64): Recording minute (0 to 58).
6. `Second` (int64): Recording second (0 to 59).
7. `GyroX` (float64): Angular velocity about the X-axis.
8. `GyroY` (float64): Angular velocity about the Y-axis.
9. `GyroZ` (float64): Angular velocity about the Z-axis (vehicle yaw rate).
10. `AccX` (float64): Linear acceleration along the X-axis (longitudinal vehicle axis).
11. `AccY` (float64): Linear acceleration along the Y-axis (lateral vehicle axis).
12. `AccZ` (float64): Linear acceleration along the Z-axis (vertical gravity axis).

---

## 3. Ground-Truth Target Label Semantics & Distributions

Ground-truth label semantics were cross-verified against `Data Collection Summary.docx`:

| Class ID | Target Maneuver Name | Abbreviation | Total Events | Total Raw Samples | Sample % | Mean Duration |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Sudden Acceleration** | **SA** | 28 events | 252 rows | 22.62% | 9.00 samples (~6.3s) |
| **2** | **Sudden Deceleration** | **SD** | 24 events | 288 rows | 25.85% | 12.00 samples (~8.5s) |
| **3** | **Sudden Right Turn** | **SRT** | 25 events | 350 rows | 31.42% | 14.00 samples (~9.2s) |
| **4** | **Sudden Left Turn** | **SLT** | 32 events | 224 rows | 20.11% | 7.00 samples (~4.2s) |
| **TOTAL** | — | — | **109 events** | **1,114 rows** | **100.00%** | **10.22 samples (~6.6s)** |

> [!IMPORTANT]
> **Absence of Normal Driving**: Dataset B contains exclusively detected risky maneuver events. There are no negative (normal driving) records. It must be modeled as a **4-class discrete event classifier**, not a binary normal vs. aggressive classifier.

---

## 4. Subject Cohorts & Event Breakdown

The 109 events were executed across three male drivers in controlled driving sessions:

| Driver Subject | Age | Vehicle Used | SA (1) | SD (2) | SRT (3) | SLT (4) | Total Events | Total Samples |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Driver-1** | 26 | Ford Fiesta 1.4 Diesel | 10 | 8 | 9 | 8 | **35 events** | 368 rows |
| **Driver-2** | 28 | Ford Fiesta 1.25 Gasoline | 10 | 7 | 9 | 9 | **35 events** | 363 rows |
| **Driver-3** | 37 | Hyundai i20 1.2 Gasoline | 8 | 9 | 7 | 15 | **39 events** | 383 rows |
| **TOTAL** | — | — | **28** | **24** | **25** | **32** | **109 events** | **1,114 rows** |

---

## 5. Verified Sensor Units & Physical Coordinates

### 5.1 Accelerometer Channels
- **Source Unit**: **g-force** ($1.0\ g \approx 9.80665\ \text{m/s}^2$).
- **Evidence**:
  - `AccZ` mean = $-0.9832\ g$ (closely matches static Earth gravity pointing downwards along sensor negative Z).
  - Acceleration magnitude mean = $1.052\ g$.
- **Mounting Configuration**: Center of dashboard with the sensor enclosure's front face pointing upwards.
- **Conversion to SI**: All acceleration channels were converted to standard $\text{m/s}^2$ using $a_{\text{ms2}} = a_g \times 9.80665$. Both raw and converted channels are preserved.

### 5.2 Gyroscope Channels
- **Source Unit**: **degrees per second (deg/s)**.
- **Evidence**:
  - `GyroZ` exhibits large dynamic excursions during turns ($\min = -50.26^\circ/\text{s}$, $\max = +45.44^\circ/\text{s}$, $\sigma = 12.05^\circ/\text{s}$). If units were radians per second, these values would imply an impossible rotation rate of ~8 revolutions per second.
- **Conversion to SI**: All angular velocity channels were converted to radians per second ($\text{rad/s}$) using $\omega_{\text{rads}} = \omega_{\text{degs}} \times \frac{\pi}{180}$. Both representations are retained.

---

## 6. Sampling Interval & Temporal Characteristics

1. **Clock Resolution**: Integer `Hour`, `Minute`, and `Second`. No sub-second millisecond counter exists.
2. **Empirical Sampling Rate**:
   - Total elapsed duration across 109 events: **721 seconds**.
   - Total rows: **1,114 rows**.
   - Effective sampling rate: **~1.55 Hz** (approximately 1.0 to 2.0 samples per second).
3. **Temporal Isolation**:
   - Each event trial lasts between 7 and 14 seconds.
   - Significant temporal gaps (minutes to hours) exist between different event trials.
   - **Conclusion**: Causal rolling windows and time-series feature operations must **never span across different event boundaries**.

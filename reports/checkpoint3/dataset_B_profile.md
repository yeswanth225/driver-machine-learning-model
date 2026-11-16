# Dataset B — Individual Profiling Report
**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Checkpoint**: 3 — Comprehensive Dataset Inspection  
**Generated**: 2026-11-16  
**Dataset DOI**: 10.17632/jj3tw8kj6h.3  
**Authors**: Asim Sinan Yuksel & Şerafettin Atmaca

---

## Overview

Dataset B is a **Raspberry Pi-based driving behavior dataset** collected with an MPU6050 IMU sensor mounted to a vehicle dashboard. It includes both raw sensor recordings and pre-computed statistical feature sets across 17 different window sizes.

---

## Hardware & Collection Setup (from `Data Collection Summary.docx`)

| Parameter | Value |
|-----------|-------|
| **Equipment** | Raspberry Pi Model B |
| **Sensor** | MPU6050 (3-axis accelerometer + 3-axis gyroscope) |
| **Mount point** | Center of dashboard, front face upward |
| **Vehicles** | Ford Fiesta 1.4 Diesel, Ford Fiesta 1.25 Gasoline, Hyundai i20 1.2 Gasoline |
| **Location** | Ankara and Isparta, Turkey |
| **Road condition** | Dry asphalt |
| **Weather** | Sunny |
| **Speed** | 30–40 km/h for turns, 70–80 km/h for acceleration/deceleration |
| **Duration** | 2 weeks |
| **Population** | 3 male drivers (ages 26, 28, 37) |

---

## Class Label Definitions (VERIFIED from documentation)

> ✅ **Ground-truth labels are CONFIRMED from `Data Collection Summary.docx`.**

The dataset was collected specifically to detect **4 risky driving behavior events**:

| Class | Behavior | Abbreviation |
|-------|----------|-------------|
| **1** | Sudden Acceleration | SA |
| **2** | Sudden Deceleration | SD |
| **3** | Sudden Right Turn | SRT |
| **4** | Sudden Left Turn | SLT |

> ⚠️ **IMPORTANT**: There is NO "normal driving" class in this dataset. **All 1,114 raw rows represent detected risky events.** This is an event-only dataset — it does not contain baseline normal driving samples.

### Event Counts (from documentation)

| Driver (Age) | SA (Class 1) | SD (Class 2) | SRT (Class 3) | SLT (Class 4) | Total |
|-------------|-------------|-------------|--------------|--------------|-------|
| Driver-1 (26) | 10 | 8 | 9 | 8 | 35 |
| Driver-2 (28) | 10 | 10 | 9 | 7 | 36 |
| Driver-3 (37) | 8 | 14 | 7 | 9 | 38 |
| **Total events** | **28** | **32** | **25** | **24** | **109 events** |

> ⚠️ **Row count vs event count discrepancy**: The documentation lists 109 total events, but `sensor_raw.csv` has 1,114 rows and class distribution shows 252/288/350/224 = 1,114. This means each "event" spans multiple rows (approximately 10 rows per event on average). The timestamp granularity is 1-second, confirming that each event is captured over ~10 seconds of sensor data.

### Class Distribution in `sensor_raw.csv`

| Class | Rows | % | Approx events |
|-------|------|---|--------------|
| 1 (SA) | 252 | 22.6% | 28 |
| 2 (SD) | 288 | 25.9% | 32 |
| 3 (SRT) | 350 | 31.4% | 25–26 |
| 4 (SLT) | 224 | 20.1% | 24–25 |

---

## `sensor_raw.csv` — Raw Sensor File

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `Class` | int64 | Behavior class (1–4) |
| `DriverID` | object | Driver-1, Driver-2, Driver-3 |
| `TaskID` | int64 | Sequential event ID (1–33 unique per driver) |
| `Hour` | int64 | Recording hour (12–15 range) |
| `Minute` | int64 | Recording minute (0–58) |
| `Second` | int64 | Recording second (0–59) |
| `GyroX` | float64 | Gyroscope X-axis |
| `GyroY` | float64 | Gyroscope Y-axis |
| `GyroZ` | float64 | Gyroscope Z-axis |
| `AccX` | float64 | Accelerometer X-axis |
| `AccY` | float64 | Accelerometer Y-axis |
| `AccZ` | float64 | Accelerometer Z-axis |

### Verified Sensor Units (MPU6050)

| Sensor | Unit | Confidence | Evidence |
|--------|------|------------|---------|
| AccX, AccY, AccZ | **g-force** | ✅ HIGH | AccZ mean = -0.983, Acc magnitude mean = 1.052 (consistent with gravity = 1g) |
| GyroX, GyroY, GyroZ | **degrees/second (deg/s)** | ✅ HIGH | GyroZ std = 12.0 (much too large for rad/s; consistent with deg/s for vehicle maneuvers) |

> ⚠️ **UNIT DIFFERENCE vs Dataset A**: Dataset A uses m/s² and rad/s (Android sensor API). Dataset B uses g-force and deg/s (MPU6050 raw output). These are incompatible without conversion. Do NOT merge raw values without unit normalization.

### Sampling Rate (B)

**~1 Hz** (1 sample per second). The timestamp has only 1-second resolution (Hour:Minute:Second). Mean rows per unique timestamp = 1.38, max = 6 (burst). This is a very low sampling rate for motion analysis — it limits the frequency content available for event detection.

### Missing Values & Duplicates

| Check | Result |
|-------|--------|
| Missing values | ✅ None (0 nulls) |
| Duplicate rows | ✅ 0 exact duplicates |
| Null in Class/DriverID | ✅ None |

### Sensor Value Ranges

| Column | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| GyroX | -14.947 | 12.779 | -0.830 | 3.404 |
| GyroY | -10.351 | 16.794 | 4.122 | 3.253 |
| GyroZ | -50.260 | 45.443 | 1.076 | 12.049 |
| AccX | -0.253 | 0.748 | 0.252 | 0.183 |
| AccY | -0.794 | 0.769 | -0.095 | 0.190 |
| AccZ | -1.368 | -0.457 | -0.983 | 0.098 |

### Outliers

| Column | Outliers (3×IQR) | Pct | Note |
|--------|-----------------|-----|------|
| GyroX | 17 | 1.5% | Minor |
| GyroY | 9 | 0.8% | Minor |
| **GyroZ** | **202** | **18.1%** | ⚠️ High — sudden turn events have extreme rotation rates |
| AccX | 0 | 0.0% | None |
| AccY | 27 | 2.4% | Minor |
| AccZ | 3 | 0.3% | Minor |

> ⚠️ **GyroZ outliers are event-related, not noise**: The 18% outlier rate in GyroZ reflects the extreme rotation rates during sudden turns (Class 3/4). These are **informative signals, not noise** — do not remove them.

### Per-Driver Distribution

| Driver | SA(1) | SD(2) | SRT(3) | SLT(4) | Total |
|--------|-------|-------|--------|--------|-------|
| Driver-1 | 90 | 96 | 126 | 56 | 368 |
| Driver-2 | 90 | 84 | 126 | 63 | 363 |
| Driver-3 | 72 | 108 | 98 | 105 | 383 |

---

## Features By Window Size — Pre-computed Feature Files

### Overview

| Property | Value |
|----------|-------|
| Number of feature files | 17 |
| Window sizes available | 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 |
| Rows per file | 1,102 (consistent across all 17 files) |
| Columns per file | 61 (1 target + 60 features) |
| All files same shape | ✅ Yes |
| Target values | 1, 2, 3, 4 (same as Class) |

### Feature Schema (61 columns)

**Target**: `Target` (int, class label 1–4)

**Features** (60 columns, 10 per axis for acc and gyro):
- `AccMeanX/Y/Z` — mean
- `AccCovX/Y/Z` — covariance (variance)
- `AccSkewX/Y/Z` — skewness
- `AccKurtX/Y/Z` — kurtosis
- `AccSumX/Y/Z` — sum
- `AccMinX/Y/Z` — minimum
- `AccMaxX/Y/Z` — maximum
- ... (equivalent set for Gyro)

> ⚠️ **Window size meaning**: The number represents seconds (since B samples at ~1 Hz). Window size 10 = 10-second window. Window size 20 = 20-second window.

> ⚠️ **Row count discrepancy**: Raw has 1,114 rows; feature files have 1,102 rows. The 12-row difference is due to windowing (early rows that don't form a complete window are dropped).

---

## Limitations

1. **No normal driving class**: Cannot be used standalone for binary normal vs. aggressive classification without supplementing with normal driving data.
2. **Very low sampling rate (1 Hz)**: Limits frequency-domain analysis and time-series deep learning.
3. **Small dataset**: 1,114 raw rows, 109 documented events across 3 drivers.
4. **Platform mismatch**: Raspberry Pi + MPU6050 ≠ smartphone sensors. Units, noise characteristics, and mounting stability differ significantly from phone-based datasets.
5. **Fixed mount**: Dashboard-mounted with known orientation (front face up) — orientation effects are consistent but not generalizable to free-mounted smartphones.
6. **Task IDs**: 33 unique TaskIDs across 3 drivers — sessions are identifiable. Leakage-safe splitting requires holding out complete TaskID groups.

## Suitability Assessment

| Task | Suitable? | Notes |
|------|-----------|-------|
| Supervised multi-class event classification | ✅ Yes | 4 labeled risky event types |
| Binary normal vs. aggressive | ⚠️ Limited | No normal class — needs augmentation |
| Deep learning (LSTM/CNN) | ❌ No | 1 Hz too low; insufficient volume |
| Personalized profiling | ⚠️ Limited | Only 3 drivers, no normal baseline |
| Feature engineering reference | ✅ Yes | 17 pre-computed feature window sets |

# Dataset C — Individual Profiling Report
**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Checkpoint**: 3 — Comprehensive Dataset Inspection  
**Generated**: 2026-11-16  
**Dataset DOI**: 10.17632/5stn873wft.1  
**Author**: Shardul Nazirkar

---

## Overview

Dataset C is a **smartphone-based driving dataset** collected during two real trips in Pune, India. It includes GPS, speed, IMU sensor data (accelerometer + gyroscope), and **binary behavior labels (0 = normal, 1 = aggressive)**. The dataset exists in two forms: unlabeled raw trip CSVs and a labeled combined CSV.

---

## Files

| File | Role | Rows | Columns | Labels |
|------|------|------|---------|--------|
| `3_FinalDatasetCsv.csv` | **Primary labeled dataset** | 14,249 | 7 | ✅ Binary (0/1) |
| `2_20210317_171452.csv` | Raw trip 2 (HH:MM:SS = 17:14:52) | 7,603 | 12 | ❌ None |
| `1_20210317_184512.csv` | Raw trip 1 (HH:MM:SS = 18:45:12) | 6,646 | 12 | ❌ None |

> ✅ **Confirmed**: `3_FinalDatasetCsv.csv` row count (14,249) = trip2 (7,603) + trip1 (6,646). The final file is a concatenation of both trips, sensor columns only (GPS/Speed/Heading dropped), with labels added.

> ✅ **Trip order confirmed**: Final file starts with trip2 data (17:14 trip), followed by trip1 data (18:45 trip). First 5 rows of `3_FinalDatasetCsv.csv` match first 5 rows of `2_20210317_171452.csv`.

---

## Label Definitions

| Label | Meaning | Row Count | Percentage |
|-------|---------|-----------|-----------|
| **0** | Normal driving | 5,805 | 40.7% |
| **1** | Aggressive driving | 8,444 | 59.3% |

> ⚠️ **Label provenance**: The labeling method is not explicitly documented in the dataset files (no README or paper DOI provided with the Mendeley dataset). Based on sensor value analysis and naming convention, labels are consistent with binary normal/aggressive classification but the exact labeling criteria (threshold-based, manual annotation, or automatic rule-based) is **unknown**.

> ⚠️ **Class imbalance**: 59.3% aggressive vs 40.7% normal. Moderate imbalance — class-weighted training or stratified sampling is advisable.

---

## `3_FinalDatasetCsv.csv` — Primary Labeled File

### Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Acc X` | float64 | m/s² | Accelerometer X-axis |
| `Acc Y` | float64 | m/s² | Accelerometer Y-axis |
| `Acc Z` | float64 | m/s² | Accelerometer Z-axis |
| `gyro_x` | float64 | rad/s | Gyroscope X-axis |
| `gyro_y` | float64 | rad/s | Gyroscope Y-axis |
| `gyro_z` | float64 | rad/s | Gyroscope Z-axis |
| `label` | int64 | — | 0=Normal, 1=Aggressive |

> **No timestamp column** in the labeled file. Sequential row order is the only temporal indicator.

### Verified Sensor Units

| Sensor | Unit | Confidence | Evidence |
|--------|------|------------|---------|
| Acc X/Y/Z | **m/s²** | ⚠️ MEDIUM | Acc magnitude mean = 1.61 (too low for m/s² gravity ≈9.8); Acc Z mean = -0.02 (not -9.8). Values suggest gravity-compensated or g-force scaled. |
| gyro_x/y/z | **rad/s** | ✅ HIGH | Gyro range ≈ ±2 rad/s, consistent with smartphone rad/s output |

> ⚠️ **Accelerometer unit uncertainty**: The near-zero Acc Z mean and low overall magnitude suggest the accelerometer data may be **gravity-compensated** (linear acceleration only), or the values may be in g-force with no dominant orientation. This differs from Dataset A where gravity component is present. The labeling as "Acc X/Y/Z" matches Android convention but the absence of ~9.8 gravity offset requires clarification during preprocessing.

### Sampling Rate

| Measurement | Value |
|------------|-------|
| Rows per second (mean) | ~10 |
| Effective sampling rate | **~10 Hz** |
| Time resolution | 1-second (from trip file `Time` column) |

> ✅ **10 Hz confirmed**: Both trip files show Mean = 9.96 rows/second, Max = 10 rows/second.

### Missing Values & Duplicates

| Check | `3_FinalDatasetCsv.csv` | `1_*.csv` | `2_*.csv` |
|-------|------------------------|-----------|-----------|
| Null values | 0 | 0 | 0 |
| Exact duplicate rows | **9** ⚠️ | 4 | 3 |

**Duplicate rows detail** (`3_FinalDatasetCsv.csv`):
- Row 2332: 1 duplicate (normal)
- Rows 7600–7602: 2 consecutive duplicates (3 identical rows, normal)
- Row 7665: 1 duplicate (normal)
- Row 13714: 1 duplicate (aggressive)
- Rows 14246–14248: 2 consecutive duplicates (3 identical rows, aggressive)

> ℹ️ Duplicate count is very low (9/14,249 = 0.06%). These likely represent sensor read collisions or timestamp conflicts in the recording app. Safe to drop before training.

### Sensor Value Ranges

| Column | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Acc X | -8.806 | 10.489 | -0.001 | 1.031 |
| Acc Y | -8.477 | 11.013 | 0.180 | 0.837 |
| Acc Z | -17.321 | 19.726 | -0.024 | 2.095 |
| gyro_x | -2.063 | 1.612 | -0.001 | 0.098 |
| gyro_y | -0.572 | 0.600 | -0.007 | 0.070 |
| gyro_z | -0.666 | 0.543 | -0.003 | 0.068 |

### Outliers (3×IQR)

| Column | Outliers | Pct | Concern Level |
|--------|----------|-----|--------------|
| Acc X | 470 | 3.3% | 🟡 Medium |
| Acc Y | 454 | 3.2% | 🟡 Medium |
| Acc Z | 392 | 2.8% | 🟡 Medium |
| gyro_x | 621 | 4.4% | 🟡 Medium |
| gyro_y | 678 | 4.8% | 🟡 Medium |
| gyro_z | 405 | 2.8% | 🟡 Medium |

> ℹ️ Outlier rates of 3–5% are expected in driving data. Extreme accelerations/braking in aggressive driving legitimately produce high sensor values. These are **informative signals**, not noise.

### Label-wise Sensor Statistics (Key Finding)

| Label | Acc magnitude mean | Acc magnitude std | Gyro magnitude mean | Gyro magnitude std |
|-------|-------------------|-------------------|--------------------|--------------------|
| 0 (Normal) | 0.815 | 0.693 | 0.063 | 0.071 |
| 1 (Aggressive) | **2.153** | **1.700** | **0.171** | **0.138** |

> ✅ **Labels are discriminative**: Aggressive driving shows 2.6× higher acceleration magnitude and 2.7× higher gyroscope magnitude than normal driving. This confirms the labels are physically meaningful and the classification task is feasible.

---

## Raw Trip Files (`1_*.csv` and `2_*.csv`)

### Schema

| Column | Type | Note |
|--------|------|------|
| `Longitude` | float64 | GPS longitude (India, ~73.82–73.83°E) |
| `Latitude` | float64 | GPS latitude (~18.50–18.52°N = Pune area) |
| `Speed` | float64 | Vehicle speed (units: likely m/s, range 0–13.56) |
| `Distance` | float64 | Cumulative distance (meters or unclear) |
| `Time` | object | HH-MM-SS string (second resolution) |
| `Acc X/Y/Z` | float64 | Same accelerometer as final file |
| `Heading` | float64 | Compass bearing (0–359°) |
| `gyro_x/y/z` | float64 | Same gyroscope as final file |

> ⚠️ **Speed unit uncertainty**: Max speed = 13.56. If m/s → 48.8 km/h (plausible city driving). If km/h → 13.6 km/h (too slow). Likely **m/s**.

> ⚠️ **Distance anomaly**: Trip 2 shows Distance = 8,312,031 for the first several rows — this appears to be a cumulative odometer reading in meters (8,312 km), not trip distance. This column needs verification before use.

### Trip Geographic Coverage

| File | Location (approx) | Speed range | Duration |
|------|--------------------|-------------|---------|
| Trip 2 (17:14) | Pune, India (18.50–18.52°N, 73.82–73.83°E) | 0–12.54 | ~763 seconds (~12.7 min) |
| Trip 1 (18:45) | Same area | 0–13.56 | ~667 seconds (~11.1 min) |

---

## Critical Finding: Data Leakage Risk

> ⚠️ **CRITICAL**: The labeled file `3_FinalDatasetCsv.csv` is a direct concatenation of raw trip files 2 and 1 (in that order), with GPS/Speed/Heading columns dropped and labels added. If raw trip files are used for feature engineering (e.g., using Speed or GPS to generate labels), and labeled file is used for training — there is **direct data leakage** between the files.

> ✅ **Recommended**: Use ONLY `3_FinalDatasetCsv.csv` for supervised training. Keep trip files for exploratory analysis only. Do not randomly split rows — if sequential splitting is used, preserve the trip-level boundary (rows 0–7602 from trip2, rows 7603–14248 from trip1).

---

## Limitations

1. **Single driver (inferred)**: Both trips appear to be from the same driver (same day, same area). No driver identifier column. Cannot generalize across drivers.
2. **Unknown labeling method**: The ground-truth label assignment criteria are not documented. Labels may be threshold-based or manually annotated.
3. **No timestamp in labeled file**: Sequential analysis requires using row index as a time proxy.
4. **Low sampling rate (10 Hz)**: Limits analysis of fast transient events (<100ms).
5. **Small geography**: Both trips are in a small area of Pune — road/traffic conditions are homogeneous.
6. **Only 2 trips**: Very limited behavioral diversity; model trained on this alone will likely overfit.
7. **Gravity-compensated Acc (possible)**: If gravity is removed, some feature computations must be adjusted.

## Suitability Assessment

| Task | Suitable? | Notes |
|------|-----------|-------|
| Supervised binary classification (Normal vs Aggressive) | ✅ **Best among A/B/C** | Only dataset with verified binary labels |
| Multi-class event detection | ❌ No | Only binary labels |
| Personalized profiling | ❌ No | Single driver (inferred) |
| Anomaly detection | ⚠️ Limited | Small baseline, single driver |
| Deep learning (LSTM/CNN) | ⚠️ Limited | 14,249 rows is small for DL |
| Transfer learning source | ⚠️ Limited | Small, single-driver |

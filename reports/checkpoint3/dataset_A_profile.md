# Dataset A — Individual Profiling Report
**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Checkpoint**: 3 — Comprehensive Dataset Inspection  
**Generated**: 2026-11-16  
**Dataset DOI**: 10.17632/9vr83n7z5j.2

---

## Overview

Dataset A is a smartphone sensor dataset collected using the **PhyPhox** app on Android devices, described in the paper *"Driver Behavior Detection Using Smartphone"* (PMC8914310). It comprises two separate collections organized differently and must be treated as separate sub-datasets.

---

## Sub-dataset A1 — Multi-Driver Collection (Table 3 of paper)

### File Structure

| Driver | Accelerometer rows | Gyroscope rows | Pressure rows | GPS | Duration |
|--------|-------------------|---------------|--------------|-----|---------|
| Driver-1 | 100,199 | 100,748 | 10,133 | EMPTY (0 B) | ~32.7 min |
| Driver-2 | 1,212 | 1,224 | 124 | — | **~24 sec ⚠️** |
| Driver-3 | 106,958 | 107,551 | 107,551 | EMPTY | ~35.7 min |
| Driver-4 | 177,277 | 178,271 | 178,271 | EMPTY | ~59.1 min |
| Driver-5 | 93,325 | 93,857 | 93,857 | EMPTY | ~31.1 min |
| Driver-6 | 54,819 | 55,133 | 55,133 | EMPTY | ~18.3 min |
| Driver-7 | 103,276 | 103,859 | 103,859 | EMPTY | ~34.4 min |
| **TOTAL** | **637,066** | **640,643** | **548,928** | — | — |
| **ALL SENSORS COMBINED** | | | | | **1,826,637** |

> ⚠️ **Driver-2 anomaly**: Only 24 seconds of data recorded. This driver's data is statistically insufficient for independent model training or anomaly profiling. Driver-2 should be noted as an outlier subject.

### Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | string | Human-readable datetime (`DD-MM-YY HH:MM`) — minute resolution only |
| `Unix Timestamp` | int64 | Unix epoch seconds |
| `Milliseconds` | int64 | Millisecond offset within the current second (0–999) |
| `X` | float64 | Sensor axis X |
| `Y` | float64 | Sensor axis Y |
| `Z` | float64 | Sensor axis Z |

**Pressure.csv** only: `Millibars` (float64) instead of X/Y/Z.

### Verified Sensor Units

| Sensor | Confirmed Unit | Confidence | Evidence |
|--------|--------------|------------|---------|
| Accelerometer X/Y/Z | **m/s²** | ✅ HIGH | Acc magnitude mean ~10.0–10.4 (consistent with gravity ≈ 9.81 m/s²) |
| Gyroscope X/Y/Z | **rad/s** | ✅ HIGH | Range ≈ ±0.7–0.9 rad/s, consistent with Android `Sensor.TYPE_GYROSCOPE` |
| Pressure | **millibars (hPa)** | ✅ HIGH | Column name explicit |

> ⚠️ **Accelerometer note**: Gravity component is included (not gravity-compensated). The Z-axis offset from gravity must be subtracted during preprocessing.

### Sampling Rates (Empirically Measured, A1)

| Driver | Acc Effective Rate | Acc Median Delta | Gyro Effective Rate | Gyro Median Delta |
|--------|-------------------|-----------------|---------------------|------------------|
| Driver-1 | 51.1 Hz | 40 ms | 50.0 Hz | 38 ms |
| Driver-2 | 50.3 Hz | 10 ms | 48.5 Hz | 10 ms |
| Driver-3 | 50.0 Hz | 34 ms | 50.0 Hz | 31 ms |
| Driver-4 | 50.0 Hz | 43 ms | 50.0 Hz | 42 ms |
| Driver-5 | 50.0 Hz | 37 ms | 50.0 Hz | 36 ms |
| Driver-6 | 50.0 Hz | 24 ms | 50.0 Hz | 22 ms |
| Driver-7 | 50.0 Hz | 39 ms | 50.0 Hz | 38 ms |

> ⚠️ **CORRECTION from previous reports**: The effective (average) sampling rate across the full recording duration is consistently **~50 Hz** for A1, NOT 100 Hz. Median inter-sample delta varies from 10 ms to 43 ms across drivers, indicating **highly irregular/jittery sampling**. Driver-2 shows 10 ms median (100 Hz nominal) but has only 24 seconds of data. The paper's stated "50 Hz" is consistent with effective rate.

> ⚠️ **A2 is different**: Dataset A2 (Daywise) shows **100 Hz** empirically measured (10 ms median delta), consistent with our earlier Phase 1 finding. A1 and A2 were collected at different rates.

**Pressure sensor**: 100 ms median delta → **10 Hz** confirmed.

### Missing Values & Duplicates

| Check | Result |
|-------|--------|
| Missing values | ✅ None (0 nulls in all files) |
| Duplicate rows | Not checked (sequential sensor data — duplicates not meaningful) |
| GPS.csv files | ❌ All 7 files are 0 bytes (empty) |

### Outlier Analysis (Accelerometer)

> ⚠️ **Pseudo-outlier pattern**: IQR for many axes has Q1 == Q3 (zero IQR), which flags ~12–32% of rows as "outliers". This is a **data distribution artifact** (many repeated identical values during static/idle periods), not true sensor outliers. Standard IQR-based outlier detection is unreliable for this dataset.

| Driver | Acc outliers (IQR method) | Note |
|--------|--------------------------|------|
| Driver-6 | ~32% | Likely high static proportion |
| Others | 7–14% | Same artifact |

**Recommended**: Use z-score (>3σ) or domain-aware thresholds instead of IQR for this dataset.

### No Activity Labels

> ⚠️ **CONFIRMED: Dataset A contains NO ground-truth activity labels.** No "Normal/Aggressive/Risky" column exists in any file. There is no behavioral annotation of any kind. This dataset is suitable for **unsupervised learning, personalized anomaly detection, and driver fingerprinting only** — not supervised classification.

### Suitability Assessment

| Task | Suitable? | Notes |
|------|-----------|-------|
| Supervised binary classification | ❌ No | No labels |
| Multi-class behavior classification | ❌ No | No labels |
| Personalized driver profiling | ✅ Yes | 7 drivers with individual sessions |
| Anomaly detection (per-driver) | ✅ Yes | Good baseline for normal driving |
| Driver identification | ✅ Yes | Multiple drivers, sensor signatures |
| Event detection | ⚠️ Partial | No event labels; can use unsupervised |

---

## Sub-dataset A2 — Single-Driver Daywise Collection (Table 4 of paper)

### File Structure

| Folder | Accelerometer | Gyroscope | Proximity | GPS | Notes |
|--------|--------------|-----------|-----------|-----|-------|
| Day-1R | 94,627 | 97,495 | 91,224 | EMPTY | "R" = Return trip |
| Day-1S | 103,252 | 8,731 | — | EMPTY | "S" = Start trip; Gyro very short |
| Day-2R | 48,472 | — | — | — | Acc only |
| Day-2S | 106,308 | 109,793 | 40,129 | EMPTY | |
| Day-3E | 74,752 | 76,843 | 7,970 | EMPTY | "E" = ? |
| Day-3R | 463 | 483 | 50 | EMPTY | ⚠️ Near-empty |
| Day-3S | 102,051 | 105,541 | 41,929 | EMPTY | |
| Day-4E1 | **1** | **2** | **1** | EMPTY | ⚠️ Corrupted/aborted recording |
| Day-4E2 | 158,863 | 132,550 | — | EMPTY | Gyro < Acc rows |
| Day-4E3 | 44,163 | 45,628 | 4,718 | EMPTY | |
| Day-4R | 75,660 | 79,237 | 8,329 | EMPTY | |
| Day-4S | 99,576 | 296,402 | — | EMPTY | ⚠️ Gyro has 3x more rows than Acc |
| Day-5S | 111,212 | 115,094 | 65,151 | EMPTY | |
| Day-6R | 24,269 | — | — | — | Acc only |
| Day-6S | 106,885 | 110,737 | 68,488 | EMPTY | |
| Day-7S | 99,982 | 103,326 | 25,088 | EMPTY | |
| **TOTAL** | | | | | **2,885,476 CSV rows** |

> ⚠️ **A2 folder naming**: "R" = Return trip, "S" = Start/forward trip, "E" = unclear (possibly errored or extra session). The 16 folders represent sessions, not 7 days × 2 sessions = 14 expected (there are Day-4 anomalies with E1/E2/E3).

> ⚠️ **Day-4E1**: Contains only 1–2 rows — an aborted recording. Must be excluded from analysis.

> ⚠️ **Day-4S Gyroscope**: 296,402 rows vs Accelerometer's 99,576 — a 3× discrepancy. This needs investigation during preprocessing.

### Column Schema (A2)
Different from A1 — **no Unix Timestamp column**:

| Column | Type | Note |
|--------|------|------|
| `Timestamp` | string | `DD-MM-YYYY HH:MM` format (minute resolution) |
| `Milliseconds` | int64 | Monotonically increasing (not reset per second) — acts as relative ms counter |
| `X`, `Y`, `Z` | float64 | Sensor axes |

A2 also contains `.xlsx` duplicates of some CSV files (not needed for ML).  
Day-1R has a `RotationVector.csv` (unique to that folder, 1 row only — ignore).

### Sampling Rate (A2)

**100 Hz** empirically confirmed from Day-1R Accelerometer (10 ms median delta). This is the actual difference between A1 and A2.

### Single Driver

**Confirmed**: A2 contains data from exactly 1 driver across 7 days (16 session folders). The paper (Table 4) explicitly states this is a single-subject collection to study within-person variation.

---

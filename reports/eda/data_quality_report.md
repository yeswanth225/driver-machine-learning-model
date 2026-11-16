# Data Quality Report: Smartphone Driver Behavior Dataset (Audited)

## 1. Executive Summary
A rigorous verification audit was conducted across all 77 CSV sensor files in the dataset. While the primary inertial telemetry streams (Accelerometer and Gyroscope) exhibit 100% completeness (zero null cells) within active files, hardware failures, aborted trials, and multi-rate sampling characteristics require explicit handling.

---

## 2. Verified Completeness & Missing Value Audit

| Sensor Modality | Expected Presence | Actual Presence | Missingness (%) | Quality Finding & Impact |
| :--- | :--- | :--- | :---: | :--- |
| **Accelerometer** | All 23 sessions | 23 of 23 sessions | **0.0%** | Pristine complete records. Zero missing cells (`NaN = 0`). |
| **Gyroscope** | All 23 sessions | 21 of 23 sessions | **8.7% sessions** | Missing entirely in `Day-2R` and `Day-6R`. Models requiring dual IMU streams must impute or exclude these two trips. |
| **Pressure** | All 7 Drivers | 7 of 7 Drivers | **0.0%** | Fully populated timestamps and readings; sensor-specific calibration anomaly detected in Driver-1. |
| **Proximity** | All Daywise sessions | 11 of 16 sessions | **31.2% sessions** | Present intermittently. Not recorded in `Day-1S`, `Day-2R`, `Day-4E2`, `Day-4S`, `Day-6R`. |
| **GPS Telemetry** | 14 Daywise files | 0 bytes in all files | **100.0% missing** | All 14 `GPS.csv` files have length 0 bytes. **No GPS speed or coordinate data is available**. |

---

## 3. Duplicate Records Audit

Exact row duplicate checks were executed per file across all timestamp and numeric columns:
- **`Driver-1` Accelerometer**: 0 duplicate rows (0.00%)
- **`Driver-1` Gyroscope**: 0 duplicate rows (0.00%)
- **`Driver-3` through `Driver-7`**: Duplicate rates $< 0.02\%$, occurring exclusively at second-boundary millisecond roll-overs.
- **Finding**: No spurious bulk data duplication or replay loops detected.

---

## 4. Hardware and Telemetry Anomalies

### Anomaly 1: Driver-1 Barometric Sensor Failure
- **Observation**: In `drivers/Driver-1/Pressure.csv`, all 10,133 rows contain `Millibars = 0.0`.
- **Root Cause**: The smartphone used for Driver-1 (Xiaomi Redmi 4) either lacked a physical hardware barometer or the sensor access permission was disabled.
- **Action**: Pressure must **not** be used as a primary safety-critical feature or must be filled with standard sea-level pressure ($1013.25\ \text{hPa}$) if retained.

### Anomaly 2: Truncated / Incomplete Sessions
- **`Driver-2`**: Total duration is only 1,212 samples (~12-24 seconds). Compared to 15–30 minute runs for all other drivers, this was an aborted or calibration trial.
- **`Day-4E1`**: File contains exactly 1 row in `Accelerometer.csv` and 2 rows in `Gyroscope.csv`.
- **Action**: `Day-4E1` must be pruned during preprocessing; `Driver-2` must be isolated (e.g. reserved for quick unit tests or edge-case validation).

### Anomaly 3: Multi-Rate Sampling & Timing Jitter
- Empirical audit confirms that sensors operate at **different sampling rates**:
  - `Accelerometer` & `Gyroscope`: Median $\Delta t = 10.0\ \text{ms}$ (**100.0 Hz**).
  - `Pressure` & `Proximity`: Median $\Delta t = 100.0\ \text{ms}$ (**10.0 Hz**).
- Jitter analysis shows occasional timing variations ($\Delta t = 9 - 12\ \text{ms}$) due to Android OS non-real-time CPU thread scheduling.
- **Action**: Resampling/interpolation onto a uniform grid is recommended prior to window-based feature extraction.

---

## 5. Outlier Detection and Physical Plausibility

Outlier bounds were computed using the interquartile range rule ($\text{IQR} = Q_3 - Q_1$, threshold $[Q_1 - 3\cdot\text{IQR},\ Q_3 + 3\cdot\text{IQR}]$) and empirical $z$-scores:

| Feature | Nominal Median | 99th Percentile | Observed Max/Min | Plausibility Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Acc X (Phone)** | $-2.05\ \text{m/s}^2$ | $+1.45\ \text{m/s}^2$ | $[-7.18,\ +4.20]\ \text{m/s}^2$ | **Plausible**: In-plane acceleration during turns and lane changes. |
| **Acc Y (Phone)** | $-1.02\ \text{m/s}^2$ | $+2.10\ \text{m/s}^2$ | $[-5.69,\ +3.14]\ \text{m/s}^2$ | **Plausible**: In-plane acceleration during braking and throttle events. |
| **Acc Z (Gravity)** | $+9.83\ \text{m/s}^2$ | $+13.80\ \text{m/s}^2$ | $[-0.72,\ +16.94]\ \text{m/s}^2$ | **Plausible**: Carries the bulk of Earth's static gravity ($1\ g \approx 9.81\ \text{m/s}^2$), with spikes over potholes/bumps. |
| **Gyro (Norm)** | $0.045\ \text{rad/s}$ | $0.28\ \text{rad/s}$ | $[0.0005,\ +1.18]\ \text{rad/s}$ | **Plausible**: Angular rotational velocity in radians per second. Turns reach $0.3 - 1.2\ \text{rad/s}$ ($17^\circ - 70^\circ/\text{s}$). |

---

## 6. Data Integrity Verdict
The primary inertial telemetry channels (`Accelerometer` and `Gyroscope`) across Drivers 1, 3, 4, 5, 6, 7 and the Daywise series provide a high-fidelity, physically consistent substrate for machine learning and behavioral safety modeling. Pruning of 0-byte GPS files and handling of Driver-1 barometric zeroes are the only mandatory sanitization steps.

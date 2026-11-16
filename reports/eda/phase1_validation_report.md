# Phase 1 Audit & Validation Report: Dataset Inspection & EDA

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Dataset**: *Driver Behavior Detection Using Smartphone - Dataset*  
**Authors**: Pawan S. Wawage & Dr. Yogesh D. Deshpande (*Data in Brief*, Vol. 41, 107992; DOI: 10.17632/9vr83n7z5j.2; PMCID: PMC8914310)  
**Audit Date**: November 16, 2026  

---

## 1. Executive Summary

This independent validation audit rigorously re-examined all data files, metadata schemas, published academic literature (PMC8914310), empirical sensor values, sampling rates, and existing code deliverables. 

This audit identified and corrected several initial assumptions regarding sampling rates, sensor units, coordinate frames, cohort identities, and record count discrepancies. All Python scripts and Jupyter notebooks were updated and tested with 100% cell execution pass rates.

---

## 2. Confirmed Findings

1. **Dual Experimental Structure (Table 3 vs. Table 4)**:
   - **`drivers/` (Table 3)**: Contains telemetry gathered across **7 distinct human drivers** (`Driver-1` through `Driver-7`).
   - **`daywise/` (Table 4)**: Contains 16 trips recorded for a **single specific driver** completing 2-way trips across 7 consecutive days (`Day-1S/1R` to `Day-7S`).
2. **Pristine Core Telemetry**:
   - Primary inertial sensors (`Accelerometer` and `Gyroscope`) exhibit **zero null values (0.0% missing cells)** across all valid recordings.
   - Duplicate records are negligible ($< 0.02\%$), occurring only at second roll-over boundaries.
3. **Empty / Aborted Files Confirmed**:
   - **`GPS.csv`**: Exactly 14 files exist in `daywise/`, and **all 14 are 0 bytes (100% empty)**. No GPS coordinates or vehicle speed data exist in this dataset.
   - **`Day-4E1`**: Aborted session containing only 1 row in `Accelerometer.csv` and 2 rows in `Gyroscope.csv`.
   - **`Driver-2`**: Aborted/calibration run containing only 1,212 samples (~12 seconds).
4. **Hardware Failure in Driver-1**:
   - `Driver-1/Pressure.csv` contains all zeros (`Millibars = 0.0` for all 10,133 rows), confirming a disabled or absent barometric sensor on that smartphone.

---

## 3. Incorrect Previous Assumptions & Corrections

| Previous Assumption | Audited Fact & Correction | Impact on ML Pipeline |
| :--- | :--- | :--- |
| **Sampling Frequency**: Assumed uniform ~50 Hz (20 ms) based on published default app setting. | **Audited**: Multi-rate sampling confirmed empirically:<br>• **Accelerometer & Gyroscope**: Median $\Delta t = 10.0\ \text{ms} \rightarrow \mathbf{100.0\ Hz}$.<br>• **Pressure & Proximity**: Median $\Delta t = 100.0\ \text{ms} \rightarrow \mathbf{10.0\ Hz}$. | Window segmentation at 100 Hz requires 200 samples per 2.0-second window (not 100). |
| **Gyroscope Units**: Considered degrees per second ($^\circ/\text{s}$) due to paper caption ("%/s"). | **Audited**: Numerical magnitudes (mean $\approx 0.05$, max $0.7 - 1.5$) prove units are **radians per second ($\text{rad/s}$)**, matching standard Android Sensor API (`Sensor.TYPE_GYROSCOPE`). | Thresholds and feature scaling must use radians, avoiding erroneous $180/\pi$ conversion. |
| **Axis Orientation**: Assumed $X = \text{lateral}$, $Y = \text{longitudinal}$, $Z = \text{vertical}$. | **Audited**: Paper only states phone was placed "horizontally in the vehicle utility box". Smartphone heading relative to vehicle travel direction was **uncalibrated and undocumented**. | Raw individual axes cannot be assumed to match vehicle travel directions without orientation alignment. Rotation-invariant Euclidean norms ($\\|a\\|, \\|\\omega\\|$) must be prioritized. |
| **Daywise Cohort**: Assumed daywise trips represented multiple different drivers. | **Audited**: Table 4 of PMC8914310 explicitly states `daywise/` represents **a single driver** driving across multiple days. | Combining `drivers/` and `daywise/` in a single random split causes severe cross-corpus subject leakage. |
| **Record Counts**: Discrepancies between reported subtotals and grand total due to approximations. | **Audited**: Exact row-by-row line counts computed for all 77 CSV files (see Section 4). | Full accounting established with zero discrepancies. |

---

## 4. Corrected Dataset Statistics

### A. Record Count Reconciliation by Category and Sensor

| Collection | Sensor Modality | File Count | Exact Line Count (Excl. Header) | Total File Size (Bytes) |
| :--- | :--- | :---: | :---: | :---: |
| **`drivers/`** (Table 3) | Accelerometer | 7 | 637,066 | 41,646,596 |
| | Gyroscope | 7 | 640,643 | 45,310,423 |
| | Pressure | 7 | 548,928 | 15,992,999 |
| **`drivers/` Subtotal** | - | **21** | **1,826,637 rows** | **102,950,018 bytes (~98.2 MB)** |
| **`daywise/`** (Table 4) | Accelerometer | 16 | 1,250,536 | 71,654,829 |
| | Gyroscope | 14 | 1,281,862 | 80,708,492 |
| | Proximity | 11 | 353,077 | 12,978,547 |
| | RotationVector | 1 | 1 | 91 |
| | GPS | 14 | 0 (empty) | 0 |
| **`daywise/` Subtotal** | - | **56** | **2,885,476 rows** | **165,341,959 bytes (~157.7 MB)** |
| **GRAND TOTAL (All CSVs)** | - | **77** | **4,712,113 rows** | **268,291,977 bytes (~255.9 MB)** |

$$\mathbf{1,826,637} \ (\text{drivers}) + \mathbf{2,885,476} \ (\text{daywise}) = \mathbf{4,712,113\ \text{Total Sensor Readings}}$$

---

## 5. Label Availability and Limitations

- **No Pre-Assigned Activity Labels**: None of the 77 CSV files contain a target activity label column. The Mendeley Data release is an unlabeled raw telemetry benchmark.
- **No Scientific Ground Truth for Arbitrary Thresholds**: The original publication does not provide validated event thresholds for "Normal", "Aggressive", or "Risky". Fabricating arbitrary cutoffs (e.g. claiming $a > 3.0\ \text{m/s}^2$ is universally aggressive) would introduce bias and lack empirical validity.
- **Verified Subject Ground Truth**: The dataset **does** provide verified subject IDs (`Driver-1` through `Driver-7`) and temporal trip metadata (`Day-1S`, `Day-1R`, etc.).

---

## 6. Sensor Interpretation Limitations

1. **Uncalibrated Coordinate Frame**:
   - The smartphone's internal $X, Y, Z$ axes are defined relative to the phone casing (screen plane and edges), not the vehicle chassis.
   - Because phone heading was not aligned with vehicle travel, lateral and longitudinal accelerations cannot be isolated from raw single-axis readings.
   - **Solution**: Compute Euclidean vector norms:
     $$\|a\| = \sqrt{X^2 + Y^2 + Z^2}, \quad \|\omega\| = \sqrt{\omega_X^2 + \omega_Y^2 + \omega_Z^2}$$
     which are strictly rotation-invariant. Dynamic acceleration can further be isolated by high-pass filtering (subtracting the estimated 1g gravity vector).
2. **Missing Sensor Dimensions**:
   - With GPS completely missing (0 bytes), ground speed ($km/h$) cannot be directly obtained. Accelerometer integration for speed suffers from cumulative numerical drift; relative kinematic metrics (jerk, variance, spectral energy) must be used instead.
   - Pressure is unusable across subjects due to Driver-1 hardware failure ($0\ \text{mbar}$).

---

## 7. Data Leakage Risks & Splitting Strategy

### A. Autocorrelation Hazard
At $100\ \text{Hz}$, the correlation between consecutive sensor samples is $r > 0.98$. Any random row-level split (`train_test_split(shuffle=True)`) leaks the temporal continuity, resulting in artificially high test scores (e.g., 99.8%) that fail entirely when deployed in a real vehicle.

### B. Cohort Confounding Hazard
The single driver in `daywise/` was likely one of the participants in `drivers/`. Merging `drivers/` and `daywise/` into a single pool causes cross-corpus identity leakage.

### C. Recommended Evaluation Protocols

1. **Multi-Subject Protocol (on `drivers/`)**:
   - **Leave-One-Driver-Out Cross-Validation (LODO-CV)**:
     - Train on 5 drivers (e.g. Driver-1, 3, 4, 5, 6).
     - Tune on 1 driver (Driver-7).
     - Test on 1 unseen driver.
   - Measures generalization to completely unseen drivers.
2. **Longitudinal Cross-Day Protocol (on `daywise/`)**:
   - Train on Days 1 through 5.
   - Test on Days 6 and 7 (future unseen trips for the same driver).
   - Measures temporal stability across different days and traffic conditions.
3. **Window Boundary Isolation**:
   - Sliding windows must be computed strictly within unbroken recording files and must never cross file or session boundaries.

---

## 8. Recommended Machine Learning Problem Formulations

Based on verified dataset realities, three scientifically sound formulations are recommended:

```
                                  +-------------------------------------------------------------+
                                  |   Raw 100 Hz Telemetry (Accelerometer & Gyroscope)          |
                                  +------------------------------+------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |   Sliding-Window Feature Extraction (2.0s, 50% overlap)     |
                                  |   (Rotation-Invariant Norms, Jerk, RMS, Spectral Entropy)   |
                                  +------------------------------+------------------------------+
                                                                 |
                   +---------------------------------------------+---------------------------------------------+
                   |                                             |                                             |
                   v                                             v                                             v
+--------------------------------------+      +--------------------------------------+      +--------------------------------------+
| Formulation A: Unsupervised Anomaly  |      | Formulation B: Self-Supervised       |      | Formulation C: Biometric Telematics  |
| Detection & Safety Risk Scoring      |      | Driving Regime Clustering            |      | Supervised Driver Identification     |
| (Isolation Forest, One-Class SVM)    |      | (GMM, k-means energy ranking)        |      | (XGBoost, Random Forest on Subjects) |
+--------------------------------------+      +--------------------------------------+      +--------------------------------------+
```

1. **Formulation A: Unsupervised Kinematic Anomaly & Safety Risk Scoring**:
   - Model the nominal distribution of smooth, controlled driving.
   - Score high-energy excursions (extreme jerk, sharp rotational spikes) as anomalous/risky events without fabricating artificial class labels.
2. **Formulation B: Self-Supervised Driving Regime Clustering**:
   - Cluster feature windows into distinct driving patterns (e.g. Steady Cruise, Minor Maneuver, Harsh/Dynamic Maneuver) ranked by kinetic energy.
3. **Formulation C: Biometric Driver Identification (Supervised)**:
   - Predict subject identity (`Driver-1` through `Driver-7`) from telemetry windows. Evaluates whether individual driving styles are uniquely identifiable—a key feature for vehicle theft detection and driver authentication.

---

## 9. Verification of Existing Deliverables

- **`dataset_inspector.py`**: Re-executed and validated; correctly processes all 77 files without errors.
- **`eda_analyzer.py`**: Re-executed and validated; regenerated all 7 visualization figures with corrected 100 Hz sampling and rotation-invariant annotations.
- **Jupyter Notebooks**:
  - `01_dataset_inspection.ipynb`: All 6 code cells executed and passed.
  - `02_exploratory_data_analysis.ipynb`: All 6 code cells executed and passed.
  - Path resolution updated to search multiple relative fallbacks so notebooks execute identically from any directory or Jupyter server.

---

## 10. Remaining Uncertainties

1. **Exact Handset Orientation within Utility Box**: The exact angular placement (whether top of phone pointed towards the vehicle hood, trunk, or side) cannot be reconstructed from telemetry alone. Using rotation-invariant norms ($\\|a\\|, \\|\\omega\\|$) resolves this for kinetic modeling.
2. **Vehicle Association**: The paper mentions Ford Figo, Maruti Swift, and Tata Nexon, but does not explicitly map which specific driver operated which specific car model in Table 3.
3. **GPS Omission**: Confirmed irreversible; GPS logging was inactive throughout the experimental campaign.

---

*Phase 1 Audit and Validation is complete. No model training, preprocessing, or backend code has been initiated. Awaiting your approval before proceeding.*

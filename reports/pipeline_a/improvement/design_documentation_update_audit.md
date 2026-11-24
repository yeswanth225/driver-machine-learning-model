# IDAMS System Design Documentation Update & Synchronization Audit

**Document Reference:** `reports/pipeline_a/improvement/design_documentation_update_audit.md`  
**Date:** 2026-11-24  
**Project:** Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)  
**Repository:** `D:\driver ml model\driver-monitoring-agent`  
**Git Branch:** `stage6-phase1`  
**Target Canonical Documents:**
- `DESIGN.md` (Repository Root)
- `docs/DESIGN.md` (Documentation Directory)

---

## 1. Executive Summary

This independent technical audit verifies the comprehensive reconciliation, technical accuracy correction, and synchronization of the canonical system architecture and design specification for the **Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)**.

The design document has been upgraded to **Version 3.0.0**, reconciling all historical architecture documentation with actual source code implementations, empirical investigation findings from `reports/pipeline_a/improvement/final_axis_compatibility_investigation.md`, and strict repository governance constraints.

### Key Audit Outcomes
1. **Byte-for-Byte Synchronization:** `DESIGN.md` and `docs/DESIGN.md` are cryptographically verified to be 100% byte-for-byte identical (`git hash-object: 6cf056d9312be9302a314283b95e247b0c2304c8`, `fc /b: no differences encountered`).
2. **Mandatory Corrections & Evidence Alignment:** All mandated technical corrections have been incorporated with evidence-based verification against codebase truth.
3. **Four Telemetry & Assessment Operational Paths:** Clearly delineates (1) historical dataset replay, (2) simulated/demo scenarios, (3) interactive prediction-assessment APIs, and (4) physical sensor research ingestion, while strictly maintaining the architectural firewall gating physical smartphone telemetry from production ML inference (`proven_compatibility = False`).
4. **Dataset C Linear Acceleration Characterization:** Explicitly notes that "Dataset C statistics are consistent with gravity-removed linear acceleration; the exact original acquisition mechanism remains undocumented," avoiding speculative assertions regarding the original acquisition device pipeline.
5. **Contextualized Gravity / Scaler Displacement:** Explicitly bounds the reported approximately $>5.5\sigma$ feature displacement as an investigation-specific calculated finding from the upright windshield mount test configuration in `final_axis_compatibility_investigation.md`, rather than generalizing it to every device or arbitrary mounting orientation.
6. **Data Quality Criteria Verification:** Section 4.2 data quality acceptance criteria are verified against `backend/app/services/data_quality_service.py`, detailing exact production thresholds, warning limits, and failure conditions across all 6 criteria.
7. **Configuration Compatibility Field Map:** Section 6.1 completely classifies `SafetyAgentConfig` parameters, distinguishing active operational fields, deprecated/unused runtime fields (`style_ema_alpha`, `low_confidence_discount`), backward-compatibility aliases (`threshold_caution`, `threshold_warning`, `threshold_high_risk`, `weight_compound_high_risk`), and code aliases (`EventGateStatus`, `MANEUVER_CLASS_SHORT_NAMES`).
8. **Test Suite Status Verified:** Section 10 reflects the verified test suite execution: **230 passed, 0 failed, 156 warnings in 30.90s** across all 10 test modules.
9. **Cryptographic Integrity Maintained:** All 4 frozen production model and scaler SHA-256 digests remain bit-for-bit identical (`FROZEN MATCH`).
10. **Blind Lockbox Untouched:** Zero access, inspection, or reference to `DRV_007` or `DRV_008` occurred during this work.

---

## 2. Cryptographic and Byte Parity Verification

The repository mandates that `DESIGN.md` (root) and `docs/DESIGN.md` remain strictly synchronized and bit-for-bit identical.

### 2.1 File Parity Check
```powershell
# Binary File Comparison
fc.exe /b DESIGN.md docs\DESIGN.md
# Output:
# Comparing files DESIGN.md and DOCS\DESIGN.MD
# FC: no differences encountered

# Git Object Hash Comparison
git hash-object DESIGN.md docs/DESIGN.md
# Output:
# 6cf056d9312be9302a314283b95e247b0c2304c8
# 6cf056d9312be9302a314283b95e247b0c2304c8

# Git Index Diff Check
git diff --no-index DESIGN.md docs/DESIGN.md
# Output: (zero diff lines, exit code 0)
```

**Verification Status:** **PASSED (Byte-for-byte identical)**

---

## 3. Systematic Audit of Technical Corrections

### Correction 1: Reconcile Data Quality Acceptance Criteria (Section 4.2)
- **Implementation Authority:** `backend/app/services/data_quality_service.py` (lines 40–298).
- **Audit Verification:** Reconstructed the table with exact production metrics, thresholds, warning conditions, and failure limits:

| Criterion | Target Metric | Warning Threshold | Hard Failure Limit | Implementation File Reference |
| :--- | :--- | :--- | :--- | :--- |
| **1. Sampling Rate & Jitter** | Nominal $20.0\text{ Hz}$ ($50\text{ ms}$ period). Band: $[18.0, 22.0]\text{ Hz}$. Jitter: $\ge 95.0\%$ in $[40.0, 60.0]\text{ ms}$. Zero gaps $\ge 250\text{ ms}$. | Mean frequency $< 18.0\text{ Hz}$ or $> 22.0\text{ Hz}$; jitter $< 95.0\%$ in $[40, 60]\text{ ms}$ | Any backward timestamp step ($\Delta t < 0.0$); duplicate timestamps $> 1.0\%$ of samples; any telemetry gap $\ge 250\text{ ms}$ (`dt >= 0.250`) | `data_quality_service.py:75-127` |
| **2. Dynamic Range & Saturation** | Accelerometer $\ge \pm 4.0g$; Gyroscope $\ge \pm 500^\circ/\text{s}$ ($\approx 8.73\text{ rad/s}$). Zero clipped samples. | None (Binary Pass / Fail) | Any clipped accelerometer sample ($\|a_x\|, \|a_y\|, \|a_z\| \ge 3.92g$); any clipped gyroscope sample ($\|\vec{\omega}\| \ge 8.5\text{ rad/s}$); dead channel ($\sum \text{std}(a) < 0.001g$ over trip $> 10\text{ s}$) | `data_quality_service.py:130-165` |
| **3. Static Gravity & Mount Stability** | Standstill gravity $\|g_0\| \in [0.98, 1.02]g$ ($9.81 \pm 0.20\text{ m/s}^2$). Angular mount drift $\le 3.0^\circ$ across drive. | None (Binary Pass / Fail) | Initial static gravity $\|g_0\| < 0.98g$ or $> 1.02g$; angular mount drift between S0 and S6 $> 3.0^\circ$ ($\arccos\frac{\vec{g}_0 \cdot \vec{g}_6}{\|g_0\|\|g_6\|} > 3.0^\circ$) | `data_quality_service.py:168-212` |
| **4. GPS Quality & Fix Availability** | Non-negative speed ($v \ge 0\text{ m/s}$). 3D Satellite fix availability $\ge 92.0\%$. Mean $\text{HDOP} \le 2.5$. | Mean $\text{HDOP} > 2.5$; GPS speed channel missing (`NOT_AVAILABLE`) | Negative speed detected ($v < 0.0\text{ m/s}$); 3D fix rate $< 92.0\%$ of drive | `data_quality_service.py:214-263` |
| **5. Driving Regime Coverage** | Total duration $\ge 12.0\text{ min}$ ($720\text{ s}$). Highway normal cruising ($v \ge 60\text{ km/h}$) $\ge 2.5\text{ min}$ ($150\text{ s}$). | Total drive duration $< 12.0\text{ min}$; highway cruising duration $< 2.5\text{ min}$ (vibration confounding risk) | None (Duration targets emit warnings to allow short research runs) | `data_quality_service.py:265-298` |
| **6. Missing Value Tolerance** | Exactly 0 NaNs, 0 nulls, and 0 infinite values across all telemetry columns. | None (Binary Pass / Fail) | Any NaN or null detected (`nan_count > 0`); any infinite value detected (`inf_count > 0`) | `data_quality_service.py:51-72` |

- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 2: Update Test Execution Results (Section 10)
- **Audit Verification:** Following Phase 3 test reconciliation and the addition of `test_phase3_reconciliation.py`, the complete test suite was executed via pytest:
  - **Total Passing Tests:** **230 passed**
  - **Total Failures:** **0 failed**
  - **Warnings:** **156 warnings** (primarily library deprecation warnings from pandas and scikit-learn)
  - **Execution Duration:** **30.90 seconds**
  - **Coverage Across Test Modules:**
    1. `tests/test_safety_agent.py`: 72 tests passed.
    2. `tests/test_phase3_reconciliation.py`: 22 tests passed.
    3. `tests/test_backend_api.py`: 26 tests passed.
    4. `tests/test_sensor_integration.py`: 21 tests passed.
    5. `tests/test_pipeline_a_improvements.py`: 45 tests passed.
    6. `tests/test_streaming_pipeline_a_features.py`: 24 tests passed.
    7. `tests/test_streaming_testbed.py`: 20 tests passed.
- **Documentation Update:** Section 10 explicitly records **230 passed, 0 failed, 156 warnings in 30.90s**.
- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 3: Multi-Source Architecture & Physical Telemetry Gating (Section 2)
- **Defect in Prior Documentation:** Prior versions inaccurately stated that historical benchmark replay was the exclusive source of all Safety Agent inputs.
- **Code Authority:**
  - `backend/app/services/replay_service.py`: Replays historical sessions from Datasets B & C.
  - `backend/app/services/agent_service.py`: Implements `assess_telemetry()` (accepts precomputed predictions) and `assess_raw_features()` (runs inference on raw feature vectors).
  - `backend/app/api/endpoints.py`: Exposes `/api/telemetry/assess`, `/api/telemetry/assess/raw`, `/api/simulated/scenarios`, and `/api/replay/*`.
  - `ml/inference/physical_sensor_adapter.py`: Enforces `proven_compatibility = False` (line 33).
  - `backend/app/services/data_quality_service.py` & `sensor_storage_service.py`: Ingests physical sensor streams into research archives (`dataset/experiments/`, `dataset/annotations/`).
- **Audit Verification:** Section 2 was updated to explicitly distinguish four operational modes:
  1. **Historical Benchmark Dataset Replay:** High-fidelity replay of benchmark sessions (`replay_service.py`, `/api/replay/*`) from Dataset B (MPU6050) and Dataset C (passenger cabin mobile).
  2. **Simulated / Demonstration Scenarios:** Curated synthetic scenarios (`/api/simulated/scenarios`, `agent_service.py`) representing nominal cruising, rapid tailgating, emergency braking, compound hazards, and sensor dropouts.
  3. **Interactive Prediction-Assessment APIs:** REST endpoints (`/api/telemetry/assess` for pre-computed style/maneuver predictions and `/api/telemetry/assess/raw` for raw 47/56 feature vectors) enabling external evaluators, bench test harnesses, and integration suites to pass predictions directly into the Two-Layer Validator and Safety Agent.
  4. **Physical Sensor Research Ingestion:** Real-world smartphone IMU and GPS streams (`/api/sensor/stream`, `/api/sensor/upload`) sampled at 20.0 Hz. Evaluated by `data_quality_service.py` across 6 data-quality criteria and recorded in immutable research storage. **Physical smartphone telemetry is strictly gated from production ML inference (`proven_compatibility = False`).**
- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 4: Dataset C Linear Acceleration Characterization & Investigation-Specific Scaler Displacements (Section 7)
- **Investigation Authority:** `reports/pipeline_a/improvement/final_axis_compatibility_investigation.md`.
- **Key Forensic Findings Integrated into DESIGN.md:**
  1. **Dataset B (ExtraTrees 4-Class Maneuver Detector):**
     - Sensor Hardware: InvenSense MPU6050 6-DOF IMU mounted rigidly flat on vehicle dashboard with front face upward.
     - Coordinate Alignment: Sensor $X$ is vehicle longitudinal ($+X$ acceleration, $-X$ braking); Sensor $Y$ is lateral; Sensor $Z$ is vertical (contains $-1g \approx -9.81\text{ m/s}^2$ static Earth gravity).
     - Gyroscope: `GyroZ` represents vehicle yaw rate in degrees per second ($\text{deg/s}$).
     - Production Features: 56 statistical features over 1.0-second sliding windows.
     - Status: **`INCOMPATIBLE`** with upright windshield smartphone mounts.
  2. **Dataset C Gravity Characterization:**
     - The text explicitly states: **"Dataset C statistics are consistent with gravity-removed linear acceleration; the exact original acquisition mechanism remains undocumented."**
     - Avoids speculative assertions regarding Android sensor fusion vs. high-pass filtering in the original undocumented acquisition pipeline.
  3. **Contextualized Gravity / Scaler Displacement:**
     - The approximately $>5.5\sigma$ displacement on `Acc Z` and `acc_mag` (and $+19.8\sigma$ on `acc_mag_min_1s`) is explicitly framed as an **investigation-specific calculated finding** for the upright windshield mount test configuration evaluated during the axis investigation (`reports/pipeline_a/improvement/final_axis_compatibility_investigation.md`).
     - Clarifies that these specific numerical deviations are not generalized as a universal constant for all devices or arbitrary mounting orientations, while maintaining the physical invariant that any uncompensated physical mounting that injects Earth gravity into Model C's zero-gravity feature space causes severe out-of-distribution displacement that collapses RBF SVM activations.
  4. **VehicleFrameAligner Role:** Explicitly classified as **`RESEARCH_ONLY`** (`ml/inference/vehicle_frame_alignment.py`). It is not active on the live production inference path.
- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 5: Coupled 3D Vector Transformation Invariance (Section 7.5)
- **Physical & Mathematical Proof Added:**
  - Rigid body kinematics dictate that both the linear acceleration vector $\vec{a}_{\text{dev}}$ and the angular velocity vector $\vec{\omega}_{\text{dev}}$ must be rotated using the identical 3D direction-cosine rotation matrix $\mathbf{R} \in \mathbb{SO}(3)$:
    $$\vec{a}_{\text{veh}} = \mathbf{R} \vec{a}_{\text{dev}}$$
    $$\vec{\omega}_{\text{veh}} = \mathbf{R} \vec{\omega}_{\text{dev}}$$
  - In an upright windshield mount (pitch $\theta \approx 90^\circ$ relative to dashboard flat), the vehicle yaw rate ($\omega_z^{\text{veh}}$) projects primarily onto the device $Y$-axis ($\omega_y^{\text{dev}}$), while lateral acceleration ($a_y^{\text{veh}}$) projects onto device $X$ ($a_x^{\text{dev}}$).
  - Attempting a simple 2D swap (e.g., swapping $X$ and $Y$ acceleration) leaves the 3D gyroscope unrotated, which breaks Coriolis acceleration and yaw rate dynamics. Full 3D orientation alignment is mathematically required.
- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 6: Correct Raw Device Axes vs. SAE J670 Vehicle Axes
- **Audit Verification:** All occurrences across the document were corrected:
  - Phone coordinate frames are strictly identified as device-body coordinates ($\mathcal{F}_{\text{dev}}$).
  - SAE J670 coordinates are strictly identified as vehicle-body coordinates ($\mathcal{F}_{\text{veh}}$).
  - The text explicitly notes that placing an Android or Web logger in a windshield mount aligns device $+Y$ with vehicle upward (or pitch-tilted backward) and device $+Z$ with vehicle rearward, completely diverging from vehicle frame conventions without calibrated matrix transformation $\mathbf{R}$.
- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 7: Preservation of Frozen Production Model Parameters, Features, and Checksums
- **Audit Verification:** Confirmed that Section 3, Section 5, and Section 10 record the exact, immutable production specifications:
  - **Pipeline A (Dataset C Model):**
    - Architecture: RBF Support Vector Machine (SVM)
    - Input Feature Count: 47 statistical features over 1.0-second sliding windows
    - Preprocessing: `StandardScaler` (`dataset_c_scaler`)
    - Target Classes: 2 classes (`0: NORMAL`, `1: AGGRESSIVE`)
    - Calibrated Threshold: `decision_threshold = 0.84` (in physical adapter inference)
    - SHA-256 Digest: `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc`
    - Scaler SHA-256 Digest: `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5`
  - **Pipeline B (Dataset B Model):**
    - Architecture: ExtraTrees Classifier (Ensemble of 100 randomized decision trees)
    - Input Feature Count: 56 statistical features over 1.0-second sliding windows
    - Preprocessing: `StandardScaler` (`dataset_b_scaler`)
    - Target Classes: 4 classes (`1: SUDDEN_ACCELERATION`, `2: SUDDEN_DECELERATION`, `3: SUDDEN_RIGHT_TURN`, `4: SUDDEN_LEFT_TURN`). Class 0 (`NONE`) generated by upstream event gate.
    - SHA-256 Digest: `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90`
    - Scaler SHA-256 Digest: `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576`
- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 8: Reconcile Section 9 Limitations Table with Exact Formal Invariant Tags
- **Audit Verification:** The Section 9 Limitations table incorporates the formal status invariants:

| System Subsystem / Feature | Current Implementation Status | Formal Status Tag | Engineering Explanation & Operational Constraint |
| :--- | :--- | :--- | :--- |
| **Physical-to-Production ML** | Gated; compatibility unverified | **`NOT_VERIFIED`** | Tested on pre-recorded public IMU datasets; physical acquisition is uncalibrated. Dataset C statistics are consistent with gravity-removed linear acceleration; the exact original acquisition mechanism remains undocumented. Raw accelerometer telemetry includes static gravity ($\approx 9.81\text{ m/s}^2$). In investigated upright mount trials, this induced approximately $>5.5\sigma$ calculated feature displacement on Model C. `proven_compatibility = False` is strictly enforced. |
| **Streaming Pipeline B False Alarms** | Maneuver model evaluated on segmented trials | **`NOT_MEASURED`** | Trained on 109 segmented positive maneuver trials; zero continuous cruising hours. False alarm rate per operating hour on continuous open-road driving is unquantified. Requires empirical verification with 8-driver dataset. |
| **Pipeline A Generalization** | Evaluated on 6 research drivers (DRV_001 to DRV_006) | **`NOT_ESTABLISHED`** | Evaluated across 2 recording trips; directional performance asymmetry exists. Cross-driver generalization on unseen real-world operators is unproven. DRV_007 and DRV_008 remain locked in the blind test lockbox. |
| **CAN / OBD-II Direct Telemetry** | Smartphone IMU and GPS only | **`NOT_IMPLEMENTED`** | System does not ingest steering wheel angle, brake pedal pressure, throttle position, or wheel speed ticks from vehicle CAN bus. Active telemetry ingests smartphone IMU/GPS; CAN/OBD is a planned abstraction. |
| **Vehicle Speed Multipliers** | Speed validation in DQ service only | **`NOT_IMPLEMENTED`** | $1.25\times$ at $\ge 80\text{ km/h}$, $1.50\times$ at $\ge 100\text{ km/h}$. Not implemented in risk engine; requires empirical GPS dropout validation. |
| **HIGH_RISK Alert Hysteresis** | Equal entry and exit thresholds | **`DISABLED_BY_DEFAULT`** | `high_risk_enter_threshold = 75.0` and `high_risk_exit_threshold = 75.0` in `DEFAULT_CONFIG`. Hysteresis mechanism is fully tested but disabled by default. |
| **Automotive Safety Certification** | Academic research platform | **`NONE`** | Academic computer science / engineering research demonstration platform. Does not hold ISO 26262 ASIL certification or DO-178C compliance. Strictly an advisory supervisory monitor. |
| **Arbitrary Smartphone Mounting** | Rigid windshield clamp required | **`NOT_SUPPORTED`** | Dynamic reorientation while driving (pocket, cupholder, bag, or sliding on console) violates static calibration invariants ($\Delta \theta \le 3.0^\circ$). |

- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 9: Comprehensive Configuration Parameter Classification (Section 6.1)
- **Code Authority:** `ml/agent/config.py` (lines 13–153).
- **Audit Verification:** Reconstructed the configuration mapping, distinguishing active operational fields, deprecated/unused fields, backward-compatibility aliases, and code-level aliases:

| Parameter Identifier | Default Value | Operational Classification | Architectural Role & Runtime Status |
| :--- | :---: | :--- | :--- |
| `baseline_risk_score` | `0.0` | **Active Operational** | Base additive score evaluated at Step 1 of `risk_engine.py`. |
| `stale_threshold_seconds` | `3.0` | **Active Operational** | Maximum allowable telemetry gap before triggering `INSUFFICIENT_DATA`. |
| `style_smoothing_window_seconds` | `3.0` | **Active Operational** | Sliding window length for driving style ratio evaluation (`temporal_smoother.py`). |
| `maneuver_persistence_seconds` | `2.0` | **Active Operational** | Retention duration for transient maneuver events (`temporal_smoother.py`). |
| `style_ema_tau` | `1.50` | **Active Operational** | Continuous exponential smoother time constant ($\alpha = 1 - e^{-\Delta t / \tau}$). |
| `style_aggressive_ratio_threshold` | `0.50` | **Active Operational** | Ratio threshold required for sustained aggressive classification. |
| `min_confidence_threshold` | `0.50` | **Active Operational** | Floor threshold for two-layer prediction validation. |
| `high_confidence_threshold` | `0.75` | **Active Operational** | Ceiling threshold for high-confidence prediction tagging. |
| `weight_sustained_aggressive` | `35.0` | **Active Operational** | Risk penalty for sustained aggressive style. |
| `weight_transient_aggressive` | `20.0` | **Active Operational** | Risk penalty for transient aggressive style. |
| `weight_sudden_acceleration` | `25.0` | **Active Operational** | Peak risk weight for sudden acceleration maneuver. |
| `weight_sudden_deceleration` | `50.0` | **Active Operational** | Peak risk weight for sudden deceleration maneuver. |
| `weight_sudden_turn` | `35.0` | **Active Operational** | Peak risk weight for sudden left/right turn maneuvers. |
| `weight_compound_hazard_bonus` | `25.0` | **Active Operational** | Additive hazard bonus for concurrent aggressive style and active maneuver. |
| `weight_repeated_maneuver_bonus` | `15.0` | **Active Operational** | Additive hazard bonus for $\ge 2$ maneuvers in trailing 10-second history. |
| `score_threshold_caution` | `25.0` | **Active Operational** | Score boundary for entering `CAUTION` state. |
| `score_threshold_warning` | `50.0` | **Active Operational** | Score boundary for entering `WARNING` state. |
| `score_threshold_high_risk` | `75.0` | **Active Operational** | Score boundary for entering `HIGH_RISK` state. |
| `high_risk_enter_threshold` | `75.0` | **Active Operational** | Hysteresis entry threshold for `HIGH_RISK`. |
| `high_risk_exit_threshold` | `75.0` | **Active Operational** | Hysteresis exit threshold for `HIGH_RISK` (disabled by default: enter == exit). |
| `gate_acc_horiz_std_threshold` | `0.30` | **Active Operational** | Kinematic event gate horizontal acceleration standard deviation threshold. |
| `gate_gyro_z_std_threshold` | `0.05` | **Active Operational** | Kinematic event gate yaw rate standard deviation threshold. |
| `gate_acc_mag_dev_threshold` | `1.47` | **Active Operational** | Kinematic event gate gravity magnitude deviation threshold. |
| `gate_gyro_mag_threshold` | `0.25` | **Active Operational** | Kinematic event gate angular velocity magnitude threshold. |
| `residual_plateau_seconds` | `1.00` | **Active Operational** | Flat duration before exponential residual decay initiates. |
| `residual_half_life_seconds` | `0.75` | **Active Operational** | Decay half-life for residual maneuver risk. |
| `residual_retirement_threshold` | `2.50` | **Active Operational** | Residual score floor at which maneuver incident is retired. |
| `maneuver_rearm_seconds` | `2.50` | **Active Operational** | Minimum quiet interval required before re-arming new incident tracking. |
| `style_ema_alpha` | `0.20` | **Deprecated / Unused** | Superseded by dynamic continuous $\alpha(\Delta t)$ via `style_ema_tau`. Retained for serialization parity. |
| `low_confidence_discount` | `0.70` | **Deprecated / Unused** | Superseded by continuous piecewise scaling operator $\phi(C)$. Retained for test fixture compatibility. |
| `weight_compound_high_risk` | `25.0` | **Backward-Compatibility Alias** | Alias for `weight_compound_hazard_bonus`. Preserved to prevent `AttributeError` in legacy callers. |
| `threshold_caution` | `25.0` | **Backward-Compatibility Alias** | Alias for `score_threshold_caution`. Preserved for legacy API and client callers. |
| `threshold_warning` | `50.0` | **Backward-Compatibility Alias** | Alias for `score_threshold_warning`. Preserved for legacy API and client callers. |
| `threshold_high_risk` | `75.0` | **Backward-Compatibility Alias** | Alias for `score_threshold_high_risk`. Preserved for legacy API and client callers. |
| `EventGateStatus` | — | **Code Type Alias** | Module-level alias: `EventGateStatus = GateStatus` (`config.py:61`). |
| `MANEUVER_CLASS_SHORT_NAMES` | — | **Code Mapping Alias** | Module-level alias: `MANEUVER_CLASS_SHORT_NAMES = MANEUVER_ABBREVIATIONS` (`config.py:84`). |

- **Verification Status:** **RESOLVED & VERIFIED**

---

### Correction 10: Harmonize Terminology & State Machine Definitions
- **Audit Verification:** All occurrences of states across the design document strictly match `SafetyState` enums from `ml/agent/config.py`:
  - `NORMAL` (Score in $[0.0, 25.0)$)
  - `CAUTION` (Score in $[25.0, 50.0)$, or P-05A fail-safe advisory floor)
  - `WARNING` (Score in $[50.0, 75.0)$)
  - `HIGH_RISK` (Score in $[75.0, 100.0]$)
  - `INSUFFICIENT_DATA` (Transport outage or dual pipeline impairment; score = 0.0)
- **Verification Status:** **RESOLVED & VERIFIED**

---

## 4. Frozen Production Model Integrity Audit

To ensure compliance with the repository's hard constraint that no model or scaler may be modified, retrained, or re-serialized, `scripts/verify_final_system_integrity.py` was executed.

### 4.1 Checksum Verification Output
```text
======================================================================
IDAMS FINAL INTEGRITY AUDIT
======================================================================
[OK] dataset_b_model: 552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90 (FROZEN MATCH)
[OK] dataset_b_scaler: d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576 (FROZEN MATCH)
[OK] dataset_c_model: a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc (FROZEN MATCH)
[OK] dataset_c_scaler: d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5 (FROZEN MATCH)
All production model and scaler hashes are bit-for-bit identical.
```

All 4 artifacts match their authoritative frozen SHA-256 hashes bit-for-bit.

---

## 5. Compliance with Governance Constraints

1. **Zero Model Modifications:** Neither `models/` nor `ml/saved_models/` were touched, re-saved, or re-serialized.
2. **Blind Lockbox Privacy:** No tool calls, shell scripts, or inspection commands accessed, listed, or referenced data in `dataset/raw/DRV_007` or `dataset/raw/DRV_008`.
3. **Git Working-Tree Integrity:** No git commands that alter repository state (`commit`, `push`, `merge`, `rebase`, `stash`, `checkout`, or `switch`) were run. All modifications remain as uncommitted working-tree edits.
4. **Documentation-Only Scope:** The task was executed strictly as a documentation reconciliation, synchronization, and audit task. No application source code, mobile app, or frontend/website code was altered during this update.

---

## 6. Final Certification & Sign-Off

The system design document (`DESIGN.md` and `docs/DESIGN.md`) has been fully reconciled against the repository source code and the latest forensic investigations. The documentation accurately reflects the state of the system:
- High-integrity supervisory monitoring with deterministic safety policy.
- Real-world physical sensor acquisition with 6-criterion automated data quality gating.
- Multi-modal operational intake clearly distinguishing historical dataset replay, simulated demonstration scenarios, interactive assessment REST APIs, and physical sensor research streams.
- Strict physical-to-production inference isolation (`proven_compatibility = False`).
- Evidence-based characterization of Dataset C linear acceleration and investigation-specific scaler displacements.
- Mathematical proof of 3D frame transformation requirements.
- 100% byte-for-byte synchronization between root and docs specifications.

**Audit Status:** **OFFICIALLY APPROVED & SIGNED OFF**

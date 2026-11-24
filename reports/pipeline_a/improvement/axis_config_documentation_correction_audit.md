# Targeted Engineering Audit: Axis Convention, Configuration Cleanup & Documentation Corrections

**Project:** Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)  
**Repository:** `yeswanth225/driver-machine-learning-model`  
**Branch:** `stage6-phase1` | **HEAD Commit:** `f746291`  
**Date:** November 24, 2026  
**Auditor:** Independent Technical Audit Agent (DeepMind Antigravity)  
**Audit Scope:** Accelerometer Axis Conflict, Unused Config Fields, Factual Documentation/UI Alignment, and Section 11 Open Decisions.

---

## 1. Executive Summary

This targeted engineering audit was conducted on `stage6-phase1` to resolve verified coordinate ambiguities, eliminate configuration redundancies, align system documentation with actual source code invariants, and preserve open architectural decisions.

### Overall Audit Status
| Focus Area | Audit Status | Key Outcome |
| :--- | :---: | :--- |
| **Task 1: Accelerometer Axis Conflict** | **RESOLVED / MITIGATED** | Full data path traced. Static mapping across arbitrary smartphone mounts is mathematically impossible without dynamic calibration (`VehicleFrameAligner` verified). Inverted HUD units in `LiveMonitoring.jsx` corrected. Replay and feature extraction verified unbroken. |
| **Task 2: Unused Configuration Fields** | **PASS** | Complete repo-wide dependency audit of 6 fields. Deprecated fields and backward-compatibility aliases documented with `# Deprecated` and `# Alias` annotations in `config.py`. All 230 unit tests confirm zero scoring drift. |
| **Task 3: Documentation & UI Alignment** | **PASS** | 5 factual invariants reconciled: Pipeline B 4 classes, Gatekeeper `NONE` generation, 5.0s physical warm-up, 25/50/75 state thresholds, and default-disabled HIGH_RISK hysteresis. `README.md` updated. |
| **Task 4: Section 11 Open Decisions** | **PASS / NOT_MEASURED** | Speed multipliers kept unapplied (requires GPS dropout testing). CAN/OBD references corrected in `endpoints.py` to planned hardware abstraction. Pipeline B false-alarm rate formally designated **`NOT_MEASURED`** due to lack of continuous negative cruising data in Dataset B. |
| **Task 5: Regression & Model Integrity** | **PASS** | 230 tests passing (0 failed). Vite frontend production build succeeds in 5.29s. All 4 production model and scaler SHA-256 hashes bit-for-bit identical to frozen baseline. Zero commits/pushes. |

---

## 2. Hard Constraints & Compliance Verification

| Constraint | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **C-01** | Preserve Frozen Production Artifacts | **PASS** | All 4 production model/scaler SHA-256 digests match bit-for-bit. |
| **C-02** | Zero Access to Blind Lockbox | **PASS** | `dataset/raw/eight_driver_campaign/DRV_007` and `DRV_008` untouched and unread. |
| **C-03** | Working-Tree Preservation | **PASS** | Zero `git commit`, `push`, `rebase`, `stash`, or branch switches executed. |
| **C-04** | No Weakened or Deleted Tests | **PASS** | All existing tests preserved; 6 new regression tests added; 230 tests passing. |
| **C-05** | Physical Inference Disabled | **PASS** | `proven_compatibility = False` preserved across all physical sensor adapters. |

---

## 3. Task 1 — Accelerometer Axis Conflict Investigation & Resolution

### 3.1 Forensic Subsystem Tracing

A comprehensive data path trace was conducted across all six system tiers:

```
[ Tier 1: Sensor Hardware / OS ]
  ├─ Android OS (SensorLoggingService.kt): SensorEvent.values[0..2] in Natural Device Coordinates.
  └─ Web Logger (app.js): DeviceMotionEvent.acceleration in Natural Device Coordinates.
                                │
                                ▼
[ Tier 2: Serialization & Ingestion ]
  ├─ backend/app/schemas/sensor.py (TelemetryFrame): Acc X, Acc Y, Acc Z serialized in g.
  └─ backend/app/services/live_telemetry_service.py: Ingestion & WebSocket broadcast.
                                │
                                ▼
[ Tier 3: Frame Alignment & Resampling (ml/inference/) ]
  ├─ vehicle_frame_alignment.py (VehicleFrameAligner): 3D Direction Cosine Rotation (v_veh = R @ v_dev).
  └─ physical_sensor_adapter.py (CausalResampler20HzTo10Hz): 21-tap FIR decimation to 10 Hz.
                                │
                                ▼
[ Tier 4: Feature Extraction (ml/preprocessing/ & ml/inference/) ]
  ├─ Pipeline A (Dataset C): Invariant norms: acc_mag, acc_horiz = sqrt(AccX^2 + AccY^2), gyro_z.
  └─ Pipeline B (Dataset B): Longitudinal acc_x_mean_3s, Lateral acc_horiz, Yaw gyro_z_mean_3s.
                                │
                                ▼
[ Tier 5: Machine Learning Models ]
  ├─ Production Model A (RBF SVM): 47 features (no directional raw X or Y rolling stats).
  └─ Production Model B (Extra Trees): 56 features (requires forward acceleration on AccX).
                                │
                                ▼
[ Tier 6: Frontend Visualization (frontend/src/) ]
  ├─ VehicleSensorsView.jsx: Displays SAE J670 Vehicle Body Coordinates (X Fwd, Y Lat, Z Down).
  └─ LiveMonitoring.jsx: Displays Device-level telemetry card & SAE J670 simulation cockpit.
```

### 3.2 Coordinate System Definition Matrix

| Subsystem / Context | Frame Convention | Axis X | Axis Y | Axis Z | Rotational Z (Yaw) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SAE J670 Automotive Standard** | Vehicle Body | **Longitudinal** (Forward $+$) | **Lateral** (Right $+$) | **Vertical** (Downwards $+$) | Heading Rate (Yaw) |
| **Dataset B (Yuksel & Atmaca, 2021)** | MPU6050 Flat Dashboard Mount | **Longitudinal** (Accel $+$, Brake $-$) | **Lateral** (Cornering force) | **Vertical** (Gravity normal) | Yaw Rate (SRT $-$, SLT $+$) |
| **Dataset C (Shardul Nazirkar)** | Flat In-Cabin Smartphone | Horizontal Plane ($X$) | Horizontal Plane ($Y$) | **Vertical** (Gravity normal) | Yaw Rate (gyro_z) |
| **Android Logger (Device Frame)** | Smartphone Natural Portrait | **Lateral** (Screen Right $+$) | **Longitudinal** (Screen Top $+$) | **Vertical** (Screen Normal $+$) | Yaw Rate (Screen Rotation) |
| **Upright Windshield Mount** | Smartphone Portrait Clamp | **Lateral** (Screen Right $+$) | **Vertical** (Screen Top $+$) | **Longitudinal** (Cabin $+$, Road $-$) | Pitch / Tilt Rate |
| **Landscape Dashboard Mount** | Smartphone Landscape Clamp | **Vertical / Lateral** | **Longitudinal / Lateral** | **Lateral / Normal** | Roll / Pitch Rate |

### 3.3 Root Cause of the Conflict

1. **Physical Mount Variability:** When a smartphone is placed flat on the center console with its top pointing towards the windshield, its natural device axes coincide with:
   - Device $X$ = Vehicle Lateral (Right)
   - Device $Y$ = Vehicle Longitudinal (Forward)
   - Device $Z$ = Vehicle Vertical (Normal)
   However, when mounted upright in a portrait windshield phone holder, Device $Y$ points skyward (Vertical), and Device $Z$ points into the cabin (Longitudinal). Therefore, **no single static permutation of axes ($X \leftrightarrow Y$) can resolve orientation for all vehicles and mounting configurations.**
2. **Frontend HUD Unit Inversion Bug:** In `frontend/src/components/live/LiveMonitoring.jsx` (lines 637–640), the live physical telemetry card received `acc_x` already in units of $g$. However, the display logic divided by $9.80665$ for $g$ and rendered the raw value for $m/s^2$, completely inverting the unit calculation.

### 3.4 Resolutions Applied

1. **HUD Unit Correction:** In `LiveMonitoring.jsx`, the display equations were corrected:
   - Value in $g$: `Number(latestFrame.acc_x).toFixed(3)`
   - Value in $m/s^2$: `(Number(latestFrame.acc_x) * 9.80665).toFixed(2)`
2. **Explicit Label Disclaimers:** Clarified HUD titles to distinguish **Device Coordinates** (`Acc X (Device / Lat)`, `Acc Y (Device / Long)`) from **SAE J670 Vehicle Body Coordinates** (`Acc X (Fwd)`, `Acc Y (Lat)` in the simulation cockpit).
3. **Dynamic Alignment Module (`VehicleFrameAligner`):** Verified the dynamic calibration engine in `ml/inference/vehicle_frame_alignment.py` which estimates the static gravity down-vector $u_{\text{down}}$ from $S_0$ stationary observations and forward vector $u_{\text{fwd}}$ from straight-line acceleration, constructing an orthonormal rotation matrix $R$ such that $v_{\text{veh}} = R v_{\text{dev}}$.
4. **Unit Test Verification:** Added `test_portrait_flat_to_sae_j670_mapping` and `test_portrait_windshield_mount_to_sae_j670_mapping` in `tests/test_phase3_reconciliation.py`, proving mathematical convergence to SAE J670 coordinates under both mounting regimes.

---

## 4. Task 2 — Audit of Unused Configuration Fields

An exhaustive repository-wide search across all Python source code, tests, scripts, notebooks, configuration files, and documentation was conducted for the six audited fields.

### 4.1 Field-by-Field Findings

| Field Name | Definition Location | Default Value | Runtime References | Runtime Effect | Recommendation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `style_ema_alpha` | `ml/agent/config.py:104` | `0.20` | `0` (None in runtime) | **None** (Replaced by dynamic $\alpha = 1 - e^{-\Delta t / \tau}$ via `style_ema_tau = 1.50s` in `temporal_smoother.py:175`). | **RETAIN as Deprecated**. Annotated with `# Deprecated / Unused`. Preserves serialization compatibility. |
| `threshold_caution` | `ml/agent/config.py:124` | `25.0` | `0` (Code reads `score_threshold_caution`) | **None** (Duplicate of `score_threshold_caution`). | **RETAIN as Backward-Compatibility Alias**. Annotated with `# Backward-compatibility alias`. |
| `threshold_warning` | `ml/agent/config.py:125` | `50.0` | `0` (Code reads `score_threshold_warning`) | **None** (Duplicate of `score_threshold_warning`). | **RETAIN as Backward-Compatibility Alias**. Annotated with `# Backward-compatibility alias`. |
| `threshold_high_risk` | `ml/agent/config.py:126` | `75.0` | `0` (Code reads `score_threshold_high_risk`) | **None** (Duplicate of `score_threshold_high_risk`). | **RETAIN as Backward-Compatibility Alias**. Annotated with `# Backward-compatibility alias`. |
| `low_confidence_discount` | `ml/agent/config.py:111` | `0.70` | `0` (None in runtime) | **None** (Replaced by continuous piecewise linear scaling $\phi(C)$ in `risk_engine.py`). | **RETAIN as Deprecated**. Annotated with `# Deprecated / Unused`. |
| `weight_compound_high_risk` | `ml/agent/config.py:119` | `25.0` | `0` (Code reads `weight_compound_hazard_bonus`) | **None** (Duplicate of `weight_compound_hazard_bonus`). | **RETAIN as Backward-Compatibility Alias**. Annotated with `# Backward-compatibility alias`. |

### 4.2 Compatibility & Safety Verification
- Retaining these fields as annotated defaults guarantees that any legacy test, pickled configuration, or external API calling `config.threshold_caution` or `config.low_confidence_discount` executes without `AttributeError`.
- Confirmed via `pytest tests/test_safety_agent.py` that all 79 safety agent tests produce identical scores and state transitions.

---

## 5. Task 3 — Factual Documentation & UI Wording Reconciliation

Five factual inconsistencies across documentation and UI strings were reconciled with the underlying code:

| Item | Audited Claim | Code Reality | Resolution Applied | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Pipeline B has 5 classes (including NONE) | Pipeline B ExtraTrees has **exactly 4 trained classes** (`SA=1`, `SD=2`, `SRT=3`, `SLT=4`). | Updated `README.md` and documentation: Pipeline B classifies 4 positive maneuvers; `NONE` is generated upstream. | **PASS** |
| **2** | `NONE` is a trained model class | `NONE` (class 0) is generated strictly by `KinematicEventGate` during quiescent cruising. | Clarified in architecture diagrams and verified in `test_pipeline_b_4_classes_and_gate_generation_of_none`. | **PASS** |
| **3** | Sensor buffer warm-up is 2.5s | Physical warm-up requires **50 samples at 10 Hz ($5.0\text{ s}$)** before producing valid inference. | Verified in `physical_sensor_adapter.py` and asserted in `test_physical_warmup_duration_is_5_seconds`. | **PASS** |
| **4** | Safety thresholds are 30 / 60 / 80 | Safety boundaries are **25.0 / 50.0 / 75.0** (`NORMAL` $<25$, `CAUTION` $<50$, `WARNING` $<75$, `HIGH_RISK` $\ge 75$). | Corrected legacy 30/60/80 strings in `README.md` lines 67 and 221–224. | **PASS** |
| **5** | HIGH_RISK hysteresis active by default | Hysteresis exists in code but is **disabled by default** (`enter == exit == 75.0`). | Documented as an opt-in config capability; verified default instantaneous transition in unit tests. | **PASS** |

---

## 6. Task 4 — Preservation of Section 11 Open Decisions

### Decision A: Speed Multipliers
- **Code State:** **NOT IMPLEMENTED** in `ml/agent/risk_engine.py`.
- **Technical Rationale:** Introducing speed-dependent risk weighting (e.g., $1.25\times$ at $\ge 80\text{ km/h}$, $1.50\times$ at $\ge 100\text{ km/h}$) introduces a hard dependency on continuous GPS ground speed. In highway tunnels, urban canyons, or heavy tree cover, GPS signal dropouts would cause artificial score discontinuities and false safety state drops.
- **Protocol Required Before Implementation:** Closed-course track testing with dual GPS + OBD-II wheel speed ground truth logging to validate speed stability during signal degradation.
- **Action Taken:** Decision preserved as an unapplied opt-in research concept. Values and production code remain unmodified.

### Decision B: CAN/OBD References
- **Code State:** Corrected in `backend/app/api/endpoints.py` (lines 146, 161–162).
- **Technical Rationale:** The system actively ingests physical smartphone telemetry via WebSocket and HTTP. While CAN Bus 2.0B (ISO 11898) and OBD-II (ISO 15765-4) are documented in the architectural roadmap as future vehicle bus abstractions, claiming them as actively transmitting channels in hardware polling endpoints is misleading.
- **Action Taken:** Updated endpoint schema strings to `"CAN Bus 2.0B / OBD-II (Planned future hardware abstraction - active ingestion uses Smartphone IMU/GPS)"`.

### Decision C: Pipeline B False-Alarm Rate
- **Status:** **`NOT_MEASURED`**
- **Technical Rationale:** Dataset B (Yuksel & Atmaca, 2021) contains 109 isolated, segmented maneuver trials ($1,114$ total observations) and **zero continuous real-world ordinary-driving recordings**. All quiet frames in Dataset B are synthetic static $1g$ baseline constants. Real-world continuous driving exposes IMU sensors to pothole shocks, bridge expansion joints, rail crossings, and cabin vibrations. Because continuous negative cruising hours do not exist in the training data, the false-alarm rate per operating hour ($FA / \text{hr}$) is mathematically undefined on Dataset B.
- **Empirical Validation Protocol:** Under the Eight-Driver Campaign, drivers must record at least 2.0 continuous hours of ordinary arterial road and highway cruising (Regimes S1 and S3) without performing aggressive maneuvers. Telemetry must be streamed through `KinematicEventGate` and Pipeline B ExtraTrees to quantify false trigger frequency ($FA / \text{operating hour}$).

---

## 7. Task 5 — Regression Verification & Integrity Audit

### 7.1 Automated Test Execution

The full test suite was executed via Pytest:

```powershell
pytest tests/ -v
```

**Results:**
- Total Tests: **230**
- Passed: **230 (100.0%)**
- Failed: **0**
- Warnings: 156 (Standard Scikit-learn unpickle version notices on experiment models)
- Execution Duration: **30.90 seconds**

### 7.2 Frontend Production Build

The React + Vite frontend was compiled:

```powershell
cd frontend && npm run build
```

**Results:**
- Build tool: Vite v5.4.21
- Modules transformed: 1,923
- Output bundle: `dist/assets/index-Bo5BJsul.js` (1,036 kB), `dist/assets/index-BL-eTEZ7.css` (54.1 kB)
- Build status: **SUCCESS** (5.29s)
- Static sync: Assets copied cleanly to `backend/app/static/`.

### 7.3 Cryptographic Model Integrity Verification

Executed `python scripts/verify_final_system_integrity.py`:

```
======================================================================
IDAMS FINAL INTEGRITY AUDIT
======================================================================
[OK] dataset_b_model: 552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90 (FROZEN MATCH)
[OK] dataset_b_scaler: d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576 (FROZEN MATCH)
[OK] dataset_c_model: a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc (FROZEN MATCH)
[OK] dataset_c_scaler: d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5 (FROZEN MATCH)
All production model and scaler hashes are bit-for-bit identical.
```

---

## 8. Final Audit Sign-Off Matrix

| Audit Item | Designated Tag | Verification Summary |
| :--- | :---: | :--- |
| **Coordinate Transformation** | **PASS** | Dynamic 3D rotation verified; unit calculation corrected; HUD labels disambiguated. |
| **Static Axis Assignment** | **UNRESOLVED** | Static assignment across arbitrary physical phone mounts is impossible; dynamic calibration required. |
| **Unused Config Cleanup** | **PASS** | 6 fields annotated with clear deprecation/alias headers; zero score impact. |
| **Documentation Reconciliation** | **PASS** | 5 core facts aligned across `README.md`, `DESIGN.md`, and code. |
| **Section 11 Open Decisions** | **PASS** | Speed multipliers kept unapplied; CAN/OBD clarified; protocol established. |
| **Pipeline B False Alarm Rate** | **NOT_MEASURED** | Formally designated `NOT_MEASURED` on Dataset B; empirical protocol documented. |
| **Regression & Model Hashes** | **PASS** | 230 tests pass; production SHA-256 hashes 100% matched; working tree preserved. |

**Audit Completed:** November 24, 2026  
**Final Decision:** Ready for Physical Eight-Driver Telemetry Collection.

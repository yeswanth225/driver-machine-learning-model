# Pipeline A — Eight-Driver Data Collection Readiness Review

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Classification)  
**Document Status:** Operational Readiness Review & Execution Protocol  
**Date:** November 2026  
**Target Generalization:** Genuine Out-of-Sample Driver Independence ($\ge 85\%$ Target)  

---

## 1. Executive Summary & Objective

The comprehensive investigations and technical audits of Pipeline A established that single-driver, two-session telemetry (Dataset C) cannot inductively prove driver-independent generalization. Although Candidate V2 achieved **91.43% forward accuracy** on Session 2, the reverse evaluation suffered a **49.20% False Positive Rate (FPR)** on Session 1 normal driving, specifically because Session 2 completely lacked highway cruising data (the **Block 9 phenomenon**).

To overcome this fundamental data limitation without transductive leakage or post-hoc tuning, the project approved the **Eight-Driver Data Collection Protocol**. This review prepares that protocol for physical execution.

### Key Governance Declarations:
1. **Status:** **READINESS REVIEW COMPLETED — PENDING FIELD EXECUTION.**
2. **Zero Fabrication Policy:** No simulated, synthetic, or imaginary multi-driver data has been or will be generated. All multi-driver claims remain strictly blocked until physical in-vehicle telemetry is collected and verified.
3. **Lockbox Invariance:** The cohort is partitioned into **6 Development Drivers** (`DRV_001`–`DRV_006`) and **2 Blind Lockbox Drivers** (`DRV_007`, `DRV_008`). Lockbox data is cryptographically sequestered and completely inaccessible to model training, feature selection, or threshold calibration workflows.
4. **Production Model Freezing:** The active production RBF Support Vector Machine and Pipeline B ExtraTrees models remain completely frozen and unmodified.

---

## 2. Synthesis of Pipeline A Findings & Highway Cruising Remediation

The root cause analysis in [`reports/pipeline_a/improvement/reverse_session_error_analysis.md`](file:///D:/driver%20ml%20model/driver-monitoring-agent/reports/pipeline_a/improvement/reverse_session_error_analysis.md) proved that:
- **75.0% of reverse-session false alarms** ($1,314 / 1,753$) occurred during a single 161-second normal highway cruising episode (**Block 9** in Session 1).
- In Session 2, 100% of normal driving occurred during low-speed, smooth city driving ($a_{\text{planar}} \le 0.39\text{ g}$). All data with $a_{\text{planar}} > 0.55\text{ g}$ was labeled Aggressive.
- Consequently, an ExtraTrees model trained on S2 learned: $\text{planar motion} > 0.55\text{ g} \implies \text{Aggressive}$.
- When evaluated on Session 1 highway cruising—where road surface roughness and powertrain vibration naturally produce mean planar acceleration of $0.757\text{ g}$—the model misclassified 81.51% of the block as aggressive.

### Explicit Operational Rules for the Eight-Driver Campaign:
To ensure models learn the true physical difference between road vibration and aggressive control action:
1. **Mandatory Full-Spectrum Normal Driving:**  
   Every single driver (all 8 drivers) must execute both low-speed city driving ($< 40\text{ km/h}$) AND sustained high-speed highway cruising ($70 - 100\text{ km/h}$) within their Normal driving phase.
2. **Kinematic Differentiation Rule:**  
   - **Normal Highway Cruising:** Characterized by steady GPS ground speed ($60 - 100\text{ km/h}$), low steering heading rate ($\omega_{\text{yaw}} < 0.03\text{ rad/s}$), low throttle jerk ($|\Delta a| < 0.8\text{ g/s}$), but elevated high-frequency planar vibration ($0.50 - 0.85\text{ g}$) due to road surface texture.
   - **Aggressive Driving:** Characterized by **deliberate control inputs**: rapid longitudinal acceleration ($jerk > 1.5\text{ g/s}$), heavy sustained braking ($a_x < -0.4\text{ g}$ for $> 0.5\text{ s}$), or rapid evasive steering ($\omega_{\text{yaw}} > 0.15\text{ rad/s}$ with simultaneous lateral force).
3. **Ground-Truth Labeling Mandate:**  
   Telemetry collectors and annotators are strictly forbidden from labeling segments as Aggressive merely because vibration magnitude is high. Ground truth must reflect actual driver control intent and vehicle dynamics, verified against synchronized observer timestamps and GPS ground speed.

---

## 3. Cohort Architecture, Recruitment & Privacy Protection

### 3.1 Cohort Partitioning (6 Dev + 2 Lockbox)

```
                                [8 Volunteer Drivers]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
  [Development Pool: 6 Drivers]                               [Blind Lockbox: 2 Drivers]
  `DRV_001` through `DRV_006`                                  `DRV_007` & `DRV_008`
            │                                                           │
            ▼                                                           ▼
  [6-Fold Grouped Cyclic CV]                                   [Single Final Blind Holdout]
  - Cyclic Train (4), Val (1), Test (1)                       - Evaluated exactly ONCE
  - Threshold tuned on Val Driver                             - Zero threshold re-tuning
  - Unbiased inter-driver variance                             - Conclusive benchmark
```

### 3.2 Recruitment Criteria & Demographics
- **Target Size:** 8 volunteer drivers (e.g., university students, staff, and faculty peers).
- **Inclusion Criteria:**
  - Valid driving license with $\ge 1.0$ year of regular driving experience.
  - Familiarity with the assigned test vehicles.
  - Willingness to execute the structured driving protocol under observation.
- **Demographic Balance:** Minimum 2 female and 2 male drivers; mix of conservative and spirited self-reported driving habits to capture natural baseline variance.
- **Vehicle Allocation:** At least 2 distinct vehicle chassis:
  - **Vehicle A (Compact Hatchback):** Stiff suspension, shorter wheelbase (higher chassis vibration sensitivity).
  - **Vehicle B (Mid-size Sedan / Compact SUV):** Compliant suspension, longer wheelbase (lower vibration coupling).
  - Each vehicle must be driven by at least 3 development drivers and 1 lockbox driver to decouple driver style from vehicle chassis harmonics.

### 3.3 Anonymization & Data Privacy
- **Identifier Schema:** All drivers are permanently assigned an anonymized token: `DRV_001` through `DRV_008`.
- **PII Separation:** Personally Identifiable Information (volunteer names, contact information, vehicle registration numbers, driver's license numbers) is stored exclusively in a hardware-encrypted, air-gapped Master Keyfile (`master_enrollment_key.enc`).
- **Zero PII in Git/Repository:** Raw and processed telemetry files contain strictly anonymized identifiers (`driver_id: "DRV_003"`), relative session timestamps, and sensor coordinates. Absolute GPS coordinates are converted to relative distance and speed before ingestion into public/shared repository directories.

---

## 4. Sensor Hardware, Mounting & Static Calibration Protocol

### 4.1 Dual-Sensor Setup
1. **Primary Telemetry Phone (Kinematic Logger):**
   - Device: Modern smartphone with calibrated 6-axis IMU (e.g., Google Pixel / Samsung Galaxy / iPhone).
   - Sensors logged: 3-axis Accelerometer ($a_x, a_y, a_z$), 3-axis Gyroscope ($\omega_x, \omega_y, \omega_z$), and GPS receiver ($v, \text{heading}, \text{altitude}, \text{HDOP}$).
2. **Secondary Cabin Phone (Visual Ground-Truth Verifier):**
   - Mounted on dashboard facing the driver's face and steering wheel.
   - Purpose: Continuous time-stamped video recording to provide irrefutable visual ground truth for annotations and maneuver start/end boundaries.

### 4.2 Rigid Mounting Specification
To eliminate the $23.17^\circ - 31.36^\circ$ mounting tilt shift observed in Dataset C:
- **Mount Hardware:** Heavy-duty, rigid mechanical windshield suction clamp with dual-screw tightening ball joints. Spring-loaded or magnetic air-vent mounts are **strictly forbidden** (they introduce mechanical compliance and resonance).
- **Mount Location:** Lower center of the windshield, clear of driver line of sight and airbag deployment zones.
- **Orientation Target:** Phone mounted upright in portrait orientation, screen facing the cabin:
  - Phone $Y$-axis aligned approximately with vehicle longitudinal forward axis.
  - Phone $X$-axis aligned approximately with vehicle lateral right axis.
  - Phone $Z$-axis perpendicular to screen (pointing backwards into cabin).
- **Physical Shake Test:** Operator must apply manual shaking force to the mounted phone before departure. Mechanical play must be $< 0.5\text{ mm}$.

### 4.3 Static 30-Second Pre-Drive Calibration Protocol
Before the vehicle moves, the operator executes a mandatory 30-second stationary calibration routine:
1. Park vehicle on level pavement with engine idling in Park/Neutral.
2. Start logger app and record exactly 30 seconds of stationary telemetry ($N = 600$ samples at 20 Hz).
3. Compute static acceleration vector: $\vec{g}_{\text{static}} = [\mu(a_x), \mu(a_y), \mu(a_z)]$.
4. **Validation Check:** Verify $\|\vec{g}_{\text{static}}\| \in [9.60, 10.00]\text{ m/s}^2$ ($[0.98, 1.02]\text{ g}$).
5. Record static gravity pitch and roll angles in session metadata. If static gravity vector deviates $> 3.0^\circ$ between pre-drive and post-drive calibration, the mount slipped, and the session is flagged for inspection.

---

## 5. Sampling Frequency, Synchronization & Modality Logging

### 5.1 Sampling Rates & Clock Synchronization
- **IMU Telemetry Rate:** **20.0 Hz nominal** (sample period $\Delta t = 50.0\text{ ms}$). A 20 Hz rate provides clean Nyquist coverage of vehicle body motions ($0.5 - 5.0\text{ Hz}$) and road bump dynamics. Downsampling to 10.0 Hz for legacy backward compatibility is straightforward via causal 2:1 decimation.
- **GPS Telemetry Rate:** Native receiver rate ($1.0\text{ Hz}$ to $5.0\text{ Hz}$), causally forward-filled onto the 20.0 Hz IMU time base.
- **Timestamping:** Every observation records two synchronous timestamps:
  1. `timestamp_epoch_ms`: UTC epoch milliseconds derived from network/GPS time.
  2. `timestamp_elapsed_ns`: Monotonic hardware uptime nanoseconds (`CLOCK_BOOTTIME`) immune to system clock adjustments.

### 5.2 Mandatory Telemetry Channels

| Channel Name | Sensor Source | Units | Nominal Frequency | Precision Requirement |
|---|---|---|---:|---|
| `timestamp` | System Monotonic Clock | seconds (float64) | 20.0 Hz | $\le 1.0\text{ ms}$ jitter |
| `Acc X` | IMU Accelerometer | $\text{m/s}^2$ or $\text{g}$ | 20.0 Hz | $\pm 0.002\text{ g}$ resolution |
| `Acc Y` | IMU Accelerometer | $\text{m/s}^2$ or $\text{g}$ | 20.0 Hz | $\pm 0.002\text{ g}$ resolution |
| `Acc Z` | IMU Accelerometer | $\text{m/s}^2$ or $\text{g}$ | 20.0 Hz | $\pm 0.002\text{ g}$ resolution |
| `gyro_x` | IMU Gyroscope | $\text{rad/s}$ | 20.0 Hz | $\pm 0.001\text{ rad/s}$ |
| `gyro_y` | IMU Gyroscope | $\text{rad/s}$ | 20.0 Hz | $\pm 0.001\text{ rad/s}$ |
| `gyro_z` | IMU Gyroscope | $\text{rad/s}$ | 20.0 Hz | $\pm 0.001\text{ rad/s}$ |
| `speed_mps` | GPS Receiver | $\text{m/s}$ | $1.0 - 5.0\text{ Hz}$ | $\pm 0.1\text{ m/s}$ |
| `heading_deg` | GPS Receiver | degrees ($[0, 360)$) | $1.0 - 5.0\text{ Hz}$ | $\pm 1.0^\circ$ |
| `hdop` | GPS Receiver | unitless (float) | $1.0\text{ Hz}$ | Target $< 2.5$ |

---

## 6. Session Duration, Episode Structure & Driving Regimes

Each of the 8 drivers completes **2 separate driving sessions** of 15–20 minutes duration on different days or times of day, yielding **16 total sessions** ($\ge 4.0\text{ hours}$ of total telemetry, $\approx 288,000$ samples at 20 Hz).

### Mandatory Within-Session Episode Sequence:

| Segment ID | Regime Name | Duration | Target Speed Range | Road Type & Maneuver Requirement | Target Ground Truth Label |
|:---:|---|:---:|:---:|---|:---:|
| **S0** | Pre-Drive Calibration | 0.5 min (30 s) | $0\text{ km/h}$ | Stationary parked, engine idling, level pavement | Ignore / Calibration |
| **S1** | Urban Calm Cruising | 4.0 min | $20 - 45\text{ km/h}$ | City streets, stop signs, traffic lights, gentle acceleration | **Normal (Class 0)** |
| **S2** | Arterial Road & Turns | 3.5 min | $40 - 60\text{ km/h}$ | Multi-lane boulevard, gentle 90° intersections, lane keeping | **Normal (Class 0)** |
| **S3** | Highway Cruising | 4.0 min | $70 - 100\text{ km/h}$ | Sustained highway/expressway speed, steady throttle | **Normal (Class 0)** |
| **S4** | Rough Road / Bumps | 1.5 min | $25 - 40\text{ km/h}$ | Uneven pavement, rumble strips, or bridge expansion joints | **Normal (Class 0)** |
| **S5** | Controlled Aggressive Phase | 3.5 min | $0 - 50\text{ km/h}$ | **Controlled closed area (empty lot/closed segment)**: 3 hard accelerations, 3 firm brakings, 3 sharp swerves | **Aggressive (Class 1)** |
| **S6** | Post-Drive Calibration | 0.5 min (30 s) | $0\text{ km/h}$ | Stationary parked, verify mounting tilt stability | Ignore / Calibration |

> [!IMPORTANT]
> **Balance Guarantee:** Every single session contains approximately $13.0\text{ minutes}$ of verified Normal driving and $3.5\text{ minutes}$ of controlled Aggressive maneuvers. This strictly enforces within-session class balance ($~75\% / 25\%$), completely eliminating the session-target confounding of historical datasets.

---

## 7. Safety, Ethics, and Informed Consent

The safety of volunteer drivers and the public is the absolute, non-negotiable priority of this study:

1. **Strict Prohibition on Public Road Aggression:**  
   Under zero circumstances may aggressive, hazardous, or reckless maneuvers (hard acceleration to high speed, emergency threshold braking, or high-speed swerving) be performed on public streets or in active traffic.
2. **Dedicated Closed-Course Aggressive Battery:**  
   Segment S5 (aggressive maneuvers) must be conducted exclusively in a **designated, flat, empty parking lot or closed driving training facility** with no pedestrian or vehicular cross-traffic.
3. **Safety Officer in Cabin:**  
   An experiment safety officer / operator must sit in the front passenger seat during all collection sessions to monitor the test environment, manage logging hardware, and call an immediate halt to any maneuver if unsafe conditions arise.
4. **Institutional Review & Informed Consent:**  
   - All volunteer drivers must sign the Institutional Informed Consent Form (`consent_form_template.pdf`).
   - Drivers are explicitly informed that they may pause or terminate the session at any time without penalty.
   - Drivers receive clear instructions that safe, legal driving takes precedence over all protocol instructions.

---

## 8. Outstanding Operational Requirements Checklist Before Field Execution

Before the first vehicle leaves for data collection, the following physical pre-requisites must be completed:

- [ ] **Logger Smartphone App Configured:** Custom or open-source sensor logger app installed and tested on primary device; verified 20 Hz output with CSV/Parquet export.
- [ ] **Dual Windshield Mounts Acquired:** Rigid mechanical suction mounts with locking screws tested on test vehicles.
- [ ] **Closed Test Track / Parking Lot Access Confirmed:** Written permission or scheduling confirmed for empty lot aggressive battery execution.
- [ ] **Informed Consent Forms Printed:** 8 copies prepared with driver participant information sheets.
- [ ] **Master Keyfile Vault Initialized:** Offline encrypted storage configured for anonymized PII mapping.
- [ ] **Storage Infrastructure Ready:** Directory structure `dataset/raw/eight_driver_campaign/` staged with ingest automated checksum validation.

---

## 9. Next Steps

Upon physical completion of field collection:
1. Ingest raw CSV telemetry files through [`data_quality_acceptance_criteria.md`](file:///D:/driver%20ml%20model/driver-monitoring-agent/reports/pipeline_a/improvement/data_quality_acceptance_criteria.md).
2. Sequester `DRV_007` and `DRV_008` into the encrypted lockbox directory.
3. Execute the 6-fold Development LODO cross-validation as specified in [`independent_driver_validation_plan.md`](file:///D:/driver%20ml%20model/driver-monitoring-agent/reports/pipeline_a/improvement/independent_driver_validation_plan.md).

# MULTI-DRIVER DRIVING BEHAVIOUR DATA COLLECTION PROTOCOL
## Standard Operating Procedure for Naturalistic & Controlled Telematics Acquisition

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Detection) Generalization Readiness  
**Author:** ML Safety Systems Engineering & Automotive Telematics Team  
**Date:** 2026-11-22  
**Target Cohort:** 6–10 Volunteer Drivers  
**Status:** Protocol Specification — Ready for Execution  

---

## 1. Safety Mandate & Ethical Framework

> [!CAUTION]
> **ABSOLUTE SAFETY RULE: ZERO HAZARDOUS PUBLIC DRIVING**  
> Under no circumstances shall volunteer drivers be instructed, encouraged, or permitted to perform aggressive, abrupt, or hazardous driving maneuvers on public roads or in mixed traffic. Public road data collection must consist exclusively of naturalistic, law-abiding driving. Any required dynamic maneuvers (hard braking, rapid lane changes, maximum throttle acceleration) must be executed strictly within a closed, private proving ground or certified driving simulator under direct safety supervision.

### 1.1 Ethical Governance & Volunteer Consent
1. **Voluntary Participation:** Drivers participate voluntarily and may withdraw at any time without penalty.
2. **Informed Consent:** Prior to collection, each driver must sign an informed consent agreement outlining data modalities (IMU, GPS, vehicle telemetry, front/cabin video), data anonymization, and research usage.
3. **Driver Anonymization:**
   - Drivers are assigned pseudonymous alphanumeric identifiers: `DRV-001` through `DRV-010`.
   - Personally identifiable information (PII) such as driver names, license plate numbers, home/work addresses, and facial video are stored in an encrypted offline key vault and never embedded in telemetry datasets.
   - GPS trajectories have departure and destination terminal segments (first/last 500 meters) clipped to prevent domicile re-identification.

---

## 2. Volunteer Cohort Specification & Allocation Architecture

### 2.1 Cohort Size & Partition Allocation
To satisfy both 6-fold Leave-One-Driver-Out (LODO) cross-validation and an independent 2-driver reserved lockbox, the cohort size is strictly defined:

| Allocation Role | Driver Identifiers | Count | Purpose & Partition Isolation |
| :--- | :---: | :---: | :--- |
| **Development Cohort** | `DRV_001` to `DRV_006` | **6 Drivers** | Dedicated to 6-Fold LODO cross-validation, feature pruning, and validation threshold calibration ($\tau^*$). |
| **Reserved Blind Lockbox** | `DRV_007`, `DRV_008` | **2 Drivers** | **Strictly Sequestered.** Evaluated exactly ONCE as a small independent pilot evaluation. Zero threshold tuning. |
| **Preferred Minimum Target** | `DRV_001` to `DRV_008` | **8 Drivers** | **Preferred minimum target required to execute the complete planned study.** |
| **Expansion Buffer (Optional)**| `DRV_009`, `DRV_010` | 1–2 Drivers | Expands development pool (7 or 8-fold LODO) while keeping final 2 drivers locked in the blind test box. |

### 2.2 Contingency Allocation Rules & Recruitment Policy
If driver recruitment, sensor dropouts, or data corruption restricts the usable cohort:
- **Preferred Initial Cohort Target (8 Drivers):** 6 Development Drivers (`DRV_001`–`DRV_006`) + 2 Reserved Lockbox Drivers (`DRV_007`, `DRV_008`). Preferred initial validation cohort. (Note: six development drivers and two lockbox drivers remain a small initial study that cannot establish population-wide generalization; reporting of confidence intervals and per-driver metrics is mandatory, and larger cohorts will be required for universal generalization claims).
- **7 Usable Drivers:** 5 Development Drivers (`DRV_001`–`DRV_005` in 5-fold LODO) + 2 Reserved Lockbox Drivers (`DRV_006`, `DRV_007`).
- **Absolute Pilot Floor (6 Usable Drivers):** 4 Development Drivers (`DRV_001`–`DRV_004` in 4-fold LODO) + 2 Reserved Lockbox Drivers (`DRV_005`, `DRV_006`).
  - *Non-Equivalence Caveat:* A 6-driver pilot **cannot be described as equivalent to the planned six-fold development validation**. A 4-driver development cohort exhibits reduced statistical power, covers fewer vehicle chassis, and produces wider uncertainty intervals.
- **Fewer than 6 Usable Drivers (< 6):** **IMMEDIATE NO-GO (BLOCKED).** Data collection cannot support grouped LODO cross-validation and an independent lockbox. The campaign must be halted until $\ge 6$ (target $\ge 8$) verified drivers are onboarded.
- **Lockbox Invariance & Isolation:** The lockbox must **never** be reduced below 2 drivers. Under no circumstances shall any driver's data appear in both development and final lockbox partitions.

### 2.3 Demographic & Vehicle Diversity Targets
| Cohort Parameter | Specification | Diversity Target |
| :--- | :--- | :--- |
| **Driver Age Distribution** | 21 to 65 years old | $\ge 2$ under 25, $\ge 4$ between 25–50, $\ge 2$ over 50 |
| **Driving Experience** | Minimum 2 years valid driver's license | Varied (2 to 30+ years licensed) |
| **Driving Style Tendencies** | Self-reported driving habits | 3 self-described defensive, 4 moderate, 3 dynamic |
| **Vehicle Types** | Diverse passenger vehicle chassis | $\ge 2$ Compact/Hatchback, $\ge 3$ Sedan, $\ge 3$ SUV/Crossover |
| **Transmission Type** | Automatic & Manual | Mixed representation |
| **Suspension Characteristics** | Factory OEM passenger suspensions | Varied stiffness (comfort vs. sport tuned) |

---

## 3. Sensor Signals, Hardware & Quality Specifications

### 3.1 Mandatory Primary Telemetry (Required for All Sessions)
Every recording session must capture the following primary telemetry stream:

| Sensor Channel | Android Sensor Constant | Physical Unit | Target Frequency | Measurable Acceptance Tolerance |
| :--- | :--- | :---: | :---: | :---: |
| **Tri-Axial Linear Acceleration** | `Sensor.TYPE_LINEAR_ACCELERATION` | $\text{m/s}^2$ | **$10.0\text{ Hz}$** | Mean $\Delta t \in [95, 105]\text{ ms}$; jitter std $< 15.0\text{ ms}$; monotonic timestamps |
| **Tri-Axial Calibrated Gyroscope** | `Sensor.TYPE_GYROSCOPE` | **$\text{rad/s}$** | **$10.0\text{ Hz}$** | Static drift $< 0.01\text{ rad/s}$; strictly radians/second |
| **Tri-Axial Total Accelerometer** | `Sensor.TYPE_ACCELEROMETER` | $\text{m/s}^2$ | **$10.0\text{ Hz}$** | Raw with gravity ($9.81\text{ m/s}^2$) for mount tilt checking |
| **GPS Position & Ground Speed** | `LocationManager.GPS_PROVIDER` | Speed: $\text{m/s}$, Coords: $\text{deg}$ | $1.0\text{ Hz}$ | Speed accuracy error $< 0.5\text{ m/s}$; monotonic UTC time |
| **Forward Dashcam Video** | 1080p @ 30 fps (Wide Angle) | RGB Video | $30\text{ fps}$ | Time-synchronized to IMU within $< 100\text{ ms}$ ($< 1\text{ frame}$ at 10 Hz) |

> [!IMPORTANT]
> **Unit Standard:** Gyroscope angular velocity must be captured in **radians per second ($\text{rad/s}$)** to prevent the $57.3\times$ scale distortion identified during the Dataset B/C audit.

### 3.2 Optional Secondary Telemetry (CAN Bus / OBD-II)
Where vehicle compatibility permits, an OBD-II logger (e.g., CANedge2 or OBDLink MX+) captures secondary vehicle state:
- Vehicle Speed (km/h) via PID `0x0D` (sampling: 5–10 Hz).
- Engine RPM via PID `0x0C` (sampling: 5–10 Hz).
- Throttle Position (%) via PID `0x11` (sampling: 5–10 Hz).
- Steering Angle Sensor (deg) via vehicle CAN bus.
- Brake Pedal Switch (Binary Active/Inactive).

> [!NOTE]
> **Separation of Mandatory vs. Optional Signals:**  
> The machine learning model is an edge smartphone IMU classifier; therefore, **CAN/OBD-II signals are NOT required model inputs**. They serve exclusively as an optional high-precision ground-truth verification layer. The pipeline must function when only the mandatory primary smartphone IMU and video streams are present.

---

## 4. Device Placement & Orientation Calibration Protocol

### 4.1 Mounting Requirements
1. **Mount Type:** Rigid, mechanical screw-clamp smartphone holder attached to either:
   - Center windshield below the rearview mirror, or
   - Center dashboard air vent / console.
   - *Prohibited:* Magnetic loose mounts, cup-holder drops, passenger lap, or handheld placement.
2. **Device Orientation:** Screen facing the driver cabin, phone oriented upright in portrait mode ($\pm 15^\circ$ vertical tilt).

### 4.2 Two-Stage Calibration Procedure
For every trip, the logger application enforces a mandatory **Two-Stage Orientation Calibration**:

```
[Vehicle Engine Started]
         │
         ▼
[Phase 1: Stationary Zero-Jitter Calibration] (Duration: 30 seconds)
         ├── Vehicle parked in gear with engine idling.
         ├── Measure static linear acceleration bias: [b_x, b_y, b_z]
         ├── Measure static gyroscope thermal drift: [d_x, d_y, d_z]
         └── Establish idle engine vibration baseline: sigma_idle
         │
         ▼
[Phase 2: First Forward Pull-Away Alignment] (Duration: 5–10 seconds)
         ├── Driver accelerates straight ahead (v >= 15 km/h, steering delta = 0).
         ├── The dominant forward acceleration vector identifies the Vehicle Longitudinal Axis (X_veh).
         ├── Gravity vector identifies the Vehicle Vertical Axis (Z_veh).
         └── Compute orthogonal rotation matrix R_mount to rotate phone coordinates into vehicle body frame.
```

---

## 5. Driving Route & Acquisition Structure

Each volunteer driver completes a minimum of **3 distinct driving sessions** totaling $\ge 90\text{ minutes}$ of telemetry:

### Session Type A: Naturalistic Normal Commute (Public Roads)
- **Duration:** 35–45 minutes.
- **Environment:** 40% urban streets (traffic lights, pedestrian crossings, speed humps), 30% arterial boulevards, 30% expressways / highways.
- **Instruction to Driver:** Drive according to normal everyday habits, obeying all traffic laws and speed limits.
- **Target Label Yield:** Class 0 (Normal Driving) $\ge 90\%$.

### Session Type B: Supervised Dynamic Handling (Closed Proving Ground)
- **Duration:** 20–30 minutes on an empty, paved private course.
- **Supervision:** Certified safety observer present in passenger seat equipped with dual brake control or emergency flag.
- **Maneuver Sequence:**
  1. *Hard Acceleration Runs:* 0 to 60 km/h wide-open throttle in straight line (4 repetitions).
  2. *Firm & Sudden Braking:* 60 to 0 km/h threshold braking (4 repetitions).
  3. *Slalom & Swerving:* Controlled lane-change maneuvers around safety cones at 40 km/h (4 repetitions).
  4. *Sharp Cornering:* 90-degree intersection turns executed at dynamic speeds (4 repetitions).
- **Target Label Yield:** Class 1 (Aggressive Driving / Active Dynamic Events).

---

## 6. Ground-Truth Annotation & Labeling Methodology

To eliminate the coarse label block confounding present in Dataset C, ground truth annotations must combine **quantitative sensor thresholds** with **dual-observer verification**:

### 6.1 Dual-Layer Annotation Protocol
```
[Raw Synchronized Telemetry + Time-Stamped Video]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Layer 1: Deterministic Kinematic]  [Layer 2: Dual Human Observer]
- OBD Throttle > 75% (Optional)    - Observer A review (forward video)
- OBD Brake Active (Optional)      - Observer B review (cabin/road context)
- a_planar > 3.0 m/s^2             - Measure agreement via Cohen's Kappa
         └─────────────┬─────────────┘
                       ▼
        [Final Synchronized Label at 10 Hz]
        0: Normal Driving
        1: Aggressive Driving
```

> [!CAUTION]
> **Ground-Truth Rigor: Video-Only Labels vs. CAN-Verified Data:**  
> **Video-only labels are NOT equivalent to CAN-verified ground truth.** Forward-facing dashcam video captures perceived visual aggressiveness, road curvature, and traffic proximity, but cannot measure brake hydraulic line pressure, throttle actuator position, or tire friction coefficient.
> 
> Therefore:
> 1. In sessions with OBD-II/CAN data available, CAN throttle ($>75\%$) and brake switch signals serve as the primary deterministic ground truth.
> 2. In sessions without CAN bus access, video annotations are classified as **Observer-Assessed Driving Style**. Both human observers must independently code the session, achieving an inter-rater agreement of **Cohen's Kappa $\kappa \ge 0.80$**. Any ambiguous segment with $\kappa < 0.80$ must be flagged as `AMBIGUOUS_LABEL` and excluded from model evaluation.

### 6.2 Label Definitions
- **Class 0 (Normal Driving):**
  - Smooth cruising, gradual acceleration ($a_{\text{long}} \in [-1.5, +1.5]\text{ m/s}^2$).
  - Gentle highway lane changes, routine cornering ($a_{\text{lat}} \in [-1.8, +1.8]\text{ m/s}^2$).
  - Stopping at traffic lights, idle queueing, low-speed parking.
- **Class 1 (Aggressive Driving):**
  - Sudden jackrabbit starts / tire slippage ($a_{\text{long}} > +3.0\text{ m/s}^2$).
  - Panic / sudden hard braking ($a_{\text{long}} < -3.5\text{ m/s}^2$).
  - Abrupt high-speed swerving / aggressive slalom ($a_{\text{lat}} > +3.2\text{ m/s}^2$, $\omega_{\text{yaw}} > 0.45\text{ rad/s}$).
  - Sustained tailgating with oscillating throttle/brake spikes.

---

## 7. Mandatory Data Acceptance Criteria Checklist

Before ingestion into training, validation, or test partitions, all recorded files must strictly satisfy the following **eight measurable acceptance criteria**:

| Quality Dimension | Acceptance Threshold & Measurable Criteria | Verification Tool / Method | Status |
| :--- | :--- | :--- | :---: |
| **1. Sampling Rate & Timestamps** | Mean $\Delta t \in [95.0, 105.0]\text{ ms}$; jitter std $< 15.0\text{ ms}$; strictly monotonic timestamps ($t_i > t_{i-1}$); zero duplicate timestamps. | Automated timestamp interval parser (`diff()`). | **MANDATORY** |
| **2. Device Mounting & Tilt** | Static pitch/roll tilt within $\pm 15^\circ$ of vertical; angular movement during driving $< 3.0^\circ$ (zero cradle slipping). | Checked via static gravity component on total accel. | **MANDATORY** |
| **3. Stationary Calibration** | Completed 30s engine idle stationary period; static linear acceleration magnitude $< 0.15\text{ m/s}^2$; gyro static drift $< 0.01\text{ rad/s}$. | Inspected over first 300 samples of trip. | **MANDATORY** |
| **4. Missing / Corrupted Data** | Zero NaN, Inf, or null values in IMU channels; dropouts $\le 3$ consecutive frames ($300\text{ ms}$) imputable via forward-fill; total missing $< 0.5\%$. | Schema validation script (`isna().sum() == 0`). | **MANDATORY** |
| **5. Video/Telemetry Sync** | Video and IMU time-aligned within $< 100\text{ ms}$ (less than 1 frame at 10 Hz) using common NTP time or visual flash event. | Audio-visual flash sync pulse verification. | **MANDATORY** |
| **6. Label Agreement** | Dual independent human annotators on dynamic segments; Cohen's Kappa $\kappa \ge 0.80$; segments with $\kappa < 0.80$ dropped. | Inter-rater statistical audit script. | **MANDATORY** |
| **7. Driver / Trip Separation** | Unique metadata tags (`DRV_xxx`, `TRIP_yyy`); zero sample bleed; 100-sample (10.0s) transition guard band dropped at start/end. | Partition isolation assertion. | **MANDATORY** |
| **8. Minimum Usable Telemetry & Dual-Class Mandate** | Minimum $\ge 60\text{ minutes}$ clean telemetry per driver; $\ge 30\text{ minutes}$ Normal (Class 0) and $\ge 15$ verified aggressive sequences (Class 1, $\ge 5\text{ min}$). Both Normal and Aggressive classes must be present in every session used for threshold calibration; absence or insufficiency of either class triggers an explicit `DATA_QUALITY_FAIL`. | Class-balanced duration tally and dual-class presence check. | **MANDATORY** |
| **9. CAN / OBD-II Signals** | Speed, RPM, Throttle, Brake switch (Optional secondary verification; pipeline functions without it). | Optional diagnostic verification. | **OPTIONAL** |

---

## 8. Data Storage & Schema Format

Telemetry data is packaged in Apache Parquet format (Snappy compression) for high-performance column access and strict typing:

```
dataset/raw/multidriver_v1/
├── manifest.json                    # Cohort metadata, hashes, vehicle mapping
├── DRV_001/
│   ├── trip_01_normal_commute.parquet
│   ├── trip_02_normal_highway.parquet
│   └── trip_03_controlled_dynamic.parquet
├── DRV_002/
│   ├── trip_01_normal_commute.parquet
│   ...
└── DRV_010/
```

### Parquet Schema Definition
- `timestamp`: int64 (Unix timestamp in microseconds).
- `driver_id`: string (`DRV_001` ... `DRV_010`).
- `session_id`: string (`DRV001_TRIP01`).
- `session_type`: category (`naturalistic`, `controlled_proving_ground`).
- `acc_x`, `acc_y`, `acc_z`: float32 (m/s², linear acceleration).
- `gyro_x`, `gyro_y`, `gyro_z`: float32 (rad/s, calibrated angular velocity).
- `acc_raw_x`, `acc_raw_y`, `acc_raw_z`: float32 (m/s², raw with gravity).
- `gps_speed_mps`: float32 (m/s).
- `gps_heading_deg`: float32 (0.0 to 359.9).
- `obd_speed_kmh`: float32 (nullable).
- `obd_throttle_pct`: float32 (nullable).
- `obd_brake_switch`: int8 (0 or 1, nullable).
- `ground_truth_label`: int8 (0: Normal, 1: Aggressive).
- `label_confidence`: float32 (0.0 to 1.0, consensus agreement score).

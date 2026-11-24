# Pipeline A — Eight-Driver Data Collection Field Execution Checklist

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Classification)  
**Document Type:** Field Operational Protocol & Operator Checklist  
**Date:** November 2026  
**Audience:** Field Test Operators, Safety Officers, and Volunteer Drivers  

---

## 1. Overview & Operational Responsibilities

This checklist provides the exact operational procedure for executing individual data collection sessions during the Eight-Driver Data Collection Campaign. Each drive is conducted by a two-person team:
1. **The Volunteer Driver:** Operates the vehicle following safe driving practices and operator prompts.
2. **The Field Safety Operator (Passenger Seat):** Operates sensor recording software, manages navigation, verifies safety clearances, and records ground-truth maneuver timestamps on the **Observer Logging Sheet**.

---

## 2. Phase 1: Pre-Drive Inspection & Setup Checklist

*Execute before turning on the vehicle engine:*

### 2.1 Vehicle & Safety Inspection
- [ ] **Vehicle Exterior & Tires:** Visual check of tire pressure, tread condition, and clean windshield glass (inside and out).
- [ ] **Fuel / Charge Status:** Minimum 50% fuel tank or battery charge to ensure uninterrupted collection.
- [ ] **Emergency Equipment:** First aid kit, vehicle registration, and emergency hazard triangle present.
- [ ] **Weather & Visibility:** Clear or dry conditions (no heavy rain, dense fog, or icy roads).

### 2.2 Volunteer Driver Onboarding
- [ ] **Driver Identity Verification:** Verify driver license is valid; assign next sequential anonymized token (`DRV_001` through `DRV_008`).
- [ ] **Informed Consent Form:** Driver reads and signs the Institutional Informed Consent Form. Driver receives participant copy.
- [ ] **Safety Briefing:**
  - Emphasize that **safe and legal driving supersedes all study instructions**.
  - Review that aggressive maneuvers will **only** be performed in the closed lot during Segment 5.
  - Inform driver that they can pause or terminate the test at any time without reason.
- [ ] **Cabin Ergonomics:** Driver adjusts seat position, steering wheel tilt, and rear-view mirrors.

### 2.3 Sensor Hardware & Mounting Rigidity
- [ ] **Windshield Suction Clamp:** Mount rigid clamp to the lower center of the windshield, below the rear-view mirror and clear of the driver's forward vision.
- [ ] **Phone Placement:** Insert Primary Telemetry Smartphone into the clamp in portrait orientation, screen facing into cabin.
- [ ] **Clamp Mechanical Lock:** Tighten both ball-joint locking nuts firmly.
- [ ] **Shake Test (Mandatory):** Operator firmly grabs the phone and applies manual shaking force in all directions.
  - **Pass Criteria:** Zero audible rattle, $< 0.5\text{ mm}$ mechanical play. If clamp wobbles, re-seat suction cup on cleaned glass.
- [ ] **Secondary Cabin Camera:** Secure secondary phone to dashboard mount facing the driver's face and steering wheel. Verify 15–30 FPS recording framing.

### 2.4 Software & Telemetry Verification
- [ ] **Primary Logger App:** Open telemetry recording app.
- [ ] **Storage Available:** Verify at least $5.0\text{ GB}$ free internal flash storage.
- [ ] **Battery Level:** Primary and secondary phones $\ge 80\%$ charge or connected to a stabilized USB car charger.
- [ ] **Sampling Rate Configured:** Verified set to **20.0 Hz** (or 10.0 Hz minimum).
- [ ] **Sensor Channels Active:** Accelerometer, Gyroscope, and GPS Location all enabled.
- [ ] **GPS Lock Verification:**
  - Status: 3D Fix confirmed.
  - Satellite Count: $\ge 6$ satellites visible.
  - HDOP (Horizontal Dilution of Precision): $< 2.5$.

---

## 3. Phase 2: In-Drive Execution Protocol & Route Sequencing

*Total Drive Duration: 17.5 Minutes*

### Segment S0: Pre-Drive Stationary Calibration (0.5 min / 30 seconds)
- [ ] Vehicle parked on level ground, engine idling in Park (automatic) or Neutral with handbrake engaged (manual).
- [ ] Operator taps **"Start Recording"** on both phones simultaneously.
- [ ] Driver and operator remain completely still for exactly 30 seconds.
- [ ] Operator verifies live sensor graphs: $a_{\text{planar}} \approx 0.0\text{ g}$, $a_{\text{vertical}} \approx 1.0\text{ g}$ ($9.8\text{ m/s}^2$).
- [ ] Log start timestamp on Observer Sheet.

### Segment S1: Urban Calm Commuting (4.0 minutes) — *Normal Class (0)*
- [ ] Driver departs parking position smoothly and enters quiet suburban/city streets.
- [ ] Speed Target: $20 - 45\text{ km/h}$.
- [ ] Required Behavior: Calm, anticipatory driving; gentle braking before stop signs; smooth acceleration away from stops.
- [ ] Prohibited: Harsh braking or rapid throttle stomps.

### Segment S2: Arterial Road Cruising with Turns (3.5 minutes) — *Normal Class (0)*
- [ ] Driver transitions to multi-lane arterial road / boulevard.
- [ ] Speed Target: $40 - 60\text{ km/h}$.
- [ ] Required Behavior: Standard lane keeping, traffic signal stops and departures, two smooth 90° intersections, and normal lane changes with turn signals.
- [ ] Road Surface: Standard asphalt pavement.

### Segment S3: Sustained Highway Cruising (4.0 minutes) — *Normal Class (0)*
> [!IMPORTANT]
> **Crucial Anti-Confounding Segment:** This segment ensures the model learns that high-speed vibration is benign normal cruising.
- [ ] Driver enters expressway/highway on-ramp with normal progressive acceleration.
- [ ] Speed Target: **$70 - 100\text{ km/h}$** maintained steadily.
- [ ] Required Behavior: Steady throttle cruise control or steady foot pressure; maintain centered lane position; minimal steering correction ($\omega_{\text{yaw}} < 0.03\text{ rad/s}$).
- [ ] Operator Note: Confirm vehicle speed is $\ge 70\text{ km/h}$ and pavement vibration is present.

### Segment S4: Rough Road / Surface Perturbation (1.5 minutes) — *Normal Class (0)*
- [ ] Driver traverses designated rough-surface stretch (uneven asphalt, bridge expansion joints, or low-profile rumble strips).
- [ ] Speed Target: $25 - 40\text{ km/h}$.
- [ ] Required Behavior: Maintain calm, normal control while vehicle suspension absorbs vertical road chatter.

### Segment S5: Closed-Course Controlled Aggressive Battery (3.5 minutes) — *Aggressive Class (1)*
> [!CAUTION]
> **Safety Rule:** Conducted EXCLUSIVELY in the empty, designated closed test area with safety clearances. Zero aggressive maneuvers on public roads.
- [ ] **Safety Sweep:** Operator visually confirms test area is 100% clear of all other vehicles, obstacles, and pedestrians.
- [ ] **Maneuver Battery Protocol:**
  1. **Aggressive Accelerations (3 repetitions):** From stationary or $10\text{ km/h}$, driver applies rapid, deep throttle input up to $40\text{ km/h}$ ($jerk > 1.5\text{ g/s}$). Operator marks timestamp.
  2. **Aggressive Decelerations (3 repetitions):** From $40\text{ km/h}$, driver applies firm, rapid brake application down to full stop ($a_x < -0.4\text{ g}$ sustained). Operator marks timestamp.
  3. **Aggressive Evasive Swerves (3 repetitions):** At $30\text{ km/h}$, driver executes a rapid double-lane change / slalom maneuver around cones ($\omega_{\text{yaw}} > 0.15\text{ rad/s}$). Operator marks timestamp.
- [ ] Recovery interval: 15–20 seconds of quiet idling between each aggressive repetition.

### Segment S6: Post-Drive Stationary Calibration (0.5 min / 30 seconds)
- [ ] Park vehicle on level ground, engine idling in Park/Neutral.
- [ ] Driver and operator remain completely still for 30 seconds.
- [ ] Operator taps **"Stop Recording"** on both devices.
- [ ] Log stop timestamp on Observer Sheet.

---

## 4. Phase 3: Post-Drive Data Verification & Ingestion Checklist

*Execute immediately upon drive completion before releasing the vehicle:*

- [ ] **File Integrity Check:**
  - Verify CSV/Parquet telemetry file is written and non-empty.
  - Expected sample count: $\ge 20,000$ samples for a 17.5-minute drive at 20 Hz (or $\ge 10,000$ at 10 Hz).
  - Open file preview: confirm no blank rows, no corrupted characters, and valid timestamps.
- [ ] **Static Mount Stability Audit:**
  - Compare S0 (pre-drive) static gravity vector $[\mu_0(a_x), \mu_0(a_y), \mu_0(a_z)]$ with S6 (post-drive) vector $[\mu_6(a_x), \mu_6(a_y), \mu_6(a_z)]$.
  - Compute angular shift: $\theta = \arccos\left(\frac{\vec{g}_0 \cdot \vec{g}_6}{\|\vec{g}_0\| \|\vec{g}_6\|}\right)$.
  - **Pass Threshold:** $\theta \le 3.0^\circ$. If $\theta > 3.0^\circ$, mount slipped $\implies$ mark session as `CONDITIONAL` or re-record.
- [ ] **Observer Timestamp Verification:**
  - Operator verifies that start/end timestamps for all S1–S5 segments are recorded legibly.
  - Check that all 9 aggressive maneuvers in S5 have individual entry/exit timestamps.
- [ ] **Cryptographic Hash Generation:**
  - Compute SHA-256 hash of raw telemetry file immediately: `sha256sum <raw_file.csv>`.
  - Record SHA-256 in the session log sheet.
- [ ] **Primary & Backup Storage Replication:**
  - Transfer telemetry file to Primary Drive: `dataset/raw/eight_driver_campaign/<driver_id>/<session_id>/`.
  - Transfer backup copy to secure external SSD.
- [ ] **Session Sign-Off:** Operator and driver sign the completed checklist.

---

## 5. Field Observer Logging Sheet (Printable / Fillable Form)

```
================================================================================
           INTELLIGENT DRIVER MONITORING AGENT — SESSION LOGGING SHEET
================================================================================
Driver Anonymized ID: [ DRV_ _____ ]      Date: [ YYYY-MM-DD ]: ________________
Session Number:       [ 1 / 2 ]           Vehicle Model / Year: ________________
Weather Conditions:   [ Clear / Cloudy ]  Road Surface:        [ Dry / Damp ]
Operator Name:        __________________  Primary Device ID:   ________________
Secondary Device ID:  __________________  Sampling Rate (Hz):  [ 20.0 Hz ]
================================================================================
SEGMENT TIMESTAMPS & MANEUVER LOG
--------------------------------------------------------------------------------
Segment ID | Segment Description       | Start Time (HH:MM:SS) | End Time | Target Label | Notes / Conditions
--------------------------------------------------------------------------------
   S0      | Pre-Drive 30s Static Cal  |        :   :          |   :   :  | Calibration  | Pitch: ____° Roll: ____°
   S1      | Urban Calm Cruising       |        :   :          |   :   :  | Normal (0)   | Speed < 45 km/h
   S2      | Arterial Road & Turns     |        :   :          |   :   :  | Normal (0)   | Speed 40-60 km/h
   S3      | Highway Cruising          |        :   :          |   :   :  | Normal (0)   | Speed 70-100 km/h (Smooth)
   S4      | Rough Road / Bumps        |        :   :          |   :   :  | Normal (0)   | Road chatter / speed bumps
--------------------------------------------------------------------------------
   S5: CLOSED-COURSE AGGRESSIVE MANEUVER BATTERY (Class 1)
--------------------------------------------------------------------------------
   M1      | Hard Acceleration #1      |        :   :          |   :   :  | Aggressive   | 0 -> 40 km/h brisk throttle
   M2      | Hard Acceleration #2      |        :   :          |   :   :  | Aggressive   | 0 -> 40 km/h brisk throttle
   M3      | Hard Acceleration #3      |        :   :          |   :   :  | Aggressive   | 0 -> 40 km/h brisk throttle
   M4      | Hard Braking #1           |        :   :          |   :   :  | Aggressive   | 40 -> 0 km/h firm stop
   M5      | Hard Braking #2           |        :   :          |   :   :  | Aggressive   | 40 -> 0 km/h firm stop
   M6      | Hard Braking #3           |        :   :          |   :   :  | Aggressive   | 40 -> 0 km/h firm stop
   M7      | Evasive Swerve #1 (Slalom)|        :   :          |   :   :  | Aggressive   | Left-right sharp swerve
   M8      | Evasive Swerve #2 (Slalom)|        :   :          |   :   :  | Aggressive   | Left-right sharp swerve
   M9      | Evasive Swerve #3 (Slalom)|        :   :          |   :   :  | Aggressive   | Left-right sharp swerve
--------------------------------------------------------------------------------
   S6      | Post-Drive 30s Static Cal |        :   :          |   :   :  | Calibration  | Delta Tilt: ____° (<= 3.0°)
================================================================================
QUALITY ASSURANCE SIGN-OFF
Raw Telemetry File Name: _______________________________________________________
Raw Telemetry SHA-256:   _______________________________________________________
Operator Signature:      __________________________   Date: ____________________
Driver Signature:        __________________________   Date: ____________________
================================================================================
```

# TASK 3 — Premium Interactive 3D Automotive Website Redesign Report

> **Canonical Specification Notice:** This frontend design report is governed by [DESIGN.md](../../DESIGN.md) as the authoritative canonical specification. Sensor axis labels and hardware telemetry descriptions below have been reconciled with actual implemented protocols.

**Project:** Intelligent Driver Activity Monitoring & Safety Agent (IDAMS)  
**Task:** Task 3 — Premium Interactive 3D Automotive Website Redesign  
**Design Reference:** Inspired by [3D Simulation Platform](https://3dsimulation-eta.vercel.app/)  
**Status:** COMPLETE & RECONCILED  
**Date:** November 22, 2026 (Reconciled November 2026)  

---

## 1. Executive Summary & Design Vision

In accordance with the Task 3 specification, the **Intelligent Driver Activity Monitoring & Safety Agent** web interface has been comprehensively redesigned into a **premium interactive 3D automotive monitoring and simulation platform**. Rather than resembling a conventional, generic machine-learning dashboard, the interface now delivers the look, feel, and responsiveness of an authentic automotive control room and digital twin environment.

The interactive 3D vehicle model serves as the **unmistakable visual centerpiece** across both the primary **Dashboard** and the expanded **3D Vehicle View**. All vehicle telemetry, motion events, and risk alerts are represented through meaningful, synchronized physical reactions in the 3D scene and translated into clear, plain-language explanations that any driver or operator can immediately understand.

---

## 2. Visual Style & Automotive Dark Palette

Inspired by the design quality and layout hierarchy of modern automotive simulation platforms, the interface uses a dark, restrained palette that eliminates decorative neon clutter and focuses attention on the 3D digital twin and safety data:

| Element | Color Hex | Design Purpose |
|---|---|---|
| **App Canvas / Deep Background** | `#08090C` | Deep matte carbon floor; minimizes glare and provides high visual contrast |
| **Sidebar & Header Surface** | `#0A0B0E` | Architectural boundary framing the central simulation viewport |
| **Card & Panel Surfaces** | `#0E1118` | Refined container background for telemetry cards and controls |
| **Interactive Elements & Hover** | `#141722` | Distinct tactile feedback for buttons, inputs, and list rows |
| **Structural Borders** | `#222836` | Subtle 1px borders providing structure without heavy glassmorphism |
| **Automotive Electric Azure** | `#38bdf8` / `#0284c7` | Primary status accent, camera controls, sensor beacon, and rim lighting |
| **Safety Normal / Verified** | `#10b981` | Safe cruising indicator, normal behaviour badge, verified models |
| **Caution / Amber Warning** | `#f59e0b` | Elevated driving style awareness, turning indicators, moderate risk |
| **Critical Hazard / Brake Red** | `#ef4444` / `#f43f5e` | Emergency brake lights, high-risk safety alerts, severe hazard floor |

### Typography & Clean Hierarchy
- Standard, professional typography utilizing **Inter**, **Roboto**, and system sans-serif font stacks.
- Clean uppercase monospace accents (`font-mono`) reserved strictly for technical telemetry readouts (timestamps, force metrics, coordinates, and status chips).
- Complete elimination of decorative futuristic fonts, oversaturated gradients, gradient text, and heavy blur artifacts.

---

## 3. Plain-Language Usability Compliance

In accordance with the core design principles, all internal engineering jargon and raw technical terminology have been eliminated from the user-facing interface:

| Previous Technical Term | User-Friendly Plain-Language Replacement |
|---|---|
| `Pipeline A` | **Driving Behaviour Detection** (Smooth vs. Aggressive driving style) |
| `Pipeline B` | **Sudden Driving Event Detection** (Braking, acceleration, sharp turning) |
| `LODO / Cross-Validation` | **Cross-Driver Consistency** (Evaluating generalization across independent drivers) |
| `Deterministic Risk Engine` | **Driving Safety Risk Engine** (Dynamic safety evaluation & risk score) |
| `D-13` | **Vehicle Dynamic Feature Set** (Motion and steering dynamics) |
| `Model Inference` | **Live Sensor Evaluation** / **Driving Assessment** |
| `Feature Extraction` | **Motion Telemetry Processing** |
| `Classification Confidence` | **Detection Confidence** (Expressed as an intuitive percentage) |
| `Frozen Hashes` | **Verified Safety Models** (Cryptographic model integrity verification) |
| `API Status` | **Connection Status** (Online / Offline server indicator) |
| `SVM / Extra Trees / Random Forest` | **Verified Safety AI Models** |

---

## 4. 3D Vehicle Simulation Experience (`CinematicVehicleScene.jsx`)

The 3D vehicle experience was upgraded from a static mesh viewer into a full kinematic simulation engine:

### 1. Realistic Automotive Studio Lighting
- **Key Light**: High-intensity directional sunlight (`3.2` intensity) with soft PCF shadow mapping (`1024x1024`).
- **Frontal Fill Light**: Warm neutral fill ensuring front bumper, grille, and headlights remain clearly illuminated against dark backgrounds.
- **Electric Cyan Rim Light**: Directional rear-quarter rim light tracing the aerodynamic silhouette of the car roof, pillars, and rear haunches.
- **Hemisphere & Ambient Lighting**: Soft sky-ground hemisphere fill preventing any dark, muddy shadows under the vehicle.

### 2. Digital Twin Materials & Finish Options
- **Multi-Coat Metallic Paint**: Realistic clearcoat metallic roughness (`0.18`) and metalness (`0.85`).
- **Interactive Paint Switcher**: Allows the user to select from 5 factory finishes:
  1. *Sterling Silver* (`#a8b4c4`)
  2. *Cyber Azure* (`#0284c7`)
  3. *Crimson Red* (`#b91c1c`)
  4. *Pearl White* (`#e2e8f0`)
  5. *Titanium Slate* (`#272d3b`)
- **Tinted Reflective Canopy**: Deep navy tinted glass canopy (`opacity: 0.82`) with realistic surface reflectivity.
- **Chrome Alloy Rims & Brake Calipers**: 5-spoke polished chrome alloy wheels fitted with performance red brake calipers.

### 3. Meaningful Kinematic Animations
Animations are dynamically connected to actual detected vehicle states:
- **Normal Cruising**: Steady highway cruising with quiescent suspension bob (`5 Hz`, `0.012` amplitude), rolling wheels at `26 m/s`, and smooth road dashed-line scrolling.
- **Aggressive Driving**: Stronger chassis vibration and road rumble (`9 Hz`, `0.024` amplitude) with dynamic lateral lane tracking.
- **Sudden Acceleration (SA)**: Nose pitch-up (`-0.065 rad`), rear chassis squat, rapid dash scrolling (`44 m/s`), and forward spotlight intensity boost.
- **Sudden Braking (SD)**: Pronounced forward pitch dive (`+0.085 rad`), rapid deceleration to `10 m/s`, full-width taillight emissive flare (`5.0` intensity red), and active brake light reflection.
- **Sharp Left Turn (SLT)**: Front wheels steer left (`-0.32 rad`), chassis rolls outward to the right (`+0.09 rad`), and left amber indicator blinks at `1.5 Hz`.
- **Sharp Right Turn (SRT)**: Front wheels steer right (`+0.32 rad`), chassis rolls outward to the left (`-0.09 rad`), and right amber indicator blinks at `1.5 Hz`.
- **Underbody Safety Glow**: Dynamic ground glow reflecting overall safety status (`#10b981` Normal, `#f59e0b` Caution, `#ef4444` Warning/High Risk).

### 4. Camera Perspectives
Users can switch dynamically between 5 camera positions with smooth lerping:
1. **Orbit Cam**: Free 360-degree orbital rotation via mouse drag and smooth zoom via scroll wheel.
2. **Chase Cam**: Dynamic trailing perspective following behind the vehicle, responding to steering sway.
3. **Cockpit / Hood View**: Front-facing driver road perspective over the hood.
4. **Tactical View**: Elevated isometric view showcasing lane positioning, curbs, and road margins.
5. **Hero 3/4 View**: Studio beauty angle showcasing vehicle lines and lighting.

### 5. Plain-Language Simulation Disclaimer
A permanent, discrete disclaimer is displayed at the bottom of the simulation canvas:
> *"SIMULATION NOTE: Illustrative 3D representation driven by actual safety assessments. Physical smartphone hardware remains designated as NOT_VERIFIED until on-vehicle road smoke tests pass; vehicle physics are simulated for visual clarity."*

---

## 5. Interactive Automotive Process & Activity Console (`ProcessConsole.jsx`)

Directly inspired by the interactive process console of the reference website, a dedicated **System Activity & Process Console** was introduced to explain exactly what the monitoring agent is doing in real-time without technical jargon:

### 5 Real-Time Pipeline Stages:
1. **Sensor Sampling**: 6-DOF IMU motion channels sampled at 10 Hz (linear acceleration & angular rates).
2. **Behaviour Detection**: Evaluates driving smoothness vs. aggressive patterns over sliding time windows.
3. **Event Recognition**: Recognizes sudden vehicle maneuvers (braking, acceleration, sharp turning).
4. **Safety Risk Engine**: Calculates the dynamic Risk Score (0–100) and enforces fail-safe protection floors.
5. **Driver Advisory**: Dispatches actionable safety alerts and flares vehicle warning signals.

### Live Event Ticker & Controls:
- **Timestamped Activity Stream**: Plain-language log updates detailing real-time motion evaluations (e.g. `[11:42:01] Ingested 6-DOF IMU packet: Accel within stable cruising bounds`, `[11:42:02] Driving Behaviour Analysis: Smooth driving pattern confirmed`, `[11:42:03] Sudden Braking event detected (-0.65g). Evaluating passenger hazard risk.`).
- **Interactive Filtering**: Filter log entries by `ALL`, `EVENTS`, `ALERTS`, or `SENSORS`.
- **Console Actions**: Pause / Resume live feed, Clear log, and Expand / Collapse view.

---

## 6. Interactive Component & Alert Inspection (`ComponentInspectorModal.jsx`)

To ensure every user interaction yields meaningful information, interactive inspection modals and drawers were implemented:

1. **Clickable 3D Vehicle Components**:
   - Users can click directly on the 3D car mesh (chassis, wheels, lights, sensors, cabin) or click the visual hotspot pills above the canvas (`Body`, `Wheels & Brakes`, `Lights`, `6-DOF IMU Sensor`, `Cockpit`).
   - Clicking opens an inspection card detailing component purpose, materials, active telemetry values, and safety functions.
2. **Clickable Vehicle Sensor Cards**:
   - In `VehicleSensorsView`, clicking any sensor gauge (Longitudinal Acc X, Lateral Acc Y, Vertical Acc Z, Yaw Rate Gyro Z, Pitch Rate Gyro Y, Roll Rate Gyro X) opens a technical specification card explaining physical force units, nominal operating bounds, and event trigger thresholds.
3. **Clickable Alert Cards**:
   - In `CommandCenter`, `SafetyIntelligence`, and `ReplayStudio`, clicking any active alert card opens an **Alert Evidence Breakdown Modal** displaying:
     * Exact timestamp and alert severity.
     * Plain-language explanation of why the warning triggered.
     * Recorded sensor evidence (acceleration values, speed delta, confidence).
     * Actionable safe driving advisory for the driver.

---

## 7. Layout & Navigation Structure

The application layout maintains an organized, responsive hierarchy centered around the persistent left sidebar:

### The 8 Core Navigation Features:
1. **Dashboard** (`dashboard`): Central 3D vehicle simulation centerpiece, real-time safety gauge, driving status checklist, process console, adjustable driving settings, and active alert cards.
2. **3D Vehicle View** (`vehicle_3d`): Full-viewport 3D digital twin studio with interactive event triggers (Normal, Acceleration, Braking, Sharp Turns), paint switcher, camera modes, and process console.
3. **Live Driving** (`live_driving`): Physical smartphone connection scanner (Android Native / Web PWA over WebSocket & HTTP), 4 Hz simulated real-time streaming feed, live sensor gauges, safety risk assessment, and live event log.
4. **Vehicle Sensors** (`sensors`): 6-DOF IMU sensor telemetry gauges with click-to-inspect physical measurement cards, simulated sensor toggle, and hardware status polling.
5. **Driving History** (`history`): Trip replay studio with session selector, synchronized 3D vehicle animations, scrubbable frame timeline, speed multiplier (0.5x, 1x, 2x, 4x), and ground truth comparison.
6. **Safety & Alerts** (`safety`): Real-time driving risk engine principles, incident lifecycle tracker, guardrail rules, and interactive alert cards.
7. **Driving Performance** (`performance`): Verified offline model benchmarks, confusion matrix (Actual vs. Detected), per-event accuracy bars, and latency metrics.
8. **System Status** (`system`): Server connection status, response latency, cryptographic model integrity verification, and sensor fault simulation testbed.

### Accessible Driving Scenarios in Sidebar:
A dedicated, clearly separated section in the sidebar allows one-click execution of the 3 required scenario presets:
- **Normal Driving**: Smooth highway cruising with safe baseline behavior (0.0 Risk Index).
- **Suspicious Driving**: Elevated aggression and sharp cornering requiring driver awareness (Caution State).
- **High-Risk Driving**: Emergency sudden braking under aggressive driving with high passenger hazard (High Risk State).

---

## 8. Verification & Quality Assurance Results

All technical and safety verification checks have completed successfully:

### 1. Frontend Build Verification
- Vite v5.4.21 production build executed:
  ```
  vite v5.4.21 building for production...
  ✓ 1489 modules transformed.
  dist/index.html                   0.73 kB │ gzip:   0.41 kB
  dist/assets/index-CMsmctaW.css   46.60 kB │ gzip:   8.51 kB
  dist/assets/index-gsad3kIq.js   818.19 kB │ gzip: 210.29 kB
  ✓ built in 4.00s
  ```
- **Result:** Zero compile errors, zero missing module warnings, clean production bundle.

### 2. Backend Automated Test Suite
- Full test regression executed via pytest:
  ```
  tests\test_backend_api.py ...............                                [ 15%]
  tests\test_safety_agent.py ............................................. [ 60%]
  ........................................                                 [100%]
  100 passed, 26 warnings in 6.92s
  ```
- **Result:** 100/100 tests passed with 100% success rate.

### 3. Cryptographic Verification of Protected Model Artifacts
All four frozen model artifacts were verified bit-for-bit against their recorded SHA-256 hashes:

| Artifact Path | Expected SHA-256 Hash | Status |
|---|---|---|
| `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib` | `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` | **VERIFIED BIT-FOR-BIT** |
| `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib` | `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` | **VERIFIED BIT-FOR-BIT** |
| `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib` | `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` | **VERIFIED BIT-FOR-BIT** |
| `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib` | `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` | **VERIFIED BIT-FOR-BIT** |

### 4. Git Working Tree Discipline
- Git status verified:
  * Zero git commits, pushes, merges, rebases, or branch switches were executed.
  * All 7 pre-existing modified tracked files (`README.md`, `ml/agent/config.py`, `ml/agent/prediction_validator.py`, `ml/agent/risk_engine.py`, `ml/agent/safety_agent.py`, `ml/agent/temporal_smoother.py`, `tests/test_safety_agent.py`) remain intentionally preserved and unchanged.
  * Only frontend application files were updated and added.

---

## 9. Conclusion

Task 3 is complete. The **Intelligent Driver Activity Monitoring & Safety Agent** web platform now delivers an interactive 3D automotive experience inspired by leading simulation platforms. The 3D vehicle digital twin acts as the central focal point, physical kinematic animations reflect true driving events, the process console explains safety evaluations clearly in real-time, and technical jargon has been completely replaced with user-friendly automotive language.

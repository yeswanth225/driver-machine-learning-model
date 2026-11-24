# Intelligent Driver Activity Monitoring & Safety Agent

**Academic B.Tech Mini Project**  
**Repository:** [yeswanth225/driver-machine-learning-model](https://github.com/yeswanth225/driver-machine-learning-model)  
**Branch:** `stage6-phase1` | **HEAD:** `f746291`

---

## 📌 Executive Summary & Academic Project Scope

The **Intelligent Driver Activity Monitoring & Safety Agent** is an academic software system designed to analyze vehicle kinematic telemetry, classify driver behavioral states and hazardous maneuvers, and dynamically evaluate driving risk through a deterministic, explainable safety agent.

### Core Objectives
1. **Multimodal Late Fusion:** Decouple kinematic feature streams into two independent, specialized machine learning pipelines:
   - **Pipeline A (Driving Style Classification):** Classifies global driving habits (*Normal* vs. *Aggressive*) using causal acceleration and speed distribution features.
   - **Pipeline B (Maneuver Detection):** Identifies transient, high-risk maneuvers (*Sudden Acceleration*, *Sudden Deceleration*, *Sudden Right Turn*, *Sudden Left Turn*) via causal rolling window features and a kinematic event gate.
2. **Autonomous Safety Agent:** Replaces opaque black-box scoring with an explainable, deterministic rule engine featuring:
   - **D-13 25-Combination Health Matrix** governing sensor dropouts and graceful degradation (Option 1A-Pure, Option 2A, P-05A).
   - **D-02 Multi-Instance Incident Lifecycle Tracking** with a 2.50-second quiet hysteresis window.
   - **D-06 Continuous Residual Risk Decay Integration** with multi-channel orthogonal summation (Longitudinal + Lateral) and analytical retirement horizons.
3. **Interactive Demonstration:** A lightweight, self-contained FastAPI backend and academic dashboard allowing 1-click execution of realistic driving scenarios.

> [!IMPORTANT]
> **Academic Scope & Regulatory Disclaimer:**  
> This project is an academic computer science / engineering mini-project developed for demonstration, reproducible research, and educational purposes. It is **not** an automotive-grade or ISO 26262 ASIL-certified safety-critical control system. It operates on pre-recorded or simulated kinematic telemetry streams and does **not** include in-cabin camera feeds (eye gaze, head pose, facial landmarks), audio sensing, or direct vehicle OBD-II/CAN-bus physical actuation.

---

## 🏗️ System Architecture

The system processes telemetry through a unidirectional, strictly causal pipeline:

```
                                  [ TELEMETRY STREAM ]
                       (Timestamp, Speed, Accel X/Y/Z, Gyro X/Y/Z)
                                           │
                                           ▼
                       [ TWO-LAYER PREDICTION VALIDATOR ]
                  (Schema, Monotonic Timestamps, Freshness Check)
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
           [ PIPELINE A (STYLE) ]                    [ PIPELINE B (MANEUVERS) ]
        • 47 Causal Kinematic Features            • 56 Causal Streaming Features
        • Pre-fitted StandardScaler               • Pre-fitted StandardScaler
        • Support Vector Machine (RBF)            • Extra Trees Classifier
        • Output: Normal / Aggressive             • Output: SA / SD / SRT / SLT
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                               [ TEMPORAL SMOOTHER ]
                        • Exponential Moving Average (EMA)
                        • D-02 Incident Lifecycle Manager (2.50s Hysteresis)
                        • Multi-Channel Residual Tracking (Longitudinal + Lateral)
                                           │
                                           ▼
                         [ DETERMINISTIC RISK ENGINE ]
                        • D-13 25-Combination Pipeline Health Matrix
                        • P-05A Minimum Caution Floor (Healthy A, Impaired B)
                        • D-13 Option 1A-Pure (Preserve Score on Dropout)
                        • D-06 Multi-Channel Continuous Exponential Decay
                        • Risk Scoring (0 - 100) & Safety State Machine
                                           │
                                           ▼
                          [ SAFETY DECISION & ALERTS ]
             • States: NORMAL (0-24.9), CAUTION (25-49.9), WARNING (50-74.9), HIGH_RISK (75-100)
             • Diagnostic Flags (DATA_QUALITY, SYSTEM_DEGRADED)
             • Explainable Incident History & Active Alerts
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
            [ FASTAPI REST API ]                      [ ACADEMIC DASHBOARD ]
         • /api/telemetry/assess                   • Live Risk Gauge & State Badge
         • /api/simulated/scenarios                • Active Incident Lifecycle Monitor
         • /api/models/status                      • 1-Click Scenario Preset Demonstrator
```

---

## 📁 Repository Directory Structure

```text
driver-monitoring-agent/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py         # REST endpoints (/health, /models/status, /assess, /scenarios)
│   │   ├── schemas/
│   │   │   └── telemetry.py         # Pydantic schemas for request/response validation
│   │   ├── services/
│   │   │   └── agent_service.py     # Singleton service managing models and safety agent
│   │   ├── static/
│   │   │   └── index.html           # Standalone, zero-dependency HTML5/JS academic dashboard
│   │   └── main.py                  # FastAPI application entrypoint with CORS & static mounting
│   └── requirements.txt             # Backend and ML dependencies
├── frontend/                        # Optional React + Vite dashboard scaffold
│   ├── src/
│   │   ├── App.jsx                  # React dashboard interface with Tailwind styling
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── agent/
│   │   ├── config.py                # Safety thresholds, cooldowns, and decay constants
│   │   ├── lifecycle_tracker.py     # D-02 IncidentLifecycleManager & ManeuverContribution
│   │   ├── prediction_validator.py  # Two-layer validation & D-13 health state classification
│   │   ├── risk_engine.py           # DeterministicLateFusionRiskEngine with D-06 residual decay
│   │   ├── safety_agent.py          # IntelligentSafetyAgent facade coordinator
│   │   └── temporal_smoother.py     # Causal EMA smoother and multi-channel incident tracking
│   ├── inference/
│   │   ├── dataset_b_classifier.py  # Pipeline B Extra Trees production inference wrapper
│   │   └── dataset_c_classifier.py  # Pipeline A SVM production inference wrapper
│   └── preprocessing/
│       └── verify_dataset_integrity.py # Dataset checksum and integrity verification
├── models/
│   └── experiments/
│       ├── dataset_b/               # Frozen Pipeline B Extra Trees model & scaler artifacts
│       └── dataset_c/               # Frozen Pipeline A SVM model & scaler artifacts
├── reports/                         # Audits, EDA reports, and model evaluation metrics
├── tests/
│   ├── test_backend_api.py          # FastAPI endpoint and integration test suite (10 tests)
│   ├── test_dataset_b_pipeline.py   # Dataset B event gate and feature tests
│   ├── test_dataset_c_binary_pipeline.py # Dataset C preprocessing & validation tests
│   ├── test_model_training_and_inference.py # Inference and model validation tests
│   └── test_safety_agent.py         # Safety agent, D-13 health matrix, and D-02/D-06 tests
└── README.md                        # Comprehensive project guide and documentation
```

---

## 📊 Dataset Descriptions & Methodologies

The project leverages two distinct, open-access vehicular telemetry datasets:

### 1. Dataset C: Driving Style Telematics
- **Source:** Mendeley Data (DOI: `10.17632/5stn873wft.1`, Shardul Nazirkar). Licensed under CC BY 4.0.
- **Task:** Binary driving style classification (`0: Normal`, `1: Aggressive`).
- **Data Modality:** Smartphone 3-axis accelerometer, gyroscope, and GPS speed data sampled at 1 Hz.
- **Preprocessing & Leakage Prevention:**
  - Duplicate sample removal and trip session boundary tagging.
  - 47 strictly causal rolling kinematic features (means, standard deviations, extrema over backward windows; forward-looking windows strictly barred).
  - Chronological split with 15-second buffer guard bands: Train ($N=9,767$, 70%), Validation ($N=1,935$, 14%), Test ($N=2,138$, 16%).

### 2. Dataset B: Maneuver Detection Telematics
- **Source:** Mendeley Data (DOI: `10.17632/jj3tw8kj6h.3`, Yuksel & Atmaca). Licensed under CC BY 4.0.
- **Task:** 4-class high-risk maneuver classification:
  1. *Sudden Acceleration* (SA)
  2. *Sudden Deceleration* (SD)
  3. *Sudden Right Turn* (SRT)
  4. *Sudden Left Turn* (SLT)
- **Data Modality:** High-frequency IMU sensor data (100 Hz resampled to 10 Hz) across multiple test drivers.
- **Preprocessing & Leakage Prevention:**
  - Kinematic Event Gate isolating 109 discrete maneuver events.
  - 56 Tier 1 strictly causal streaming kinematic features (speed, longitudinal acceleration, lateral acceleration, jerk, and rotational invariants).
  - Stratified Group Split by Driver ID ensuring zero event and driver leakage across partitions (Train: 76 events, Val: 16 events, Test: 17 events; plus 3-fold Leave-One-Driver-Out cross-validation).

---

## 🔬 Machine Learning Models & Measured Evaluation Metrics

All models were evaluated on strictly held-out test sets. The metrics below represent actual measured empirical results:

### Measured Performance Summary

| Pipeline | Model Architecture | Partition | Accuracy | Balanced Acc. | Macro F1 | Critical Class Performance |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Pipeline A**<br>(Driving Style) | **Support Vector Machine (RBF)**<br>$C=2.0$, $\gamma='scale'$, $\tau^*=0.84$ | Validation<br>Test | 70.85%<br>**69.93%** | 71.30%<br>**70.83%** | 0.7070<br>**0.6970** | Aggressive Recall: **84.25%**<br>Normal Precision: **80.67%**<br>False Positive Rate: **42.59%** |
| **Pipeline B**<br>(Maneuver Detection) | **Extra Trees Classifier**<br>200 Trees, Depth 10, Balanced | Val (Event)<br>Val (Sample)<br>Test (Event)<br>Test (Sample)<br>3-Fold LODO | **100.0%**<br>79.75%<br>**88.24%**<br>**80.57%**<br>**93.06%** | **100.0%**<br>80.52%<br>87.50%<br>81.08%<br>— | **1.0000**<br>0.8102<br>**0.8786**<br>**0.8155**<br>**0.9272** | SA F1: **0.7750**<br>SD F1: **0.8750**<br>SRT F1: **0.8037**<br>SLT F1: **0.7463** |

### 🔬 Pipeline A Generalization Benchmark & Scientific Findings (Dataset C)

An exhaustive research benchmark across 8 model families (RBF SVM, ExtraTrees, Random Forest, XGBoost, HistGradientBoosting, compact 1D CNN, voting ensemble, and Logistic Regression baseline) evaluated the research target of $\ge 90\%$ performance across independent validation data:

| Model Architecture | Features | S1 $\rightarrow$ S2 Acc | S1 $\rightarrow$ S2 F1 | S1 $\rightarrow$ S2 FPR | S2 $\rightarrow$ S1 Acc | S2 $\rightarrow$ S1 F1 | S2 $\rightarrow$ S1 FPR | Mean Cross-Session Acc | Mean Macro-F1 | Mean FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 47 | 82.52% | 82.01% | 1.03% | 78.79% | 77.93% | 38.79% | 80.66% | 79.97% | 19.91% |
| **Refined SVM (RBF)** | 47 | 87.53% | 86.88% | 1.82% | 76.69% | 75.24% | 45.50% | 82.11% | 81.06% | 23.66% |
| **ExtraTrees (Candidate 1)** | **47** | **91.43%** | **90.79%** | **1.82%** | **74.71%** | **72.80%** | **49.99%** | **83.07%** | **81.79%** | **25.91%** |
| **ExtraTrees (Advanced)** | **85** | **91.65%** | **91.02%** | **1.78%** | **74.59%** | **72.56%** | **50.83%** | **83.12%** | **81.79%** | **26.30%** |
| **Random Forest** | 47 | 88.42% | 87.78% | 1.22% | 74.29% | 72.23% | 51.14% | 81.36% | 80.01% | 26.18% |
| **XGBoost (Tuned)** | 47 | 84.80% | 84.22% | 1.12% | 75.44% | 73.69% | 48.50% | 80.12% | 78.95% | 24.81% |
| **HistGradientBoosting** | 47 | 86.48% | 85.85% | 1.36% | 74.77% | 72.88% | 49.73% | 80.62% | 79.37% | 25.55% |
| **Compact 1D CNN** | Raw IMU | 85.19% | 84.41% | 5.24% | 71.86% | 69.10% | 56.41% | 78.52% | 76.76% | 30.83% |

#### Critical Forensic Findings & Mathematical Bottlenecks:
1. **Directional Asymmetry:** While ExtraTrees reaches **91.43% accuracy** and passes 7 of 10 targets in the Session 1 $\rightarrow$ Session 2 direction, the reverse transfer drops to **74.71%** with a surging 49.99% false alarm rate. The cross-session mean ceiling across both directions is **83.07% accuracy** (and pooled accuracy is **82.49%**).
2. **Physical Mounting Drift ($31.36^\circ$ Offset):** Vector analysis of static accelerometer values demonstrates a $31.36^\circ$ physical orientation tilt between Session 1 ($\vec{a}_1 = [-0.048, 0.177, -0.031]$) and Session 2 ($\vec{a}_2 = [0.055, 0.183, -0.028]$), injecting systematic covariate shift into raw-axis sensors.
3. **Trip Class Prior Mismatch:** Session 1 contains 48.2% aggressive driving, while Session 2 contains 66.8% aggressive driving. Training on Session 2 induces an aggressive prior that over-triggers false alarms on Session 1.
4. **Single-Driver, 17-Episode Constraint:** Dataset C contains telemetry from exactly **one driver** across only **17 continuous driving episodes** ($N_{\text{eff}} \approx 707$ independent samples). It is mathematically impossible to evaluate "unseen-driver generalization" on Dataset C alone.

#### Evaluation Methodology Distinctions:
- **Sample-Level Classification:** Evaluates each individual 100 ms time step; susceptible to high temporal autocorrelation.
- **Event-Level Detection:** Evaluates contiguous maneuver episodes bounded by onset and completion timestamps via temporal IoU bipartite matching.
- **Session-Independent Evaluation:** Validates generalization across separate recording trips of the *same driver* separated by non-recording intervals (1h 17m).
- **Driver-Independent Evaluation:** Strictly validates generalization on *completely unseen drivers* whose data was excluded from training, feature selection, and scaler fitting (e.g. Leave-One-Driver-Out).

#### Approved 8-Driver Data Collection Protocol:
To achieve genuine $\ge 90\%$ driver-independent performance, the project established an operational 8-driver protocol:
- **Cohort:** 8 licensed drivers (6 development drivers in 6-fold LODO cross-validation + 2 reserved blind lockbox drivers).
- **Sensor Rigidity:** Windshield mechanical screw-clamp mount, mandatory 15-second static calibration prior to engine start.
- **Safety Protocol:** 100% naturalistic driving on public roads (zero aggressive maneuvers on public highways); controlled aggressive maneuvers (hard braking, slalom, aggressive acceleration) restricted strictly to a private, closed proving ground supervised by a driving instructor.
- **Model Deployment Decision:** Existing production baseline (RBF SVM) remains active and cryptographically verified. Candidate models remain preserved in the research vault until multi-driver physical validation is completed.

### Cryptographic SHA-256 Hashes of Frozen Production Artifacts
To guarantee model integrity and reproducible research, production artifacts are cryptographically frozen:

| Pipeline | Artifact File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Pipeline B Model** | `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib` | `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` |
| **Pipeline B Scaler** | `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib` | `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` |
| **Pipeline A Model** | `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib` | `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` |
| **Pipeline A Scaler** | `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib` | `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` |

---

## 🛡️ Safety Agent Architecture & Approved Policies

The Stage 6 Safety Agent fuses predictions from both pipelines into an explainable, deterministic risk score ($0.0 \dots 100.0$) mapped to four safety states:
- **`NORMAL` (0.00 – 24.99):** Baseline safe driving conditions.
- **`CAUTION` (25.00 – 49.99):** Elevated awareness (e.g., degraded sensor health, mild style aggression).
- **`WARNING` (50.00 – 74.99):** Immediate risk detected (e.g., severe sudden deceleration or repeated sharp cornering).
- **`HIGH_RISK` (75.00 – 100.00):** Severe compound hazard (e.g., aggressive driver undergoing sudden deceleration or swerving).

### 1. D-13 25-Combination Pipeline Health Matrix
The system evaluates pipeline availability across 5 discrete health states: `HEALTHY`, `DEGRADED`, `MISSING`, `INVALID`, and `QUIESCENT_BYPASS` ($5 \times 5 = 25$ total combinations):
- **Option 1A-Pure (Pipeline A Impairment with Active Pipeline B):**  
  When Pipeline A is `DEGRADED`, `MISSING`, or `INVALID` and Pipeline B detects an active maneuver, the agent preserves the ordinary confidence-scaled calculated behavioral score and ordinary score-to-state mapping without imposing an artificial CAUTION floor, while appending diagnostic flags (`DATA_QUALITY`, `SYSTEM_DEGRADED`).
- **Option 2A (Pipeline A Impairment during Quiescence):**  
  When Pipeline A is impaired and Pipeline B reports `QUIESCENT_BYPASS` (nominal cruising without maneuver), the system safely returns `INSUFFICIENT_DATA` with `DATA_QUALITY` warnings, avoiding false safety assertions.
- **P-05A Minimum CAUTION Floor (Pipeline B Impairment during Active Driving):**  
  When Pipeline A is `HEALTHY` and Pipeline B is `DEGRADED`, `MISSING`, or `INVALID` during active driving, the system enforces a minimum state of `CAUTION` (clamping safe scores upwards) to signal reduced monitoring coverage to the driver.

### 2. D-02 Multi-Instance Incident Lifecycle Tracking
Implemented via `IncidentLifecycleManager`:
- **2.50s Quiet Hysteresis Window:** An incident remains active as long as excitation persists. Once excitation ceases, an uninterrupted quiet period of $2.50\text{ s}$ must elapse before the channel re-arms for a new incident.
- **Intra-Incident Continuation ($t_0$ Preservation):** Re-excitation within the $2.50\text{ s}$ window is classified as an intra-incident continuation, preserving the original onset timestamp $t_0$ and preventing artificial inflation of the recent incident counter.
- **Separate Incident Allocation:** Any excitation occurring after $2.50\text{ s}$ of quiet receives a newly incremented `event_id` and increments the maneuver occurrence count.

### 3. D-06 Continuous Residual Risk Decay Integration
Implemented via `ManeuverContribution`:
- **Orthogonal Summation:** Risk contributions from orthogonal axes (Longitudinal: Acceleration/Deceleration vs. Lateral: Left/Right Turn) are tracked independently and summed orthogonally:
  $$\text{Residual} = R_{\text{longitudinal}}(t) + R_{\text{lateral}}(t)$$
- **Continuous Decay Mechanics:** When a maneuver ceases excitation at $t_{\text{end}}$, its risk contribution remains at peak weight ($W$) for a $1.00\text{ s}$ plateau, after which it decays exponentially with a half-life of $0.75\text{ s}$:
  $$R(t) = W \times 2^{-\frac{(t - t_{\text{end}} - 1.00)}{0.75}}$$
- **Analytical Retirement Horizon:** A contribution is retired completely when its residual risk drops below $2.50$ points:
  $$t_{\text{retire}} = t_{\text{end}} + 1.00 + 0.75 \times \log_2\left(\frac{W}{2.50}\right)$$
  *(For Sudden Deceleration $W=50.0$, $t_{\text{retire}} \approx t_{\text{end}} + 4.24\text{ s}$; for Turns $W=35.0$, $t_{\text{retire}} \approx t_{\text{end}} + 3.86\text{ s}$)*.

---

## 🚀 Installation & Local Environment Setup

The project runs on standard Python 3.10+ and requires no GPU or external database setup.

### 1. Clone the Repository
```bash
git clone https://github.com/yeswanth225/driver-machine-learning-model.git
cd driver-machine-learning-model
```

### 2. Set Up Python Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

## 💻 Running the Application & Academic Dashboard

### Method 1: All-in-One Dashboard (Recommended)
The backend comes bundled with a responsive, standalone academic dashboard mounted directly at the root URL.

1. Start the FastAPI backend:
   ```bash
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```
   *(Or access the Swagger API docs at `http://127.0.0.1:8000/docs`)*.

### Method 2: Running the React Frontend (Optional)
If you wish to run the standalone React/Vite development server:
```bash
cd frontend
npm install
npm run dev
```
The Vite dashboard will launch at `http://localhost:5173/` and communicate with the FastAPI backend at port 8000.

---

## 🧪 Comprehensive Testing & Verification

The project includes an extensive test suite verifying mathematical correctness, D-13 health matrices, D-02/D-06 lifecycles, streaming maneuver detection, and API endpoints.

To run all 175 automated tests:
```bash
python -m pytest tests -v
```

### Test Suite Breakdown (175 Passed / 175 Total)
- `tests/test_backend_api.py` (15 tests): Validates health checks, frozen model SHA-256 hashes, REST telemetry assessment, scenario retrieval, agent reset, hardware diagnostics, and schema validation.
- `tests/test_safety_agent.py` (85 tests): Validates the 25-combination health matrix, D-13 Option 1A-Pure, Option 2A, P-05A, D-02 2.50s hysteresis, and LT-01 through LT-10 lifecycle tests.
- `tests/test_streaming_testbed.py` (21 tests): Validates boundary-independent streaming maneuver detection, onset confirmation, quiet hysteresis, end-of-stream flushing, repeated flush calls, and temporal IoU bipartite matching.
- `tests/test_model_training_and_inference.py` (18 tests): Validates inference pipelines, feature quarantine compliance, and frozen artifact integrity.
- `tests/test_dataset_b_pipeline.py` (22 tests): Validates event gate, kinematics, unit conversions, and zero-leakage driver splitting.
- `tests/test_dataset_c_binary_pipeline.py` (14 tests): Validates Dataset C preprocessing, session tagging, and causal feature engineering.

---

## 🎯 B.Tech Mini-Project College Demonstration Guide

During your project demonstration or viva, use the **1-Click Preset Demonstration Scenarios** built directly into the dashboard.

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEMONSTRATION WALKTHROUGH FOR EXAMINERS                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Scenario 1: Nominal Highway Cruising
- **Action:** Click `Nominal Highway Cruising`.
- **Telemetry:** Speed = 65 km/h, Normal driving style ($c=0.92$), no active maneuver.
- **Expected Output:**
  - Risk Score: **0.00 / 100**
  - Safety State: **NORMAL (Green)**
  - Explanation: "Normal driving style without hazardous maneuvers."

### Scenario 2: Sudden Harsh Braking (Warning + Residual Decay)
- **Action:** Click `Sudden Harsh Braking`.
- **Telemetry:** Longitudinal deceleration ($a_x = -6.2 \text{ m/s}^2$), Sudden Deceleration maneuver ($c=0.95$).
- **Expected Output:**
  - Risk Score: **60.00 / 100**
  - Safety State: **WARNING (Amber)**
  - Active Alert: `DECELERATION_WARNING`
  - Lifecycle: Active incident on `DECEL` channel. When quiet begins, observe risk residual smoothly decay over a 4.24s horizon.

### Scenario 3: Compound Critical Hazard
- **Action:** Click `Compound Hazard (Swerving + Aggressive)`.
- **Telemetry:** Aggressive driving style ($c=0.91$) combined with Sudden Left Turn ($c=0.94$).
- **Expected Output:**
  - Risk Score: **84.50 / 100**
  - Safety State: **HIGH_RISK (Red)**
  - Active Alert: `COMPOUND_HAZARD` (severity: CRITICAL, +25 pt non-linear compound penalty).

### Scenario 4: Sensor Dropout (D-13 Option 1A-Pure)
- **Action:** Click `Pipeline A Sensor Dropout (Option 1A-Pure)`.
- **Telemetry:** Pipeline A disconnected (`DEGRADED` / `MISSING`), Pipeline B detects Sudden Right Turn ($c=0.90$).
- **Expected Output:**
  - Safety State: **CAUTION (Yellow)** (Calculated behavioral score $\approx 31.50$ preserved).
  - Diagnostic Flags: `DATA_QUALITY`, `SYSTEM_DEGRADED` (Signals camera/style sensor failure without masking the maneuver).

### Scenario 5: Pipeline B Sensor Failure (P-05A Fail-Safe)
- **Action:** Click `Pipeline B Sensor Failure (P-05A Failsafe)`.
- **Telemetry:** Pipeline A `HEALTHY` (Normal style), Pipeline B disconnected (`MISSING`).
- **Expected Output:**
  - Safety State: **CAUTION (Yellow)**
  - Explanation: "P-05A Safety Enforcement: Reduced sensor coverage mandates minimum CAUTION floor during active driving."

### Scenario 6: Dual-Pipeline Communication Outage (Option 2A)
- **Action:** Click `Dual-Pipeline Outage (Option 2A)`.
- **Telemetry:** Both pipelines disconnected (`MISSING`).
- **Expected Output:**
  - Safety State: **INSUFFICIENT_DATA (Grey)**
  - Explanation: "Option 2A Failsafe: Telemetry streams insufficient for safety assessment."

---

## 📱 Physical Sensor Integration & Real-World Telemetry Acquisition

The system features an end-to-end **Physical Sensor Acquisition, Ingestion, and Quality Validation Pipeline** designed to bridge real-world smartphone kinematics (Android/iOS) with the core safety intelligence engine.

```
 [ SMARTPHONE 20 Hz ] ──> [ INGESTION API ] ──> [ QUALITY ENGINE ] ──> [ 2:1 CAUSAL DECIMATOR ] ──> [ ML & SAFETY AGENT ] ──> [ LIVE DASHBOARD ]
 (Android APK / PWA)      (/api/sensor/*)       (6 Quality Gates)      (10 Hz Model Parity)         (D-13 Risk Engine)         (/live_monitoring)
```

### 1. Mobile Acquisition Clients
- **Native Android Application (`mobile/android/`):**
  - Native Kotlin Android Studio project targeting Android 14 (API 34).
  - Background logging via `ForegroundService` with `PARTIAL_WAKE_LOCK`.
  - Subscribes to `Sensor.TYPE_ACCELEROMETER` and `Sensor.TYPE_GYROSCOPE` at `SENSOR_DELAY_FASTEST` ($\approx 20\text{ Hz}$).
  - Hardware monotonic clock synchronization (`SystemClock.elapsedRealtimeNanos()`).
  - Atomic buffered CSV writer, local SQLite/file persistence, client-side SHA-256 calculation, and multipart HTTP upload (`DataUploader.kt`).
- **Cross-Platform Web Logger PWA (`mobile/web-logger/`):**
  - Zero-installation mobile logger accessible directly over HTTPS at `http://<HOST_IP>:8000/sensor-logger/`.
  - Subscribes to HTML5 `devicemotion` and `navigator.geolocation` APIs.
  - Internal 50 ms tick loop providing stable 20.0 Hz resampling with sub-millisecond precision.
  - Real-time mobile HUD displaying rolling sampling rate, elapsed duration, GPS speed, and driving regime selector (`S0` to `S6`).
  - Web Crypto API SHA-256 generation and direct upload to `/api/sensor/upload`.

### 2. Automated Quality Control & Governance Engine
Every incoming physical recording is automatically audited by `DataQualityService` across **6 mandatory acceptance criteria**:
1. **Sampling Frequency & Jitter:** Nominal 20 Hz ($18.0 - 22.0\text{ Hz}$), zero gaps $\ge 250\text{ ms}$, strictly monotonic.
2. **Dynamic Range & Saturation:** Full-scale $\ge \pm 4g$, zero clipped samples ($< 3.92g$), zero dead channels.
3. **Static Gravitational Calibration & Mount Stability:** Pre-drive stationary baseline $|g_0| \in [0.98, 1.02]g$, post-drive mount drift $\le 3.0^\circ$.
4. **GPS Ground Speed Consistency:** Zero negative speeds, mean $\text{HDOP} \le 2.5$.
5. **Driving Regime Representation:** Minimum durations across city (`S1`), highway cruising (`S3`), rough-road (`S4`), and aggressive maneuvers (`S5`).
6. **Data Hygiene:** Strictly zero NaNs, nulls, or infinite values.

**Governance & Lockbox Isolation:**
- Development Driver Pool: `DRV_001` through `DRV_006` are stored in `dataset/raw/eight_driver_campaign/`.
- Blind Lockbox Holdout: `DRV_007` and `DRV_008` are sequestered; any developmental access attempt is strictly intercepted and returns **HTTP 403 Forbidden**.
- Duplicate Ingestion Protection: Identical uploads are detected via SHA-256 and rejected as `DUPLICATE_REJECTED`.

### 3. Sampling Compatibility Adapter (10 Hz vs. 20 Hz)
To eliminate temporal feature distortion when feeding 20 Hz physical telemetry into models trained at 10 Hz:
- **Path A (Causal 2:1 Anti-Aliasing Decimator):**
  $$y[k] = \frac{1}{2} x[2k] + \frac{1}{2} x[2k-1]$$
  Executes causal downsampling with **zero future lookahead**, ensuring exact physical feature alignment with frozen production models.
- **Path B (Native 20 Hz Feature Extractor):**
  Matches physical time integration durations ($[20, 40, 60, 100]$ samples for $1\text{s}, 2\text{s}, 3\text{s}, 5\text{s}$), scales jerk by $\Delta t = 0.05\text{ s}$, and employs continuous-time matched EMA smoothing coefficients ($\alpha_{\text{fast}} = 0.1056, \alpha_{\text{slow}} = 0.0408$).
- **Buffer Warm-Up Enforcement:** Requires at least 50 downsampled observations ($5.0\text{ s}$) before producing valid ML style inferences (`PhysicalSensorInferenceAdapter`).

### 4. Live Monitoring & Observer Annotation
- **Frontend Dual-Mode Selector:** Switch seamlessly between **"Physical Smartphone Sensor (Live 20 Hz)"** and **"Historical Dataset Analysis / Demonstration Stream"**.
- **Honest Hardware Verification:** If no physical device is transmitting, the dashboard reports `Physical Hardware: NOT_VERIFIED` with instructions for launching the mobile logger.
- **Observer Ground-Truth Tagging:** In-car observers can tag real-time maneuvers (`NORMAL`, `AGGRESSIVE`, `SUDDEN_ACCELERATION`, `SUDDEN_DECELERATION`, `SUDDEN_RIGHT_TURN`, `SUDDEN_LEFT_TURN`, `UNCERTAIN`) stored immutably in `dataset/annotations/ground_truth_manifest.json`.

---

## 📡 Backend API Reference

### Base URL: `http://127.0.0.1:8000/api`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health, uptime, and active model status. |
| `GET` | `/models/status` | Real-time verification of frozen model artifact SHA-256 hashes. |
| `POST` | `/telemetry/assess` | Primary assessment endpoint (accepts structured or raw telemetry). |
| `POST` | `/telemetry/assess/raw` | Direct evaluation of raw sensor signals (accel, gyro, speed). |
| `GET` | `/simulated/scenarios` | Returns the 6 preset demonstration scenarios for viva presentations. |
| `POST` | `/agent/reset` | Resets the safety agent, EMA smoother, and incident tracker state. |
| `POST` | `/sensor/device/register` | Registers an Android/iOS smartphone device for physical acquisition. |
| `POST` | `/sensor/session/start` | Initializes a new physical telemetry recording session. |
| `POST` | `/sensor/telemetry/frame` | Ingests a single real-time 20 Hz frame from the mobile logger. |
| `POST` | `/sensor/upload` | Multipart CSV upload, SHA-256 verification, and automated quality audit. |
| `GET` | `/sensor/sessions` | Lists all ingested physical recording sessions and manifests. |
| `GET` | `/sensor/sessions/{d}/{s}/quality` | Retrieves the 6-criterion Data Quality Report for a session. |
| `GET` | `/sensor/live/status` | Real-time device connection state, measured frequency, and latest frame. |
| `GET` | `/sensor/live/frames` | Retrieves recent rolling frames for live dashboard charts. |
| `POST` | `/sensor/annotations` | Records an independent human observer ground-truth annotation. |
| `GET` | `/sensor/annotations/{session_id}` | Retrieves all ground-truth annotations for a session. |
| `WS` | `/sensor/live/stream` | Full-duplex WebSocket channel streaming 20 Hz telemetry frames. |
| `GET` | `/system/hardware-status` | Dynamic hardware status reflecting real-time smartphone connectivity. |

---

## 🚀 Quick Start & Execution Guide

### 1. Launching the Backend Server
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger API docs: `http://127.0.0.1:8000/docs`
- Mobile Web Sensor Logger: `http://127.0.0.1:8000/sensor-logger/`

### 2. Launching the Frontend Live Monitoring Dashboard
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173/` in your browser. Navigate to **Live Monitoring** to view the dual-mode 20 Hz cockpit.

### 3. Running the Complete Verification Test Suite
```powershell
pytest tests/ -v
```
Executes all **208 tests** spanning backend APIs, data pipelines, model artifacts, safety agent rules, causal resampling adapters, and physical sensor ingestion.

---

## ⚠️ Academic Limitations & Future Enhancements

### Known Academic Limitations
1. **Smartphone IMU Proxy Data:** The models were trained and validated on public smartphone IMU datasets (Dataset B and Dataset C). Real vehicular dynamics exhibit distinct chassis vibrations and suspension filtering.
2. **Deterministic Late Fusion:** The safety agent utilizes rule-based late fusion rather than end-to-end deep multimodal fusion. While this provides complete safety explainability and auditability, cross-modal latent correlations are not modeled.
3. **Absence of Cabin Vision:** The system does not monitor driver gaze, blink duration (PERCLOS), or smartphone distraction directly; it infers risk purely through vehicle kinematic response.

### Future Research Directions
- **Physical Embedded Edge Deployment:** Port the inference engine to an NVIDIA Jetson Orin Nano or Raspberry Pi 5 with an MPU-6050 accelerometer/gyroscope HAT.
- **OBD-II / CAN-Bus Integration:** Replace smartphone-derived speed with high-precision wheel speed sensors via an ELM327 OBD-II adapter.
- **Multimodal Vision Integration:** Incorporate an in-cabin camera feed using MediaPipe Face Mesh for eye-aspect-ratio (EAR) drowsiness detection.

---

## 👨‍💻 Authors & Academic Attribution

- **B.Tech Capstone Mini Project:** Intelligent Driver Activity Monitoring & Safety Agent
- **Original Dataset Authors:**
  - Dataset B: *Yuksel & Atmaca*, Mendeley Data, DOI: `10.17632/jj3tw8kj6h.3` (CC BY 4.0)
  - Dataset C: *Shardul Nazirkar*, Mendeley Data, DOI: `10.17632/5stn873wft.1` (CC BY 4.0)
- **Frameworks & Libraries:** Python, Scikit-Learn, FastAPI, Pydantic, Pytest, Tailwind CSS, Vite.

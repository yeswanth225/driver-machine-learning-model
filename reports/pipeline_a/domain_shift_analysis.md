# DOMAIN SHIFT ANALYSIS & DRIFT MONITORING FRAMEWORK
## Quantitative Detection & Evaluation of Physical Telematics Shifts in Production

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A Robustness & Safety Engineering  
**Author:** Senior Machine Learning Systems Architect & Telematics Reliability Lead  
**Date:** 2026-11-22  
**Scope:** Telematics Domain Shift Identification, Statistical Detection, and Operational Guardrails  
**Policy Status:** Architectural Design & Offline Monitoring Specification (Zero Unverified Production Code)  

---

## 1. Executive Summary

A core lesson from the forensic audit of Dataset C is that physical telematics datasets suffer from significant **domain shift**. Even within the same driver and vehicle, a 75-minute interval between Session 1 and Session 2 introduced a statistically significant DC lateral shift ($p < 10^{-8}$) and heading rotation shift ($p < 10^{-22}$). Across different vehicles, smartphone mounts, and human drivers, these domain shifts multiply in magnitude.

If an ML model trained in Vehicle A is deployed into Vehicle B without domain-shift awareness, physical differences in mount stiffness, engine idle harmonics, or road pavement will be misinterpreted as aggressive driving, triggering false safety alarms.

This document establishes:
1. A taxonomy of six distinct physical domain shifts in automotive telematics.
2. Rigorous quantitative metrics to detect and monitor domain drift.
3. An online drift detection architecture.
4. Strict safety policies prohibiting unverified heuristic corrections from entering the production runtime.

---

## 2. Taxonomy of Telematics Domain Shifts

```
                                [TELEMATICS DOMAIN SHIFT]
                                           │
         ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
         ▼                   ▼                           ▼                   ▼
[1. Mounting & Angle]  [2. Vehicle Chassis]        [3. Road Environment]  [4. Driver Habit]
- Cradle tilt          - Suspension stiffness     - Asphalt vs cobbles   - Throttle aggressiveness
- Vibration resonance  - Engine idle harmonics    - Speed humps, seams   - Baseline speed bias
```

### 2.1 Sensor Orientation & Cradle Tilt Shifts
- **Physical Mechanism:** Smartphones are mounted at arbitrary angles depending on windshield rake and cradle design. A phone tilted backward at $25^\circ$ projects a component of Earth's gravity ($g \cdot \sin 25^\circ \approx 4.14\text{ m/s}^2$) onto its longitudinal axis ($a_y$).
- **Impact on Inference:** If linear acceleration hardware fusion fails to fully compensate for tilt, a static DC offset shifts the model's operating point, causing constant false positive acceleration alarms.

### 2.2 Device Mounting Rigidity & Resonance
- **Physical Mechanism:** Different phone mounts exhibit vastly different mechanical transfer functions:
  - *Rigid Windshield Suction Mount:* High-frequency transmission ($>20\text{ Hz}$), low displacement amplitude.
  - *Flexible Gooseneck / Vent Clip:* Low resonant frequency ($4\text{--}8\text{ Hz}$), large harmonic swaying during cornering.
- **Impact on Inference:** Oscillating mount bounce directly corrupts `gyro_x` and `gyro_y` (roll/pitch), artificially inflating `gyro_mag` in naive unseparated feature sets.

### 2.3 Environmental & Road Surface Shifts
- **Physical Mechanism:** Road surfaces vary from smooth asphalt to grooved concrete, cobblestones, and potholed urban corridors.
- **Impact on Inference:** Uneven pavement induces continuous vertical acceleration ($a_z$) chatter ($\sigma_z > 2.0\text{ m/s}^2$), which naive models confuse with high-speed aggressive driving.

### 2.4 Driver Behavioral Baseline (Habitual Style Shifts)
- **Physical Mechanism:** Human driving baselines vary widely:
  - A conservative driver in heavy traffic averages lateral acceleration $< 0.8\text{ m/s}^2$.
  - An assertive courier averages lateral acceleration $> 1.8\text{ m/s}^2$ while remaining completely in control.
- **Impact on Inference:** A fixed universal threshold flags assertive drivers perpetually, while failing to detect aggressive degradation in conservative drivers.

### 2.5 Sensor Hardware Calibration & Thermal Drift
- **Physical Mechanism:** MEMS accelerometers and gyroscopes in consumer smartphones exhibit temperature-dependent zero-bias drift:
  $$\Delta b_{\text{gyro}}(T) \approx 0.05^\circ/\text{s} \text{ per } 10^\circ\text{C rise in device temperature}.$$
- **Impact on Inference:** During long trips with GPS and charging active, phone temperature can rise from $25^\circ\text{C}$ to $45^\circ\text{C}$, introducing a progressive baseline drift.

### 2.6 Class-Prior Shifts (Contextual Exposure)
- **Physical Mechanism:** In stop-and-go morning rush hour, aggressive opportunities are near zero ($P(\text{Aggressive}) < 0.05$). On late-night open arterial roads, aggressive overtaking opportunities increase ($P(\text{Aggressive}) \approx 0.40$).
- **Impact on Inference:** Static Bayesian priors produce high false alarm rates during low-prior traffic conditions.

---

## 3. Quantitative Shift Detection Metrics

To monitor telemetry streams without relying on ground-truth labels, the system evaluates four statistical distance metrics on sliding windows ($W = 300\text{ seconds}$):

### 3.1 Population Stability Index (PSI) as an Observability Metric
The Population Stability Index measures the empirical shift in feature distributions between a reference baseline $\mathcal{P}$ (training distribution) and the live operational distribution $\mathcal{Q}$:

$$\text{PSI} = \sum_{b=1}^{B} \left( Q_b - P_b \right) \cdot \ln\left( \frac{Q_b}{P_b} \right)$$

> [!IMPORTANT]
> **PSI is an Observability Indicator, NOT a Safety Actuation Threshold:**  
> In automotive telematics, PSI thresholds (such as 0.10 or 0.25) are statistical rules-of-thumb from financial credit scoring; they have **NOT** been independently safety-certified for real-time automotive control. PSI serves exclusively as an offline diagnostic indicator and telemetry quality monitor. It must NEVER be used to directly override risk scoring or force driving state transitions.

---

## 4. Online Drift Monitoring & Approved Data-Quality Architecture

### 4.1 Safety Flaw Analysis of Heuristic Risk Clamping
A previously suggested heuristic proposed: *"If $\text{PSI} \ge 0.25$, force style risk $= 0.0$."*  
**Critical Safety Review & Rejection:**
1. **Masking Genuine Aggressive Driving:** If a driver operates aggressively on a bumpy road or in an unmodeled vehicle, high vibration will elevate PSI. Forcing style risk to 0.0 would artificially suppress genuine aggressive driving evidence, generating an unjustified "Safe / Normal Driving" assessment.
2. **False Equivalence:** Unreliable Pipeline A input must **never** be treated as evidence of normal driving. Absence of reliable style evidence is not proof of safe driving.
3. **Preservation of Pipeline B Evidence:** Dynamic maneuver detection (Pipeline B) must remain fully authoritative even when continuous driving style (Pipeline A) is degraded.

### 4.2 Approved Data-Quality Handling Architecture
Rather than inventing ad-hoc risk floors or modifying the risk engine, the system follows the **already approved Stage 6 Safety Agent architecture** (`ml/agent/prediction_validator.py` and `risk_engine.py`):

```
[Incoming Telemetry Stream]
             │
             ├──► [Drift & Quality Monitor (Passive)]
             │           │
             │           ▼ (If severe hardware corruption / sensor failure detected)
             │    Set pipeline_a_health = "DEGRADED" (or "INVALID")
             │
             ├──► [Pipeline A Inference Engine]
             │           │ (If degraded, style_prediction = None, confidence = 0.0)
             │           ▼
             ├──► [RawPredictionInput Payload]
             │      ├── style_prediction: Optional[int] (None when degraded; NEVER defaulted to 0)
             │      ├── style_confidence: 0.0
             │      ├── pipeline_a_health: "DEGRADED"
             │      ├── maneuver_prediction: int (From Pipeline B)
             │      └── pipeline_b_health: "HEALTHY" / "QUIESCENT_BYPASS"
             │
             ▼
[PredictionValidator (ml/agent/prediction_validator.py)]
  ├── Layer 1: Asserts timestamp validity and bounds.
  ├── Layer 2: Validates pipeline health signals independently:
  │     ├── Pipeline A = "DEGRADED" -> flags data_quality_status = "DEGRADED".
  │     └── Pipeline B = "HEALTHY" -> has_usable_maneuver = True (is_valid = True).
  │     └── EXPLICIT POLICY: Never defaults missing style to Normal Driving.
  ▼
[DeterministicRiskEngine (ml/agent/risk_engine.py)]
  ├── Style Rules (Rule 2 SUSTAINED_AGGRESSIVE_STYLE): Requires valid style_pred == 1.
  │     └── When style_pred is None, Rule 2 does NOT fire (style risk = 0.0).
  ├── Maneuver Rules (Rule 1, 3, 4, 5, 6): Fully active and driven by Pipeline B.
  │     └── Any Sudden Acceleration, Deceleration, or Turn escalates risk normally.
  └── Degraded State Reporting: Active alert emitted notifying telemetry degradation
        without compromising driver intervention on active maneuvers.
```

---

## 5. Strict Safety Policy: Prohibition of Unverified Corrections

> [!WARNING]
> **MANDATORY POLICY ON UNVERIFIED RUNTIME HEURISTICS**  
> Under no circumstances shall ad-hoc mathematical offsets (such as subtracting arbitrary DC constants, scaling raw inputs by uncalibrated multipliers, or hardcoding per-vehicle thresholds) be injected into the production inference codebase without formal validation.
> 
> Unverified heuristic corrections risk:
> 1. Inadvertently masking genuine high-g crash or emergency avoidance maneuvers.
> 2. Introducing race conditions or non-deterministic state drift into the Safety Agent.
> 3. Invalidating the audited frozen artifact hashes.

### Approved Remediation Mechanism
If domain shift or sensor corruption is detected:
- The system flags `pipeline_a_health = "DEGRADED"` in `RawPredictionInput`.
- The `PredictionValidator` produces a `ValidatedPrediction` with `data_quality_status = "DEGRADED"`, withholding untrusted style predictions while leaving Pipeline B maneuver detection fully armed and operational.
- Zero changes to production code or frozen model files are executed.

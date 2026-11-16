# Stage 6 Rule Engine Specification: Deterministic Safety Rules, Scoring & Alerting

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** `ml/agent/risk_engine.py`  
**Stage:** Stage 6 — Intelligent Safety Monitoring Agent  
**Status:** Completed & Validated  
**Version:** 1.0.0  

---

## 1. Design Philosophy & Safety Objectives

The Safety Rule Engine provides deterministic, audit-compliant reasoning over driver behavior. Unlike end-to-end black-box deep neural networks whose internal failure modes cannot be mathematically bounded or explained to human operators, this rule engine combines the statistical pattern-recognition strengths of trained ML classifiers with the absolute safety guarantees of formal decision logic.

### Core Guarantees:
1. **Mathematical Determinism:** Given identical input states and temporal contexts, the engine produces identical risk scores, safety classifications, and explanations ($f(x_t, h_t) = f(x_t, h_t)$ strictly holds).
2. **Explainability by Design:** Every penalty, status transition, and emitted alert is directly traceable to one or more human-interpretable causal factors (e.g., "Harsh deceleration detected during sustained aggressive driving").
3. **No Phantom Risk / False Alarms:** Transient 1-sample spikes are filtered by temporal smoothing before rule execution; low-confidence classifications are scaled down smoothly.
4. **Zero-Latency Escalation:** While redundant alerts are throttled to prevent cognitive fatigue, escalating threats immediately pierce cooldown barriers.

---

## 2. Risk Score Formulation & State Partitioning

### 2.1 Continuous Risk Score Function
The composite risk score $R(t) \in [0.0, 100.0]$ is computed as the sum of a baseline score $R_0$, style risk $R_{\text{style}}$, maneuver risk $R_{\text{maneuver}}$, and compound hazard interaction risk $R_{\text{compound}}$, subject to confidence discounting $\phi(C)$ and strict clamping:

$$R(t) = \operatorname{clip}\left( R_0 + \phi(C_S) \cdot R_{\text{style}}(t) + \phi(C_M) \cdot R_{\text{maneuver}}(t) + R_{\text{compound}}(t), \ 0.0, \ 100.0 \right)$$

Where:
- $R_0 = 10.0$ (Baseline driving nominal operational risk).
- $C_S$ is the smoothed driving style confidence.
- $C_M$ is the active maneuver confidence.
- $\phi(C)$ is the confidence scaling operator defined in Rule 8.

### 2.2 Safety State Partitioning
The continuous risk score maps directly into five mutually exclusive safety states:

| Safety State | Score Interval | Operational Definition & System Posture |
| :--- | :---: | :--- |
| **`INSUFFICIENT_DATA`** | $\text{N/A}$ | Sensor outage, packet dropout, stale telemetry ($>3.0\,\text{s}$), or uninitialized buffers. System cannot certify driving state. |
| **`NORMAL`** | $[0.0, 30.0)$ | Nominal driving. Smooth longitudinal and lateral vehicle dynamics. No alerts emitted. |
| **`CAUTION`** | $[30.0, 55.0)$ | Elevated risk. Sustained aggressive driving style or isolated abrupt acceleration. Advisory alert emitted. |
| **`WARNING`** | $[55.0, 75.0)$ | Severe operational deviation. Sudden braking or sharp cornering detected. Warning alert emitted. |
| **`HIGH_RISK`** | $[75.0, 100.0]$ | Critical safety hazard. Confluence of aggressive driving and abrupt maneuvers. Immediate critical alert emitted. |

---

## 3. Formal Rule Specifications (Rules 1 to 9)

```mermaid
flowchart TD
    IN[Smoothed Context Ingress] --> R9{Rule 9:\nMissing / Stale?}
    R9 -- Yes --> ST_INS["State: INSUFFICIENT_DATA\nRisk Score: 0.0\nAlert: DATA_QUALITY WARNING"]
    R9 -- No --> R1[Rule 1: Base Nominal Score = 10.0]

    R1 --> R23{Style Check}
    R23 -- Sustained Aggressive --> R2["Rule 2: Base Style +35.0\nExplanation: Sustained Aggressive Style"]
    R23 -- Transient Aggressive --> R3["Rule 3: Base Style +15.0\nExplanation: Transient Aggressive Style"]
    R23 -- Normal --> R1_OK["Style Risk: +0.0"]

    R1_OK & R2 & R3 --> R456{Active Maneuver Check}
    R456 -- Sudden Accel --> R4["Rule 4: Maneuver Risk +30.0\nAlert: WARNING"]
    R456 -- Sudden Decel --> R5["Rule 5: Maneuver Risk +45.0\nAlert: WARNING"]
    R456 -- Sudden Turn --> R6["Rule 6: Maneuver Risk +40.0\nAlert: WARNING"]
    R456 -- None --> RM_OK["Maneuver Risk: +0.0"]

    RM_OK & R4 & R5 & R6 --> R7{Rule 7:\nCompound Hazard?\n(Sustained Aggressive +\nManeuver Active)}
    R7 -- Yes --> R7_APP["Compound Bonus: +25.0\nAlert: CRITICAL"]
    R7 -- No --> R8["Rule 8: Apply Confidence Discount phi(C)"]
    R7_APP --> R8

    R8 --> CLAMP["Score Clamping: [0, 100]\nState Assignment & Cooldown Gate"]
```

---

### Rule 1: Normal Driving Baseline
- **Precondition:** Data quality is `VALID`, driving style is `NORMAL`, and active maneuver is `NONE`.
- **Score Contribution:**
  $$R_1 = 10.0$$
- **Safety State:** `NORMAL` ($10.0 < 30.0$).
- **Alert Dispatch:** None.
- **Synthesized Explanation:**
  `"Driving behavior is normal and stable with no abnormal maneuvers detected."`

---

### Rule 2: Sustained Aggressive Driving Style
- **Precondition:** Driver style classified as `AGGRESSIVE` continuously over the temporal sliding window ($R_{\text{agg}} \ge 0.60$ with $N \ge 5$).
- **Score Contribution:**
  $$R_{\text{style}} = +35.0 \cdot \phi(C_S)$$
- **Effective Score (without maneuvers):**
  $$R(t) = 10.0 + 35.0 = 45.0 \implies \mathbf{CAUTION}$$
- **Alert Dispatch:**
  - Severity: `WARNING`
  - Category: `DRIVING_STYLE`
  - Cooldown: $4.0\,\text{s}$
- **Synthesized Explanation:**
  `"Sustained aggressive driving style detected (confidence: {conf:.1%}, duration: {dur:.1f}s)."`

---

### Rule 3: Transient / Isolated Aggressive Driving
- **Precondition:** Isolated aggressive classification where $R_{\text{agg}} < 0.60$ or duration $< 1.0\,\text{s}$.
- **Score Contribution:**
  $$R_{\text{style}} = +15.0 \cdot \phi(C_S)$$
- **Effective Score (without maneuvers):**
  $$R(t) = 10.0 + 15.0 = 25.0 \implies \mathbf{NORMAL}$$
- **Alert Dispatch:** None (logged internally as advisory). Prevents nuisance alerts on intermittent throttle inputs.
- **Synthesized Explanation:**
  `"Transient aggressive behavior detected; monitoring stability."`

---

### Rule 4: Sudden Longitudinal Acceleration
- **Precondition:** Pipeline B detects `SUDDEN_ACCELERATION` (active within $2.0\,\text{s}$ persistence window).
- **Score Contribution:**
  $$R_{\text{maneuver}} = +30.0 \cdot \phi(C_M)$$
- **Effective Score (under Normal style):**
  $$R(t) = 10.0 + 30.0 = 40.0 \implies \mathbf{CAUTION}$$
- **Alert Dispatch:**
  - Severity: `WARNING`
  - Category: `MANEUVER`
  - Cooldown: $3.0\,\text{s}$
- **Synthesized Explanation:**
  `"Abrupt longitudinal acceleration detected (confidence: {conf:.1%})."`

---

### Rule 5: Sudden Longitudinal Deceleration (Harsh Braking)
- **Precondition:** Pipeline B detects `SUDDEN_DECELERATION` (active within $2.0\,\text{s}$ persistence window). Harsh braking indicates imminent forward collision risk or severe traffic turbulence.
- **Score Contribution:**
  $$R_{\text{maneuver}} = +45.0 \cdot \phi(C_M)$$
- **Effective Score (under Normal style):**
  $$R(t) = 10.0 + 45.0 = 55.0 \implies \mathbf{WARNING}$$
- **Alert Dispatch:**
  - Severity: `WARNING`
  - Category: `MANEUVER`
  - Cooldown: $3.0\,\text{s}$
- **Synthesized Explanation:**
  `"Severe longitudinal braking / sudden deceleration detected (confidence: {conf:.1%})."`

---

### Rule 6: Sudden Lateral Maneuvers (Sharp Cornering)
- **Precondition:** Pipeline B detects `SUDDEN_RIGHT_TURN` or `SUDDEN_LEFT_TURN`. High lateral jerk indicates potential roll-over hazard or aggressive lane swapping.
- **Score Contribution:**
  $$R_{\text{maneuver}} = +40.0 \cdot \phi(C_M)$$
- **Effective Score (under Normal style):**
  $$R(t) = 10.0 + 40.0 = 50.0 \implies \mathbf{CAUTION}$$
- **Alert Dispatch:**
  - Severity: `WARNING`
  - Category: `MANEUVER`
  - Cooldown: $3.0\,\text{s}$
- **Synthesized Explanation:**
  `"Sharp lateral manoeuvre ({turn_direction}) detected (confidence: {conf:.1%})."`

---

### Rule 7: Compound High-Risk Hazard Interaction
- **Precondition:** Simultaneous occurrence of **Sustained Aggressive Driving** (Rule 2) AND **Abrupt Maneuver** (Rule 4, 5, or 6).
- **Physical Justification:** An abrupt maneuver performed in isolation on a clear road may represent an evasive safety maneuver; however, when executed by an already aggressive driver, it indicates dangerous, reckless, or erratic vehicle handling.
- **Score Contribution:**
  $$R_{\text{compound}} = +25.0$$
- **Effective Score Calculation:**
  $$R(t) = 10.0 + 35.0 (\text{Style}) + 45.0 (\text{Braking}) + 25.0 (\text{Compound}) = 115.0 \xrightarrow{\text{clamped}} \mathbf{100.0} \implies \mathbf{HIGH\_RISK}$$
  $$R(t) = 10.0 + 35.0 (\text{Style}) + 40.0 (\text{Turn}) + 25.0 (\text{Compound}) = 110.0 \xrightarrow{\text{clamped}} \mathbf{100.0} \implies \mathbf{HIGH\_RISK}$$
  $$R(t) = 10.0 + 35.0 (\text{Style}) + 30.0 (\text{Accel}) + 25.0 (\text{Compound}) = 100.0 \implies \mathbf{HIGH\_RISK}$$
- **Alert Dispatch:**
  - Severity: `CRITICAL`
  - Category: `COMPOUND_HAZARD`
  - Cooldown: $5.0\,\text{s}$ (with immediate escalation bypass)
- **Synthesized Explanation:**
  `"CRITICAL COMPOUND HAZARD: Abrupt manoeuvre ({maneuver_name}) executed during sustained aggressive driving. Immediate driver intervention recommended."`

---

### Rule 8: Low-Confidence Proportional Discount
- **Precondition:** Model confidence $C$ is below the operational discount threshold ($C_{\text{discount}} = 0.60$).
- **Mathematical Scaling Operator:**
  $$\phi(C) = \begin{cases} 
      1.0 & \text{if } C \ge C_{\text{discount}} \\
      \frac{C}{C_{\text{discount}}} & \text{if } 0.0 \le C < C_{\text{discount}}
  \end{cases}$$
- **Safety Objective:** Prevents uncertain or borderline classifier outputs near decision boundaries from inducing aggressive state escalations.
- **Example:**
  - Harsh Braking ($+45.0$) at $C_M = 0.30$:
    $$\phi(0.30) = \frac{0.30}{0.60} = 0.50 \implies \text{Penalty} = 45.0 \cdot 0.50 = +22.5$$
    Total score $= 10.0 + 22.5 = 32.5 \implies \mathbf{CAUTION}$ (instead of `WARNING`).

---

### Rule 9: Telemetry Quality & Staleness Faults
- **Precondition:** Data quality status is tagged as `MISSING`, `INVALID`, or `STALE` ($(t_{\text{current}} - t) > 3.0\,\text{s}$).
- **Score Override:**
  $$R(t) = 0.0$$
- **Safety State Override:**
  $$\text{State} \equiv \mathbf{INSUFFICIENT\_DATA}$$
- **Alert Dispatch:**
  - Severity: `WARNING`
  - Category: `DATA_QUALITY`
  - Cooldown: $5.0\,\text{s}$
- **Synthesized Explanation:**
  `"Telemetry data quality fault: {data_quality_status}. Safety state cannot be certified."`

---

## 4. Alert Dispatch, Deduplication & Escalation Mechanics

### 4.1 Deduplication State Management
High-frequency telemetry (10 Hz) will produce identical rule firings every 100 ms while a condition persists. The engine tracks `_last_alert_time` and `_last_alert_severity` per `AlertCategory`:

```python
category = candidate_alert.category
last_time = self._last_alert_time.get(category, -1e9)
last_sev = self._last_alert_severity.get(category, AlertSeverity.INFO)
cooldown = self.config.alert_cooldowns.get(category, 3.0)

# Escalation Bypass Check
is_escalation = candidate_alert.severity.value > last_sev.value
in_cooldown = (current_time - last_time) < cooldown

if is_escalation or not in_cooldown:
    self._last_alert_time[category] = current_time
    self._last_alert_severity[category] = candidate_alert.severity
    active_alerts.append(candidate_alert)
```

### 4.2 Severity Priority Matrix
Alert severities have an ordinal ranking:
$$\text{INFO } (0) < \text{WARNING } (1) < \text{CRITICAL } (2)$$

When a driver moves from an isolated `WARNING` alert (e.g., sudden acceleration) directly into a `CRITICAL` alert (e.g., compound aggressive turn), the `is_escalation` flag triggers `True`. This **immediately bypasses** the cooldown period, ensuring safety-critical escalations are never silenced or delayed by prior minor warnings.

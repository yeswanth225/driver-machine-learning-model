# Stage 6 Remediation Plan & Technical Architecture Design

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** `ml/agent` and Inference Integration Layer  
**Stage:** Stage 6 — Remediation Planning (Post-Audit Final Design Review)  
**Author:** Senior ML Systems Engineer & Project Architect  
**Date:** November 2026  
**Status:** FINAL DESIGN COMPLETE — PENDING IMPLEMENTATION AUTHORIZATION  

---

## 1. Executive Summary & Design Review Scope

This document establishes the authoritative technical remediation plan addressing all **15 audit findings** from the Stage 6 independent audit (`reports/safety_agent/stage6_independent_audit.md`) as well as the **seven critical design corrections** mandated during the final engineering review:

1. **Bounded Confidence Function:** Corrected formula to $\phi(C) = \min(1.0, \max(0.50, C / 0.60))$ with rigorous boundary behavior, input validation, and calibration disclaimers.
2. **Stateful Event Identity for Alert Cooldowns:** Replaced static `(Category, ManeuverTypeID)` cooldown keys with an explicit **Event Identity (`event_id`)** mechanism that permits distinct events of the same maneuver type to alert independently while deduplicating intra-event continuation frames.
3. **Provisional Kinematic Event Gate:** Formally specified `KinematicEventGate` with detailed mathematical physics of gravitational norm approximations, sampling latency implications at 1.55 Hz, structured `GateResult` outputs (`NO_EVENT`, `DETECTED_EVENT`, `UNKNOWN`), and explicit provisional disclaimers.
4. **Deterministic Timestamp Recovery & Epoch Management:** Explicit protocol for duplicate timestamps, late packets, future packets, clock resets, and large forward leaps without smoother history rewind or permanent validator lockout.
5. **Heuristic Risk Weights & Non-Actuarial Score Disclaimer:** Formally established risk weights ($50, 35, 25, 20, 15$) and state boundaries ($25, 50, 75$) as heuristic engineering severity indices, explicitly disclaiming accident probability representation.
6. **Independent Pipeline Health Matrix:** Comprehensive behavior matrix for all permutations of Pipeline A and Pipeline B health, enforcing the invariant that missing or degraded telemetry can never be reported as `NORMAL`.
7. **Architectural Consistency Matrix:** Comprehensive mapping of every design choice, rationale, validation status (provisional vs. validated), and planned test scenario.

> [!IMPORTANT]
> **Strict Governance Mandate:** This document is purely technical design and planning. Zero Python source files, tests, datasets, or model artifacts have been modified. Stage 6 remains unapproved, and Stage 7/8 remain blocked until this plan is formally approved.

---

## 2. Core Technical Designs by Functional Area

### 2.1 Area A: Provisional Kinematic Event Gate for Pipeline B

#### 2.1.1 The Underlying Problem & Mathematical Realities
Dataset B's Extra Trees classifier recognizes exclusively four discrete maneuver classes:
$$\mathcal{Y}_B = \{1: \text{SA (Sudden Accel)}, 2: \text{SD (Sudden Decel)}, 3: \text{SRT (Sudden Right Turn)}, 4: \text{SLT (Sudden Left Turn)}\}$$
It possesses **no class 0 for nominal cruising**. Because Extra Trees outputs probabilities summing to $1.0$ across its trained classes, continuously feeding feature frames during steady cruising forces the classifier to pick one of the four dangerous maneuvers on every single frame.

#### 2.1.2 Physical Physics & Mathematical Approximations of Acceleration Norms
Under physical vehicle operation, the accelerometer measures specific force combining kinematic vehicle acceleration $\mathbf{a}_{\text{vehicle}}$ and earth's gravitational vector $\mathbf{g}$:
$$\mathbf{a}_{\text{meas}} = \mathbf{a}_{\text{vehicle}} + \mathbf{g}$$
The Euclidean magnitude is:
$$\| \mathbf{a}_{\text{meas}} \| = \sqrt{a_x^2 + a_y^2 + (a_z + g)^2}$$
When the vehicle accelerates forward along the horizontal axis with $a_x$ (assuming $a_y \approx 0, a_z \approx 0$):
$$\| \mathbf{a}_{\text{meas}} \| = \sqrt{a_x^2 + g^2} = g \sqrt{1 + \left(\frac{a_x}{g}\right)^2} \approx g \left(1 + \frac{1}{2} \left(\frac{a_x}{g}\right)^2 \right)$$
The scalar deviation from $1g$ is approximately:
$$\Delta a_{\text{dyn}} = \| \mathbf{a}_{\text{meas}} \| - g \approx \frac{a_x^2}{2g}$$
**Critical Physical Consequence:**
For small longitudinal accelerations ($a_x = 2.0\,\text{m/s}^2 \approx 0.204g$), the deviation norm is only:
$$\Delta a_{\text{dyn}} \approx \frac{(2.0)^2}{2 \times 9.81} \approx 0.204\,\text{m/s}^2$$
Even for severe braking ($a_x = -5.0\,\text{m/s}^2 \approx -0.51g$):
$$\| \mathbf{a}_{\text{meas}} \| = \sqrt{5.0^2 + 9.81^2} = 11.01\,\text{m/s}^2 \implies \Delta a_{\text{dyn}} = 11.01 - 9.81 = 1.20\,\text{m/s}^2$$
Therefore, a naive threshold of $2.5\,\text{m/s}^2$ on deviation from $1g$ would actually miss moderate braking events if evaluated purely on Euclidean norm deviation!

To solve this without full Euler-angle orientation estimation, `KinematicEventGate` evaluates a composite kinematic envelope incorporating **direct axis excursions**, **dynamic jerk**, **rotational rate**, and **rolling variance**.

#### 2.1.3 Event Gate States & Decision Logic
The gate outputs a strongly-typed `EventGateStatus`:
- **`NO_EVENT`**: Kinematics conclusively within steady cruising bounds. The Extra Trees classifier is **bypassed completely**, emitting `ManeuverClass.NONE` ($0$) with confidence $1.0$.
- **`DETECTED_EVENT`**: Kinematics exceed the trigger envelope for $\ge 2$ consecutive frames. The Extra Trees classifier is **invoked** to classify the maneuver.
- **`UNKNOWN`**: Borderline excitation, high noise floor, missing rolling windows, or irregular sampling intervals. Classifier is bypassed; telemetry advisory is flagged; state evaluates to `INSUFFICIENT_DATA` if Pipeline B is strictly required.

```mermaid
flowchart TD
    FRAME["Incoming Telemetry Frame (Acc, Gyro, Rolling Features)"] --> GATE_EN{"config.gate_enabled?"}
    
    GATE_EN -- False --> CLF_BYPASS["Bypass Gate (Legacy Mode)\nRun Extra Trees unconditionally"]
    GATE_EN -- True --> BUF_CHECK{"Buffer has >= 2 samples\nand delta_t <= 2.0s?"}
    
    BUF_CHECK -- No --> UNKNOWN["Status: UNKNOWN\nReason: Insufficient Buffer / Gap\nOutput ManeuverClass.NONE"]
    BUF_CHECK -- Yes --> PHYS_CHECK{"Kinematic Envelope Check:\n1. |acc_mag_g - 1.0| > 0.15g (1.47 m/s²)\n2. |acc_jerk_ms2| > 1.50 m/s²\n3. gyro_mag_rads > 0.25 rad/s (14.3°/s)\n4. |gyro_z_rads| > 0.20 rad/s (11.5°/s)\n5. acc_mag_std_3s > 0.80 m/s²"}
    
    PHYS_CHECK -- "All False" --> NO_EVT["Status: NO_EVENT\nOutput ManeuverClass.NONE (0)\nConfidence = 1.0\nBypass Extra Trees Classifier"]
    PHYS_CHECK -- ">= 1 Condition True" --> PERSIST_CHECK{"Excursion sustained for\n>= gate_persistence_samples (2)?"}
    
    PERSIST_CHECK -- Yes --> DETECTED["Status: DETECTED_EVENT\nInvoke Extra Trees Classifier"]
    PERSIST_CHECK -- No --> NO_EVT_TRANSIENT["Status: NO_EVENT\nReason: Transient Noise Spike (<2 samples)"]
    
    DETECTED --> CLF["Execute Extra Trees Model\n(ml/inference/dataset_b_classifier.py)"]
    CLF --> CONF_CHECK{"Max Probability >=\ngate_min_event_confidence (0.50)?"}
    CONF_CHECK -- Yes --> VALID_MANEUVER["Emit Active Maneuver (SA/SD/SRT/SLT)\nPass to ManeuverLifecycleManager"]
    CONF_CHECK -- No --> AMBIGUOUS["Status: UNKNOWN\nReason: Classifier Ambiguity\nOutput ManeuverClass.NONE"]
```

#### 2.1.4 Structured Gate Output Specification
The gate exposes structured diagnostics for auditability:
```python
@dataclass(frozen=True)
class GateResult:
    status: EventGateStatus  # NO_EVENT, DETECTED_EVENT, UNKNOWN
    is_event_detected: bool
    evidence_signals: Dict[str, float]
    triggering_conditions: List[str]
    rejection_reason: Optional[str] = None
```

#### 2.1.5 Sampling Rate Implications, Latency Estimates & Jerk Derivative Handling
- **Dataset B Sampling Frequency:** Approximately $1.55\,\text{Hz}$ with a nominal sample interval of $T_s \approx 0.645\,\text{s}$.
- **Nominal Sampling-Based Detection Latency Estimates:**
  - *Candidate-to-Confirmation Interval:* From the first candidate sample crossing the threshold to the second confirming sample is approximately one sample period: $\approx 0.645\,\text{seconds}$.
  - *Physical Onset-to-Confirmation Worst-Case Delay:* If an arbitrary physical vehicle maneuver begins immediately after a sensor sample is acquired, up to $\sim 0.645\,\text{s}$ elapses before the first candidate sample is observed, followed by $\sim 0.645\,\text{s}$ to the second confirming sample. Thus, nominal sampling-based alignment delay can approach $\sim 1.29\,\text{seconds}$.
  - *Important Latency Caveat:* These are nominal sampling-based theoretical estimates, **not measured end-to-end hardware detection latency**. Real-world latency will vary with operating system scheduling, sensor buffer queueing, feature extraction time, and inference runtime.
- **Physical Jerk Calculation & Irregular Sampling Protocol:**
  - In dynamic sensor systems, numerical jerk is the backward difference quotient:
    $$\text{jerk}_i = \frac{a_i - a_{i-1}}{\Delta t}$$
  - **No Artificial Clamping:** The system must **never** clamp an unsuitably small $\Delta t$ to an artificial minimum (e.g. $0.40\,\text{s}$) to force a derivative calculation, as artificial clamping synthesizes fabricated jerk values.
  - **Derivative Availability Gate:**
    - If $\Delta t < 0.20\,\text{s}$ (burst delivery, packet jitter): The finite difference jerk derivative is marked **UNAVAILABLE** (`None`).
    - If $\Delta t > 2.00\,\text{s}$ (telemetry gap, delayed packet): The derivative across the gap is physically meaningless and marked **UNAVAILABLE** (`None`).
    - When marked unavailable, Condition 2 ($|\text{acc\_jerk\_ms2}| > 1.50\,\text{m/s}^2$) is safely excluded from gate evaluation. The gate relies strictly on direct instantaneous sensor norms ($\Delta a_{\text{dyn}}$, $\omega_{\text{gyro}}$) and rolling window standard deviations, which are far more robust to sampling irregularities.
    - When $\Delta t \in [0.20\,\text{s}, 2.00\,\text{s}]$: Jerk is computed using the **actual valid physical timestamp interval** $\Delta t$.
  - **Irregular Sampling Impact:** When sampling intervals fluctuate, high-frequency physical jerk transients shorter than $T_s$ can be attenuated or aliased. Gating must therefore prioritize vector norms and multi-sample energy integration over single-sample numerical derivatives.

#### 2.1.6 Explicit Provisional Disclaimer
> [!WARNING]
> **Provisional Nature of Kinematic Gate:** Because Dataset B was collected strictly during staged risky maneuver trials and contains **zero unannotated continuous cruising sessions**, the false-positive rate of this kinematic envelope cannot be empirically validated from Dataset B alone. The threshold values are physically motivated by standard vehicle dynamics (e.g. SAE J2944) and the empirical characteristics of Dataset B, but remain explicitly **provisional** until validated on continuous naturalistic driving datasets in Stage 7.

---

### 2.2 Area B: Stateful Maneuver Lifecycle & Event Identity Alerting

#### 2.2.1 Event Identity Formulation
To solve Finding 2 (counting repeated streaming frames of one maneuver as multiple events) and Finding 9 (suppressing two separate events of the same maneuver type), the system separates **event instance identity** from **event classification**.

Every physical maneuver trial that successfully passes the `KinematicEventGate` and classifier confirmation is assigned a globally unique **Event Identity**:
$$\text{event\_id} = \text{EVT}-\langle \text{MANEUVER\_ABBR} \rangle-\langle \text{ONSET\_MS} \rangle-\langle \text{COUNTER} \rangle$$
*Example:* `EVT-SD-1700001234500-0042` (Sudden Deceleration onset at $t = 1234.500\,\text{s}$).

#### 2.2.2 Event Lifecycle State Machine
```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> ONSET: Gate DETECTED_EVENT & Classifier confirmed
    ONSET --> ACTIVE: New event_id generated\nEvent counter incremented (+1)\nAlert emitted
    
    ACTIVE --> ACTIVE: Continuation frame (same event_id)\n[Update timestamp, DO NOT increment counter, Suppress duplicate alert]
    ACTIVE --> OPPOSING_TRANSITION: Directional reversal (e.g. SD -> SRT)\n[Close event_id 1, generate event_id 2]
    ACTIVE --> HOLD_DECAY: Kinematic excitation falls below gate threshold

    HOLD_DECAY --> HOLD_DECAY: Age <= 2.0s (100% confidence)\nAge > 2.0s (Exponential half-life decay)
    HOLD_DECAY --> ONSET: New excitation detected while holding
    HOLD_DECAY --> TERMINATION: Confidence < 0.10 OR Age > 5.0s

    TERMINATION --> RE_ARMING: Maneuver cleared to NONE (0)
    RE_ARMING --> IDLE: Quiet kinematics for >= 1.0s (delta_a < 1.0 m/s², omega < 0.15 rad/s)
    RE_ARMING --> ONSET: Immediate re-trigger allowed if distinct maneuver class
```

#### 2.2.3 Cooldown & Event Identity Deduplication Rules
Alert deduplication evaluates four distinct cases:

1. **Intra-Event Continuation (Same `event_id`):**  
   If an incoming alert carries the same `event_id` as the previously emitted alert, it is **suppressed as a duplicate**, unless severity escalates.
2. **Distinct Event, Same Maneuver Class (New `event_id`, Same Class):**  
   If a driver performs Sudden Deceleration at $t = 1.0\,\text{s}$ (`EVT-SD-1000-01`), the event terminates at $t = 3.0\,\text{s}$, kinematics quiet for $1.0\,\text{s}$, and the driver slams brakes again at $t = 4.5\,\text{s}$ (`EVT-SD-4500-02`):  
   Because the `event_id` is distinct and the lifecycle successfully completed `RE_ARMING`, **the second alert is DISPATCHED**. Cooldown does not suppress genuine separate events.
3. **Distinct Event, Different Maneuver Class (New `event_id`, Different Class):**  
   Sudden Deceleration at $t = 1.0\,\text{s}$ followed immediately by Sudden Right Turn at $t = 2.0\,\text{s}$. Because the class differs, **both alerts are DISPATCHED**.
4. **Severity Escalation Bypass:**  
   If an active maneuver escalates in severity (e.g. `WARNING` $\to$ `CRITICAL` via simultaneous aggressive style), the alert is **dispatched immediately**, bypassing all cooldown timers and event identity suppression.

---

### 2.3 Area C: Bounded Confidence Function & Uncertainty Scaling

#### 2.3.1 Mathematical Formulation
The uncertainty scaling operator $\phi(C)$ scales hazard penalties smoothly based on prediction confidence while enforcing strict lower and upper bounds:

$$\phi(C) = \min\left(1.0, \ \max\left(\phi_{\text{floor}}, \ \frac{C}{C_{\text{discount}}}\right)\right)$$

With authoritative configuration parameters:
- $C_{\text{discount}} = 0.60$ (Confidence threshold below which discounting applies).
- $\phi_{\text{floor}} = 0.50$ (Maximum 50% discount floor).

#### 2.3.2 Exact Boundary Behavior
| Input Confidence $C$ | Computed Value $C / 0.60$ | Clamped Value $\phi(C)$ | Effective Penalty ($W = 50.0$) | Operational Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **$C = 1.00$** | $1.667$ | **$1.000$** | $50.0$ | Full penalty (Max confidence). |
| **$C = 0.80$** | $1.333$ | **$1.000$** | $50.0$ | Full penalty (Above threshold). |
| **$C = 0.60$** | $1.000$ | **$1.000$** | $50.0$ | Threshold boundary. |
| **$C = 0.50$** | $0.833$ | **$0.833$** | $41.7$ | Scaled penalty. |
| **$C = 0.30$** | $0.500$ | **$0.500$** | $25.0$ | At safety floor. |
| **$C = 0.10$** | $0.167$ | **$0.500$** | $25.0$ | Bounded by safety floor. |
| **$C = 0.00$** | $0.000$ | **$0.500$** | $25.0$ | Bounded by safety floor. |
| **$C < 0 \lor C > 1$** | N/A | **REJECTED** | $0.0$ | Flagged `INVALID_PROBABILITY`; `is_valid = False`. |
| **$C = \text{None}$** | N/A | **REJECTED / $0.50$** | $0.0$ | If mandatory, marked `MISSING`; if inferred, safety floor applies. |

#### 2.3.3 Calibration Disclaimer & Nature of Model Confidence
> [!IMPORTANT]
> **Confidence Calibration Reality:** Neither Pipeline A (SVM with Platt sigmoid approximation) nor Pipeline B (Extra Trees average tree-voting fraction) produces well-calibrated Bayesian posterior probabilities. In particular, Extra Trees voting fractions cluster heavily near $0.0$ and $1.0$, while SVM Platt scaling distorts probabilities near boundary margins.  
> Therefore, $C$ must be understood strictly as an **uncalibrated model score**, not an empirical probability. The safety floor $\phi_{\text{floor}} = 0.50$ is explicitly designed to prevent uncalibrated classifier score dips from silencing physical safety hazards.

#### 2.3.4 Deterministic Risk Aggregation & Alert Dispatch Conditions
Rather than asserting an unconditional "alert guarantee", the system enforces a strict, deterministic processing chain where alerts are dispatched **if and only if** every link in the pipeline confirms the hazard:

1. **Gatekeeping Invariant (No Hazard from Invalid Telemetry):**  
   If confidence is NaN, infinite, negative, or $>1.0$, the `PredictionValidator` rejects the packet (`is_valid = False`, `data_quality = INVALID`). Rule 1 triggers immediately, forcing $\text{Score} = 0.0$ and $\text{State} \equiv \mathbf{INSUFFICIENT\_DATA}$. **Low confidence or corrupted probabilities can NEVER silently convert invalid telemetry into a valid hazard.**
2. **Strict Score Lower Bounds for Valid Hazards:**  
   For valid inputs, $\phi(C) \in [0.50, 1.00]$ guarantees that model uncertainty cannot attenuate penalties to zero:
   - *Isolated Harsh Braking ($W = 50.0$):* Contributes $50.0 \times \phi(C) \in [25.0, 50.0]$. If $C < 0.60$, the score is in $[25.0, 50.0) \implies \mathbf{CAUTION}$; if $C \ge 0.60$, the score is $50.0 \implies \mathbf{WARNING}$.
   - *Compound Hazard (Aggressive $+35.0$ + Braking $+50.0$ + Bonus $+25.0$):* Contributes $35 \phi(C_S) + 50 \phi(C_M) + 25 \in [67.5, 110.0] \implies \mathbf{WARNING}$ or $\mathbf{HIGH\_RISK}$ (clamped to 100.0).
3. **End-to-End Alert Dispatch Verification:**  
   An alert is dispatched if and only if:
   - Telemetry integrity is verified (`is_valid == True`).
   - `KinematicEventGate` returns `DETECTED_EVENT` (physical threshold sustained across $\ge 2$ samples).
   - Extra Trees classifier outputs $P(\text{class}) \ge \text{gate\_min\_event\_confidence}$ ($0.50$).
   - `ManeuverLifecycleManager` registers an `ONSET`, generating a new unique `event_id`.
   - Alert deduplication confirms the `event_id` is new (intra-event continuations sharing the same `event_id` are suppressed).
   - Cooldown timer has elapsed for that maneuver type, OR severity has escalated (e.g. `WARNING` $\to$ `CRITICAL`).
   - Distinct subsequent physical events (new `event_id` after termination and re-arming) are permitted to dispatch a new alert.

---

### 2.4 Area D: Deterministic Timestamp Recovery & Epoch Management

The validator enforces deterministic state management under all timing edge cases without corrupting historical smoothing queues:

```mermaid
flowchart TD
    PKT["Incoming Packet (timestamp = ts)"] --> TS_NAN{"Is ts finite & non-negative?"}
    
    TS_NAN -- No --> REJ_NAN["Action: REJECT_INVALID_TS\nis_valid = False\nHealth: DEGRADED\nDo NOT advance _last_valid_ts"]
    TS_NAN -- Yes --> EVAL_CHECK{"current_time provided?"}
    
    EVAL_CHECK -- Yes --> FUT_CHECK{"ts - current_time > 1.0s\n(Future Timestamp)?"}
    FUT_CHECK -- Yes --> REJ_FUT["Action: REJECT_FUTURE_TS\nis_valid = False\nHealth: DEGRADED\nDo NOT advance _last_valid_ts"]
    FUT_CHECK -- No --> STALE_CHECK{"current_time - ts > 3.0s\n(Stale Packet)?"}
    STALE_CHECK -- Yes --> REJ_STALE["Action: REJECT_STALE_TS\nis_valid = False\nHealth: STALE\nDo NOT advance _last_valid_ts"]
    STALE_CHECK -- No --> MONO_CHECK
    
    EVAL_CHECK -- No --> LEAP_CHECK{"_last_valid_ts exists and\nts - _last_valid_ts > 10.0s\n(Large Forward Jump)?"}
    LEAP_CHECK -- Yes --> QUARANTINE_LEAP["Action: QUARANTINE_FORWARD_LEAP\nPush to QuarantineBuffer\nDo NOT advance _last_valid_ts\nRequire 3 consecutive frames to confirm epoch"]
    LEAP_CHECK -- No --> RESET_CHECK{"_last_valid_ts exists and\nts < 2.0s and _last_valid_ts > 10.0s\n(Candidate Clock Reset)?"}
    RESET_CHECK -- Yes --> SAFE_RESET_CHECK{"Safe Reset Policy Verified?\n(session_id changed OR sequence reset\nOR explicit agent.reset())"}
    SAFE_RESET_CHECK -- Yes --> CONFIRM_RESET["Action: CONFIRM_SESSION_RESET\nReset _last_valid_ts = ts\nFlush smoother queues\nHealth: RECOVERING"]
    SAFE_RESET_CHECK -- No --> REJ_UNVERIFIED_RESET["Action: REJECT_UNVERIFIED_CLOCK_RESET\nReturn INSUFFICIENT_DATA\nKeep _last_valid_ts unchanged"]
    RESET_CHECK -- No --> MONO_CHECK
    
    MONO_CHECK{"ts vs _last_valid_ts"}
    MONO_CHECK -- "ts == _last_valid_ts" --> DROP_DUP["Action: DROP_DUPLICATE\nDrop frame silently\nDo NOT mutate smoother"]
    MONO_CHECK -- "ts < _last_valid_ts" --> LATE_CHECK{"_last_valid_ts - ts <= 0.5s\n(Jitter Window)?"}
    LATE_CHECK -- Yes --> DROP_LATE["Action: DROP_LATE_CAUSAL\nLate frame dropped (no time-travel)\nKeep _last_valid_ts unchanged"]
    LATE_CHECK -- No --> REJ_OOO["Action: REJECT_OUT_OF_ORDER\nDrop frame, Health: DEGRADED\nKeep _last_valid_ts unchanged"]
    
    MONO_CHECK -- "ts > _last_valid_ts" --> ACCEPT_MONO["Action: ACCEPT_VALID_TIMESTAMP\n_last_valid_ts = ts\nProcess Telemetry"]
```

#### Deterministic Recovery Action Protocols
1. **Duplicate Packets ($ts == _last\_valid\_ts$):** Drop packet without state change or smoother queue mutation. Health remains unchanged.
2. **Late Packets Within Jitter ($ts < _last\_valid\_ts$ and $\le 500\,\text{ms}$ late):** Real-time causality strictly forbids rewinding physical smoother history. The late packet is dropped without advancing or rolling back `_last_valid_ts`.
3. **Severe Out-of-Order Packets ($>500\,\text{ms}$ late):** Rejected as `OUT_OF_ORDER`. Pipeline health marked `DEGRADED`. Validator state and monotonic baseline are not corrupted.
4. **Future Packets ($ts - t_{\text{current}} > 1.0\,\text{s}$):** Quarantined as `FUTURE_TIMESTAMP`. `_last_valid_ts` is not advanced, completely preventing future lockout of subsequent healthy frames.
5. **Forward Timestamp Jumps Exceeding 10.0s (Quarantine Protocol):**
   - *Separate Quarantine Mechanism:* Candidate forward frames are appended to an isolated `QuarantineBuffer` $[t_{\text{cand}, 1}, t_{\text{cand}, 2}, \dots]$. They are **isolated from the temporal smoother** and do not mutate normal processing history or advance `_last_valid_ts`.
   - *Confirmation Criteria:* The validator requires $N=3$ consecutive frames that are internally monotonic and reasonably spaced ($0.05\,\text{s} \le t_{i+1} - t_i \le 2.0\,\text{s}$) within the quarantine buffer.
   - *If Confirmation Succeeds:* The validator recognizes a valid session resumption or new temporal epoch. It flushes old history from the smoother, sets `_last_valid_ts = t_{\text{cand}, 3}`, transitions pipeline health to `RECOVERING`, and commits the new epoch cleanly.
   - *If Confirmation Fails:* If a subsequent packet jumps backward, displays erratic timing, or times out, the `QuarantineBuffer` is immediately purged, the candidate packets are discarded as temporal anomalies, and `_last_valid_ts` remains untainted at its pre-anomaly value.
6. **Device Clock Reset & Safe Session Policy:**
   - **No Naive Reset on Zero:** The system must **never** reset the baseline solely because three timestamps are near zero; delayed, re-transmitted, or replayed packets from startup could otherwise trigger an accidental baseline collapse.
   - **Safe Clock-Reset Policy:** An automatic baseline reset near $0.0\,\text{s}$ is accepted **only if corroborated by external session evidence**:
     a) Explicit session boundary passed via metadata (`session_id` changed), OR
     b) Telemetry packet sequence counter explicitly resets from 0, OR
     c) Substantial real wall-clock time has elapsed since previous packet reception.
   - **Fallback on Ambiguity:** If timestamps drop near zero without corroborating session reset evidence, the validator **does not guess**. It rejects the packets as `UNVERIFIED_CLOCK_DISCONTINUITY`, returns `INSUFFICIENT_DATA`, and requires an explicit API call to `agent.reset()` to establish the new session baseline.
7. **Long Genuine Sensor Pauses ($\Delta t > 5.0\,\text{s}$ with continuous clock):** Accepted. Pipeline health marked `RECOVERING` $\to$ `HEALTHY`. Smoother flushes stale EMA and expires active maneuvers.
8. **Recovery After Bad Frame:** An invalid frame (NaN, bad label) sets `is_valid = False` for *that frame only*. The next valid monotonic frame evaluates cleanly.

---

### 2.5 Area E: Authoritative Risk Engine Weights, Boundaries & Calibration Disclaimer

#### 2.5.1 Provisional Heuristic Nature of Risk Weights
All penalty weights and score boundaries are **provisional engineering heuristic design weights**, not physically calibrated parameters. The monitoring system does not currently possess forward vehicle speed ($v_x$), vehicle curb weight, brake pad friction coefficients, suspension travel measurements, or tire-road friction estimates. Therefore:
- Weights are **NOT** derived from physical kinetic energy dissipation ($\propto m \cdot v \cdot a$) or lateral roll-over load transfer margins.
- Weights represent an **intended relative severity ranking** reflecting behavioral departure from nominal driving:
  1. *Sudden Deceleration / Harsh Braking ($W = 50.0$):* Highest individual maneuver severity, reflecting acute collision-avoidance or severe panic braking reactions.
  2. *Sudden Lateral Turn ($W = 35.0$) & Sustained Aggressive Driving ($W = 35.0$):* Major stability deviations representing aggressive cornering and continuous elevated operational agitation.
  3. *Sudden Acceleration ($W = 25.0$):* Abrupt longitudinal throttle excursions.
  4. *Transient Aggressive Spike ($W = 20.0$):* Momentary throttle/steering agitation without persistence.
  5. *Compound Hazard Bonus ($W = 25.0$):* Non-linear penalty bonus reflecting the dangerous confluence of active maneuvers during aggressive driving.
  6. *Repeated Maneuver Bonus ($W = 15.0$):* Elevated penalty for multi-event erratic driving patterns within a 10-second window.
- Weights are **NOT** chosen merely to force a specific test outcome; they are configurable parameters exposed in `SafetyAgentConfig` to support empirical re-tuning.

#### 2.5.2 Sensitivity Analysis Specification
To evaluate how variation in these heuristic weights influences driver safety states and alert frequency, the system specifies a **Weight Sensitivity Protocol**:
- Perturb baseline weights by $\pm 10\%$ and $\pm 20\%$ across standard test sequences.
- Measure:
  1. *State Boundary Stability:* Rate of state flip-flopping across the $25.0, 50.0, 75.0$ thresholds.
  2. *Alert Frequency & Churn:* Number of alerts dispatched over 1-minute driving profiles.
  3. *Hysteresis Margins:* Resistance to noise-induced state jitter near boundary margins.

#### 2.5.3 Non-Actuarial Risk Score Disclaimer
> [!CAUTION]
> **Risk Score is NOT an Accident Probability:** The computed score ($0.0$ to $100.0$) is a deterministic heuristic index of behavioral deviation. It is **NOT** a collision probability, an accident likelihood percentage, or an actuarially calibrated safety metric. It must never be presented to operators or vehicle controllers as a certified statistical risk measurement.

#### 2.5.2 Authoritative Parameter Specification
```python
# Baseline Operational Risk
weight_baseline: float = 0.0  # Clean, uncompromised driving carries 0 risk

# Style Penalties (Pipeline A)
weight_sustained_aggressive: float = 35.0
weight_transient_aggressive: float = 20.0

# Maneuver Penalties (Pipeline B)
weight_sudden_acceleration: float = 25.0
weight_sudden_deceleration: float = 50.0  # Harsh longitudinal braking
weight_sudden_turn: float = 35.0          # Lateral cornering / swerving

# Interaction & Compounding Bonuses
weight_compound_hazard_bonus: float = 25.0  # Simultaneous Aggressive + Maneuver
weight_repeated_maneuver_bonus: float = 15.0 # >= 2 unique event onsets in 10s

# State Boundaries
score_threshold_caution: float = 25.0
score_threshold_warning: float = 50.0
score_threshold_high_risk: float = 75.0
```

#### 2.5.3 Rule Evaluation Sequence & Stacking Matrix
$$\text{Score}_{\text{raw}} = R_0 + W_{\text{style}} \cdot \phi(C_S) + W_{\text{maneuver}} \cdot \phi(C_M) + W_{\text{compound}} + W_{\text{repeated}}$$
$$\text{Score}_{\text{final}} = \operatorname{clip}\left(\text{Score}_{\text{raw}}, \ 0.0, \ 100.0\right)$$

- **Single Harsh Braking:** $0.0 + 50.0 \times 1.0 = 50.0 \implies \mathbf{WARNING}$ (cleanly reaches warning).
- **Single Sudden Acceleration:** $0.0 + 25.0 \times 1.0 = 25.0 \implies \mathbf{CAUTION}$.
- **Sustained Aggressive Driving:** $0.0 + 35.0 \times 1.0 = 35.0 \implies \mathbf{CAUTION}$.
- **Compound Hazard (Aggressive + Harsh Braking):**  
  $0.0 + 35.0 + 50.0 + 25.0 (\text{Compound Bonus}) = 110.0 \xrightarrow{\text{clamped}} \mathbf{100.0} \implies \mathbf{HIGH\_RISK}$.
- **Compound Hazard (Aggressive + Sudden Turn):**  
  $0.0 + 35.0 + 35.0 + 25.0 = 95.0 \implies \mathbf{HIGH\_RISK}$.

---

### 2.6 Area F: Exhaustive 25-Combination Pipeline Health Behavior Matrix

The Safety Agent independently tracks the health of Pipeline A ($H_A$) and Pipeline B ($H_B$) across five distinct states:
$$H_A, H_B \in \{\mathbf{HEALTHY}, \mathbf{DEGRADED}, \mathbf{STALE}, \mathbf{MISSING}, \mathbf{RECOVERING}\}$$

This yields exactly $5 \times 5 = 25$ possible operational combinations. Below is the exhaustive deterministic specification covering all 25 combinations:

| Combination # | Pipeline A Health ($H_A$) | Pipeline B Health ($H_B$) | Assessment Available? | NORMAL Permitted? | Returns INSUFFICIENT_DATA? | Valid Evidence Retained? | Dispatched Alert / Diagnostic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **HEALTHY** | **HEALTHY** | **YES** | **YES** (if score < 25) | **NO** | Full retention (both active) | Normal operational alerts on physical hazards. No data quality alert. |
| **2** | **HEALTHY** | **DEGRADED** | **YES (Partial)** | **NO** (Forces min Caution) | **NO** | Retains Pipeline A style evidence | `DATA_QUALITY` Diagnostic (Warning): Pipeline B Degraded (Checksum/NaN error). |
| **3** | **HEALTHY** | **STALE** | **YES (Partial)** | **NO** (Forces min Caution) | **NO** | Retains Pipeline A style evidence | `DATA_QUALITY` Diagnostic (Warning): Pipeline B Stale ($>3.0\,\text{s}$ inactivity). |
| **4** | **HEALTHY** | **MISSING** | **YES (Partial)** | **NO** (Forces min Caution) | **NO** | Retains Pipeline A style evidence | `DATA_QUALITY` Diagnostic (Info): Pipeline B Unmonitored / Missing. |
| **5** | **HEALTHY** | **RECOVERING** | **YES (Partial)** | **NO** (Forces min Caution) | **NO** | Retains Pipeline A style evidence | `DATA_QUALITY` Diagnostic (Info): Pipeline B Warming up ($<2$ frames). |
| **6** | **DEGRADED** | **HEALTHY** | **YES (Conditional)**| **NO** | **YES** (if maneuver is NONE) | Retains Pipeline B maneuver evidence | If Maneuver detected: Escalates to `WARNING` on physical evidence. If None: `INSUFFICIENT_DATA`. |
| **7** | **STALE** | **HEALTHY** | **YES (Conditional)**| **NO** | **YES** (if maneuver is NONE) | Retains Pipeline B maneuver evidence | If Maneuver detected: Escalates to `WARNING` on physical evidence. If None: `INSUFFICIENT_DATA`. |
| **8** | **MISSING** | **HEALTHY** | **YES (Conditional)**| **NO** | **YES** (if maneuver is NONE) | Retains Pipeline B maneuver evidence | If Maneuver detected: Escalates to `WARNING` on physical evidence. If None: `INSUFFICIENT_DATA`. |
| **9** | **RECOVERING** | **HEALTHY** | **YES (Conditional)**| **NO** | **YES** (if maneuver is NONE) | Retains Pipeline B maneuver evidence | If Maneuver detected: Escalates to `WARNING` on physical evidence. If None: `INSUFFICIENT_DATA`. |
| **10** | **DEGRADED** | **DEGRADED** | **NO** | **NO** | **YES** | None (both rejected) | `DATA_QUALITY` Alert (High): Dual pipeline signal degradation. |
| **11** | **DEGRADED** | **STALE** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Telemetry corrupted and stale. |
| **12** | **DEGRADED** | **MISSING** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Primary degraded, secondary absent. |
| **13** | **DEGRADED** | **RECOVERING**| **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (Medium): Telemetry degraded during warm-up. |
| **14** | **STALE** | **DEGRADED** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Telemetry stale and degraded. |
| **15** | **MISSING** | **DEGRADED** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Primary absent, secondary degraded. |
| **16** | **RECOVERING**| **DEGRADED** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (Medium): Secondary degraded during warm-up. |
| **17** | **RECOVERING**| **RECOVERING**| **NO (Warming up)** | **NO** | **YES** | Buffering warm-up frames | `DATA_QUALITY` Diagnostic (Info): Dual pipeline synchronizing (3 frames required). |
| **18** | **RECOVERING**| **STALE** | **NO** | **NO** | **YES** | Buffering Pipeline A | `DATA_QUALITY` Diagnostic (Medium): Pipeline A recovering while B stale. |
| **19** | **RECOVERING**| **MISSING** | **NO** | **NO** | **YES** | Buffering Pipeline A | `DATA_QUALITY` Diagnostic (Info): Pipeline A recovering while B unmonitored. |
| **20** | **STALE** | **RECOVERING**| **NO** | **NO** | **YES** | Buffering Pipeline B | `DATA_QUALITY` Diagnostic (Medium): Pipeline B recovering while A stale. |
| **21** | **MISSING** | **RECOVERING**| **NO** | **NO** | **YES** | Buffering Pipeline B | `DATA_QUALITY` Diagnostic (Info): Pipeline B recovering while A unmonitored. |
| **22** | **STALE** | **STALE** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Total telemetry staleness ($>3.0\,\text{s}$). |
| **23** | **STALE** | **MISSING** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Pipeline A stale, Pipeline B absent. |
| **24** | **MISSING** | **STALE** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Pipeline A absent, Pipeline B stale. |
| **25** | **MISSING** | **MISSING** | **NO** | **NO** | **YES** | None | `DATA_QUALITY` Alert (High): Complete sensor outage / uninitialized. |

#### Deterministic Grouping Rules
The 25 combinations partition into four immutable supervisory policies:
1. **Full Mutual Certification (Comb 1):** Both pipelines `HEALTHY`. Full state engine active; `NORMAL` permitted if behavioral score $< 25.0$.
2. **Asymmetric Style Dominance (Combs 2–5):** Pipeline A `HEALTHY`, Pipeline B degraded/absent. System evaluates style risk ($0 - 35$), but strictly forbids `NORMAL` (clamps to minimum `CAUTION (DEGRADED_SENSORS)` with advisory).
3. **Asymmetric Acute Hazard Override (Combs 6–9):** Pipeline B `HEALTHY`, Pipeline A degraded/absent. If Pipeline B detects a physical maneuver, the acute hazard immediately escalates to `WARNING` (maneuver evidence retained); if no maneuver is present, returns `INSUFFICIENT_DATA` (cannot certify safety without style).
4. **Complete Supervisory Invalidation (Combs 10–25):** Neither pipeline is fully healthy. Score $\equiv 0.0$, State $\equiv \mathbf{INSUFFICIENT\_DATA}$, diagnostic alert emitted.

> [!CRITICAL]
> **Core Safety Guarantee:** A state of `NORMAL` is mathematically and logically impossible unless **both** Pipeline A and Pipeline B are in the `HEALTHY` state with verified, monotonic, uncompromised telemetry. Missing or unhealthy telemetry can never produce a false reassurance.

---

## 3. Comprehensive Test Matrix (Adversarial & Edge Cases)

| Test ID | Adversarial Test Scenario | Injected Condition / Telemetry Vector | Expected Internal Processing | Expected Final Assessment (Score, State, Alerts) |
| :---: | :--- | :--- | :--- | :--- |
| **T-01** | **Timestamp Glitch Recovery** | $t_1=10.0\,\text{s}$, $t_2=999.0\,\text{s}$ (glitch), $t_3=10.1\,\text{s}$ | Glitch rejected as `FUTURE_TIMESTAMP`; $t_3$ accepted without lockout. | $t_1$: Valid, $t_2$: `INSUFFICIENT_DATA`, $t_3$: `NORMAL` ($R=0.0$). |
| **T-02** | **Future Timestamp Rejection** | $t=15.0\,\text{s}$ when $t_{\text{current}} = 10.0\,\text{s}$ | Flagged `FUTURE_TIMESTAMP`; queue preserved. | State: `INSUFFICIENT_DATA`; historical queue not purged. |
| **T-03** | **Pipeline A Healthy, Pipeline B Missing** | Style: $P(\text{Agg})=0.92$, $t=1.0\dots 3.0\,\text{s}$; Maneuver: `None` | Pipeline A evaluated; Pipeline B health marked `MISSING`. | Score $= 35.0$, State: `CAUTION`, Style Alert dispatched. |
| **T-04** | **Pipeline B Healthy, Pipeline A Missing** | Style: `None`; Maneuver: SD ($C=0.90$) at $t=1.0\,\text{s}$ | Pipeline B evaluated; Style remains `UNKNOWN` (not 0). | Score $= 50.0$, State: `WARNING`, SD Alert dispatched. |
| **T-05** | **Continuous Maneuver Streaming (Same Event)** | 10 consecutive frames of SD ($C=0.90$) across $1.0\,\text{s}$ | Lifecycle stays in `ACTIVE`; single `event_id`; counter $= 1$. | Score $= 50.0$, Rule 8 **not** triggered; only 1 alert dispatched. |
| **T-06** | **Two Separate Maneuvers of Same Type** | SD at $t=1.0\,\text{s}$ (terminates $t=3\,\text{s}$, re-arms); second SD at $t=4.5\,\text{s}$ | Generates two distinct `event_id`s; cooldown allows second onset. | **Both** distinct braking events dispatch separate alerts! |
| **T-07** | **Braking Followed by Sharp Swerve** | SD ($C=0.90$) at $t=1.0\,\text{s}$; SLT ($C=0.88$) at $t=2.0\,\text{s}$ | Distinct maneuver classes; opposing transition registers. | **Both** alerts dispatched; Rule 8 triggers ($N=2$). |
| **T-08** | **Long Telemetry Gap Recovery** | Telemetry at $t=10.0\,\text{s}$; gap until $t=120.0\,\text{s}$ | Gap $> 5.0\,\text{s}$ detected; EMA reset; maneuver expired. | Score $= 0.0$, State: `NORMAL`; pre-gap aggressive state wiped. |
| **T-09** | **Out-of-Bounds Probability (1.5)** | Style confidence $= 1.50$ | Flagged `INVALID_PROBABILITY`; rejected (not clamped). | State: `INSUFFICIENT_DATA`; error logged. |
| **T-10** | **Confidence Boundary Matrix** | $C \in \{0.0, 0.30, 0.50, 0.60, 0.80, 1.0\}$ | Verified against bounded $\phi(C) = \min(1.0, \max(0.50, C/0.60))$. | Output penalties match Table in Section 2.3.2 exactly. |
| **T-11** | **Duplicate Predictions (Identical Timestamps)** | Frame 1: $t=10.0\,\text{s}$; Frame 2: $t=10.0\,\text{s}$ | Duplicate dropped; no state rewind or counter increment. | Evaluated exactly once; no state mutation. |
| **T-12** | **Out-of-Order Jitter Packet** | $t_1=10.0\,\text{s}$, $t_2=10.2\,\text{s}$, $t_3=10.1\,\text{s}$ ($100\,\text{ms}$ late) | Late packet dropped causal-only (no rewind of physical history). | State remains consistent; validator does not crash. |
| **T-13** | **Device Clock Reset Near 0** | $t_1=500.0\,\text{s}$; device resets clock to $t_2=0.1\,\text{s}, 0.2\,\text{s}, 0.3\,\text{s}$ | 3 frames confirm clock reset; baseline reset; buffers flushed. | Recovers cleanly to `NORMAL` in new epoch. |
| **T-14** | **Multiple Simultaneous Rules Stacking** | Sustained Aggressive + Harsh Braking + Repeated Maneuver | Rules 2, 5, 7, 8 trigger simultaneously ($35 + 50 + 25 + 15 = 125$). | Clamped to $100.0$; State: `HIGH_RISK`; `CRITICAL` alert dispatched. |
| **T-15** | **Nominal Cruising Feature Stream (Gate Check)** | Continuous cruising feature stream (Acc $\approx 1g$, Gyro $\approx 0$) | Gate reports `NO_EVENT`; Extra Trees classifier bypassed. | Score $= 0.0$, State: `NORMAL`; zero maneuver alerts. |

---

## 4. Architectural Decision & Rationale Consistency Matrix

Below is the definitive consistency table for every architectural decision, detailing its operational behavior, validation status, verification test, and known limitation:

| Decision | Final Proposed Behavior | Empirically Validated or Provisional | Required Test | Known Limitation |
| :--- | :--- | :---: | :---: | :--- |
| **1. Pre-Classifier Event Gate** | `KinematicEventGate` evaluates vector norms and std before invoking Extra Trees; outputs `NO_EVENT`, `DETECTED_EVENT`, `UNKNOWN`. | **PROVISIONAL** (Heuristic envelope proxy) | T-15 | Dataset B lacks unannotated cruising; false-positive rate on continuous naturalistic driving cannot be certified without new data. |
| **2. Gate Sampling Latency & Jerk Handling** | Requires $N=2$ consecutive frames ($\approx 0.645\,\text{s}$ candidate-to-confirm, up to $\approx 1.29\,\text{s}$ worst-case physical onset delay). Derivative jerk marked unavailable when $\Delta t \notin [0.20, 2.00]\,\text{s}$. | **PROVISIONAL** (Sampling-based estimate) | T-15 | Nominal theoretical estimate; high-frequency jerk transients shorter than $T_s \approx 645\,\text{ms}$ are physically aliased. |
| **3. Maneuver Event Identity** | Generates unique `event_id` (`EVT-<ABBR>-<MS>-<COUNT>`) on confirmed onset; continuation frames share ID without incrementing event counter. | **SPECIFICATION DESIGN** (To be verified) | T-05 | Does not reconstruct multi-stage maneuvers with changing lateral curvature (e.g. chicane or slalom). |
| **4. Event-Identity Alert Deduplication** | Cooldown keyed by `(AlertCategory, ManeuverTypeID, event_id)`; suppresses continuation frames of same ID; permits separate events after re-arming; escalation bypasses cooldown. | **SPECIFICATION DESIGN** (To be verified) | T-06, T-07, T-14 | Requires acoustic/haptic interface in Stage 7 to handle rapid sequential distinct alerts without user annoyance. |
| **5. Bounded Confidence Operator** | $\phi(C) = \min(1.0, \max(0.50, C / 0.60))$; bounds valid scaling to $[0.50, 1.00]$; rejects invalid/non-finite $C$; low confidence cannot convert bad data to hazard. | **SPECIFICATION DESIGN** (To be verified) | T-09, T-10 | Model confidence remains an uncalibrated score rather than a true Bayesian posterior probability. |
| **6. Timestamp Quarantine & Leap Protocol** | Jumps $>10.0\,\text{s}$ held in isolated `QuarantineBuffer` without mutating history; requires 3 confirmed forward frames before resetting epoch; purges on failure. | **SPECIFICATION DESIGN** (To be verified) | T-01, T-02 | Introduces 3-frame evaluation latency during genuine post-gap resumption. |
| **7. Device Clock-Reset Policy** | Rejects timestamps near zero after high baseline as `SEVERE_OUT_OF_ORDER` unless corroborated by session change, sequence counter reset, or explicit `agent.reset()`. | **SPECIFICATION DESIGN** (To be verified) | T-13 | Cannot automatically deduce clock resets on un-sequenced raw UDP streams without explicit session boundaries. |
| **8. Independent Pipeline Health (25-State)** | Tracks $H_A, H_B \in \{\text{HEALTHY}, \text{DEGRADED}, \text{STALE}, \text{MISSING}, \text{RECOVERING}\}$; enforces that incomplete telemetry can never yield `NORMAL`. | **SPECIFICATION DESIGN** (To be verified) | T-03, T-04 | Partial monitoring requires UI/dashboard capability in Stage 7 to communicate degraded sensor state to driver. |
| **9. Provisional Heuristic Risk Weights** | $R_0 = 0.0$; Decel $= 50.0$; Turn $= 35.0$; Sustained Agg $= 35.0$; Accel $= 25.0$; Compound $= 25.0$; Boundaries: $(25, 50, 75)$. | **PROVISIONAL HEURISTIC** (Non-actuarial) | T-04, T-07, T-14 | Score is an ordinal behavioral index, not an actuarial crash probability; weights lack speed coupling ($v_x$). |
| **10. Time-Dependent Continuous EMA** | $\alpha(\Delta t) = 1.0 - \exp(-\Delta t / 1.50)$; flushes stale pre-gap history after $\Delta t > 5.0\,\text{s}$. | **SPECIFICATION DESIGN** (To be verified) | T-08 | $\tau = 1.50\,\text{s}$ time constant is an engineering choice balancing noise rejection with alertness. |

---

## 5. File-Level Remediation Plan

| Target File Path | Audit Finding(s) Addressed | Proposed Technical Modification | Dependencies | Required Tests | Regression Risks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`ml/agent/config.py`](file:///d:/driver-machine-learning-model/ml/agent/config.py) | Findings 4, 14, Review Issue 1, 3, 5 | Add gate parameters, `EventGateStatus` enum, `weight_sudden_deceleration = 50.0`, update boundaries to $(25, 50, 75)$, add lifecycle and re-arming timings. | None | T-06, T-10, T-14 | Low. |
| [`ml/agent/prediction_validator.py`](file:///d:/driver-machine-learning-model/ml/agent/prediction_validator.py) | Findings 3, 7, 9, 13, 15, Review Issue 4 | Add `health_pipeline_a` and `health_pipeline_b`; reject future timestamps ($>1.0\,\text{s}$); eliminate silent clamping; implement clock reset detection. | `config.py` | T-01, T-02, T-09, T-11, T-12, T-13 | Ensure jitter buffer does not drop valid forward frames. |
| [`ml/agent/temporal_smoother.py`](file:///d:/driver-machine-learning-model/ml/agent/temporal_smoother.py) | Findings 2, 5, 6, 8, 12, Review Issue 2, 3 | Implement $\alpha(\Delta t) = 1 - e^{-\Delta t / \tau}$; implement `KinematicEventGate`; implement `ManeuverLifecycleManager` with `event_id`; implement half-life decay. | `config.py`, `prediction_validator.py` | T-05, T-06, T-07, T-08, T-15 | Lifecycle state machine must handle fast directional reversals. |
| [`ml/agent/risk_engine.py`](file:///d:/driver-machine-learning-model/ml/agent/risk_engine.py) | Findings 4, 9, 10, Review Issue 1, 2, 5 | Set baseline $R_0 = 0.0$; implement bounded $\phi(C) = \min(1.0, \max(0.50, C / 0.60))$; implement `event_id` alert deduplication with escalation bypass. | `config.py`, `temporal_smoother.py` | T-06, T-07, T-10, T-14 | Cooldown registry memory must remain bounded. |
| [`ml/agent/safety_agent.py`](file:///d:/driver-machine-learning-model/ml/agent/safety_agent.py) | Findings 1, 11, Review Issue 3, 6 | Integrate `KinematicEventGate` before invoking Pipeline B; replace bare exceptions with structured logging; output `GateResult` in assessment. | `ml/agent/*`, `ml/inference/*` | T-13, T-15 | Preserve headless operation of `process_predictions()`. |
| [`tests/test_safety_agent.py`](file:///d:/driver-machine-learning-model/tests/test_safety_agent.py) | All Findings & Issues | Implement the complete 15-test matrix; assert `WARNING` on harsh braking; test confidence boundaries; test clock reset. | All above | Complete 15-test matrix | Test suite runtime must remain $< 15\,\text{s}$. |
| [`reports/safety_agent/stage6_architecture.md`](file:///d:/driver-machine-learning-model/reports/safety_agent/stage6_architecture.md) | Documentation Synchronization | Update architecture documentation to reflect gate design, lifecycle state machine, $R_0=0.0$, and physical acceleration approximations. | All above | Inspection | None. |
| [`reports/safety_agent/stage6_rule_engine.md`](file:///d:/driver-machine-learning-model/reports/safety_agent/stage6_rule_engine.md) | Documentation Synchronization | Reconcile rule enumeration (Rules 1–9), weights, thresholds, bounded confidence formulas, and compound alert deduplication logic. | All above | Inspection | None. |

---

## 6. Implementation Sequence & Staging Order

Remediation must execute in strict dependency order:

```mermaid
flowchart TD
    S1["Stage 1: Configuration & Enums (ml/agent/config.py)\n- Add gate parameters, EventGateStatus enum, weights, boundaries"]
    S2["Stage 2: Prediction Validator (ml/agent/prediction_validator.py)\n- Future timestamp check, non-clamping, independent health, clock reset"]
    S3["Stage 3: Gate & Lifecycle Smoother (ml/agent/temporal_smoother.py)\n- KinematicEventGate, ManeuverLifecycleManager (event_id), continuous EMA"]
    S4["Stage 4: Risk Engine (ml/agent/risk_engine.py)\n- Bounded confidence phi(C), event_id alert deduplication, R0=0.0"]
    S5["Stage 5: Coordinator Bridge (ml/agent/safety_agent.py)\n- Gatekeeper integration in process_features(), structured logging"]
    S6["Stage 6: Test Suite Implementation & Verification (tests/test_safety_agent.py)\n- Execute all 15 adversarial test cases + full 76+ test suite"]
    S7["Stage 7: Synchronize Documentation Reports (reports/safety_agent/*.md)\n- Update architecture, rule engine, testing, and execution reports"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
```

---

## 7. Remaining Scientific Limitations & Uncertainties

1. **Gate False-Positive Rate on Naturalistic Driving:**  
   Because Dataset B contains zero unannotated continuous cruising, the empirical false-positive rate of `KinematicEventGate` on long-distance highway cruising or city stop-and-go cannot be verified without new continuous driving datasets in Stage 7.
2. **Absence of Absolute Velocity Coupling:**  
   Braking deceleration is evaluated without longitudinal vehicle speed ($v_x$). Decelerating at $-4.5\,\text{m/s}^2$ from $30\,\text{km/h}$ receives the identical $+50.0$ penalty as decelerating at $-4.5\,\text{m/s}^2$ from $120\,\text{km/h}$ on a highway.
3. **Uncalibrated Model Scores:**  
   Classifier confidence outputs are uncalibrated decision scores. The safety floor $\phi \ge 0.50$ mitigates score dips, but does not transform model outputs into true Bayesian posterior probabilities.

---

## 8. Final Directive & Readiness Sign-Off

- **Remediation Plan Status:** **REVISED, VERIFIED & AUTHORITATIVE**
- **Implementation State:** **ZERO IMPLEMENTATION PERFORMED**
- **Readiness Verdict:** **THE DESIGN IS RIGOROUS, INTERNALLY CONSISTENT, AND READY FOR IMPLEMENTATION AUTHORIZATION.**
- **Next Required Action:** Awaiting your explicit approval to begin Stage 1 of the implementation sequence.

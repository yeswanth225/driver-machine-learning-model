# Stage 6 Architecture Design: Continuous Residual Risk Integration

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** `ml/agent` Production Architecture  
**Stage:** Stage 6 — Intelligent Safety Monitoring Agent (Controlled Remediation)  
**Document Type:** Architectural Investigation & Integration Design (READ-ONLY)  
**Date:** November 2026  
**Status:** PROPOSED ARCHITECTURAL DESIGN — READ-ONLY INVESTIGATION — AWAITING IMPLEMENTATION AUTHORIZATION  

---

## 1. Executive Summary

During the Stage 6 Checkpoint 1 verification of **Defect D-06 (Continuous Residual Risk Decay)**, the mathematical implementation and analytical boundary behavior of `DeterministicRiskEngine.calculate_residual_risk()` were validated and provisionally accepted. However, production call-site inspection revealed that `calculate_residual_risk()` is **not currently connected** into the live production risk-scoring pipeline (`DeterministicRiskEngine.evaluate()`).

Connecting continuous residual decay into the live pipeline cannot be accomplished via a naive 1-line function call in `evaluate()`. A live integration requires tracking **independent excitation-end timestamps**, supporting **simultaneous orthogonal maneuver decay** (e.g., harsh braking concurrent with abrupt steering), preventing **continuation frames from resetting decay timers**, and enforcing **incident identity preservation across a 2.50-second quiet hysteresis window**.

In strict compliance with governance rules, this document provides the formal architectural design to achieve production integration safely, decoupling computational rolling buffers from physical incident lifecycles, and aligning with the co-requisite requirements of **D-02 (Incident Counting)**, **D-09 (Dissimilar Cooldown)**, and **D-10 (Modality Confidence Scaling)**.

---

## 2. Current Architecture & Verified Call Flow

### 2.1 Current Ingress & Evaluation Call Flow
In the current implementation, sensor telemetry traverses four sequential components:

```mermaid
flowchart TD
    INGRESS["1. Telemetry Ingress\nfeatures_b (1.55 Hz), features_c (10.0 Hz)"] --> AGENT["2. IntelligentSafetyAgent.process_features()"]
    
    subgraph AGENT_PIPELINE ["Live Production Path"]
        GATE["KinematicEventGate.evaluate(features_b)\n-> VALID_QUIESCENT / VALID_EXCITED"]
        CLF["Classifier Inference\nPipeline A (SVM) + Pipeline B (Extra Trees)"]
        VAL["PredictionValidator.validate()\n-> ValidatedPrediction"]
        SMOOTH["TemporalSmoother.update()\n-> SmoothedContext"]
        RISK["DeterministicRiskEngine.evaluate()\n-> RuleEvaluationResult"]
        OUT["SafetyAssessment Synthesizer"]
    end

    AGENT --> GATE
    GATE --> CLF
    CLF --> VAL
    VAL --> SMOOTH
    SMOOTH --> RISK
    RISK --> OUT
```

### 2.2 Production Call-Site Inspection Findings
1. [`ml/agent/risk_engine.py`](file:///d:/driver%20ml%20model/driver-monitoring-agent/ml/agent/risk_engine.py#L274-L350):
   ```python
   # Rule 4: SUDDEN_ACCELERATION_MANEUVER
   if ctx.is_maneuver_active and ctx.active_maneuver == ManeuverClass.SUDDEN_ACCELERATION.value:
       raw_score += self.config.weight_sudden_acceleration
   # Rule 5: SUDDEN_DECELERATION_MANEUVER
   elif ctx.is_maneuver_active and ctx.active_maneuver == ManeuverClass.SUDDEN_DECELERATION.value:
       raw_score += self.config.weight_sudden_deceleration
   # Rule 6: SUDDEN_TURN_MANEUVER
   elif ctx.is_maneuver_active and ctx.active_maneuver in [
       ManeuverClass.SUDDEN_RIGHT_TURN.value, ManeuverClass.SUDDEN_LEFT_TURN.value
   ]:
       raw_score += self.config.weight_sudden_turn
   ```
   **Verified Finding:** Rules 4, 5, and 6 evaluate via an `if / elif / elif` ladder that adds static, fixed weights ($25.0, 50.0, 35.0$). `calculate_residual_risk()` is defined as a class method but is **never invoked**.

2. [`ml/agent/temporal_smoother.py`](file:///d:/driver%20ml%20model/driver-monitoring-agent/ml/agent/temporal_smoother.py#L191-L245):
   ```python
   # Maneuver Ingress & Continuation
   if m_pred is not None and m_pred != ManeuverClass.NONE.value:
       is_continuation = (
           self._active_maneuver == m_pred
           and self._active_maneuver_timestamp is not None
           and (ts - self._active_maneuver_timestamp) <= self.config.maneuver_persistence_seconds
       )
       self._active_maneuver = m_pred
       self._active_maneuver_conf = max(self._active_maneuver_conf, m_conf) if is_continuation else m_conf
       self._active_maneuver_timestamp = ts  # <--- OVERWRITTEN ON EVERY FRAME

   # Maneuver Expiration
   if self._active_maneuver != ManeuverClass.NONE.value and self._active_maneuver_timestamp is not None:
       maneuver_age = ts - self._active_maneuver_timestamp
       if 0.0 <= maneuver_age <= self.config.maneuver_persistence_seconds:  # <--- CUTOFF AT 2.0s
           is_maneuver_active = True
       else:
           self._active_maneuver = ManeuverClass.NONE.value
           self._active_maneuver_conf = 0.0
           is_maneuver_active = False
   ```
   **Verified Finding:** `self._active_maneuver` is a single scalar integer. `self._active_maneuver_timestamp` is updated on every frame. When `ts - self._active_maneuver_timestamp > 2.0\,\text{s}`, the maneuver is forcibly cleared to `NONE`.

---

## 3. Root Cause Analysis of Integration Blockers

Connecting `calculate_residual_risk()` directly into the current code without architectural updates causes five fatal defects:

```
+----------------------------------------------------------------------------------------------------+
|                                    INTEGRATION BLOCKERS SUMMARY                                     |
+----------------------------------------------------------------------------------------------------+
| 1. Downstream Premature Truncation:                                                                |
|    TemporalSmoother forces is_maneuver_active = False at dt = 2.0s.                                 |
|    D-06 requires continuous exponential decay until t_retire = 4.241s (W=50.0) or 3.856s (W=35.0).  |
|    Connecting naive decay causes residual risk to cliff-drop abruptly from 19.84 pts to 0.00 pts!   |
+----------------------------------------------------------------------------------------------------+
| 2. Continuation Timestamp Overwriting:                                                             |
|    _active_maneuver_timestamp = ts runs on every streaming frame.                                  |
|    Passing (ts - _active_maneuver_timestamp) resets the decay timer to 0 on every ongoing frame.   |
+----------------------------------------------------------------------------------------------------+
| 3. Single-Maneuver Scalar Bottleneck:                                                              |
|    SmoothedContext holds only ONE active_maneuver. A Sudden Turn at t=2.0s overwrites Sudden      |
|    Decel from t=1.0s, annihilating the ongoing deceleration residual decay.                        |
+----------------------------------------------------------------------------------------------------+
| 4. Mutual Exclusivity in Risk Rules:                                                               |
|    Rules 4, 5, and 6 use elif ladders, prohibiting simultaneous risk summation of orthogonal       |
|    maneuver hazards (Decel + Turn).                                                                |
+----------------------------------------------------------------------------------------------------+
| 5. Cross-Defect Couplings:                                                                         |
|    Modifying persistence touches D-02 (incident boundaries). Multi-maneuver alerts touch D-09      |
|    (cooldowns). Confidence scaling touches D-10 (three-tier scaling).                              |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Architectural Analysis: Answers to the Nine Questions

### 4.1 Where should each maneuver's excitation-end timestamp be stored?
**Answer: In an encapsulated `ManeuverLifecycleTracker` within `TemporalSmoother`, passed downstream via `SmoothedContext.active_contributions`.**

- **Rationale:** Physical kinematic sensors operate at $1.55\,\text{Hz}$ and $10.0\,\text{Hz}$. Determining when physical excitation ends requires evaluating the kinematic stream (`KinematicEventGate` status and classifier outputs over time). This temporal filtering is the explicit domain of `TemporalSmoother`.
- Each active maneuver contribution is represented as a structured record:
  ```python
  @dataclass
  class ManeuverContribution:
      event_id: str                      # Unique physical incident identity
      maneuver_class: int                # ManeuverClass enum value (1=SA, 2=SD, 3=SRT, 4=SLT)
      initial_weight: float              # W (e.g. 50.0 for SD, 35.0 for Turn)
      confidence: float                  # Uncalibrated model confidence C
      onset_timestamp: float             # Physical onset timestamp (t_onset)
      excitation_end_timestamp: Optional[float]  # None while excited; locked once excitation ends
      is_excited: bool                   # True while sensor energy exceeds gate threshold
  ```
- `SmoothedContext` exposes a mapping `active_contributions: Dict[int, ManeuverContribution]` keyed by maneuver class.

---

### 4.2 How can deceleration and turning contributions decay independently?
**Answer: By maintaining independent multi-channel maneuver slots and summing orthogonal modality residuals.**

- Longitudinal dynamics (Deceleration/Acceleration) and lateral dynamics (Turning) are physically orthogonal. A driver slamming the brakes at $t=1.0\,\text{s}$ and swerving at $t=2.0\,\text{s}$ generates two distinct physical hazards.
- In `TemporalSmoother`, maintain independent active slots:
  ```python
  self._active_channels: Dict[int, ManeuverContribution] = {}
  ```
- When Sudden Deceleration arrives, it occupies slot `ManeuverClass.SUDDEN_DECELERATION` with its own $t_{\text{end}}$.
- When Sudden Right Turn arrives, it occupies slot `ManeuverClass.SUDDEN_RIGHT_TURN` with its own $t_{\text{end}}$.
- In `DeterministicRiskEngine`, the total maneuver risk is computed by iterating over all active channels:
  $$R_{\text{maneuver\_total}}(t) = \sum_{m \in \text{ctx.active\_contributions.values()}} R_m(t)$$
  Where for each contribution:
  $$\Delta t_m = \begin{cases} 0.0 & \text{if } m.\text{is\_excited} \\ t - m.\text{excitation\_end\_timestamp} & \text{if not } m.\text{is\_excited} \end{cases}$$
  $$R_m(t) = \text{calculate\_residual\_risk}(m.\text{initial\_weight}, \Delta t_m) \cdot \phi(m.\text{confidence})$$
- Because $\Delta t_m$ is calculated independently from each channel's locked $t_{\text{end}}$, Deceleration decays on its timeline ($t_{\text{retire}} = t_{\text{end}, \text{decel}} + 4.241\,\text{s}$) while Turning decays on its separate timeline ($t_{\text{retire}} = t_{\text{end}, \text{turn}} + 3.856\,\text{s}$).

---

### 4.3 How can ongoing frames avoid restarting an existing contribution's decay timer?
**Answer: By strictly enforcing a two-phase state transition: mutable tracking during excitation, immutable locking upon excitation termination.**

```mermaid
stateDiagram-v2
    [*] --> EXCITATION: Onset Frame (Gate VALID_EXCITED)
    EXCITATION --> EXCITATION: Continuation Frames (Update last_active_ts; decay timer NOT started)
    EXCITATION --> DECAY_LOCKED: Gate returns VALID_QUIESCENT (Physical Excitation Ends)
    
    note right of DECAY_LOCKED
        excitation_end_timestamp = last_active_ts
        LOCK TIMESTAMP IMMUTABLY
        is_excited = False
    end note

    DECAY_LOCKED --> DECAY_LOCKED: Streaming Frames (dt = ts - excitation_end_timestamp)
    DECAY_LOCKED --> RETIRED: dt > t_retire (Residual strictly < 2.50 pts)
    RETIRED --> [*]: Pruned from active channels
```

- While `is_excited == True`, `last_active_ts = ts` updates, but no decay occurs ($\Delta t \equiv 0.0\,\text{s}$).
- The moment the gate detects quiet kinematics or the maneuver concludes, `excitation_end_timestamp` is locked to `last_active_ts` and `is_excited` transitions to `False`.
- On all subsequent frames, `excitation_end_timestamp` is **read-only and immutable**. Ingress frames at $t_1, t_2, \dots$ provide the current time $ts$ to evaluate decay, but **never overwrite** `excitation_end_timestamp`.

---

### 4.4 How can a new physical excitation re-arm lifecycle timers without incorrectly restarting an older contribution's residual decay?
**Answer: By decoupling Incident Generational Identity (`event_id`) from residual decay tracking.**

When a new physical excitation of maneuver class $M$ arrives:
1. **Case A: New excitation belongs to an orthogonal class (e.g., Turn after Decel):**  
   It enters a separate channel slot. The older deceleration residual continues its decay undisturbed.
2. **Case B: New excitation belongs to the same class (e.g., Decel 2 while Decel 1 residual is decaying):**  
   - If re-arming conditions are met ($\ge 2.50\,\text{s}$ quiet buffer), a new incident identity `EVT-SD-<ts>-02` is created.
   - The new excitation requires $50.0\,\text{pts}$ base risk.
   - The decaying residual ($R_{\text{old}} \le 25.0\,\text{pts}$) is subsumed by the new full-magnitude excitation ($50.0 \ge R_{\text{old}}$).
   - The channel transitions back to `is_excited = True` under the new `event_id`, resetting decay for the *new* event while cleanly superseding the decaying residual without double-counting base weights.

---

### 4.5 How can the architecture preserve incident identity until 2.50 seconds of uninterrupted quiet?
**Answer: By implementing an explicit `uninterrupted_quiet_since` hysteresis timer in `TemporalSmoother`.**

- **Hysteresis Specification ($T_{\text{re-arm}} = 2.50\,\text{seconds}$):**  
  A physical incident envelope does not close immediately upon the first sub-threshold sample. Driving often exhibits momentary hesitations or multi-pulse braking.
- **Timer Mechanics:**
  ```python
  if gate_status == GateStatus.VALID_QUIESCENT:
      if self._uninterrupted_quiet_since is None:
          self._uninterrupted_quiet_since = ts
      quiet_elapsed = ts - self._uninterrupted_quiet_since
      if quiet_elapsed >= 2.50:
          self._active_incident_closed = True
  elif gate_status == GateStatus.VALID_EXCITED:
      # Kinematic excitation re-occurs!
      if not self._active_incident_closed:
          # Intra-incident re-excitation: retain existing event_id
          self._uninterrupted_quiet_since = None
      else:
          # Quiet threshold was met: re-arm and generate new event_id
          self._start_new_incident(ts)
  ```
- This guarantees that incident identity (`event_id`) is preserved across brief gaps ($< 2.50\,\text{s}$) and only closes when kinematics remain uninterruptedly quiet for $\ge 2.50\,\text{s}$.

---

### 4.6 How can computational window rollover remain separate from physical incident identity?
**Answer: By enforcing strict architectural isolation between Ring Buffers (computational windows) and the Incident Registry (physical lifecycle).**

| Functional Domain | Component | State Storage | Pruning / Lifetime Policy |
| :--- | :--- | :--- | :--- |
| **Computational Moving Windows** | `TemporalSmoother._history` | Circular `deque` of `(ts, ValidatedPrediction)` | Time-based eviction: strictly pruned when $t < ts - 10.0\,\text{s}$. Used solely for rolling ratios and EMA. |
| **Physical Incident Lifecycles** | `ManeuverLifecycleTracker` | Dictionary of `ManeuverContribution` records | Lifecycle-based eviction: pruned when residual strictly retires ($t > t_{\text{retire}}$) and quiet re-arming completes ($\ge 2.50\,\text{s}$). |

- When an old telemetry sample at $t=1.0\,\text{s}$ is evicted from `_history` at $t=11.1\,\text{s}$, it has **zero impact** on physical incidents or active residuals. Physical residuals retire according to analytical time $t_{\text{retire}}$, completely decoupled from buffer rollover.

---

### 4.7 How should the risk engine sum independent contributions without double-counting?
**Answer: By enforcing modal independence for orthogonal classes, ceiling-clamping for identical classes, and single-instance bonus evaluation.**

$$\text{Score}_{\text{raw}} = R_0 + R_{\text{style}}(t) + \sum_{k \in \text{modalities}} R_k(t) + R_{\text{compound}}(t) + R_{\text{repeated}}(t)$$

1. **Orthogonal Maneuver Modalities ($R_k$):**  
   Longitudinal Deceleration ($W=50.0$) and Lateral Steering ($W=35.0$) sum directly:
   $$R_{\text{maneuvers}} = R_{\text{decel}}(\Delta t_D) + R_{\text{turn}}(\Delta t_T)$$
2. **Identical-Class Ceiling Clamping:**  
   If overlapping residuals exist within the same class, the contribution is capped at $\max_i(R_{k, i}) \le W_k$. Two consecutive brake taps can never produce $50 + 50 = 100$ base braking penalty.
3. **Compound Bonus ($W_{\text{compound}} = 25.0$):**  
   Evaluated as a scalar indicator: added **at most once per frame** if *any* maneuver channel is active concurrently with an aggressive driving style. It is never multiplied by the number of active maneuvers.
4. **Final Clamping:**  
   $$\text{Score}_{\text{final}} = \min(100.0, \max(0.0, \text{Score}_{\text{raw}}))$$

---

### 4.8 How can the design preserve D-02 incident counting, D-09 cooldown and D-10 confidence scaling?
**Answer:**
- **Preserving D-02 (Event Deduplication & Streaming Counting):**  
  `event_id` is assigned at initial onset. Continuation frames share the identical `event_id`. `recent_maneuver_count` counts unique `event_id` values within the past $10.0\,\text{seconds}$. Ongoing frames of the same maneuver do not increment the counter or spuriously trigger Rule 8 bonuses.
- **Preserving D-09 (Dissimilar Maneuver Cooldown Bypass):**  
  Alert deduplication in `DeterministicRiskEngine` partitions alert timers by `(AlertCategory.MANEUVER, maneuver_class)`. When Sudden Deceleration triggers at $t=1.0\,\text{s}$ and Sudden Turn triggers at $t=2.0\,\text{s}$, the Turn alert checks the lateral cooldown slot, which is uninitialized. The Turn alert is dispatched with zero suppression, preserving D-09.
- **Preserving D-10 (Three-Tier Confidence Scaling):**  
  Each contribution scales its residual independently by its own modality confidence:
  $$R_k(t) = \text{calculate\_residual\_risk}(W_k, \Delta t_k) \cdot \phi(C_k)$$
  Where $\phi(C) = 0.0$ for $C < 0.40$, $C / 0.60$ for $0.40 \le C < 0.60$, and $1.0$ for $C \ge 0.60$. A high-confidence style ($C=0.95$) never artificially inflates an uncertain maneuver ($C=0.35$), preserving D-10.

---

### 4.9 What regression tests are required to verify the complete lifecycle?
**Answer: A dedicated 10-test lifecycle regression suite (`test_t17_residual_risk_lifecycle_matrix`):**

1. **Analytical Boundary Decay Lifecycle:** Verify continuous half-life decay from $W \to W/2 \to 2.50 \to 0.00$.
2. **Continuation Frame Invariance:** 5 consecutive frames during excitation do not advance decay timer; decay starts only after excitation ends.
3. **Orthogonal Dual-Maneuver Decay:** Decel ($W=50.0$) and Turn ($W=35.0$) decay independently on separate clocks; Decel retires at $4.241\,\text{s}$ post-excitation while Turn retires at $3.856\,\text{s}$.
4. **Dissimilar Alert Dispatch (D-09):** Turn alert emits immediately during ongoing Decel residual decay.
5. **Modality Confidence Scaling (D-10):** Low-confidence maneuver ($C=0.35$) produces zero residual even when style confidence is $0.95$.
6. **Re-Arming Quiet Hysteresis ($2.50\,\text{s}$):** Re-excitation at $\Delta t_{\text{quiet}} = 1.5\,\text{s}$ retains `event_id`; re-excitation at $\Delta t_{\text{quiet}} = 2.6\,\text{s}$ increments counter and generates new `event_id`.
7. **Buffer Rollover Independence:** Active residual decay is unaffected when telemetry advances past the $10.0\,\text{s}$ circular buffer pruning horizon.
8. **Same-Class Double-Counting Prevention:** Consecutive brake pulses within decay window cap at $50.0\,\text{pts}$ base weight.
9. **Quiescent Zero-Residual Invariant:** Cruising telemetry maintains $R_{\text{residual}} \equiv 0.00$.
10. **Global Score Clamping:** Multiple overlapping hazards stack deterministically and clamp cleanly to $100.0$.

---

## 5. Target Architecture & Component Responsibilities

```mermaid
classDiagram
    class IntelligentSafetyAgent {
        +gate: KinematicEventGate
        +validator: PredictionValidator
        +smoother: TemporalSmoother
        +risk_engine: DeterministicRiskEngine
        +process_predictions()
        +process_features()
    }

    class TemporalSmoother {
        -_active_channels: Dict[int, ManeuverContribution]
        -_uninterrupted_quiet_since: Optional[float]
        -_history: deque
        -_maneuver_events: deque
        +update(val_pred): SmoothedContext
    }

    class ManeuverContribution {
        +event_id: str
        +maneuver_class: int
        +initial_weight: float
        +confidence: float
        +onset_timestamp: float
        +excitation_end_timestamp: Optional[float]
        +is_excited: bool
    }

    class SmoothedContext {
        +timestamp: float
        +style_ema: float
        +active_contributions: Dict[int, ManeuverContribution]
        +active_maneuver: Optional[int]
        +active_maneuver_confidence: float
        +active_maneuver_age_seconds: float
        +is_maneuver_active: bool
        +recent_maneuver_count: int
    }

    class DeterministicRiskEngine {
        +calculate_residual_risk()
        +evaluate(ctx): RuleEvaluationResult
    }

    IntelligentSafetyAgent --> TemporalSmoother
    IntelligentSafetyAgent --> DeterministicRiskEngine
    TemporalSmoother --> ManeuverContribution
    TemporalSmoother --> SmoothedContext
    DeterministicRiskEngine ..> SmoothedContext
```

### Component Contracts
1. **`TemporalSmoother`:**
   - Ingests `ValidatedPrediction`.
   - Maintains `_active_channels: Dict[int, ManeuverContribution]`.
   - Locks `excitation_end_timestamp` when excitation terminates.
   - Prunes channels whose analytical retirement time has passed ($t > t_{\text{retire}}$).
   - Generates `SmoothedContext` containing `active_contributions`.
2. **`SmoothedContext`:**
   - Carries `active_contributions` dictionary.
   - Provides backward-compatible scalar properties (`active_maneuver`, `active_maneuver_confidence`, `is_maneuver_active`) mapped to the highest-severity active contribution for legacy callers.
3. **`DeterministicRiskEngine`:**
   - Iterates over `ctx.active_contributions`.
   - Evaluates continuous residual risk using `calculate_residual_risk()`.
   - Multiplies each contribution by $\phi(C)$.
   - Sums independent orthogonal contributions without double-counting.

---

## 6. Event Lifecycle & Residual Decay Sequence Diagram

The sequence diagram below traces an incident where Sudden Deceleration ($t=1.0\,\text{s}$) is followed by Sudden Right Turn ($t=2.0\,\text{s}$):

```mermaid
sequenceDiagram
    autonumber
    actor Driver
    participant Gate as KinematicEventGate
    participant Smoother as TemporalSmoother
    participant RiskEngine as DeterministicRiskEngine

    Note over Driver, RiskEngine: T = 1.0s: Sudden Deceleration Onset
    Driver->>Gate: Harsh Braking Telemetry (sigma_a = 0.85)
    Gate-->>Smoother: VALID_EXCITED (Decel confirmed)
    Smoother->>Smoother: Open Channel[SD]: event_id=EVT-SD-1000-01, W=50.0, is_excited=True
    Smoother-->>RiskEngine: SmoothedContext(active_contributions={SD: is_excited=True})
    RiskEngine->>RiskEngine: SD: dt=0 -> R=50.0. Score=50.0 (WARNING). Alert: ALT-MAN-SD

    Note over Driver, RiskEngine: T = 1.8s: Deceleration Excitation Ends
    Driver->>Gate: Telemetry quiets (sigma_a = 0.15)
    Gate-->>Smoother: VALID_QUIESCENT
    Smoother->>Smoother: Lock SD: excitation_end_ts = 1.8s, is_excited=False

    Note over Driver, RiskEngine: T = 2.0s: Sudden Turn Onset (Concurrent with Decel Plateau)
    Driver->>Gate: Abrupt Steering Telemetry (omega_z = 0.35)
    Gate-->>Smoother: VALID_EXCITED (Turn confirmed)
    Smoother->>Smoother: Open Channel[SRT]: event_id=EVT-SRT-2000-02, W=35.0, is_excited=True
    Smoother-->>RiskEngine: SmoothedContext(active_contributions={SD: dt=0.2s, SRT: is_excited=True})
    RiskEngine->>RiskEngine: SD in Plateau (dt=0.2s <= 1.0s) -> R=50.0
    RiskEngine->>RiskEngine: SRT Excited -> R=35.0
    RiskEngine->>RiskEngine: Total Maneuver Risk = 50.0 + 35.0 = 85.0 (HIGH_RISK)
    RiskEngine->>RiskEngine: Alert: ALT-MAN-SRT (Dispatched! Bypasses Decel cooldown via D-09)

    Note over Driver, RiskEngine: T = 3.55s: Decel Half-Life (dt = 1.75s post-excitation)
    Driver->>Gate: Quiet Cruising
    Smoother->>Smoother: SD: dt = 3.55 - 1.8 = 1.75s -> R = 25.0 pts
    Smoother->>Smoother: SRT: dt = 3.55 - 2.6 = 0.95s (Plateau) -> R = 35.0 pts
    RiskEngine->>RiskEngine: Total Maneuver Risk = 25.0 + 35.0 = 60.0 (WARNING)

    Note over Driver, RiskEngine: T = 6.041s: Decel Analytical Boundary (t_retire)
    Smoother->>Smoother: SD: dt = 4.241446s == t_retire -> R = 2.50 pts (Retained)

    Note over Driver, RiskEngine: T = 6.050s: Decel Retirement
    Smoother->>Smoother: SD: dt = 4.250s > t_retire -> R = 0.00 pts (RETIRED & PRUNED)
    Smoother-->>RiskEngine: SmoothedContext(active_contributions={SRT: Decaying})
```

---

## 7. Proposed API & Schema Modifications

### 7.1 `SmoothedContext` Schema Additions
In [`ml/agent/temporal_smoother.py`](file:///d:/driver%20ml%20model/driver-monitoring-agent/ml/agent/temporal_smoother.py):
```python
@dataclass
class SmoothedContext:
    timestamp: float
    style_ema: float
    style_aggressive_ratio: float
    is_sustained_aggressive: bool
    is_transient_aggressive: bool
    current_style: Optional[int]
    current_style_confidence: float
    
    # NEW: Multi-channel active contributions dictionary
    active_contributions: Dict[int, ManeuverContribution] = field(default_factory=dict)
    
    # Backward-compatible scalar properties
    active_maneuver: Optional[int] = None
    active_maneuver_confidence: float = 0.0
    active_maneuver_age_seconds: float = 0.0
    is_maneuver_active: bool = False
    recent_maneuver_count: int = 0
    recent_maneuver_history: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = True
    data_quality_status: str = "VALID"
```

### 7.2 Backward Compatibility
To prevent breaking existing unit tests that inspect `ctx.active_maneuver` directly:
- `active_maneuver`: Populated with the highest-risk active maneuver class (e.g. `SUDDEN_DECELERATION`).
- `active_maneuver_confidence`: Populated with the confidence of that highest-risk maneuver.
- `is_maneuver_active`: Evaluates to `len(self.active_contributions) > 0`.
- All legacy tests expecting scalar access remain 100% operational without modification.

---

## 8. Defect Dependencies & Implementation Sequence

Production integration of continuous residual risk touches four interconnected defects:

```mermaid
flowchart LR
    D06["D-06: Residual Decay\n(Math & Boundary Approved)"] --> D08["D-08: Missing Style\n(Validator / Smoother)"]
    D08 --> D09["D-09: Dissimilar Maneuver Cooldown\n(Multi-Channel Alert Deduplication)"]
    D09 --> D10["D-10: Modality Confidence Scaling\n(phi(C) per contribution)"]
    D10 --> INT["Integrated Production Pipeline\n(Multi-Channel Residual Decay)"]
```

### Implementation Staging Order:
1. **Checkpoint 2 (D-08):** Ensure missing style predictions do not default to normal driving.
2. **Checkpoint 3 (D-09):** Refactor `DeterministicRiskEngine` alert deduplication to partition cooldowns by `(AlertCategory, maneuver_class)`, allowing dissimilar maneuvers to emit alerts without suppression.
3. **Checkpoint 4 (D-10):** Implement unified three-tier confidence scaling $\phi(C)$ independently for style and maneuvers.
4. **Checkpoint 5 (Production Residual Integration):** Integrate `ManeuverLifecycleTracker` into `TemporalSmoother`, update `SmoothedContext`, and sum independent residuals in `DeterministicRiskEngine.evaluate()`.

---

## 9. Comprehensive Test Matrix (10 Lifecycle Test Cases)

| Test ID | Operational Scenario | Injected Conditions | Expected Lifecycle Progression | Expected Scores / State |
| :---: | :--- | :--- | :--- | :--- |
| **LT-01** | **Single Decel Full Lifecycle** | SD ($W=50.0$) at $t=1.0\,\text{s}$; ends $t=1.8\,\text{s}$. | $t \in [1.8, 2.8]\,\text{s}$: Plateau ($R=50.0$); $t=3.55\,\text{s}$: Half-life ($R=25.0$); $t=6.041\,\text{s}$: Boundary ($R=2.50$); $t=6.050\,\text{s}$: Retired ($R=0.0$). | $50.0 \to 25.0 \to 2.50 \to 0.00$. State: $\mathbf{WARNING} \to \mathbf{CAUTION} \to \mathbf{NORMAL}$. |
| **LT-02** | **Continuation Frame Invariance** | SD across 4 frames ($t = 1.0, 1.4, 1.8, 2.2\,\text{s}$). | Excitation ends at $t=2.2\,\text{s}$. Plateau holds until $t=3.2\,\text{s}$. Decay timer does NOT reset on intermediate frames. | Single incident ID; decay clock starts strictly at $t=2.2\,\text{s}$. |
| **LT-03** | **Dual Orthogonal Maneuver Decay** | SD at $t=1.0\,\text{s}$ ($t_{\text{end}}=1.8\,\text{s}$); SRT at $t=2.0\,\text{s}$ ($t_{\text{end}}=2.6\,\text{s}$). | At $t=2.5\,\text{s}$: SD in plateau ($50.0$), SRT in plateau ($35.0$). Total $= 85.0\,\text{pts}$. Decel retires at $t=6.041\,\text{s}$; Turn retires at $t=6.456\,\text{s}$. | Combined: $\mathbf{HIGH\_RISK}$ ($85.0$). Separate retirement timestamps. |
| **LT-04** | **Dissimilar Maneuver Alerting (D-09)** | SD at $t=1.0\,\text{s}$; SRT at $t=2.0\,\text{s}$. | Both alerts dispatched. Turn is NOT silenced by Decel cooldown. | Two distinct alerts: `ALT-MAN-SD` and `ALT-MAN-SRT`. |
| **LT-05** | **Modality Uncertainty Scaling (D-10)** | SD ($C=0.35$); Style ($C=0.95$). | SD has $C < 0.40 \implies \phi(C) = 0.0$. Style contributes $35.0 \times 1.0 = 35.0$. | Score $= 35.0$ ($\mathbf{CAUTION}$); uncertain maneuver contributes $0.0\,\text{pts}$. |
| **LT-06** | **Quiet Hysteresis Re-Arming ($2.50\,\text{s}$)** | SD ends $t=2.0\,\text{s}$. Re-excitation at $t=3.5\,\text{s}$ ($1.5\,\text{s}$ quiet); second re-excitation at $t=6.5\,\text{s}$ ($3.0\,\text{s}$ quiet). | At $t=3.5\,\text{s}$: same `event_id` (intra-incident resumption). At $t=6.5\,\text{s}$: new `event_id` (re-armed after $>2.50\,\text{s}$ quiet). | Event counter $= 1$ at $t=3.5\,\text{s}$; increments to $2$ at $t=6.5\,\text{s}$. |
| **LT-07** | **Buffer Rollover Independence** | SD at $t=1.0\,\text{s}$; advance telemetry to $t=12.0\,\text{s}$. | Raw sample at $t=1.0\,\text{s}$ evicted from $10\,\text{s}$ deque at $t=11.1\,\text{s}$. Incident lifecycle and decay proceed normally. | No crash, no premature eviction of decay channels. |
| **LT-08** | **Same-Class Double-Counting Cap** | SD 1 at $t=1.0\,\text{s}$; SD 2 at $t=2.5\,\text{s}$ (during SD 1 decay). | Same class ceiling applies: $\max(R_1, R_2) \le 50.0$. | Longitudinal braking risk never exceeds $50.0\,\text{pts}$. |
| **LT-09** | **Continuous Cruising Invariant** | 100 frames of flat highway cruising. | Gate reports `VALID_QUIESCENT`. Zero active channels. | Score $\equiv 0.00$; State $\equiv \mathbf{NORMAL}$. |
| **LT-10** | **Non-Linear Bonus Stacking** | Sustained Aggressive $+35.0$, Decel $+50.0$, Turn $+35.0$, Compound $+25.0$. | Raw sum $= 145.0\,\text{pts}$. Clamped to $100.0$. | Score $= 100.0$; State: $\mathbf{HIGH\_RISK}$. Alert: `CRITICAL`. |

---

## 10. Risks, Edge Cases & Unresolved Questions

1. **Jitter & High-Frequency Sampling Jumps:**
   If incoming telemetry packets exhibit timestamp jitter ($\Delta t < 50\,\text{ms}$), discrete sampling can create momentary artificial quiet gaps. The $2.50\,\text{s}$ quiet hysteresis successfully shields against false re-arming under jitter.
2. **Compound Maneuvers with Directional Reversal (Chicane / Slalom):**
   A rapid left-to-right swerve trial presents two successive turning classes (`SUDDEN_LEFT_TURN` followed immediately by `SUDDEN_RIGHT_TURN`). The architecture treats opposing turns as distinct orthogonal instances, correctly tracking both lateral pulses while capping total lateral contribution at $W_{\text{turn}} = 35.0\,\text{pts}$.
3. **Sensor Dropout During Active Decay:**
   If telemetry packets cease while a residual is actively decaying, the validator flags `STALE` after $3.0\,\text{seconds}$. Rule 1 triggers, transitioning the state to `INSUFFICIENT_DATA`. When telemetry resumes, the elapsed physical time is evaluated against $t_{\text{retire}}$, ensuring stale residuals do not linger artificially across outages.

---

## 11. Strict Governance Halt & Recommendation

- **Investigation Status:** **100% COMPLETE & VERIFIED (READ-ONLY)**.
- **Production Code Status:** Zero modifications made to production Python files during this investigation.
- **Recommendation:** Do not attempt premature integration of residual risk in isolation. Authorize remediation of **D-08 (Missing Style)**, **D-09 (Dissimilar Cooldown)**, and **D-10 (Confidence Scaling)** first, then land the integrated multi-channel residual decay pipeline in Checkpoint 5.

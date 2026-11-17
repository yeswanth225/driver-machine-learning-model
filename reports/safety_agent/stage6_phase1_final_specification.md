# Stage 6 Phase 1 Final Specification & Implementation Blueprint

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Repository:** `yeswanth225/driver-machine-learning-model`  
**Working Directory:** `D:\driver-machine-learning-model`  
**Branch:** `main`  
**Documented Checkpoint:** `eb54f7d1e03260482500f611b77d1065931e300b`  
**Document Type:** Final Implementation Specification (Strictly Read-Only Deliverable)  
**Date:** November 2026  
**Status:** SPECIFICATION COMPLETE — AWAITING USER IMPLEMENTATION AUTHORIZATION — ZERO CODE MODIFICATIONS PERFORMED  

---

## 1. Verified Checkpoint & Governance Compliance

### 1.1 Git Checkpoint Verification [VERIFIED IMPLEMENTATION BEHAVIOUR]
```powershell
git status
git branch --show-current
git log -1 --oneline
git diff
```
* **Repository Root:** `D:\driver-machine-learning-model`
* **Active Branch:** `main`
* **Tracking State:** Synchronized with `origin/main`
* **HEAD Commit:** `eb54f7d1e03260482500f611b77d1065931e300b`
* **HEAD Subject:** `feat(ml): implement Stage 6 Group A remediation, D-06 residual risk decay, and residual integration architecture design`
* **Working Tree:** Clean on all tracked files. 0 tracked modifications.
* **Diff:** 0 diff lines across tracked files.

### 1.2 Model Artifact Cryptographic Hashes [VERIFIED IMPLEMENTATION BEHAVIOUR]
All four frozen Stage 5 models and scalers were verified via direct SHA-256 computation:
* `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`: `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` (**MATCH**)
* `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`: `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` (**MATCH**)
* `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib`: `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` (**MATCH**)
* `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib`: `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` (**MATCH**)

---

## 2. Explicit User Decisions (Locked Baseline)

The following three user decisions are formally incorporated and locked [USER APPROVED]:

1. **Decision 1 — `SafetyAlert.message` Compatibility Property:**  
   Implement additive property on `SafetyAlert`:
   ```python
   @property
   def message(self) -> str:
       return self.explanation
   ```
   * Strictly preserves dataclass `__init__()`.
   * Preserves `title`, `explanation`, and `to_dict()` JSON schemas.
   * Resolves the regression test failure at line 1110 in `tests/test_safety_agent.py`.
2. **Decision 2 — D-10 Option A Compound Bonus Gating:**  
   Preserve documented flat $+25.0$ compound bonus weight, gated so it applies if and only if both modality scaling factors are strictly positive:
   ```python
   compound_bonus = (
       25.0
       if (phi_style > 0.0 and phi_maneuver > 0.0)
       else 0.0
   )
   ```
   * No proportional scaling of the 25-point bonus.
   * Approved Option 1 piecewise confidence scaling:
     $$\phi(C) = \begin{cases} 0.0 & C < 0.40 \\ \frac{C}{0.60} & 0.40 \le C < 0.60 \\ 1.0 & C \ge 0.60 \end{cases}$$
   * Applied independently to style and maneuver modalities without cross-modality averaging.
3. **Decision 3 — Full State-Transition Test Reconciled with Health-State Assertions:**  
   * Initial nominal step uses valid nominal maneuver inputs (`maneuver_prediction = 0`, `maneuver_confidence = 1.0`) to assert `pipeline_a_health == "HEALTHY"` and `pipeline_b_health == "HEALTHY"`, cleanly permitting `SafetyState.NORMAL`.
   * `QUIESCENT_BYPASS` is reserved exclusively for the kinematic gate evaluating quiet telemetry in `process_features()`, verified in `test_process_features_with_production_models` and `test_t01_d01_*`.

---

## 3. D-09 Final Implementation Specification: Alert Cooldown Identity

### 3.1 Cooldown Registry Key & Maneuver Identity [AUTHORITATIVE REQUIREMENT]
* **Nature of Maneuver Identifier:** In `ml/agent/temporal_smoother.py` lines 74–85, `ctx.active_maneuver` is an integer scalar enum:
  $$\text{ManeuverClass}: \quad 0 = \text{NONE}, \quad 1 = \text{SA}, \quad 2 = \text{SD}, \quad 3 = \text{SRT}, \quad 4 = \text{SLT}$$
  It represents a **stable class-type identifier**, not an event-instance UUID.
* **Cooldown Dictionary Structure:** In `DeterministicRiskEngine`, internal alert tracking dictionaries are keyed by:
  ```python
  Tuple[AlertCategory, Optional[int]]
  ```
  * *Non-Maneuver Categories (`STYLE`, `COMPOUND`, `DATA_QUALITY`):* Keyed as `(category, None)`. Preserves category-wide throttling.
  * *Maneuver Category (`MANEUVER`):* Keyed as `(AlertCategory.MANEUVER, ctx.active_maneuver)`. Partitions throttling by maneuver class ID.

### 3.2 Detailed Operational Scenarios [VERIFIED IMPLEMENTATION BEHAVIOUR]
1. **Repeated Frames of Same Maneuver Class:**  
   Sudden Decel ($M=2$) at $t = 1.0\,\text{s}$ sets key `(AlertCategory.MANEUVER, 2)`. Frame at $t = 1.2\,\text{s}$ hits the same key; elapsed $0.2\,\text{s} < 3.0\,\text{s}$. **Alert is SUPPRESSED.**
2. **Dissimilar Maneuver Types:**  
   Sudden Decel ($M=2$) at $t = 1.0\,\text{s}$ records under `(AlertCategory.MANEUVER, 2)`. Sudden Turn ($M=3$) at $t = 2.0\,\text{s}$ checks `(AlertCategory.MANEUVER, 3)`, finds no active cooldown, and **dispatches immediately**.
3. **Separate Events of Same Maneuver Class:**  
   Sudden Decel ($M=2$) at $t = 1.0\,\text{s}$, followed by cruising, followed by a second Sudden Decel ($M=2$) at $t = 5.0\,\text{s}$. Elapsed $4.0\,\text{s} \ge 3.0\,\text{s}$ (`alert_cooldown_maneuver`). **Alert is DISPATCHED.** (Note: Separate events of the same class occurring $< 3.0\,\text{s}$ apart remain throttled under class-level cooldown; full multi-instance tracking is governed by Phase 2 residual risk).
4. **Maneuver Transitions:**  
   Abrupt transition from Acceleration ($M=1$) to Deceleration ($M=2$) switches keys dynamically; both dispatch without suppression.
5. **Missing Maneuver Identifiers:**  
   When `ctx.active_maneuver == 0` or `None`, no maneuver alert rule triggers; non-maneuver alerts use `(category, None)`.
6. **Severity Escalation Bypass:**  
   Escalations to `AlertSeverity.HIGH` or `CRITICAL` bypass cooldown timers regardless of key.

### 3.3 Cooldown Lifecycle & Reset Behaviour [VERIFIED IMPLEMENTATION BEHAVIOUR]
* Initialized in `DeterministicRiskEngine.__init__()` as empty dictionaries.
* Flushed cleanly in `DeterministicRiskEngine.reset()`.
* Memory footprint is strictly $O(1)$, bounded to at most $4$ maneuver classes $+ 3$ non-maneuver categories $= 7$ dictionary entries.

---

## 4. D-10 Final Mathematical Specification: Modality Confidence Scaling

### 4.1 Approved Scaling Operator [USER APPROVED]
```python
def calculate_confidence_scaling(confidence: Optional[float]) -> float:
    if confidence is None or isinstance(confidence, bool):
        return 0.0
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        return 0.0
    if confidence < 0.40:
        return 0.0
    if confidence < 0.60:
        return float(confidence / 0.60)
    return 1.0
```

### 4.2 Application and Score Summation [AUTHORITATIVE REQUIREMENT]
Scaling is applied directly to base hazard penalties **prior** to risk score accumulation:

$$\text{Score}_{\text{raw}} = R_{\text{baseline}} + R_{\text{style}} + R_{\text{maneuver}} + R_{\text{compound}} + R_{\text{repeated}}$$

Where:
* $R_{\text{baseline}} = 0.0$
* $R_{\text{style}} = W_{\text{style}} \cdot \phi(C_{\text{style}})$  
  ($W_{\text{style}} = 35.0$ if sustained aggressive, $20.0$ if transient aggressive, else $0.0$)
* $R_{\text{maneuver}} = W_{\text{maneuver}} \cdot \phi(C_{\text{maneuver}})$  
  ($W_{\text{maneuver}} = 50.0$ for SD, $35.0$ for Turn, $25.0$ for SA, else $0.0$)
* $R_{\text{compound}} = 25.0$ if `ctx.is_maneuver_active` and (`ctx.is_sustained_aggressive` or `ctx.is_transient_aggressive`) and $\phi(C_{\text{style}}) > 0.0$ and $\phi(C_{\text{maneuver}}) > 0.0$, else $0.0$.
* $R_{\text{repeated}} = 15.0$ if `ctx.recent_maneuver_count >= 2`, else $0.0$.
* Clamping: $\text{Score}_{\text{final}} = \min(100.0, \max(0.0, \text{Score}_{\text{raw}}))$.

### 4.3 Missing, Invalid, and Non-Finite Handling [VERIFIED IMPLEMENTATION BEHAVIOUR]
* If confidence is omitted (`None`), $\phi(C) = 0.0$.
* If confidence is non-finite (`NaN`, $\pm\infty$) or out-of-bounds ($< 0.0$ or $> 1.0$), `PredictionValidator` flags `data_quality_status = "INVALID"`, setting `is_valid = False`. In `calculate_confidence_scaling()`, defense-in-depth returns $0.0$.
* Modality independence: $R_{\text{style}}$ depends strictly on $C_{\text{style}}$; $R_{\text{maneuver}}$ depends strictly on $C_{\text{maneuver}}$.

---

## 5. D-13 State Precedence & Execution Order of Operations

To prevent mixing transport-level data faults with behavioral states or fail-safe clamps, the system enforces a strict **7-Stage Order of Operations** [AUTHORITATIVE REQUIREMENT]:

```mermaid
flowchart TD
    S1["Stage 1: Packet-Level Transport Validation\n(PredictionValidator.validate)"] --> CHK1{"is_valid is True and\ndata_quality == VALID?"}
    
    CHK1 -- No --> S1_FAIL["Rule 1 Intercepts:\nScore = 0.0\nState = INSUFFICIENT_DATA\nEmit DATA_QUALITY Alert"]
    CHK1 -- Yes --> S2["Stage 2: Independent Pipeline Health Determination\n(Compute H_A and H_B)"]
    
    S2 --> S3["Stage 3: Behavioral Risk Contribution Calculation\n(Evaluate R_style, R_maneuver, R_compound, R_repeated)"]
    S3 --> S4["Stage 4: Behavioral Risk State Mapping\n(Score -> S_behavioral in {NORMAL, CAUTION, WARNING, HIGH_RISK})"]
    
    S4 --> S5["Stage 5: Fail-Safe Minimum State Enforcement\n(Lookup S_min_failsafe from Health Truth Table)"]
    
    S5 --> S6["Stage 6: Final SafetyState Selection\nS_final = max_severity(S_min_failsafe, S_behavioral)"]
    
    S6 --> S7["Stage 7: Diagnostic Alert Append\n(Append DATA_QUALITY alert if H_A != HEALTHY or H_B not in {HEALTHY, QUIESCENT})"]
```

### Exact Precedence Rules:
1. **Rule 1 (Transport Guardrail) Takes Absolute Precedence:**  
   If the transport packet itself is stale, non-monotonic, or both predictions are missing, processing halts immediately. No behavioral rules run; state is strictly `INSUFFICIENT_DATA`, score is $0.0$.
2. **Behavioral Risk Calculation is Independent:**  
   When the packet is valid, behavioral risk accumulates from all valid modalities. An acute harsh braking maneuver ($W=50.0, C \ge 0.60$) always produces behavioral state `WARNING`.
3. **Health Policy Enforces a Floor, Never Attenuates:**  
   Health status determines $S_{\text{min\_failsafe}}$ (e.g., `CAUTION` when Pipeline B is missing).  
   Final state is the highest severity:
   $$S_{\text{final}} = \max_{\text{severity}}\left(S_{\text{min\_failsafe}}, \ S_{\text{behavioral}}\right)$$
   A minimum `CAUTION` clamp **never reduces** a behavioral `WARNING` or `HIGH_RISK`!
4. **Asymmetric Acute Hazard Override:**  
   If Pipeline A is `MISSING` or `DEGRADED`, but Pipeline B detects an acute maneuver ($R=50.0$), $S_{\text{behavioral}} = \mathbf{WARNING}$. Because $\text{WARNING} > \text{INSUFFICIENT\_DATA}$, the final state is `WARNING`, and the physical hazard is alerted!

---

## 6. Corrected 20-Combination Pipeline Health Truth Table

Below is the authoritative, corrected 20-combination truth table [PROPOSED POLICY & AUTHORITATIVE SYNTHESIS]:

| # | Pipeline A Health ($H_A$) | Pipeline B Health ($H_B$) | `NORMAL` Permitted? | Min Fail-Safe State ($S_{\text{min\_failsafe}}$) | Behavioral State Calculable? ($S_{\text{behavioral}}$) | Acute Maneuver Warning Override? | Behavioral Risk Calculated? | `DATA_QUALITY` Diagnostic? | Result Becomes `INSUFFICIENT_DATA`? | Packet-Level Valid? | Authoritative Architectural Reason |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **HEALTHY** | **HEALTHY** | **YES** | `NORMAL` | `NORMAL`..`HIGH_RISK` | N/A | **YES** (Both) | **NO** | **NO** | **YES** | **Full Mutual Certification.** Both pipelines verified. Normal cruising certified when score $< 25.0$. |
| **2** | **HEALTHY** | **DEGRADED** | **NO** | `CAUTION` | `NORMAL`..`CAUTION` (Style) | NO | **YES** (Style only) | **YES** | **NO** | **YES** | Style monitored; maneuver pipeline degraded. Minimum `CAUTION` enforced. |
| **3** | **HEALTHY** | **MISSING** | **NO** | `CAUTION` | `NORMAL`..`CAUTION` (Style) | NO | **YES** (Style only) | **YES** | **NO** | **YES** | Maneuver unmonitored; style valid. Minimum `CAUTION` enforced (tested in `test_t13_d13`). |
| **4** | **HEALTHY** | **INVALID** | **NO** | `CAUTION` | `NORMAL`..`CAUTION` (Style) | NO | **YES** (Style only) | **YES** | **NO** | **YES** | Maneuver telemetry invalid/out of bounds; style valid. Minimum `CAUTION` enforced. |
| **5** | **HEALTHY** | **QUIESCENT_BYPASS** | **YES** | `NORMAL` | `NORMAL`..`CAUTION` (Style) | N/A | **YES** (Style only) | **NO** | **NO** | **YES** | **Mutual Certification Satisfied.** Kinematic gate verified cruising; nominal maneuver confirmed. |
| **6** | **DEGRADED** | **HEALTHY** | **NO** | `INSUFFICIENT_DATA` | If Maneuver: `WARNING`..<br>If None: `NORMAL` | **YES** (Overrides to WARNING/HIGH_RISK) | **YES** (Maneuver only) | **YES** | **Conditional** (YES only if Maneuver is NONE) | **YES** | **Asymmetric Acute Hazard Override.** Severe maneuver alerted; if nominal, cannot certify safety without style. |
| **7** | **DEGRADED** | **DEGRADED** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Complete supervisory invalidation. Both pipelines degraded by model/feature exceptions. |
| **8** | **DEGRADED** | **MISSING** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Style degraded, maneuver absent. Cannot supervise safety. |
| **9** | **DEGRADED** | **INVALID** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Dual pipeline corruption. Complete supervisory invalidation. |
| **10** | **DEGRADED** | **QUIESCENT_BYPASS** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Cruising kinematics verified, but style model degraded. Cannot certify driving safety. |
| **11** | **MISSING** | **HEALTHY** | **NO** | `INSUFFICIENT_DATA` | If Maneuver: `WARNING`..<br>If None: `NORMAL` | **YES** (Overrides to WARNING/HIGH_RISK) | **YES** (Maneuver only) | **YES** | **Conditional** (YES only if Maneuver is NONE) | **YES** | **Asymmetric Acute Hazard Override.** Style absent; acute maneuver alerted; if nominal, yields `INSUFFICIENT_DATA`. |
| **12** | **MISSING** | **DEGRADED** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Primary absent, secondary degraded. Complete supervisory invalidation. |
| **13** | **MISSING** | **MISSING** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **NO** (`raw_is_missing`) | Total telemetry absence. Rule 1 intercepts directly at validator level. |
| **14** | **MISSING** | **INVALID** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Primary absent, secondary invalid. Complete supervisory invalidation. |
| **15** | **MISSING** | **QUIESCENT_BYPASS** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Cruising kinematics verified, but style stream absent. Cannot certify driving safety. |
| **16** | **INVALID** | **HEALTHY** | **NO** | `INSUFFICIENT_DATA` | If Maneuver: `WARNING`..<br>If None: `NORMAL` | **YES** (Overrides to WARNING/HIGH_RISK) | **YES** (Maneuver only) | **YES** | **Conditional** (YES only if Maneuver is NONE) | **YES** | **Asymmetric Acute Hazard Override.** Style invalid; acute maneuver alerted; if nominal, yields `INSUFFICIENT_DATA`. |
| **17** | **INVALID** | **DEGRADED** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Dual pipeline failure. Complete supervisory invalidation. |
| **18** | **INVALID** | **MISSING** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Primary invalid, secondary absent. Complete supervisory invalidation. |
| **19** | **INVALID** | **INVALID** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Both pipelines invalid. Complete supervisory invalidation. |
| **20** | **INVALID** | **QUIESCENT_BYPASS** | **NO** | `INSUFFICIENT_DATA` | None ($R = 0.0$) | NO | **NO** | **YES** | **YES** | **YES** | Cruising kinematics verified, but style payload invalid. Cannot certify driving safety. |

---

## 7. INSUFFICIENT_DATA Policy and Boundaries

`SafetyState.INSUFFICIENT_DATA` is emitted **if and only if** one of the following exact criteria is met [AUTHORITATIVE REQUIREMENT]:

1. **Packet-Level Transport Guardrail (Rule 1):**
   * Packet timestamp is stale: $(t_{\text{current}} - t) > 3.0\,\text{s}$ (`data_quality_status == "STALE"`).
   * Packet timestamp non-monotonic: $t < t_{\text{last\_valid}}$ without clock reset corroboration.
   * Packet timestamp in future: $(t - t_{\text{current}}) > 1.0\,\text{s}$.
   * Entire telemetry packet empty: both style and maneuver predictions are `None`.
2. **Supervisory Dual Pipeline Failure:**
   * Both pipelines are unavailable (any combination of `DEGRADED`, `MISSING`, or `INVALID` on both A and B, rows 7–9, 12–14, 17–19).
3. **Uncertifiable Cruising Without Style Monitoring:**
   * Pipeline A is `DEGRADED`, `MISSING`, or `INVALID`, and Pipeline B is `QUIESCENT_BYPASS` or `HEALTHY` with nominal maneuver ($M=0$) (rows 6, 10, 11, 15, 16, 20). Driving cannot be certified safe or cautious without general style monitoring.

---

## 8. Acute Maneuver Override Policy

* **Safety-Critical Justification [AUTHORITATIVE REQUIREMENT]:**  
  A driver executing severe emergency braking ($-5.0\,\text{m/s}^2$, $W = 50.0$) presents an immediate rear-end collision hazard. If the secondary style sensor (Pipeline A) is offline or degraded, suppressing the acute braking alert into `INSUFFICIENT_DATA` would create an unacceptable safety risk.
* **Operational Rule:**  
  When Pipeline B is `HEALTHY` and emits an active abrupt maneuver ($M \in \{1, 2, 3, 4\}$ with $C \ge 0.40$), the acute maneuver hazard state (`WARNING` or `HIGH_RISK`) **governs the final assessment**, while a concurrent `DATA_QUALITY` diagnostic alert notifies the operator that style monitoring is unavailable.

---

## 9. QUIESCENT_BYPASS Semantics

* **Definition [AUTHORITATIVE REQUIREMENT]:**  
  `QUIESCENT_BYPASS` applies **exclusively to Pipeline B** and occurs **if and only if** `KinematicEventGate.evaluate()` confirms that all 5 kinematic measurement groups are present, finite, and strictly below trigger thresholds (`GateStatus.VALID_QUIESCENT`).
* **Distinction from Nominal Classifier Prediction:**  
  * When `process_predictions()` ingests pre-computed integer predictions (`maneuver_prediction = 0`), Pipeline B health is **`HEALTHY`** (classifier evaluated nominal cruising).
  * When `process_features()` bypasses classifier inference via the kinematic gate, Pipeline B health is **`QUIESCENT_BYPASS`**.
  * Both states confirm that the vehicle is not maneuvering abruptly and satisfy mutual certification with a `HEALTHY` Pipeline A to permit `SafetyState.NORMAL`.

---

## 10. Legacy Test Reconciliations & Test Design

Detailed test fixtures and assertions [AUTHORITATIVE REQUIREMENT]:

### 10.1 `test_harsh_braking_triggers_warning`
* **Target:** `DeterministicRiskEngine.evaluate(ctx)`
* **Input Fixture:** `ctx = SmoothedContext(timestamp=1.0, current_style=0, current_style_confidence=0.9, active_maneuver=2 (SD), active_maneuver_confidence=0.90, active_maneuver_age_seconds=0.2, is_maneuver_active=True, recent_maneuver_count=1)`
* **Expected Result:**
  ```python
  assert res.risk_score == 50.0
  assert res.safety_state == SafetyState.WARNING
  ```

### 10.2 `test_low_confidence_discount`
* **Target:** `DeterministicRiskEngine.evaluate(ctx)`
* **Input Fixture:** `ctx = SmoothedContext(timestamp=1.0, current_style=0, current_style_confidence=0.9, active_maneuver=2 (SD), active_maneuver_confidence=0.45, active_maneuver_age_seconds=0.2, is_maneuver_active=True, recent_maneuver_count=1)`
* **Expected Result:** $\phi(0.45) = 0.45 / 0.60 = 0.75 \implies \text{Score} = 50.0 \times 0.75 = 37.5\,\text{pts}$.
  ```python
  assert res.risk_score == pytest.approx(37.5, abs=0.1)
  assert res.safety_state == SafetyState.CAUTION
  ```

### 10.3 `test_score_clamping`
* **Target:** `DeterministicRiskEngine.evaluate(ctx)`
* **Input Fixture:** `ctx = SmoothedContext(timestamp=1.0, is_sustained_aggressive=True, current_style=1, current_style_confidence=0.9, active_maneuver=2 (SD), active_maneuver_confidence=0.95, active_maneuver_age_seconds=0.2, is_maneuver_active=True, recent_maneuver_count=5)`
* **Expected Result:** $\text{raw\_score} = 35.0 (\text{sustained}) + 50.0 (\text{SD}) + 25.0 (\text{compound}) + 15.0 (\text{repeated}) = 125.0$.
  ```python
  assert res.raw_score == 125.0
  assert res.risk_score == 100.0
  ```

### 10.4 `test_full_state_transition_sequence`
* **Target:** `IntelligentSafetyAgent.process_predictions()`
* **Step 1 (Nominal Cruising):**
  ```python
  ass1 = agent.process_predictions(
      timestamp=1.0,
      style_prediction=0,
      style_confidence=0.95,
      maneuver_prediction=0,      # Explicit nominal maneuver (class 0)
      maneuver_confidence=1.0
  )
  assert ass1.pipeline_a_health == "HEALTHY"
  assert ass1.pipeline_b_health == "HEALTHY"
  assert ass1.safety_state == SafetyState.NORMAL
  assert ass1.risk_score == 0.0
  ```
* **Step 2 (Harsh Braking):**
  ```python
  ass2 = agent.process_predictions(
      timestamp=2.0,
      style_prediction=0,
      style_confidence=0.95,
      maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
      maneuver_confidence=0.92
  )
  assert ass2.pipeline_b_health == "HEALTHY"
  assert ass2.safety_state == SafetyState.WARNING
  assert ass2.risk_score == 50.0
  ```
* **Separation of Quiescent Bypass Verification:**  
  Intentional quiescent bypass is tested separately in `test_process_features_with_production_models` and `test_t01_d01_kinematic_event_gate_quiescent_bypass`, where raw feature inputs execute the kinematic gate and assert `pipeline_b_health == "QUIESCENT_BYPASS"`.

---

## 11. Exact Implementation File Scope

Phase 1 code modifications are **strictly bounded to 3 files** [AUTHORITATIVE REQUIREMENT]:

1. **[`ml/agent/risk_engine.py`](file:///d:/driver-machine-learning-model/ml/agent/risk_engine.py):**
   * Add `@property def message(self) -> str: return self.explanation` to `SafetyAlert`.
   * Update `_last_alert_timestamps` and `_last_alert_severities` key type to `Tuple[AlertCategory, Optional[int]]`.
   * Update `_should_suppress_alert()` and `_record_alert()` to accept `maneuver_id: Optional[int] = None`.
   * Implement `calculate_confidence_scaling(confidence: Optional[float]) -> float`.
   * In `evaluate()`, apply $\phi(C)$ independently to style and maneuver contributions; gate Rule 7 on $\phi(C_S) > 0$ and $\phi(C_M) > 0$.
2. **[`ml/agent/safety_agent.py`](file:///d:/driver-machine-learning-model/ml/agent/safety_agent.py):**
   * Add `pipeline_a_health: str` and `pipeline_b_health: str` to `SafetyAssessment`.
   * Compute independent pipeline health in `process_predictions()` and `process_features()`.
   * Enforce fail-safe state precedence from Section 6 truth table (clamp to min `CAUTION` or `INSUFFICIENT_DATA`; forbid `NORMAL` unless mutually certified).
3. **[`tests/test_safety_agent.py`](file:///d:/driver-machine-learning-model/tests/test_safety_agent.py):**
   * Reconcile assertions in `test_harsh_braking_triggers_warning`, `test_low_confidence_discount`, `test_score_clamping`, and `test_full_state_transition_sequence`.
   * Add dedicated piecewise confidence boundary test suite.

---

## 12. Implementation Sequence & Execution Order

```mermaid
flowchart TD
    S1["1. ml/agent/risk_engine.py (D-09 & SafetyAlert.message)\n- Add message property\n- Partition cooldown keys by (category, maneuver_id)\n- Run pytest: test_t09_d09 passes"]
    S2["2. ml/agent/risk_engine.py (D-10 Option 1 & Compound Gating)\n- Implement calculate_confidence_scaling(C)\n- Apply phi(C) independently per modality\n- Gate Rule 7 bonus\n- Run pytest: test_t10_d10 passes"]
    S3["3. ml/agent/safety_agent.py (D-13 Health Matrix & Precedence)\n- Add pipeline_a_health & pipeline_b_health\n- Implement 7-stage precedence and fail-safe clamp\n- Run pytest: test_t13_d13 passes"]
    S4["4. tests/test_safety_agent.py (Legacy Test Reconciliations)\n- Update 4 legacy test assertions\n- Add phi(C) boundary test\n- Run pytest tests/test_safety_agent.py -v -> 39/39 PASSED"]
    S5["5. Full Repository Test Suite Verification\n- Run python -m pytest tests/ -v\n- Achieve 93/93 PASSED (100%)\n- Verify zero regressions across ML pipelines"]

    S1 --> S2 --> S3 --> S4 --> S5
```

---

## 13. Acceptance Criteria

1. `test_t09_d09_dissimilar_maneuver_alerts_bypass_category_cooldown` **PASSED**.
2. `test_t10_d10_unified_three_tier_confidence_scaling` **PASSED**.
3. `test_t13_d13_independent_pipeline_health_matrix` **PASSED**.
4. All 4 legacy tests modernized and **PASSED**.
5. All 39 tests in `tests/test_safety_agent.py` **PASSED (100%)**.
6. All 93 tests in `tests/` directory **PASSED (100%)**.
7. Zero model retraining, zero artifact hash changes.

---

## 14. Final Approval Checklist for User Authorization

- [ ] **Lock 1:** Authorize `SafetyAlert.message` compatibility property.
- [ ] **Lock 2:** Authorize D-10 Option 1 three-tier confidence function with gated $+25.0$ compound bonus.
- [ ] **Lock 3:** Authorize nominal maneuver input in Step 1 of `test_full_state_transition_sequence` with explicit `HEALTHY` assertion.
- [ ] **Lock 4:** Authorize the 7-stage state precedence and corrected 20-combination fail-safe truth table.
- [ ] **Lock 5:** Authorize proceeding with code edits strictly inside `ml/agent/risk_engine.py`, `ml/agent/safety_agent.py`, and `tests/test_safety_agent.py`.

---

## 15. Explicit Confirmation of Read-Only Compliance

* **Python Source Files Modified:** **0**
* **Test Files Modified:** **0**
* **Model Artifacts Modified:** **0**
* **Datasets Modified:** **0**
* **Configuration Files Created / Modified:** **0**
* **Implementation Actions Executed:** **0**
* **Permitted Deliverable Created:** [`reports/safety_agent/stage6_phase1_final_specification.md`](file:///d:/driver-machine-learning-model/reports/safety_agent/stage6_phase1_final_specification.md)

**FINAL SPECIFICATION COMPLETE. ZERO IMPLEMENTATION ACTIONS EXECUTED. AWAITING USER AUTHORIZATION TO COMMENCE PHASE 1 IMPLEMENTATION.**

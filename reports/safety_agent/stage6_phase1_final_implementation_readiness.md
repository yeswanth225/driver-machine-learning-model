# Stage 6 Phase 1 Final Specification Consolidation & Implementation Readiness Audit

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Repository:** `yeswanth225/driver-machine-learning-model`  
**Working Directory:** `D:\driver-machine-learning-model`  
**Active Branch:** `main`  
**Current HEAD Commit:** `eb54f7d1e03260482500f611b77d1065931e300b`  
**Document Type:** Final Implementation Readiness Audit & Authoritative Specification (Strictly Read-Only Deliverable)  
**Date:** November 2026  
**Status:** SPECIFICATION CONSOLIDATED — ALL USER DECISIONS LOCKED — ZERO RESIDUAL AMBIGUITIES — READY FOR USER IMPLEMENTATION APPROVAL  

---

## 1. Executive Summary & Verification Checkpoint

### 1.1 Repository Checkpoint & Remote Synchronization Audit [VERIFIED IMPLEMENTATION BEHAVIOUR]
```powershell
git status
git branch --show-current
git log -1 --oneline
git fetch origin
git log -1 --oneline origin/main
```
* **Local Working Directory:** `D:\driver-machine-learning-model`
* **Active Branch:** `main`
* **Remote Tracking:** `origin/main` (Synchronized, up-to-date)
* **Local HEAD Commit:** `eb54f7d1e03260482500f611b77d1065931e300b`
* **Remote HEAD Commit:** `eb54f7d1e03260482500f611b77d1065931e300b`
* **Commit Subject:** `feat(ml): implement Stage 6 Group A remediation, D-06 residual risk decay, and residual integration architecture design`
* **Tracked Modifications:** **0** (Working tree is strictly clean on all tracked files)
* **Untracked Reports (Preserved Historical Documentation):**
  - `reports/safety_agent/stage6_phase1_remediation_plan.md`
  - `reports/safety_agent/stage6_phase1_specification_reconciliation.md`
  - `reports/safety_agent/stage6_phase1_specification_closure_addendum.md`
  - `reports/safety_agent/stage6_phase1_final_specification.md`
  - `reports/safety_agent/stage6_phase1_preimplementation_audit.md`
  - `reports/safety_agent/stage6_phase1_final_policy_clarification.md`
  - `reports/safety_agent/stage6_phase1_final_implementation_readiness.md` (this authoritative deliverable)
* **Master Documentation Audit:**  
  `MASTER_PROJECT_DOCUMENTATION.md` is an external/historical tracking document not present in local git tree. All authoritative Stage 5 artifact hashes and architectural invariants are verified against `reports/model_training/final_model_selection_report.md`, `reports/safety_agent/stage6_remediation_execution_report.md`, and direct on-disk cryptographic computation.

### 1.2 Cryptographic Verification of Stage 5 Model Artifacts [VERIFIED IMPLEMENTATION BEHAVIOUR]
All four frozen Stage 5 joblib artifacts were re-verified byte-for-byte via SHA-256:
* `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`:  
  `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` (**MATCH**)
* `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`:  
  `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` (**MATCH**)
* `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib`:  
  `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` (**MATCH**)
* `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib`:  
  `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` (**MATCH**)

---

## 2. All Locked User Decisions (Authoritative Baseline)

The following decisions are **FINAL, APPROVED, and LOCKED**. They govern the consolidated implementation specification:

### 2.1 D-09 — Alert Cooldown Partitioning & SafetyAlert Compatibility [USER APPROVED]
1. **Cooldown Registry Keying:**  
   In `DeterministicRiskEngine`, internal alert timestamps and severities are keyed by:
   ```python
   Tuple[AlertCategory, Optional[int]]
   ```
   - **Non-Maneuver Categories (`STYLE`, `COMPOUND`, `DATA_QUALITY`):** Keyed as `(category, None)`. Preserves category-wide throttling.
   - **Maneuver Category (`MANEUVER`):** Keyed as `(AlertCategory.MANEUVER, ctx.active_maneuver)`. Partitions alert throttling by maneuver class ID ($1=\text{SA}, 2=\text{SD}, 3=\text{SRT}, 4=\text{SLT}$).
   - **Behavior:** Repeated frames of the same maneuver class within 3.0s are throttled. Dissimilar maneuver classes bypass cooldown and dispatch immediately upon onset.
2. **`SafetyAlert.message` Compatibility Property:**  
   Additive property on `SafetyAlert`:
   ```python
   @property
   def message(self) -> str:
       return self.explanation
   ```
   - Strictly preserves dataclass `__init__()` signature.
   - Preserves `title`, `explanation`, and `to_dict()` JSON schemas.
   - Resolves existing regression test failure at line 1110 in `tests/test_safety_agent.py`.

### 2.2 D-10 — Independent Confidence Scaling & Gated Compound Bonus [USER APPROVED]
1. **Approved Piecewise Confidence Scaling Operator (Option 1):**
   $$\phi(C) = \begin{cases} 0.0 & C < 0.40 \\ \frac{C}{0.60} & 0.40 \le C < 0.60 \\ 1.0 & C \ge 0.60 \end{cases}$$
   - Below $0.40$: Deadband. Modality risk penalty is $0.0$.
   - $[0.40, 0.60)$: Linear ramp from $0.6667$ to $1.0000$.
   - $\ge 0.60$: Saturated confidence. Full hazard weight applied.
   - Missing (`None`), non-numeric, or non-finite (`NaN`, $\pm\infty$) confidence strictly yields $\phi(C) = 0.0$.
2. **Strict Modality Independence:**  
   - $R_{\text{style}} = W_{\text{style}} \cdot \phi(C_{\text{style}})$
   - $R_{\text{maneuver}} = W_{\text{maneuver}} \cdot \phi(C_{\text{maneuver}})$
   - Confidence is **never averaged** across modalities.
3. **Approved Compound Hazard Bonus Policy (Option A):**  
   $$R_{\text{compound}} = \begin{cases} 25.0 & \text{if } \text{is\_compound\_hazard\_active} \land \phi(C_{\text{style}}) > 0.0 \land \phi(C_{\text{maneuver}}) > 0.0 \\ 0.0 & \text{otherwise} \end{cases}$$
   - The original compound hazard condition (`ctx.is_maneuver_active and (ctx.is_sustained_aggressive or ctx.is_transient_aggressive)`) is **strictly preserved**.
   - Bonus is **flat $+25.0$ points**. It is **never proportionally scaled**.
4. **Elimination of Global Post-Accumulation Multiplier:**  
   The legacy scaling `final_score = raw_score * confidence_factor` is **completely deleted**.

### 2.3 D-13 — Independent Pipeline Health Architecture [USER APPROVED]
1. **Canonical Health States:**  
   Retain exactly 5 canonical states: `HEALTHY`, `DEGRADED`, `MISSING`, `INVALID`, `QUIESCENT_BYPASS`.
2. **Separation from Packet Staleness:**  
   Packet-level staleness (`data_quality_status == "STALE"`) remains a transport guardrail attribute in `PredictionValidator`, distinct from individual pipeline health.
3. **D-13 Policy Decision 1 (LOCKED: Option 1A — Active Maneuver Override):**  
   When Pipeline A is unavailable (`DEGRADED`, `MISSING`, or `INVALID`) and Pipeline B is `HEALTHY` with an active abrupt maneuver ($M \in \{1, 2, 3, 4\}$):
   - The system preserves the **ordinary calculated behavioral state** $S_{\text{behavioral}} = \text{MapScore}(W_{\text{maneuver}} \cdot \phi(C_M))$.
   - Does **not** artificially force a minimum `WARNING` state.
   - Sudden Deceleration ($W=50.0$):
     - $C = 0.40 \implies \phi(0.40) = 0.6667 \implies 33.33\,\text{pts} \implies \mathbf{CAUTION}$
     - $C = 0.50 \implies \phi(0.50) = 0.8333 \implies 41.67\,\text{pts} \implies \mathbf{CAUTION}$
     - $C = 0.60 \implies \phi(0.60) = 1.0000 \implies 50.00\,\text{pts} \implies \mathbf{WARNING}$
   - Sudden Turn ($W=35.0$): $C \in [0.40, 1.0] \implies 23.33\dots 35.0\,\text{pts} \implies \mathbf{CAUTION}$.
   - Sudden Acceleration ($W=25.0$): $C \in [0.40, 1.0] \implies 16.67\dots 25.0\,\text{pts} \implies \mathbf{CAUTION}$.
   - Appends a `DATA_QUALITY` diagnostic alert identifying that Pipeline A is unavailable.
4. **D-13 Policy Decision 2 (LOCKED: Option 2A — Nominal Cruising Demotion):**  
   When Pipeline A is unavailable (`DEGRADED`, `MISSING`, or `INVALID`) and Pipeline B reports nominal cruising ($M=0$) or `QUIESCENT_BYPASS`:
   - Returns **`SafetyState.INSUFFICIENT_DATA`**.
   - Appends a `DATA_QUALITY` diagnostic alert.
   - Does **not** permit `NORMAL`. Cruising safety cannot be certified without driving style monitoring.
   - Does **not** treat intentional `QUIESCENT_BYPASS` as a Pipeline B failure.

---

## 3. Complete 20-Combination D-13 Truth Table

Every single combination of Pipeline A ($4$ states) and Pipeline B ($5$ states) is explicitly specified:

| # | Pipeline A Health ($H_A$) | Pipeline B Health ($H_B$) | Operationally Valid? | Behavioral Risk Permitted? | `NORMAL` Permitted? | Final Safety-State Policy | `DATA_QUALITY` Alert Required? | Expected Alert Category & Diagnostic Meaning | Supporting Source / Requirement |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| **1** | **HEALTHY** | **HEALTHY** | **YES** | **YES** (Both) | **YES** | $S_{\text{final}} = S_{\text{behavioral}}$. Score $< 25.0 \implies \text{NORMAL}$. | **NO** | None. Nominal full mutual streaming. | Authoritative Baseline (`stage6_rule_engine.md`) |
| **2** | **HEALTHY** | **DEGRADED** | **YES** | **YES** (Style) | **NO** | Min `CAUTION` clamp: If $S_{\text{behavioral}} == \text{NORMAL} \implies \text{CAUTION}$; else $S_{\text{behavioral}}$. | **YES** | `DATA_QUALITY`, `WARNING`: "Maneuver pipeline degraded by inference exception." | Authoritative Requirement D-13 & Stage 6 Architecture |
| **3** | **HEALTHY** | **MISSING** | **YES** | **YES** (Style) | **NO** | Min `CAUTION` clamp: If $S_{\text{behavioral}} == \text{NORMAL} \implies \text{CAUTION}$; else $S_{\text{behavioral}}$. | **YES** | `DATA_QUALITY`, `INFO`: "Maneuver pipeline telemetry omitted." | **AUTHORITATIVE REQUIREMENT** (`test_t13_d13`) |
| **4** | **HEALTHY** | **INVALID** | **YES** | **YES** (Style) | **NO** | Min `CAUTION` clamp: If $S_{\text{behavioral}} == \text{NORMAL} \implies \text{CAUTION}$; else $S_{\text{behavioral}}$. | **YES** | `DATA_QUALITY`, `WARNING`: "Maneuver prediction out-of-bounds or non-finite." | Authoritative Requirement D-13 & D-15 |
| **5** | **HEALTHY** | **QUIESCENT_BYPASS** | **YES** | **YES** (Style) | **YES** | $S_{\text{final}} = S_{\text{behavioral}}$. Normal style $\implies \text{NORMAL}$. Aggressive $\implies \text{CAUTION}$. | **NO** | None. Gate verified physical cruising. | Authoritative Requirement D-01 (`test_t01_d01`) |
| **6** | **DEGRADED** | **HEALTHY** | **YES** | **YES** (Maneuver) | **NO** | If Active Maneuver: Ordinary $S_{\text{behavioral}}$ (`CAUTION`/`WARNING`).<br>If Nominal ($M=0$): `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `WARNING`: "Driving style pipeline degraded by exception." | Locked Policy Decisions 1A & 2A |
| **7** | **DEGRADED** | **DEGRADED** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Dual pipeline failure: Both pipelines degraded." | Authoritative Requirement (Rule 9 / Outage) |
| **8** | **DEGRADED** | **MISSING** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Supervisory failure: Style degraded, Maneuver missing." | Authoritative Requirement (Rule 9) |
| **9** | **DEGRADED** | **INVALID** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Dual pipeline failure: Style degraded, Maneuver invalid." | Authoritative Requirement (Rule 9) |
| **10** | **DEGRADED** | **QUIESCENT_BYPASS** | **YES** | **NO** | **NO** | Strictly `INSUFFICIENT_DATA`. Cruising uncertified without style. | **YES** | `DATA_QUALITY`, `WARNING`: "Cruising quiet, but Style pipeline degraded." | Locked Policy Decision 2A |
| **11** | **MISSING** | **HEALTHY** | **YES** | **YES** (Maneuver) | **NO** | If Active Maneuver: Ordinary $S_{\text{behavioral}}$ (`CAUTION`/`WARNING`).<br>If Nominal ($M=0$): `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `INFO`: "Driving style telemetry missing." | Locked Policy Decisions 1A & 2A |
| **12** | **MISSING** | **DEGRADED** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Supervisory failure: Style missing, Maneuver degraded." | Authoritative Requirement (Rule 9) |
| **13** | **MISSING** | **MISSING** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA` (Rule 1 transport abort). | **YES** | `DATA_QUALITY`, `HIGH`: "Total telemetry absence: Empty packet." | Authoritative Requirement (`PredictionValidator`, Rule 1) |
| **14** | **MISSING** | **INVALID** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Supervisory failure: Style missing, Maneuver invalid." | Authoritative Requirement (Rule 9) |
| **15** | **MISSING** | **QUIESCENT_BYPASS** | **YES** | **NO** | **NO** | Strictly `INSUFFICIENT_DATA`. Cruising uncertified without style. | **YES** | `DATA_QUALITY`, `INFO`: "Cruising quiet, but Style telemetry missing." | Locked Policy Decision 2A |
| **16** | **INVALID** | **HEALTHY** | **YES** | **YES** (Maneuver) | **NO** | If Active Maneuver: Ordinary $S_{\text{behavioral}}$ (`CAUTION`/`WARNING`).<br>If Nominal ($M=0$): `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `WARNING`: "Driving style prediction invalid." | Locked Policy Decisions 1A & 2A |
| **17** | **INVALID** | **DEGRADED** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Dual pipeline failure: Style invalid, Maneuver degraded." | Authoritative Requirement (Rule 9) |
| **18** | **INVALID** | **MISSING** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Supervisory failure: Style invalid, Maneuver missing." | Authoritative Requirement (Rule 9) |
| **19** | **INVALID** | **INVALID** | **NO** | **NO** ($R=0$) | **NO** | Strictly `INSUFFICIENT_DATA`. | **YES** | `DATA_QUALITY`, `HIGH`: "Dual pipeline failure: Both pipelines invalid." | Authoritative Requirement (Rule 9) |
| **20** | **INVALID** | **QUIESCENT_BYPASS** | **YES** | **NO** | **NO** | Strictly `INSUFFICIENT_DATA`. Cruising uncertified without style. | **YES** | `DATA_QUALITY`, `WARNING`: "Cruising quiet, but Style prediction invalid." | Locked Policy Decision 2A |

---

## 4. Transport Validation and Pipeline Health Separation

### 4.1 Strict Two-Stage Separation of Concerns

```mermaid
flowchart TD
    RAW["Raw Ingress Packet\n(ts, style_pred, style_conf, maneuver_pred, maneuver_conf)"] --> S1["Stage 1: PredictionValidator.validate()\n(Transport & Schema Integrity)"]
    
    S1 --> CHK1{"is_valid is True and\ndata_quality == 'VALID'?"}
    
    CHK1 -- No --> S1_HALT["Terminal Transport Abort (Rule 1):\nRisk Score = 0.0\nState = INSUFFICIENT_DATA\nEmit Transport DATA_QUALITY Alert"]
    
    CHK1 -- Yes --> S2["Stage 2: DeterministicRiskEngine.evaluate()\n(Accumulate Behavioral Risk R_raw)"]
    S2 --> S3["Stage 3: Map Score to S_behavioral\n(NORMAL, CAUTION, WARNING, HIGH_RISK)"]
    
    S3 --> S4["Stage 4: IntelligentSafetyAgent Coordinator\n(Evaluate Pipeline A & B Health Strings)"]
    S4 --> S5{"20-Row D-13 Truth Table Enforcement"}
    
    S5 -->|Mutual Certification Met| EMIT_NORM["Permit SafetyState.NORMAL\n(if Score < 25.0)"]
    S5 -->|Active Maneuver + A Down| EMIT_BEHAV["Emit Ordinary S_behavioral\nAppend DATA_QUALITY Alert (A down)"]
    S5 -->|Cruising + A Down| EMIT_INS["Emit INSUFFICIENT_DATA\nAppend DATA_QUALITY Alert (A down)"]
    S5 -->|B Impaired + A Healthy| EMIT_CLAMP["Clamp to min CAUTION\nAppend DATA_QUALITY Alert (B down)"]
    S5 -->|Dual Impairment| EMIT_FAIL["Emit INSUFFICIENT_DATA\nAppend DATA_QUALITY Alert (Dual down)"]
```

### 4.2 Exact Source Citations Proving Execution Precedence

1. **Transport Guardrail in `PredictionValidator.validate()`:**  
   In [`ml/agent/prediction_validator.py` lines 70–175](file:///d:/driver-machine-learning-model/ml/agent/prediction_validator.py#L70-L175):
   - Inspects timestamp staleness ($> 3.0\text{s}$), future timestamps ($> 1.0\text{s}$), monotonicity violation, empty packets (`raw_is_missing`), and non-numeric/non-finite fields.
   - If any violation occurs: `is_valid = False` and `data_quality_status = "STALE" | "MISSING" | "INVALID"`.
2. **Terminal Short-Circuit in `DeterministicRiskEngine.evaluate()` (Rule 1):**  
   In [`ml/agent/risk_engine.py` lines 179–220](file:///d:/driver-machine-learning-model/ml/agent/risk_engine.py#L179-L220):
   ```python
   if not ctx.is_valid or ctx.data_quality_status != "VALID":
       return RuleEvaluationResult(
           timestamp=ts,
           safety_state=SafetyState.INSUFFICIENT_DATA,
           risk_score=0.0,
           ...
       )
   ```
   Rule 1 executes an immediate `return`. Rules 2–8 are completely bypassed.
3. **Impossibility of Overwriting Transport INSUFFICIENT_DATA:**  
   Because Rule 1 returns `SafetyState.INSUFFICIENT_DATA` directly to `process_predictions()`, no subsequent behavioral scoring occurs.
4. **Independent Pipeline Health Derivation:**  
   In `IntelligentSafetyAgent`:
   - `pipeline_a_health`: Derived by inspecting Style inputs. If `raw_input.style_prediction is None` $\implies$ `"MISSING"`. If validation errors exist $\implies$ `"INVALID"`. If classifier inference raises exception $\implies$ `"DEGRADED"`. If prediction $\in \{0, 1\}$ and confidence valid $\implies$ `"HEALTHY"`.
   - `pipeline_b_health`: Derived by inspecting Maneuver inputs or Kinematic Gate. If `raw_input.maneuver_prediction is None` $\implies$ `"MISSING"`. If validation errors exist $\implies$ `"INVALID"`. If classifier inference raises exception $\implies$ `"DEGRADED"`. If gate reports `VALID_QUIESCENT` $\implies$ `"QUIESCENT_BYPASS"`. If classifier predicts $\in \{1, 2, 3, 4\}$ with valid confidence $\implies$ `"HEALTHY"`.
5. **Provenance of `QUIESCENT_BYPASS`:**  
   - Provenance in commit `eb54f7d` resolving Defect D-01.
   - Extra Trees was trained on Dataset B, containing *only* 4 maneuver classes and **no nominal cruising class**.
   - `KinematicEventGate.evaluate()` in [`ml/agent/safety_agent.py` lines 82–245](file:///d:/driver-machine-learning-model/ml/agent/safety_agent.py#L82-L245) inspects 5 physical kinematic groups. If all are sub-threshold, it returns `GateStatus.VALID_QUIESCENT`.
   - `IntelligentSafetyAgent.process_features()` intercepts this status, bypassing Extra Trees inference and setting `pipeline_b_health = "QUIESCENT_BYPASS"`.
   - Verified by regression test `test_t01_d01_kinematic_event_gate_quiescent_bypass` (lines 611–638).

---

## 5. Formal State-Selection Algorithm (Authoritative Implementation Logic)

The following logic will be implemented in `ml/agent/safety_agent.py`:

```python
def resolve_safety_assessment(
    val_pred: ValidatedPrediction,
    pipeline_a_health: str,
    pipeline_b_health: str,
    eval_result: RuleEvaluationResult,
    ctx: SmoothedContext
) -> Tuple[SafetyState, List[Dict[str, Any]]]:
    """
    Consolidated implementation resolving the authoritative 20-row D-13 truth table
    incorporating locked Option 1A (Active Maneuver) and Option 2A (Nominal Cruising).
    """
    alerts_list = [
        {
            "alert_id": a.alert_id,
            "timestamp": a.timestamp,
            "severity": a.severity.value,
            "category": a.category.value,
            "title": a.title,
            "triggering_condition": a.triggering_condition,
            "explanation": a.explanation,
            "advisory": a.safe_driving_advisory,
            "related_predictions": a.related_predictions
        }
        for a in eval_result.active_alerts
    ]

    # STAGE 1: Transport Validation Guardrail (Rule 1)
    if not val_pred.is_valid or val_pred.data_quality_status != "VALID":
        return SafetyState.INSUFFICIENT_DATA, alerts_list

    is_a_down = pipeline_a_health in ("DEGRADED", "MISSING", "INVALID")
    is_b_down = pipeline_b_health in ("DEGRADED", "MISSING", "INVALID")

    # STAGE 2: Dual Pipeline Failure (Rows 7-9, 12-14, 17-19)
    if is_a_down and is_b_down:
        alerts_list.append(_create_data_quality_alert_dict(
            val_pred.timestamp,
            severity="HIGH",
            title="Supervisory Pipeline Failure",
            explanation=f"Both Pipeline A ({pipeline_a_health}) and Pipeline B ({pipeline_b_health}) are unavailable."
        ))
        return SafetyState.INSUFFICIENT_DATA, alerts_list

    # STAGE 3: Pipeline A Unavailable with Healthy Pipeline B (Rows 6, 11, 16)
    if is_a_down and pipeline_b_health == "HEALTHY":
        alerts_list.append(_create_data_quality_alert_dict(
            val_pred.timestamp,
            severity="WARNING" if pipeline_a_health != "MISSING" else "INFO",
            title="Style Monitoring Offline",
            explanation=f"Driving style pipeline is {pipeline_a_health}. Maneuver monitoring remains active."
        ))
        is_active_maneuver = (
            ctx.is_maneuver_active
            and ctx.active_maneuver in (1, 2, 3, 4)
            and ctx.active_maneuver_confidence >= 0.40
        )
        if is_active_maneuver:
            # OPTION 1A LOCKED: Return ordinary calculated behavioral state
            return eval_result.safety_state, alerts_list
        else:
            # OPTION 2A LOCKED: Cruising uncertified without style
            return SafetyState.INSUFFICIENT_DATA, alerts_list

    # STAGE 4: Pipeline A Unavailable during Quiescent Cruising (Rows 10, 15, 20)
    if is_a_down and pipeline_b_health == "QUIESCENT_BYPASS":
        alerts_list.append(_create_data_quality_alert_dict(
            val_pred.timestamp,
            severity="WARNING" if pipeline_a_health != "MISSING" else "INFO",
            title="Style Monitoring Offline",
            explanation=f"Vehicle cruising quietly, but style pipeline is {pipeline_a_health}."
        ))
        # OPTION 2A LOCKED: Cruising uncertified without style
        return SafetyState.INSUFFICIENT_DATA, alerts_list

    # STAGE 5: Pipeline B Impaired with Healthy Pipeline A (Rows 2, 3, 4)
    if is_b_down:
        alerts_list.append(_create_data_quality_alert_dict(
            val_pred.timestamp,
            severity="WARNING" if pipeline_b_health != "MISSING" else "INFO",
            title="Maneuver Monitoring Offline",
            explanation=f"Maneuver pipeline is {pipeline_b_health}. Style monitoring remains active."
        ))
        # Minimum CAUTION clamp: Never permit NORMAL; never downgrade WARNING/HIGH_RISK
        if eval_result.safety_state == SafetyState.NORMAL:
            return SafetyState.CAUTION, alerts_list
        return eval_result.safety_state, alerts_list

    # STAGE 6: Full Mutual Certification (Rows 1 & 5)
    # Pipeline A is HEALTHY and Pipeline B is HEALTHY or QUIESCENT_BYPASS
    return eval_result.safety_state, alerts_list
```

---

## 6. Exact Implementation Scope & File Boundaries

Phase 1 code modifications are **strictly bounded to exactly three files**:

### 6.1 `ml/agent/risk_engine.py`
1. **In `SafetyAlert` Dataclass:**
   ```python
   @property
   def message(self) -> str:
       return self.explanation
   ```
2. **In `DeterministicRiskEngine.__init__()` and `reset()`:**
   - Update `_last_alert_timestamps` and `_last_alert_severities` type annotation to:
     `Dict[Tuple[AlertCategory, Optional[int]], float]` and `Dict[Tuple[AlertCategory, Optional[int]], AlertSeverity]`.
3. **In `_should_suppress_alert()` and `_record_alert()`:**
   - Accept optional `maneuver_id: Optional[int] = None`.
   - Build key: `key = (category, maneuver_id if category == AlertCategory.MANEUVER else None)`.
4. **Implement `calculate_confidence_scaling()`:**
   ```python
   @staticmethod
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
5. **In `evaluate()`:**
   - Compute independent scaling:
     `phi_style = self.calculate_confidence_scaling(ctx.current_style_confidence)`
     `phi_maneuver = self.calculate_confidence_scaling(ctx.active_maneuver_confidence)`
   - Apply independent style scaling:
     `if ctx.is_sustained_aggressive: raw_score += self.config.weight_sustained_aggressive * phi_style`
     `elif ctx.is_transient_aggressive: raw_score += self.config.weight_transient_aggressive * phi_style`
   - Apply independent maneuver scaling:
     `raw_score += self.config.weight_sudden_deceleration * phi_maneuver` (similarly for turn and accel).
   - Gate flat compound bonus (Option A):
     `if ctx.is_maneuver_active and (ctx.is_sustained_aggressive or ctx.is_transient_aggressive) and phi_style > 0.0 and phi_maneuver > 0.0:`
     `    raw_score += self.config.weight_compound_hazard_bonus  # +25.0`
   - Pass `maneuver_id=ctx.active_maneuver` when checking and recording maneuver alerts.
   - Delete legacy post-accumulation discount block (`raw_score * confidence_factor`).
   - Clamp final score: `final_score = min(100.0, max(0.0, float(raw_score)))`.

### 6.2 `ml/agent/safety_agent.py`
1. **In `SafetyAssessment` Dataclass:**
   - Add fields:
     ```python
     pipeline_a_health: str = "HEALTHY"
     pipeline_b_health: str = "HEALTHY"
     ```
   - Update `to_dict()`: `pipeline_a_health` and `pipeline_b_health` are automatically serialized via `asdict()`.
2. **In `process_predictions()` and `process_features()`:**
   - Compute `pipeline_a_health` and `pipeline_b_health`.
   - Apply the formal state-selection algorithm with Option 1A and Option 2A logic.
   - Populate `pipeline_a_health` and `pipeline_b_health` on returned `SafetyAssessment`.

### 6.3 `tests/test_safety_agent.py`
1. **Modernize 4 Legacy Tests:**
   - `test_harsh_braking_triggers_warning`: Assert `res.risk_score == 50.0` and `res.safety_state == SafetyState.WARNING`.
   - `test_low_confidence_discount`: Assert `res.risk_score == pytest.approx(37.5, abs=0.1)` and `res.safety_state == SafetyState.CAUTION`.
   - `test_score_clamping`: Assert `res.raw_score == 125.0` and `res.risk_score == 100.0` and `res.safety_state == SafetyState.HIGH_RISK`.
   - `test_full_state_transition_sequence`:
     - Step 1: Pass `maneuver_prediction = 0, maneuver_confidence = 1.0`. Assert `pipeline_a_health == "HEALTHY"`, `pipeline_b_health == "HEALTHY"`, `safety_state == SafetyState.NORMAL`.
     - Step 2: Assert `ass2.safety_state == SafetyState.WARNING`, `ass2.risk_score == 50.0`.
2. **Add Dedicated Verification Test Suites:**
   - `test_t10_d10_piecewise_confidence_boundaries`: $C \in [0.0, 0.39, 0.40, 0.50, 0.60, 0.80, 1.0, \text{NaN}, \text{None}, \text{inf}]$.
   - `test_t13_d13_active_maneuver_ordinary_state_under_option_1a`: Sudden Decel at $0.40 \implies \text{CAUTION}$, at $0.60 \implies \text{WARNING}$.
   - `test_t13_d13_nominal_cruising_insufficient_data_under_option_2a`: Pipeline A missing, Pipeline B nominal $\implies \text{INSUFFICIENT\_DATA}$.

---

## 7. Detailed Regression Test Matrix & Coverage Verification

| Test Identifier | Target Defect / Feature | Test Inputs & Fixture | Expected Outcomes & Assertions |
| :--- | :---: | :--- | :--- |
| `test_t09_d09_dissimilar_maneuver_alerts` | D-09 | $t=1.0\text{s}$: Sudden Decel ($M=2$); $t=2.0\text{s}$: Sudden Turn ($M=3$). | Both alerts dispatch. `len(res2.active_alerts) == 1`. |
| `test_t09_d09_same_maneuver_suppression` | D-09 | $t=1.0\text{s}$: Sudden Decel ($M=2$); $t=1.5\text{s}$: Sudden Decel ($M=2$). | Frame 2 alert is throttled. `len(res2.active_alerts) == 0`. |
| `test_t09_d09_alert_message_property` | D-09 | Construct `SafetyAlert` without `message` param. Access `alert.message`. | `alert.message == alert.explanation`. Constructor intact. |
| `test_t10_d10_piecewise_scaling` | D-10 | Evaluate `calculate_confidence_scaling(C)` at boundary points. | $0.0, 0.0, 0.6667, 0.8333, 1.0, 1.0, 0.0, 0.0$. |
| `test_t10_d10_compound_bonus_gating` | D-10 | Style aggressive ($C_S=0.35 \implies \phi=0$); Maneuver active ($C_M=0.90 \implies \phi=1$). | Compound bonus is strictly $0.0$. Raw score $= 50.0$ (no $+25$). |
| `test_t13_d13_pipeline_b_missing` | D-13 | Ingest Style normal ($C_S=0.95$), Maneuver `None`. | `H_A == "HEALTHY"`, `H_B == "MISSING"`, `State == CAUTION`. |
| `test_t13_d13_option_1a_acute_override` | D-13 | Pipeline A missing; Pipeline B Sudden Decel ($C_M=0.40$). | `State == CAUTION`, `Score == 33.33`, `DATA_QUALITY` alert present. |
| `test_t13_d13_option_1a_warning` | D-13 | Pipeline A missing; Pipeline B Sudden Decel ($C_M=0.60$). | `State == WARNING`, `Score == 50.0`, `DATA_QUALITY` alert present. |
| `test_t13_d13_option_2a_cruising` | D-13 | Pipeline A missing; Pipeline B nominal ($M=0, C_M=1.0$). | `State == INSUFFICIENT_DATA`, `DATA_QUALITY` alert present. |
| `test_t13_d13_option_2a_quiescent` | D-13 | Pipeline A missing; Pipeline B `QUIESCENT_BYPASS`. | `State == INSUFFICIENT_DATA`, `DATA_QUALITY` alert present. |
| `test_t01_d01_quiescent_bypass` | D-01 / D-13 | Raw quiet IMU features to `process_features()`. | `H_B == "QUIESCENT_BYPASS"`, `Maneuver == "None / Nominal"`. |
| `test_t16_artifact_hashes` | T-16 | Cryptographic SHA-256 computation of all 4 Stage 5 artifacts. | Exact match byte-for-byte across all four files. |

---

## 8. Legacy Test Modernization Table

| Test Name | Pre-Audit Code & Assertions | Root Cause of Failure | Modernized Assertions [APPROVED] |
| :--- | :--- | :--- | :--- |
| `test_harsh_braking_triggers_warning` | `assert res.risk_score == 35.0`<br>`assert res.safety_state == SafetyState.CAUTION` | Harsh braking weight remediated from $35.0$ to $50.0$ under D-14 to trigger `WARNING`. | `assert res.risk_score == 50.0`<br>`assert res.safety_state == SafetyState.WARNING` |
| `test_low_confidence_discount` | `assert res.risk_score == 24.5` (flat $0.70$ discount on $35.0$) | D-10 Option 1 applies continuous piecewise scaling: $50.0 \times \phi(0.45) = 50.0 \times 0.75 = 37.5\,\text{pts}$. | `assert res.risk_score == pytest.approx(37.5, abs=0.1)`<br>`assert res.safety_state == SafetyState.CAUTION` |
| `test_score_clamping` | `assert res.raw_score == 110.0`<br>`assert res.risk_score == 100.0` | Weight update: $35 (\text{style}) + 50 (\text{SD}) + 25 (\text{compound}) + 15 (\text{repeated}) = 125.0$. | `assert res.raw_score == 125.0`<br>`assert res.risk_score == 100.0`<br>`assert res.safety_state == SafetyState.HIGH_RISK` |
| `test_full_state_transition_sequence` | Step 1 passed `maneuver_prediction=None`, asserting `SafetyState.NORMAL`. | D-13 forbids `NORMAL` when Pipeline B is missing (`None`). Step 1 must pass nominal inputs. | Step 1: `maneuver_prediction=0, maneuver_confidence=1.0`. Assert `H_A=="HEALTHY", H_B=="HEALTHY", State==NORMAL`. Step 2: `ass2.safety_state==WARNING`. |

---

## 9. Risk & Ambiguity Analysis

* **Unresolved Architectural Decisions:** **0**  
  Every policy, truth table row, and state precedence rule has been locked.
* **Compatibility Risk (`SafetyAlert.message`):** **ZERO**  
  Implemented as a pure property returning `self.explanation`. No dataclass constructor or serialization changes.
* **Regression Risk across ML Pipelines:** **ZERO**  
  Non-Stage 6 test suite already passes 100% (54/54 tests in `test_dataset_b_pipeline.py`, `test_dataset_c_binary_pipeline.py`, `test_model_training_and_inference.py`).
* **Artifact Integrity Risk:** **ZERO**  
  Zero model retraining. All four Stage 5 hashes remain byte-for-byte identical.

---

## 10. Exact Acceptance Criteria

Upon completion of Phase 1 implementation:
1. `test_t09_d09_dissimilar_maneuver_alerts_bypass_category_cooldown` **PASSED**.
2. `test_t10_d10_unified_three_tier_confidence_scaling` **PASSED**.
3. `test_t13_d13_independent_pipeline_health_matrix` **PASSED**.
4. All four modernized legacy tests **PASSED**.
5. All 39 tests in `tests/test_safety_agent.py` **PASSED (100%)**.
6. All 93 tests in repository `tests/` directory **PASSED (100%)**.
7. Zero modifications to `pytest.ini`.
8. Zero modifications to model artifacts or datasets.

---

## 11. Final Readiness Verdict

```
================================================================================
FINAL VERDICT: READY FOR USER IMPLEMENTATION APPROVAL
================================================================================
```

All three defect specifications (D-09, D-10, D-13) are unified, mathematically verified, and internally consistent.  
All 20 combinations of the D-13 truth table have explicit, operational outcomes.  
Both newly approved policy decisions (Option 1A and Option 2A) are fully consolidated.  
Transport validation precedence is proven with source citations.  
The 3-file implementation scope is verified and sufficient.  
Zero technical ambiguities remain.

---

## 12. Explicit Confirmation of Read-Only Compliance

* **Python Source Files Modified:** **0**
* **Test Files Modified:** **0**
* **Model Artifacts Modified:** **0**
* **Datasets Modified:** **0**
* **Configuration Files Modified:** **0**
* **Implementation Actions Executed:** **0**
* **Authoritative Deliverable Created:** [`reports/safety_agent/stage6_phase1_final_implementation_readiness.md`](file:///d:/driver-machine-learning-model/reports/safety_agent/stage6_phase1_final_implementation_readiness.md)

**AWAITING EXPLICIT USER AUTHORIZATION TO COMMENCE PHASE 1 IMPLEMENTATION.**

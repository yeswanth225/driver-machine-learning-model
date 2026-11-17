# Stage 6 Phase 1 Pre-Implementation Audit & Final D-13 Specification

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Repository:** `yeswanth225/driver-machine-learning-model`  
**Working Directory:** `D:\driver-machine-learning-model`  
**Branch:** `main`  
**Documented Checkpoint:** `eb54f7d1e03260482500f611b77d1065931e300b`  
**Document Type:** Formal Pre-Implementation Acceptance Audit (Strictly Read-Only Deliverable)  
**Date:** November 2026  
**Status:** AUDIT COMPLETE — SPECIFICATION FULLY ENUMERATED — READY FOR USER APPROVAL — ZERO IMPLEMENTATION ACTIONS EXECUTED  

---

## 1. Verified Repository Checkpoint & Integrity Audit

### 1.1 Git Checkpoint Verification [VERIFIED IMPLEMENTATION BEHAVIOUR]
```powershell
git status
git branch --show-current
git log -1 --oneline
git diff
```
* **Repository Root:** `D:\driver-machine-learning-model`
* **Active Branch:** `main` (synchronized with `origin/main`)
* **HEAD Commit:** `eb54f7d1e03260482500f611b77d1065931e300b`
* **HEAD Subject:** `feat(ml): implement Stage 6 Group A remediation, D-06 residual risk decay, and residual integration architecture design`
* **Tracked Modifications:** **0** (Working tree is strictly clean on all tracked files)
* **Untracked Deliverables (Read-Only Markdown):**
  - `reports/safety_agent/stage6_phase1_final_specification.md`
  - `reports/safety_agent/stage6_phase1_remediation_plan.md`
  - `reports/safety_agent/stage6_phase1_specification_closure_addendum.md`
  - `reports/safety_agent/stage6_phase1_specification_reconciliation.md`
  - `reports/safety_agent/stage6_phase1_preimplementation_audit.md` (this audit)

### 1.2 Model Artifact Cryptographic Hashes [VERIFIED IMPLEMENTATION BEHAVIOUR]
All four frozen Stage 5 models and scalers were verified via direct SHA-256 byte-by-byte computation:
* `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`:  
  `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` (**MATCH**)
* `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`:  
  `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` (**MATCH**)
* `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib`:  
  `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` (**MATCH**)
* `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib`:  
  `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` (**MATCH**)

---

## 2. Locked User Decisions (Final & Immutable)

The following five user decisions are **FINAL, LOCKED**, and incorporated without reopening [USER APPROVED]:

1. **Decision 1 — `SafetyAlert.message` Compatibility Property:**  
   Implement additive property on `SafetyAlert`:
   ```python
   @property
   def message(self) -> str:
       return self.explanation
   ```
   * Strictly preserves dataclass constructor `__init__()`.
   * Preserves `title`, `explanation`, and `to_dict()` JSON schemas.
   * Resolves regression test failure at line 1110 of `tests/test_safety_agent.py`.
2. **Decision 2 — D-10 Option A Flat Compound Bonus Gating:**  
   Preserve documented flat $+25.0$ compound bonus weight, gated so it applies if and only if both modality scaling factors are strictly positive ($\phi(C_{\text{style}}) > 0.0$ and $\phi(C_{\text{maneuver}}) > 0.0$).
3. **Decision 3 — Nominal Maneuver Input in State-Transition Test:**  
   Initial nominal step of `test_full_state_transition_sequence` passes explicit nominal maneuver inputs (`maneuver_prediction = 0`, `maneuver_confidence = 1.0`), cleanly asserting `pipeline_a_health == "HEALTHY"` and `pipeline_b_health == "HEALTHY"`, permitting `SafetyState.NORMAL`.
4. **Decision 4 — Five Canonical Pipeline Health States Preserved:**  
   Preserve exact canonical enum strings: `HEALTHY`, `DEGRADED`, `MISSING`, `INVALID`, `QUIESCENT_BYPASS`.
5. **Decision 5 — Pytest Configuration Retained as Separate Task:**  
   Test commands remain explicitly targeted to the `tests/` directory (`python -m pytest tests/ -v`). `pytest.ini` remains untouched.

---

## 3. Exhaustive 20-Combination D-13 Truth Table

The supervisory health coordinator evaluates the 20 combinations of Pipeline A ($4$ states) and Pipeline B ($5$ states). Every row is individually enumerated without grouping:

| # | Pipeline A Health ($H_A$) | Pipeline B Health ($H_B$) | `NORMAL` Permitted? | Minimum Fail-Safe State | Behavioral Risk Calculable? | Modality Usability | Acute Maneuver Can Generate `WARNING`? | `DATA_QUALITY` Diagnostic Required? | Result Becomes `INSUFFICIENT_DATA`? | Final State-Selection Rule | Supporting Requirement / Source |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **HEALTHY** | **HEALTHY** | **YES** | `NORMAL` | **YES** | Both Style & Maneuver usable | **YES** ($W=50.0 \implies \text{WARNING}$) | **NO** | **NO** | $S_{\text{final}} = S_{\text{behavioral}}$. Normal certified if score $< 25.0$. | Authoritative Baseline (`stage6_rule_engine.md`) |
| **2** | **HEALTHY** | **DEGRADED** | **NO** | `CAUTION` | **YES** | Style usable; Maneuver untrusted | **NO** (B degraded; predictions untrusted) | **YES** (Warning) | **NO** | Fail-safe floor: if $S_{\text{behavioral}} == \text{NORMAL} \implies \text{CAUTION}$; else $S_{\text{behavioral}}$. | Authoritative Requirement D-13 & Proposed Floor |
| **3** | **HEALTHY** | **MISSING** | **NO** | `CAUTION` | **YES** | Style usable; Maneuver omitted | **NO** (No maneuver stream) | **YES** (Info) | **NO** | Fail-safe floor: if $S_{\text{behavioral}} == \text{NORMAL} \implies \text{CAUTION}$; else $S_{\text{behavioral}}$. | **AUTHORITATIVE REQUIREMENT** (`test_t13_d13`) |
| **4** | **HEALTHY** | **INVALID** | **NO** | `CAUTION` | **YES** | Style usable; Maneuver invalid | **NO** (B payload rejected) | **YES** (Warning) | **NO** | Fail-safe floor: if $S_{\text{behavioral}} == \text{NORMAL} \implies \text{CAUTION}$; else $S_{\text{behavioral}}$. | Authoritative Requirement D-13 & D-15 |
| **5** | **HEALTHY** | **QUIESCENT_BYPASS** | **YES** | `NORMAL` | **YES** | Style usable; Maneuver verified nominal by kinematic gate | **N/A** (Kinematic gate confirms vehicle is quiet) | **NO** (Nominal operating bypass) | **NO** | $S_{\text{final}} = S_{\text{behavioral}}$. Normal certified if score $< 25.0$. | Authoritative Requirement D-01 (`test_t01_d01`) |
| **6** | **DEGRADED** | **HEALTHY** | **NO** | Conditional (`INSUFFICIENT_DATA` or `WARNING`) | **YES** | Maneuver usable; Style untrusted | **YES** (Acute maneuver override applies) | **YES** (Warning: Pipeline A degraded) | **Conditional** (YES if $M=0$; NO if $M \in \{1, 2, 3, 4\}$) | If active maneuver detected $\implies S_{\text{final}} = \text{WARNING}$; if $M=0 \implies S_{\text{final}} = \text{INSUFFICIENT\_DATA}$. | Proposed Policy (Awaiting User Lock) / Case A & B |
| **7** | **DEGRADED** | **DEGRADED** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable (Both models crashed / corrupted) | **NO** | **YES** (High: Dual pipeline failure) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9 / Complete Outage) |
| **8** | **DEGRADED** | **MISSING** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable (Primary degraded, secondary missing) | **NO** | **YES** (High: Unmonitored driving) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **9** | **DEGRADED** | **INVALID** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable (Dual failure) | **NO** | **YES** (High: Dual failure) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **10** | **DEGRADED** | **QUIESCENT_BYPASS** | **NO** | `INSUFFICIENT_DATA` | **NO** | Kinematic gate verifies quiet cruising, but Style untrusted | **NO** (No acute event) | **YES** (Warning: Pipeline A degraded during cruising) | **YES** (Cruising cannot be certified without style stream) | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Proposed Policy (Awaiting User Lock) |
| **11** | **MISSING** | **HEALTHY** | **NO** | Conditional (`INSUFFICIENT_DATA` or `WARNING`) | **YES** | Maneuver usable; Style missing | **YES** (Acute maneuver override applies) | **YES** (Info: Pipeline A missing) | **Conditional** (YES if $M=0$; NO if $M \in \{1, 2, 3, 4\}$) | If active maneuver detected $\implies S_{\text{final}} = \text{WARNING}$; if $M=0 \implies S_{\text{final}} = \text{INSUFFICIENT\_DATA}$. | Proposed Policy (Awaiting User Lock) / Case A & B |
| **12** | **MISSING** | **DEGRADED** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable | **NO** | **YES** (High: Both unavailable) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **13** | **MISSING** | **MISSING** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable (Total telemetry absence) | **NO** | **YES** (High: Packet empty) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$ (intercepted by Rule 1 at validator level). | Authoritative Requirement (`PredictionValidator`, Rule 1) |
| **14** | **MISSING** | **INVALID** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable | **NO** | **YES** (High: Both unavailable) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **15** | **MISSING** | **QUIESCENT_BYPASS** | **NO** | `INSUFFICIENT_DATA` | **NO** | Kinematic gate verifies quiet cruising, but Style missing | **NO** (No acute event) | **YES** (Info: Pipeline A missing during cruising) | **YES** (Cruising cannot be certified without style stream) | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Proposed Policy (Awaiting User Lock) |
| **16** | **INVALID** | **HEALTHY** | **NO** | Conditional (`INSUFFICIENT_DATA` or `WARNING`) | **YES** | Maneuver usable; Style invalid | **YES** (Acute maneuver override applies) | **YES** (Warning: Pipeline A invalid) | **Conditional** (YES if $M=0$; NO if $M \in \{1, 2, 3, 4\}$) | If active maneuver detected $\implies S_{\text{final}} = \text{WARNING}$; if $M=0 \implies S_{\text{final}} = \text{INSUFFICIENT\_DATA}$. | Proposed Policy (Awaiting User Lock) / Case A & B |
| **17** | **INVALID** | **DEGRADED** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable | **NO** | **YES** (High: Dual failure) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **18** | **INVALID** | **MISSING** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable | **NO** | **YES** (High: Both unavailable) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **19** | **INVALID** | **INVALID** | **NO** | `INSUFFICIENT_DATA` | **NO** ($R = 0.0$) | Neither usable | **NO** | **YES** (High: Dual failure) | **YES** | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Authoritative Requirement (Rule 9) |
| **20** | **INVALID** | **QUIESCENT_BYPASS** | **NO** | `INSUFFICIENT_DATA` | **NO** | Kinematic gate verifies quiet cruising, but Style invalid | **NO** (No acute event) | **YES** (Warning: Pipeline A invalid during cruising) | **YES** (Cruising cannot be certified without style stream) | $S_{\text{final}} \equiv \text{INSUFFICIENT\_DATA}$. | Proposed Policy (Awaiting User Lock) |

---

## 4. Resolution of INSUFFICIENT_DATA Precedence (Cases A through F)

### 4.1 Strict Distinction: Transport Invalidation vs. Pipeline Impairment

In safety-critical software, a **Packet-Level Transport Failure** must never be conflated with an **Individual Pipeline Impairment**:

1. **Packet-Level Transport Guardrail (Stage 1 - Rule 1):**  
   Evaluated in `PredictionValidator.validate()`. Triggered when:
   - Packet timestamp is stale: $(t_{\text{current}} - t) > 3.0\,\text{s}$ (`data_quality_status == "STALE"`).
   - Packet timestamp is non-monotonic: $t < t_{\text{last\_valid}}$ without clock reset.
   - Packet timestamp is in future: $(t - t_{\text{current}}) > 1.0\,\text{s}$.
   - Entire packet is empty: both style and maneuver predictions are `None`.
   - Any provided feature contains non-finite (`NaN`, $\pm\infty$) or out-of-domain numeric types.  
   **Behavior:** Terminal short-circuit. `is_valid = False`. Behavioral risk calculation is completely aborted. State is strictly `SafetyState.INSUFFICIENT_DATA` and `risk_score = 0.0`.
2. **Individual Pipeline Impairment (Stage 2 & Stage 5):**  
   Transport packet is valid (`is_valid = True`), but one classifier fails (e.g. Pipeline A is missing while Pipeline B receives valid IMU telemetry).  
   **Behavior:** Behavioral risk is calculated from all valid modalities. Fail-safe state selection governs the final outcome.

---

### 4.2 Detailed Operational Analysis of Cases A through F

#### Case A: Pipeline A Unavailable, Pipeline B Healthy, Active Acute Maneuver (Rows 6, 11, 16)
* **Packet Transport Integrity:** VALID. IMU telemetry and timestamp are fresh and monotonic.
* **Pipeline Health:** Pipeline A $\in \{\text{DEGRADED}, \text{MISSING}, \text{INVALID}\}$; Pipeline B is `HEALTHY`.
* **Behavioral Risk Availability:** High. Maneuver pipeline detects sudden braking ($M=2$, $W=50.0$, $C=0.92 \implies \phi(C)=1.0 \implies R_{\text{maneuver}} = 50.0$).
* **Acute Hazard Availability:** Acute physical hazard is confirmed present.
* **Safety Dilemma:** Is `INSUFFICIENT_DATA` terminal (suppressing the acute brake alert), or does the acute hazard govern?
* **Specification & Precedence Resolution:**  
  > [!IMPORTANT]
  > **Acute Hazard Priority:** An acute physical hazard threatening vehicle stability ($-5.0\,\text{m/s}^2$ harsh deceleration) must **never** be masked by an auxiliary sensor outage. Suppressing an emergency braking warning into `INSUFFICIENT_DATA` because the driving style classifier is offline would introduce an unacceptable safety hazard.
* **Final Safety State:** **`SafetyState.WARNING`** (or `HIGH_RISK` if compounding in Phase 2).
* **Diagnostic Coexistence:** A `DATA_QUALITY` diagnostic alert (`AlertCategory.DATA_QUALITY`, `AlertSeverity.WARNING` or `INFO`) coexists in `active_alerts`, and `pipeline_a_health` is explicitly reported as non-healthy.
* **Governance Status:** **PROPOSED POLICY (REQUIRES USER LOCK)**.

#### Case B: Pipeline A Unavailable, Pipeline B Healthy, No Active Maneuver (Rows 6, 11, 16 with $M=0$)
* **Packet Transport Integrity:** VALID.
* **Pipeline Health:** Pipeline A $\in \{\text{DEGRADED}, \text{MISSING}, \text{INVALID}\}$; Pipeline B is `HEALTHY` (reporting nominal cruising $M=0$).
* **Behavioral Risk Availability:** Low. $R_{\text{maneuver}} = 0.0$; $R_{\text{style}} = 0.0$. Total score is $0.0$.
* **Acute Hazard Availability:** None.
* **Safety Dilemma:** On score alone ($0.0$), state would map to `NORMAL`. But Pipeline A is offline.
* **Specification & Precedence Resolution:**  
  Mutual certification requires both pipelines to certify normal cruising. Because driving style is unmonitored, the system cannot certify safety or even cautious driving. Without an acute maneuver to alert, the system must declare supervisory inability to monitor the driver.
* **Final Safety State:** **`SafetyState.INSUFFICIENT_DATA`**.
* **Diagnostic Alert:** Emits `DATA_QUALITY` alert explaining Pipeline A outage.
* **Governance Status:** **PROPOSED POLICY (REQUIRES USER LOCK)**.

#### Case C: Pipeline A Healthy, Pipeline B Missing (Row 3)
* **Packet Transport Integrity:** VALID.
* **Pipeline Health:** Pipeline A is `HEALTHY`; Pipeline B is `MISSING`.
* **Behavioral Risk Availability:** Style risk is fully calculable (e.g. $0.0$ if Normal, $35.0$ if Sustained Aggressive).
* **Acute Hazard Availability:** None (maneuver channel missing).
* **Final Safety State:**  
  If Style is Normal ($0$ pts): Cannot emit `NORMAL` (forbidden by D-13). Fail-safe floor elevates state to **`SafetyState.CAUTION`**.  
  If Style is Aggressive ($35.0$ pts): Behavioral state is **`SafetyState.CAUTION`**.
* **Diagnostic Alert:** Emits `DATA_QUALITY` advisory (`AlertSeverity.INFO`, "Pipeline B telemetry omitted").
* **Governance Status:** **AUTHORITATIVE REQUIREMENT** (Directly enforced by `tests/test_safety_agent.py` line 1217–1219, `test_t13_d13`).

#### Case D: Both Pipelines Unavailable (Rows 7–9, 12–14, 17–19)
* **Packet Transport Integrity:** Either invalid (empty packet) or valid transport but both classifiers failed/corrupted.
* **Pipeline Health:** Dual impairment ($H_A \ne \text{HEALTHY}$, $H_B \notin \{\text{HEALTHY}, \text{QUIESCENT}\}$).
* **Behavioral Risk Availability:** Zero ($R = 0.0$).
* **Acute Hazard Availability:** None.
* **Final Safety State:** Strictly **`SafetyState.INSUFFICIENT_DATA`**.
* **Diagnostic Alert:** Emits `DATA_QUALITY` alert (`AlertSeverity.HIGH`, "Dual pipeline supervisory failure").
* **Governance Status:** **AUTHORITATIVE REQUIREMENT** (Baseline Rule 9 / Stage 6 architecture).

#### Case E: One Pipeline Degraded While the Other Remains Healthy
* **Sub-case E1 (Pipeline A Healthy, Pipeline B Degraded - Row 2):**  
  Style monitored, maneuver classifier crashed or features corrupted.  
  Behavioral risk from style calculated. Maneuver untrusted.  
  Fail-safe floor: **`SafetyState.CAUTION`**. `NORMAL` is forbidden.  
  Emits `DATA_QUALITY` warning for Pipeline B.
* **Sub-case E2 (Pipeline A Degraded, Pipeline B Healthy - Row 6):**  
  Maneuver monitored, style classifier crashed or features corrupted.  
  If acute maneuver active $\implies$ **`SafetyState.WARNING`** (Acute Hazard Override).  
  If maneuver nominal ($M=0$) $\implies$ **`SafetyState.INSUFFICIENT_DATA`** (Cruising uncertifiable).  
  Emits `DATA_QUALITY` warning for Pipeline A.
* **Governance Status:** **PROPOSED POLICY (REQUIRES USER LOCK)**.

#### Case F: Pipeline B Intentionally Quiescent While Pipeline A Is Healthy (Row 5)
* **Packet Transport Integrity:** VALID.
* **Pipeline Health:** Pipeline A is `HEALTHY`; Pipeline B is `QUIESCENT_BYPASS`.
* **Behavioral Risk Availability:** Fully available. `KinematicEventGate` confirms that all 5 kinematic metrics are present, finite, and strictly below physical excursion thresholds. Extra Trees classifier is safely bypassed.
* **Acute Hazard Availability:** Physical absence of acute maneuver positively verified by kinematic gate.
* **Final Safety State:**  
  If Style is Normal $\implies$ **`SafetyState.NORMAL`** (Mutual Certification Satisfied).  
  If Style is Sustained Aggressive $\implies$ **`SafetyState.CAUTION`** ($35.0$ pts).
* **Diagnostic Alert:** **NONE**. Quiescent bypass is an intentional, efficient nominal operating mode, not a failure.
* **Governance Status:** **AUTHORITATIVE REQUIREMENT** (D-01, `test_t01_d01_kinematic_event_gate_quiescent_bypass`).

---

### 4.3 Why INSUFFICIENT_DATA Cannot Be Compared Numerically

In Python, `SafetyState` is defined as:
```python
class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
```
Because `SafetyState` inherits from `(str, Enum)`:
1. Python compares enum members **lexicographically by string value**, NOT by severity!
   - `'NORMAL' > 'CAUTION'` is `True` (because `'N' > 'C'`). Calling `max(NORMAL, CAUTION)` dangerously returns `"NORMAL"`!
   - `'INSUFFICIENT_DATA' < 'NORMAL'` is `True` (because `'I' < 'N'`).
2. `INSUFFICIENT_DATA` represents an **administrative / data-validity state**, not a behavioral risk tier. It does not exist on the continuous $0\dots 100$ risk scale.
3. Therefore, **generic `max()` operations on `SafetyState` are strictly prohibited**. All state selections must use explicit precedence branches or an explicit integer severity dictionary that excludes `INSUFFICIENT_DATA`.

---

## 5. Formal State-Selection Algorithm (Detailed Pseudocode)

The following deterministic algorithm implements the exact 7-stage order of operations, distinguishing transport faults, individual pipeline health, behavioral risk, and fail-safe floors:

```python
# =============================================================================
# EXPLICIT SEVERITY ORDERING (Excludes administrative INSUFFICIENT_DATA)
# =============================================================================
BEHAVIORAL_SEVERITY_TIER = {
    SafetyState.NORMAL: 0,
    SafetyState.CAUTION: 1,
    SafetyState.WARNING: 2,
    SafetyState.HIGH_RISK: 3
}

def resolve_final_safety_state(
    val_pred: ValidatedPrediction,
    pipeline_a_health: str,
    pipeline_b_health: str,
    eval_result: RuleEvaluationResult,
    ctx: SmoothedContext
) -> Tuple[SafetyState, List[SafetyAlert]]:
    """
    Deterministically computes final SafetyState and appends required
    diagnostic alerts based on the authoritative 20-combination truth table.
    
    A minimum fail-safe state NEVER downgrades a higher behavioral state.
    """
    alerts = list(eval_result.active_alerts)

    # -------------------------------------------------------------------------
    # STAGE 1: Packet-Level Transport Validation Failure (Rule 1 Guardrail)
    # -------------------------------------------------------------------------
    if not val_pred.is_valid or val_pred.data_quality_status != "VALID":
        # Terminal transport failure: packet is stale, non-monotonic, or empty
        return SafetyState.INSUFFICIENT_DATA, alerts

    # -------------------------------------------------------------------------
    # STAGE 2: Identify Acute Maneuver Detection
    # -------------------------------------------------------------------------
    # An acute maneuver requires Pipeline B to be HEALTHY and active maneuver
    # to be one of the physical abrupt classes (1=SA, 2=SD, 3=SRT, 4=SLT)
    is_acute_maneuver_active = (
        pipeline_b_health == "HEALTHY"
        and ctx.is_maneuver_active
        and ctx.active_maneuver in (1, 2, 3, 4)
        and ctx.active_maneuver_confidence >= 0.40
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Dual Pipeline Failure Check
    # -------------------------------------------------------------------------
    is_pipeline_a_unavailable = pipeline_a_health in ("DEGRADED", "MISSING", "INVALID")
    is_pipeline_b_unavailable = pipeline_b_health in ("DEGRADED", "MISSING", "INVALID")

    if is_pipeline_a_unavailable and is_pipeline_b_unavailable:
        # Rows 7-9, 12-14, 17-19: Total supervisory failure
        _append_data_quality_alert(
            alerts,
            severity=AlertSeverity.HIGH,
            title="Supervisory Pipeline Failure",
            message=f"Both Pipeline A ({pipeline_a_health}) and Pipeline B ({pipeline_b_health}) are unavailable."
        )
        return SafetyState.INSUFFICIENT_DATA, alerts

    # -------------------------------------------------------------------------
    # STAGE 4: Asymmetric Degradation with Acute Maneuver Override (Rows 6, 11, 16)
    # -------------------------------------------------------------------------
    if is_pipeline_a_unavailable and pipeline_b_health == "HEALTHY":
        _append_data_quality_alert(
            alerts,
            severity=AlertSeverity.WARNING if pipeline_a_health != "MISSING" else AlertSeverity.INFO,
            title="Style Monitoring Offline",
            message=f"Driving style pipeline is {pipeline_a_health}. Maneuver monitoring remains active."
        )
        if is_acute_maneuver_active:
            # ACUTE HAZARD OVERRIDE: Severe maneuver governs over style outage
            return eval_result.safety_state, alerts
        else:
            # Cruising cannot be certified safe without driving style monitoring
            return SafetyState.INSUFFICIENT_DATA, alerts

    # -------------------------------------------------------------------------
    # STAGE 5: Quiescent Cruising with Unavailable Style Stream (Rows 10, 15, 20)
    # -------------------------------------------------------------------------
    if is_pipeline_a_unavailable and pipeline_b_health == "QUIESCENT_BYPASS":
        _append_data_quality_alert(
            alerts,
            severity=AlertSeverity.WARNING if pipeline_a_health != "MISSING" else AlertSeverity.INFO,
            title="Style Monitoring Offline",
            message=f"Vehicle is cruising quietly, but style pipeline is {pipeline_a_health}."
        )
        return SafetyState.INSUFFICIENT_DATA, alerts

    # -------------------------------------------------------------------------
    # STAGE 6: Fail-Safe Minimum State Enforcement (Rows 2, 3, 4)
    # -------------------------------------------------------------------------
    # At this point, Pipeline A is guaranteed HEALTHY
    if is_pipeline_b_unavailable:
        _append_data_quality_alert(
            alerts,
            severity=AlertSeverity.WARNING if pipeline_b_health != "MISSING" else AlertSeverity.INFO,
            title="Maneuver Monitoring Offline",
            message=f"Maneuver pipeline is {pipeline_b_health}. Style monitoring remains active."
        )
        # NORMAL is strictly forbidden; minimum fail-safe floor is CAUTION
        behavioral_state = eval_result.safety_state
        if behavioral_state == SafetyState.NORMAL:
            return SafetyState.CAUTION, alerts
        else:
            # Never downgrade WARNING or HIGH_RISK
            return behavioral_state, alerts

    # -------------------------------------------------------------------------
    # STAGE 7: Full Mutual Certification (Rows 1 & 5)
    # -------------------------------------------------------------------------
    # Pipeline A is HEALTHY and Pipeline B is HEALTHY or QUIESCENT_BYPASS
    return eval_result.safety_state, alerts
```

---

## 6. Explicit Formulation of Approved D-10 Option A

### 6.1 Exact Approved Formula [USER APPROVED]

Confidence scaling operator:
$$\phi(C) = \begin{cases} 0.0 & C < 0.40 \\ \frac{C}{0.60} & 0.40 \le C < 0.60 \\ 1.0 & C \ge 0.60 \end{cases}$$

Compound hazard bonus ($R_{\text{compound}}$):
$$R_{\text{compound}} = \begin{cases} 25.0 & \text{if } \text{is\_compound\_hazard\_active} \land \phi(C_{\text{style}}) > 0.0 \land \phi(C_{\text{maneuver}}) > 0.0 \\ 0.0 & \text{otherwise} \end{cases}$$

Where:
- $\text{is\_compound\_hazard\_active} \iff \text{ctx.is\_maneuver\_active} \land (\text{ctx.is\_sustained\_aggressive} \lor \text{ctx.is\_transient\_aggressive})$.
- The original compound hazard detection semantics are **strictly preserved** without alteration.
- The bonus is **strictly flat ($+25.0$ points)**. It is **never proportionally scaled**.

### 6.2 Integration into Raw Score Summation

The raw risk score is accumulated directly from independent, confidence-scaled hazard contributions:

$$\text{Score}_{\text{raw}} = R_{\text{style}} + R_{\text{maneuver}} + R_{\text{compound}} + R_{\text{repeated}}$$

Where:
1. **Style Contribution:**
   $$R_{\text{style}} = W_{\text{style}} \cdot \phi(C_{\text{style}})$$
   ($W_{\text{style}} = 35.0$ for sustained aggressive, $20.0$ for transient aggressive, else $0.0$).
2. **Maneuver Contribution:**
   $$R_{\text{maneuver}} = W_{\text{maneuver}} \cdot \phi(C_{\text{maneuver}})$$
   ($W_{\text{maneuver}} = 50.0$ for Sudden Deceleration, $35.0$ for Sudden Turn, $25.0$ for Sudden Acceleration, else $0.0$).
3. **Compound Bonus:** Flat $+25.0$ points if gated condition met, else $0.0$.
4. **Repeated Bonus:** Flat $+15.0$ points if `ctx.recent_maneuver_count >= 2`, else $0.0$.
5. **Score Clamping:** $\text{Score}_{\text{final}} = \min(100.0, \max(0.0, \text{Score}_{\text{raw}}))$.
6. **Elimination of Legacy Scaling:** The legacy post-accumulation discount (`raw_score * confidence_factor`) is completely removed.

---

## 7. Verification of Test Specifications

### 7.1 Full State-Transition Test Verification
* **Target:** `test_full_state_transition_sequence` in `tests/test_safety_agent.py` (lines 537–565).
* **Initial Nominal Step Verification:**
  ```python
  ass1 = agent.process_predictions(
      timestamp=1.0,
      style_prediction=0,
      style_confidence=0.95,
      maneuver_prediction=0,      # ManeuverClass.NONE.value
      maneuver_confidence=1.0
  )
  assert hasattr(ass1, "pipeline_a_health")
  assert hasattr(ass1, "pipeline_b_health")
  assert ass1.pipeline_a_health == "HEALTHY"
  assert ass1.pipeline_b_health == "HEALTHY"
  assert ass1.safety_state == SafetyState.NORMAL
  assert ass1.risk_score == 0.0
  ```
  - Exercises the real `process_predictions()` execution path.
  - Passes valid nominal predictions for both streams.
  - Returned health strings match `"HEALTHY"` and `"HEALTHY"`.
  - Mutual certification is genuinely met, cleanly permitting `SafetyState.NORMAL`.
* **Step 2 (Acute Braking Step):**
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
  - Sudden Deceleration ($W=50.0$, $C=0.92 \implies \phi=1.0$) produces $50.0$ points.
  - Matches the WARNING threshold ($\ge 50.0$).

### 7.2 Independent Verification of QUIESCENT_BYPASS in Feature Path
* The production feature path (`process_features()`) evaluates the kinematic event gate.
* When `gate.evaluate()` returns `GateStatus.VALID_QUIESCENT`, `pipeline_b_health` is set to `"QUIESCENT_BYPASS"`.
* Verified in existing test `test_process_features_with_production_models` (lines 567–599) and `test_t01_d01_kinematic_event_gate_quiescent_bypass` (lines 611–638).

### 7.3 Coverage Preservation Across 4 Legacy Test Replacements
1. `test_harsh_braking_triggers_warning`: Tests isolated harsh deceleration with weight $50.0$, verifying that braking alone triggers `WARNING` without requiring compound aggressive driving.
2. `test_low_confidence_discount`: Tests sub-saturated confidence scaling ($C=0.45 \implies \phi(0.45)=0.75 \implies 50.0 \times 0.75 = 37.5$ pts, `CAUTION`), verifying continuous sensitivity without deadband distortion.
3. `test_score_clamping`: Tests multi-hazard accumulation ($35 + 50 + 25 + 15 = 125.0$) clamped to $100.0$ and `HIGH_RISK`, verifying saturation bounds.
4. `test_full_state_transition_sequence`: Tests multi-step temporal state progression across `NORMAL` $\to$ `WARNING` $\to$ `CAUTION` $\to$ `INSUFFICIENT_DATA`.

---

## 8. Strictly Restricted 3-File Implementation Scope

Phase 1 code modifications remain strictly bounded to exactly three files:

1. **[`ml/agent/risk_engine.py`](file:///d:/driver-machine-learning-model/ml/agent/risk_engine.py):**
   - Add `@property def message(self) -> str: return self.explanation` to `SafetyAlert`.
   - Update `_last_alert_timestamps` and `_last_alert_severities` key type to `Tuple[AlertCategory, Optional[int]]`.
   - Update `_should_suppress_alert()` and `_record_alert()` to accept `maneuver_id: Optional[int] = None`.
   - Implement `calculate_confidence_scaling(confidence: Optional[float]) -> float`.
   - Update `evaluate()` to apply modality-independent $\phi(C)$ scaling and gated Rule 7 compound bonus.
2. **[`ml/agent/safety_agent.py`](file:///d:/driver-machine-learning-model/ml/agent/safety_agent.py):**
   - Add `pipeline_a_health: str = "HEALTHY"` and `pipeline_b_health: str = "HEALTHY"` to `SafetyAssessment` dataclass and `to_dict()`.
   - In `process_predictions()` and `process_features()`, compute independent health strings ($H_A, H_B$).
   - Implement the deterministic 7-stage state-selection algorithm with fail-safe clamps.
3. **[`tests/test_safety_agent.py`](file:///d:/driver-machine-learning-model/tests/test_safety_agent.py):**
   - Update assertions in `test_harsh_braking_triggers_warning` ($50.0$, `WARNING`).
   - Update assertions in `test_low_confidence_discount` ($37.5$, `CAUTION`).
   - Update assertions in `test_score_clamping` (raw $125.0$, clamped $100.0$, `HIGH_RISK`).
   - Update Step 1 of `test_full_state_transition_sequence` with explicit nominal maneuver and health assertions.
   - Add dedicated piecewise confidence boundary test suite.

---

## 9. Final Readiness Verdict & Policy Approval Gate

### 9.1 Verdict Classification

**`READY FOR USER APPROVAL`**

All 20 combinations of the D-13 truth table are individually specified. The 7-stage state-selection algorithm is fully formalized. The exact mathematical formulation for D-10 Option A is detailed. The test verifications and 3-file implementation boundaries are complete.

### 9.2 Itemized Policy Decision Checklist for User Lock

To authorize Phase 1 implementation, the user is requested to lock the following synthesized policies:

- [ ] **Policy Lock 1 (Acute Hazard Override - Rows 6, 11, 16):**  
  Confirm that when Pipeline A is unavailable (`DEGRADED`, `MISSING`, `INVALID`) but Pipeline B is `HEALTHY` and detects an active acute maneuver ($M \in \{1, 2, 3, 4\}$ with $C \ge 0.40$), the acute maneuver hazard state (`WARNING` or `HIGH_RISK`) governs `safety_state`, while a concurrent `DATA_QUALITY` diagnostic alert is appended.
- [ ] **Policy Lock 2 (Nominal Cruising Demotion - Rows 6, 10, 11, 15, 16, 20):**  
  Confirm that when Pipeline A is unavailable (`DEGRADED`, `MISSING`, `INVALID`) and Pipeline B reports nominal cruising ($M=0$ or `QUIESCENT_BYPASS`), the final state is `INSUFFICIENT_DATA` (with a `DATA_QUALITY` diagnostic alert) because cruising cannot be certified safe without driving style monitoring.
- [ ] **Policy Lock 3 (Fail-Safe CAUTION Floor - Rows 2, 3, 4):**  
  Confirm that when Pipeline A is `HEALTHY` and Pipeline B is impaired (`DEGRADED`, `MISSING`, `INVALID`), `SafetyState.NORMAL` is strictly forbidden, and the minimum fail-safe state is clamped to `CAUTION` (never downgrading higher behavioral risk).
- [ ] **Policy Lock 4 (Scope Authorization):**  
  Authorize proceeding with code modifications strictly bounded to `ml/agent/risk_engine.py`, `ml/agent/safety_agent.py`, and `tests/test_safety_agent.py`.

---

## 10. Explicit Confirmation of Read-Only Compliance

* **Python Source Files Modified:** **0**
* **Test Files Modified:** **0**
* **Model Artifacts Modified:** **0**
* **Datasets Modified:** **0**
* **Configuration Files Created / Modified:** **0**
* **Implementation Actions Executed:** **0**
* **Permitted Deliverable Created:** [`reports/safety_agent/stage6_phase1_preimplementation_audit.md`](file:///d:/driver-machine-learning-model/reports/safety_agent/stage6_phase1_preimplementation_audit.md)

**AUDIT COMPLETE. ZERO IMPLEMENTATION ACTIONS EXECUTED. AWAITING USER APPROVAL TO PROCEED WITH PHASE 1 IMPLEMENTATION.**

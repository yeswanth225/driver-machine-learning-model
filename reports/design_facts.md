# IDAMS Canonical Design Facts

This document compiles read-only, forensically audited code facts extracted directly from the codebase of `yeswanth225/driver-machine-learning-model`. Values are quoted verbatim with exact file:line references. Where a concept was claimed in prior documentation but does not exist in the implementation, it is explicitly marked as `NOT FOUND`.

---

## 1. CONFIG (`ml/agent/config.py`)

### Verbatim Enums

```python
# ml/agent/config.py:13-19
class SafetyState(str, Enum):
    """Unified Driver Safety States."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
```

```python
# ml/agent/config.py:22-25
class DrivingStyle(int, Enum):
    """Pipeline A (Dataset C) Class Labels."""
    NORMAL = 0
    AGGRESSIVE = 1
```

```python
# ml/agent/config.py:28-34
class ManeuverClass(int, Enum):
    """Pipeline B (Dataset B) Class Labels."""
    NONE = 0
    SUDDEN_ACCELERATION = 1
    SUDDEN_DECELERATION = 2
    SUDDEN_RIGHT_TURN = 3
    SUDDEN_LEFT_TURN = 4
```

```python
# ml/agent/config.py:37-43
class AlertSeverity(str, Enum):
    """Alert Severity Levels."""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
```

```python
# ml/agent/config.py:46-51
class AlertCategory(str, Enum):
    """Alert Functional Categories."""
    STYLE = "STYLE"
    MANEUVER = "MANEUVER"
    COMPOUND = "COMPOUND"
    DATA_QUALITY = "DATA_QUALITY"
```

```python
# ml/agent/config.py:54-58
class GateStatus(str, Enum):
    """Kinematic Event Gate Evaluation Status."""
    VALID_QUIESCENT = "VALID_QUIESCENT"
    VALID_EXCITED = "VALID_EXCITED"
    INVALID_OR_INSUFFICIENT = "INVALID_OR_INSUFFICIENT"
```

```python
# Health Status Enums: NOT FOUND as Enum classes in ml/agent/config.py.
# Health statuses are implemented as string literals across prediction_validator.py and risk_engine.py:
# - Pipeline Health (ml/agent/prediction_validator.py:116, 133, 158, 169, 192, 223):
#     "HEALTHY", "QUIESCENT_BYPASS", "DEGRADED", "MISSING", "INVALID"
# - Data Quality Status (ml/agent/prediction_validator.py:244, 247, 250, 253, 259, 261, 263):
#     "VALID", "STALE", "MISSING", "INVALID", "DEGRADED"
```

### Verbatim `SafetyAgentConfig` Dataclass & Defaults

```python
# ml/agent/config.py:87-152
@dataclass(frozen=True)
class SafetyAgentConfig:
    """Centralized, immutable configuration parameters for the Safety Monitoring Agent."""

    # 0. Baseline Score
    baseline_risk_score: float = 0.0

    # 1. Temporal & Staleness Thresholds (Seconds)
    stale_threshold_seconds: float = 3.0
    future_timestamp_threshold_seconds: float = 1.0
    max_forward_leap_seconds: float = 10.0
    history_window_seconds: float = 10.0
    style_smoothing_window_seconds: float = 3.0
    maneuver_persistence_seconds: float = 2.0
    maneuver_history_window_seconds: float = 10.0

    # 2. Smoothing Weights
    style_ema_alpha: float = 0.20
    style_ema_tau: float = 1.50
    style_aggressive_ratio_threshold: float = 0.50

    # 3. Confidence Thresholds
    min_confidence_threshold: float = 0.50
    high_confidence_threshold: float = 0.75
    low_confidence_discount: float = 0.70  # 30% reduction for low confidence

    # 4. Deterministic Risk Score Weights (0 to 100 scale)
    weight_sustained_aggressive: float = 35.0
    weight_transient_aggressive: float = 20.0
    weight_sudden_acceleration: float = 25.0
    weight_sudden_deceleration: float = 50.0
    weight_sudden_turn: float = 35.0
    weight_compound_high_risk: float = 25.0
    weight_compound_hazard_bonus: float = 25.0
    weight_repeated_maneuver_bonus: float = 15.0

    # 5. Risk Score to State Boundaries
    threshold_caution: float = 25.0
    threshold_warning: float = 50.0
    threshold_high_risk: float = 75.0
    score_threshold_caution: float = 25.0
    score_threshold_warning: float = 50.0
    score_threshold_high_risk: float = 75.0
    high_risk_enter_threshold: float = 75.0
    high_risk_exit_threshold: float = 75.0

    # 6. Alert Deduplication Cooldown Timers (Seconds)
    alert_cooldown_style: float = 5.0
    alert_cooldown_maneuver: float = 3.0
    alert_cooldown_compound: float = 2.0
    alert_cooldown_data_quality: float = 5.0

    # 7. Kinematic Event Gate Thresholds (Pipeline B)
    gate_enabled: bool = True
    gate_acc_horiz_std_threshold: float = 0.30
    gate_gyro_z_std_threshold: float = 0.05
    gate_acc_mag_dev_threshold: float = 1.47  # m/s^2 deviation from 1g (~0.15g)
    gate_gyro_mag_threshold: float = 0.25     # rad/s (~14.3 deg/s)

    # 8. Residual Risk Decay & Incident Lifecycle (D-02 & D-06)
    residual_plateau_seconds: float = 1.00
    residual_half_life_seconds: float = 0.75
    residual_retirement_threshold: float = 2.50
    maneuver_rearm_seconds: float = 2.50
```

---

## 2. SCORE FLOW (`ml/agent/risk_engine.py`)

Order of operations in `DeterministicRiskEngine.evaluate(ctx: SmoothedContext)`:

1. **Layer 1 Transport & Packet Integrity Guard** (`ml/agent/risk_engine.py:260-300`):
   - If `not ctx.is_valid`: Emits `DATA_QUALITY` alert (`AlertSeverity.HIGH`, cooldown `alert_cooldown_data_quality=5.0s`), short-circuits evaluation immediately, returning `safety_state = SafetyState.INSUFFICIENT_DATA`, `risk_score = 0.0`, `raw_score = 0.0`, `confidence_factor = 0.0`.
2. **Fail-Safe Condition 1 — Dual Pipeline Impairment** (`ml/agent/risk_engine.py:333-355`):
   - If `h_a in ("DEGRADED", "MISSING", "INVALID", "QUIESCENT_BYPASS") and h_b in ("DEGRADED", "MISSING", "INVALID")`: Emits `Dual Pipeline Supervisory Failure` alert (`AlertSeverity.HIGH`), short-circuits returning `SafetyState.INSUFFICIENT_DATA`, `risk_score = 0.0`.
3. **Fail-Safe Condition 2 — Cruising during Pipeline A Outage** (`ml/agent/risk_engine.py:359-385`):
   - If `h_a in ("DEGRADED", "MISSING", "INVALID") and (h_b == "QUIESCENT_BYPASS" or (h_b == "HEALTHY" and not ctx.is_maneuver_active))`: Emits `Driving Style Telemetry {h_a.capitalize()}` alert (`INFO` if missing, else `MEDIUM`), short-circuits returning `SafetyState.INSUFFICIENT_DATA`, `risk_score = 0.0`.
4. **Modality Confidence Scaling Computation $\phi(C)$** (`ml/agent/risk_engine.py:142-168, 394-402`):
   - Scaling function:
     $$\phi(C) = \begin{cases} 0.0 & \text{if } C < 0.40 \\ \frac{C}{0.60} & \text{if } 0.40 \le C < 0.60 \\ 1.0 & \text{if } C \ge 0.60 \end{cases}$$
   - Computed independently for style: $\phi_{\text{style}} = \phi(\text{current\_style\_confidence})$ if $h_a \in (\text{"HEALTHY"}, \text{"QUIESCENT\_BYPASS"})$ else $0.0$.
   - Computed independently for maneuver: $\phi_{\text{maneuver}} = \phi(\text{active\_maneuver\_confidence})$ if $h_b \in (\text{"HEALTHY"}, \text{"QUIESCENT\_BYPASS"})$ else $0.0$.
5. **Driving Style Score Contribution** (`ml/agent/risk_engine.py:405-448`):
   - Only evaluated if `h_a == "HEALTHY"`:
     - **Rule 2: Sustained Aggressive** (`ctx.is_sustained_aggressive and phi_style > 0.0`):
       $$\text{raw\_score} \mathrel{+}= \text{weight\_sustained\_aggressive (35.0)} \times \phi_{\text{style}}$$
       Emits `AlertSeverity.MEDIUM` alert with cooldown `alert_cooldown_style (5.0s)`.
     - **Rule 3: Transient Aggressive** (`ctx.is_transient_aggressive and phi_style > 0.0`):
       $$\text{raw\_score} \mathrel{+}= \text{weight\_transient\_aggressive (20.0)} \times \phi_{\text{style}}$$
       (No alert emitted; log explanation only).
6. **Maneuver Contribution & Channel Combination Rule** (`ml/agent/risk_engine.py:450-584`):
   - Only evaluated if `h_b in ("HEALTHY", "QUIESCENT_BYPASS")`:
     - If `ctx.active_contributions` present (multi-channel decay enabled):
       - For each contribution $m$, compute elapsed decay time $dt_{\text{decay}} = \max(0.0, ts - t_{\text{end}})$ if not excited, else $0.0$.
       - Compute residual risk via `calculate_residual_risk()` (`ml/agent/risk_engine.py:82-139`):
         $$R(dt) = W_{\text{initial}} \cdot 2^{-\frac{dt - t_{\text{plateau}}}{t_{1/2}}}$$
         with $t_{\text{plateau}}=1.00\text{s}$, $t_{1/2}=0.75\text{s}$, $R_{\text{retire}}=2.50$, analytical boundary $t_{\text{retire}} = t_{\text{plateau}} + t_{1/2} \log_2(W_{\text{initial}} / R_{\text{retire}})$. For $dt > t_{\text{retire}}$, $R(dt) = 0.0$.
       - Scale: $R_{\text{scaled}} = R(dt) \times \phi(\text{confidence})$.
       - **Within-channel rule:** Take $\max()$ across events within the same channel:
         - Sudden Acceleration (1) $\to$ $\text{longitudinal\_risk} = \max(\text{longitudinal\_risk}, R_{\text{scaled}})$
         - Sudden Deceleration (2) $\to$ $\text{longitudinal\_risk} = \max(\text{longitudinal\_risk}, R_{\text{scaled}})$
         - Sudden Turns (3, 4) $\to$ $\text{lateral\_risk} = \max(\text{lateral\_risk}, R_{\text{scaled}})$
       - **Cross-channel rule (Orthogonal Summation):** Additive stack:
         $$\text{raw\_score} \mathrel{+}= (\text{longitudinal\_risk} + \text{lateral\_risk})$$
7. **Rule 8: Repeated Erratic Maneuver Bonus** (`ml/agent/risk_engine.py:696-700`):
   - If `ctx.recent_maneuver_count >= 2`:
     $$\text{raw\_score} \mathrel{+}= \text{weight\_repeated\_maneuver\_bonus (15.0)}$$
8. **Rule 7: Compound High Risk Hazard Bonus** (`ml/agent/risk_engine.py:705-744`):
   - Gated condition: requires $h_a, h_b \in (\text{"HEALTHY"}, \text{"QUIESCENT\_BYPASS"})$ AND `ctx.is_maneuver_active` AND (`ctx.is_sustained_aggressive` or `ctx.is_transient_aggressive`) AND $\phi_{\text{style}} > 0.0$ AND $\phi_{\text{maneuver}} > 0.0$.
   - Bonus:
     $$\text{raw\_score} \mathrel{+}= \text{weight\_compound\_hazard\_bonus (25.0)}$$
   - Emits `AlertSeverity.CRITICAL` alert with cooldown `alert_cooldown_compound (2.0s)`.
9. **Final Score Clamping** (`ml/agent/risk_engine.py:748`):
   $$\text{final\_score} = \min(100.0, \max(0.0, \text{float}(\text{raw\_score})))$$
10. **Behavioral State Mapping & Hysteresis** (`ml/agent/risk_engine.py:226-246, 751`):
    ```python
    enter_thresh = getattr(self.config, "high_risk_enter_threshold", 75.0)
    exit_thresh = getattr(self.config, "high_risk_exit_threshold", 75.0)
    if self._last_safety_state == SafetyState.HIGH_RISK:
        is_high_risk = final_score >= exit_thresh
    else:
        is_high_risk = final_score >= enter_thresh

    if is_high_risk:
        state = SafetyState.HIGH_RISK
    elif final_score >= self.config.score_threshold_warning (50.0):
        state = SafetyState.WARNING
    elif final_score >= self.config.score_threshold_caution (25.0):
        state = SafetyState.CAUTION
    else:
        state = SafetyState.NORMAL
    self._last_safety_state = state
    ```
11. **Fail-Safe State Arbitration Overrides** (`ml/agent/risk_engine.py:756-782`):
    - **Condition 3** (`h_a in ("DEGRADED", "MISSING", "INVALID") and h_b == "HEALTHY" and ctx.is_maneuver_active`): Preserves calculated behavioral score and state without artificial floor (`ml/agent/risk_engine.py:757-768`).
    - **Condition 4** (`h_a == "HEALTHY" and h_b in ("DEGRADED", "MISSING", "INVALID")`): Enforces a fail-safe CAUTION floor; if `state == SafetyState.NORMAL`, overrides to `state = SafetyState.CAUTION` (`ml/agent/risk_engine.py:770-782`).

---

## 3. HEALTH MATRIX

Code handles the following 4 structural conditions across the 5 possible pipeline health values (`HEALTHY`, `QUIESCENT_BYPASS`, `DEGRADED`, `MISSING`, `INVALID`):

| Pipeline A Health ($h_a$) | Pipeline B Health ($h_b$) | Code Branch | Resulting State | Floor / Policy | Emitted Alert | Lines in `risk_engine.py` |
|:---|:---|:---|:---|:---|:---|:---|
| `HEALTHY` or `QUIESCENT_BYPASS` | `HEALTHY` or `QUIESCENT_BYPASS` | Normal Behavioral Scoring | Calculated from score ($<25 \to \text{NORMAL}$, $25-49.9 \to \text{CAUTION}$, $50-74.9 \to \text{WARNING}$, $\ge 75 \to \text{HIGH\_RISK}$) | None | Behavioral alerts as triggered | L390-L751 |
| `HEALTHY` | `DEGRADED`, `MISSING`, or `INVALID` | Condition 4 | $\max(\text{Calculated State}, \text{CAUTION})$ | Enforces `CAUTION` floor if calculated state is `NORMAL` | `Maneuver Telemetry {h_b.capitalize()}` (`INFO` if missing, else `MEDIUM`) | L770-L782 |
| `DEGRADED`, `MISSING`, or `INVALID` | `HEALTHY` (Active Maneuver: $M \ne 0$) | Condition 3 | Calculated from Pipeline B active maneuver score | None (Option 1A-Pure; ordinary behavioral score preserved) | `Driving Style Telemetry {h_a.capitalize()}` (`INFO` if missing, else `MEDIUM`) | L757-L768 |
| `DEGRADED`, `MISSING`, or `INVALID` | `HEALTHY` (Cruising: $M = 0$) | Condition 2 | `INSUFFICIENT_DATA` (Score = 0.0) | Hard fail-safe pause | `Driving Style Telemetry {h_a.capitalize()}` (`INFO` if missing, else `MEDIUM`) | L359-L385 |
| `DEGRADED`, `MISSING`, or `INVALID` | `QUIESCENT_BYPASS` | Condition 2 | `INSUFFICIENT_DATA` (Score = 0.0) | Hard fail-safe pause | `Driving Style Telemetry {h_a.capitalize()}` (`INFO` if missing, else `MEDIUM`) | L359-L385 |
| `DEGRADED`, `MISSING`, `INVALID`, or `QUIESCENT_BYPASS` | `DEGRADED`, `MISSING`, or `INVALID` | Condition 1 | `INSUFFICIENT_DATA` (Score = 0.0) | Hard fail-safe pause | `Dual Pipeline Supervisory Failure` (`HIGH`) | L333-L355 |

### Unhandled Combinations
- **None unhandled:** The above conditions partition all $5 \times 5 = 25$ theoretical combinations of `(pipeline_a_health, pipeline_b_health)`.
- **Note on `h_a == "QUIESCENT_BYPASS"`:** In `prediction_validator.py:116`, Pipeline A defaults to `DrivingStyle.NORMAL` with confidence 1.0 when `pipeline_a_health == "QUIESCENT_BYPASS"`. Condition 1 explicitly catches `h_a == "QUIESCENT_BYPASS"` when $h_b$ is degraded/missing/invalid.

---

## 4. LIFECYCLE TRACKER (`ml/agent/lifecycle_tracker.py`)

- **States per Channel (`ManeuverContribution.is_excited`)** (`ml/agent/lifecycle_tracker.py:23-35`):
  1. `is_excited = True` (Active excitation: vehicle is physically executing abrupt acceleration, braking, or turning).
  2. `is_excited = False` (Decay: physical sensor excitation has ceased; contribution decays continuously according to $R(dt)$).
  3. Retired / Pruned (Channel deleted once $dt_{\text{decay}} > t_{\text{retire}}$).
- **State Transitions** (`ml/agent/lifecycle_tracker.py:109-221`):
  - Inactive $\to$ Active Excited: Triggered on new non-zero maneuver prediction (`maneuver_prediction in (1, 2, 3, 4)`). Sets `onset_timestamp = ts`, generates unique `event_id = EVT-{class}-{epoch_ms}-{counter:02d}`.
  - Active Excited $\to$ Intra-Event Continuation: Same class persists while excited. Preserves `onset_timestamp` ($t_0$), updates `last_active_timestamp = ts`, takes $\max(\text{confidence})$.
  - Active Excited $\to$ Decay: Sensor excitation ceases (`cls != current_active_class`). Locks `excitation_end_timestamp = last_active_timestamp`.
  - Decay $\to$ Re-Excitation (Intra-Incident): Maneuver of same class re-occurs before quiet hysteresis expires ($dt_{\text{quiet}} < 2.50\text{s}$). Retains original `event_id` and $t_0$, sets `is_excited = True`.
  - Decay $\to$ Separate Incident: Maneuver of same class occurs after quiet hysteresis ($dt_{\text{quiet}} \ge 2.50\text{s}$). Increments incident counter, generates new `event_id`, sets new $t_0$.
  - Decay $\to$ Retired: $dt_{\text{decay}} > t_{\text{retire}}$. Contribution removed from `_active_channels`.
- **Hysteresis Duration:**
  - `maneuver_rearm_seconds = 2.50s` (`ml/agent/config.py:150`; `ml/agent/lifecycle_tracker.py:107`).
- **Where Called From:**
  - Called from `TemporalSmoother.update()` at `ml/agent/temporal_smoother.py:207-212`:
    ```python
    active_contributions, recent_maneuver_count, recent_maneuver_history = self.lifecycle_manager.update(
        ts=ts,
        maneuver_prediction=m_pred,
        maneuver_confidence=m_conf,
        is_quiescent_bypass=is_quiescent
    )
    ```
- **Wired into SafetyAgent Output?**
  - **Internally Wired:** Yes. `active_contributions` is packed into `SmoothedContext.active_contributions` (`temporal_smoother.py:262`), which directly drives multi-channel residual decay risk scoring in `DeterministicRiskEngine.evaluate()` (`risk_engine.py:451-583`).
  - **Exposed on Public Dataclass:** **No.** `SafetyAssessment` (`ml/agent/safety_agent.py:42-61`) does not contain `active_contributions` or `incident_history` fields.

---

## 5. TEMPORAL SMOOTHER (`ml/agent/temporal_smoother.py`)

- **Real EMA Time Constant $\tau$:**
  - $\tau = 1.50\text{ s}$ (`ml/agent/config.py:105`: `style_ema_tau: float = 1.50`).
  - Discrete alpha formula: $\alpha = 1.0 - e^{-\Delta t / \tau}$ (`ml/agent/temporal_smoother.py:175`).
  - Default telemetry gap reset: $\Delta t > 5.0\text{ s} \implies \text{EMA} = \text{raw\_target}$ (`ml/agent/temporal_smoother.py:171-172`).
- **Aggressive-Ratio Window & Threshold:**
  - Rolling Window: `style_smoothing_window_seconds = 3.0 s` (`ml/agent/config.py:99`; `ml/agent/temporal_smoother.py:180`).
  - Ratio Threshold: `style_aggressive_ratio_threshold = 0.50` (50%) (`ml/agent/config.py:106`; `ml/agent/temporal_smoother.py:193`).
  - EMA Threshold: `style_ema >= 0.65` (`ml/agent/temporal_smoother.py:194`).
  - Sustained Aggressive Trigger: `style_aggressive_ratio >= 0.50 or style_ema >= 0.65` (`ml/agent/temporal_smoother.py:192-195`).
- **Minimum Samples:**
  - Ratio requires $\ge 1$ sample in 3.0s window; otherwise ratio is $0.0$ (`ml/agent/temporal_smoother.py:186-190`).
  - Initial EMA assigns first sample directly: `self._style_ema = raw_target` (`ml/agent/temporal_smoother.py:166`).
- **Maneuver Persistence / Decay / Expiry:**
  - `maneuver_persistence_seconds = 2.0 s` (`ml/agent/config.py:100`; `ml/agent/temporal_smoother.py:223`).
  - Plateau: `residual_plateau_seconds = 1.00 s` (`ml/agent/config.py:147`).
  - Half-life: `residual_half_life_seconds = 0.75 s` (`ml/agent/config.py:148`).
  - Retirement threshold: `residual_retirement_threshold = 2.50 pts` (`ml/agent/config.py:149`).
  - Re-arm hysteresis: `maneuver_rearm_seconds = 2.50 s` (`ml/agent/config.py:150`).
- **History Window:**
  - `history_window_seconds = 10.0 s` (`ml/agent/config.py:98`; `ml/agent/temporal_smoother.py:149`).
  - `maneuver_history_window_seconds = 10.0 s` (`ml/agent/config.py:101`; `ml/agent/lifecycle_tracker.py:223`).

---

## 6. MODELS

Verified read-only from `scripts/verify_final_system_integrity.py` and serialized artifacts:

### Pipeline A (Driving Style Binary Classifier)
- **Model Class:** `sklearn.svm._classes.SVC`
- **Hyperparameters:**
  ```python
  {'C': 2.0, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced',
   'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale',
   'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42,
   'shrinking': True, 'tol': 0.001, 'verbose': False}
  ```
- **Number of Input Features (`n_features_in_`):** `47`
- **Class Labels (`classes_`):** `[0, 1]` (`0: Normal Driving`, `1: Aggressive Driving`)
- **Probability Decision Threshold:** `0.84` (`prob_aggressive >= 0.84` triggers prediction = 1).
  - **What "0.84" actually is:** It is the tuned binary probability decision threshold (`threshold = 0.84`) in `ml/inference/physical_sensor_adapter.py:56, 126`. It is NOT a temporal time constant.
- **Sampling Rate:** `10.0 Hz` (`ml/preprocessing/pipeline_dataset_c.py:45`).
- **SHA-256 Checksums (`scripts/verify_final_system_integrity.py:11-12`):**
  - Model (`dataset_c_model`): `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc`
  - Scaler (`dataset_c_scaler`): `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5`

### Pipeline B (4-Class Maneuver Classifier)
- **Model Class:** `sklearn.ensemble._forest.ExtraTreesClassifier`
- **Hyperparameters:**
  ```python
  {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini',
   'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None,
   'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 3,
   'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 200,
   'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
  ```
- **Number of Input Features (`n_features_in_`):** `56`
- **Class Labels (`classes_`):** `[1, 2, 3, 4]` (`1: Sudden Acceleration`, `2: Sudden Deceleration`, `3: Sudden Right Turn`, `4: Sudden Left Turn`)
- **Probability Decision Threshold:** `argmax(predict_proba)` / default multi-class decision boundary.
- **Sampling Rate:** `1.0 Hz` nominal in Dataset B (`ml/preprocessing/pipeline_dataset_b.py:80`), evaluated over ~1.55 Hz sliding windows.
- **SHA-256 Checksums (`scripts/verify_final_system_integrity.py:9-10`):**
  - Model (`dataset_b_model`): `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90`
  - Scaler (`dataset_b_scaler`): `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576`

---

## 7. SAMPLING AND WARM-UP (`ml/inference/sampling_compatibility_adapter.py`)

- **Decimator Formula in Use (Default):**
  - Causal 2-tap moving average (`ml/inference/sampling_compatibility_adapter.py:60, 128-140`):
    $$y[k] = 0.5 \cdot x[2k] + 0.5 \cdot x[2k-1]$$
  - Group delay: $0.5\text{ samples} = 25.0\text{ ms}$ at 20 Hz.
- **Optional FIR Option & Flag:**
  - 21-tap causal windowed-sinc low-pass FIR filter (`ml/inference/sampling_compatibility_adapter.py:30-51, 62-64`):
    - Cutoff frequency: $f_c = 4.0\text{ Hz}$, sampling rate $f_s = 20.0\text{ Hz}$, linear phase.
    - Group delay: $10\text{ samples} = 500.0\text{ ms}$ at 20 Hz.
    - Stopband attenuation: $> 40\text{ dB}$. Unity DC gain ($\sum c = 1.0$).
  - Selector Flag: `filter_type: str = "2tap"` (pass `filter_type="fir_21tap"`).
  - Research status: `is_research_filter = True` when `filter_type != "2tap"`.
- **Warm-Up Sample Count:**
  - `50` samples counted at `10.0 Hz` (`ml/inference/physical_sensor_adapter.py:102-111`).
  - Equivalent incoming physical samples: `100` raw samples at `20.0 Hz`.
  - Warm-up duration: exactly `5.00 seconds`.
- **Feature Counts per Window:**
  - Production Pipeline A (Model A, Dataset C): `47` features (`ml/inference/physical_sensor_adapter.py:50-53`).
  - Production Pipeline B (Model B, Dataset B): `56` features (`ml/inference/dataset_b_classifier.py:61-62`).
  - Candidate V2 Pipeline A (ExtraTrees): `110` features in config, pruned to `102` features (`ml/inference/physical_sensor_adapter.py:58-68`).

---

## 8. AXES

### Actual Axis Conventions

| Subsystem / Location | File & Line | `Acc X` | `Acc Y` | `Acc Z` | `gyro_x` | `gyro_y` | `gyro_z` |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Android Logger** | `mobile/.../SensorLoggingService.kt:178-187` | Lateral (screen right $+$) | Longitudinal (screen top $+$) | Vertical (screen front $+$) | Pitch (rotation around X) | Roll (rotation around Y) | Yaw (rotation around Z) |
| **Web Logger** | `mobile/web-logger/app.js:50-64` | `acc.x` (DeviceMotion) | `acc.y` (DeviceMotion) | `acc.z` (DeviceMotion) | `beta` $\to$ Pitch | `gamma` $\to$ Roll | `alpha` $\to$ Yaw |
| **CSV Telemetry Schema** | `reports/.../physical_sensor_integration_architecture.md:136-141` | Lateral (left $+$, right $-$) | Longitudinal (fwd $+$, brake $-$) | Vertical (down $+$, up $-$) | Pitch rate ($\text{rad/s}$) | Roll rate ($\text{rad/s}$) | Yaw rate ($\text{rad/s}$) |
| **Dataset B Preprocessing** | `ml/preprocessing/pipeline_dataset_b.py:6, 178-180` | MPU6050 dashboard mount (front face upward) | MPU6050 dashboard mount | MPU6050 dashboard mount | GyroX (rad/s) | GyroY (rad/s) | GyroZ (rad/s) |
| **Dataset C Preprocessing** | `ml/preprocessing/pipeline_dataset_c.py:124-127, 153-155` | Planar Horizontal $\sqrt{X^2+Y^2}$ | Planar Horizontal $\sqrt{X^2+Y^2}$ | Vertical normal | Angular velocity | Angular velocity | Yaw rate dynamics |
| **Frontend Live HUD** | `frontend/.../LiveMonitoring.jsx:864-895` | `Acc X (Fwd)` [Longitudinal] | `Acc Y (Lat)` [Lateral] | `Acc Z (Vert)` [Vertical] | `Roll Rate` | `Pitch Rate` | `Yaw Rate` |
| **Frontend Sensor View** | `frontend/.../VehicleSensorsView.jsx:103-123` | `Acc Longitudinal (X)` | `Acc Lateral (Y)` | `Acc Vertical (Z)` | Roll rate | Pitch rate | `Angular Velocity (Yaw)` |
| **SAE J670 Standard** | `ml/inference/vehicle_frame_alignment.py:205-211` | Forward ($u_x$) | Right / Lateral ($u_y$) | Down / Vertical ($u_z$) | Roll ($\omega_x$) | Pitch ($\omega_y$) | Yaw ($\omega_z$) |

### Axis Disagreements Flagged
- **FLAG 1 (Longitudinal vs Lateral Inversion):** Android logger, web logger, and CSV schema map `Acc X = Lateral` and `Acc Y = Longitudinal`. In contrast, frontend gauges (`LiveMonitoring.jsx:864`, `VehicleSensorsView.jsx:103`) and standard SAE J670 automotive coordinates map `Acc X = Longitudinal (Forward)` and `Acc Y = Lateral`.
- **FLAG 2 (Gyroscope Axis Swapping):** Android and web loggers map `gyro_x = Pitch` and `gyro_y = Roll`. The frontend (`LiveMonitoring.jsx:888-895`) labels `gyroX = Roll Rate` and `gyroY = Pitch Rate`.

---

## 9. DATA QUALITY (`backend/app/services/data_quality_service.py`)

All 6 quality gates with exact numeric limits:

1. **Criterion 6: Missing Values & Data Hygiene** (`backend/app/services/data_quality_service.py:53-71`):
   - `nan_count == 0` (Exactly 0 NaNs).
   - `inf_count == 0` (Exactly 0 Infs).
   - Status: `FAIL` if `nan_count > 0 or inf_count > 0`, else `PASS`.
2. **Criterion 1: Sampling Frequency & Jitter** (`backend/app/services/data_quality_service.py:76-127`):
   - Monotonicity: zero backwards timestamp steps (`backwards == 0`). If $> 0 \to \text{FAIL}$.
   - Duplicates: `duplicates <= 0.01 * total_samples` ($\le 1.0\%$). If $> 1.0\% \to \text{FAIL}$.
   - Max gap: zero gaps $\ge 250.0\text{ ms}$ (`dt >= 0.250`). If $\ge 1 \to \text{FAIL}$.
   - Nominal frequency band: $[18.0, 22.0]\text{ Hz}$ (corresponds to nominal period $[45.45, 55.56]\text{ ms}$). If outside $\to \text{WARNING}$.
   - Inter-sample jitter window: $\ge 95.0\%$ of intervals must fall in $[40.0, 60.0]\text{ ms}$ (`(dt >= 0.040) & (dt <= 0.060)`). If $< 95.0\% \to \text{WARNING}$.
   - Status: `PASS` if no FAIL or WARNING conditions triggered.
3. **Criterion 2: Dynamic Range & Saturation** (`backend/app/services/data_quality_service.py:132-165`):
   - Full scale range: zero clipped samples where $|a_x| \ge 3.92g$ or $|a_y| \ge 3.92g$ or $|a_z| \ge 3.92g$ ($98\%$ of $\pm 4.0g$).
   - Gyroscope saturation: zero samples with $\text{gyro\_mag} \ge 8.5\text{ rad/s}$ ($487\text{ deg/s}$).
   - Dead channel check: $\text{std}(a_x) + \text{std}(a_y) + \text{std}(a_z) \ge 0.001g$ if session duration $> 10.0\text{ s}$.
   - Status: `FAIL` if clipped $> 0$ or dead channel detected, else `PASS`.
4. **Criterion 3: Static Gravity & Mount Stability** (`backend/app/services/data_quality_service.py:170-211`):
   - Initial Static Gravity Magnitude: $|g_0| \in [0.98, 1.02]g$ ($9.81 \pm 0.20\text{ m/s}^2$) evaluated on segment `S0` (or first 30s / 600 samples). If outside $\to \text{FAIL}$.
   - Post-drive mount drift: angular deviation $\theta_{\text{drift}} \le 3.0^\circ$ between S0 and S6. If $> 3.0^\circ \to \text{FAIL}$.
   - Status: `PASS` if both pass.
5. **Criterion 4: GPS Ground Speed & Satellite Fix** (`backend/app/services/data_quality_service.py:216-262`):
   - Speed non-negativity: zero samples with $\text{speed\_mps} < 0.0$. If $> 0 \to \text{FAIL}$.
   - GPS 3D fix coverage: $\ge 92.0\%$ of session with valid coordinates ($\text{lat} \ne 0.0, \text{lon} \ne 0.0$). If $< 92.0\% \to \text{FAIL}$.
   - Satellite geometry: mean $\text{HDOP} \le 2.5$. If $> 2.5 \to \text{WARNING}$.
   - Status: `PASS` if all met; `NOT_AVAILABLE` (with WARNING) if `speed_mps` channel missing.
6. **Criterion 5: Driving Regime Coverage & Highway Anti-Confounding** (`backend/app/services/data_quality_service.py:267-297`):
   - Total session duration: $\ge 720.0\text{ s}$ ($12.0\text{ min}$). If $< 720.0\text{ s} \to \text{WARNING}$.
   - Highway normal cruising duration ($\text{speed} \ge 60.0\text{ km/h}$): $\ge 150.0\text{ s}$ ($2.5\text{ min}$). If $< 150.0\text{ s} \to \text{WARNING}$.
   - Status: `PASS` if both met.

---

## 10. BACKEND API

All routes are mounted under `/api` in `backend/app/main.py:33-34`.

### General Telemetry & Agent Endpoints (`backend/app/api/endpoints.py`)

| Method | Path | Purpose | Line Reference |
|:---|:---|:---|:---|
| `GET` | `/api/health` | Backend API health check | `endpoints.py:38` |
| `GET` | `/api/models/status` | Verify ML model artifacts, existence, and SHA-256 hashes | `endpoints.py:44` |
| `POST` | `/api/telemetry/assess` | Ingest multi-modal predictions, validate, smooth, score risk | `endpoints.py:50` |
| `POST` | `/api/telemetry/assess/raw` | Ingest raw feature vectors, run Model A/B inference, score risk | `endpoints.py:65` |
| `GET` | `/api/simulated/scenarios` | List demonstration scenarios | `endpoints.py:80` |
| `POST` | `/api/agent/reset` | Reset temporal smoothing buffers and incident timers | `endpoints.py:86` |
| `GET` | `/api/replay/sessions` | List recorded driving sessions | `endpoints.py:93` |
| `GET` | `/api/replay/sessions/{session_id}` | Get telemetry detail for recorded replay session | `endpoints.py:102` |
| `GET` | `/api/analytics/overview` | Get verified evaluation and benchmark analytics | `endpoints.py:111` |
| `GET` | `/api/analytics/unified-registry` | Get multi-model registry and performance benchmarks | `endpoints.py:117` |
| `GET` | `/api/system/hardware-status` | Inspect live physical sensor hardware | `endpoints.py:126` |

### Physical Sensor Integration Endpoints (`backend/app/api/sensor_endpoints.py`)

| Method | Path | Purpose | Line Reference |
|:---|:---|:---|:---|
| `POST` | `/api/sensor/device/register` | Register physical smartphone device token | `sensor_endpoints.py:46` |
| `POST` | `/api/sensor/session/start` | Initialize physical telemetry recording session | `sensor_endpoints.py:62` |
| `POST` | `/api/sensor/telemetry/frame` | Ingest single real-time 20 Hz telemetry frame | `sensor_endpoints.py:89` |
| `POST` | `/api/sensor/upload` | Upload completed raw telemetry CSV (validates 6 gates) | `sensor_endpoints.py:102` |
| `GET` | `/api/sensor/sessions` | List all ingested physical driving sessions | `sensor_endpoints.py:175` |
| `GET` | `/api/sensor/sessions/{driver_id}/{session_id}/quality` | Get 6-criterion Data Quality Report for session | `sensor_endpoints.py:181` |
| `GET` | `/api/sensor/live/status` | Get live device connection status and measured frequency | `sensor_endpoints.py:212` |
| `GET` | `/api/sensor/live/frames` | Get rolling telemetry frames for frontend charts | `sensor_endpoints.py:221` |
| `POST` | `/api/sensor/annotations` | Record human ground-truth annotation | `sensor_endpoints.py:230` |
| `GET` | `/api/sensor/annotations/{session_id}` | Get annotations for session | `sensor_endpoints.py:242` |
| `WS` | `/api/sensor/live/stream` | Bi-directional WebSocket streaming telemetry channel | `sensor_endpoints.py:251` |

### Storage, Upload Limits & Error Handling
- `MAX_UPLOAD_BYTES = 50 * 1024 * 1024` (50 MB) (`sensor_endpoints.py:117`). Exceeding returns `HTTP 413 Payload Too Large`.
- **Driver Whitelist Logic (`backend/app/services/sensor_storage_service.py:32-33, 75-86`):**
  - Whitelist: `DEVELOPMENT_DRIVERS = {"DRV_001", "DRV_002", "DRV_003", "DRV_004", "DRV_005", "DRV_006"}`.
  - Sequestered Lockbox: `RESERVED_LOCKBOX_DRIVERS = {"DRV_007", "DRV_008"}`.
- **Error Codes:**
  - `HTTP 403 Forbidden`: Attempting to access lockbox driver (`DRV_007`, `DRV_008`) or path traversal in `session_id` (`sensor_endpoints.py:57, 84, 166, 193`).
  - `HTTP 400 Bad Request`: Driver not in whitelist or schema validation failure (`sensor_endpoints.py:59, 86, 99, 168, 195, 239`).
  - `HTTP 409 Conflict`: File already exists or duplicate session collision (`sensor_endpoints.py:170`).

---

## 11. FRONTEND

- **Stream Rate Received:**
  - Simulation Stream: `4.0 Hz` (`250 ms` timer: `frontend/src/components/live/LiveMonitoring.jsx:148`).
  - Live Physical WebSocket: Telemetry emitted at `20.0 Hz` (`50 ms` periods from mobile).
- **Rate at which React State Updates:**
  - WebSocket frames are buffered in `wsFrameBufferRef` and flushed to React state at `10.0 Hz` (`100 ms` interval) (`frontend/src/components/live/LiveMonitoring.jsx:82-94`).
- **Risk Gauge Color Bands & Thresholds (`frontend/src/components/common/RiskGauge.jsx:15-24`):**
  - $\ge 75.0$ or `state === 'HIGH_RISK'` / `'CRITICAL'`: Red (`#ef4444`).
  - $\ge 50.0$ or `state === 'WARNING'`: Orange (`#f97316`).
  - $\ge 25.0$ or `state === 'CAUTION'`: Amber (`#eab308`).
  - $< 25.0$ (`state === 'NORMAL'`): Black / White (`currentColor`).
- **How `INSUFFICIENT_DATA` Renders (`frontend/src/components/common/RiskGauge.jsx:5-20, 65-70`):**
  - Triggered if `state === 'INSUFFICIENT_DATA'` or `score === null` or `score === undefined`.
  - Center readout renders `"--"` (never `0.0`).
  - Label text renders `"No Telemetry"` (instead of `"Risk Index"`).
  - Active telemetry ring is de-energized (`strokeDashoffset = arcLength`, neutral styling `text-neutral-400 dark:text-neutral-600 opacity-30`).
- **List of Pages / Routes (`frontend/src/App.jsx:176-218`):**
  1. `'dashboard'` $\to$ `<CommandCenter />`
  2. `'live_monitoring'` $\to$ `<LiveMonitoring />`
  3. `'driving_behaviour'` $\to$ `<Vehicle3DView />`
  4. `'safety_alerts'` $\to$ `<SafetyIntelligence />`
  5. `'trip_history'` $\to$ `<ReplayStudio />`
  6. `'driving_insights'` $\to$ `<DrivingAnalytics />`
  7. `'settings'` $\to$ `<SystemDiagnostics />`
- **Verification Strings in Code:**
  - `LiveMonitoring.jsx:427`: `Physical Hardware State: {isPhysConnected ? 'VERIFIED_ACTIVE' : 'NOT_VERIFIED'}`.
  - `App.jsx:46`: `explanation: '... Telemetry nominal (Hardware: NOT_VERIFIED until on-road test).'`.
  - `App.jsx:229`: `Models Verified [4/4]`.
  - `SystemDiagnostics.jsx:480, 483, 512`: `Verified Safety Models Manifest`, `SHA-256 Integrity Verified`, `Cryptographically Verified`.
- **CAN / OBD / USB References in Code:**
  - `backend/app/api/endpoints.py:146`: `"CAN Bus 2.0B / OBD-II (Optional auxiliary link)"`.
  - `frontend/src/components/views/VehicleSensorsView.jsx:77`: `6-DOF IMU Kinematic Signals, High-Rate Polling Channels & Bus Topology`.
  - **Code Reality:** No CAN-bus, OBD-II, or USB IMU drivers exist in the implementation. Telemetry ingestion operates exclusively over HTTP and WebSocket from mobile/web sensor loggers.

---

## 12. REPO FACTS

### Git Status & Branch
- **Active Branch:** `stage6-phase1`
- **Uncommitted Changed Files (`git status --short`):**
  - `M README.md`
  - `M docs/architecture.md`
  - `M ml/agent/config.py`
  - `M ml/agent/prediction_validator.py`
  - `M ml/agent/risk_engine.py`
  - `M ml/agent/safety_agent.py`
  - `M ml/agent/temporal_smoother.py`
  - `M reports/safety_agent/phase1b_two_layer_validator_design_note.md`
  - `M reports/safety_agent/residual_risk_integration_design.md`
  - `M reports/safety_agent/stage6_architecture.md`
  - `M tests/test_safety_agent.py`
- **Untracked Additions (Stage 6 / 7 Phase 1):**
  - `DESIGN.md`, `docs/DESIGN.md`, `backend/app/`, `frontend/`, `ml/agent/lifecycle_tracker.py`, `ml/inference/physical_sensor_adapter.py`, `ml/inference/sampling_compatibility_adapter.py`, `ml/inference/vehicle_frame_alignment.py`, `mobile/`, `tests/test_phase3_reconciliation.py`, `tests/test_sensor_integration.py`, etc.

### Real Pytest Count
- **Passed:** `224`
- **Failed:** `0`
- **Warnings:** `154`
- **Total Test Count:** `224` (in 33.50s)

### Directory Tree to Depth 3 (Excluding `node_modules`, `dist`, `.git`, `DRV_007`, `DRV_008`)

```text
driver-monitoring-agent/
  ├── backend/
    ├── app/
      ├── agents/
      ├── api/
      ├── core/
      ├── database/
      ├── models/
      ├── schemas/
      ├── services/
      ├── static/
      └── __init__.py
      └── main.py
    └── requirements.txt
  ├── dataset/
    ├── annotations/
    ├── derived/
      ├── eight_driver_campaign/
    ├── experiments/
      ├── pipeline_a/
    ├── metadata/
    ├── processed/
    ├── raw/
      ├── 9vr83n7z5j-2/
      ├── dataset_02_jj3tw8kj6h/
      ├── dataset_03_5stn873wft/
      ├── dataset_04_eishkaran/
      ├── dataset_05_shashwatwork/
      ├── dataset_06_yotamkatz/
      ├── dataset_07_vitorrf/
      ├── daywise/
      ├── drivers/
      ├── eight_driver_campaign/
      └── DOWNLOAD_MANIFEST.md
    └── .gitkeep
  ├── docs/
    └── architecture.md
    └── DATASET_LICENSES_AND_ATTRIBUTION.md
    └── DESIGN.md
  ├── frontend/
    ├── src/
      ├── components/
      ├── lib/
      ├── services/
      └── App.jsx
      └── index.css
      └── main.jsx
    └── .gitkeep
    └── index.html
    └── package-lock.json
    └── package.json
    └── tailwind.config.js
    └── vite.config.js
  ├── ml/
    ├── agent/
      └── __init__.py
      └── config.py
      └── lifecycle_tracker.py
      └── prediction_validator.py
      └── risk_engine.py
      └── safety_agent.py
      └── temporal_smoother.py
    ├── evaluation/
      └── __init__.py
      └── benchmark_stage5.py
      └── run_streaming_evaluation.py
      └── streaming_testbed.py
    ├── inference/
      └── __init__.py
      └── dataset_b_classifier.py
      └── dataset_c_classifier.py
      └── physical_sensor_adapter.py
      └── sampling_compatibility_adapter.py
      └── streaming_pipeline_a_features.py
      └── vehicle_frame_alignment.py
    ├── notebooks/
      └── 01_dataset_inspection.ipynb
      └── 02_exploratory_data_analysis.ipynb
    ├── preprocessing/
      └── __init__.py
      └── pipeline_dataset_a.py
      └── pipeline_dataset_b.py
      └── pipeline_dataset_c.py
      └── ...
    ├── saved_models/
      └── dataset_c_binary_model_v1.0.0.joblib
      └── dataset_c_config_v1.0.0.json
      └── dataset_c_extratrees_v1.0.0.joblib
      └── dataset_c_knn_v1.0.0.joblib
      └── dataset_c_metrics_v1.0.0.json
      └── dataset_c_rf_v1.0.0.joblib
      └── dataset_c_scaler_v1.0.0.joblib
      └── dataset_c_svm_v1.0.0.joblib
      └── ...
  ├── mobile/
    ├── android/
      ├── app/
      ├── gradle/
      └── build.gradle
    └── web-logger/
      └── app.js
      └── index.html
      └── style.css
  ├── models/
    ├── dataset_b_model.joblib
    ├── dataset_b_scaler.joblib
    ├── dataset_c_binary_model.joblib
    ├── dataset_c_scaler.joblib
    └── experiments/
      ├── dataset_b/
      ├── dataset_c/
      └── pipeline_a/
  ├── notebooks/
    ├── Driver_Activity_EDA.ipynb
    ├── Driver_Activity_ML_Models.ipynb
    ├── README.md
    └── outputs/
  ├── reports/
    ├── eda/
    ├── frontend/
    ├── model_performance_audit/
    ├── pipeline_a/
    ├── preprocessing/
    ├── safety_agent/
    ├── stage7/
    └── design_facts.md
  ├── scripts/
    └── advanced_pipeline_a_research.py
    └── audit_phase1_baseline.py
    └── audit_pipeline_a_data.py
    └── build_eda_notebook.py
    └── build_ml_notebook.py
    └── canonical_pipeline_a_evaluator.py
    └── check_agent_dataset_c.py
    └── check_dataset_c.py
    └── diagnose_reverse_session_errors.py
    └── engineer_pipeline_a_features.py
    └── independent_audit_pipeline_a.py
    └── package_pipeline_a_candidate.py
    └── phase1_bottleneck_investigation.py
    └── run_controlled_robustness_experiments.py
    └── run_model_experiments.py
    └── run_pipeline_a_improvement_experiments.py
    └── train_pipeline_a_improvements.py
    └── verify_final_system_integrity.py
    └── verify_research_report.py
  ├── tests/
    └── __init__.py
    └── test_backend_api.py
    └── test_dataset_b_pipeline.py
    └── test_dataset_c_binary_pipeline.py
    └── test_model_training_and_inference.py
    └── test_phase3_reconciliation.py
    └── test_pipeline_a_improvements.py
    └── test_safety_agent.py
    └── test_sensor_integration.py
    └── test_streaming_pipeline_a_features.py
    └── test_streaming_testbed.py
  ├── DESIGN.md
  └── README.md
```

---

## 13. UNIMPLEMENTED

The following concepts described in legacy design reports or research proposals are NOT implemented in the executable code:

1. **Vehicle Speed Multipliers:**
   - *Doc Claim:* Risk scores scaled by $1.25\times$ for $v \ge 80\text{ km/h}$ and $1.50\times$ for $v \ge 100\text{ km/h}$.
   - *Code Truth:* `NOT FOUND`. No speed multiplier logic exists in `DeterministicRiskEngine`, `TemporalSmoother`, or `SafetyAgentConfig`. `SmoothedContext` does not have a speed field.
2. **Leave-One-Driver-Out (LODO) Evaluation on Production Models:**
   - *Doc Claim:* Production Model A validated via cross-driver leave-one-out partitions.
   - *Code Truth:* `NOT FOUND`. Production Model A (Dataset C) was derived from 2 sequential trips recorded by a single subject; LODO experiments exist only as research scripts (`lodo_benchmark_results.json`) on unpromoted models.
3. **Coordinate Frame Alignment in Production Inference Path:**
   - *Doc Claim:* Automated dynamic phone-to-vehicle body alignment applied to streaming physical data.
   - *Code Truth:* `NOT FOUND in production path`. `VehicleFrameAligner` (`ml/inference/vehicle_frame_alignment.py`) is implemented as an opt-in research module, but is not called within `PhysicalSensorInferenceAdapter.process_raw_20hz_sample()` or `IntelligentSafetyAgent.process_features()`.
4. **Pipeline B False-Alarm Rate Measurement on Continuous Driving:**
   - *Doc Claim:* False alarm rate of maneuver classification measured during continuous cruising.
   - *Code Truth:* `NOT FOUND`. Dataset B contains only 109 isolated positive maneuver trials with no continuous negative cruising sessions.
5. **CAN-Bus / OBD-II / USB IMU Hardware Drivers:**
   - *Doc Claim:* Direct vehicle bus hardware polling.
   - *Code Truth:* `NOT FOUND`. No serial, CAN-bus, or OBD-II hardware drivers exist; hardware status relies entirely on mobile WebSocket/HTTP telemetry.
6. **SafetyState `CRITICAL`:**
   - *Doc Claim:* `CRITICAL` is the fourth state boundary (80–100).
   - *Code Truth:* `NOT FOUND`. The enum is `SafetyState.HIGH_RISK` (`ml/agent/config.py:18`). `CRITICAL` exists only as an `AlertSeverity`.

---

## 14. UNVERIFIED FACTS

1. **Physical On-Road Behavior on Actual Vehicles:**  
   The code interfaces and mathematics of the 20 Hz ingestion pipeline, 6 quality gates, causal FIR decimator, and vehicle-frame aligner were verified via automated unit and integration tests (`test_sensor_integration.py`, `test_phase3_reconciliation.py`). Live on-road vehicular dynamics could not be physically executed in this development environment and remain honestly marked as `NOT_VERIFIED until on-road test`.
2. **Blind Lockbox Partitions `DRV_007` and `DRV_008`:**  
   In strict compliance with lockbox isolation constraints, the contents, directories, and files of `DRV_007` and `DRV_008` were not accessed, inspected, or verified.

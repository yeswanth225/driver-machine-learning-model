"""
Deterministic Safety Rule Engine & Explainable Risk Scoring Module
===================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

import math
import numbers
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    SafetyState,
    DrivingStyle,
    ManeuverClass,
    AlertSeverity,
    AlertCategory,
    STYLE_LABEL_NAMES,
    MANEUVER_LABEL_NAMES
)
from ml.agent.temporal_smoother import SmoothedContext


@dataclass
class SafetyAlert:
    """Structured, actionable safety alert."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    category: AlertCategory
    title: str
    triggering_condition: str
    explanation: str
    related_predictions: Dict[str, Any]
    safe_driving_advisory: str

    @property
    def message(self) -> str:
        """Additive compatibility property returning self.explanation."""
        return self.explanation


@dataclass
class RuleEvaluationResult:
    """Result of deterministic rule engine evaluation."""
    timestamp: float
    safety_state: SafetyState
    risk_score: float  # Bounded [0.0, 100.0]
    raw_score: float
    confidence_factor: float
    triggered_rules: List[str]
    rule_explanations: List[str]
    overall_explanation: str
    active_alerts: List[SafetyAlert] = field(default_factory=list)


class DeterministicRiskEngine:
    """
    Evaluates transparent, configurable engineering rules to produce an explainable
    driving risk score (0 to 100) and discrete safety state.
    """

    def __init__(self, config: SafetyAgentConfig = DEFAULT_CONFIG):
        self.config = config
        self._last_alert_timestamps: Dict[AlertCategory, float] = {}
        self._last_alert_severities: Dict[AlertCategory, AlertSeverity] = {}
        self._alert_counter: int = 0

    def reset(self):
        """Reset internal alert cooldown timers and counters."""
        self._last_alert_timestamps.clear()
        self._last_alert_severities.clear()
        self._alert_counter = 0

    @staticmethod
    def calculate_residual_risk(
        initial_weight: float,
        elapsed_seconds: float,
        plateau_seconds: float = 1.00,
        half_life_seconds: float = 0.75,
        retirement_threshold: float = 2.50
    ) -> float:
        """
        Calculate continuous exponential residual-risk decay following a plateau.
        
        Specification:
            R(dt) = W * 2^(-(dt - plateau_seconds) / half_life_seconds)
            
        Analytical Retirement Boundary:
            t_retire = plateau_seconds + half_life_seconds * log2(initial_weight / retirement_threshold)
            
        Retirement Policy (strictly analytical without tolerance/epsilon):
            - dt <= plateau_seconds: Full initial weight W is retained.
            - plateau_seconds < dt < t_retire: Positive residual R(dt) is retained.
            - dt == t_retire: Exactly retirement_threshold (2.50) is retained.
            - dt > t_retire: Retired strictly to 0.00 pts.
            
        Parameter Validation:
            - plateau_seconds, half_life_seconds, retirement_threshold must be finite and > 0.0.
            - initial_weight must be finite and > 0.0. If initial_weight < retirement_threshold, returns 0.0.
            - elapsed_seconds must be finite. Negative elapsed_seconds safely returns initial_weight (within plateau).
            - Invalid, non-finite, or boolean parameters safely return 0.0.
        """
        params = (initial_weight, elapsed_seconds, plateau_seconds, half_life_seconds, retirement_threshold)
        if not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in params):
            return 0.0
        if not all(math.isfinite(p) for p in params):
            return 0.0
        if plateau_seconds <= 0.0 or half_life_seconds <= 0.0 or retirement_threshold <= 0.0:
            return 0.0
        if initial_weight <= 0.0 or initial_weight < retirement_threshold:
            return 0.0

        # Plateau: retain full initial weight for elapsed time <= plateau_seconds (including negative dt)
        if elapsed_seconds <= plateau_seconds:
            return float(initial_weight)

        # Analytical retirement boundary
        t_retire = float(plateau_seconds) + float(half_life_seconds) * math.log2(
            float(initial_weight) / float(retirement_threshold)
        )

        # Strict boundary evaluation
        if elapsed_seconds > t_retire:
            return 0.0
        if elapsed_seconds == t_retire:
            return float(retirement_threshold)

        # Elapsed time is strictly below analytical boundary: compute exponential decay
        decay_time = float(elapsed_seconds) - float(plateau_seconds)
        decay_factor = 2.0 ** (-decay_time / float(half_life_seconds))
        residual = float(initial_weight) * decay_factor
        return float(residual)

    @staticmethod
    def calculate_confidence_scaling(confidence: Optional[float]) -> float:
        """
        D-10 Authoritative Modality Confidence Scaling Operator.
        
        Specification:
            phi(C) = 0.0          for C < 0.40
            phi(C) = C / 0.60     for 0.40 <= C < 0.60
            phi(C) = 1.0          for C >= 0.60
            
        Defense-in-depth:
            - None or boolean returns 0.0.
            - Non-numeric or non-finite (NaN, +/-inf) returns 0.0.
            - Out-of-bounds (C < 0.0 or C > 1.0) returns 0.0.
            - Supports Python float/int and NumPy numeric types (np.floating, np.integer).
        """
        if confidence is None or isinstance(confidence, bool):
            return 0.0
        if not isinstance(confidence, numbers.Real) or not math.isfinite(confidence):
            return 0.0
        c = float(confidence)
        if c < 0.0 or c > 1.0:
            return 0.0
        if c < 0.40:
            return 0.0
        if c < 0.60:
            return float(c / 0.60)
        return 1.0

    def _should_suppress_alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        timestamp: float,
        cooldown_seconds: float
    ) -> bool:
        """
        Check if an alert should be suppressed by cooldown.
        Bypasses cooldown if severity escalates to HIGH or CRITICAL.
        """
        if category not in self._last_alert_timestamps:
            return False

        last_ts = self._last_alert_timestamps[category]
        last_sev = self._last_alert_severities.get(category, AlertSeverity.INFO)
        elapsed = timestamp - last_ts

        # Severity escalation bypass: never suppress escalation to HIGH or CRITICAL
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        if severity_order.get(severity, 0) > severity_order.get(last_sev, 0):
            return False

        return elapsed < cooldown_seconds

    def _record_alert(self, alert: SafetyAlert):
        """Record alert emission for cooldown tracking."""
        self._last_alert_timestamps[alert.category] = alert.timestamp
        self._last_alert_severities[alert.category] = alert.severity

    def evaluate(self, ctx: SmoothedContext) -> RuleEvaluationResult:
        """
        Evaluate smoothed context against deterministic safety rules and D-13 fail-safe table.
        Produces explainable risk score, safety state, and deduplicated alerts.
        """
        ts = ctx.timestamp
        h_a = getattr(ctx, "pipeline_a_health", "HEALTHY")
        h_b = getattr(ctx, "pipeline_b_health", "HEALTHY")

        # -------------------------------------------------------------
        # Rule 1: Layer 1 Transport & Packet Integrity Guard
        # -------------------------------------------------------------
        if not ctx.is_valid:
            explanation = (
                f"Safety monitoring paused: Telemetry data quality is {ctx.data_quality_status}. "
                "Predictions are unavailable, invalid, or stale. Sensor integrity check required."
            )
            alerts = []
            if not self._should_suppress_alert(
                AlertCategory.DATA_QUALITY,
                AlertSeverity.HIGH,
                ts,
                self.config.alert_cooldown_data_quality
            ):
                self._alert_counter += 1
                alert = SafetyAlert(
                    alert_id=f"ALT-DATA-{int(ts * 1000)}-{self._alert_counter:04d}",
                    timestamp=ts,
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.DATA_QUALITY,
                    title="Sensor Telemetry Disruption",
                    triggering_condition=f"data_quality_status == '{ctx.data_quality_status}'",
                    explanation=explanation,
                    related_predictions={
                        "data_quality": ctx.data_quality_status,
                        "timestamp": ts
                    },
                    safe_driving_advisory="Drive cautiously. Telemetry monitoring is temporarily disrupted."
                )
                alerts.append(alert)
                self._record_alert(alert)

            return RuleEvaluationResult(
                timestamp=ts,
                safety_state=SafetyState.INSUFFICIENT_DATA,
                risk_score=0.0,
                raw_score=0.0,
                confidence_factor=0.0,
                triggered_rules=["DATA_QUALITY_ANOMALY"],
                rule_explanations=[explanation],
                overall_explanation=explanation,
                active_alerts=alerts
            )

        alerts: List[SafetyAlert] = []

        def _emit_dq_alert(title: str, condition: str, explanation: str, severity: AlertSeverity = AlertSeverity.MEDIUM):
            if not self._should_suppress_alert(
                AlertCategory.DATA_QUALITY,
                severity,
                ts,
                self.config.alert_cooldown_data_quality
            ):
                self._alert_counter += 1
                dq_alert = SafetyAlert(
                    alert_id=f"ALT-DATA-{int(ts * 1000)}-{self._alert_counter:04d}",
                    timestamp=ts,
                    severity=severity,
                    category=AlertCategory.DATA_QUALITY,
                    title=title,
                    triggering_condition=condition,
                    explanation=explanation,
                    related_predictions={
                        "pipeline_a_health": h_a,
                        "pipeline_b_health": h_b,
                        "timestamp": ts
                    },
                    safe_driving_advisory="Drive with heightened caution. Monitoring subsystem impaired."
                )
                alerts.append(dq_alert)
                self._record_alert(dq_alert)

        # -------------------------------------------------------------
        # Condition 1: Dual Pipeline Impairment (Rows 7, 8, 9, 12, 13, 14, 17, 18, 19)
        # -------------------------------------------------------------
        if h_a in ("DEGRADED", "MISSING", "INVALID") and h_b in ("DEGRADED", "MISSING", "INVALID"):
            explanation = (
                f"Dual pipeline failure: Driving style is {h_a} and maneuver detection is {h_b}. "
                "System cannot certify driving safety."
            )
            _emit_dq_alert(
                title="Dual Pipeline Supervisory Failure",
                condition=f"pipeline_a_health == '{h_a}' and pipeline_b_health == '{h_b}'",
                explanation=explanation,
                severity=AlertSeverity.HIGH
            )
            return RuleEvaluationResult(
                timestamp=ts,
                safety_state=SafetyState.INSUFFICIENT_DATA,
                risk_score=0.0,
                raw_score=0.0,
                confidence_factor=0.0,
                triggered_rules=["DUAL_PIPELINE_OUTAGE"],
                rule_explanations=[explanation],
                overall_explanation=explanation,
                active_alerts=alerts
            )

        # -------------------------------------------------------------
        # Condition 2: Cruising during Pipeline A Outage (Rows 6, 10, 11, 15, 16, 20 with M=0)
        # -------------------------------------------------------------
        is_cruising_during_a_outage = (
            h_a in ("DEGRADED", "MISSING", "INVALID") and
            (h_b == "QUIESCENT_BYPASS" or (h_b == "HEALTHY" and not ctx.is_maneuver_active))
        )
        if is_cruising_during_a_outage:
            sev = AlertSeverity.INFO if h_a == "MISSING" else AlertSeverity.MEDIUM
            explanation = (
                f"Driving style pipeline is {h_a}. Vehicle is cruising, but driving state cannot be "
                "certified safe without continuous driving style monitoring."
            )
            _emit_dq_alert(
                title=f"Driving Style Telemetry {h_a.capitalize()}",
                condition=f"pipeline_a_health == '{h_a}' and not is_maneuver_active",
                explanation=explanation,
                severity=sev
            )
            return RuleEvaluationResult(
                timestamp=ts,
                safety_state=SafetyState.INSUFFICIENT_DATA,
                risk_score=0.0,
                raw_score=0.0,
                confidence_factor=0.0,
                triggered_rules=["STYLE_PIPELINE_OUTAGE"],
                rule_explanations=[explanation],
                overall_explanation=explanation,
                active_alerts=alerts
            )

        # -------------------------------------------------------------
        # Behavioral Risk Scoring from Viable Modalities
        # -------------------------------------------------------------
        raw_score = 0.0
        triggered_rules: List[str] = []
        rule_explanations: List[str] = []

        # D-10: Independent Modality Confidence Scaling
        phi_style = (
            self.calculate_confidence_scaling(ctx.current_style_confidence)
            if h_a == "HEALTHY" else 0.0
        )
        phi_maneuver = (
            self.calculate_confidence_scaling(ctx.active_maneuver_confidence)
            if h_b in ("HEALTHY", "QUIESCENT_BYPASS") else 0.0
        )

        # Style Rules (only if Pipeline A is HEALTHY)
        if h_a == "HEALTHY":
            # Rule 2: SUSTAINED_AGGRESSIVE_STYLE
            if ctx.is_sustained_aggressive and phi_style > 0.0:
                raw_score += self.config.weight_sustained_aggressive * phi_style
                triggered_rules.append("SUSTAINED_AGGRESSIVE_STYLE")
                expl = (
                    f"Sustained aggressive driving style detected over recent window "
                    f"(Aggressive Ratio: {ctx.style_aggressive_ratio:.0%}, EMA: {ctx.style_ema:.2f})."
                )
                rule_explanations.append(expl)

                if not self._should_suppress_alert(
                    AlertCategory.STYLE,
                    AlertSeverity.MEDIUM,
                    ts,
                    self.config.alert_cooldown_style
                ):
                    self._alert_counter += 1
                    alert = SafetyAlert(
                        alert_id=f"ALT-STY-{int(ts * 1000)}-{self._alert_counter:04d}",
                        timestamp=ts,
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.STYLE,
                        title="Sustained Aggressive Driving",
                        triggering_condition="style_aggressive_ratio >= 0.50 or style_ema >= 0.65",
                        explanation=expl,
                        related_predictions={
                            "style": "Aggressive Driving",
                            "aggressive_ratio": ctx.style_aggressive_ratio,
                            "confidence": ctx.current_style_confidence
                        },
                        safe_driving_advisory="Moderate acceleration and maintain smooth driving inputs to preserve vehicle stability."
                    )
                    alerts.append(alert)
                    self._record_alert(alert)

            # Rule 3: TRANSIENT_AGGRESSIVE_STYLE
            elif ctx.is_transient_aggressive and phi_style > 0.0:
                raw_score += self.config.weight_transient_aggressive * phi_style
                triggered_rules.append("TRANSIENT_AGGRESSIVE_STYLE")
                rule_explanations.append(
                    f"Transient aggressive driving spike observed (Confidence: {ctx.current_style_confidence:.0%})."
                )

        # Maneuver Rules (only if Pipeline B is HEALTHY or QUIESCENT_BYPASS)
        if h_b in ("HEALTHY", "QUIESCENT_BYPASS"):
            # Rule 4: SUDDEN_ACCELERATION_MANEUVER
            if ctx.is_maneuver_active and ctx.active_maneuver == ManeuverClass.SUDDEN_ACCELERATION.value and phi_maneuver > 0.0:
                raw_score += self.config.weight_sudden_acceleration * phi_maneuver
                triggered_rules.append("SUDDEN_ACCELERATION_MANEUVER")
                expl = f"Sudden acceleration maneuver detected (Confidence: {ctx.active_maneuver_confidence:.0%})."
                rule_explanations.append(expl)

                if not self._should_suppress_alert(
                    AlertCategory.MANEUVER,
                    AlertSeverity.MEDIUM,
                    ts,
                    self.config.alert_cooldown_maneuver
                ):
                    self._alert_counter += 1
                    alert = SafetyAlert(
                        alert_id=f"ALT-MAN-{int(ts * 1000)}-{self._alert_counter:04d}",
                        timestamp=ts,
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.MANEUVER,
                        title="Sudden Acceleration Detected",
                        triggering_condition="active_maneuver == SUDDEN_ACCELERATION",
                        explanation=expl,
                        related_predictions={
                            "maneuver": "Sudden Acceleration",
                            "confidence": ctx.active_maneuver_confidence
                        },
                        safe_driving_advisory="Ease off the throttle to prevent abrupt forward weight transfer."
                    )
                    alerts.append(alert)
                    self._record_alert(alert)

            # Rule 5: SUDDEN_DECELERATION_MANEUVER (Harsh Braking)
            elif ctx.is_maneuver_active and ctx.active_maneuver == ManeuverClass.SUDDEN_DECELERATION.value and phi_maneuver > 0.0:
                raw_score += self.config.weight_sudden_deceleration * phi_maneuver
                triggered_rules.append("SUDDEN_DECELERATION_MANEUVER")
                expl = (
                    f"Sudden deceleration / harsh braking detected (Confidence: {ctx.active_maneuver_confidence:.0%}), "
                    "posing rear-end collision and passenger destabilization hazards."
                )
                rule_explanations.append(expl)

                if not self._should_suppress_alert(
                    AlertCategory.MANEUVER,
                    AlertSeverity.HIGH,
                    ts,
                    self.config.alert_cooldown_maneuver
                ):
                    self._alert_counter += 1
                    alert = SafetyAlert(
                        alert_id=f"ALT-MAN-{int(ts * 1000)}-{self._alert_counter:04d}",
                        timestamp=ts,
                        severity=AlertSeverity.HIGH,
                        category=AlertCategory.MANEUVER,
                        title="Harsh Braking Detected",
                        triggering_condition="active_maneuver == SUDDEN_DECELERATION",
                        explanation=expl,
                        related_predictions={
                            "maneuver": "Sudden Deceleration",
                            "confidence": ctx.active_maneuver_confidence
                        },
                        safe_driving_advisory="Increase following distance to allow progressive, controlled braking."
                    )
                    alerts.append(alert)
                    self._record_alert(alert)

            # Rule 6: SUDDEN_TURN_MANEUVER (Left or Right)
            elif ctx.is_maneuver_active and ctx.active_maneuver in [
                ManeuverClass.SUDDEN_RIGHT_TURN.value,
                ManeuverClass.SUDDEN_LEFT_TURN.value
            ] and phi_maneuver > 0.0:
                raw_score += self.config.weight_sudden_turn * phi_maneuver
                m_name = MANEUVER_LABEL_NAMES.get(ctx.active_maneuver, "Sudden Turn")
                triggered_rules.append("SUDDEN_TURN_MANEUVER")
                expl = (
                    f"Abrupt lateral steering maneuver detected ({m_name}, Confidence: {ctx.active_maneuver_confidence:.0%}), "
                    "elevating vehicle rollover and lateral skid risks."
                )
                rule_explanations.append(expl)

                if not self._should_suppress_alert(
                    AlertCategory.MANEUVER,
                    AlertSeverity.HIGH,
                    ts,
                    self.config.alert_cooldown_maneuver
                ):
                    self._alert_counter += 1
                    alert = SafetyAlert(
                        alert_id=f"ALT-MAN-{int(ts * 1000)}-{self._alert_counter:04d}",
                        timestamp=ts,
                        severity=AlertSeverity.HIGH,
                        category=AlertCategory.MANEUVER,
                        title=f"Abrupt Lateral Turn ({m_name})",
                        triggering_condition="active_maneuver in (SUDDEN_RIGHT_TURN, SUDDEN_LEFT_TURN)",
                        explanation=expl,
                        related_predictions={
                            "maneuver": m_name,
                            "confidence": ctx.active_maneuver_confidence
                        },
                        safe_driving_advisory="Smooth out steering inputs before cornering to maintain tire traction."
                    )
                    alerts.append(alert)
                    self._record_alert(alert)

            # Rule 8: REPEATED_ERRATIC_MANEUVERS
            if ctx.recent_maneuver_count >= 2:
                raw_score += self.config.weight_repeated_maneuver_bonus
                triggered_rules.append("REPEATED_ERRATIC_MANEUVERS")
                expl = f"Repeated erratic maneuvers detected ({ctx.recent_maneuver_count} within 10s window)."
                rule_explanations.append(expl)

        # Rule 7: COMPOUND_HIGH_RISK_HAZARD (Simultaneous Aggressive Style + Maneuver)
        # Gated Compound Bonus (D-10 Authoritative Requirement):
        # Applies if and only if both modality scaling factors are strictly positive.
        if (
            h_a == "HEALTHY" and
            h_b in ("HEALTHY", "QUIESCENT_BYPASS") and
            ctx.is_maneuver_active and
            (ctx.is_sustained_aggressive or ctx.is_transient_aggressive) and
            phi_style > 0.0 and
            phi_maneuver > 0.0
        ):
            raw_score += self.config.weight_compound_hazard_bonus
            triggered_rules.append("COMPOUND_HIGH_RISK_HAZARD")
            expl = (
                "Compound hazard: Abrupt maneuver executed simultaneously during aggressive driving style! "
                "High probability of vehicle instability."
            )
            rule_explanations.append(expl)

            if not self._should_suppress_alert(
                AlertCategory.COMPOUND,
                AlertSeverity.CRITICAL,
                ts,
                self.config.alert_cooldown_compound
            ):
                self._alert_counter += 1
                alert = SafetyAlert(
                    alert_id=f"ALT-CMP-{int(ts * 1000)}-{self._alert_counter:04d}",
                    timestamp=ts,
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.COMPOUND,
                    title="CRITICAL: Compound Risk Maneuver",
                    triggering_condition="is_maneuver_active and (is_sustained_aggressive or is_transient_aggressive) and phi_style > 0 and phi_maneuver > 0",
                    explanation=expl,
                    related_predictions={
                        "style": "Aggressive Driving",
                        "maneuver": MANEUVER_LABEL_NAMES.get(ctx.active_maneuver, "Unknown"),
                        "maneuver_confidence": ctx.active_maneuver_confidence
                    },
                    safe_driving_advisory="IMMEDIATE ACTION: Reduce speed and stabilize vehicle handling immediately."
                )
                alerts.append(alert)
                self._record_alert(alert)

        # Final Risk Score Calculation (Modality-specific scaling replaces legacy Rule 9 discount)
        confidence_factor = 1.0
        final_score = min(100.0, max(0.0, float(raw_score)))

        # Behavioral State Mapping
        if final_score >= self.config.score_threshold_high_risk:
            state = SafetyState.HIGH_RISK
        elif final_score >= self.config.score_threshold_warning:
            state = SafetyState.WARNING
        elif final_score >= self.config.score_threshold_caution:
            state = SafetyState.CAUTION
        else:
            state = SafetyState.NORMAL

        # -------------------------------------------------------------
        # Fail-Safe State Arbitration & Diagnostic Alerts
        # -------------------------------------------------------------
        # Condition 3: Active Maneuver during Pipeline A Outage (Rows 6, 11, 16 with M != 0) - Policy 1A-Clamped
        if h_a in ("DEGRADED", "MISSING", "INVALID") and h_b == "HEALTHY" and ctx.is_maneuver_active:
            if state == SafetyState.NORMAL:
                state = SafetyState.CAUTION
            sev = AlertSeverity.INFO if h_a == "MISSING" else AlertSeverity.MEDIUM
            expl_dq = f"Driving style pipeline is {h_a}. Active maneuver evaluated under fail-safe safety floor."
            _emit_dq_alert(
                title=f"Driving Style Telemetry {h_a.capitalize()}",
                condition=f"pipeline_a_health == '{h_a}' and is_maneuver_active",
                explanation=expl_dq,
                severity=sev
            )
            rule_explanations.append(expl_dq)

        # Condition 4: Impaired Pipeline B with Healthy Pipeline A (Rows 2, 3, 4) - P-05A
        elif h_a == "HEALTHY" and h_b in ("DEGRADED", "MISSING", "INVALID"):
            if state == SafetyState.NORMAL:
                state = SafetyState.CAUTION
            sev = AlertSeverity.INFO if h_b == "MISSING" else AlertSeverity.MEDIUM
            expl_dq = f"Maneuver detection pipeline is {h_b}. System operating under fail-safe CAUTION advisory floor."
            _emit_dq_alert(
                title=f"Maneuver Telemetry {h_b.capitalize()}",
                condition=f"pipeline_b_health == '{h_b}'",
                explanation=expl_dq,
                severity=sev
            )
            rule_explanations.append(expl_dq)

        if not rule_explanations:
            overall_explanation = "Driving behavior is nominal. Vehicle inputs are within safe operating bounds."
        else:
            overall_explanation = " | ".join(rule_explanations)

        return RuleEvaluationResult(
            timestamp=ts,
            safety_state=state,
            risk_score=final_score,
            raw_score=raw_score,
            confidence_factor=confidence_factor,
            triggered_rules=triggered_rules,
            rule_explanations=rule_explanations,
            overall_explanation=overall_explanation,
            active_alerts=alerts
        )

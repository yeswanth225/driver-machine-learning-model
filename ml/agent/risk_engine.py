"""
Deterministic Safety Rule Engine & Explainable Risk Scoring Module
===================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

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
        Evaluate smoothed context against deterministic safety rules.
        Produces explainable risk score, safety state, and deduplicated alerts.
        """
        ts = ctx.timestamp

        # -------------------------------------------------------------
        # Rule 1: DATA_QUALITY_CHECK (Guardrail)
        # -------------------------------------------------------------
        if not ctx.is_valid or ctx.data_quality_status != "VALID":
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

        # -------------------------------------------------------------
        # Normal Driving Rule Processing
        # -------------------------------------------------------------
        raw_score = 0.0
        triggered_rules: List[str] = []
        rule_explanations: List[str] = []
        alerts: List[SafetyAlert] = []

        # Confidence tracking
        confs = []
        if ctx.current_style == DrivingStyle.AGGRESSIVE.value:
            confs.append(ctx.current_style_confidence)
        if ctx.is_maneuver_active:
            confs.append(ctx.active_maneuver_confidence)
        avg_concern_conf = float(sum(confs) / len(confs)) if confs else 1.0

        # Rule 2: SUSTAINED_AGGRESSIVE_STYLE
        if ctx.is_sustained_aggressive:
            raw_score += self.config.weight_sustained_aggressive
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
        elif ctx.is_transient_aggressive:
            raw_score += self.config.weight_transient_aggressive
            triggered_rules.append("TRANSIENT_AGGRESSIVE_STYLE")
            rule_explanations.append(
                f"Transient aggressive driving spike observed (Confidence: {ctx.current_style_confidence:.0%})."
            )

        # Rule 4: SUDDEN_ACCELERATION_MANEUVER
        if ctx.is_maneuver_active and ctx.active_maneuver == ManeuverClass.SUDDEN_ACCELERATION.value:
            raw_score += self.config.weight_sudden_acceleration
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
        elif ctx.is_maneuver_active and ctx.active_maneuver == ManeuverClass.SUDDEN_DECELERATION.value:
            raw_score += self.config.weight_sudden_deceleration
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
        ]:
            raw_score += self.config.weight_sudden_turn
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

        # Rule 7: COMPOUND_HIGH_RISK_HAZARD (Simultaneous Aggressive Style + Maneuver)
        if ctx.is_maneuver_active and (ctx.is_sustained_aggressive or ctx.is_transient_aggressive):
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
                    triggering_condition="is_maneuver_active and (is_sustained_aggressive or is_transient_aggressive)",
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

        # Rule 8: REPEATED_ERRATIC_MANEUVERS
        if ctx.recent_maneuver_count >= 2:
            raw_score += self.config.weight_repeated_maneuver_bonus
            triggered_rules.append("REPEATED_ERRATIC_MANEUVERS")
            expl = f"Repeated erratic maneuvers detected ({ctx.recent_maneuver_count} within 10s window)."
            rule_explanations.append(expl)

        # Rule 9: LOW_CONFIDENCE_DISCOUNT
        confidence_factor = 1.0
        if raw_score > 0 and avg_concern_conf < self.config.min_confidence_threshold:
            confidence_factor = self.config.low_confidence_discount
            triggered_rules.append("LOW_CONFIDENCE_DISCOUNT")
            rule_explanations.append(
                f"Risk score discounted by {(1.0 - confidence_factor):.0%} due to low model confidence ({avg_concern_conf:.0%})."
            )

        # Final Risk Score Calculation
        final_score = raw_score * confidence_factor
        final_score = min(100.0, max(0.0, float(final_score)))

        # State Mapping
        if final_score >= self.config.score_threshold_high_risk:
            state = SafetyState.HIGH_RISK
        elif final_score >= self.config.score_threshold_warning:
            state = SafetyState.WARNING
        elif final_score >= self.config.score_threshold_caution:
            state = SafetyState.CAUTION
        else:
            state = SafetyState.NORMAL

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

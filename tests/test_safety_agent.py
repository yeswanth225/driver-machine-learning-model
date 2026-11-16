"""
Unit & Integration Tests: Intelligent Driver Activity Monitoring & Safety Agent
=============================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Safety Agent Tests
"""

import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ml.agent.config import (
    SafetyState,
    DrivingStyle,
    ManeuverClass,
    AlertSeverity,
    AlertCategory,
    SafetyAgentConfig,
    DEFAULT_CONFIG
)
from ml.agent.prediction_validator import (
    RawPredictionInput,
    ValidatedPrediction,
    PredictionValidator
)
from ml.agent.temporal_smoother import (
    SmoothedContext,
    TemporalSmoother
)
from ml.agent.risk_engine import (
    DeterministicRiskEngine,
    RuleEvaluationResult,
    SafetyAlert
)
from ml.agent.safety_agent import (
    SafetyAssessment,
    IntelligentSafetyAgent
)
from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier
from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset" / "processed"


class TestPredictionValidator:
    """Tests for schema, bounds, finiteness, and staleness validation."""

    @pytest.fixture
    def validator(self):
        return PredictionValidator()

    def test_valid_input_passes(self, validator):
        raw = RawPredictionInput(
            timestamp=10.0,
            style_prediction=DrivingStyle.NORMAL.value,
            style_confidence=0.92,
            maneuver_prediction=ManeuverClass.NONE.value,
            maneuver_confidence=0.88
        )
        res = validator.validate(raw, current_time=10.5)
        assert res.is_valid is True
        assert res.is_stale is False
        assert res.data_quality_status == "VALID"
        assert len(res.validation_errors) == 0

    def test_missing_predictions_rejected(self, validator):
        raw = RawPredictionInput(timestamp=10.0)
        res = validator.validate(raw, current_time=10.5)
        assert res.is_valid is False
        assert res.data_quality_status == "MISSING"
        assert any("Missing prediction" in err for err in res.validation_errors)

    def test_stale_timestamp_rejected(self, validator):
        raw = RawPredictionInput(
            timestamp=10.0,
            style_prediction=0,
            style_confidence=0.9
        )
        # Evaluated at 15.0 seconds (5s elapsed > 3.0s threshold)
        res = validator.validate(raw, current_time=15.0)
        assert res.is_valid is False
        assert res.is_stale is True
        assert res.data_quality_status == "STALE"
        assert any("Stale telemetry" in err for err in res.validation_errors)

    def test_invalid_class_labels_handled(self, validator):
        raw = RawPredictionInput(
            timestamp=10.0,
            style_prediction=99,  # invalid
            style_confidence=0.8,
            maneuver_prediction=88,  # invalid
            maneuver_confidence=0.8
        )
        res = validator.validate(raw)
        assert res.is_valid is False
        assert res.data_quality_status == "INVALID"
        assert res.style_prediction is None
        assert res.maneuver_prediction is None

    def test_invalid_confidence_bounds_clamped(self, validator):
        raw = RawPredictionInput(
            timestamp=10.0,
            style_prediction=1,
            style_confidence=1.5,  # > 1.0
            maneuver_prediction=1,
            maneuver_confidence=-0.3  # < 0.0
        )
        res = validator.validate(raw)
        assert res.style_confidence == 1.0
        assert res.maneuver_confidence == 0.0

    def test_non_finite_inputs_handled(self, validator):
        raw = RawPredictionInput(
            timestamp=float("nan"),
            style_prediction=1,
            style_confidence=float("inf")
        )
        res = validator.validate(raw)
        assert res.is_valid is False
        assert res.timestamp == 0.0


class TestTemporalSmoother:
    """Tests for causal multi-rate temporal smoothing and persistence."""

    @pytest.fixture
    def smoother(self):
        return TemporalSmoother()

    def test_isolated_aggressive_prediction_does_not_trigger_sustained(self, smoother):
        # 5 normal readings followed by 1 aggressive reading
        for i in range(5):
            val = ValidatedPrediction(
                timestamp=float(i) * 0.1,
                style_prediction=0,
                style_confidence=0.9,
                maneuver_prediction=0,
                maneuver_confidence=0.9,
                is_valid=True,
                is_stale=False,
                data_quality_status="VALID"
            )
            ctx = smoother.update(val)
        assert ctx.is_sustained_aggressive is False

        # 1 isolated aggressive pulse
        val_agg = ValidatedPrediction(
            timestamp=0.6,
            style_prediction=1,
            style_confidence=0.85,
            maneuver_prediction=0,
            maneuver_confidence=0.9,
            is_valid=True,
            is_stale=False,
            data_quality_status="VALID"
        )
        ctx_agg = smoother.update(val_agg)
        assert ctx_agg.is_sustained_aggressive is False
        assert ctx_agg.is_transient_aggressive is True

    def test_sustained_aggressive_predictions_trigger_state(self, smoother):
        # 10 consecutive aggressive predictions
        for i in range(10):
            val = ValidatedPrediction(
                timestamp=float(i) * 0.1,
                style_prediction=1,
                style_confidence=0.9,
                maneuver_prediction=0,
                maneuver_confidence=0.9,
                is_valid=True,
                is_stale=False,
                data_quality_status="VALID"
            )
            ctx = smoother.update(val)
        assert ctx.is_sustained_aggressive is True
        assert ctx.style_aggressive_ratio >= 0.50

    def test_maneuver_persistence_and_expiry(self, smoother):
        # Maneuver arrives at t=1.0
        val_m = ValidatedPrediction(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.9,
            maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
            maneuver_confidence=0.95,
            is_valid=True,
            is_stale=False,
            data_quality_status="VALID"
        )
        ctx1 = smoother.update(val_m)
        assert ctx1.is_maneuver_active is True
        assert ctx1.active_maneuver == ManeuverClass.SUDDEN_DECELERATION.value

        # At t=2.0 (1.0s elapsed < 2.0s persistence), no new maneuver prediction provided
        val_hold = ValidatedPrediction(
            timestamp=2.0,
            style_prediction=0,
            style_confidence=0.9,
            maneuver_prediction=ManeuverClass.NONE.value,
            maneuver_confidence=0.0,
            is_valid=True,
            is_stale=False,
            data_quality_status="VALID"
        )
        ctx2 = smoother.update(val_hold)
        assert ctx2.is_maneuver_active is True
        assert ctx2.active_maneuver == ManeuverClass.SUDDEN_DECELERATION.value

        # At t=3.5 (2.5s elapsed > 2.0s persistence), maneuver should have expired
        val_exp = ValidatedPrediction(
            timestamp=3.5,
            style_prediction=0,
            style_confidence=0.9,
            maneuver_prediction=ManeuverClass.NONE.value,
            maneuver_confidence=0.0,
            is_valid=True,
            is_stale=False,
            data_quality_status="VALID"
        )
        ctx3 = smoother.update(val_exp)
        assert ctx3.is_maneuver_active is False
        assert ctx3.active_maneuver == ManeuverClass.NONE.value

    def test_history_window_pruning(self, smoother):
        # Add sample at t=0.0
        val0 = ValidatedPrediction(
            timestamp=0.0,
            style_prediction=0,
            style_confidence=0.9,
            maneuver_prediction=0,
            maneuver_confidence=0.0,
            is_valid=True,
            is_stale=False,
            data_quality_status="VALID"
        )
        smoother.update(val0)

        # Advance to t=15.0 (> 10s history cutoff)
        val15 = ValidatedPrediction(
            timestamp=15.0,
            style_prediction=0,
            style_confidence=0.9,
            maneuver_prediction=0,
            maneuver_confidence=0.0,
            is_valid=True,
            is_stale=False,
            data_quality_status="VALID"
        )
        smoother.update(val15)
        assert len(smoother._history) == 1
        assert smoother._history[0][0] == 15.0


class TestDeterministicRiskEngine:
    """Tests for deterministic scoring, rule activations, state transitions, and clamping."""

    @pytest.fixture
    def engine(self):
        return DeterministicRiskEngine()

    def test_normal_driving_produces_normal_state(self, engine):
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.95,
            active_maneuver=0,
            active_maneuver_confidence=0.0,
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=False,
            recent_maneuver_count=0
        )
        res = engine.evaluate(ctx)
        assert res.safety_state == SafetyState.NORMAL
        assert res.risk_score == 0.0
        assert len(res.triggered_rules) == 0
        assert len(res.active_alerts) == 0

    def test_insufficient_data_produces_insufficient_state(self, engine):
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.0,
            active_maneuver=0,
            active_maneuver_confidence=0.0,
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=False,
            recent_maneuver_count=0,
            is_valid=False,
            data_quality_status="STALE"
        )
        res = engine.evaluate(ctx)
        assert res.safety_state == SafetyState.INSUFFICIENT_DATA
        assert "DATA_QUALITY_ANOMALY" in res.triggered_rules

    def test_sudden_acceleration_triggers_rule(self, engine):
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_ACCELERATION.value,
            active_maneuver_confidence=0.85,
            active_maneuver_age_seconds=0.5,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res = engine.evaluate(ctx)
        assert "SUDDEN_ACCELERATION_MANEUVER" in res.triggered_rules
        assert res.risk_score == 25.0
        assert res.safety_state == SafetyState.CAUTION

    def test_harsh_braking_triggers_warning(self, engine):
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.90,
            active_maneuver_age_seconds=0.2,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res = engine.evaluate(ctx)
        assert "SUDDEN_DECELERATION_MANEUVER" in res.triggered_rules
        assert res.risk_score == 35.0
        assert res.safety_state == SafetyState.CAUTION

    def test_compound_high_risk_hazard(self, engine):
        # Maneuver (Sudden Turn: 30) + Sustained Aggressive (35) + Compound Bonus (25) = 90 -> HIGH_RISK
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.85,
            style_aggressive_ratio=0.80,
            is_sustained_aggressive=True,
            is_transient_aggressive=False,
            current_style=1,
            current_style_confidence=0.90,
            active_maneuver=ManeuverClass.SUDDEN_RIGHT_TURN.value,
            active_maneuver_confidence=0.92,
            active_maneuver_age_seconds=0.4,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res = engine.evaluate(ctx)
        assert "COMPOUND_HIGH_RISK_HAZARD" in res.triggered_rules
        assert "SUSTAINED_AGGRESSIVE_STYLE" in res.triggered_rules
        assert "SUDDEN_TURN_MANEUVER" in res.triggered_rules
        assert res.risk_score >= 75.0
        assert res.safety_state == SafetyState.HIGH_RISK
        assert any(a.severity == AlertSeverity.CRITICAL for a in res.active_alerts)

    def test_low_confidence_discount(self, engine):
        # Harsh braking (35), but confidence is 0.45 (< 0.50 threshold) -> 35 * 0.70 = 24.5
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.45,
            active_maneuver_age_seconds=0.2,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res = engine.evaluate(ctx)
        assert "LOW_CONFIDENCE_DISCOUNT" in res.triggered_rules
        assert res.risk_score == pytest.approx(24.5, abs=0.1)
        assert res.safety_state == SafetyState.NORMAL

    def test_score_clamping(self, engine):
        # Artificially stack every rule
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.9,
            style_aggressive_ratio=0.9,
            is_sustained_aggressive=True,
            is_transient_aggressive=False,
            current_style=1,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.95,
            active_maneuver_age_seconds=0.2,
            is_maneuver_active=True,
            recent_maneuver_count=5  # repeated bonus (15)
        )
        # raw score = 35 (sustained) + 35 (SD) + 25 (compound) + 15 (repeated) = 110
        res = engine.evaluate(ctx)
        assert res.raw_score == 110.0
        assert res.risk_score == 100.0  # clamped to 100

    def test_scoring_determinism(self, engine):
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.5,
            style_aggressive_ratio=0.4,
            is_sustained_aggressive=False,
            is_transient_aggressive=True,
            current_style=1,
            current_style_confidence=0.8,
            active_maneuver=ManeuverClass.SUDDEN_LEFT_TURN.value,
            active_maneuver_confidence=0.85,
            active_maneuver_age_seconds=0.1,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res1 = engine.evaluate(ctx)
        engine.reset()
        res2 = engine.evaluate(ctx)
        assert res1.risk_score == res2.risk_score
        assert res1.safety_state == res2.safety_state
        assert res1.triggered_rules == res2.triggered_rules


class TestAlertDeduplication:
    """Tests for alert deduplication, cooldown timers, and severity escalation bypass."""

    def test_rapid_consecutive_alerts_suppressed_by_cooldown(self):
        engine = DeterministicRiskEngine()
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_ACCELERATION.value,
            active_maneuver_confidence=0.85,
            active_maneuver_age_seconds=0.1,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res1 = engine.evaluate(ctx)
        assert len(res1.active_alerts) == 1

        # Identical prediction arrives 0.1s later (within 3.0s maneuver cooldown)
        ctx2 = SmoothedContext(
            timestamp=1.1,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_ACCELERATION.value,
            active_maneuver_confidence=0.85,
            active_maneuver_age_seconds=0.2,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res2 = engine.evaluate(ctx2)
        # Suppressed! Zero alerts generated
        assert len(res2.active_alerts) == 0

    def test_severity_escalation_bypasses_cooldown(self):
        engine = DeterministicRiskEngine()
        # 1. Medium severity alert at t=1.0 (Sudden Acceleration)
        ctx1 = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_ACCELERATION.value,
            active_maneuver_confidence=0.85,
            active_maneuver_age_seconds=0.1,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res1 = engine.evaluate(ctx1)
        assert len(res1.active_alerts) == 1
        assert res1.active_alerts[0].severity == AlertSeverity.MEDIUM

        # 2. Critical severity alert at t=1.2 (Compound Hazard)
        ctx2 = SmoothedContext(
            timestamp=1.2,
            style_ema=0.8,
            style_aggressive_ratio=0.8,
            is_sustained_aggressive=True,
            is_transient_aggressive=False,
            current_style=1,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.95,
            active_maneuver_age_seconds=0.1,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res2 = engine.evaluate(ctx2)
        # Must NOT be suppressed: escalated to CRITICAL!
        assert len(res2.active_alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in res2.active_alerts)


class TestIntelligentSafetyAgentIntegration:
    """Full end-to-end integration tests for IntelligentSafetyAgent."""

    @pytest.fixture
    def agent(self):
        return IntelligentSafetyAgent()

    def test_full_state_transition_sequence(self, agent):
        # 1. Start with Nominal Driving
        ass1 = agent.process_predictions(timestamp=1.0, style_prediction=0, style_confidence=0.95)
        assert ass1.safety_state == SafetyState.NORMAL
        assert ass1.risk_score == 0.0

        # 2. Abrupt braking maneuver occurs at t=2.0
        ass2 = agent.process_predictions(
            timestamp=2.0,
            style_prediction=0,
            style_confidence=0.95,
            maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
            maneuver_confidence=0.92
        )
        assert ass2.safety_state == SafetyState.CAUTION
        assert ass2.risk_score >= 35.0

        # 3. Compound Aggression + Turning at t=2.5
        ass3 = agent.process_predictions(
            timestamp=2.5,
            style_prediction=1,
            style_confidence=0.92,
            maneuver_prediction=ManeuverClass.SUDDEN_RIGHT_TURN.value,
            maneuver_confidence=0.94
        )
        assert ass3.safety_state == SafetyState.HIGH_RISK
        assert ass3.risk_score >= 75.0

        # 4. Telemetry is interrupted (Stale reading at t=10.0 evaluated at current_time=15.0)
        ass4 = agent.process_predictions(
            timestamp=10.0,
            style_prediction=0,
            style_confidence=0.9,
            current_time=15.0
        )
        assert ass4.safety_state == SafetyState.INSUFFICIENT_DATA
        assert ass4.data_quality_status == "STALE"

    def test_process_features_with_production_models(self):
        # Load real test samples and real inference classifiers
        clf_c = DatasetCBinaryClassifier(
            model_dir=str(REPO_ROOT / "models" / "experiments" / "dataset_c"),
            model_path=str(REPO_ROOT / "models" / "experiments" / "dataset_c" / "dataset_c_support_vector_machine_exp_v1.joblib"),
            scaler_path=str(REPO_ROOT / "models" / "experiments" / "dataset_c" / "dataset_c_scaler_exp_v1.joblib"),
            config_path=str(REPO_ROOT / "models" / "experiments" / "dataset_c" / "dataset_c_experiment_summary.json"),
            decision_threshold=0.84
        )
        clf_b = DatasetBManeuverClassifier(model_name="extra_trees")

        agent = IntelligentSafetyAgent(classifier_c=clf_c, classifier_b=clf_b)

        test_c_df = pd.read_csv(DATA_DIR / "dataset_c_test.csv")[clf_c.feature_names]
        test_b_df = pd.read_csv(DATA_DIR / "dataset_b_test.csv")[clf_b.feature_names]

        # Process first sample
        assessment = agent.process_features(
            timestamp=1.0,
            features_c=test_c_df.iloc[[0]],
            features_b=test_b_df.iloc[[0]]
        )

        assert assessment.is_valid is True
        assert assessment.safety_state in [SafetyState.NORMAL, SafetyState.CAUTION, SafetyState.WARNING, SafetyState.HIGH_RISK]
        assert 0.0 <= assessment.risk_score <= 100.0
        assert assessment.driving_style in ["Normal Driving", "Aggressive Driving"]
        assert assessment.maneuver in [
            "None / Nominal", "Sudden Acceleration", "Sudden Deceleration",
            "Sudden Right Turn", "Sudden Left Turn"
        ]
        assert isinstance(assessment.to_dict(), dict)

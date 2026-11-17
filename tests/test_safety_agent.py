"""
Unit & Integration Tests: Intelligent Driver Activity Monitoring & Safety Agent
=============================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Safety Agent Tests
"""

import math
import hashlib
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
    GateStatus,
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
    GateResult,
    KinematicEventGate,
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

    def test_p01_numpy_scalar_types_accepted(self, validator):
        """P-01: Verify numpy integer and float scalar types are accepted and sanitized."""
        raw = RawPredictionInput(
            timestamp=np.float64(10.0),
            style_prediction=np.int64(1),
            style_confidence=np.float32(0.92),
            maneuver_prediction=np.int32(2),
            maneuver_confidence=np.float64(0.88)
        )
        res = validator.validate(raw, current_time=10.5)
        assert res.is_valid is True
        assert res.data_quality_status == "VALID"
        assert len(res.validation_errors) == 0
        assert type(res.style_prediction) is int
        assert type(res.maneuver_prediction) is int
        assert type(res.style_confidence) is float
        assert type(res.maneuver_confidence) is float
        assert res.style_prediction == 1
        assert res.maneuver_prediction == 2
        assert abs(res.style_confidence - 0.92) < 1e-4
        assert abs(res.maneuver_confidence - 0.88) < 1e-4

    def test_p01_boolean_inputs_strictly_rejected(self, validator):
        """P-01: Verify bool inputs are strictly rejected (cannot bypass numeric checks)."""
        # Boolean style prediction
        res1 = validator.validate(RawPredictionInput(timestamp=10.0, style_prediction=True, style_confidence=0.9))
        assert res1.is_valid is False
        assert any("Style prediction must be an integer" in err for err in res1.validation_errors)

        res2 = validator.validate(RawPredictionInput(timestamp=10.0, style_prediction=False, style_confidence=0.9))
        assert res2.is_valid is False
        assert any("Style prediction must be an integer" in err for err in res2.validation_errors)

        # Boolean maneuver prediction
        res3 = validator.validate(RawPredictionInput(timestamp=10.0, maneuver_prediction=True, maneuver_confidence=0.9))
        assert res3.is_valid is False
        assert any("Maneuver prediction must be an integer" in err for err in res3.validation_errors)

        # Boolean confidence
        res4 = validator.validate(RawPredictionInput(timestamp=10.0, style_prediction=1, style_confidence=True))
        assert res4.is_valid is False
        assert any("Style confidence" in err and "must be a finite float" in err for err in res4.validation_errors)

        # Boolean timestamp
        res5 = validator.validate(RawPredictionInput(timestamp=True, style_prediction=1, style_confidence=0.9))
        assert res5.is_valid is False
        assert any("Invalid timestamp value" in err for err in res5.validation_errors)

    def test_p01_nan_and_infinite_confidences_rejected(self, validator):
        """P-01: Verify NaN, +inf, and -inf confidence values are caught and flagged."""
        res_nan = validator.validate(RawPredictionInput(
            timestamp=10.0,
            style_prediction=1,
            style_confidence=float("nan"),
            maneuver_prediction=1,
            maneuver_confidence=np.nan
        ))
        assert res_nan.is_valid is False
        assert any("Style confidence" in err for err in res_nan.validation_errors)
        assert any("Maneuver confidence" in err for err in res_nan.validation_errors)

        res_inf = validator.validate(RawPredictionInput(
            timestamp=10.0,
            style_prediction=1,
            style_confidence=float("inf"),
            maneuver_prediction=1,
            maneuver_confidence=float("-inf")
        ))
        assert res_inf.is_valid is False
        assert any("Style confidence" in err for err in res_inf.validation_errors)
        assert any("Maneuver confidence" in err for err in res_inf.validation_errors)

    def test_p01_process_predictions_boundary_coercion(self):
        """P-01: Verify IntelligentSafetyAgent.process_predictions coerces numpy types and rejects bool."""
        agent = IntelligentSafetyAgent()

        # Valid numpy types coerced at boundary
        ass_valid = agent.process_predictions(
            timestamp=np.float64(1.0),
            style_prediction=np.int64(0),
            style_confidence=np.float32(0.95),
            maneuver_prediction=np.int32(0),
            maneuver_confidence=np.float64(0.90)
        )
        assert ass_valid.is_valid is True
        assert ass_valid.safety_state == SafetyState.NORMAL

        # Boolean input rejected without silent int coercion
        ass_bool = agent.process_predictions(
            timestamp=2.0,
            style_prediction=True,
            style_confidence=0.95
        )
        assert ass_bool.is_valid is False

    def test_p01_numpy_float32_boundary_thresholds_040_060(self, validator):
        """P-01: Verify numpy float32 boundary values at 0.40 and 0.60 are accepted and sanitized."""
        for conf_val in [np.float32(0.40), np.float32(0.60)]:
            raw = RawPredictionInput(
                timestamp=10.0,
                style_prediction=np.int64(1),
                style_confidence=conf_val,
                maneuver_prediction=np.int32(1),
                maneuver_confidence=conf_val
            )
            res = validator.validate(raw)
            assert res.is_valid is True
            assert res.data_quality_status == "VALID"
            assert len(res.validation_errors) == 0
            assert type(res.style_confidence) is float
            assert type(res.maneuver_confidence) is float
            assert abs(res.style_confidence - float(conf_val)) < 1e-6
            assert abs(res.maneuver_confidence - float(conf_val)) < 1e-6

        # Test at process_predictions boundary as well
        agent = IntelligentSafetyAgent()
        for conf_val in [np.float32(0.40), np.float32(0.60)]:
            ass = agent.process_predictions(
                timestamp=1.0,
                style_prediction=np.int64(0),
                style_confidence=conf_val,
                maneuver_prediction=np.int32(0),
                maneuver_confidence=conf_val
            )
            assert ass.is_valid is True


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


class TestStage6RemediationRegression:
    """
    Authoritative Stage 6 Regression Test Suite.
    Directly tests defects D-01 through D-15 and production integration T-16.
    Every test targets actual production behavior with non-tautological assertions.
    """

    # -------------------------------------------------------------------------
    # T-01 (Defect D-01): Pipeline B Continuous False Maneuvers / Event Gate
    # -------------------------------------------------------------------------
    def test_t01_d01_kinematic_event_gate_quiescent_bypass(self):
        """
        D-01: Extra Trees classifier lacks a nominal class. Continuous ingestion of
        quiescent features must be intercepted by an upstream kinematic event gate
        to avoid classifying normal cruising as an emergency maneuver.
        """
        clf_b = DatasetBManeuverClassifier(model_name="extra_trees")
        agent = IntelligentSafetyAgent(classifier_b=clf_b)
        
        # Synthetic quiescent cruising frame: flat acceleration, zero angular velocity
        row = {col: 0.0 for col in clf_b.feature_names}
        row["acc_mag_g"] = 1.0
        row["acc_mag_ms2"] = 9.81
        row["acc_z_ms2"] = 9.81
        row["AccZ"] = 9.81
        quiescent_features_b = pd.DataFrame([row])
        
        assessment = agent.process_features(
            timestamp=1.0,
            features_b=quiescent_features_b,
            features_c=None
        )
        assert assessment.maneuver == "None / Nominal", (
            f"D-01 Failure: Quiescent telemetry falsely classified as maneuver: '{assessment.maneuver}'"
        )
        assert assessment.maneuver_id == ManeuverClass.NONE.value
        assert assessment.risk_score == 0.0

    def test_t01_d01_hardened_invalid_and_edge_case_telemetry(self):
        """
        D-01 Hardening: KinematicEventGate must validate input completeness, types,
        and finiteness before kinematic threshold checks.
        Invalid, missing, or non-finite telemetry must return INVALID_OR_INSUFFICIENT,
        must NEVER fabricate 'None / Nominal' with confidence 1.0, and must evaluate
        to INSUFFICIENT_DATA rather than cruising.
        Valid quiet telemetry must bypass, and valid excited telemetry must invoke classifier.
        """
        clf_b = DatasetBManeuverClassifier(model_name="extra_trees")
        agent = IntelligentSafetyAgent(classifier_b=clf_b)
        gate = agent.gate

        # Template for baseline valid quiet frame
        base_quiet = {col: 0.0 for col in clf_b.feature_names}
        base_quiet["acc_horiz_std_3s"] = 0.10
        base_quiet["gyro_z_std_3s"] = 0.02
        base_quiet["acc_mag_ms2"] = 9.81
        base_quiet["gyro_mag_rads"] = 0.05
        base_quiet["gyro_z_rads"] = 0.02

        # 1. Invalid Case A: Empty DataFrame
        empty_cases = [pd.DataFrame(), pd.DataFrame([{}])]
        for empty_df in empty_cases:
            res = gate.evaluate(empty_df)
            assert res.status == GateStatus.INVALID_OR_INSUFFICIENT
            assert res.is_event_detected is False
            ass = agent.process_features(timestamp=1.0, features_b=empty_df)
            assert ass.maneuver != "None / Nominal", "Empty frame must not fabricate nominal maneuver!"
            assert ass.maneuver_id is None
            assert ass.maneuver_confidence == 0.0
            assert ass.safety_state == SafetyState.INSUFFICIENT_DATA

        # 2. Invalid Case B: All-NaN row
        nan_df = pd.DataFrame([{col: np.nan for col in clf_b.feature_names}])
        res_nan = gate.evaluate(nan_df)
        assert res_nan.status == GateStatus.INVALID_OR_INSUFFICIENT
        ass_nan = agent.process_features(timestamp=2.0, features_b=nan_df)
        assert ass_nan.maneuver != "None / Nominal", "All-NaN frame must not fabricate nominal maneuver!"
        assert ass_nan.maneuver_id is None
        assert ass_nan.maneuver_confidence == 0.0
        assert ass_nan.safety_state == SafetyState.INSUFFICIENT_DATA

        # 3. Invalid Case C: Missing one required feature group (e.g., gyro_z missing)
        miss_df = pd.DataFrame([dict(base_quiet)])
        del miss_df["gyro_z_rads"]
        del miss_df["GyroZ"]
        res_miss = gate.evaluate(miss_df)
        assert res_miss.status == GateStatus.INVALID_OR_INSUFFICIENT
        assert "MISSING_REQUIRED_FEATURE" in res_miss.reason
        ass_miss = agent.process_features(timestamp=3.0, features_b=miss_df)
        assert ass_miss.maneuver != "None / Nominal"
        assert ass_miss.maneuver_id is None
        assert ass_miss.safety_state == SafetyState.INSUFFICIENT_DATA

        # 4. Invalid Case D: NaN in each required kinematic feature individually
        required_features = ["acc_horiz_std_3s", "gyro_z_std_3s", "acc_mag_ms2", "gyro_mag_rads", "gyro_z_rads"]
        for feat in required_features:
            f_dict = dict(base_quiet)
            f_dict[feat] = np.nan
            f_df = pd.DataFrame([f_dict])
            res_f = gate.evaluate(f_df)
            assert res_f.status == GateStatus.INVALID_OR_INSUFFICIENT, f"NaN in {feat} must be INVALID_OR_INSUFFICIENT!"
            ass_f = agent.process_features(timestamp=4.0, features_b=f_df)
            assert ass_f.maneuver != "None / Nominal"
            assert ass_f.maneuver_id is None
            assert ass_f.safety_state == SafetyState.INSUFFICIENT_DATA

        # 5. Invalid Case E: Positive infinity in kinematic feature
        inf_dict = dict(base_quiet)
        inf_dict["acc_mag_ms2"] = np.inf
        res_inf = gate.evaluate(pd.DataFrame([inf_dict]))
        assert res_inf.status == GateStatus.INVALID_OR_INSUFFICIENT
        assert "NON_FINITE_VALUE" in res_inf.reason
        ass_inf = agent.process_features(timestamp=5.0, features_b=pd.DataFrame([inf_dict]))
        assert ass_inf.maneuver != "None / Nominal"
        assert ass_inf.maneuver_id is None

        # 6. Invalid Case F: Negative infinity in kinematic feature
        ninf_dict = dict(base_quiet)
        ninf_dict["gyro_z_rads"] = -np.inf
        res_ninf = gate.evaluate(pd.DataFrame([ninf_dict]))
        assert res_ninf.status == GateStatus.INVALID_OR_INSUFFICIENT
        assert "NON_FINITE_VALUE" in res_ninf.reason
        ass_ninf = agent.process_features(timestamp=6.0, features_b=pd.DataFrame([ninf_dict]))
        assert ass_ninf.maneuver != "None / Nominal"
        assert ass_ninf.maneuver_id is None

        # 7. Invalid Case G: Numeric strings and non-numeric types
        str_dict = dict(base_quiet)
        str_dict["acc_horiz_std_3s"] = "0.10"
        res_str = gate.evaluate(pd.DataFrame([str_dict]))
        assert res_str.status == GateStatus.INVALID_OR_INSUFFICIENT
        assert "INVALID_NUMERIC_TYPE" in res_str.reason

        none_dict = dict(base_quiet)
        none_dict["acc_mag_ms2"] = None
        none_dict["acc_mag_g"] = None
        res_none = gate.evaluate(pd.DataFrame([none_dict]))
        assert res_none.status == GateStatus.INVALID_OR_INSUFFICIENT

        # 8. Valid Quiet Telemetry: Must bypass classifier and emit nominal with conf=1.0
        res_quiet = gate.evaluate(pd.DataFrame([base_quiet]))
        assert res_quiet.status == GateStatus.VALID_QUIESCENT
        assert res_quiet.is_event_detected is False
        ass_quiet = agent.process_features(timestamp=8.0, features_b=pd.DataFrame([base_quiet]))
        assert ass_quiet.maneuver == "None / Nominal"
        assert ass_quiet.maneuver_id == ManeuverClass.NONE.value
        assert ass_quiet.maneuver_confidence == 1.0
        assert ass_quiet.risk_score == 0.0

        # 9. Valid Sudden Braking: Must trigger VALID_EXCITED and invoke classifier
        brake_dict = dict(base_quiet)
        brake_dict["acc_horiz_std_3s"] = 0.85
        brake_dict["acc_mag_ms2"] = 11.20
        res_brake = gate.evaluate(pd.DataFrame([brake_dict]))
        assert res_brake.status == GateStatus.VALID_EXCITED
        assert res_brake.is_event_detected is True
        ass_brake = agent.process_features(timestamp=9.0, features_b=pd.DataFrame([brake_dict]))
        assert ass_brake.maneuver_id != ManeuverClass.NONE.value
        assert ass_brake.maneuver != "None / Nominal"

        # 10. Valid Sudden Turning: Must trigger VALID_EXCITED and invoke classifier
        turn_dict = dict(base_quiet)
        turn_dict["gyro_z_std_3s"] = 0.15
        turn_dict["gyro_z_rads"] = 0.35
        turn_dict["gyro_mag_rads"] = 0.40
        res_turn = gate.evaluate(pd.DataFrame([turn_dict]))
        assert res_turn.status == GateStatus.VALID_EXCITED
        assert res_turn.is_event_detected is True
        ass_turn = agent.process_features(timestamp=10.0, features_b=pd.DataFrame([turn_dict]))
        assert ass_turn.maneuver_id != ManeuverClass.NONE.value

        # 11. Valid Acceleration: Must trigger VALID_EXCITED and invoke classifier
        accel_dict = dict(base_quiet)
        accel_dict["acc_horiz_std_3s"] = 0.55
        accel_dict["acc_mag_ms2"] = 11.0
        res_accel = gate.evaluate(pd.DataFrame([accel_dict]))
        assert res_accel.status == GateStatus.VALID_EXCITED
        ass_accel = agent.process_features(timestamp=11.0, features_b=pd.DataFrame([accel_dict]))
        assert ass_accel.maneuver_id != ManeuverClass.NONE.value

        # 12. Noisy Telemetry Immediately Below and Above Thresholds
        # Sub-threshold: acc_horiz_std=0.29 (< 0.30)
        sub_dict = dict(base_quiet)
        sub_dict["acc_horiz_std_3s"] = 0.29
        sub_dict["gyro_z_std_3s"] = 0.049
        sub_dict["acc_mag_ms2"] = 9.80665 + 1.46
        res_sub = gate.evaluate(pd.DataFrame([sub_dict]))
        assert res_sub.status == GateStatus.VALID_QUIESCENT
        assert res_sub.is_event_detected is False

        # Super-threshold: acc_horiz_std=0.31 (>= 0.30)
        sup_dict = dict(base_quiet)
        sup_dict["acc_horiz_std_3s"] = 0.31
        sup_dict["gyro_z_std_3s"] = 0.049
        sup_dict["acc_mag_ms2"] = 9.80665 + 1.46
        res_sup = gate.evaluate(pd.DataFrame([sup_dict]))
        assert res_sup.status == GateStatus.VALID_EXCITED
        assert res_sup.is_event_detected is True

    # -------------------------------------------------------------------------
    # T-02 (Defect D-02): Maneuver Event-Count Inflation across Consecutive Frames
    # -------------------------------------------------------------------------
    def test_t02_d02_consecutive_maneuver_streaming_deduplication(self):
        """
        D-02: Streaming consecutive frames of the same physical maneuver must NOT
        append multiple distinct events to _maneuver_events or trigger Rule 8.
        """
        agent = IntelligentSafetyAgent()
        
        # Stream 5 consecutive frames of the same Sudden Deceleration event across 0.8s
        timestamps = [1.0, 1.2, 1.4, 1.6, 1.8]
        assessments = []
        for ts in timestamps:
            ass = agent.process_predictions(
                timestamp=ts,
                style_prediction=0,
                style_confidence=0.90,
                maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
                maneuver_confidence=0.90
            )
            assessments.append(ass)
            
        final_assessment = assessments[-1]
        assert "REPEATED_ERRATIC_MANEUVERS" not in final_assessment.triggered_rules, (
            f"D-02 Failure: Rule 8 falsely triggered on single continuous maneuver! Rules: {final_assessment.triggered_rules}"
        )
        assert agent.smoother.get_context(1.8).recent_maneuver_count == 1, (
            f"D-02 Failure: Maneuver count inflated to {agent.smoother.get_context(1.8).recent_maneuver_count}"
        )

    # -------------------------------------------------------------------------
    # T-03 (Defect D-03): Out-of-Order Timestamp Permanent Lockout Recovery
    # -------------------------------------------------------------------------
    def test_t03_d03_out_of_order_timestamp_lockout_recovery(self):
        """
        D-03: An out-of-order packet or clock jump must not permanently lock the
        validator into an INVALID state for all subsequent valid monotonic frames.
        """
        validator = PredictionValidator()
        
        # Frame 1: Valid initial packet
        r1 = validator.validate(RawPredictionInput(timestamp=10.0, style_prediction=0, style_confidence=0.9))
        assert r1.is_valid is True
        
        # Frame 2: Glitch / Out-of-order jump packet
        r2 = validator.validate(RawPredictionInput(timestamp=999.0, style_prediction=0, style_confidence=0.9))
        
        # Frame 3: Normal chronological packet continuing from t=10.0
        r3 = validator.validate(RawPredictionInput(timestamp=10.1, style_prediction=0, style_confidence=0.9))
        
        assert r3.is_valid is True, (
            f"D-03 Failure: Validator permanently locked out after clock jump. Errors: {r3.validation_errors}"
        )
        assert r3.data_quality_status == "VALID"

    # -------------------------------------------------------------------------
    # T-04 (Defect D-04): Documentation vs. Code Parameter Divergence
    # -------------------------------------------------------------------------
    def test_t04_d04_risk_engine_parameters_match_specification(self):
        """
        D-04: Asserts that risk configuration parameters match the authoritative
        specification: baseline R0 = 0.0, Decel = 50.0, Turn = 35.0, Compound = 25.0.
        """
        config = DEFAULT_CONFIG
        assert config.baseline_risk_score == 0.0, f"D-04 Failure: Expected baseline R0=0.0, got {config.baseline_risk_score}"
        assert config.weight_sudden_deceleration == 50.0, f"D-04 Failure: Expected Decel=50.0, got {config.weight_sudden_deceleration}"
        assert config.weight_sudden_turn == 35.0, f"D-04 Failure: Expected Turn=35.0, got {config.weight_sudden_turn}"
        assert config.weight_compound_high_risk == 25.0, f"D-04 Failure: Expected Compound=25.0, got {config.weight_compound_high_risk}"
        assert config.threshold_caution == 25.0
        assert config.threshold_warning == 50.0
        assert config.threshold_high_risk == 75.0

    # -------------------------------------------------------------------------
    # T-05 (Defect D-05): Continuous Time-Dependent EMA Invariance
    # -------------------------------------------------------------------------
    def test_t05_d05_continuous_time_dependent_ema_invariance(self):
        """
        D-05: Discrete fixed-rate EMA (alpha = 0.20) must be replaced by continuous
        time-dependent EMA alpha(dt) = 1 - exp(-dt / tau). Decay over total elapsed
        time dt = 1.0s must be invariant across 20 Hz vs 2 Hz discretization.
        """
        smoother_fast = TemporalSmoother()
        smoother_slow = TemporalSmoother()
        
        # Initialize both with aggressive target
        v_init = ValidatedPrediction(timestamp=0.0, is_valid=True, style_prediction=1, style_confidence=1.0)
        smoother_fast.update(v_init)
        smoother_slow.update(v_init)
        
        # Stream A: 20 steps of dt = 0.05s
        for i in range(1, 21):
            smoother_fast.update(ValidatedPrediction(timestamp=i * 0.05, is_valid=True, style_prediction=0, style_confidence=1.0))
            
        # Stream B: 2 steps of dt = 0.50s
        for i in range(1, 3):
            smoother_slow.update(ValidatedPrediction(timestamp=i * 0.50, is_valid=True, style_prediction=0, style_confidence=1.0))
            
        ema_fast = smoother_fast.get_context(1.0).style_ema
        ema_slow = smoother_slow.get_context(1.0).style_ema
        
        assert math.isclose(ema_fast, ema_slow, abs_tol=0.05), (
            f"D-05 Failure: Discrete EMA is sample-rate dependent! 20Hz EMA={ema_fast:.4f} vs 2Hz EMA={ema_slow:.4f}"
        )

    # -------------------------------------------------------------------------
    # T-06 (Defect D-06): Continuous Half-Life Decay & Strict 2.50pt Retirement
    # -------------------------------------------------------------------------
    def test_t06_d06_continuous_exponential_maneuver_decay_and_retirement(self):
        """
        D-06: Maneuver residual risk must follow continuous half-life decay (tau_1/2 = 0.75s)
        after a 1.0s plateau, and strictly retire to 0.00 pts when elapsed time exceeds the
        analytical boundary:
            t_retire = plateau_seconds + half_life_seconds * log2(initial_weight / retirement_threshold)

        For each weight (W=50.0 and W=35.0), verifies:
        1. Analytical retirement boundary (dt == t_retire) -> exactly 2.50 pts.
        2. Immediately preceding representable float (math.nextafter(t_retire, -inf)) -> > 2.50 pts.
        3. Immediately following representable float (math.nextafter(t_retire, +inf)) -> 0.00 pts.
        4. Clearly post-boundary time (dt = t_retire + 0.5s) -> 0.00 pts.
        5. Exact 1.00s plateau boundary (dt = 1.00s) -> full weight W.
        6. Half-life point (dt = 1.75s) -> W / 2.0.
        7. Pre-plateau points (dt = 0.0s, dt = 0.50s) -> full weight W; decay onset (dt = 1.01s) -> R < W.
        8. Negative dt handled safely (dt = -1.0s) -> full weight W within plateau.
        9. Exhaustive parameter validation (non-positive, non-finite, sub-threshold, booleans, invalid types).
        10. Determinism across repeated calls.
        """
        engine = DeterministicRiskEngine()
        assert hasattr(engine, "calculate_residual_risk"), (
            "D-06 Failure: DeterministicRiskEngine lacks calculate_residual_risk() method!"
        )

        for W in [50.0, 35.0]:
            p = 1.00
            h = 0.75
            thresh = 2.50
            t_retire = p + h * math.log2(W / thresh)
            t_prev = math.nextafter(t_retire, -math.inf)
            t_next = math.nextafter(t_retire, math.inf)

            # 1. Analytical retirement boundary -> exactly 2.50 retained
            r_exact = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=t_retire)
            assert r_exact == thresh, f"Expected exactly {thresh} at analytical boundary, got {r_exact}"

            # 2. Immediately preceding representable float -> residual retained strictly > 2.50
            r_prev = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=t_prev)
            assert r_prev > thresh, f"Expected residual > {thresh} at preceding float, got {r_prev}"
            assert math.isclose(r_prev, thresh, abs_tol=1e-9)

            # 3. Immediately following representable float -> strictly 0.00
            r_next = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=t_next)
            assert r_next == 0.00, f"Expected 0.00 at next float after boundary, got {r_next}"

            # 4. Clearly post-boundary time -> strictly 0.00
            r_post = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=t_retire + 0.5)
            assert r_post == 0.00, f"Expected 0.00 at clearly post-boundary time, got {r_post}"

            # 5. Exact 1.00-second plateau boundary -> full weight W
            r_plateau_end = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=1.00)
            assert r_plateau_end == W, f"Expected full weight {W} at plateau end, got {r_plateau_end}"

            # 6. Half-life point (dt = 1.75s) -> exactly W / 2.0
            r_half_life = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=1.75)
            assert math.isclose(r_half_life, W / 2.0, abs_tol=1e-12), f"Expected {W/2.0} at 1 half-life, got {r_half_life}"

            # 7. Plateau points and decay onset
            assert engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=0.0) == W
            assert engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=0.50) == W
            r_decay_start = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=1.01)
            assert thresh < r_decay_start < W, f"Expected decay onset at dt=1.01, got {r_decay_start}"

            # 8. Negative dt handled safely (retains initial weight within plateau without overflow)
            r_neg = engine.calculate_residual_risk(initial_weight=W, elapsed_seconds=-1.0)
            assert r_neg == W, f"Negative dt should safely retain full weight within plateau, got {r_neg}"

        # 9. Exhaustive parameter validation
        # Non-positive parameters
        assert engine.calculate_residual_risk(50.0, 1.0, plateau_seconds=0.0) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, plateau_seconds=-1.0) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, half_life_seconds=0.0) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, half_life_seconds=-0.75) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, retirement_threshold=0.0) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, retirement_threshold=-2.5) == 0.0
        assert engine.calculate_residual_risk(initial_weight=0.0, elapsed_seconds=1.0) == 0.0
        assert engine.calculate_residual_risk(initial_weight=-50.0, elapsed_seconds=1.0) == 0.0

        # Sub-threshold initial weight (initial_weight < retirement_threshold)
        assert engine.calculate_residual_risk(initial_weight=2.0, elapsed_seconds=0.5, retirement_threshold=2.50) == 0.0
        # Exactly equal to threshold at dt=0 -> returns threshold (2.50)
        assert engine.calculate_residual_risk(initial_weight=2.50, elapsed_seconds=0.0, retirement_threshold=2.50) == 2.50
        # Exactly equal to threshold after plateau -> returns 0.0
        assert engine.calculate_residual_risk(initial_weight=2.50, elapsed_seconds=1.5, retirement_threshold=2.50) == 0.0

        # Non-finite inputs
        assert engine.calculate_residual_risk(float("nan"), 1.0) == 0.0
        assert engine.calculate_residual_risk(50.0, float("nan")) == 0.0
        assert engine.calculate_residual_risk(float("inf"), 1.0) == 0.0
        assert engine.calculate_residual_risk(50.0, float("inf")) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, plateau_seconds=float("inf")) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, half_life_seconds=float("nan")) == 0.0
        assert engine.calculate_residual_risk(50.0, 1.0, retirement_threshold=float("inf")) == 0.0

        # Booleans and invalid types rejected safely
        assert engine.calculate_residual_risk(True, 1.0) == 0.0
        assert engine.calculate_residual_risk(50.0, False) == 0.0
        assert engine.calculate_residual_risk("invalid", 1.0) == 0.0
        assert engine.calculate_residual_risk(50.0, None) == 0.0

        # 10. Determinism across repeated calls
        results = [engine.calculate_residual_risk(50.0, 2.5) for _ in range(100)]
        assert len(set(results)) == 1, "calculate_residual_risk is non-deterministic across repeated calls!"

    # -------------------------------------------------------------------------
    # T-07 (Defect D-07): Future Timestamp Rejection & Queue Preservation
    # -------------------------------------------------------------------------
    def test_t07_d07_future_timestamp_rejection_and_queue_preservation(self):
        """
        D-07: Ingress packets dated in the future (t > current_time) must be rejected
        and must NOT purge history buffers in TemporalSmoother.
        """
        validator = PredictionValidator()
        smoother = TemporalSmoother()
        
        # Populate history at t = 10.0s
        v1 = validator.validate(RawPredictionInput(timestamp=10.0, style_prediction=1, style_confidence=0.9), current_time=10.0)
        smoother.update(v1)
        assert len(smoother._history) == 1
        
        # Inject future timestamp (ts = 15.0s when current_time = 10.0s)
        raw_future = RawPredictionInput(timestamp=15.0, style_prediction=1, style_confidence=0.9)
        v_future = validator.validate(raw_future, current_time=10.0)
        
        assert v_future.is_valid is False, "D-07 Failure: Future timestamp was accepted as valid!"
        assert any("future" in err.lower() for err in v_future.validation_errors)
        assert len(smoother._history) == 1

    # -------------------------------------------------------------------------
    # T-08 (Defect D-08): Missing Style Prediction Not Defaulted to Normal
    # -------------------------------------------------------------------------
    def test_t08_d08_missing_style_prediction_not_defaulted_to_normal(self):
        """
        D-08: When a packet contains only maneuver data (style_prediction=None),
        TemporalSmoother must NOT coerce style to 0 (Normal Driving).
        """
        smoother = TemporalSmoother()
        
        # Feed packet with style = None
        v = ValidatedPrediction(
            timestamp=1.0,
            is_valid=True,
            style_prediction=None,
            style_confidence=None,
            maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
            maneuver_confidence=0.90
        )
        smoother.update(v)
        ctx = smoother.get_context(1.0)
        
        assert ctx.current_style is None, (
            f"D-08 Failure: Missing style prediction was silently coerced to {ctx.current_style} (Normal)!"
        )

    # -------------------------------------------------------------------------
    # T-09 (Defect D-09): Dissimilar Maneuver Alerts Bypass Category Cooldown
    # -------------------------------------------------------------------------
    def test_t09_d09_dissimilar_maneuver_alerts_bypass_category_cooldown(self):
        """
        D-09: Sudden Deceleration followed immediately by Sudden Turn must emit
        two distinct alerts; the turn alert must not be suppressed by category cooldown.
        """
        engine = DeterministicRiskEngine()
        
        # Event 1: Sudden Decel at t = 1.0s
        ctx1 = SmoothedContext(
            timestamp=1.0,
            style_ema=0.1,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.90,
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res1 = engine.evaluate(ctx1)
        assert len(res1.active_alerts) >= 1
        
        # Event 2: Sudden Turn at t = 2.0s (1.0s elapsed < 3.0s category cooldown)
        ctx2 = SmoothedContext(
            timestamp=2.0,
            style_ema=0.1,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.9,
            active_maneuver=ManeuverClass.SUDDEN_RIGHT_TURN.value,
            active_maneuver_confidence=0.90,
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=True,
            recent_maneuver_count=2
        )
        res2 = engine.evaluate(ctx2)
        
        assert len(res2.active_alerts) >= 1, (
            "D-09 Failure: Dissimilar maneuver alert (Turn) was silenced by Deceleration cooldown!"
        )
        assert any("Turn" in a.message for a in res2.active_alerts)

    # -------------------------------------------------------------------------
    # T-10 (Defect D-10): Unified Three-Tier Confidence Scaling
    # -------------------------------------------------------------------------
    def test_t10_d10_unified_three_tier_confidence_scaling(self):
        """
        D-10: Risk contribution must scale independently by modality:
        phi(C) = 0 for C < 0.40; phi(C) = C/0.60 for 0.40 <= C < 0.60; phi(C) = 1.0 for C >= 0.60.
        A high-confidence style (0.95) must NOT mask an uncertain maneuver (0.35).
        """
        engine = DeterministicRiskEngine()
        
        # Maneuver with C = 0.35 (noise deadband) with high style confidence (0.95)
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.1,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.95,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.35,  # C < 0.40 -> 0.0 pts!
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res = engine.evaluate(ctx)
        
        assert res.risk_score == 0.0, (
            f"D-10 Failure: Low-confidence maneuver (C=0.35) contributed risk! Score: {res.risk_score}"
        )

    # -------------------------------------------------------------------------
    # T-11 (Defect D-11): Inference Exceptions Logged and Flagged Degraded
    # -------------------------------------------------------------------------
    def test_t11_d11_inference_exceptions_logged_and_flagged_degraded(self):
        """
        D-11: Bare except Exception: blocks in safety_agent.py must not silently
        mask quarantined features or schema errors. Pipeline health must become DEGRADED.
        """
        agent = IntelligentSafetyAgent(auto_load_classifiers=True)
        
        # Pass features with anti-causal quarantined features that raise ValueError in classifier_b
        bad_features_b = pd.DataFrame([{
            "event_duration_samples": 10,
            "event_progress_ratio": 0.5,
            "acc_horiz_mean_3s": 1.0
        }])
        
        assessment = agent.process_features(
            timestamp=1.0,
            features_b=bad_features_b,
            features_c=None
        )
        
        assert assessment.is_valid is False or assessment.data_quality_status in ["DEGRADED", "ERROR", "DEGRADED_SENSORS"], (
            f"D-11 Failure: Inference exception was silently masked! Status: {assessment.data_quality_status}, Valid: {assessment.is_valid}"
        )

    # -------------------------------------------------------------------------
    # T-12 (Defect D-12): Telemetry Gap Resets Stale Style EMA
    # -------------------------------------------------------------------------
    def test_t12_d12_telemetry_gap_resets_stale_style_ema(self):
        """
        D-12: A multi-minute telemetry dropout (> 5.0s) must reset pre-gap
        aggressive EMA rather than carrying it into the new session.
        """
        smoother = TemporalSmoother()
        
        # Session 1: Aggressive driving at t = 10.0s
        for t in np.linspace(1.0, 10.0, 10):
            smoother.update(ValidatedPrediction(timestamp=t, is_valid=True, style_prediction=1, style_confidence=0.95))
        assert smoother.get_context(10.0).style_ema > 0.70
        
        # Long gap: 110s elapse -> t = 120.0s with Normal Driving
        smoother.update(ValidatedPrediction(timestamp=120.0, is_valid=True, style_prediction=0, style_confidence=0.95))
        ctx = smoother.get_context(120.0)
        
        assert ctx.style_ema < 0.25, (
            f"D-12 Failure: Pre-gap aggressive EMA persisted across 110s gap! EMA: {ctx.style_ema}"
        )

    # -------------------------------------------------------------------------
    # T-13 (Defect D-13): Independent Pipeline Health Matrix
    # -------------------------------------------------------------------------
    def test_t13_d13_independent_pipeline_health_matrix(self):
        """
        D-13: System must track Pipeline A and Pipeline B health independently,
        forbidding a NORMAL state when Pipeline B is missing/degraded.
        """
        agent = IntelligentSafetyAgent()
        
        # Ingest only Pipeline A (style=Normal), Pipeline B missing
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.95,
            maneuver_prediction=None,
            maneuver_confidence=None
        )
        
        assert hasattr(assessment, "pipeline_a_health") and hasattr(assessment, "pipeline_b_health"), (
            "D-13 Failure: SafetyAssessment lacks independent pipeline health fields!"
        )
        assert assessment.pipeline_a_health == "HEALTHY"
        assert assessment.pipeline_b_health == "MISSING"
        assert assessment.safety_state != SafetyState.NORMAL, (
            "D-13 Failure: System emitted NORMAL state despite Pipeline B being completely missing!"
        )

    # -------------------------------------------------------------------------
    # T-14 (Defect D-14): Harsh Braking Triggers WARNING Unweakened
    # -------------------------------------------------------------------------
    def test_t14_d14_harsh_braking_triggers_warning_unweakened(self):
        """
        D-14: Sudden Deceleration (Harsh Braking) was specified to trigger WARNING,
        not CAUTION. Weight must be 50.0 pts, reaching the WARNING threshold (50.0).
        """
        engine = DeterministicRiskEngine()
        
        ctx = SmoothedContext(
            timestamp=1.0,
            style_ema=0.0,
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=0,
            current_style_confidence=0.95,
            active_maneuver=ManeuverClass.SUDDEN_DECELERATION.value,
            active_maneuver_confidence=0.90,
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=True,
            recent_maneuver_count=1
        )
        res = engine.evaluate(ctx)
        
        assert res.risk_score >= 50.0, f"D-14 Failure: Harsh braking risk score is {res.risk_score} (< 50.0)"
        assert res.safety_state == SafetyState.WARNING, (
            f"D-14 Failure: Harsh braking produced {res.safety_state.value}, expected WARNING!"
        )

    # -------------------------------------------------------------------------
    # T-15 (Defect D-15): Invalid Confidence Rejection Policy
    # -------------------------------------------------------------------------
    def test_t15_d15_invalid_confidence_rejection_policy(self):
        """
        D-15: Confidence values outside [0.0, 1.0] must be explicitly rejected
        with is_valid=False and INVALID status, eliminating deceptive clamping claims.
        """
        validator = PredictionValidator()
        
        # Test out-of-bounds confidence (1.5)
        raw_high = RawPredictionInput(timestamp=1.0, style_prediction=0, style_confidence=1.5)
        res_high = validator.validate(raw_high)
        assert res_high.is_valid is False
        assert res_high.data_quality_status == "INVALID"
        assert any("confidence" in err.lower() for err in res_high.validation_errors)
        
        # Test negative confidence (-0.2)
        raw_neg = RawPredictionInput(timestamp=1.0, style_prediction=0, style_confidence=-0.2)
        res_neg = validator.validate(raw_neg)
        assert res_neg.is_valid is False
        assert res_neg.data_quality_status == "INVALID"

    # -------------------------------------------------------------------------
    # T-16 (Integration): Production Artifact Hashes & Decision Threshold
    # -------------------------------------------------------------------------
    def test_t16_production_artifact_hashes_threshold_and_end_to_end(self):
        """
        T-16: Verifies SHA-256 hashes of all 4 Stage 5 artifacts, proves Dataset C
        scaler execution, and tests exact threshold boundary behavior (0.8399, 0.8400, 0.8401).
        """
        # 1. Verify artifact hashes
        artifacts = {
            "b_model": (REPO_ROOT / "models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib",
                        "552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90"),
            "b_scaler": (REPO_ROOT / "models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib",
                         "d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576"),
            "c_model": (REPO_ROOT / "models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib",
                        "a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc"),
            "c_scaler": (REPO_ROOT / "models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib",
                        "d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5"),
        }
        for name, (path, expected_hash) in artifacts.items():
            assert path.exists(), f"Artifact missing: {path}"
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            
        # 2. Instantiate canonical wrappers
        clf_c = DatasetCBinaryClassifier(model_dir=str(REPO_ROOT / "models" / "experiments" / "dataset_c"))
        clf_b = DatasetBManeuverClassifier(model_name="extra_trees")
        
        assert clf_c.decision_threshold == 0.84, f"Expected 0.84 threshold, got {clf_c.decision_threshold}"
        assert clf_c.requires_scaler is True
        assert clf_b.requires_scaling is False
        
        # 3. Test threshold boundary operator (>= 0.84)
        class MockClassifier(DatasetCBinaryClassifier):
            def __init__(self, prob_val, base):
                self.decision_threshold = base.decision_threshold
                self.prob_val = prob_val
            def predict_proba(self, X):
                return np.array([[1.0 - self.prob_val, self.prob_val]])
                
        dummy = pd.DataFrame([{"dummy": 1}])
        # Strictly below: 0.8399 -> 0
        assert DatasetCBinaryClassifier.predict(MockClassifier(0.8399, clf_c), dummy)[0] == 0
        # Exactly equal: 0.8400 -> 1
        assert DatasetCBinaryClassifier.predict(MockClassifier(0.8400, clf_c), dummy)[0] == 1
        # Strictly above: 0.8401 -> 1
        assert DatasetCBinaryClassifier.predict(MockClassifier(0.8401, clf_c), dummy)[0] == 1


# =============================================================================
# Stage 6 Phase 1C: Comprehensive 20-Row D-13 Fail-Safe Health Matrix Tests
# =============================================================================
class TestD13FailSafeHealthMatrix:
    """
    Validates all 20 reachable rows of the authoritative D-13 Fail-Safe Truth Table.
    Integrates P-05A (Healthy A + Impaired B clamps to CAUTION floor) and
    Policy 1A-Clamped (Active maneuver during A outage clamped >= CAUTION,
    cruising during A outage returns INSUFFICIENT_DATA).
    Treats QUIESCENT_BYPASS as healthy-by-design.
    """

    def test_row01_healthy_a_healthy_b_allows_normal(self):
        """Row 1: HEALTHY A + HEALTHY B allows NORMAL when driving is nominal."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.95,
            maneuver_prediction=0,
            maneuver_confidence=0.90
        )
        assert assessment.pipeline_a_health == "HEALTHY"
        assert assessment.pipeline_b_health == "HEALTHY"
        assert assessment.safety_state == SafetyState.NORMAL
        assert not any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row02_healthy_a_degraded_b_p05a_clamps_to_caution(self):
        """Row 2 (P-05A): HEALTHY A + DEGRADED B clamps nominal style to CAUTION and raises DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.95,
            pipeline_b_health="DEGRADED"
        )
        assert assessment.pipeline_a_health == "HEALTHY"
        assert assessment.pipeline_b_health == "DEGRADED"
        assert assessment.safety_state == SafetyState.CAUTION
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row03_healthy_a_missing_b_p05a_clamps_to_caution(self):
        """Row 3 (P-05A): HEALTHY A + MISSING B clamps nominal style to CAUTION and raises DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.95,
            maneuver_prediction=None,
            maneuver_confidence=None
        )
        assert assessment.pipeline_a_health == "HEALTHY"
        assert assessment.pipeline_b_health == "MISSING"
        assert assessment.safety_state == SafetyState.CAUTION
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row04_healthy_a_invalid_b_p05a_clamps_to_caution(self):
        """Row 4 (P-05A): HEALTHY A + INVALID B clamps nominal style to CAUTION and raises DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.95,
            maneuver_prediction=99,
            maneuver_confidence=0.90
        )
        assert assessment.pipeline_a_health == "HEALTHY"
        assert assessment.pipeline_b_health == "INVALID"
        assert assessment.safety_state == SafetyState.CAUTION
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row05_healthy_a_quiescent_bypass_b_allows_normal(self):
        """Row 5: HEALTHY A + QUIESCENT_BYPASS B is healthy-by-design; allows NORMAL with no DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=0,
            style_confidence=0.95,
            pipeline_b_health="QUIESCENT_BYPASS"
        )
        assert assessment.pipeline_a_health == "HEALTHY"
        assert assessment.pipeline_b_health == "QUIESCENT_BYPASS"
        assert assessment.safety_state == SafetyState.NORMAL
        assert not any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row06_degraded_a_healthy_b_policy1a_clamped(self):
        """Row 6 (Policy 1A-Clamped): DEGRADED A + HEALTHY B returns INSUFFICIENT_DATA if cruising,
        or calculates behavioral risk clamped >= CAUTION if active maneuver."""
        agent = IntelligentSafetyAgent()
        # Subcase A: Nominal Cruising (M=0)
        ass_cruise = agent.process_predictions(
            timestamp=1.0,
            pipeline_a_health="DEGRADED",
            maneuver_prediction=0,
            maneuver_confidence=0.90
        )
        assert ass_cruise.pipeline_a_health == "DEGRADED"
        assert ass_cruise.pipeline_b_health == "HEALTHY"
        assert ass_cruise.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in ass_cruise.active_alerts)

        # Subcase B: Active Maneuver (Harsh Braking M=2)
        agent.reset()
        ass_man = agent.process_predictions(
            timestamp=1.0,
            pipeline_a_health="DEGRADED",
            maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
            maneuver_confidence=0.90
        )
        assert ass_man.pipeline_a_health == "DEGRADED"
        assert ass_man.pipeline_b_health == "HEALTHY"
        assert ass_man.safety_state == SafetyState.WARNING
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in ass_man.active_alerts)

    def test_row07_degraded_a_degraded_b_insufficient_data(self):
        """Row 7: DEGRADED A + DEGRADED B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            pipeline_a_health="DEGRADED",
            pipeline_b_health="DEGRADED"
        )
        assert assessment.pipeline_a_health == "DEGRADED"
        assert assessment.pipeline_b_health == "DEGRADED"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row08_degraded_a_missing_b_insufficient_data(self):
        """Row 8: DEGRADED A + MISSING B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            pipeline_a_health="DEGRADED",
            pipeline_b_health="MISSING"
        )
        assert assessment.pipeline_a_health == "DEGRADED"
        assert assessment.pipeline_b_health == "MISSING"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row09_degraded_a_invalid_b_insufficient_data(self):
        """Row 9: DEGRADED A + INVALID B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            pipeline_a_health="DEGRADED",
            pipeline_b_health="INVALID"
        )
        assert assessment.pipeline_a_health == "DEGRADED"
        assert assessment.pipeline_b_health == "INVALID"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row10_degraded_a_quiescent_bypass_b_insufficient_data(self):
        """Row 10: DEGRADED A + QUIESCENT_BYPASS B returns INSUFFICIENT_DATA because style is uncertifiable."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            pipeline_a_health="DEGRADED",
            pipeline_b_health="QUIESCENT_BYPASS"
        )
        assert assessment.pipeline_a_health == "DEGRADED"
        assert assessment.pipeline_b_health == "QUIESCENT_BYPASS"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row11_missing_a_healthy_b_policy1a_clamped(self):
        """Row 11 (Policy 1A-Clamped): MISSING A + HEALTHY B returns INSUFFICIENT_DATA if cruising,
        or calculates behavioral risk clamped >= CAUTION if active maneuver."""
        agent = IntelligentSafetyAgent()
        # Subcase A: Nominal Cruising (M=0)
        ass_cruise = agent.process_predictions(
            timestamp=1.0,
            style_prediction=None,
            maneuver_prediction=0,
            maneuver_confidence=0.90
        )
        assert ass_cruise.pipeline_a_health == "MISSING"
        assert ass_cruise.pipeline_b_health == "HEALTHY"
        assert ass_cruise.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in ass_cruise.active_alerts)

        # Subcase B: Active Maneuver (Sudden Acceleration M=1)
        agent.reset()
        ass_man = agent.process_predictions(
            timestamp=1.0,
            style_prediction=None,
            maneuver_prediction=ManeuverClass.SUDDEN_ACCELERATION.value,
            maneuver_confidence=0.90
        )
        assert ass_man.pipeline_a_health == "MISSING"
        assert ass_man.pipeline_b_health == "HEALTHY"
        assert ass_man.safety_state == SafetyState.CAUTION
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in ass_man.active_alerts)

    def test_row12_missing_a_degraded_b_insufficient_data(self):
        """Row 12: MISSING A + DEGRADED B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=None,
            pipeline_b_health="DEGRADED"
        )
        assert assessment.pipeline_a_health == "MISSING"
        assert assessment.pipeline_b_health == "DEGRADED"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row13_missing_a_missing_b_insufficient_data(self):
        """Row 13: MISSING A + MISSING B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=None,
            maneuver_prediction=None
        )
        assert assessment.pipeline_a_health == "MISSING"
        assert assessment.pipeline_b_health == "MISSING"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row14_missing_a_invalid_b_insufficient_data(self):
        """Row 14: MISSING A + INVALID B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=None,
            maneuver_prediction=99,
            maneuver_confidence=0.90
        )
        assert assessment.pipeline_a_health == "MISSING"
        assert assessment.pipeline_b_health == "INVALID"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row15_missing_a_quiescent_bypass_b_insufficient_data(self):
        """Row 15: MISSING A + QUIESCENT_BYPASS B returns INSUFFICIENT_DATA because style is uncertifiable."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=None,
            pipeline_b_health="QUIESCENT_BYPASS"
        )
        assert assessment.pipeline_a_health == "MISSING"
        assert assessment.pipeline_b_health == "QUIESCENT_BYPASS"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row16_invalid_a_healthy_b_policy1a_clamped(self):
        """Row 16 (Policy 1A-Clamped): INVALID A + HEALTHY B returns INSUFFICIENT_DATA if cruising,
        or calculates behavioral risk clamped >= CAUTION if active maneuver."""
        agent = IntelligentSafetyAgent()
        # Subcase A: Nominal Cruising (M=0)
        ass_cruise = agent.process_predictions(
            timestamp=1.0,
            style_prediction=99,
            style_confidence=0.90,
            maneuver_prediction=0,
            maneuver_confidence=0.90
        )
        assert ass_cruise.pipeline_a_health == "INVALID"
        assert ass_cruise.pipeline_b_health == "HEALTHY"
        assert ass_cruise.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in ass_cruise.active_alerts)

        # Subcase B: Active Maneuver (Sudden Right Turn M=3)
        agent.reset()
        ass_man = agent.process_predictions(
            timestamp=1.0,
            style_prediction=99,
            style_confidence=0.90,
            maneuver_prediction=ManeuverClass.SUDDEN_RIGHT_TURN.value,
            maneuver_confidence=0.90
        )
        assert ass_man.pipeline_a_health == "INVALID"
        assert ass_man.pipeline_b_health == "HEALTHY"
        assert ass_man.safety_state == SafetyState.CAUTION
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in ass_man.active_alerts)

    def test_row17_invalid_a_degraded_b_insufficient_data(self):
        """Row 17: INVALID A + DEGRADED B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=99,
            pipeline_b_health="DEGRADED"
        )
        assert assessment.pipeline_a_health == "INVALID"
        assert assessment.pipeline_b_health == "DEGRADED"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row18_invalid_a_missing_b_insufficient_data(self):
        """Row 18: INVALID A + MISSING B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=99,
            maneuver_prediction=None
        )
        assert assessment.pipeline_a_health == "INVALID"
        assert assessment.pipeline_b_health == "MISSING"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row19_invalid_a_invalid_b_insufficient_data(self):
        """Row 19: INVALID A + INVALID B returns INSUFFICIENT_DATA and DQ alert."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=99,
            maneuver_prediction=99
        )
        assert assessment.pipeline_a_health == "INVALID"
        assert assessment.pipeline_b_health == "INVALID"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)

    def test_row20_invalid_a_quiescent_bypass_b_insufficient_data(self):
        """Row 20: INVALID A + QUIESCENT_BYPASS B returns INSUFFICIENT_DATA because style is uncertifiable."""
        agent = IntelligentSafetyAgent()
        assessment = agent.process_predictions(
            timestamp=1.0,
            style_prediction=99,
            pipeline_b_health="QUIESCENT_BYPASS"
        )
        assert assessment.pipeline_a_health == "INVALID"
        assert assessment.pipeline_b_health == "QUIESCENT_BYPASS"
        assert assessment.safety_state == SafetyState.INSUFFICIENT_DATA
        assert any(a["category"] == AlertCategory.DATA_QUALITY.value for a in assessment.active_alerts)



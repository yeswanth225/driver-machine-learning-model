"""
Intelligent Driving Safety Monitoring Agent
===========================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

import time
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
import numpy as np

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    SafetyState,
    DrivingStyle,
    ManeuverClass,
    GateStatus,
    EventGateStatus,
    STYLE_LABEL_NAMES,
    MANEUVER_LABEL_NAMES
)
from ml.agent.prediction_validator import (
    RawPredictionInput,
    ValidatedPrediction,
    PredictionValidator
)
from ml.agent.temporal_smoother import TemporalSmoother, SmoothedContext
from ml.agent.risk_engine import DeterministicRiskEngine, RuleEvaluationResult, SafetyAlert
from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier
from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier


@dataclass
class SafetyAssessment:
    """Unified Safety Assessment Output produced by the Safety Monitoring Agent."""
    timestamp: float
    safety_state: SafetyState
    risk_score: float  # Bounded [0.0, 100.0]
    driving_style: str
    driving_style_id: Optional[int]
    driving_style_confidence: float
    maneuver: str
    maneuver_id: Optional[int]
    maneuver_confidence: float
    confidence_summary: Dict[str, float]
    triggered_rules: List[str]
    explanation: str
    active_alerts: List[Dict[str, Any]]
    data_quality_status: str
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to clean dictionary format."""
        d = asdict(self)
        d["safety_state"] = self.safety_state.value
        return d


@dataclass(frozen=True)
class GateResult:
    """Strongly-typed diagnostic result emitted by KinematicEventGate."""
    status: GateStatus
    is_event_detected: bool
    reason: str

    def __iter__(self):
        """Enables tuple unpacking: (status, reason) = gate.evaluate(...)"""
        return iter((self.status, self.reason))


class KinematicEventGate:
    """Provisional Kinematic Event Gate for Pipeline B (Dataset B)."""

    def __init__(self, config: SafetyAgentConfig = DEFAULT_CONFIG):
        self.config = config

    def evaluate(self, features: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> GateResult:
        """
        Evaluate whether the kinematic feature frame indicates an active physical maneuver.
        Explicitly validates input completeness, types, and finiteness before threshold checks.
        """
        if not getattr(self.config, "gate_enabled", True):
            return GateResult(
                status=GateStatus.VALID_EXCITED,
                is_event_detected=True,
                reason="GATE_DISABLED"
            )

        # 1. Structural / Type Validation
        if features is None:
            return GateResult(
                status=GateStatus.INVALID_OR_INSUFFICIENT,
                is_event_detected=False,
                reason="NULL_TELEMETRY: Feature frame is None."
            )

        if isinstance(features, dict):
            if not features:
                return GateResult(
                    status=GateStatus.INVALID_OR_INSUFFICIENT,
                    is_event_detected=False,
                    reason="EMPTY_TELEMETRY: Feature dictionary is empty."
                )
            row = features
        elif isinstance(features, pd.DataFrame):
            if features.empty or len(features) == 0:
                return GateResult(
                    status=GateStatus.INVALID_OR_INSUFFICIENT,
                    is_event_detected=False,
                    reason="EMPTY_TELEMETRY: Feature DataFrame has 0 rows."
                )
            row = features.iloc[0].to_dict()
        elif isinstance(features, pd.Series):
            if features.empty:
                return GateResult(
                    status=GateStatus.INVALID_OR_INSUFFICIENT,
                    is_event_detected=False,
                    reason="EMPTY_TELEMETRY: Feature Series is empty."
                )
            row = features.to_dict()
        else:
            return GateResult(
                status=GateStatus.INVALID_OR_INSUFFICIENT,
                is_event_detected=False,
                reason=f"INVALID_DATA_TYPE: Expected DataFrame, Series, or dict, got {type(features).__name__}."
            )

        # 2. Extract and validate each required kinematic measurement group
        def _extract_finite(keys: List[str], group_name: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
            found_key = None
            for k in keys:
                if k in row and row[k] is not None:
                    found_key = k
                    break
            if found_key is None:
                return None, None, f"MISSING_REQUIRED_FEATURE: Missing feature in group '{group_name}' (expected one of {keys})."

            raw_val = row[found_key]
            if isinstance(raw_val, str):
                return None, found_key, f"INVALID_NUMERIC_TYPE: Key '{found_key}' has string value '{raw_val}'."
            if not isinstance(raw_val, (int, float, np.number)):
                return None, found_key, f"INVALID_NUMERIC_TYPE: Key '{found_key}' has non-numeric type {type(raw_val).__name__}."

            try:
                f_val = float(raw_val)
            except (ValueError, TypeError) as ex:
                return None, found_key, f"INVALID_NUMERIC_VALUE: Key '{found_key}' failed float conversion ({ex})."

            if math.isnan(f_val):
                return None, found_key, f"NON_FINITE_VALUE: Key '{found_key}' is NaN."
            if math.isinf(f_val):
                return None, found_key, f"NON_FINITE_VALUE: Key '{found_key}' is infinite ({f_val})."

            return f_val, found_key, None

        # Group 1: Horizontal acceleration variation
        acc_horiz_std, k1, err1 = _extract_finite(["acc_horiz_std_3s", "acc_horiz_std_5s"], "acc_horiz_std")
        if err1 is not None:
            return GateResult(status=GateStatus.INVALID_OR_INSUFFICIENT, is_event_detected=False, reason=err1)

        # Group 2: Yaw rate variation
        gyro_z_std, k2, err2 = _extract_finite(["gyro_z_std_3s", "gyro_z_std_5s"], "gyro_z_std")
        if err2 is not None:
            return GateResult(status=GateStatus.INVALID_OR_INSUFFICIENT, is_event_detected=False, reason=err2)

        # Group 3: Total acceleration magnitude
        acc_mag, k3, err3 = _extract_finite(["acc_mag_ms2", "acc_mag_g"], "acc_mag")
        if err3 is not None:
            return GateResult(status=GateStatus.INVALID_OR_INSUFFICIENT, is_event_detected=False, reason=err3)

        # Group 4: Total rotational magnitude
        gyro_mag, k4, err4 = _extract_finite(["gyro_mag_rads", "gyro_mag_degs"], "gyro_mag")
        if err4 is not None:
            return GateResult(status=GateStatus.INVALID_OR_INSUFFICIENT, is_event_detected=False, reason=err4)

        # Group 5: Direct yaw rate
        gyro_z, k5, err5 = _extract_finite(["gyro_z_rads", "GyroZ"], "gyro_z")
        if err5 is not None:
            return GateResult(status=GateStatus.INVALID_OR_INSUFFICIENT, is_event_detected=False, reason=err5)

        # 3. Unit Normalizations
        # Dynamic acceleration magnitude excursion from 1g
        if k3 == "acc_mag_g":
            acc_mag_dev = abs(acc_mag - 1.0) * 9.80665
        else:
            acc_mag_dev = abs(acc_mag - 9.80665)

        # Rotational magnitude in rad/s
        if k4 == "gyro_mag_degs":
            gyro_mag_rads = float(np.deg2rad(gyro_mag))
        else:
            gyro_mag_rads = gyro_mag

        # Yaw rate in rad/s
        if k5 == "GyroZ":
            gyro_z_rads = abs(float(np.deg2rad(gyro_z)))
        else:
            gyro_z_rads = abs(gyro_z)

        # 4. Kinematic Excursion Envelope Checks
        if acc_horiz_std >= getattr(self.config, "gate_acc_horiz_std_threshold", 0.30):
            return GateResult(
                status=GateStatus.VALID_EXCITED,
                is_event_detected=True,
                reason=f"EXCITED_ACC_HORIZ_STD_{acc_horiz_std:.3f}"
            )

        if gyro_z_std >= getattr(self.config, "gate_gyro_z_std_threshold", 0.05):
            return GateResult(
                status=GateStatus.VALID_EXCITED,
                is_event_detected=True,
                reason=f"EXCITED_GYRO_Z_STD_{gyro_z_std:.3f}"
            )

        if acc_mag_dev >= getattr(self.config, "gate_acc_mag_dev_threshold", 1.47):
            return GateResult(
                status=GateStatus.VALID_EXCITED,
                is_event_detected=True,
                reason=f"EXCITED_ACC_MAG_DEV_{acc_mag_dev:.3f}"
            )

        if gyro_mag_rads >= getattr(self.config, "gate_gyro_mag_threshold", 0.25):
            return GateResult(
                status=GateStatus.VALID_EXCITED,
                is_event_detected=True,
                reason=f"EXCITED_GYRO_MAG_{gyro_mag_rads:.3f}"
            )

        if gyro_z_rads >= 0.20:
            return GateResult(
                status=GateStatus.VALID_EXCITED,
                is_event_detected=True,
                reason=f"EXCITED_GYRO_Z_{gyro_z_rads:.3f}"
            )

        return GateResult(
            status=GateStatus.VALID_QUIESCENT,
            is_event_detected=False,
            reason="QUIESCENT_CRUISING"
        )


class IntelligentSafetyAgent:
    """
    Top-level coordinator combining Pipeline A (Driving Style) and Pipeline B (Maneuvers)
    via prediction validation, temporal smoothing, and a deterministic safety rule engine.
    """

    def __init__(
        self,
        config: SafetyAgentConfig = DEFAULT_CONFIG,
        classifier_c: Optional[DatasetCBinaryClassifier] = None,
        classifier_b: Optional[DatasetBManeuverClassifier] = None,
        auto_load_classifiers: bool = False
    ):
        self.config = config
        self.validator = PredictionValidator(config=config)
        self.smoother = TemporalSmoother(config=config)
        self.risk_engine = DeterministicRiskEngine(config=config)
        self.gate = KinematicEventGate(config=config)

        self.classifier_c = classifier_c
        self.classifier_b = classifier_b

        if auto_load_classifiers:
            if self.classifier_c is None:
                self.classifier_c = DatasetCBinaryClassifier()
            if self.classifier_b is None:
                self.classifier_b = DatasetBManeuverClassifier(model_name="extra_trees")

    def reset(self):
        """Reset internal history, smoothing buffers, and alert timers."""
        self.smoother.reset()
        self.risk_engine.reset()

    def process_predictions(
        self,
        timestamp: float,
        style_prediction: Optional[int] = None,
        style_confidence: Optional[float] = None,
        maneuver_prediction: Optional[int] = None,
        maneuver_confidence: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> SafetyAssessment:
        """
        Process pre-computed predictions from Pipeline A and/or Pipeline B.
        """
        raw_input = RawPredictionInput(
            timestamp=timestamp,
            style_prediction=style_prediction,
            style_confidence=style_confidence,
            maneuver_prediction=maneuver_prediction,
            maneuver_confidence=maneuver_confidence
        )

        # Step 1: Validate Predictions & Telemetry Integrity
        val_pred = self.validator.validate(raw_input, current_time=current_time)

        # Step 2: Causal Temporal Smoothing & Multi-Rate Alignment
        smoothed_ctx = self.smoother.update(val_pred)

        # Step 3: Deterministic Rule Engine & Risk Scoring
        eval_result = self.risk_engine.evaluate(smoothed_ctx)

        # Format Human-Readable Output
        style_id = val_pred.style_prediction
        style_name = STYLE_LABEL_NAMES.get(style_id, "Unknown") if style_id is not None else "Unknown"

        maneuver_id = val_pred.maneuver_prediction
        maneuver_name = MANEUVER_LABEL_NAMES.get(maneuver_id, "None / Nominal") if maneuver_id is not None else "Unknown"

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

        return SafetyAssessment(
            timestamp=val_pred.timestamp,
            safety_state=eval_result.safety_state,
            risk_score=eval_result.risk_score,
            driving_style=style_name,
            driving_style_id=style_id,
            driving_style_confidence=float(val_pred.style_confidence),
            maneuver=maneuver_name,
            maneuver_id=maneuver_id,
            maneuver_confidence=float(val_pred.maneuver_confidence),
            confidence_summary={
                "style_confidence": float(val_pred.style_confidence),
                "maneuver_confidence": float(val_pred.maneuver_confidence),
                "confidence_factor": float(eval_result.confidence_factor)
            },
            triggered_rules=eval_result.triggered_rules,
            explanation=eval_result.overall_explanation,
            active_alerts=alerts_list,
            data_quality_status=val_pred.data_quality_status,
            is_valid=val_pred.is_valid
        )

    def process_features(
        self,
        timestamp: float,
        features_c: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        features_b: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        current_time: Optional[float] = None
    ) -> SafetyAssessment:
        """
        End-to-end processing: Run classifier inference on raw feature frames,
        extract predictions and confidences, and generate safety assessment.
        """
        style_pred = None
        style_conf = None
        maneuver_pred = None
        maneuver_conf = None

        # Run Pipeline A (Style) if features provided
        if features_c is not None and self.classifier_c is not None:
            if isinstance(features_c, dict):
                features_c = pd.DataFrame([features_c])
            try:
                probs_c = self.classifier_c.predict_proba(features_c)
                # Model decision threshold
                thresh = getattr(self.classifier_c, "decision_threshold", 0.5)
                if thresh is None:
                    thresh = 0.5
                p_agg = float(probs_c[0, 1])
                style_pred = 1 if p_agg >= thresh else 0
                style_conf = p_agg if style_pred == 1 else float(probs_c[0, 0])
            except Exception:
                style_pred = None
                style_conf = None

        # Run Pipeline B (Maneuver) if features provided
        if features_b is not None and self.classifier_b is not None:
            if isinstance(features_b, dict):
                features_b = pd.DataFrame([features_b])
            try:
                gate_result = self.gate.evaluate(features_b)
                if gate_result.status == GateStatus.VALID_QUIESCENT:
                    maneuver_pred = ManeuverClass.NONE.value
                    maneuver_conf = 1.0
                elif gate_result.status == GateStatus.VALID_EXCITED:
                    preds_b = self.classifier_b.predict(features_b)
                    probs_b = self.classifier_b.predict_proba(features_b)
                    pred_cls = int(preds_b[0])
                    maneuver_pred = pred_cls
                    maneuver_conf = float(probs_b[0, pred_cls - 1])
                else:  # GateStatus.INVALID_OR_INSUFFICIENT
                    maneuver_pred = None
                    maneuver_conf = None
            except Exception:
                maneuver_pred = None
                maneuver_conf = None

        return self.process_predictions(
            timestamp=timestamp,
            style_prediction=style_pred,
            style_confidence=style_conf,
            maneuver_prediction=maneuver_pred,
            maneuver_confidence=maneuver_conf,
            current_time=current_time
        )

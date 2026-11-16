"""
Intelligent Driving Safety Monitoring Agent
===========================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    SafetyState,
    DrivingStyle,
    ManeuverClass,
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
                preds_b = self.classifier_b.predict(features_b)
                probs_b = self.classifier_b.predict_proba(features_b)
                pred_cls = int(preds_b[0])
                maneuver_pred = pred_cls
                maneuver_conf = float(probs_b[0, pred_cls - 1])
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

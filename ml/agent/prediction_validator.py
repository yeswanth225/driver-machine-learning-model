"""
Prediction Validation & Input Integrity Verification Module
============================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    DrivingStyle,
    ManeuverClass,
    STYLE_LABEL_NAMES,
    MANEUVER_LABEL_NAMES
)


@dataclass
class RawPredictionInput:
    """Raw prediction payload received from model inference pipelines or streaming telemetry."""
    timestamp: float
    style_prediction: Optional[int] = None
    style_confidence: Optional[float] = None
    maneuver_prediction: Optional[int] = None
    maneuver_confidence: Optional[float] = None
    sensor_health: str = "OK"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatedPrediction:
    """Validated, sanitized prediction ready for temporal smoothing and risk scoring."""
    timestamp: float
    style_prediction: Optional[int]
    style_confidence: float
    maneuver_prediction: Optional[int]
    maneuver_confidence: float
    is_valid: bool
    is_stale: bool
    data_quality_status: str  # "VALID", "STALE", "MISSING", "INVALID"
    validation_errors: List[str] = field(default_factory=list)


class PredictionValidator:
    """
    Validates model predictions and telemetry for schema conformance, value bounds,
    finiteness, and timestamp staleness.
    """

    def __init__(self, config: SafetyAgentConfig = DEFAULT_CONFIG):
        self.config = config
        self._last_valid_timestamp: Optional[float] = None

    def validate(
        self,
        raw_input: RawPredictionInput,
        current_time: Optional[float] = None
    ) -> ValidatedPrediction:
        """
        Validate input payload and produce a sanitized ValidatedPrediction.
        Never defaults missing predictions to a normal driving state.
        """
        errors: List[str] = []
        is_stale = False

        # 1. Timestamp Verification
        ts = raw_input.timestamp
        if not isinstance(ts, (int, float)) or math.isnan(ts) or math.isinf(ts) or ts < 0.0:
            errors.append(f"Invalid timestamp value: {ts}. Timestamp must be a finite non-negative number.")
            ts = 0.0

        # Staleness Check (relative to provided current_time or last valid timestamp)
        eval_time = current_time if current_time is not None else ts
        if eval_time is not None and (eval_time - ts) > self.config.stale_threshold_seconds:
            is_stale = True
            errors.append(
                f"Stale telemetry detected: timestamp {ts:.2f} is {(eval_time - ts):.2f}s behind current time {eval_time:.2f}s (Threshold: {self.config.stale_threshold_seconds}s)."
            )

        # Monotonicity check (warning only)
        if self._last_valid_timestamp is not None and ts < self._last_valid_timestamp:
            errors.append(
                f"Temporal anomaly: timestamp {ts:.2f} decreased from previous timestamp {self._last_valid_timestamp:.2f}."
            )

        # 2. Driving Style Prediction Verification (Pipeline A)
        style_pred = raw_input.style_prediction
        style_conf = raw_input.style_confidence

        if style_pred is not None:
            if not isinstance(style_pred, (int, np_int_type := int)):
                errors.append(f"Style prediction must be an integer, got: {type(style_pred)}")
                style_pred = None
            elif style_pred not in [DrivingStyle.NORMAL.value, DrivingStyle.AGGRESSIVE.value]:
                errors.append(f"Style prediction {style_pred} outside valid domain {{0, 1}}.")
                style_pred = None

        if style_conf is not None:
            if not isinstance(style_conf, (int, float)) or math.isnan(style_conf) or math.isinf(style_conf):
                errors.append(f"Style confidence {style_conf} must be a finite float.")
                style_conf = 0.0
            elif not (0.0 <= style_conf <= 1.0):
                errors.append(f"Style confidence {style_conf} outside [0.0, 1.0] range.")
                style_conf = max(0.0, min(1.0, float(style_conf)))
        else:
            style_conf = 0.0

        # 3. Maneuver Prediction Verification (Pipeline B)
        maneuver_pred = raw_input.maneuver_prediction
        maneuver_conf = raw_input.maneuver_confidence

        if maneuver_pred is not None:
            if not isinstance(maneuver_pred, int):
                errors.append(f"Maneuver prediction must be an integer, got: {type(maneuver_pred)}")
                maneuver_pred = None
            elif maneuver_pred not in [
                ManeuverClass.NONE.value,
                ManeuverClass.SUDDEN_ACCELERATION.value,
                ManeuverClass.SUDDEN_DECELERATION.value,
                ManeuverClass.SUDDEN_RIGHT_TURN.value,
                ManeuverClass.SUDDEN_LEFT_TURN.value
            ]:
                errors.append(f"Maneuver prediction {maneuver_pred} outside valid domain {{0, 1, 2, 3, 4}}.")
                maneuver_pred = None

        if maneuver_conf is not None:
            if not isinstance(maneuver_conf, (int, float)) or math.isnan(maneuver_conf) or math.isinf(maneuver_conf):
                errors.append(f"Maneuver confidence {maneuver_conf} must be a finite float.")
                maneuver_conf = 0.0
            elif not (0.0 <= maneuver_conf <= 1.0):
                errors.append(f"Maneuver confidence {maneuver_conf} outside [0.0, 1.0] range.")
                maneuver_conf = max(0.0, min(1.0, float(maneuver_conf)))
        else:
            maneuver_conf = 0.0

        raw_has_style = raw_input.style_prediction is not None
        raw_has_maneuver = raw_input.maneuver_prediction is not None
        raw_is_missing = not raw_has_style and not raw_has_maneuver

        # 4. Check for Complete Data Absence
        has_style = style_pred is not None
        has_maneuver = maneuver_pred is not None
        if not has_style and not has_maneuver:
            errors.append("Missing prediction: Neither Driving Style nor Maneuver prediction was provided or valid.")

        # Determine Data Quality Status
        if is_stale:
            data_quality = "STALE"
            is_valid = False
        elif raw_is_missing:
            data_quality = "MISSING"
            is_valid = False
        elif len(errors) > 0:
            data_quality = "INVALID"
            is_valid = False
        else:
            data_quality = "VALID"
            is_valid = True
            self._last_valid_timestamp = ts

        return ValidatedPrediction(
            timestamp=ts,
            style_prediction=style_pred,
            style_confidence=float(style_conf),
            maneuver_prediction=maneuver_pred,
            maneuver_confidence=float(maneuver_conf),
            is_valid=is_valid,
            is_stale=is_stale,
            data_quality_status=data_quality,
            validation_errors=errors
        )

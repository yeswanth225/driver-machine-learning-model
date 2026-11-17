"""
Prediction Validation & Input Integrity Verification Module
============================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

import math
import numbers
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
    pipeline_a_health: Optional[str] = None
    pipeline_b_health: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatedPrediction:
    """Validated, sanitized prediction ready for temporal smoothing and risk scoring."""
    timestamp: float
    style_prediction: Optional[int] = None
    style_confidence: float = 0.0
    maneuver_prediction: Optional[int] = None
    maneuver_confidence: float = 0.0
    is_valid: bool = True
    is_stale: bool = False
    data_quality_status: str = "VALID"  # "VALID", "STALE", "MISSING", "INVALID", "DEGRADED"
    pipeline_a_health: str = "HEALTHY"
    pipeline_b_health: str = "HEALTHY"
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
        layer1_errors: List[str] = []
        is_stale = False

        # 1. Timestamp Verification (Layer 1 Transport Integrity)
        ts = raw_input.timestamp
        if isinstance(ts, bool) or not isinstance(ts, numbers.Real) or not math.isfinite(ts) or ts < 0.0:
            layer1_errors.append(f"Invalid timestamp value: {ts}. Timestamp must be a finite non-negative number.")
            ts = 0.0
        else:
            ts = float(ts)

        # Temporal Checks
        if current_time is not None and (ts - current_time) > getattr(self.config, "future_timestamp_threshold_seconds", 1.0):
            layer1_errors.append(
                f"Future timestamp detected: timestamp {ts:.2f}s is ahead of current time {current_time:.2f}s."
            )

        if current_time is not None and (current_time - ts) > self.config.stale_threshold_seconds:
            is_stale = True
            layer1_errors.append(
                f"Stale telemetry detected: timestamp {ts:.2f} is {(current_time - ts):.2f}s behind current time {current_time:.2f}s (Threshold: {self.config.stale_threshold_seconds}s)."
            )

        if self._last_valid_timestamp is not None:
            if ts < self._last_valid_timestamp:
                layer1_errors.append(
                    f"Temporal anomaly: timestamp {ts:.2f} decreased from previous timestamp {self._last_valid_timestamp:.2f}."
                )
            elif (ts - self._last_valid_timestamp) > getattr(self.config, "max_forward_leap_seconds", 10.0):
                layer1_errors.append(
                    f"Excessive forward timestamp leap: timestamp {ts:.2f} increased by {(ts - self._last_valid_timestamp):.2f}s from {self._last_valid_timestamp:.2f}s."
                )

        # 2. Driving Style Prediction Verification (Layer 2 Pipeline A)
        style_pred = raw_input.style_prediction
        style_conf = raw_input.style_confidence
        style_errors: List[str] = []

        if raw_input.pipeline_a_health is not None:
            pipeline_a_health = raw_input.pipeline_a_health
            if style_pred is not None and isinstance(style_pred, numbers.Integral) and not isinstance(style_pred, bool):
                style_pred = int(style_pred)
            if style_conf is not None and isinstance(style_conf, numbers.Real) and not isinstance(style_conf, bool) and math.isfinite(style_conf):
                style_conf = max(0.0, min(1.0, float(style_conf)))
            else:
                style_conf = 0.0
        elif style_pred is None:
            pipeline_a_health = "MISSING"
            style_conf = 0.0
        else:
            if isinstance(style_pred, bool) or not isinstance(style_pred, numbers.Integral):
                style_errors.append(f"Style prediction must be an integer, got: {type(style_pred)}")
                style_pred = None
            else:
                style_pred = int(style_pred)
                if style_pred not in [DrivingStyle.NORMAL.value, DrivingStyle.AGGRESSIVE.value]:
                    style_errors.append(f"Style prediction {style_pred} outside valid domain {{0, 1}}.")
                    style_pred = None

            if style_conf is not None:
                if isinstance(style_conf, bool) or not isinstance(style_conf, numbers.Real) or not math.isfinite(style_conf):
                    style_errors.append(f"Style confidence {style_conf} must be a finite float.")
                    style_conf = 0.0
                elif not (0.0 <= style_conf <= 1.0):
                    style_errors.append(f"Style confidence {style_conf} outside [0.0, 1.0] range.")
                    style_conf = max(0.0, min(1.0, float(style_conf)))
                else:
                    style_conf = float(style_conf)
            else:
                style_conf = 0.0

            if style_errors:
                pipeline_a_health = "INVALID"
            else:
                pipeline_a_health = "HEALTHY"

        # 3. Maneuver Prediction Verification (Layer 2 Pipeline B)
        maneuver_pred = raw_input.maneuver_prediction
        maneuver_conf = raw_input.maneuver_confidence
        maneuver_errors: List[str] = []

        if raw_input.pipeline_b_health is not None:
            pipeline_b_health = raw_input.pipeline_b_health
            if pipeline_b_health == "QUIESCENT_BYPASS":
                if maneuver_pred is None:
                    maneuver_pred = ManeuverClass.NONE.value
                elif isinstance(maneuver_pred, numbers.Integral) and not isinstance(maneuver_pred, bool):
                    maneuver_pred = int(maneuver_pred)
                if maneuver_conf is None or not math.isfinite(maneuver_conf):
                    maneuver_conf = 1.0
                elif isinstance(maneuver_conf, numbers.Real) and not isinstance(maneuver_conf, bool):
                    maneuver_conf = float(maneuver_conf)
                else:
                    maneuver_conf = 1.0
            else:
                if maneuver_pred is not None and isinstance(maneuver_pred, numbers.Integral) and not isinstance(maneuver_pred, bool):
                    maneuver_pred = int(maneuver_pred)
                if maneuver_conf is not None and isinstance(maneuver_conf, numbers.Real) and not isinstance(maneuver_conf, bool) and math.isfinite(maneuver_conf):
                    maneuver_conf = max(0.0, min(1.0, float(maneuver_conf)))
                else:
                    maneuver_conf = 0.0
        elif raw_input.metadata.get("quiescent_bypass") or raw_input.metadata.get("gate_status") == "VALID_QUIESCENT":
            pipeline_b_health = "QUIESCENT_BYPASS"
            maneuver_pred = ManeuverClass.NONE.value
            maneuver_conf = 1.0
        elif maneuver_pred is None:
            pipeline_b_health = "MISSING"
            maneuver_conf = 0.0
        else:
            if isinstance(maneuver_pred, bool) or not isinstance(maneuver_pred, numbers.Integral):
                maneuver_errors.append(f"Maneuver prediction must be an integer, got: {type(maneuver_pred)}")
                maneuver_pred = None
            else:
                maneuver_pred = int(maneuver_pred)
                if maneuver_pred not in [
                    ManeuverClass.NONE.value,
                    ManeuverClass.SUDDEN_ACCELERATION.value,
                    ManeuverClass.SUDDEN_DECELERATION.value,
                    ManeuverClass.SUDDEN_RIGHT_TURN.value,
                    ManeuverClass.SUDDEN_LEFT_TURN.value
                ]:
                    maneuver_errors.append(f"Maneuver prediction {maneuver_pred} outside valid domain {{0, 1, 2, 3, 4}}.")
                    maneuver_pred = None

            if maneuver_conf is not None:
                if isinstance(maneuver_conf, bool) or not isinstance(maneuver_conf, numbers.Real) or not math.isfinite(maneuver_conf):
                    maneuver_errors.append(f"Maneuver confidence {maneuver_conf} must be a finite float.")
                    maneuver_conf = 0.0
                elif not (0.0 <= maneuver_conf <= 1.0):
                    maneuver_errors.append(f"Maneuver confidence {maneuver_conf} outside [0.0, 1.0] range.")
                    maneuver_conf = max(0.0, min(1.0, float(maneuver_conf)))
                else:
                    maneuver_conf = float(maneuver_conf)
            else:
                maneuver_conf = 0.0

            if maneuver_errors:
                pipeline_b_health = "INVALID"
            else:
                pipeline_b_health = "HEALTHY"

        all_errors = layer1_errors + style_errors + maneuver_errors

        # Check for Complete Data Absence
        raw_is_missing = (raw_input.style_prediction is None and raw_input.maneuver_prediction is None and pipeline_b_health != "QUIESCENT_BYPASS")
        if raw_is_missing:
            all_errors.append("Missing prediction: Neither Driving Style nor Maneuver prediction was provided or valid.")

        has_usable_style = (pipeline_a_health == "HEALTHY")
        has_usable_maneuver = (pipeline_b_health in ("HEALTHY", "QUIESCENT_BYPASS"))

        # Determine Data Quality Status and is_valid
        if is_stale:
            data_quality = "STALE"
            is_valid = False
        elif layer1_errors:
            data_quality = "INVALID"
            is_valid = False
        elif raw_is_missing:
            data_quality = "MISSING"
            is_valid = False
        elif not has_usable_style and not has_usable_maneuver:
            data_quality = "INVALID" if (style_errors or maneuver_errors) else "MISSING"
            is_valid = False
        else:
            is_valid = True
            self._last_valid_timestamp = ts
            if pipeline_a_health == "HEALTHY" and pipeline_b_health in ("HEALTHY", "QUIESCENT_BYPASS"):
                data_quality = "VALID"
            elif pipeline_a_health == "DEGRADED" or pipeline_b_health == "DEGRADED":
                data_quality = "DEGRADED"
            elif pipeline_a_health == "INVALID" or pipeline_b_health == "INVALID":
                data_quality = "INVALID"
            else:
                data_quality = "VALID"

        return ValidatedPrediction(
            timestamp=ts,
            style_prediction=style_pred,
            style_confidence=float(style_conf),
            maneuver_prediction=maneuver_pred,
            maneuver_confidence=float(maneuver_conf),
            is_valid=is_valid,
            is_stale=is_stale,
            data_quality_status=data_quality,
            pipeline_a_health=pipeline_a_health,
            pipeline_b_health=pipeline_b_health,
            validation_errors=all_errors
        )

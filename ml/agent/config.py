"""
Safety Agent Configuration & Domain Enums
==========================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any


class SafetyState(str, Enum):
    """Unified Driver Safety States."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DrivingStyle(int, Enum):
    """Pipeline A (Dataset C) Class Labels."""
    NORMAL = 0
    AGGRESSIVE = 1


class ManeuverClass(int, Enum):
    """Pipeline B (Dataset B) Class Labels."""
    NONE = 0
    SUDDEN_ACCELERATION = 1
    SUDDEN_DECELERATION = 2
    SUDDEN_RIGHT_TURN = 3
    SUDDEN_LEFT_TURN = 4


class AlertSeverity(str, Enum):
    """Alert Severity Levels."""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    """Alert Functional Categories."""
    STYLE = "STYLE"
    MANEUVER = "MANEUVER"
    COMPOUND = "COMPOUND"
    DATA_QUALITY = "DATA_QUALITY"


STYLE_LABEL_NAMES: Dict[int, str] = {
    0: "Normal Driving",
    1: "Aggressive Driving"
}

MANEUVER_LABEL_NAMES: Dict[int, str] = {
    0: "None / Nominal",
    1: "Sudden Acceleration",
    2: "Sudden Deceleration",
    3: "Sudden Right Turn",
    4: "Sudden Left Turn"
}

MANEUVER_ABBREVIATIONS: Dict[int, str] = {
    0: "NOM",
    1: "SA",
    2: "SD",
    3: "SRT",
    4: "SLT"
}


@dataclass(frozen=True)
class SafetyAgentConfig:
    """Centralized, immutable configuration parameters for the Safety Monitoring Agent."""

    # 1. Temporal & Staleness Thresholds (Seconds)
    stale_threshold_seconds: float = 3.0
    history_window_seconds: float = 10.0
    style_smoothing_window_seconds: float = 3.0
    maneuver_persistence_seconds: float = 2.0
    maneuver_history_window_seconds: float = 10.0

    # 2. Smoothing Weights
    style_ema_alpha: float = 0.20
    style_aggressive_ratio_threshold: float = 0.50

    # 3. Confidence Thresholds
    min_confidence_threshold: float = 0.50
    high_confidence_threshold: float = 0.75
    low_confidence_discount: float = 0.70  # 30% reduction for low confidence

    # 4. Deterministic Risk Score Weights (0 to 100 scale)
    weight_sustained_aggressive: float = 35.0
    weight_transient_aggressive: float = 20.0
    weight_sudden_acceleration: float = 25.0
    weight_sudden_deceleration: float = 35.0
    weight_sudden_turn: float = 30.0
    weight_compound_hazard_bonus: float = 25.0
    weight_repeated_maneuver_bonus: float = 15.0

    # 5. Risk Score to State Boundaries
    score_threshold_caution: float = 25.0
    score_threshold_warning: float = 50.0
    score_threshold_high_risk: float = 75.0

    # 6. Alert Deduplication Cooldown Timers (Seconds)
    alert_cooldown_style: float = 5.0
    alert_cooldown_maneuver: float = 3.0
    alert_cooldown_compound: float = 2.0
    alert_cooldown_data_quality: float = 5.0


DEFAULT_CONFIG = SafetyAgentConfig()

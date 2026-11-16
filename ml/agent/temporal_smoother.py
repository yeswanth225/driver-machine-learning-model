"""
Temporal Smoothing & Multi-Rate Telemetry Alignment Module
==========================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    DrivingStyle,
    ManeuverClass,
    STYLE_LABEL_NAMES,
    MANEUVER_LABEL_NAMES
)
from ml.agent.prediction_validator import ValidatedPrediction


@dataclass
class SmoothedContext:
    """Smoothed, multi-rate aligned context used by the Deterministic Risk Engine."""
    timestamp: float
    # Style Dynamics (10 Hz Pipeline A)
    style_ema: float  # Exponential moving average of aggressive probability [0.0, 1.0]
    style_aggressive_ratio: float  # Fraction of aggressive predictions in recent window
    is_sustained_aggressive: bool
    is_transient_aggressive: bool
    current_style: int  # 0: Normal, 1: Aggressive
    current_style_confidence: float

    # Maneuver Dynamics (~1.55 Hz Pipeline B)
    active_maneuver: int  # 0: None, 1: SA, 2: SD, 3: SRT, 4: SLT
    active_maneuver_confidence: float
    active_maneuver_age_seconds: float
    is_maneuver_active: bool
    recent_maneuver_count: int  # Number of abrupt maneuvers in past 10s window
    recent_maneuver_history: List[Dict[str, Any]] = field(default_factory=list)

    # Data Quality
    is_valid: bool = True
    data_quality_status: str = "VALID"


class TemporalSmoother:
    """
    Maintains a bounded historical buffer to causally smooth high-frequency (10 Hz)
    style predictions and persist low-frequency (~1.55 Hz) discrete maneuvers.
    """

    def __init__(self, config: SafetyAgentConfig = DEFAULT_CONFIG):
        self.config = config
        self._history: deque = deque()  # stores (timestamp, ValidatedPrediction)

        # Style state
        self._style_ema: float = 0.0
        self._last_style_timestamp: Optional[float] = None

        # Maneuver state
        self._active_maneuver: int = ManeuverClass.NONE.value
        self._active_maneuver_conf: float = 0.0
        self._active_maneuver_timestamp: Optional[float] = None

        # Maneuver event log: list of (timestamp, maneuver_id, confidence)
        self._maneuver_events: deque = deque()

    def reset(self):
        """Reset all smoother histories and internal state."""
        self._history.clear()
        self._style_ema = 0.0
        self._last_style_timestamp = None
        self._active_maneuver = ManeuverClass.NONE.value
        self._active_maneuver_conf = 0.0
        self._active_maneuver_timestamp = None
        self._maneuver_events.clear()

    def update(self, val_pred: ValidatedPrediction) -> SmoothedContext:
        """
        Process a new validated prediction frame, update causal smoothing buffers,
        and return the aligned SmoothedContext.
        """
        ts = val_pred.timestamp

        if not val_pred.is_valid:
            return SmoothedContext(
                timestamp=ts,
                style_ema=self._style_ema,
                style_aggressive_ratio=0.0,
                is_sustained_aggressive=False,
                is_transient_aggressive=False,
                current_style=val_pred.style_prediction if val_pred.style_prediction is not None else 0,
                current_style_confidence=val_pred.style_confidence,
                active_maneuver=ManeuverClass.NONE.value,
                active_maneuver_confidence=0.0,
                active_maneuver_age_seconds=0.0,
                is_maneuver_active=False,
                recent_maneuver_count=0,
                is_valid=False,
                data_quality_status=val_pred.data_quality_status
            )

        # Append to bounded sliding window
        self._history.append((ts, val_pred))

        # Prune history older than history_window_seconds (10s)
        cutoff_time = ts - self.config.history_window_seconds
        while self._history and self._history[0][0] < cutoff_time:
            self._history.popleft()

        while self._maneuver_events and self._maneuver_events[0][0] < cutoff_time:
            self._maneuver_events.popleft()

        # -------------------------------------------------------------
        # 1. Smooth Pipeline A: Driving Style (10.0 Hz Stream)
        # -------------------------------------------------------------
        current_style = val_pred.style_prediction if val_pred.style_prediction is not None else 0
        current_style_conf = val_pred.style_confidence

        if val_pred.style_prediction is not None:
            # Target probability for aggressive driving
            raw_target = float(val_pred.style_prediction) * current_style_conf
            alpha = self.config.style_ema_alpha
            if self._last_style_timestamp is None:
                self._style_ema = raw_target
            else:
                self._style_ema = alpha * raw_target + (1.0 - alpha) * self._style_ema
            self._last_style_timestamp = ts

        # Windowed Aggressive Ratio (past style_smoothing_window_seconds = 3.0s)
        style_cutoff = ts - self.config.style_smoothing_window_seconds
        recent_style_samples = [
            p.style_prediction
            for t, p in self._history
            if t >= style_cutoff and p.style_prediction is not None
        ]
        if recent_style_samples:
            aggressive_count = sum(1 for s in recent_style_samples if s == DrivingStyle.AGGRESSIVE.value)
            style_aggressive_ratio = float(aggressive_count / len(recent_style_samples))
        else:
            style_aggressive_ratio = 0.0

        is_sustained_aggressive = (
            style_aggressive_ratio >= self.config.style_aggressive_ratio_threshold
            or self._style_ema >= 0.65
        )
        is_transient_aggressive = (
            not is_sustained_aggressive and current_style == DrivingStyle.AGGRESSIVE.value and current_style_conf >= 0.60
        )

        # -------------------------------------------------------------
        # 2. Smooth Pipeline B: Maneuver Detection (~1.55 Hz Stream)
        # -------------------------------------------------------------
        m_pred = val_pred.maneuver_prediction
        m_conf = val_pred.maneuver_confidence

        if m_pred is not None and m_pred != ManeuverClass.NONE.value:
            # Register new active maneuver
            self._active_maneuver = m_pred
            self._active_maneuver_conf = m_conf
            self._active_maneuver_timestamp = ts
            self._maneuver_events.append((ts, m_pred, m_conf))

        # Check maneuver persistence
        is_maneuver_active = False
        maneuver_age = 0.0
        if self._active_maneuver != ManeuverClass.NONE.value and self._active_maneuver_timestamp is not None:
            maneuver_age = ts - self._active_maneuver_timestamp
            if 0.0 <= maneuver_age <= self.config.maneuver_persistence_seconds:
                is_maneuver_active = True
            else:
                # Expired
                self._active_maneuver = ManeuverClass.NONE.value
                self._active_maneuver_conf = 0.0
                is_maneuver_active = False

        recent_maneuver_count = len(self._maneuver_events)
        recent_maneuver_history = [
            {
                "timestamp": t,
                "maneuver_id": mid,
                "maneuver_name": MANEUVER_LABEL_NAMES.get(mid, "Unknown"),
                "confidence": conf
            }
            for t, mid, conf in list(self._maneuver_events)
        ]

        return SmoothedContext(
            timestamp=ts,
            style_ema=float(self._style_ema),
            style_aggressive_ratio=float(style_aggressive_ratio),
            is_sustained_aggressive=is_sustained_aggressive,
            is_transient_aggressive=is_transient_aggressive,
            current_style=current_style,
            current_style_confidence=current_style_conf,
            active_maneuver=self._active_maneuver if is_maneuver_active else ManeuverClass.NONE.value,
            active_maneuver_confidence=float(self._active_maneuver_conf) if is_maneuver_active else 0.0,
            active_maneuver_age_seconds=float(maneuver_age),
            is_maneuver_active=is_maneuver_active,
            recent_maneuver_count=recent_maneuver_count,
            recent_maneuver_history=recent_maneuver_history,
            is_valid=True,
            data_quality_status=val_pred.data_quality_status
        )

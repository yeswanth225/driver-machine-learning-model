"""
Temporal Smoothing & Multi-Rate Telemetry Alignment Module
==========================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 6 - Intelligent Safety Monitoring Agent
"""

from collections import deque
from dataclasses import dataclass, field
import math
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
from ml.agent.lifecycle_tracker import ManeuverContribution, IncidentLifecycleManager


@dataclass
class SmoothedContext:
    """Smoothed, multi-rate aligned context used by the Deterministic Risk Engine."""
    timestamp: float
    # Style Dynamics (10 Hz Pipeline A)
    style_ema: float = 0.0  # Exponential moving average of aggressive probability [0.0, 1.0]
    style_aggressive_ratio: float = 0.0  # Fraction of aggressive predictions in recent window
    is_sustained_aggressive: bool = False
    is_transient_aggressive: bool = False
    current_style: Optional[int] = None  # 0: Normal, 1: Aggressive, None if missing
    current_style_confidence: Optional[float] = 0.0

    # Multi-channel Active/Decaying Maneuver Contributions (D-02 & D-06)
    active_contributions: Dict[int, ManeuverContribution] = field(default_factory=dict)

    # Maneuver Dynamics (~1.55 Hz Pipeline B)
    active_maneuver: int = 0  # 0: None, 1: SA, 2: SD, 3: SRT, 4: SLT
    active_maneuver_confidence: float = 0.0
    active_maneuver_age_seconds: float = 0.0
    is_maneuver_active: bool = False
    recent_maneuver_count: int = 0  # Number of abrupt maneuvers in past 10s window
    recent_maneuver_history: List[Dict[str, Any]] = field(default_factory=list)

    # Data Quality & Pipeline Health
    is_valid: bool = True
    data_quality_status: str = "VALID"
    pipeline_a_health: str = "HEALTHY"
    pipeline_b_health: str = "HEALTHY"


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
        self._last_maneuver_onset_time: Optional[float] = None

        # Maneuver event log: list of (timestamp, maneuver_id, confidence)
        self._maneuver_events: deque = deque()
        self._last_context: Optional[SmoothedContext] = None
        self.lifecycle_manager = IncidentLifecycleManager(config)

    def reset(self):
        """Reset all smoother histories and internal state."""
        self._history.clear()
        self._style_ema = 0.0
        self._last_style_timestamp = None
        self._active_maneuver = ManeuverClass.NONE.value
        self._active_maneuver_conf = 0.0
        self._active_maneuver_timestamp = None
        self._last_maneuver_onset_time = None
        self._maneuver_events.clear()
        self._last_context = None
        self.lifecycle_manager.reset()

    def get_context(self, ts: Optional[float] = None) -> SmoothedContext:
        """Return the current or most recent SmoothedContext."""
        if self._last_context is not None:
            return self._last_context
        active_contribs = self.lifecycle_manager.active_contributions
        return SmoothedContext(
            timestamp=ts or 0.0,
            style_ema=float(self._style_ema),
            style_aggressive_ratio=0.0,
            is_sustained_aggressive=False,
            is_transient_aggressive=False,
            current_style=None,
            current_style_confidence=0.0,
            active_contributions=active_contribs,
            active_maneuver=self._active_maneuver,
            active_maneuver_confidence=float(self._active_maneuver_conf),
            active_maneuver_age_seconds=0.0,
            is_maneuver_active=self._active_maneuver != ManeuverClass.NONE.value,
            recent_maneuver_count=len(self._maneuver_events),
            is_valid=True,
            data_quality_status="VALID"
        )

    def update(self, val_pred: ValidatedPrediction) -> SmoothedContext:
        """
        Process a new validated prediction frame, update causal smoothing buffers,
        and return the aligned SmoothedContext.
        """
        ts = val_pred.timestamp

        if not val_pred.is_valid:
            ctx = SmoothedContext(
                timestamp=ts,
                style_ema=self._style_ema,
                style_aggressive_ratio=0.0,
                is_sustained_aggressive=False,
                is_transient_aggressive=False,
                current_style=val_pred.style_prediction,
                current_style_confidence=val_pred.style_confidence if val_pred.style_confidence is not None else 0.0,
                active_maneuver=ManeuverClass.NONE.value,
                active_maneuver_confidence=0.0,
                active_maneuver_age_seconds=0.0,
                is_maneuver_active=False,
                recent_maneuver_count=0,
                is_valid=False,
                data_quality_status=val_pred.data_quality_status,
                pipeline_a_health=val_pred.pipeline_a_health,
                pipeline_b_health=val_pred.pipeline_b_health
            )
            self._last_context = ctx
            return ctx

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
        current_style = val_pred.style_prediction
        current_style_conf = val_pred.style_confidence if val_pred.style_confidence is not None else 0.0

        if val_pred.style_prediction is not None:
            # Target probability for aggressive driving
            raw_target = float(val_pred.style_prediction) * current_style_conf
            if self._last_style_timestamp is None:
                self._style_ema = raw_target
            else:
                dt = ts - self._last_style_timestamp
                if dt < 0:
                    dt = 0.0
                if dt > 5.0:  # Telemetry gap reset (> 5.0s)
                    self._style_ema = raw_target
                else:
                    tau = getattr(self.config, "style_ema_tau", 1.50)
                    alpha = 1.0 - math.exp(-dt / tau)
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
        # 2. Smooth Pipeline B: Maneuver Detection & Lifecycle (~1.55 Hz Stream)
        # -------------------------------------------------------------
        m_pred = val_pred.maneuver_prediction
        m_conf = val_pred.maneuver_confidence
        is_quiescent = (val_pred.pipeline_b_health == "QUIESCENT_BYPASS")

        active_contributions, recent_maneuver_count, recent_maneuver_history = self.lifecycle_manager.update(
            ts=ts,
            maneuver_prediction=m_pred,
            maneuver_confidence=m_conf,
            is_quiescent_bypass=is_quiescent
        )

        # Legacy buffer synchronization for test backward compatibility
        self._maneuver_events = deque([
            (h["timestamp"], h["maneuver_id"], h["confidence"])
            for h in recent_maneuver_history
        ])

        if active_contributions:
            active_excited = [
                c for c in active_contributions.values()
                if c.is_excited or (ts - (c.excitation_end_timestamp or c.last_active_timestamp)) <= self.config.maneuver_persistence_seconds
            ]
            if active_excited:
                primary_contrib = max(
                    active_excited,
                    key=lambda c: (c.initial_weight, c.confidence)
                )
                is_maneuver_active = True
                active_maneuver = primary_contrib.maneuver_class
                active_maneuver_conf = primary_contrib.confidence
                active_maneuver_age = ts - primary_contrib.onset_timestamp
                self._active_maneuver = active_maneuver
                self._active_maneuver_conf = active_maneuver_conf
                self._active_maneuver_timestamp = primary_contrib.last_active_timestamp
            else:
                is_maneuver_active = False
                active_maneuver = ManeuverClass.NONE.value
                active_maneuver_conf = 0.0
                active_maneuver_age = 0.0
                self._active_maneuver = ManeuverClass.NONE.value
                self._active_maneuver_conf = 0.0
                self._active_maneuver_timestamp = None
        else:
            is_maneuver_active = False
            active_maneuver = ManeuverClass.NONE.value
            active_maneuver_conf = 0.0
            active_maneuver_age = 0.0
            self._active_maneuver = ManeuverClass.NONE.value
            self._active_maneuver_conf = 0.0
            self._active_maneuver_timestamp = None

        ctx = SmoothedContext(
            timestamp=ts,
            style_ema=float(self._style_ema),
            style_aggressive_ratio=float(style_aggressive_ratio),
            is_sustained_aggressive=is_sustained_aggressive,
            is_transient_aggressive=is_transient_aggressive,
            current_style=current_style,
            current_style_confidence=current_style_conf,
            active_contributions=active_contributions,
            active_maneuver=active_maneuver,
            active_maneuver_confidence=float(active_maneuver_conf),
            active_maneuver_age_seconds=float(active_maneuver_age),
            is_maneuver_active=is_maneuver_active,
            recent_maneuver_count=recent_maneuver_count,
            recent_maneuver_history=recent_maneuver_history,
            is_valid=True,
            data_quality_status=val_pred.data_quality_status,
            pipeline_a_health=val_pred.pipeline_a_health,
            pipeline_b_health=val_pred.pipeline_b_health
        )
        self._last_context = ctx
        return ctx

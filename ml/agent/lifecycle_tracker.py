"""
Incident Lifecycle Tracker & Maneuver Contribution Manager (D-02 & D-06)

Fulfills Stage 6 Phase 2 authoritative specifications:
- D-02: Multi-instance Incident Lifecycle Tracking with 2.50s quiet hysteresis.
- D-06: Continuous Residual Risk Decay Integration with analytical boundary retirement.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import math

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    ManeuverClass,
    MANEUVER_CLASS_SHORT_NAMES,
    MANEUVER_LABEL_NAMES,
)


@dataclass
class ManeuverContribution:
    """
    Structured record of an active physical maneuver contribution (D-06).
    """
    event_id: str                              # Unique physical incident identity (e.g. EVT-SD-1000-01)
    maneuver_class: int                        # ManeuverClass enum value (1=SA, 2=SD, 3=SRT, 4=SLT)
    initial_weight: float                      # Base penalty weight W (e.g. 50.0 for SD, 35.0 for Turn)
    confidence: float                          # Model confidence C
    onset_timestamp: float                     # Physical onset timestamp (t0, immutable during continuation)
    excitation_end_timestamp: Optional[float] = None  # Locked when sensor excitation ceases
    is_excited: bool = True                    # True during active physical maneuver; False during decay
    last_active_timestamp: float = 0.0         # Timestamp of most recent active frame


class IncidentLifecycleManager:
    """
    Tracks streaming maneuver incidents, distinguishes separate incidents via 2.50s quiet hysteresis,
    preserves event onset t0 across continuation frames, and manages decaying residual channels.
    """

    def __init__(self, config: SafetyAgentConfig = DEFAULT_CONFIG):
        self.config = config
        self._active_channels: Dict[int, ManeuverContribution] = {}
        self._incident_counter: int = 0
        self._uninterrupted_quiet_since: Optional[float] = None
        self._last_excitation_end_times: Dict[int, float] = {}
        self._active_incident_ids: Dict[int, str] = {}
        self._incident_history: List[Dict[str, Any]] = []

    def reset(self):
        """Reset internal incident tracking state."""
        self._active_channels.clear()
        self._incident_counter = 0
        self._uninterrupted_quiet_since = None
        self._last_excitation_end_times.clear()
        self._active_incident_ids.clear()
        self._incident_history.clear()

    @property
    def active_contributions(self) -> Dict[int, ManeuverContribution]:
        """Return shallow copy of current active/decaying contributions."""
        return dict(self._active_channels)

    def get_initial_weight(self, maneuver_class: int) -> float:
        """Resolve authoritative base weight for a maneuver class."""
        if maneuver_class == ManeuverClass.SUDDEN_ACCELERATION.value:
            return float(self.config.weight_sudden_acceleration)
        elif maneuver_class == ManeuverClass.SUDDEN_DECELERATION.value:
            return float(self.config.weight_sudden_deceleration)
        elif maneuver_class in (
            ManeuverClass.SUDDEN_RIGHT_TURN.value,
            ManeuverClass.SUDDEN_LEFT_TURN.value,
        ):
            return float(self.config.weight_sudden_turn)
        return 0.0

    def calculate_retirement_horizon(self, initial_weight: float) -> float:
        """Calculate analytical boundary retirement time post-excitation."""
        plateau = getattr(self.config, "residual_plateau_seconds", 1.00)
        half_life = getattr(self.config, "residual_half_life_seconds", 0.75)
        retire_thresh = getattr(self.config, "residual_retirement_threshold", 2.50)
        if initial_weight <= retire_thresh:
            return plateau
        return plateau + half_life * math.log2(float(initial_weight) / float(retire_thresh))

    def update(
        self,
        ts: float,
        maneuver_prediction: Optional[int],
        maneuver_confidence: Optional[float],
        is_quiescent_bypass: bool = False
    ) -> Tuple[Dict[int, ManeuverContribution], int, List[Dict[str, Any]]]:
        """
        Process a new telemetry frame and advance incident lifecycles.
        
        Returns:
            (active_contributions, recent_incident_count, recent_incident_history)
        """
        m_conf = float(maneuver_confidence) if (maneuver_confidence is not None and math.isfinite(maneuver_confidence)) else 0.0
        has_active_maneuver = (
            maneuver_prediction is not None
            and maneuver_prediction in (1, 2, 3, 4)
            and not is_quiescent_bypass
        )
        rearm_sec = getattr(self.config, "maneuver_rearm_seconds", 2.50)

        if has_active_maneuver:
            m_class = int(maneuver_prediction)
            self._uninterrupted_quiet_since = None

            if m_class in self._active_channels:
                existing = self._active_channels[m_class]
                if existing.is_excited:
                    # Intra-event continuation: preserve onset_timestamp (t0)
                    existing.confidence = max(existing.confidence, m_conf)
                    existing.last_active_timestamp = ts
                else:
                    # In decay state: check quiet duration for intra-incident re-excitation
                    dt_quiet = ts - (existing.excitation_end_timestamp or existing.last_active_timestamp)
                    if dt_quiet < rearm_sec:
                        # Intra-incident re-excitation: retain event_id and t0
                        existing.is_excited = True
                        existing.excitation_end_timestamp = None
                        existing.confidence = max(existing.confidence, m_conf)
                        existing.last_active_timestamp = ts
                    else:
                        # Hysteresis elapsed: start new incident of the same class
                        self._incident_counter += 1
                        prefix = MANEUVER_CLASS_SHORT_NAMES.get(m_class, f"M{m_class}")
                        event_id = f"EVT-{prefix}-{int(ts * 1000):04d}-{self._incident_counter:02d}"
                        self._active_incident_ids[m_class] = event_id
                        w = self.get_initial_weight(m_class)
                        new_contrib = ManeuverContribution(
                            event_id=event_id,
                            maneuver_class=m_class,
                            initial_weight=w,
                            confidence=m_conf,
                            onset_timestamp=ts,
                            excitation_end_timestamp=None,
                            is_excited=True,
                            last_active_timestamp=ts
                        )
                        self._active_channels[m_class] = new_contrib
                        self._incident_history.append({
                            "timestamp": ts,
                            "event_id": event_id,
                            "maneuver_id": m_class,
                            "maneuver_name": MANEUVER_LABEL_NAMES.get(m_class, "Unknown"),
                            "confidence": m_conf
                        })
            else:
                # Channel not active: check last excitation end time
                last_end = self._last_excitation_end_times.get(m_class)
                dt_quiet = (ts - last_end) if last_end is not None else (rearm_sec + 1.0)
                if last_end is not None and dt_quiet < rearm_sec and m_class in self._active_incident_ids:
                    # Intra-incident resumption
                    event_id = self._active_incident_ids[m_class]
                    w = self.get_initial_weight(m_class)
                    contrib = ManeuverContribution(
                        event_id=event_id,
                        maneuver_class=m_class,
                        initial_weight=w,
                        confidence=m_conf,
                        onset_timestamp=last_end,
                        excitation_end_timestamp=None,
                        is_excited=True,
                        last_active_timestamp=ts
                    )
                    self._active_channels[m_class] = contrib
                else:
                    # Genuine new incident
                    self._incident_counter += 1
                    prefix = MANEUVER_CLASS_SHORT_NAMES.get(m_class, f"M{m_class}")
                    event_id = f"EVT-{prefix}-{int(ts * 1000):04d}-{self._incident_counter:02d}"
                    self._active_incident_ids[m_class] = event_id
                    w = self.get_initial_weight(m_class)
                    contrib = ManeuverContribution(
                        event_id=event_id,
                        maneuver_class=m_class,
                        initial_weight=w,
                        confidence=m_conf,
                        onset_timestamp=ts,
                        excitation_end_timestamp=None,
                        is_excited=True,
                        last_active_timestamp=ts
                    )
                    self._active_channels[m_class] = contrib
                    self._incident_history.append({
                        "timestamp": ts,
                        "event_id": event_id,
                        "maneuver_id": m_class,
                        "maneuver_name": MANEUVER_LABEL_NAMES.get(m_class, "Unknown"),
                        "confidence": m_conf
                    })
        else:
            if self._uninterrupted_quiet_since is None:
                self._uninterrupted_quiet_since = ts

        # Process channel excitation transitions and decay retirements
        current_active_class = int(maneuver_prediction) if has_active_maneuver else None
        channels_to_prune: List[int] = []

        for cls, contrib in self._active_channels.items():
            if cls != current_active_class:
                if contrib.is_excited:
                    # Excitation concluded on this frame: lock end timestamp immutably
                    contrib.is_excited = False
                    contrib.excitation_end_timestamp = contrib.last_active_timestamp
                    self._last_excitation_end_times[cls] = contrib.last_active_timestamp

                # Check analytical retirement
                dt_decay = ts - (contrib.excitation_end_timestamp or contrib.last_active_timestamp)
                t_retire = self.calculate_retirement_horizon(contrib.initial_weight)
                if dt_decay > t_retire:
                    channels_to_prune.append(cls)

        for cls in channels_to_prune:
            del self._active_channels[cls]

        # Prune historical incidents older than 10.0 seconds
        window_cutoff = ts - getattr(self.config, "maneuver_history_window_seconds", 10.0)
        self._incident_history = [
            inc for inc in self._incident_history if inc["timestamp"] >= window_cutoff
        ]

        recent_count = len(self._incident_history)
        return dict(self._active_channels), recent_count, list(self._incident_history)

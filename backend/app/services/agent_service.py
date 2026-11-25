"""
Safety Agent Service: Model Management, Telemetry Orchestration, and Scenario Simulation.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from ml.agent.safety_agent import IntelligentSafetyAgent, SafetyAssessment
from ml.agent.config import (
    DEFAULT_CONFIG,
    SafetyState,
    DrivingStyle,
    ManeuverClass,
    STYLE_LABEL_NAMES,
    MANEUVER_LABEL_NAMES,
)
from backend.app.schemas.telemetry import (
    TelemetryPredictionRequest,
    RawFeatureTelemetryRequest,
    AssessmentResponse,
    AlertDetail,
    IncidentDetail,
    ArtifactDetail,
    ModelStatusResponse,
    ScenarioPreset,
)

def _find_repo_root() -> Path:
    _cur = Path(__file__).resolve()
    for p in [
        _cur.parents[3] if len(_cur.parents) > 3 else None,
        _cur.parents[2] if len(_cur.parents) > 2 else None,
        Path.cwd(),
        _cur.parent,
    ]:
        if p and ((p / "models" / "experiments").exists() or (p / "backend").exists()):
            return p
    return _cur.parents[3] if len(_cur.parents) > 3 else Path.cwd()

ROOT_DIR = _find_repo_root()

FROZEN_ARTIFACTS = [
    {
        "name": "Pipeline B Maneuver Extra Trees Model",
        "pipeline": "Pipeline B (Maneuver)",
        "path": "models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib",
        "expected_sha256": "552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90",
    },
    {
        "name": "Pipeline B Maneuver Feature Scaler",
        "pipeline": "Pipeline B (Maneuver)",
        "path": "models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib",
        "expected_sha256": "d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576",
    },
    {
        "name": "Pipeline A Driving Style SVM Model",
        "pipeline": "Pipeline A (Style)",
        "path": "models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib",
        "expected_sha256": "a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc",
    },
    {
        "name": "Pipeline A Driving Style Feature Scaler",
        "pipeline": "Pipeline A (Style)",
        "path": "models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib",
        "expected_sha256": "d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5",
    },
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().lower()


class AgentService:
    """Singleton service coordinating the Safety Agent and ML pipelines."""

    _instance: Optional["AgentService"] = None

    def __init__(self):
        self.agent = IntelligentSafetyAgent(config=DEFAULT_CONFIG, auto_load_classifiers=True)

    @classmethod
    def get_instance(cls) -> "AgentService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset_agent(self):
        """Reset internal agent history and incident lifecycle timers."""
        self.agent.reset()

    def get_model_status(self) -> ModelStatusResponse:
        """Verify model artifacts on disk against frozen cryptographic hashes."""
        details: List[ArtifactDetail] = []
        all_ok = True

        for art in FROZEN_ARTIFACTS:
            rel_path = art["path"]
            abs_path = ROOT_DIR / rel_path
            exists = abs_path.is_file()
            actual_hash = compute_sha256(abs_path) if exists else None
            is_match = (actual_hash == art["expected_sha256"]) if actual_hash else False

            if not is_match:
                all_ok = False

            details.append(
                ArtifactDetail(
                    name=art["name"],
                    pipeline=art["pipeline"],
                    path=rel_path,
                    exists=exists,
                    expected_sha256=art["expected_sha256"],
                    actual_sha256=actual_hash,
                    is_verified=is_match,
                )
            )

        status_text = "All production artifacts verified" if all_ok else "Artifact hash mismatch or missing file"
        return ModelStatusResponse(status=status_text, all_verified=all_ok, artifacts=details)

    def assess_telemetry(self, req: TelemetryPredictionRequest) -> AssessmentResponse:
        """Process pre-computed telemetry predictions through the safety agent."""
        assessment = self.agent.process_predictions(
            timestamp=req.timestamp,
            style_prediction=req.style_prediction,
            style_confidence=req.style_confidence,
            maneuver_prediction=req.maneuver_prediction,
            maneuver_confidence=req.maneuver_confidence,
            pipeline_a_health=req.pipeline_a_health,
            pipeline_b_health=req.pipeline_b_health,
        )
        return self._format_assessment_response(assessment)

    def assess_raw_features(self, req: RawFeatureTelemetryRequest) -> AssessmentResponse:
        """Process raw sensor telemetry features through classifiers and safety agent."""
        df_b = pd.DataFrame([req.features_b]) if req.features_b else None
        df_c = pd.DataFrame([req.features_c]) if req.features_c else None

        assessment = self.agent.process_features(
            timestamp=req.timestamp,
            features_b=df_b,
            features_c=df_c,
        )
        return self._format_assessment_response(assessment)

    def _format_assessment_response(self, assessment: SafetyAssessment) -> AssessmentResponse:
        """Transform SafetyAssessment dataclass into Pydantic AssessmentResponse."""
        alerts = [
            AlertDetail(
                alert_id=a.get("alert_id", "UNKNOWN"),
                timestamp=a.get("timestamp", assessment.timestamp),
                severity=a.get("severity", "MEDIUM"),
                category=a.get("category", "GENERAL"),
                title=a.get("title", "Safety Alert"),
                explanation=a.get("explanation", ""),
                safe_driving_advisory=a.get("safe_driving_advisory", "Maintain defensive driving habits."),
            )
            for a in assessment.active_alerts
        ]

        active_incidents = [
            IncidentDetail(
                event_id=c.event_id,
                maneuver_id=c.maneuver_class,
                maneuver_name=MANEUVER_LABEL_NAMES.get(c.maneuver_class, "Unknown"),
                initial_weight=c.initial_weight,
                confidence=c.confidence,
                is_excited=c.is_excited,
                onset_timestamp=c.onset_timestamp,
                excitation_end_timestamp=c.excitation_end_timestamp,
            )
            for c in getattr(self.agent.smoother.lifecycle_manager, "active_contributions", {}).values()
        ]

        return AssessmentResponse(
            timestamp=assessment.timestamp,
            safety_state=assessment.safety_state.value,
            risk_score=round(assessment.risk_score, 2),
            driving_style=assessment.driving_style,
            driving_style_id=assessment.driving_style_id,
            driving_style_confidence=round(assessment.driving_style_confidence, 4),
            maneuver=assessment.maneuver,
            maneuver_id=assessment.maneuver_id,
            maneuver_confidence=round(assessment.maneuver_confidence, 4),
            pipeline_a_health=assessment.pipeline_a_health,
            pipeline_b_health=assessment.pipeline_b_health,
            is_valid=assessment.is_valid,
            data_quality_status=assessment.data_quality_status,
            triggered_rules=assessment.triggered_rules,
            explanation=assessment.explanation,
            active_alerts=alerts,
            active_incidents=active_incidents,
            confidence_summary={k: round(v, 4) for k, v in assessment.confidence_summary.items()},
        )

    def get_scenarios(self) -> List[ScenarioPreset]:
        """Preset scenarios designed for college mini-project demonstration."""
        return [
            ScenarioPreset(
                id="SCN-01",
                name="Nominal Highway Cruising",
                category="Normal Driving",
                description="Smooth highway cruising with normal driving style and quiescent vehicle dynamics. Mutual pipeline certification satisfied.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=1.0,
                    style_prediction=DrivingStyle.NORMAL.value,
                    style_confidence=0.95,
                    maneuver_prediction=0,
                    maneuver_confidence=0.98,
                    pipeline_a_health="HEALTHY",
                    pipeline_b_health="HEALTHY",
                ),
                expected_state="NORMAL",
                expected_score_range="0.00 pts",
            ),
            ScenarioPreset(
                id="SCN-02",
                name="Emergency Sudden Braking",
                category="Hazard Event",
                description="Driver slams on brakes (Harsh Deceleration, W=50.0). Triggers WARNING state unweakened with passenger destabilization advisory.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=2.0,
                    style_prediction=DrivingStyle.NORMAL.value,
                    style_confidence=0.90,
                    maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
                    maneuver_confidence=0.92,
                    pipeline_a_health="HEALTHY",
                    pipeline_b_health="HEALTHY",
                ),
                expected_state="WARNING",
                expected_score_range="50.00 pts",
            ),
            ScenarioPreset(
                id="SCN-03",
                name="Compound Critical Hazard (Aggressive + Swerve)",
                category="Critical Hazard",
                description="Aggressive tailgating followed by an abrupt lateral right turn swerve. Activates +25.0 compound hazard bonus and CRITICAL alert.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=3.0,
                    style_prediction=DrivingStyle.AGGRESSIVE.value,
                    style_confidence=0.92,
                    maneuver_prediction=ManeuverClass.SUDDEN_RIGHT_TURN.value,
                    maneuver_confidence=0.88,
                    pipeline_a_health="HEALTHY",
                    pipeline_b_health="HEALTHY",
                ),
                expected_state="HIGH_RISK",
                expected_score_range="95.00 - 100.00 pts",
            ),
            ScenarioPreset(
                id="SCN-04",
                name="Pipeline A Sensor Dropout (Option 1A-Pure)",
                category="Fault Tolerance",
                description="Pipeline A (Style camera/telemetry) disconnects during an active maneuver. Option 1A-Pure preserves calculated risk without artificial floor and raises DATA_QUALITY diagnostic.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=4.0,
                    style_prediction=None,
                    style_confidence=None,
                    maneuver_prediction=ManeuverClass.SUDDEN_ACCELERATION.value,
                    maneuver_confidence=0.85,
                    pipeline_a_health="MISSING",
                    pipeline_b_health="HEALTHY",
                ),
                expected_state="CAUTION",
                expected_score_range="25.00 pts",
            ),
            ScenarioPreset(
                id="SCN-05",
                name="Pipeline B Offline during Cruising (P-05A)",
                category="Fail-Safe Clamping",
                description="Pipeline B maneuver IMU offline while driving style is normal. Policy P-05A clamps safety state to minimum CAUTION floor.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=5.0,
                    style_prediction=DrivingStyle.NORMAL.value,
                    style_confidence=0.90,
                    maneuver_prediction=None,
                    maneuver_confidence=None,
                    pipeline_a_health="HEALTHY",
                    pipeline_b_health="MISSING",
                ),
                expected_state="CAUTION",
                expected_score_range="0.00 pts (Clamped CAUTION floor)",
            ),
            ScenarioPreset(
                id="SCN-06",
                name="Dual-Pipeline Communication Outage (Option 2A)",
                category="System Fail-Safe",
                description="Simultaneous dropout of both telemetry pipelines. Triggers INSUFFICIENT_DATA fail-safe and high-severity DATA_QUALITY alert.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=6.0,
                    style_prediction=None,
                    style_confidence=None,
                    maneuver_prediction=None,
                    maneuver_confidence=None,
                    pipeline_a_health="MISSING",
                    pipeline_b_health="MISSING",
                ),
                expected_state="INSUFFICIENT_DATA",
                expected_score_range="0.00 pts",
            ),
            ScenarioPreset(
                id="SCN-07",
                name="Low-Confidence Maneuver Telemetry (D-10 Scaling)",
                category="Noise Filtering",
                description="Maneuver detector fires with very low confidence (C=0.35 < 0.40). D-10 piecewise scaling zeroes the score, completely suppressing false alarms.",
                telemetry=TelemetryPredictionRequest(
                    timestamp=7.0,
                    style_prediction=DrivingStyle.NORMAL.value,
                    style_confidence=0.85,
                    maneuver_prediction=ManeuverClass.SUDDEN_DECELERATION.value,
                    maneuver_confidence=0.35,
                    pipeline_a_health="HEALTHY",
                    pipeline_b_health="HEALTHY",
                ),
                expected_state="NORMAL",
                expected_score_range="0.00 pts",
            ),
        ]

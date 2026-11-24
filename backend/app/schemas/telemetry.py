"""
Pydantic Schemas for Telemetry Ingestion, Inference, and Safety Assessments.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Intelligent Driver Monitoring Agent API"
    version: str = "1.0.0"


class ArtifactDetail(BaseModel):
    name: str
    pipeline: str
    path: str
    exists: bool
    expected_sha256: str
    actual_sha256: Optional[str] = None
    is_verified: bool = False


class ModelStatusResponse(BaseModel):
    status: str
    all_verified: bool
    artifacts: List[ArtifactDetail]


class TelemetryPredictionRequest(BaseModel):
    timestamp: float = Field(..., description="Telemetry timestamp in seconds")
    style_prediction: Optional[int] = Field(default=None, description="0: Normal, 1: Aggressive, None: Missing")
    style_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in [0.0, 1.0]")
    maneuver_prediction: Optional[int] = Field(default=None, description="0: None, 1: SA, 2: SD, 3: SRT, 4: SLT, None: Missing")
    maneuver_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in [0.0, 1.0]")
    pipeline_a_health: Optional[str] = Field(default=None, description="Modality health override (HEALTHY, DEGRADED, MISSING, INVALID, QUIESCENT_BYPASS)")
    pipeline_b_health: Optional[str] = Field(default=None, description="Modality health override (HEALTHY, DEGRADED, MISSING, INVALID, QUIESCENT_BYPASS)")


class RawFeatureTelemetryRequest(BaseModel):
    timestamp: float = Field(..., description="Telemetry timestamp in seconds")
    features_b: Optional[Dict[str, float]] = Field(default=None, description="IMU sensor features for Pipeline B (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, etc.)")
    features_c: Optional[Dict[str, float]] = Field(default=None, description="Steering & vehicle speed features for Pipeline A")


class AlertDetail(BaseModel):
    alert_id: str
    timestamp: float
    severity: str
    category: str
    title: str
    explanation: str
    safe_driving_advisory: str


class IncidentDetail(BaseModel):
    event_id: str
    maneuver_id: int
    maneuver_name: str
    initial_weight: float
    confidence: float
    is_excited: bool
    onset_timestamp: float
    excitation_end_timestamp: Optional[float] = None


class AssessmentResponse(BaseModel):
    timestamp: float
    safety_state: str
    risk_score: float
    driving_style: str
    driving_style_id: Optional[int]
    driving_style_confidence: float
    maneuver: str
    maneuver_id: Optional[int]
    maneuver_confidence: float
    pipeline_a_health: str
    pipeline_b_health: str
    is_valid: bool
    data_quality_status: str
    triggered_rules: List[str]
    explanation: str
    active_alerts: List[AlertDetail]
    active_incidents: List[IncidentDetail] = []
    confidence_summary: Dict[str, float] = {}


class ScenarioPreset(BaseModel):
    id: str
    name: str
    category: str
    description: str
    telemetry: TelemetryPredictionRequest
    expected_state: str
    expected_score_range: str


class ReplaySessionSummary(BaseModel):
    id: str
    title: str
    driver_id: str
    duration_seconds: float
    total_frames: int
    event_counts: Dict[str, int]
    description: str


class ReplayFrame(BaseModel):
    frame_index: int
    timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    speed: float
    steering_angle: float
    ground_truth_label: Optional[str] = None
    telemetry_request: TelemetryPredictionRequest
    features_b: Optional[Dict[str, float]] = None


class ReplaySessionDetail(BaseModel):
    id: str
    title: str
    driver_id: str
    duration_seconds: float
    total_frames: int
    event_counts: Dict[str, int]
    description: str
    frames: List[ReplayFrame]


class HardwareStatusResponse(BaseModel):
    connected: bool
    status: str
    message: str
    port: Optional[str] = None
    protocol: Optional[str] = None
    supported_sensors: List[str]
    missing_requirements: List[str]


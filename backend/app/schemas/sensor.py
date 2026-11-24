"""
Pydantic Schemas for Physical Sensor Telemetry, Device Registration, Ingestion, and Quality Reporting.
=====================================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DeviceRegistrationRequest(BaseModel):
    device_name: str = Field(..., description="Device model and manufacturer, e.g. Pixel 8 Pro")
    os_version: str = Field(..., description="Operating system version, e.g. Android 14")
    app_version: str = Field(default="1.0.0", description="Sensor logger application version")
    assigned_driver_id: str = Field(..., description="Assigned anonymized driver token, e.g. DRV_001")
    hardware_capabilities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Sensors available")


class DeviceRegistrationResponse(BaseModel):
    device_token: str
    assigned_driver_id: str
    status: str = "REGISTERED"
    target_frequency_hz: float = 20.0
    message: str = "Device registered successfully for physical telemetry acquisition"


class SessionStartRequest(BaseModel):
    device_token: str
    driver_id: str
    vehicle_id: str = Field(..., description="Vehicle token, e.g. VEH_01 (Hatchback) or VEH_02 (Sedan)")
    initial_pitch_deg: Optional[float] = Field(default=None, description="Pre-drive static pitch angle")
    initial_roll_deg: Optional[float] = Field(default=None, description="Pre-drive static roll angle")
    notes: Optional[str] = Field(default=None, description="Operational notes")


class SessionStartResponse(BaseModel):
    session_id: str
    driver_id: str
    vehicle_id: str
    status: str = "ACTIVE"
    start_epoch_ms: int
    message: str = "Recording session initialized. Begin S0 30-second static calibration."


class TelemetryFrame(BaseModel):
    timestamp: float = Field(..., description="Elapsed seconds since session start")
    timestamp_epoch_ms: int = Field(..., description="UTC epoch timestamp in milliseconds")
    acc_x: float = Field(..., alias="Acc X", description="Lateral acceleration in g")
    acc_y: float = Field(..., alias="Acc Y", description="Longitudinal acceleration in g")
    acc_z: float = Field(..., alias="Acc Z", description="Vertical acceleration in g")
    gyro_x: float = Field(..., description="Pitch rate in rad/s")
    gyro_y: float = Field(..., description="Roll rate in rad/s")
    gyro_z: float = Field(..., description="Yaw rate in rad/s")
    speed_mps: Optional[float] = Field(default=0.0, description="GPS ground speed in m/s")
    heading_deg: Optional[float] = Field(default=0.0, description="GPS heading in degrees")
    latitude: Optional[float] = Field(default=0.0, description="GPS latitude")
    longitude: Optional[float] = Field(default=0.0, description="GPS longitude")
    altitude: Optional[float] = Field(default=0.0, description="Altitude in meters")
    hdop: Optional[float] = Field(default=1.0, description="GPS HDOP")
    segment_id: Optional[str] = Field(default="S1", description="Driving regime S0-S6")

    class Config:
        populate_by_name = True


class BatchTelemetryIngestRequest(BaseModel):
    session_id: str
    driver_id: str
    frames: List[TelemetryFrame]


class IngestionUploadResponse(BaseModel):
    session_id: str
    driver_id: str
    file_name: str
    file_size_bytes: int
    sample_count: int
    duration_seconds: float
    sha256: str
    data_quality_status: str
    is_duplicate: bool = False
    message: str


class QualityCriterionResult(BaseModel):
    criterion_name: str
    status: str  # PASS, FAIL, WARNING, NOT_AVAILABLE
    measured_value: Any
    threshold: str
    details: str


class DataQualityReport(BaseModel):
    session_id: str
    driver_id: str
    timestamp_generated: str
    overall_status: str  # PASS, FAIL, WARNING
    total_samples: int
    duration_seconds: float
    sampling_frequency_hz: float
    criteria_results: List[QualityCriterionResult]
    regime_breakdown: Dict[str, float]
    mount_drift_deg: Optional[float] = None
    recommendation: str


class LiveDeviceStatusResponse(BaseModel):
    connected: bool
    status: str  # STREAMING, STANDBY, DISCONNECTED
    device_token: Optional[str] = None
    driver_id: Optional[str] = None
    session_id: Optional[str] = None
    actual_frequency_hz: float = 0.0
    latest_frame: Optional[Dict[str, Any]] = None
    last_heartbeat_epoch_ms: Optional[int] = None
    quality_status: str = "NOT_AVAILABLE"
    message: str = ""

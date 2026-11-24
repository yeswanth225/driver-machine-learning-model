"""
FastAPI Route Endpoints for Telemetry Assessment, Pipeline Status, and Demonstrations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends

from backend.app.schemas.telemetry import (
    HealthResponse,
    ModelStatusResponse,
    TelemetryPredictionRequest,
    RawFeatureTelemetryRequest,
    AssessmentResponse,
    ScenarioPreset,
    ReplaySessionSummary,
    ReplaySessionDetail,
    HardwareStatusResponse,
)
from backend.app.services.agent_service import AgentService
from backend.app.services.replay_service import ReplayService
from backend.app.services.analytics_service import AnalyticsService

router = APIRouter()


def get_agent_service() -> AgentService:
    return AgentService.get_instance()


def get_replay_service() -> ReplayService:
    return ReplayService.get_instance()


def get_analytics_service() -> AnalyticsService:
    return AnalyticsService.get_instance()


@router.get("/health", response_model=HealthResponse, summary="API Health Check")
def health_check():
    """Verify backend API responsiveness."""
    return HealthResponse()


@router.get("/models/status", response_model=ModelStatusResponse, summary="Verify ML Model Artifacts & Hashes")
def get_models_status(service: AgentService = Depends(get_agent_service)):
    """Check existence and cryptographic SHA-256 integrity of all 4 frozen production artifacts."""
    return service.get_model_status()


@router.post("/telemetry/assess", response_model=AssessmentResponse, summary="Assess Telemetry Predictions")
def assess_telemetry(
    request: TelemetryPredictionRequest,
    service: AgentService = Depends(get_agent_service),
):
    """
    Ingest multi-modal predictions, execute two-layer validation,
    multi-instance lifecycle tracking, and continuous residual risk scoring.
    """
    try:
        return service.assess_telemetry(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Telemetry assessment error: {str(e)}")


@router.post("/telemetry/assess/raw", response_model=AssessmentResponse, summary="Assess Raw Sensor Telemetry")
def assess_raw_features(
    request: RawFeatureTelemetryRequest,
    service: AgentService = Depends(get_agent_service),
):
    """
    Ingest raw feature vectors, run Pipeline A & Pipeline B inference,
    evaluate KinematicEventGate, and produce safety assessment.
    """
    try:
        return service.assess_raw_features(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature inference error: {str(e)}")


@router.get("/simulated/scenarios", response_model=List[ScenarioPreset], summary="List Demonstration Scenarios")
def get_demonstration_scenarios(service: AgentService = Depends(get_agent_service)):
    """Retrieve pre-configured scenarios illustrating normal, hazard, compound, and fail-safe behaviors."""
    return service.get_scenarios()


@router.post("/agent/reset", summary="Reset Agent State & Incident Timers")
def reset_agent(service: AgentService = Depends(get_agent_service)):
    """Reset temporal smoothing buffers, incident hysteresis timers, and alert cooldowns."""
    service.reset_agent()
    return {"status": "success", "message": "Safety Agent internal state reset successfully"}


@router.get("/replay/sessions", response_model=List[ReplaySessionSummary], summary="List Recorded Driving Sessions")
def list_replay_sessions(replay_service: ReplayService = Depends(get_replay_service)):
    """Retrieve historical driving sessions captured from real test vehicle deployments."""
    try:
        return replay_service.get_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load replay sessions: {str(e)}")


@router.get("/replay/sessions/{session_id}", response_model=ReplaySessionDetail, summary="Get Recorded Driving Session Detail")
def get_replay_session(session_id: str, replay_service: ReplayService = Depends(get_replay_service)):
    """Retrieve frame-by-frame telemetry and model inference outputs for a specific recorded session."""
    session = replay_service.get_session_detail(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Replay session '{session_id}' not found")
    return session


@router.get("/analytics/overview", summary="Get Verified Evaluation & Benchmark Analytics")
def get_analytics_overview(analytics_service: AnalyticsService = Depends(get_analytics_service)):
    """Retrieve cryptographically verified test metrics, confusion matrices, and streaming latency statistics."""
    return analytics_service.get_overview()


@router.get("/analytics/unified-registry", summary="Get Unified Multi-Model Registry & Performance Benchmarks")
def get_unified_registry(analytics_service: AnalyticsService = Depends(get_analytics_service)):
    """
    Retrieve the dynamic multi-model registry comparing all evaluated models across
    Pipeline A, Pipeline B, and the Safety Agent, including distinct metric winners.
    """
    return analytics_service.get_unified_registry()


@router.get("/system/hardware-status", response_model=HardwareStatusResponse, summary="Inspect Live Physical Sensor Hardware")
def get_hardware_status():
    """
    Inspect physical vehicle sensor infrastructure (Mobile Smartphone IMU/GPS; CAN/OBD planned future hardware abstraction).
    Honest reporting: Detects connected mobile smartphone sensor logger or returns offline status.
    """
    from backend.app.services.live_telemetry_service import LiveTelemetryService
    live_status = LiveTelemetryService.get_instance().get_live_status()

    if live_status.connected:
        return HardwareStatusResponse(
            connected=True,
            status=live_status.status,
            message=f"Physical smartphone sensor logger active ({live_status.device_token}, {live_status.actual_frequency_hz:.1f} Hz).",
            port="Mobile Sensor Logger (WebSocket / HTTP)",
            protocol=f"Mobile Sensor Telemetry @ {live_status.actual_frequency_hz:.1f}Hz",
            supported_sensors=[
                "Smartphone 3-Axis Accelerometer (10-20 Hz)",
                "Smartphone 3-Axis Gyroscope (10-20 Hz)",
                "Smartphone GPS Ground Speed & Heading",
                "CAN Bus 2.0B / OBD-II (Planned future hardware abstraction - active ingestion uses Smartphone IMU/GPS)",
            ],
            missing_requirements=[],
        )

    return HardwareStatusResponse(
        connected=False,
        status="DISCONNECTED",
        message="No physical smartphone logger or vehicle sensor hardware currently transmitting.",
        port=None,
        protocol=None,
        supported_sensors=[
            "Smartphone 3-Axis Accelerometer (10-20 Hz)",
            "Smartphone 3-Axis Gyroscope (10-20 Hz)",
            "Smartphone GPS Ground Speed & Heading",
            "CAN Bus 2.0B / OBD-II (Planned future hardware abstraction - active ingestion uses Smartphone IMU/GPS)",
        ],
        missing_requirements=[
            "Launch Driver Sensor Logger app or open http://<host>:8000/sensor-logger/ on smartphone",
            "Connect phone to same local Wi-Fi or hotspot",
            "Attach smartphone to rigid windshield mount and start acquisition",
        ],
    )


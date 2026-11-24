"""
FastAPI Route Endpoints for Physical Sensor Integration, Device Registration, Ingestion, and Quality Reporting.
=============================================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: backend/app/api/sensor_endpoints.py
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from pathlib import Path

from backend.app.schemas.sensor import (
    DeviceRegistrationRequest,
    DeviceRegistrationResponse,
    SessionStartRequest,
    SessionStartResponse,
    TelemetryFrame,
    IngestionUploadResponse,
    DataQualityReport,
    LiveDeviceStatusResponse,
)
from backend.app.services.sensor_storage_service import SensorStorageService, BASE_RAW_DIR
from backend.app.services.data_quality_service import DataQualityService
from backend.app.services.live_telemetry_service import LiveTelemetryService
from backend.app.services.annotation_service import AnnotationService, AnnotationRecord

sensor_router = APIRouter(prefix="/sensor", tags=["Physical Sensor Integration"])


def get_storage_service() -> SensorStorageService:
    return SensorStorageService.get_instance()


def get_quality_service() -> DataQualityService:
    return DataQualityService.get_instance()


def get_live_service() -> LiveTelemetryService:
    return LiveTelemetryService.get_instance()


def get_annotation_service() -> AnnotationService:
    return AnnotationService.get_instance()


@sensor_router.post("/device/register", response_model=DeviceRegistrationResponse, summary="Register Physical Smartphone Device")
def register_device(
    req: DeviceRegistrationRequest,
    live_service: LiveTelemetryService = Depends(get_live_service),
    storage_service: SensorStorageService = Depends(get_storage_service),
):
    """Registers an Android/iOS smartphone device for physical sensor acquisition."""
    try:
        storage_service.validate_driver_id(req.assigned_driver_id)
        return live_service.register_device(req)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@sensor_router.post("/session/start", response_model=SessionStartResponse, summary="Start Physical Telemetry Recording Session")
def start_session(
    req: SessionStartRequest,
    live_service: LiveTelemetryService = Depends(get_live_service),
    storage_service: SensorStorageService = Depends(get_storage_service),
):
    """Initializes a new physical telemetry recording session."""
    try:
        storage_service.validate_driver_id(req.driver_id)
        import time
        session_id = f"sess_{int(time.time())}_{req.driver_id}"
        live_service.set_active_session(session_id, req.driver_id, req.device_token)

        return SessionStartResponse(
            session_id=session_id,
            driver_id=req.driver_id,
            vehicle_id=req.vehicle_id,
            status="ACTIVE",
            start_epoch_ms=int(time.time() * 1000),
            message="Session active. Execute S0 30-second static calibration.",
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@sensor_router.post("/telemetry/frame", summary="Ingest Single Real-Time Telemetry Frame")
async def ingest_telemetry_frame(
    frame: TelemetryFrame,
    live_service: LiveTelemetryService = Depends(get_live_service),
):
    """Ingests a real-time 20 Hz frame from the mobile device and broadcasts to live dashboards."""
    try:
        await live_service.broadcast_frame(frame)
        return {"status": "ACK", "timestamp": frame.timestamp}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame ingestion error: {str(e)}")


@sensor_router.post("/upload", response_model=IngestionUploadResponse, summary="Upload Completed Raw Telemetry Recording (CSV)")
async def upload_raw_telemetry(
    driver_id: str = Form(...),
    session_id: str = Form(...),
    client_sha256: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    storage_service: SensorStorageService = Depends(get_storage_service),
    quality_service: DataQualityService = Depends(get_quality_service),
):
    """
    Ingests completed raw telemetry CSV, verifies SHA-256 integrity,
    runs automated 6-criterion Data Quality acceptance evaluation, and stores immutably.
    """
    try:
        # Enforce maximum upload size limit (50 MB) to prevent memory exhaustion
        MAX_UPLOAD_BYTES = 50 * 1024 * 1024
        csv_bytes = await file.read()
        if len(csv_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Payload Too Large: File exceeds maximum allowed size of 50 MB.")

        storage_res = storage_service.store_raw_recording(
            driver_id=driver_id,
            session_id=session_id,
            csv_bytes=csv_bytes,
            client_sha256=client_sha256,
        )

        if storage_res.get("is_duplicate", False):
            return IngestionUploadResponse(
                session_id=storage_res["session_id"],
                driver_id=driver_id,
                file_name=file.filename or "unknown.csv",
                file_size_bytes=len(csv_bytes),
                sample_count=0,
                duration_seconds=0.0,
                sha256=storage_res["sha256"],
                data_quality_status="DUPLICATE_REJECTED",
                is_duplicate=True,
                message=storage_res["message"],
            )

        # Run automated data quality validation
        df = storage_service.get_session_telemetry_df(driver_id, session_id)
        if df is not None:
            report = quality_service.evaluate_session_df(df, driver_id, session_id)
            sess_dir = BASE_RAW_DIR / driver_id / session_id
            quality_service.save_quality_reports(report, sess_dir)
            quality_status = report.overall_status
        else:
            quality_status = "NOT_EVALUATED"

        return IngestionUploadResponse(
            session_id=session_id,
            driver_id=driver_id,
            file_name=storage_res["file_name"],
            file_size_bytes=storage_res["file_size_bytes"],
            sample_count=storage_res["sample_count"],
            duration_seconds=storage_res["duration_seconds"],
            sha256=storage_res["sha256"],
            data_quality_status=quality_status,
            is_duplicate=False,
            message="Raw telemetry recording stored and validated successfully.",
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileExistsError as fe:
        raise HTTPException(status_code=409, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error during ingestion: {str(e)}")


@sensor_router.get("/sessions", summary="List All Ingested Physical Driving Sessions")
def list_sessions(storage_service: SensorStorageService = Depends(get_storage_service)):
    """Retrieves metadata manifests for all ingested physical recording sessions."""
    return storage_service.list_all_sessions()


@sensor_router.get("/sessions/{driver_id}/{session_id}/quality", response_model=DataQualityReport, summary="Get Data Quality Report for Session")
def get_session_quality_report(
    driver_id: str,
    session_id: str,
    storage_service: SensorStorageService = Depends(get_storage_service),
    quality_service: DataQualityService = Depends(get_quality_service),
):
    """Retrieves or re-computes the 6-criterion Data Quality Report for a session."""
    try:
        storage_service.validate_driver_id(driver_id)
        storage_service.validate_session_id(session_id)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    report_json = BASE_RAW_DIR / driver_id / session_id / "data_quality_report.json"
    if report_json.exists():
        import json
        with open(report_json, "r") as f:
            return json.load(f)

    df = storage_service.get_session_telemetry_df(driver_id, session_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found for driver '{driver_id}'.")

    report = quality_service.evaluate_session_df(df, driver_id, session_id)
    quality_service.save_quality_reports(report, BASE_RAW_DIR / driver_id / session_id)
    return report


@sensor_router.get("/live/status", response_model=LiveDeviceStatusResponse, summary="Get Live Physical Sensor Hardware & Connection Status")
def get_live_status(live_service: LiveTelemetryService = Depends(get_live_service)):
    """
    Returns real-time physical device connection state, measured frequency,
    and latest telemetry frame.
    """
    return live_service.get_live_status()


@sensor_router.get("/live/frames", summary="Get Recent Telemetry Frames for Frontend Charts")
def get_recent_frames(
    count: int = 50,
    live_service: LiveTelemetryService = Depends(get_live_service)
):
    """Retrieves recent rolling frames for live dashboard rendering."""
    return live_service.get_recent_frames(count)


@sensor_router.post("/annotations", summary="Record Independent Human Observer Ground-Truth Annotation")
def create_annotation(
    record: AnnotationRecord,
    annotation_service: AnnotationService = Depends(get_annotation_service)
):
    """Records an observer ground-truth annotation for a specific session time window."""
    try:
        return annotation_service.create_annotation(record)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@sensor_router.get("/annotations/{session_id}", summary="Get Annotations for Session")
def get_annotations(
    session_id: str,
    annotation_service: AnnotationService = Depends(get_annotation_service)
):
    """Retrieves all human ground-truth annotations for a session."""
    return annotation_service.get_session_annotations(session_id)


@sensor_router.websocket("/live/stream")
async def websocket_telemetry_stream(
    websocket: WebSocket,
    live_service: LiveTelemetryService = Depends(get_live_service)
):
    """Bi-directional WebSocket streaming telemetry channel."""
    await websocket.accept()
    live_service.register_websocket(websocket)
    try:
        while True:
            # Keep-alive or handle client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        live_service.unregister_websocket(websocket)
    except Exception:
        live_service.unregister_websocket(websocket)

"""
Integration & Unit Test Suite for Physical Sensor Acquisition, Ingestion, Quality Engine & ML Compatibility.
=============================================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: tests/test_sensor_integration.py
"""

import io
import time
import math
import hashlib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.sensor_storage_service import (
    SensorStorageService,
    DEVELOPMENT_DRIVERS,
    RESERVED_LOCKBOX_DRIVERS,
)
from backend.app.services.data_quality_service import (
    DataQualityService,
    DataQualityReport,
)
from backend.app.services.annotation_service import (
    AnnotationService,
    AnnotationRecord,
)
from backend.app.services.live_telemetry_service import LiveTelemetryService
from ml.inference.sampling_compatibility_adapter import (
    CausalResampler20HzTo10Hz,
    Native20HzCausalFeatureExtractor,
)
from ml.inference.physical_sensor_adapter import PhysicalSensorInferenceAdapter


client = TestClient(app)


# =====================================================================
# 1. LOCKBOX DRIVER ISOLATION TESTS
# =====================================================================

def test_lockbox_isolation_rules():
    """Verify that development drivers are accepted and lockbox drivers are strictly forbidden."""
    storage = SensorStorageService.get_instance()

    # Valid development drivers
    for drv in DEVELOPMENT_DRIVERS:
        storage.validate_driver_id(drv)

    # Blind lockbox drivers MUST raise PermissionError
    for locked in RESERVED_LOCKBOX_DRIVERS:
        with pytest.raises(PermissionError) as exc_info:
            storage.validate_driver_id(locked)
        assert "Lockbox" in str(exc_info.value) or "LOCKBOX" in str(exc_info.value)

    # Invalid driver patterns MUST raise ValueError
    for invalid in ["DRV_099", "DRIVER_1", "TEST_001", "DRV001"]:
        with pytest.raises(ValueError):
            storage.validate_driver_id(invalid)


def test_lockbox_rejection_via_endpoints():
    """Verify API endpoints return HTTP 403 Forbidden when attempting to register or upload lockbox drivers."""
    # Attempt registration with DRV_007
    reg_resp = client.post(
        "/api/sensor/device/register",
        json={
            "device_name": "Pixel 7 Pro",
            "os_version": "Android 14",
            "app_version": "1.0.0",
            "assigned_driver_id": "DRV_007",
        },
    )
    assert reg_resp.status_code == 403
    assert "Lockbox" in reg_resp.json()["detail"] or "LOCKBOX" in reg_resp.json()["detail"]

    # Attempt registration with DRV_008
    reg_resp2 = client.post(
        "/api/sensor/device/register",
        json={
            "device_name": "Pixel 7 Pro",
            "os_version": "Android 14",
            "app_version": "1.0.0",
            "assigned_driver_id": "DRV_008",
        },
    )
    assert reg_resp2.status_code == 403


# =====================================================================
# 2. DATA INGESTION & DUPLICATE REJECTION TESTS
# =====================================================================

def _generate_synthetic_20hz_csv(n_seconds: int = 35, inject_gap: bool = False, gravity_offset: float = 0.0) -> bytes:
    """Helper to generate valid 20 Hz CSV telemetry bytes matching MANDATORY_CSV_COLUMNS."""
    rows = []
    n_samples = n_seconds * 20
    t0_ms = 1700000000000

    for i in range(n_samples):
        # 50 ms step
        dt = i * 0.05
        if inject_gap and i == 200:
            dt += 0.300  # 300 ms gap

        t_elapsed = dt
        epoch_ms = t0_ms + int(dt * 1000)

        # Baseline: static vertical gravity ~ 1.0g + gravity_offset
        ax = float(np.random.normal(0.0, 0.02))
        ay = float(np.random.normal(0.0, 0.02))
        az = float(-1.0 + gravity_offset + np.random.normal(0.0, 0.01))

        gx = float(np.random.normal(0.0, 0.005))
        gy = float(np.random.normal(0.0, 0.005))
        gz = float(np.random.normal(0.0, 0.005))

        speed_mps = 18.0 + 0.1 * math.sin(dt)
        heading_deg = 90.0
        lat = 37.7749 + dt * 0.00001
        lon = -122.4194 + dt * 0.00001
        alt = 50.0
        hdop = 1.2
        seg_id = "S0" if dt <= 30.0 else "S1"

        rows.append(f"{t_elapsed:.4f},{epoch_ms},{ax:.4f},{ay:.4f},{az:.4f},{gx:.5f},{gy:.5f},{gz:.5f},{speed_mps:.2f},{heading_deg:.1f},{lat:.6f},{lon:.6f},{alt:.1f},{hdop:.1f},{seg_id}")

    header = "timestamp,timestamp_epoch_ms,Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z,speed_mps,heading_deg,latitude,longitude,altitude,hdop,segment_id\n"
    return (header + "\n".join(rows)).encode("utf-8")


def test_duplicate_upload_rejection():
    """Verify that uploading the exact same CSV twice detects duplicate SHA-256 and rejects."""
    import shutil
    from backend.app.services.sensor_storage_service import BASE_RAW_DIR
    for sess_id in ["sess_test_duplicate_01", "sess_test_duplicate_02"]:
        p = BASE_RAW_DIR / "DRV_001" / sess_id
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    storage = SensorStorageService.get_instance()
    csv_bytes = _generate_synthetic_20hz_csv(n_seconds=10)

    try:
        # First store
        res1 = storage.store_raw_recording(
            driver_id="DRV_001",
            session_id="sess_test_duplicate_01",
            csv_bytes=csv_bytes,
        )
        assert res1["is_duplicate"] is False
        assert res1["sha256"] == hashlib.sha256(csv_bytes).hexdigest()

        # Second store of the exact same content
        res2 = storage.store_raw_recording(
            driver_id="DRV_001",
            session_id="sess_test_duplicate_02",
            csv_bytes=csv_bytes,
        )
        assert res2["is_duplicate"] is True
        assert "Duplicate" in res2["message"]
    finally:
        for sess_id in ["sess_test_duplicate_01", "sess_test_duplicate_02"]:
            p = BASE_RAW_DIR / "DRV_001" / sess_id
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)


# =====================================================================
# 3. DATA QUALITY ENGINE TESTS (6 ACCEPTANCE CRITERIA)
# =====================================================================

def test_quality_service_healthy_session():
    """Verify clean 20 Hz synthetic session passes all quality checks."""
    csv_bytes = _generate_synthetic_20hz_csv(n_seconds=40)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    quality = DataQualityService.get_instance()
    report = quality.evaluate_session_df(df, driver_id="DRV_002", session_id="sess_healthy_01")

    assert report.overall_status in ["PASS", "WARNING"]
    c1 = next(c for c in report.criteria_results if "Criterion 1" in c.criterion_name)
    assert c1.status == "PASS"

    c2 = next(c for c in report.criteria_results if "Criterion 2" in c.criterion_name)
    assert c2.status == "PASS"

    c3 = next(c for c in report.criteria_results if "Criterion 3" in c.criterion_name)
    assert c3.status == "PASS"


def test_quality_service_gap_rejection():
    """Verify that timestamp gaps >= 250 ms are flagged as FAIL."""
    csv_bytes = _generate_synthetic_20hz_csv(n_seconds=40, inject_gap=True)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    quality = DataQualityService.get_instance()
    report = quality.evaluate_session_df(df, driver_id="DRV_002", session_id="sess_gap_01")

    c1 = next(c for c in report.criteria_results if "Criterion 1" in c.criterion_name)
    assert c1.status == "FAIL"
    assert c1.measured_value["large_gaps"] > 0


def test_quality_service_gravity_calibration_failure():
    """Verify that abnormal static gravity magnitude fails static calibration."""
    # Offset gravity by -0.3g (norm ~ 1.30g, outside [0.95, 1.05] bounds)
    csv_bytes = _generate_synthetic_20hz_csv(n_seconds=40, gravity_offset=-0.3)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    quality = DataQualityService.get_instance()
    report = quality.evaluate_session_df(df, driver_id="DRV_003", session_id="sess_grav_fail")

    c3 = next(c for c in report.criteria_results if "Criterion 3" in c.criterion_name)
    assert c3.status == "FAIL"


def test_quality_service_mount_drift():
    """Verify that post-drive mount rotation > 3.0 degrees triggers drift failure."""
    csv_bytes = _generate_synthetic_20hz_csv(n_seconds=40)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Add S6 post-drive segment with 10 degree rotation
    df["segment_id"] = "S1"
    df.loc[:100, "segment_id"] = "S0"
    df.loc[len(df)-100:, "segment_id"] = "S6"

    theta = math.radians(10.0)
    s6_idx = df[df["segment_id"] == "S6"].index
    ay_old = df.loc[s6_idx, "Acc Y"].copy()
    az_old = df.loc[s6_idx, "Acc Z"].copy()
    df.loc[s6_idx, "Acc Y"] = ay_old * math.cos(theta) - az_old * math.sin(theta)
    df.loc[s6_idx, "Acc Z"] = ay_old * math.sin(theta) + az_old * math.cos(theta)

    quality = DataQualityService.get_instance()
    report = quality.evaluate_session_df(df, driver_id="DRV_003", session_id="sess_drift_fail")

    c3 = next(c for c in report.criteria_results if "Criterion 3" in c.criterion_name)
    assert c3.status == "FAIL"
    assert c3.measured_value["mount_drift_deg"] > 3.0


# =====================================================================
# 4. SAMPLING FREQUENCY COMPATIBILITY ADAPTERS (10 HZ vs 20 HZ)
# =====================================================================

def test_causal_resampler_decimation():
    """Verify Path A: Causal 20Hz -> 10Hz resampler emits 1 sample per 2 input frames with zero lookahead."""
    resampler = CausalResampler20HzTo10Hz()

    # Feed frame 1 -> should return None (waiting for 2nd sample)
    s1 = {"timestamp": 1.00, "timestamp_epoch_ms": 1700000001000, "Acc X": 0.1, "Acc Y": 0.2, "Acc Z": -1.0, "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.03, "speed_mps": 10.0}
    out1 = resampler.process_sample(s1)
    assert out1 is None
    assert resampler.output_count == 0

    # Feed frame 2 -> should return decimated sample
    s2 = {"timestamp": 1.05, "timestamp_epoch_ms": 1700000001050, "Acc X": 0.12, "Acc Y": 0.22, "Acc Z": -1.02, "gyro_x": 0.012, "gyro_y": 0.022, "gyro_z": 0.032, "speed_mps": 10.1}
    out2 = resampler.process_sample(s2)
    assert out2 is not None
    assert resampler.output_count == 1
    assert "timestamp" in out2
    assert "Acc X" in out2
    assert pytest.approx(out2["Acc X"], 1e-4) == 0.11

    # Reset
    resampler.reset()
    assert resampler.output_count == 0


def test_native_20hz_extractor_physical_time_windows():
    """Verify Path B: Native 20Hz extractor window lengths match physical durations."""
    extractor = Native20HzCausalFeatureExtractor()

    # Feed 100 samples (5.0 seconds at 20 Hz)
    feats = None
    for i in range(100):
        t = 1.0 + i * 0.05
        sample = {
            "timestamp": t,
            "Acc X": 0.05 * math.sin(i * 0.1),
            "Acc Y": 0.02 * math.cos(i * 0.1),
            "Acc Z": -1.0,
            "gyro_x": 0.001,
            "gyro_y": 0.001,
            "gyro_z": 0.002,
            "speed_mps": 15.0,
        }
        feats = extractor.process_sample(sample)

    assert extractor.sample_count == 100
    assert extractor.is_warmed_up is True
    assert len(feats) == len(extractor.feature_names)


def test_physical_sensor_inference_adapter_warmup():
    """Verify that PhysicalSensorInferenceAdapter honors warm-up requirements before predicting."""
    adapter = PhysicalSensorInferenceAdapter(use_production_model=False)

    # Feed 40 frames at 20 Hz (2.0 seconds) -> still in warm-up
    for i in range(40):
        sample = {
            "timestamp": 1.0 + i * 0.05,
            "timestamp_epoch_ms": int(1000 + i * 50),
            "Acc X": 0.05,
            "Acc Y": 0.02,
            "Acc Z": -1.0,
            "gyro_x": 0.001,
            "gyro_y": 0.001,
            "gyro_z": 0.002,
            "speed_mps": 15.0,
        }
        res = adapter.process_raw_20hz_sample(sample)
        assert res["status"] in ["WARMING_UP", "WAITING_DECIMATION"]

    # Feed 80 more frames (total 120 frames at 20 Hz = 60 decimated 10 Hz samples, satisfying 50 warm-up)
    last_res = None
    for i in range(40, 120):
        sample = {
            "timestamp": 1.0 + i * 0.05,
            "timestamp_epoch_ms": int(1000 + i * 50),
            "Acc X": 0.05,
            "Acc Y": 0.02,
            "Acc Z": -1.0,
            "gyro_x": 0.001,
            "gyro_y": 0.001,
            "gyro_z": 0.002,
            "speed_mps": 15.0,
        }
        last_res = adapter.process_raw_20hz_sample(sample)

    assert last_res["status"] in ["ACTIVE_INFERENCE", "WAITING_DECIMATION"]
    if last_res["status"] == "ACTIVE_INFERENCE":
        assert "driving_style" in last_res
        assert "confidence" in last_res


# =====================================================================
# 5. LIVE ENDPOINTS & TELEMETRY STREAMING TESTS
# =====================================================================

def test_device_registration_and_live_status():
    """Verify registering a device updates live status and hardware status endpoint."""
    reg_resp = client.post(
        "/api/sensor/device/register",
        json={
            "device_name": "Google Pixel 8",
            "os_version": "Android 14",
            "app_version": "1.0.0",
            "assigned_driver_id": "DRV_001",
        },
    )
    assert reg_resp.status_code == 200
    token = reg_resp.json()["device_token"]
    assert token.startswith("dev_")

    # Start session
    sess_resp = client.post(
        "/api/sensor/session/start",
        json={
            "device_token": token,
            "driver_id": "DRV_001",
            "vehicle_id": "VH_CIVIC_2022",
            "notes": "Arterial Highway S1",
        },
    )
    assert sess_resp.status_code == 200
    assert sess_resp.json()["status"] == "ACTIVE"

    # Ingest a frame
    frame_resp = client.post(
        "/api/sensor/telemetry/frame",
        json={
            "timestamp": 1.0,
            "timestamp_epoch_ms": 1700000001000,
            "Acc X": 0.05,
            "Acc Y": -0.02,
            "Acc Z": -1.0,
            "gyro_x": 0.001,
            "gyro_y": 0.002,
            "gyro_z": 0.003,
            "speed_mps": 18.5,
            "latitude": 37.7749,
            "longitude": -122.4194,
        },
    )
    assert frame_resp.status_code == 200

    # Query live status
    live_resp = client.get("/api/sensor/live/status")
    assert live_resp.status_code == 200
    live_data = live_resp.json()
    assert live_data["status"] in ["CONNECTED", "STREAMING", "STANDBY"]
    assert live_data["driver_id"] == "DRV_001"

    # Query system hardware status
    hw_resp = client.get("/api/system/hardware-status")
    assert hw_resp.status_code == 200
    hw_data = hw_resp.json()
    assert hw_data["connected"] is True
    assert hw_data["status"] in ["ONLINE", "STREAMING"]


# =====================================================================
# 6. OBSERVER GROUND-TRUTH ANNOTATION TESTS
# =====================================================================

def test_observer_annotations_lifecycle():
    """Verify observer ground truth annotation recording and retrieval."""
    sess_id = f"sess_annot_test_{int(time.time())}"

    # Create annotation
    annot_payload = {
        "session_id": sess_id,
        "driver_id": "DRV_004",
        "annotator_id": "IN_CAR_OBSERVER_01",
        "start_timestamp": 12.5,
        "end_timestamp": 15.0,
        "label": "SUDDEN_ACCELERATION",
        "evidence": "Sharp throttle stomp to merge into fast lane",
    }
    create_resp = client.post("/api/sensor/annotations", json=annot_payload)
    assert create_resp.status_code == 200
    res_data = create_resp.json()
    assert res_data["annotation_id"].startswith("ann_")
    assert res_data["label"] == "SUDDEN_ACCELERATION"

    # Retrieve annotations for session
    get_resp = client.get(f"/api/sensor/annotations/{sess_id}")
    assert get_resp.status_code == 200
    annots = get_resp.json()
    assert len(annots) >= 1
    assert annots[0]["label"] == "SUDDEN_ACCELERATION"


# =====================================================================
# 7. SECURITY, PATH TRAVERSAL & LOCKBOX INTEGRITY TESTS
# =====================================================================

def test_path_traversal_session_id_rejection():
    """Verify that path traversal in session_id raises PermissionError / 403."""
    storage = SensorStorageService.get_instance()

    with pytest.raises(PermissionError):
        storage.validate_session_id("../../malicious_escape")

    with pytest.raises(PermissionError):
        storage.validate_session_id("sess_normal/../escape")

    # Special characters check
    with pytest.raises(ValueError):
        storage.validate_session_id("sess_invalid!@#$")


def test_lockbox_isolation_on_quality_endpoint():
    """Verify that lockbox drivers DRV_007/DRV_008 cannot be inspected via quality endpoints."""
    resp = client.get("/api/sensor/sessions/DRV_007/sess_secret/quality")
    assert resp.status_code == 403
    assert "Lockbox" in resp.json()["detail"] or "LOCKBOX" in resp.json()["detail"]

    resp2 = client.get("/api/sensor/sessions/DRV_008/sess_secret/quality")
    assert resp2.status_code == 403


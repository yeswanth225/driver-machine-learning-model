# Live Monitoring & Physical Sensor API Reference

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Backend REST Endpoints & WebSocket Telemetry Protocol  
**Document ID:** `IDAMS-API-SENSOR-01`  
**Status:** Live & Implemented  
**Date:** November 2026  

---

## 1. Overview & Architecture

The Physical Sensor Integration suite exposes a unified set of REST endpoints and WebSocket channels under the `/api/sensor` route prefix. These interfaces handle device registration, real-time 20 Hz frame ingestion, bulk CSV uploads with cryptographic validation, automated quality reporting, and ground-truth observer tagging.

---

## 2. Authentication & Governance Headers

- **Development Pool:** `DRV_001` through `DRV_006` are accepted for all registration, recording, and upload endpoints.
- **Blind Lockbox Holdout:** Any request specifying `DRV_007` or `DRV_008` is immediately intercepted and rejected with `403 Forbidden`.

---

## 3. Detailed Endpoint Reference

### 3.1 Device Registration
`POST /api/sensor/device/register`

Registers an Android or iOS smartphone device for physical sensor acquisition.

**Request Body (`application/json`):**
```json
{
  "device_name": "Google Pixel 8",
  "os_version": "Android 14",
  "app_version": "1.0.0",
  "assigned_driver_id": "DRV_001",
  "hardware_capabilities": {
    "has_accelerometer": true,
    "has_gyroscope": true,
    "has_gps": true
  }
}
```

**Success Response (`200 OK`):**
```json
{
  "device_token": "dev_1700000000_DRV_001",
  "assigned_driver_id": "DRV_001",
  "status": "REGISTERED",
  "target_frequency_hz": 20.0,
  "message": "Smartphone registered successfully for physical telemetry acquisition."
}
```

**Error Codes:**
- `400 Bad Request`: Invalid driver ID format (e.g. `DRV_099`).
- `403 Forbidden`: Attempting to register under sequestered lockbox drivers (`DRV_007`, `DRV_008`).

---

### 3.2 Session Initialization
`POST /api/sensor/session/start`

Initializes a new physical recording session and links incoming telemetry frames to a specific vehicle and route.

**Request Body:**
```json
{
  "device_token": "dev_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "vehicle_id": "VEH_CIVIC_2022",
  "initial_pitch_deg": 12.4,
  "initial_roll_deg": -1.2,
  "notes": "Arterial Highway S1 collection run"
}
```

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "vehicle_id": "VEH_CIVIC_2022",
  "status": "ACTIVE",
  "start_epoch_ms": 1700000000000,
  "message": "Session active. Execute S0 30-second static calibration."
}
```

---

### 3.3 Real-Time Frame Ingestion
`POST /api/sensor/telemetry/frame`

Ingests a single 20 Hz frame from the mobile logger and broadcasts it across active WebSockets.

**Request Body:**
```json
{
  "timestamp": 1.250,
  "timestamp_epoch_ms": 1700000001250,
  "Acc X": 0.042,
  "Acc Y": -0.015,
  "Acc Z": -0.985,
  "gyro_x": 0.0012,
  "gyro_y": -0.0008,
  "gyro_z": 0.0005,
  "speed_mps": 18.2,
  "heading_deg": 88.5,
  "latitude": 37.774929,
  "longitude": -122.419416,
  "altitude": 48.2,
  "hdop": 1.1,
  "segment_id": "S1"
}
```

**Success Response (`200 OK`):**
```json
{
  "status": "ACK",
  "timestamp": 1.250
}
```

---

### 3.4 Raw Recording Bulk Upload & Quality Audit
`POST /api/sensor/upload`

Uploads a completed raw `.csv` telemetry file, verifies client SHA-256 integrity, detects duplicates, executes the 6-criterion automated quality evaluation, and stores files immutably.

**Content-Type:** `multipart/form-data`
- `driver_id` (Text Form Field): e.g. `DRV_001`
- `session_id` (Text Form Field): e.g. `sess_1700000000_DRV_001`
- `client_sha256` (Text Form Field, Optional): SHA-256 digest computed on mobile client
- `file` (Binary File): CSV file upload

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "file_name": "raw_telemetry.csv",
  "file_size_bytes": 1420850,
  "sample_count": 28400,
  "duration_seconds": 1420.0,
  "sha256": "3a7b1c4d9e0f11a2b3c4...",
  "data_quality_status": "PASS",
  "is_duplicate": false,
  "message": "Raw telemetry recording stored and validated successfully."
}
```

**Duplicate Detection Response (`200 OK`):**
```json
{
  "session_id": "sess_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "file_size_bytes": 1420850,
  "sample_count": 0,
  "duration_seconds": 0.0,
  "sha256": "3a7b1c4d9e0f11a2b3c4...",
  "data_quality_status": "DUPLICATE_REJECTED",
  "is_duplicate": true,
  "message": "Duplicate recording detected. Identical telemetry already stored in session 'sess_1700000000_DRV_001'."
}
```

---

### 3.5 Live Hardware & Stream Status
`GET /api/sensor/live/status`

Returns dynamic hardware connection state, measured frequency, and latest frame.

**Success Response (`200 OK`):**
```json
{
  "connected": true,
  "status": "STREAMING",
  "device_token": "dev_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "session_id": "sess_1700000000_DRV_001",
  "actual_frequency_hz": 19.98,
  "latest_frame": {
    "timestamp": 14.50,
    "Acc X": 0.02,
    "Acc Y": -0.01,
    "Acc Z": -0.98,
    "gyro_z": 0.002
  },
  "last_heartbeat_epoch_ms": 1700000014500,
  "quality_status": "ACTIVE",
  "message": "Physical telemetry streaming active at 20.0 Hz."
}
```

---

### 3.6 Observer Ground-Truth Annotation
`POST /api/sensor/annotations`

Records an independent human ground-truth annotation marking an observed maneuver time window.

**Request Body:**
```json
{
  "session_id": "sess_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "annotator_id": "OBS_01",
  "start_timestamp": 45.2,
  "end_timestamp": 47.8,
  "label": "SUDDEN_DECELERATION",
  "evidence": "Rapid braking to avoid animal entering roadway",
  "verification_status": "VERIFIED"
}
```

**Success Response (`200 OK`):**
```json
{
  "annotation_id": "ann_1700000045_1",
  "session_id": "sess_1700000000_DRV_001",
  "driver_id": "DRV_001",
  "start_timestamp": 45.2,
  "end_timestamp": 47.8,
  "label": "SUDDEN_DECELERATION",
  "annotator_id": "OBS_01",
  "evidence": "Rapid braking to avoid animal entering roadway",
  "verification_status": "VERIFIED",
  "created_at": "2026-11-24T18:30:45Z"
}
```

---

### 3.7 WebSocket Streaming Channel
`WebSocket /api/sensor/live/stream`

Full-duplex real-time streaming channel transmitting 20 Hz telemetry frames and frequency heartbeats to connected dashboards.

**Sample WebSocket Frame Payload:**
```json
{
  "type": "TELEMETRY_FRAME",
  "data": {
    "timestamp": 22.45,
    "timestamp_epoch_ms": 1700000022450,
    "Acc X": 0.05,
    "Acc Y": -0.02,
    "Acc Z": -0.99,
    "gyro_x": 0.001,
    "gyro_y": 0.002,
    "gyro_z": 0.001,
    "speed_mps": 24.1,
    "segment_id": "S2"
  },
  "frequency_hz": 20.0
}
```

---
*Maintained by the IDAMS Backend & Telemetry Engineering Group.*

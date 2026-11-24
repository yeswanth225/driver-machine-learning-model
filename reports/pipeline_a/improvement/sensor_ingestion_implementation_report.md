# Sensor Ingestion & Acquisition Implementation Report

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Ingestion Pipeline, Native Android App & Web PWA Logger  
**Document ID:** `IDAMS-IMP-INGEST-01`  
**Status:** Complete & Verified  
**Date:** November 2026  

---

## 1. Overview

This report details the technical implementation of the Smartphone Acquisition and Ingestion subsystem developed for physical vehicle telemetry acquisition.

To accommodate heterogeneous field hardware and testing environments, two distinct client loggers were engineered:
1. A **Native Android Studio Application** (`mobile/android/`) designed for sustained, high-reliability background logging with Android Foreground Services and wake locks.
2. A **Cross-Platform Mobile Web PWA Logger** (`mobile/web-logger/`) deployable directly over HTTPS with zero native compilation overhead.

Both clients feed into a unified, cryptographically verified FastAPI backend ingestion service.

---

## 2. Mobile Acquisition Clients

### 2.1 Native Android Application Architecture
- **Location:** `mobile/android/`
- **Minimum SDK:** Android API Level 26 (Android 8.0 Oreo)  
- **Target SDK:** Android API Level 34 (Android 14 UpsideDownCake)  
- **Language:** Kotlin 1.9.22  
- **Build System:** Gradle Kotlin DSL (`build.gradle.kts`)

#### Key Components:
1. **`SensorLoggingService.kt` (Foreground Service):**
   - Utilizes `android.app.Service` with `FOREGROUND_SERVICE_TYPE_LOCATION` and `FOREGROUND_SERVICE_TYPE_HEALTH` to prevent Android Doze mode termination.
   - Acquires `PARTIAL_WAKE_LOCK` and registers sensor listeners:
     - `Sensor.TYPE_ACCELEROMETER` at `SENSOR_DELAY_FASTEST` ($\approx 20\text{ ms}$).
     - `Sensor.TYPE_GYROSCOPE` at `SENSOR_DELAY_FASTEST`.
     - `LocationManager.GPS_PROVIDER` for high-precision GNSS position, speed, and bearing.
   - Implements a monotonic clock synchronizer using `SystemClock.elapsedRealtimeNanos()` to eliminate clock drift from NTP updates.
   - Writes incrementally to an atomic buffered CSV stream on internal storage:
     `context.filesDir/sessions/<session_id>/raw_telemetry.csv`.

2. **`MainActivity.kt` & HUD Interface:**
   - Real-time mobile HUD displaying rolling sampling rate ($\text{Hz}$), elapsed recording duration, current vehicle speed, and active driving regime selector (`S0` to `S6`).
   - One-tap regime transition buttons ensuring explicit ground-truth phase demarcation.

3. **`DataUploader.kt` (HTTP Ingestion Client):**
   - Built on `OkHttp3` with streaming `MultipartBody` upload.
   - Computes local SHA-256 digest of the finalized CSV file prior to transmission.
   - Sends client digest in the upload payload for end-to-end cryptographic integrity verification.

---

### 2.2 Cross-Platform Mobile Web Logger (PWA)
- **Location:** `mobile/web-logger/`
- **Hosted At:** Mounted statically at `/sensor-logger/` on the FastAPI backend.
- **Tech Stack:** HTML5, Modern Vanilla ES6 JavaScript, Web Manifest, Web Crypto API.

#### Key Features:
1. **Zero-Installation Field Deployment:** Field drivers can open the logger in Chrome for Android or Mobile Safari by navigating to the hosted port or scanning the dashboard QR code.
2. **Standardized Sensor Ingestion:**
   - Subscribes to `window.addEventListener('devicemotion')` and extracts `accelerationIncludingGravity` and `rotationRate`.
   - Subscribes to `navigator.geolocation.watchPosition` with `enableHighAccuracy: true`.
3. **Internal Fixed-Rate 20 Hz Resampling Timer:**
   - Runs a 50 ms monotonic tick loop (`performance.now()`) buffering instantaneous inertial states.
   - Interpolates sensor readings across micro-jitter intervals to produce clean 20 Hz intervals.
4. **Client-Side Cryptography & Export:**
   - In-memory CSV generation formatted to the standard 15-column schema.
   - SHA-256 hash generation via `crypto.subtle.digest('SHA-256', ...)`.
   - Direct HTTP POST to `/api/sensor/upload` or instant local `.csv` file download.

---

## 3. Backend Ingestion & Storage Architecture

### 3.1 `SensorStorageService` (`backend/app/services/sensor_storage_service.py`)
The storage service coordinates raw telemetry persistence, integrity checking, and governance:

1. **Development Driver Pool Segregation:**
   - Enforces driver ID validation:
     $$\mathcal{D}_{\text{dev}} = \{\text{DRV\_001}, \text{DRV\_002}, \text{DRV\_003}, \text{DRV\_004}, \text{DRV\_005}, \text{DRV\_006}\}$$
   - Any upload matching $\mathcal{D}_{\text{lockbox}} = \{\text{DRV\_007}, \text{DRV\_008}\}$ is rejected immediately with HTTP 403 Forbidden.

2. **Cryptographic Checksum Verification & Deduplication:**
   - The backend recalculates `hashlib.sha256(csv_bytes).hexdigest()`.
   - If `client_sha256` was provided, the service verifies bit-level match before storage.
   - Global session manifests are scanned for identical hashes; duplicate re-uploads return `is_duplicate: True` with status `DUPLICATE_REJECTED`, preventing redundant evaluation or disk bloat.

3. **Storage Layout & Manifests:**
   Every accepted recording creates an isolated session directory:
   ```
   dataset/raw/eight_driver_campaign/<driver_id>/<session_id>/
   ├── raw_telemetry.csv
   ├── metadata.json
   ├── data_quality_report.json
   └── data_quality_report.md
   ```

   **Sample `metadata.json`:**
   ```json
   {
     "session_id": "sess_1700000000_DRV_001",
     "driver_id": "DRV_001",
     "file_name": "raw_telemetry.csv",
     "file_size_bytes": 1048576,
     "sha256": "3a7b1c4d9e0f...",
     "sample_count": 24000,
     "duration_seconds": 1200.0,
     "nominal_frequency_hz": 20.0,
     "ingested_at": "2026-11-24T18:00:00Z"
   }
   ```

---

## 4. Integration with Quality Validation

Immediately upon writing the raw CSV file to disk, the ingestion handler invokes `DataQualityService.evaluate_session_df()`:
1. Loads the stored CSV into a strictly validated pandas DataFrame.
2. Runs all 6 quality acceptance criteria.
3. Automatically persists `data_quality_report.json` and human-readable `data_quality_report.md` alongside the raw data.
4. Returns the overall quality status (`PASS`, `WARNING`, or `FAIL`) in the HTTP upload response.

---

## 5. Verification & Compliance Summary

| Test Case | Description | Result |
|:---|:---|:---|
| `test_duplicate_upload_rejection` | Identical CSV uploaded twice yields `is_duplicate=True` | **PASSED** |
| `test_lockbox_isolation_rules` | Attempts to access DRV_007/DRV_008 raise `PermissionError` | **PASSED** |
| `test_lockbox_rejection_via_endpoints` | API returns HTTP 403 Forbidden on lockbox tokens | **PASSED** |
| `test_quality_service_healthy_session` | 20 Hz recording meets all criteria | **PASSED** |

---
*Certified by the IDAMS Systems Engineering Group.*

# Physical Sensor Data Collection Field Guide & Protocol

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Eight-Driver Data Collection Campaign Field Operations  
**Document ID:** `IDAMS-SOP-COLLECTION-01`  
**Status:** Approved Standard Operating Procedure  
**Date:** November 2026  

---

## 1. Objectives & Safety Invariants

This guide establishes the mandatory Standard Operating Procedure (SOP) for field researchers, drivers, and in-car observers executing the physical driving data collection campaign.

### Critical Safety Invariants:
1. **Safety First:** Data collection must NEVER compromise vehicular safety or violate local traffic laws.
2. **Dedicated Roles:** Every collection vehicle MUST have two occupants:
   - A licensed **Primary Driver** focused entirely on vehicle operation.
   - An **In-Car Observer** seated in the front passenger seat responsible for operating the logger software, monitoring sampling frequency, and recording real-time ground-truth maneuver timestamps.
3. **Aggressive Maneuver Execution:** Aggressive driving maneuvers (regime `S5`) must ONLY be executed on designated open, low-traffic arterial roads or closed proving grounds with clear lines of sight.

---

## 2. Equipment Setup & Smartphone Rigging

### 2.1 Hardware Requirements
- **Primary Logger Device:** Modern smartphone (Google Pixel 7/8, Samsung Galaxy S22/S23, or iPhone 13/14/15) running Android 12+ or iOS 16+.
- **Rigid Phone Mount:** Heavy-duty mechanical clamping mount with suction cup or bolted bracket.
  - *Prohibited:* Magnetic mounts, air-vent clip mounts, or flexible gooseneck holders (prone to vibration induced angular slippage).
- **Power Supply:** 12V automotive USB-PD fast charger (minimum 18W) with reinforced USB-C/Lightning cable.
- **Vehicle Selection:** Development campaign uses three diverse vehicle form factors:
  - `VEH_01`: Compact Hatchback (e.g., Honda Jazz / VW Polo)
  - `VEH_02`: Mid-Size Sedan (e.g., Honda Civic / Toyota Corolla)
  - `VEH_03`: Compact SUV (e.g., Hyundai Creta / Kia Seltos)

### 2.2 Mount Alignment & Sensor Calibration
1. Attach mount to the center of the windshield below the rearview mirror or rigid dashboard top.
2. Fasten the smartphone in **Landscape Orientation**:
   - Screen facing the rear of the vehicle (toward the driver/observer).
   - Phone body parallel to the vehicle's transverse axis.
3. Tighten all adjustment thumbscrews firmly.
4. Verify mechanical rigidity: lightly tap the phone edge; angular oscillation must settle within $< 100\text{ ms}$.

---

## 3. Seven-Stage Drive Protocol (S0 through S6)

Each recording session must follow the chronological 7-stage sequence:

| Phase | Regime Description | Min. Duration | Target Speed Range | Mandatory Driving Characteristics |
|:---|:---|:---:|:---:|:---|
| **S0** | **Pre-Drive Static Calibration** | 30 seconds | $0\text{ km/h}$ (Stationary) | Engine running at idle; vehicle parked on level ground; zero occupant movement. Calibrates local gravity vector ($1.0g$). |
| **S1** | **City Stop-and-Go Driving** | 3.0 minutes | $0 - 45\text{ km/h}$ | Normal urban traffic, traffic light stops, pedestrian crossings, smooth gentle throttle and braking. |
| **S2** | **Normal Arterial Road Driving** | 3.0 minutes | $40 - 70\text{ km/h}$ | Multi-lane arterial road, moderate traffic flow, gradual lane changes with turn signals, gentle curves. |
| **S3** | **Smooth Highway Cruising** | 3.0 minutes | $80 - 100\text{ km/h}$ | Continuous highway speed, steady throttle, minimal lateral acceleration ($< 0.15g$). Focus of reverse-session false positive audit. |
| **S4** | **Rough-Road & Vibration Normal** | 2.0 minutes | $20 - 50\text{ km/h}$ | Normal driving over speed humps, bridge expansion joints, concrete seams, or textured asphalt. Establishes vibration baseline. |
| **S5** | **Controlled Aggressive Maneuvers** | 2.5 minutes | Varied | Sudden throttle stomps ($a_x > 0.35g$), firm emergency braking ($a_x < -0.55g$), rapid slalom lane changes ($a_y > 0.35g$, $\omega_z > 0.30\text{ rad/s}$). |
| **S6** | **Post-Drive Static Verification** | 30 seconds | $0\text{ km/h}$ (Stationary) | Vehicle parked on level ground; verifies phone mount did not slip ($> 3.0^\circ$ drift triggers session rejection). |

---

## 4. Ingestion & Quality Acceptance Workflow

### Step 1: Ending the Drive & Generating Local Checksum
1. Upon completing stage `S6`, tap **"Stop Recording"** on the mobile logger.
2. The logger finalizes the internal CSV stream, closes file handles, and automatically computes the client SHA-256 digest:
   ```
   SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   ```

### Step 2: Uploading Telemetry to Ingestion Gateway
1. Ensure the logger smartphone is connected to vehicle Wi-Fi or cellular data.
2. Enter the assigned driver token (e.g., `DRV_001` through `DRV_006`) and unique session ID (e.g., `sess_20261125_drv001_s1`).
3. Tap **"Upload to Backend"** or upload via the web dashboard at `/sensor-logger/`.

### Step 3: Automated Quality Gate Evaluation
The backend immediately executes `DataQualityService.evaluate_session_df()` evaluating the 6 acceptance criteria:
1. **Sampling Frequency:** Mean frequency $19.0 - 21.0\text{ Hz}$, zero gaps $\ge 250\text{ ms}$, strictly monotonic.
2. **Dynamic Range:** Zero clipped accelerometer samples ($|a| < 3.92g$), zero dead channels.
3. **Static Calibration:** Initial static gravity $|g_0| \in [0.98, 1.02]g$, post-drive mount drift $\le 3.0^\circ$.
4. **GPS Consistency:** Zero negative speeds, valid satellite geometry (mean $\text{HDOP} \le 2.5$).
5. **Regime Coverage:** Sufficient duration across all regimes (`S0` through `S6`).
6. **Data Hygiene:** Exactly zero NaNs, nulls, or infinite values.

### Step 4: Session Acceptance or Re-Drive Decision
- **Status `PASS`:** Session is accepted and archived immutably in `dataset/raw/eight_driver_campaign/`.
- **Status `WARNING`:** Flagged for manual review by the lead research engineer.
- **Status `FAIL`:** Session is automatically flagged as rejected. The field crew must inspect mount tightness or sensor settings and re-execute the driving session.

---
*Authorized by the IDAMS Field Operations & Safety Compliance Directorate.*

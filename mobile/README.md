# Mobile Smartphone Sensor Logger Applications

This directory contains two acquisition tools for capturing physical driving telemetry at **20.0 Hz**:

1. **Native Android Application (`mobile/android/`) [PREFERRED CANONICAL PATH]**:  
   Full Kotlin / Android Studio Gradle project with an Android Foreground Service (`SensorLoggingService.kt`) and partial wake lock. Designed specifically for multi-minute, sustained vehicular data collection without background timer throttling.
2. **Mobile Web / PWA Sensor Logger (`mobile/web-logger/`) [SUPPLEMENTARY DEMONSTRATION]**:  
   Lightweight, zero-install Web / PWA application running on Chrome for Android or Safari. Features real-time frequency estimation, GPS speed integration, local CSV storage, and direct HTTP upload to the FastAPI backend.  
   > [!WARNING]  
   > **Mobile Browser Throttling Advisory:** Mobile web browsers aggressively throttle or suspend JavaScript timers (`setInterval`) when the screen turns off or the tab loses focus. Therefore, the PWA is intended for rapid demonstration, laboratory testing, and validation; the **Native Android Application** is the designated canonical acquisition tool for the official 8-driver scientific campaign.

---

## Telemetry Schema (Canonical Acceptance Contract)

Both applications output CSV files conforming strictly to the following 15-column schema:

| Column | Unit | Description |
|---|---|---|
| `timestamp` | seconds (float64) | Relative time elapsed since session start |
| `timestamp_epoch_ms` | milliseconds (int64) | UTC epoch milliseconds from GPS/Network clock |
| `Acc X` | g (float) | Lateral body acceleration ($1.0\text{ g} = 9.80665\text{ m/s}^2$) |
| `Acc Y` | g (float) | Longitudinal vehicle acceleration |
| `Acc Z` | g (float) | Vertical acceleration (nominal $-1.0\text{ g}$ upright) |
| `gyro_x` | rad/s (float) | Pitch angular velocity |
| `gyro_y` | rad/s (float) | Roll angular velocity |
| `gyro_z` | rad/s (float) | Yaw angular velocity (vehicle turning rate) |
| `speed_mps` | m/s (float) | GPS ground speed ($1\text{ m/s} = 3.6\text{ km/h}$) |
| `heading_deg` | degrees (float) | GPS bearing ($0^\circ - 360^\circ$) |
| `latitude` | degrees (float) | GPS Latitude (WGS 84) |
| `longitude` | degrees (float) | GPS Longitude (WGS 84) |
| `altitude` | meters (float) | GPS Elevation |
| `hdop` | unitless (float) | Horizontal Dilution of Precision |
| `segment_id` | string | Active regime tag (`S0` through `S6`) |

---

## Option 1: Running the Mobile Web / PWA Logger (Immediate Setup)

1. **Serve the Web Logger:**
   The backend automatically serves this web logger at `http://<your-pc-ip>:8000/sensor-logger/` or you can open `mobile/web-logger/index.html` on your mobile browser.
2. **Open on Smartphone:**  
   Connect your smartphone to the same Wi-Fi network as your PC and open `http://<pc-ip>:8000/sensor-logger/` in Google Chrome.
3. **Mount Smartphone in Vehicle:**  
   Place the phone in the rigid windshield suction clamp.
4. **Select Driver and Regime:**  
   Choose your assigned token (e.g. `DRV_001`) and start with `S0 - Pre-Drive 30s Static Calibration`.
5. **Tap START:**  
   The HUD displays real-time Hz (nominal 20 Hz), duration, samples logged, and GPS speed.
6. **Upload or Download:**  
   Upon stopping, tap **"UPLOAD TO BACKEND"** to transmit directly to `/api/sensor/upload`, or tap **"DOWNLOAD CSV"** to save locally.

---

## Option 2: Building the Native Android APK

1. **Open in Android Studio:**  
   Open the `mobile/android` directory in Android Studio (Giraffe or newer).
2. **Sync Gradle:**  
   Allow Gradle to download Android SDK 34 and dependencies (`okhttp`, `kotlinx-coroutines`).
3. **Build APK:**  
   Run `Build > Build Bundle(s) / APK(s) > Build APK(s)`.
4. **Install on Phone:**  
   Transfer `app-debug.apk` to your Android device via USB debugging or direct download:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```
5. **Grant Permissions:**  
   Grant Location and Body Sensors permissions when prompted.

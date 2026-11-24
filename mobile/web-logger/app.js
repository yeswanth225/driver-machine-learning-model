/**
 * Driver Sensor Logger — Mobile Web / PWA Core Engine
 * Captures 3-Axis Accel, 3-Axis Gyro, and GPS at 20.0 Hz nominal rate.
 * Writes strictly monotonic timestamps matching the project acceptance schema.
 */

(function () {
  'use strict';

  // State Variables
  let isRecording = false;
  let recordingStartTime = 0;
  let sampleCount = 0;
  let samplesBuffer = [];
  let samplingTimer = null;
  let currentSessionId = '';
  let lastSampleTime = 0;
  let measuredFrequency = 0.0;
  let freqWindow = [];

  // Live sensor cache
  let currentAcc = { x: 0.0, y: 0.0, z: -0.98 }; // in g
  let currentGyro = { x: 0.0, y: 0.0, z: 0.0 };  // in rad/s
  let currentGps = { lat: 0.0, lon: 0.0, speedMps: 0.0, headingDeg: 0.0, alt: 0.0, hdop: 1.0, fix: false };

  // DOM Elements
  const statusBadge = document.getElementById('statusBadge');
  const freqVal = document.getElementById('freqVal');
  const durationVal = document.getElementById('durationVal');
  const samplesVal = document.getElementById('samplesVal');
  const speedVal = document.getElementById('speedVal');
  const driverSelect = document.getElementById('driverSelect');
  const segmentSelect = document.getElementById('segmentSelect');
  const backendUrl = document.getElementById('backendUrl');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const btnUpload = document.getElementById('btnUpload');
  const btnExport = document.getElementById('btnExport');
  const liveSensors = document.getElementById('liveSensors');
  const logBox = document.getElementById('logBox');

  function log(msg) {
    const time = new Date().toLocaleTimeString();
    logBox.innerHTML = `[${time}] ${msg}<br>` + logBox.innerHTML;
  }

  // 1. Initialize Motion Sensors
  function initSensors() {
    if (window.DeviceMotionEvent) {
      window.addEventListener('devicemotion', (event) => {
        const acc = event.accelerationIncludingGravity || event.acceleration;
        if (acc && acc.x !== null) {
          // Convert m/s^2 to g (1 g = 9.80665 m/s^2)
          currentAcc.x = acc.x / 9.80665;
          currentAcc.y = acc.y / 9.80665;
          currentAcc.z = acc.z / 9.80665;
        }
        const rot = event.rotationRate;
        if (rot && rot.alpha !== null) {
          // Convert deg/s to rad/s
          currentGyro.x = (rot.beta || 0) * (Math.PI / 180.0);
          currentGyro.y = (rot.gamma || 0) * (Math.PI / 180.0);
          currentGyro.z = (rot.alpha || 0) * (Math.PI / 180.0);
        }
      }, true);
      log('Hardware IMU listener registered (DeviceMotionEvent).');
    } else {
      log('WARNING: DeviceMotionEvent not supported on this browser.');
    }

    // 2. Initialize Geolocation
    if ('geolocation' in navigator) {
      navigator.geolocation.watchPosition(
        (pos) => {
          currentGps.lat = pos.coords.latitude;
          currentGps.lon = pos.coords.longitude;
          currentGps.speedMps = pos.coords.speed !== null && pos.coords.speed >= 0 ? pos.coords.speed : 0.0;
          currentGps.headingDeg = pos.coords.heading !== null && !isNaN(pos.coords.heading) ? pos.coords.heading : 0.0;
          currentGps.alt = pos.coords.altitude || 0.0;
          currentGps.hdop = pos.coords.accuracy ? Math.max(1.0, pos.coords.accuracy / 5.0) : 1.0;
          currentGps.fix = true;

          speedVal.textContent = `${(currentGps.speedMps * 3.6).toFixed(1)} km/h`;
        },
        (err) => {
          log(`GPS notice: ${err.message}`);
        },
        { enableHighAccuracy: true, maximumAge: 1000, timeout: 5000 }
      );
      log('GPS listener registered (High Accuracy).');
    }
  }

  // Request iOS permission if needed
  async function requestPermissionsIfNeeded() {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const response = await DeviceMotionEvent.requestPermission();
        log(`Motion sensor permission: ${response}`);
      } catch (err) {
        log(`Permission error: ${err.message}`);
      }
    }
  }

  // 3. Start Recording
  async function startRecording() {
    await requestPermissionsIfNeeded();

    isRecording = true;
    recordingStartTime = performance.now();
    sampleCount = 0;
    samplesBuffer = [];
    currentSessionId = `sess_${Date.now()}`;
    lastSampleTime = performance.now();
    freqWindow = [];

    statusBadge.textContent = 'RECORDING 20Hz';
    statusBadge.className = 'status-badge status-active';

    btnStart.disabled = true;
    btnStart.className = 'btn-start btn-disabled';
    btnStop.disabled = false;
    btnStop.className = 'btn-stop';
    btnUpload.disabled = true;
    btnUpload.className = 'btn-upload btn-disabled';
    btnExport.disabled = true;
    btnExport.className = 'btn-export btn-disabled';
    driverSelect.disabled = true;

    log(`Started 20 Hz recording for ${driverSelect.value} (Session: ${currentSessionId}).`);

    // Target 20.0 Hz = exactly 50 ms loop
    samplingTimer = setInterval(sampleStep, 50);
  }

  // 4. Sample Step (20 Hz loop)
  function sampleStep() {
    if (!isRecording) return;

    const nowPerf = performance.now();
    const elapsedSec = (nowPerf - recordingStartTime) / 1000.0;
    const epochMs = Date.now();

    // Frequency measurement
    const dt = (nowPerf - lastSampleTime) / 1000.0;
    lastSampleTime = nowPerf;
    if (dt > 0.001) {
      freqWindow.push(1.0 / dt);
      if (freqWindow.length > 20) freqWindow.shift();
      measuredFrequency = freqWindow.reduce((a, b) => a + b, 0) / freqWindow.length;
    }

    const segment = segmentSelect.value;

    // Build canonical CSV line matching schema:
    // timestamp,timestamp_epoch_ms,Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z,speed_mps,heading_deg,latitude,longitude,altitude,hdop,segment_id
    const row = [
      elapsedSec.toFixed(4),
      epochMs,
      currentAcc.x.toFixed(6),
      currentAcc.y.toFixed(6),
      currentAcc.z.toFixed(6),
      currentGyro.x.toFixed(6),
      currentGyro.y.toFixed(6),
      currentGyro.z.toFixed(6),
      currentGps.speedMps.toFixed(2),
      currentGps.headingDeg.toFixed(1),
      currentGps.lat.toFixed(6),
      currentGps.lon.toFixed(6),
      currentGps.alt.toFixed(1),
      currentGps.hdop.toFixed(2),
      segment
    ];

    samplesBuffer.push(row.join(','));
    sampleCount++;

    // Update HUD every 10 samples (2 Hz display refresh)
    if (sampleCount % 10 === 0) {
      freqVal.textContent = `${measuredFrequency.toFixed(1)} Hz`;
      samplesVal.textContent = sampleCount.toLocaleString();

      const mins = Math.floor(elapsedSec / 60);
      const secs = Math.floor(elapsedSec % 60);
      durationVal.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

      liveSensors.innerHTML = `
        Acc (X,Y,Z): [ ${currentAcc.x.toFixed(3)}, ${currentAcc.y.toFixed(3)}, ${currentAcc.z.toFixed(3)} ] g<br>
        Gyro (X,Y,Z): [ ${currentGyro.x.toFixed(3)}, ${currentGyro.y.toFixed(3)}, ${currentGyro.z.toFixed(3)} ] rad/s<br>
        GPS: Lat ${currentGps.lat.toFixed(4)}, Lon ${currentGps.lon.toFixed(4)}, HDOP: ${currentGps.hdop.toFixed(1)}
      `;
    }
  }

  // 5. Stop Recording
  function stopRecording() {
    isRecording = false;
    if (samplingTimer) clearInterval(samplingTimer);

    statusBadge.textContent = 'STOPPED';
    statusBadge.className = 'status-badge status-standby';

    btnStart.disabled = false;
    btnStart.className = 'btn-start';
    btnStop.disabled = true;
    btnStop.className = 'btn-stop btn-disabled';
    btnUpload.disabled = false;
    btnUpload.className = 'btn-upload';
    btnExport.disabled = false;
    btnExport.className = 'btn-export';
    driverSelect.disabled = false;

    log(`Recording stopped. Total samples: ${sampleCount} (${samplesBuffer.length} rows buffered).`);
  }

  // 6. Generate CSV Content
  function generateCsvContent() {
    const header = 'timestamp,timestamp_epoch_ms,Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z,speed_mps,heading_deg,latitude,longitude,altitude,hdop,segment_id\n';
    return header + samplesBuffer.join('\n') + '\n';
  }

  // 7. Download CSV locally
  function exportCsv() {
    const csvContent = generateCsvContent();
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `raw_telemetry_${driverSelect.value}_${currentSessionId}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    log(`CSV downloaded: raw_telemetry_${driverSelect.value}_${currentSessionId}.csv`);
  }

  // 8. Upload Directly to Backend
  async function uploadToBackend() {
    const base = backendUrl.value.trim().replace(/\/$/, '');
    const uploadUrl = `${base}/api/sensor/upload`;
    log(`Uploading to ${uploadUrl}...`);

    btnUpload.disabled = true;
    btnUpload.textContent = 'UPLOADING...';

    const csvContent = generateCsvContent();
    const blob = new Blob([csvContent], { type: 'text/csv' });

    // Compute SHA-256 in browser
    const hashBuffer = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(csvContent));
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const sha256 = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

    const formData = new FormData();
    formData.append('driver_id', driverSelect.value);
    formData.append('session_id', currentSessionId);
    formData.append('client_sha256', sha256);
    formData.append('file', blob, `raw_telemetry_${currentSessionId}.csv`);

    try {
      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const resJson = await response.json();
      log(`Upload successful! Server SHA-256: ${resJson.sha256.substring(0, 16)}...`);
      log(`Quality Status: ${resJson.data_quality_status || 'ACCEPTED'}`);
      alert('Upload Successful! Verified by Backend.');
    } catch (err) {
      log(`Upload failed: ${err.message}`);
      alert(`Upload Failed: ${err.message}`);
    } finally {
      btnUpload.disabled = false;
      btnUpload.textContent = 'UPLOAD TO BACKEND';
    }
  }

  // Event Listeners
  btnStart.addEventListener('click', startRecording);
  btnStop.addEventListener('click', stopRecording);
  btnExport.addEventListener('click', exportCsv);
  btnUpload.addEventListener('click', uploadToBackend);

  // Initialize
  initSensors();
})();

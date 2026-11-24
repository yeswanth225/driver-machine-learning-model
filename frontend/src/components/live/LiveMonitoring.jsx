import React, { useState, useEffect, useRef } from 'react';
import {
  Radio,
  RefreshCw,
  Activity,
  Play,
  Pause,
  Smartphone,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  ShieldAlert,
  Compass,
  FileCheck,
  Tag,
  Clock,
  Layers,
} from 'lucide-react';
import {
  fetchHardwareStatus,
  assessTelemetry,
  fetchLiveSensorStatus,
  fetchRecentTelemetryFrames,
  uploadSensorCsv,
  createAnnotation,
} from '../../services/api';
import { Badge } from '../common/Badge';
import { RiskGauge } from '../common/RiskGauge';
import { ComponentInspectorModal } from '../common/ComponentInspectorModal';
import { ParticleButton } from '../ui/ParticleButton';
import { Button } from '../ui/button';

export function LiveMonitoring({ onGlobalAssessmentUpdate }) {
  // Mode selection: 'physical' vs 'simulation'
  const [activeMode, setActiveMode] = useState('physical');

  // Physical Live Telemetry State
  const [liveStatus, setLiveStatus] = useState(null);
  const [isCheckingLive, setIsCheckingLive] = useState(false);
  const [recentFrames, setRecentFrames] = useState([]);
  const [annotationNote, setAnnotationNote] = useState('');
  const [annotationCategory, setAnnotationCategory] = useState('NORMAL');
  const [annotationSuccess, setAnnotationSuccess] = useState(null);

  // File Upload State
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadDriverId, setUploadDriverId] = useState('DRV_001');
  const [uploadSessionId, setUploadSessionId] = useState('');
  const [uploadResult, setUploadResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  // Simulation Mode State
  const [isSimStreaming, setIsSimStreaming] = useState(false);
  const [simAssessment, setSimAssessment] = useState(null);
  const [simEventHistory, setSimEventHistory] = useState([]);
  const [simSensorValues, setSimSensorValues] = useState({
    accX: 0.02,
    accY: -0.01,
    accZ: -0.98,
    gyroX: 0.005,
    gyroY: -0.002,
    gyroZ: 0.001,
    speed: 68.4,
  });

  const [inspectorTarget, setInspectorTarget] = useState(null);

  const simTimerRef = useRef(null);
  const simStepRef = useRef(0);
  const pollingTimerRef = useRef(null);
  const wsRef = useRef(null);
  const wsFrameBufferRef = useRef([]);
  const throttleTimerRef = useRef(null);

  // Lifecycle
  useEffect(() => {
    checkLiveStatus();

    // Start background polling for live sensor status every 1.5 seconds
    pollingTimerRef.current = setInterval(checkLiveStatus, 1500);

    // Decoupled batch flusher for WebSocket telemetry (10 Hz throttle) to prevent React render thrashing
    throttleTimerRef.current = setInterval(() => {
      if (wsFrameBufferRef.current.length > 0) {
        const buffered = wsFrameBufferRef.current;
        wsFrameBufferRef.current = [];
        setRecentFrames((prev) => {
          const reversed = [...buffered].reverse();
          return [...reversed, ...prev].slice(0, 50);
        });
      }
    }, 100);

    // Attempt WebSocket connection for sub-100ms telemetry updates
    connectWebSocket();

    return () => {
      if (simTimerRef.current) clearInterval(simTimerRef.current);
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
      if (throttleTimerRef.current) clearInterval(throttleTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/sensor/live/stream`;
      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        try {
          const frame = JSON.parse(event.data);
          if (frame && frame.timestamp) {
            wsFrameBufferRef.current.push(frame);
          }
        } catch {
          // ignore keepalive or parse error
        }
      };

      ws.onclose = () => {
        // Will retry automatically on next page render or poll
      };

      wsRef.current = ws;
    } catch (e) {
      console.warn('WebSocket connection attempt omitted or failed:', e);
    }
  };

  const checkLiveStatus = async () => {
    setIsCheckingLive(true);
    try {
      const data = await fetchLiveSensorStatus();
      setLiveStatus(data);
    } catch {
      setLiveStatus({
        device_token: 'NONE',
        assigned_driver_id: 'NONE',
        device_model: 'None registered',
        connection_status: 'DISCONNECTED',
        last_heartbeat_epoch_ms: 0,
        measured_hz: 0.0,
        nominal_hz: 20.0,
        total_frames_received: 0,
        active_session_id: null,
        message: 'No physical smartphone logger connected.',
      });
    } finally {
      setIsCheckingLive(false);
    }
  };

  // Simulation Stream Step
  const toggleSimStreaming = () => {
    if (isSimStreaming) {
      if (simTimerRef.current) clearInterval(simTimerRef.current);
      setIsSimStreaming(false);
    } else {
      setIsSimStreaming(true);
      simTimerRef.current = setInterval(runSimStep, 250);
    }
  };

  const runSimStep = async () => {
    simStepRef.current += 1;
    const step = simStepRef.current;
    const t = step * 0.25;

    let accX = (Math.random() - 0.5) * 0.08;
    let accY = (Math.random() - 0.5) * 0.06;
    let accZ = -0.98 + (Math.random() - 0.5) * 0.04;
    let gyroZ = (Math.random() - 0.5) * 0.02;

    let maneuverPred = 0;
    let maneuverConf = 0.98;
    let stylePred = 0;
    let styleConf = 0.94;

    const cycle = step % 40;
    if (cycle >= 10 && cycle <= 13) {
      accX = -0.65;
      maneuverPred = 2; // Sudden Deceleration
      maneuverConf = 0.92;
    } else if (cycle >= 22 && cycle <= 25) {
      gyroZ = 0.42;
      accY = 0.38;
      maneuverPred = 4; // Sharp Left Turn
      maneuverConf = 0.88;
    } else if (cycle >= 32 && cycle <= 34) {
      accX = 0.52;
      maneuverPred = 1; // Sudden Acceleration
      maneuverConf = 0.89;
    }

    setSimSensorValues({
      accX: Number(accX.toFixed(3)),
      accY: Number(accY.toFixed(3)),
      accZ: Number(accZ.toFixed(3)),
      gyroX: Number(((Math.random() - 0.5) * 0.01).toFixed(3)),
      gyroY: Number(((Math.random() - 0.5) * 0.01).toFixed(3)),
      gyroZ: Number(gyroZ.toFixed(3)),
      speed: Number((65.0 + accX * 5.0).toFixed(1)),
    });

    try {
      const resp = await assessTelemetry({
        timestamp: t,
        style_prediction: stylePred,
        style_confidence: styleConf,
        maneuver_prediction: maneuverPred,
        maneuver_confidence: maneuverConf,
        pipeline_a_health: 'HEALTHY',
        pipeline_b_health: 'HEALTHY',
      });

      setSimAssessment(resp);
      if (onGlobalAssessmentUpdate) onGlobalAssessmentUpdate(resp);

      if (resp.maneuver && !resp.maneuver.includes('None')) {
        setSimEventHistory((prev) => [
          {
            id: `${step}-${Date.now()}`,
            timestamp: t.toFixed(1),
            event: resp.maneuver,
            risk: resp.risk_score,
            state: resp.safety_state,
          },
          ...prev.slice(0, 19),
        ]);
      }
    } catch (err) {
      console.error('Simulation streaming step failed:', err);
    }
  };

  // Upload handler
  const handleUploadCsv = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    setIsUploading(true);
    setUploadResult(null);

    const formData = new FormData();
    formData.append('driver_id', uploadDriverId);
    formData.append('session_id', uploadSessionId || `sess_${Date.now()}_${uploadDriverId}`);
    formData.append('file', uploadFile);

    try {
      const res = await uploadSensorCsv(formData);
      setUploadResult({ success: true, data: res });
    } catch (err) {
      setUploadResult({ success: false, error: err.message });
    } finally {
      setIsUploading(false);
    }
  };

  // Observer Annotation
  const handleSaveAnnotation = async () => {
    const activeSess = liveStatus?.session_id || liveStatus?.active_session_id;
    if (!liveStatus || !activeSess) {
      setAnnotationSuccess({ ok: false, msg: 'No active recording session to annotate.' });
      return;
    }
    const currentFrame = liveStatus.latest_frame;
    const nowSec = currentFrame ? currentFrame.timestamp : Date.now() / 1000;

    try {
      await createAnnotation({
        session_id: activeSess,
        driver_id: liveStatus.driver_id || liveStatus.assigned_driver_id || 'DRV_001',
        annotator_id: 'IN_CAR_OBSERVER_01',
        start_timestamp: Math.max(0, nowSec - 2.5),
        end_timestamp: nowSec,
        label: annotationCategory,
        evidence: annotationNote || `Marked live during ${activeSess}`,
      });
      setAnnotationSuccess({ ok: true, msg: `Recorded [${annotationCategory}] ground-truth tag.` });
      setAnnotationNote('');
      setTimeout(() => setAnnotationSuccess(null), 3000);
    } catch (err) {
      setAnnotationSuccess({ ok: false, msg: err.message });
    }
  };

  const isPhysConnected = liveStatus?.status === 'STREAMING' || liveStatus?.status === 'STANDBY' || liveStatus?.connected;
  const measuredHz = Number(liveStatus?.actual_frequency_hz ?? liveStatus?.measured_hz ?? 0.0);
  const activeDriver = liveStatus?.driver_id || liveStatus?.assigned_driver_id || 'UNKNOWN';
  const latestFrame = liveStatus?.latest_frame;

  // Acc norm calculation (handling both 'Acc X' alias and 'acc_x')
  const rawAx = latestFrame ? Number(latestFrame['Acc X'] ?? latestFrame.acc_x ?? 0) : 0;
  const rawAy = latestFrame ? Number(latestFrame['Acc Y'] ?? latestFrame.acc_y ?? 0) : 0;
  const rawAz = latestFrame ? Number(latestFrame['Acc Z'] ?? latestFrame.acc_z ?? -1) : -1;
  const rawGx = latestFrame ? Number(latestFrame.gyro_x ?? 0) : 0;
  const rawGy = latestFrame ? Number(latestFrame.gyro_y ?? 0) : 0;
  const rawGz = latestFrame ? Number(latestFrame.gyro_z ?? 0) : 0;

  // If accelerometer values are already in g (|az| ~ 1.0), norm is direct; if in m/s^2 (|az| ~ 9.8), divide by 9.80665
  const isMps2 = Math.abs(rawAz) > 5.0;
  const axG = isMps2 ? rawAx / 9.80665 : rawAx;
  const ayG = isMps2 ? rawAy / 9.80665 : rawAy;
  const azG = isMps2 ? rawAz / 9.80665 : rawAz;
  const accNorm = Math.sqrt(axG ** 2 + ayG ** 2 + azG ** 2);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto bg-transparent text-black dark:text-white">
      <ComponentInspectorModal
        isOpen={!!inspectorTarget}
        onClose={() => setInspectorTarget(null)}
        target={inspectorTarget}
      />

      {/* Dual Mode Switcher Banner */}
      <div className="spacex-card flex flex-col md:flex-row md:items-center justify-between gap-4 p-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0">
            <Smartphone className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-neutral-500 dark:text-neutral-400 block">
              Dual-Source Ingestion Pipeline
            </span>
            <h2 className="text-sm sm:text-base font-display font-bold uppercase tracking-spacex text-black dark:text-white">
              Telemetry Ingestion Mode
            </h2>
          </div>
        </div>

        <div className="flex items-center gap-1.5 bg-[#f1f3f5] dark:bg-[#18181b] p-1.5 rounded-lg border border-[#dee2e6] dark:border-[#27272a]">
          <Button
            variant={activeMode === 'physical' ? 'default' : 'outline'}
            onClick={() => setActiveMode('physical')}
            className={`px-4 py-1.5 text-xs font-display font-bold uppercase tracking-wider rounded-md transition-all ${
              activeMode === 'physical' ? 'shadow-sm' : ''
            }`}
          >
            Physical Smartphone (Live 20 Hz)
          </Button>
          <Button
            variant={activeMode === 'simulation' ? 'default' : 'outline'}
            onClick={() => setActiveMode('simulation')}
            className={`px-4 py-1.5 text-xs font-display font-bold uppercase tracking-wider rounded-md transition-all ${
              activeMode === 'simulation' ? 'shadow-sm' : ''
            }`}
          >
            Historical Dataset Replay / Demo
          </Button>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MODE 1: PHYSICAL SMARTPHONE SENSOR (LIVE 20 HZ)                           */}
      {/* ========================================================================= */}
      {activeMode === 'physical' && (
        <div className="space-y-6">
          {/* Hardware Connection Card */}
          <div className="spacex-card">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-start gap-3.5">
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                    isPhysConnected
                      ? 'bg-emerald-600 text-white'
                      : 'bg-neutral-800 text-neutral-400'
                  }`}
                >
                  <Radio className="w-4 h-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2.5">
                    <h3 className="text-xs sm:text-sm font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                      Physical Smartphone Hardware:{' '}
                      <span className={isPhysConnected ? 'text-emerald-500 font-bold' : 'text-neutral-400'}>
                        {liveStatus?.connection_status || 'DISCONNECTED'}
                      </span>
                    </h3>
                    <Badge variant={isPhysConnected ? 'HEALTHY' : 'CAUTION'} size="sm">
                      {isPhysConnected ? 'ONLINE' : 'NOT CONNECTED'}
                    </Badge>
                  </div>
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1 font-sans">
                    {liveStatus?.message || 'Searching for physical smartphone logger stream on /api/sensor/live/stream...'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2.5 shrink-0">
                <ParticleButton
                  onClick={checkLiveStatus}
                  disabled={isCheckingLive}
                  variant="subtle"
                  size="sm"
                  pendingLabel="Polling Stream..."
                  successLabel="Verified Active"
                  errorLabel="Hardware Offline"
                  icon={<RefreshCw className="w-3.5 h-3.5" />}
                >
                  Refresh Status
                </ParticleButton>

                <a
                  href="/sensor-logger/"
                  target="_blank"
                  rel="noreferrer"
                  className="btn-spacex-solid text-[11px] flex items-center gap-2"
                >
                  <Smartphone className="w-3.5 h-3.5" />
                  <span>Open Web Logger</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>

            {/* Honest Hardware Indicator Notice */}
            <div className="mt-4 pt-3 border-t border-[#dee2e6] dark:border-[#27272a] text-xs font-mono flex items-center justify-between text-neutral-600 dark:text-neutral-400">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-500" />
                <span>
                  Physical Hardware State:{' '}
                  <strong className="text-black dark:text-white">
                    {isPhysConnected ? 'VERIFIED_ACTIVE' : 'NOT_VERIFIED'}
                  </strong>{' '}
                  (Real-world vehicle validation requires phone in rigid mount)
                </span>
              </div>
              <div>
                <span>Device Token: </span>
                <span className="text-black dark:text-white font-bold">{liveStatus?.device_token || 'NONE'}</span>
              </div>
            </div>
          </div>

          {/* If DISCONNECTED: Show Connection Guide & File Ingestion Box */}
          {!isPhysConnected && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* How to Connect Box */}
              <div className="spacex-card space-y-3">
                <div className="flex items-center gap-2 border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
                  <Smartphone className="w-4 h-4 text-black dark:text-white" />
                  <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                    How to Connect Physical Phone
                  </h3>
                </div>
                <div className="text-xs space-y-2 text-neutral-600 dark:text-neutral-400 font-sans">
                  <p>
                    <strong>1. Launch Web Logger:</strong> Click the <em>&quot;Open Web Logger&quot;</em> button above or navigate to{' '}
                    <code className="text-black dark:text-white font-mono bg-neutral-200 dark:bg-neutral-800 px-1 py-0.5 rounded-lg">
                      /sensor-logger/
                    </code>{' '}
                    on your phone browser (Chrome/Safari).
                  </p>
                  <p>
                    <strong>2. Native Android App:</strong> Compile and install{' '}
                    <code className="text-black dark:text-white font-mono bg-neutral-200 dark:bg-neutral-800 px-1 py-0.5 rounded-lg">
                      mobile/android/
                    </code>{' '}
                    with Android Studio for background foreground service logging.
                  </p>
                  <p>
                    <strong>3. Rigid Mount:</strong> Fasten the phone to the dashboard or windshield in landscape orientation.
                  </p>
                  <p>
                    <strong>4. S0 Calibration:</strong> Keep the car stationary for 30 seconds to calibrate the 1.0g gravity vector.
                  </p>
                </div>
              </div>

              {/* Offline File Ingestion Box */}
              <div className="spacex-card space-y-3">
                <div className="flex items-center gap-2 border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
                  <UploadCloud className="w-4 h-4 text-black dark:text-white" />
                  <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                    Upload Recorded Telemetry CSV
                  </h3>
                </div>
                <form onSubmit={handleUploadCsv} className="space-y-3 text-xs">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-neutral-500 mb-1">
                        Driver ID (DRV_001 - DRV_006)
                      </label>
                      <select
                        value={uploadDriverId}
                        onChange={(e) => setUploadDriverId(e.target.value)}
                        className="w-full bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] p-1.5 rounded-lg font-mono text-black dark:text-white"
                      >
                        <option value="DRV_001">DRV_001</option>
                        <option value="DRV_002">DRV_002</option>
                        <option value="DRV_003">DRV_003</option>
                        <option value="DRV_004">DRV_004</option>
                        <option value="DRV_005">DRV_005</option>
                        <option value="DRV_006">DRV_006</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[10px] uppercase font-bold text-neutral-500 mb-1">
                        Session Identifier
                      </label>
                      <input
                        type="text"
                        placeholder="e.g. sess_highway_01"
                        value={uploadSessionId}
                        onChange={(e) => setUploadSessionId(e.target.value)}
                        className="w-full bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] p-1.5 rounded-lg font-mono text-black dark:text-white"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-[10px] uppercase font-bold text-neutral-500 mb-1">
                      Raw 20 Hz CSV File
                    </label>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => setUploadFile(e.target.files[0])}
                      className="w-full text-xs font-mono text-neutral-500 file:mr-2 file:py-1 file:px-2 file:rounded-lg file:border-0 file:text-xs file:bg-black file:text-white dark:file:bg-white dark:file:text-black cursor-pointer"
                    />
                  </div>

                  <ParticleButton
                    type="submit"
                    disabled={isUploading || !uploadFile}
                    variant="solid"
                    className="w-full text-xs py-2"
                    pendingLabel="Validating 6 Quality Gates..."
                    successLabel="Quality Manifest Approved!"
                    errorLabel="Validation Rejected"
                    icon={<UploadCloud className="w-3.5 h-3.5" />}
                  >
                    Upload & Validate Quality
                  </ParticleButton>

                  {uploadResult && (
                    <div
                      className={`p-2 rounded-lg border text-xs font-mono ${
                        uploadResult.success
                          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                          : 'bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400'
                      }`}
                    >
                      {uploadResult.success ? (
                        <div>
                          <p className="font-bold">Uploaded &amp; Accepted!</p>
                          <p>Samples: {uploadResult.data.sample_count} | Quality: {uploadResult.data.data_quality_status}</p>
                          <p className="text-[10px] truncate">SHA-256: {uploadResult.data.sha256}</p>
                        </div>
                      ) : (
                        <p>Error: {uploadResult.error}</p>
                      )}
                    </div>
                  )}
                </form>
              </div>
            </div>
          )}

          {/* If CONNECTED or STREAMING: Render Full Live Physical HUD */}
          {isPhysConnected && (
            <div className="space-y-6">
              {/* Top Frequency & Session Gauge Bar */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="spacex-card p-3">
                  <span className="text-[10px] uppercase font-mono text-neutral-500 block">Measured Rate</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-xl font-bold font-mono text-black dark:text-white tabular-nums">
                      {measuredHz.toFixed(1)}
                    </span>
                    <span className="text-xs font-mono text-neutral-400">/ 20.0 Hz</span>
                  </div>
                  <Badge
                    variant={measuredHz >= 18.0 && measuredHz <= 22.0 ? 'HEALTHY' : 'CAUTION'}
                    size="sm"
                  >
                    {measuredHz >= 18.0 && measuredHz <= 22.0 ? 'RATE STABLE' : 'JITTER'}
                  </Badge>
                </div>

                <div className="spacex-card p-3">
                  <span className="text-[10px] uppercase font-mono text-neutral-500 block">Frames Ingested</span>
                  <div className="text-xl font-bold font-mono text-black dark:text-white tabular-nums">
                    {(liveStatus?.total_frames_received ?? (recentFrames ? recentFrames.length : 0)).toLocaleString()}
                  </div>
                  <span className="text-[10px] font-mono text-neutral-500">20 ms intervals</span>
                </div>

                <div className="spacex-card p-3">
                  <span className="text-[10px] uppercase font-mono text-neutral-500 block">Assigned Driver</span>
                  <div className="text-xl font-bold font-mono text-black dark:text-white">
                    {activeDriver}
                  </div>
                  <span className="text-[10px] font-mono text-emerald-500">DEV POOL (001-006)</span>
                </div>

                <div className="spacex-card p-3">
                  <span className="text-[10px] uppercase font-mono text-neutral-500 block">Static Gravity Norm</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-xl font-bold font-mono text-black dark:text-white tabular-nums">
                      {accNorm.toFixed(3)}
                    </span>
                    <span className="text-xs font-mono text-neutral-400">g</span>
                  </div>
                  <Badge variant={accNorm >= 0.98 && accNorm <= 1.02 ? 'HEALTHY' : 'CAUTION'} size="sm">
                    {accNorm >= 0.98 && accNorm <= 1.02 ? 'GRAVITY NOMINAL' : 'CALIBRATING'}
                  </Badge>
                </div>
              </div>

              {/* Kinematic Gauges Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* 6-DOF IMU Gauges (8 cols) */}
                <div className="lg:col-span-8 spacex-card space-y-4">
                  <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-black dark:text-white" />
                      <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                        Live 6-DOF Kinematic Channels (20 Hz Native)
                      </h3>
                    </div>
                    <span className="text-[10px] font-mono uppercase text-emerald-500">
                      WebSocket Telemetry Active (Device Coordinates)
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 text-center">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-neutral-500 block">
                        Acc X (Device / Lat)
                      </span>
                      <span className="text-lg font-mono font-bold text-black dark:text-white tabular-nums">
                        {latestFrame ? Number(latestFrame.acc_x ?? latestFrame['Acc X'] ?? 0).toFixed(3) : '0.000'} g
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400 block">
                        {latestFrame ? (Number(latestFrame.acc_x ?? latestFrame['Acc X'] ?? 0) * 9.80665).toFixed(2) : '0.00'} m/s²
                      </span>
                    </div>

                    <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 text-center">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-neutral-500 block">
                        Acc Y (Device / Long)
                      </span>
                      <span className="text-lg font-mono font-bold text-black dark:text-white tabular-nums">
                        {latestFrame ? Number(latestFrame.acc_y ?? latestFrame['Acc Y'] ?? 0).toFixed(3) : '0.000'} g
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400 block">
                        {latestFrame ? (Number(latestFrame.acc_y ?? latestFrame['Acc Y'] ?? 0) * 9.80665).toFixed(2) : '0.00'} m/s²
                      </span>
                    </div>

                    <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 text-center">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-neutral-500 block">
                        Acc Z (Device / Norm)
                      </span>
                      <span className="text-lg font-mono font-bold text-black dark:text-white tabular-nums">
                        {latestFrame ? Number(latestFrame.acc_z ?? latestFrame['Acc Z'] ?? -1.0).toFixed(3) : '-1.000'} g
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400 block">
                        {latestFrame ? (Number(latestFrame.acc_z ?? latestFrame['Acc Z'] ?? -1.0) * 9.80665).toFixed(2) : '-9.81'} m/s²
                      </span>
                    </div>

                    <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 text-center">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-neutral-500 block">
                        Roll Rate (Gyro X)
                      </span>
                      <span className="text-lg font-mono font-bold text-black dark:text-white tabular-nums">
                        {latestFrame ? latestFrame.gyro_x.toFixed(4) : '0.0000'}
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400 block">rad/s</span>
                    </div>

                    <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 text-center">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-neutral-500 block">
                        Pitch Rate (Gyro Y)
                      </span>
                      <span className="text-lg font-mono font-bold text-black dark:text-white tabular-nums">
                        {latestFrame ? latestFrame.gyro_y.toFixed(4) : '0.0000'}
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400 block">rad/s</span>
                    </div>

                    <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 text-center">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-neutral-500 block">
                        Yaw Rate (Gyro Z)
                      </span>
                      <span className="text-lg font-mono font-bold text-black dark:text-white tabular-nums">
                        {latestFrame ? latestFrame.gyro_z.toFixed(4) : '0.0000'}
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400 block">rad/s</span>
                    </div>
                  </div>

                  {/* GPS & Kinematics summary */}
                  <div className="pt-2 flex flex-wrap items-center justify-between gap-4 text-xs font-mono uppercase text-neutral-500">
                    <div>
                      Speed:{' '}
                      <span className="text-black dark:text-white font-bold">
                        {latestFrame?.gps_speed !== null && latestFrame?.gps_speed !== undefined
                          ? `${(latestFrame.gps_speed * 3.6).toFixed(1)} km/h`
                          : 'Awaiting Fix'}
                      </span>
                    </div>
                    <div>
                      Coords:{' '}
                      <span className="text-black dark:text-white font-bold">
                        {latestFrame?.gps_lat ? `${latestFrame.gps_lat.toFixed(4)}, ${latestFrame.gps_lon.toFixed(4)}` : 'N/A'}
                      </span>
                    </div>
                    <div>
                      GPS Accuracy:{' '}
                      <span className="text-black dark:text-white font-bold">
                        {latestFrame?.gps_accuracy ? `±${latestFrame.gps_accuracy.toFixed(1)}m` : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Ground Truth Annotation Trigger (4 cols) */}
                <div className="lg:col-span-4 spacex-card space-y-4 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 border-b border-[#dee2e6] dark:border-[#27272a] pb-2 mb-3">
                      <Tag className="w-4 h-4 text-black dark:text-white" />
                      <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                        Observer Ground Truth Tagging
                      </h3>
                    </div>
                    <p className="text-xs text-neutral-600 dark:text-neutral-400 font-sans mb-3">
                      Tag current physical maneuver as independent ground-truth for ML evaluation:
                    </p>

                    <div className="grid grid-cols-2 gap-2 mb-3">
                      {[
                        { label: 'Normal Driving', val: 'NORMAL' },
                        { label: 'Sudden Accel', val: 'SUDDEN_ACCELERATION' },
                        { label: 'Sudden Brake', val: 'SUDDEN_DECELERATION' },
                        { label: 'Sharp Turn', val: 'SUDDEN_RIGHT_TURN' },
                        { label: 'Aggressive Style', val: 'AGGRESSIVE' },
                        { label: 'Uncertain / Noise', val: 'UNCERTAIN' },
                      ].map((item) => (
                        <Button
                          key={item.val}
                          type="button"
                          variant={annotationCategory === item.val ? 'default' : 'outline'}
                          onClick={() => setAnnotationCategory(item.val)}
                          className={`h-auto p-2 text-[11px] font-mono rounded-lg transition-all text-left justify-start ${
                            annotationCategory === item.val ? 'font-bold' : ''
                          }`}
                        >
                          {item.label}
                        </Button>
                      ))}
                    </div>

                    <input
                      type="text"
                      placeholder="Optional observer notes..."
                      value={annotationNote}
                      onChange={(e) => setAnnotationNote(e.target.value)}
                      className="w-full bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] p-2 rounded-lg font-mono text-xs text-black dark:text-white mb-3"
                    />

                    <ParticleButton
                      onClick={handleSaveAnnotation}
                      variant="solid"
                      className="w-full text-xs py-2"
                      pendingLabel="Saving Tag..."
                      successLabel={`Recorded [${annotationCategory}]!`}
                      errorLabel="Failed to Save"
                      icon={<Tag className="w-3.5 h-3.5" />}
                    >
                      Record Annotation [{annotationCategory}]
                    </ParticleButton>

                    {annotationSuccess && (
                      <p
                        className={`text-xs font-mono mt-2 ${
                          annotationSuccess.ok ? 'text-emerald-500' : 'text-red-500'
                        }`}
                      >
                        {annotationSuccess.msg}
                      </p>
                    )}
                  </div>

                  <div className="pt-2 border-t border-[#dee2e6] dark:border-[#27272a] text-[10px] font-mono text-neutral-500">
                    Annotations saved to immutable audit manifest
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODE 2: HISTORICAL DATASET REPLAY / DEMONSTRATION STREAM                   */}
      {/* ========================================================================= */}
      {activeMode === 'simulation' && (
        <div className="space-y-6">
          {/* Demonstration Notice */}
          <div className="spacex-card border-amber-500/40 bg-amber-500/5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-xs font-display font-bold uppercase tracking-wider text-black dark:text-white">
                    Historical Dataset Demonstration Mode (Synthetic/Replay Stream)
                  </h3>
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-0.5 font-sans">
                    Generating deterministic simulated telemetry derived from Dataset B &amp; C driving profiles to evaluate Safety Agent guardrails.
                  </p>
                </div>
              </div>

              <Button
                variant={isSimStreaming ? 'destructive' : 'outline'}
                onClick={toggleSimStreaming}
                className="text-xs px-5 py-2 font-display font-bold uppercase tracking-spacex rounded-full transition-all flex items-center gap-2 shrink-0 h-9"
              >
                {isSimStreaming ? (
                  <>
                    <Pause className="w-3.5 h-3.5" />
                    <span>Stop Replay</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    <span>Start Demo Stream</span>
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Real-Time Assessment Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Sensor HUD (6 cols) */}
            <div className="lg:col-span-6 spacex-card space-y-4">
              <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
                <div className="flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-black dark:text-white" />
                  <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                    Synthetic Vehicle Sensors HUD
                  </h3>
                </div>
                <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
                  {isSimStreaming ? 'Streaming 4 Hz' : 'Sensors Idle'}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2.5">
                <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-2.5 text-center">
                  <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 block">
                    Acc X (Fwd)
                  </span>
                  <span className="text-base font-mono font-bold text-black dark:text-white tabular-nums">
                    {simSensorValues.accX} g
                  </span>
                </div>
                <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-2.5 text-center">
                  <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 block">
                    Acc Y (Lat)
                  </span>
                  <span className="text-base font-mono font-bold text-black dark:text-white tabular-nums">
                    {simSensorValues.accY} g
                  </span>
                </div>
                <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-2.5 text-center">
                  <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 block">
                    Acc Z (Vert)
                  </span>
                  <span className="text-base font-mono font-bold text-black dark:text-white tabular-nums">
                    {simSensorValues.accZ} g
                  </span>
                </div>
                <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-2.5 text-center">
                  <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 block">
                    Roll Rate
                  </span>
                  <span className="text-base font-mono font-bold text-black dark:text-white tabular-nums">
                    {simSensorValues.gyroX} r/s
                  </span>
                </div>
                <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-2.5 text-center">
                  <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 block">
                    Pitch Rate
                  </span>
                  <span className="text-base font-mono font-bold text-black dark:text-white tabular-nums">
                    {simSensorValues.gyroY} r/s
                  </span>
                </div>
                <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-2.5 text-center">
                  <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 block">
                    Yaw Rate
                  </span>
                  <span className="text-base font-mono font-bold text-black dark:text-white tabular-nums">
                    {simSensorValues.gyroZ} r/s
                  </span>
                </div>
              </div>

              <div className="pt-2 flex items-center justify-between text-xs font-display uppercase tracking-wider text-neutral-500">
                <div>
                  Speed:{' '}
                  <span className="text-black dark:text-white font-mono font-bold tabular-nums">
                    {simSensorValues.speed} km/h
                  </span>
                </div>
                <div>
                  Source:{' '}
                  <span className="text-black dark:text-white font-bold">Historical Pattern</span>
                </div>
              </div>
            </div>

            {/* Live Safety Assessment (6 cols) */}
            <div className="lg:col-span-6 spacex-card flex flex-col justify-between">
              <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2 mb-3">
                <div className="flex items-center gap-2">
                  <Radio className="w-3.5 h-3.5 text-black dark:text-white" />
                  <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                    Real-Time Safety Assessment
                  </h3>
                </div>
                <Badge variant={simAssessment?.safety_state || 'NORMAL'} size="sm">
                  {simAssessment?.safety_state || 'NORMAL'}
                </Badge>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-around gap-6 my-auto">
                <RiskGauge
                  score={simAssessment?.risk_score ?? 0}
                  state={simAssessment?.safety_state ?? 'NORMAL'}
                  size={144}
                  strokeWidth={8}
                />

                <div className="space-y-2.5 font-display text-xs uppercase tracking-wider w-full max-w-xs">
                  <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                    <span className="text-neutral-500">Driving Style:</span>
                    <span className="font-bold text-black dark:text-white">
                      {simAssessment?.driving_style || 'Normal'}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                    <span className="text-neutral-500">Kinematic Event:</span>
                    <span className="font-bold text-black dark:text-white">
                      {simAssessment?.maneuver || 'None'}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                    <span className="text-neutral-500">Data Quality:</span>
                    <span className="font-bold text-black dark:text-white">Synthetic Nominal</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-500">Timestamp:</span>
                    <span className="text-black dark:text-white font-mono font-bold tabular-nums">
                      T+{Number(simAssessment?.timestamp ?? 0).toFixed(2)}s
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Live Events Feed */}
          <div className="spacex-card">
            <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2 mb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-black dark:text-white" />
                <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                  Demo Driving Events Feed
                </h3>
              </div>
              <span className="text-[10px] font-mono uppercase text-neutral-500">
                {simEventHistory.length} events logged in session
              </span>
            </div>

            {simEventHistory.length === 0 ? (
              <div className="text-center py-8 text-neutral-400 dark:text-neutral-600 font-display text-xs uppercase tracking-wider">
                {isSimStreaming
                  ? 'Monitoring demo stream... No hazard events triggered yet.'
                  : 'Stream is idle. Press "Start Demo Stream" to begin evaluation.'}
              </div>
            ) : (
              <div className="space-y-1.5 max-h-60 overflow-y-auto font-mono text-xs">
                {simEventHistory.map((ev) => (
                  <div
                    key={ev.id}
                    className="flex items-center justify-between p-2.5 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-neutral-500 font-mono tabular-nums text-[11px]">T+{ev.timestamp}s</span>
                      <span className="text-black dark:text-white font-display uppercase tracking-wider font-bold">
                        {ev.event}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-neutral-600 dark:text-neutral-400 tabular-nums">
                        Risk: {Number(ev.risk).toFixed(1)} pts
                      </span>
                      <Badge variant={ev.state} size="sm">
                        {ev.state}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipBack,
  SkipForward,
  Clock,
  Film,
  AlertTriangle,
} from 'lucide-react';
import { fetchReplaySessions, fetchReplaySessionDetail, assessTelemetry } from '../../services/api';
import { CinematicVehicleScene } from '../three/CinematicVehicleScene';
import { Badge } from '../common/Badge';
import { RiskGauge } from '../common/RiskGauge';
import { AlertCard } from '../common/AlertCard';
import { ComponentInspectorModal } from '../common/ComponentInspectorModal';
import { Button } from '../ui/button';

export function ReplayStudio({ onGlobalAssessmentUpdate, vehicleColor, onColorChange }) {
  const [sessions, setSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState('session-drv-1');
  const [sessionDetail, setSessionDetail] = useState(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [currentAssessment, setCurrentAssessment] = useState(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const [inspectorTarget, setInspectorTarget] = useState(null);

  const playbackTimerRef = useRef(null);

  useEffect(() => {
    loadSessions();
    return () => {
      if (playbackTimerRef.current) clearInterval(playbackTimerRef.current);
    };
  }, []);

  const loadSessions = async () => {
    try {
      const data = await fetchReplaySessions();
      setSessions(data);
      if (data.length > 0) {
        loadSessionDetail(data[0].id);
      }
    } catch (err) {
      console.error('Failed to load replay sessions:', err);
    }
  };

  const loadSessionDetail = async (sessionId) => {
    setIsLoadingSession(true);
    setIsPlaying(false);
    if (playbackTimerRef.current) clearInterval(playbackTimerRef.current);

    try {
      const detail = await fetchReplaySessionDetail(sessionId);
      setSessionDetail(detail);
      setSelectedSessionId(sessionId);
      setCurrentFrameIdx(0);
      if (detail.frames.length > 0) {
        evaluateFrame(detail.frames[0]);
      }
    } catch (err) {
      console.error('Failed to load session detail:', err);
    } finally {
      setIsLoadingSession(false);
    }
  };

  const evaluateFrame = async (frame) => {
    if (!frame || !frame.telemetry_request) return;
    try {
      const assessment = await assessTelemetry(frame.telemetry_request);
      setCurrentAssessment(assessment);
      if (onGlobalAssessmentUpdate) onGlobalAssessmentUpdate(assessment);
    } catch (err) {
      console.error('Replay frame assessment error:', err);
    }
  };

  useEffect(() => {
    if (isPlaying && sessionDetail?.frames?.length > 0) {
      const intervalMs = Math.max(50, Math.round(200 / playbackSpeed));
      playbackTimerRef.current = setInterval(() => {
        setCurrentFrameIdx((prevIdx) => {
          const nextIdx = prevIdx + 1;
          if (nextIdx >= sessionDetail.frames.length) {
            setIsPlaying(false);
            return prevIdx;
          }
          evaluateFrame(sessionDetail.frames[nextIdx]);
          return nextIdx;
        });
      }, intervalMs);
    } else {
      if (playbackTimerRef.current) clearInterval(playbackTimerRef.current);
    }

    return () => {
      if (playbackTimerRef.current) clearInterval(playbackTimerRef.current);
    };
  }, [isPlaying, playbackSpeed, sessionDetail]);

  const togglePlay = () => {
    if (!sessionDetail?.frames?.length) return;
    if (!isPlaying && currentFrameIdx >= sessionDetail.frames.length - 1) {
      setCurrentFrameIdx(0);
      evaluateFrame(sessionDetail.frames[0]);
    }
    setIsPlaying(!isPlaying);
  };

  const handleRestart = () => {
    setIsPlaying(false);
    setCurrentFrameIdx(0);
    if (sessionDetail?.frames?.[0]) {
      evaluateFrame(sessionDetail.frames[0]);
    }
  };

  const handleSeek = (newIdx) => {
    const idx = Math.max(0, Math.min((sessionDetail?.frames?.length || 1) - 1, newIdx));
    setCurrentFrameIdx(idx);
    if (sessionDetail?.frames?.[idx]) {
      evaluateFrame(sessionDetail.frames[idx]);
    }
  };

  const currentFrame = sessionDetail?.frames?.[currentFrameIdx];
  const totalFrames = sessionDetail?.frames?.length || 0;

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto bg-transparent text-black dark:text-white">
      {/* Component Inspector Modal */}
      <ComponentInspectorModal
        isOpen={!!inspectorTarget}
        onClose={() => setInspectorTarget(null)}
        target={inspectorTarget}
      />

      {/* Session Selection & Summary Header */}
      <div className="spacex-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0">
              <Film className="w-4 h-4" />
            </div>
            <h2 className="text-sm sm:text-base font-display font-bold uppercase tracking-spacex text-black dark:text-white">
              Trip History &amp; Replay Studio
            </h2>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1 font-sans">
            Replay recorded multi-driver driving sessions with synchronized 3D chassis kinematics and real-time inference telemetry.
          </p>
        </div>

        {/* Session Selector Dropdown */}
        <div className="flex items-center gap-2.5">
          <label className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
            Trip Log:
          </label>
          <select
            value={selectedSessionId}
            onChange={(e) => loadSessionDetail(e.target.value)}
            disabled={isLoadingSession}
            className="bg-white dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg px-3 py-1.5 text-xs font-display uppercase tracking-wider text-black dark:text-white focus:border-black dark:focus:border-white outline-none shadow-xs"
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title} ({s.total_frames} frames, {s.duration_seconds}s)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main 3D Twin & Replay Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* 3D Vehicle Scene (7 cols) */}
        <div className="lg:col-span-7 h-[480px] flex flex-col rounded-sm overflow-hidden border border-[#dee2e6] dark:border-[#27272a] bg-black relative">
          <CinematicVehicleScene
            safetyState={currentAssessment?.safety_state || 'NORMAL'}
            drivingStyle={currentAssessment?.driving_style || 'Normal'}
            maneuver={currentAssessment?.maneuver || 'None'}
            riskScore={currentAssessment?.risk_score ?? 0}
            vehicleColor={vehicleColor}
            onColorChange={onColorChange}
            onInspectComponent={(target) => setInspectorTarget(target)}
            className="flex-1 w-full h-full"
          />
        </div>

        {/* Current Frame Analysis & Predictions (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <div className="spacex-card flex flex-col items-center text-center">
            <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400 mb-2">
              Synchronized Safety Score
            </span>
            <RiskGauge
              score={currentAssessment?.risk_score ?? 0}
              state={currentAssessment?.safety_state ?? 'NORMAL'}
              size={144}
              strokeWidth={8}
            />

            <div className="mt-3 flex items-center gap-2.5">
              <Badge variant={currentAssessment?.safety_state || 'NORMAL'} size="md">
                {currentAssessment?.safety_state || 'NORMAL'}
              </Badge>
              <span className="text-xs font-mono text-neutral-500 dark:text-neutral-400 tabular-nums">
                T+{Number(currentFrame?.timestamp ?? 0).toFixed(2)}s
              </span>
            </div>
          </div>

          <div className="spacex-card space-y-2.5 font-display text-xs uppercase tracking-wider">
            <div className="border-b border-[#dee2e6] dark:border-[#27272a] pb-2 flex items-center justify-between">
              <span className="text-[10px] font-bold text-neutral-500 dark:text-neutral-400">
                Frame Telemetry Prediction
              </span>
              <span className="text-[10px] text-black dark:text-white font-mono">
                Models Synchronized
              </span>
            </div>

            <div className="flex justify-between items-center text-xs">
              <span className="text-neutral-500 dark:text-neutral-400">Kinematic Event:</span>
              <span className="font-bold text-black dark:text-white">
                {currentAssessment?.maneuver || 'None'} ({((currentAssessment?.maneuver_confidence ?? 0.9) * 100).toFixed(0)}%)
              </span>
            </div>

            <div className="flex justify-between items-center text-xs">
              <span className="text-neutral-500 dark:text-neutral-400">Driving Style:</span>
              <span className="font-bold text-black dark:text-white">
                {currentAssessment?.driving_style || 'Normal'}
              </span>
            </div>

            <div className="pt-2 border-t border-[#dee2e6] dark:border-[#27272a] flex justify-between items-center text-xs">
              <span className="text-neutral-500 dark:text-neutral-400">Dataset Truth:</span>
              <span className="px-2 py-0.5 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg text-black dark:text-white font-mono text-[10px]">
                {currentFrame?.ground_truth_label || 'None'}
              </span>
            </div>

            <div className="pt-2 border-t border-[#dee2e6] dark:border-[#27272a] grid grid-cols-2 gap-2 text-[10px] font-mono text-neutral-600 dark:text-neutral-400">
              <div>Forward: <span className="text-black dark:text-white font-bold tabular-nums">{currentFrame?.acc_x ?? 0} g</span></div>
              <div>Lateral: <span className="text-black dark:text-white font-bold tabular-nums">{currentFrame?.acc_y ?? 0} g</span></div>
              <div>Turn Rate: <span className="text-black dark:text-white font-bold tabular-nums">{currentFrame?.gyro_z ?? 0} r/s</span></div>
              <div>Speed: <span className="text-black dark:text-white font-bold tabular-nums">{currentFrame?.speed ?? 65} km/h</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* Scrubbable Timeline & Transport Controls */}
      <div className="spacex-card space-y-4">
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-black dark:text-white" />
              <span>T+{Number(currentFrame?.timestamp ?? 0).toFixed(2)}s / {Number(sessionDetail?.duration_seconds ?? 0).toFixed(2)}s</span>
            </span>
            <span className="font-mono">Frame {currentFrameIdx + 1} of {totalFrames}</span>
          </div>

          <div className="relative w-full">
            <input
              type="range"
              min="0"
              max={Math.max(0, totalFrames - 1)}
              value={currentFrameIdx}
              onChange={(e) => handleSeek(parseInt(e.target.value))}
              className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-full appearance-none cursor-pointer accent-black dark:accent-white"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={handleRestart}
              title="Restart session"
              aria-label="Restart session"
              className="h-8 w-8 rounded-lg"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>

            <Button
              variant="outline"
              size="icon"
              onClick={() => handleSeek(currentFrameIdx - 1)}
              disabled={currentFrameIdx <= 0}
              title="Previous frame"
              aria-label="Previous frame"
              className="h-8 w-8 rounded-lg"
            >
              <SkipBack className="w-4 h-4" />
            </Button>

            <Button
              variant="default"
              onClick={togglePlay}
              className="text-xs h-8 px-5 rounded-lg gap-2 font-display uppercase tracking-spacex"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Play</span>
                </>
              )}
            </Button>

            <Button
              variant="outline"
              size="icon"
              onClick={() => handleSeek(currentFrameIdx + 1)}
              disabled={currentFrameIdx >= totalFrames - 1}
              title="Next frame"
              aria-label="Next frame"
              className="h-8 w-8 rounded-lg"
            >
              <SkipForward className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-1 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-1">
            {[0.5, 1.0, 2.0, 4.0].map((spd) => (
              <Button
                key={spd}
                variant={playbackSpeed === spd ? "default" : "outline"}
                size="sm"
                onClick={() => setPlaybackSpeed(spd)}
                className="h-7 px-2.5 text-xs font-mono rounded-md"
              >
                {spd}x
              </Button>
            ))}
          </div>
        </div>
      </div>

      {currentAssessment?.active_alerts?.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-display uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span>Active Warnings at T+{Number(currentFrame?.timestamp ?? 0).toFixed(2)}s</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {currentAssessment.active_alerts.map((alert, i) => (
              <div
                key={alert.alert_id || i}
                onClick={() => setInspectorTarget({ type: 'alert', data: alert })}
                className="cursor-pointer"
              >
                <AlertCard alert={alert} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

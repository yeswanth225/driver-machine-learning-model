import React, { useState } from 'react';
import { CinematicVehicleScene } from '../three/CinematicVehicleScene';
import { ProcessConsole } from '../common/ProcessConsole';
import { ComponentInspectorModal } from '../common/ComponentInspectorModal';
import { RiskGauge } from '../common/RiskGauge';
import { Badge } from '../common/Badge';
import { AlertCard } from '../common/AlertCard';
import { ParticleButton } from '../ui/ParticleButton';
import { Button } from '@/components/ui/button';
import {
  Sliders,
  AlertTriangle,
  Activity,
  Check,
  ShieldCheck,
} from 'lucide-react';

export function CommandCenter({
  assessment,
  telemetryState,
  onAssessCustom,
  onResetAgent,
  vehicleColor,
  onColorChange,
}) {
  const [showSettings, setShowSettings] = useState(false);
  const [feedbackToast, setFeedbackToast] = useState(null);
  const [inspectorTarget, setInspectorTarget] = useState(null);

  // Manual override form state
  const [manualForm, setManualForm] = useState({
    style_prediction: telemetryState.style_prediction ?? 0,
    style_confidence: telemetryState.style_confidence ?? 0.95,
    maneuver_prediction: telemetryState.maneuver_prediction ?? 0,
    maneuver_confidence: telemetryState.maneuver_confidence ?? 0.92,
    sensor_health: telemetryState.pipeline_b_health || 'HEALTHY',
  });

  const handleApplySettings = (e) => {
    e.preventDefault();
    onAssessCustom({
      timestamp: (assessment?.timestamp || 0) + 1.0,
      style_prediction: manualForm.style_prediction === 'NONE' ? null : parseInt(manualForm.style_prediction),
      style_confidence: manualForm.style_prediction === 'NONE' ? null : parseFloat(manualForm.style_confidence),
      maneuver_prediction: manualForm.maneuver_prediction === 'NONE' ? null : parseInt(manualForm.maneuver_prediction),
      maneuver_confidence: manualForm.maneuver_prediction === 'NONE' ? null : parseFloat(manualForm.maneuver_confidence),
      pipeline_a_health: manualForm.sensor_health === 'MISSING' ? 'MISSING' : 'HEALTHY',
      pipeline_b_health: manualForm.sensor_health,
    });

    setFeedbackToast('Telemetry override applied to safety agent');
    setTimeout(() => setFeedbackToast(null), 3000);
  };

  const currentStyle = assessment?.driving_style || 'Normal';
  const currentManeuver = assessment?.maneuver || 'None / Nominal';
  const safetyState = assessment?.safety_state || 'NORMAL';
  const riskScore = assessment?.risk_score ?? 0.0;
  const alerts = assessment?.active_alerts || [];

  const getSafetyHeadline = (state) => {
    switch (state) {
      case 'NORMAL':
        return { label: 'Nominal Driving', desc: 'Kinematic telemetry within verified safety tolerances.', color: 'text-black dark:text-white' };
      case 'CAUTION':
        return { label: 'Elevated Caution', desc: 'Dynamic variance detected. Heightened monitoring engaged.', color: 'text-amber-600 dark:text-amber-400' };
      case 'WARNING':
        return { label: 'Safety Threshold Exceeded', desc: 'Hazard detected. Physical threshold limit triggered.', color: 'text-red-600 dark:text-red-400' };
      case 'HIGH_RISK':
      case 'CRITICAL':
        return { label: 'Critical Incident', desc: 'Compound emergency hazard. Immediate correction required.', color: 'text-red-700 dark:text-red-400' };
      default:
        return { label: 'Telemetry Monitoring', desc: 'Evaluating real-time IMU and kinematic channels.', color: 'text-neutral-500 dark:text-neutral-400' };
    }
  };

  const headline = getSafetyHeadline(safetyState);

  const getEventFriendlyName = (m) => {
    const s = String(m).toLowerCase();
    if (s.includes('accel')) return 'Sudden Acceleration';
    if (s.includes('decel') || s.includes('brak')) return 'Sudden Braking';
    if (s.includes('left')) return 'Sharp Left Turn';
    if (s.includes('right')) return 'Sharp Right Turn';
    return 'Stable Cruising';
  };

  const isAggressive = currentStyle.toLowerCase().includes('aggressive');

  const handleInspectAlert = (alert) => {
    setInspectorTarget({
      type: 'alert',
      data: alert,
    });
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto bg-transparent text-black dark:text-white">
      {/* Toast Notification */}
      {feedbackToast && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-2.5 bg-black dark:bg-white text-white dark:text-black rounded-sm font-display text-xs uppercase tracking-spacex shadow-xl">
          <Check className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-600 shrink-0" />
          <span>{feedbackToast}</span>
        </div>
      )}

      {/* Component Inspector Modal */}
      <ComponentInspectorModal
        isOpen={!!inspectorTarget}
        onClose={() => setInspectorTarget(null)}
        target={inspectorTarget}
      />

      {/* Top Grid: 3D Vehicle Showcase + Telemetry Safety Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* 3D Vehicle Digital Twin View (8 cols) */}
        <div className="lg:col-span-8 h-[520px] flex flex-col rounded-sm overflow-hidden border border-[#dee2e6] dark:border-[#27272a] bg-black relative">
          <CinematicVehicleScene
            safetyState={safetyState}
            drivingStyle={currentStyle}
            maneuver={currentManeuver}
            riskScore={riskScore}
            confidence={assessment?.maneuver_confidence ?? 0.95}
            vehicleColor={vehicleColor}
            onColorChange={onColorChange}
            onInspectComponent={(target) => setInspectorTarget(target)}
            className="flex-1 w-full h-full"
          />
        </div>

        {/* Safety Overview Instrument (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          {/* Risk Gauge Card */}
          <div className="spacex-card flex flex-col items-center justify-center text-center">
            <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400 mb-2">
              Mission Safety Overview
            </span>

            <RiskGauge score={riskScore} state={headline.label} size={168} strokeWidth={8} />

            <div className="mt-3">
              <h3 className={`text-base font-display font-bold uppercase tracking-spacex ${headline.color}`}>
                {headline.label}
              </h3>
              <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1 px-4 leading-relaxed font-sans">
                {headline.desc}
              </p>
            </div>

            <div className="w-full mt-4 pt-3 border-t border-[#dee2e6] dark:border-[#27272a] flex items-center justify-around font-display text-xs uppercase tracking-wider">
              <div>
                <span className="text-[9px] text-neutral-500 dark:text-neutral-400 block mb-1">Style</span>
                <Badge variant={isAggressive ? 'CAUTION' : 'NORMAL'} size="sm">
                  {isAggressive ? 'Aggressive' : 'Smooth'}
                </Badge>
              </div>
              <div className="w-px h-6 bg-[#dee2e6] dark:bg-[#27272a]" />
              <div>
                <span className="text-[9px] text-neutral-500 dark:text-neutral-400 block mb-1">Sensors</span>
                <Badge variant={assessment?.pipeline_b_health || 'HEALTHY'} size="sm">
                  {assessment?.pipeline_b_health === 'HEALTHY' ? 'Nominal' : 'Limited'}
                </Badge>
              </div>
            </div>
          </div>

          {/* Telemetry Channels Summary */}
          <div className="spacex-card space-y-2.5">
            <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
              <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
                Telemetry Channels
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(!showSettings)}
                className="gap-1.5 h-7 px-2.5 text-[10px] rounded-lg"
              >
                <Sliders className="w-3 h-3" />
                <span>{showSettings ? 'Hide Controls' : 'Manual Override'}</span>
              </Button>
            </div>

            <div className="space-y-1.5 text-xs font-sans">
              <div className="flex items-center justify-between p-2 rounded-lg bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a]">
                <span className="text-neutral-600 dark:text-neutral-400 text-xs">Driving Style:</span>
                <span className={`font-display font-bold uppercase tracking-wider ${isAggressive ? 'text-amber-600 dark:text-amber-400' : 'text-black dark:text-white'}`}>
                  {isAggressive ? 'Aggressive' : 'Smooth Cruising'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a]">
                <span className="text-neutral-600 dark:text-neutral-400 text-xs">Kinematic Event:</span>
                <span className="font-display font-bold uppercase tracking-wider text-black dark:text-white">
                  {getEventFriendlyName(currentManeuver)}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a]">
                <span className="text-neutral-600 dark:text-neutral-400 text-xs">Event Confidence:</span>
                <span className="font-mono text-black dark:text-white font-bold tabular-nums">
                  {((assessment?.maneuver_confidence ?? 0.95) * 100).toFixed(0)}%
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-lg bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a]">
                <span className="text-neutral-600 dark:text-neutral-400 text-xs">Active Alerts:</span>
                <span className={`font-display font-bold uppercase tracking-wider ${alerts.length > 0 ? 'text-red-600 dark:text-red-400' : 'text-black dark:text-white'}`}>
                  {alerts.length > 0 ? `${alerts.length} Active` : '0 Nominal'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Real-Time Telemetry Event Console */}
      <ProcessConsole
        safetyState={safetyState}
        drivingStyle={currentStyle}
        maneuver={currentManeuver}
        riskScore={riskScore}
        sensorHealth={assessment?.pipeline_b_health || 'HEALTHY'}
      />

      {/* Manual Override & Simulation Panel */}
      {showSettings && (
        <div className="spacex-card space-y-4">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-3 mb-2">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-black dark:text-white" />
              <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                Manual Telemetry Override &amp; Sensor Simulation
              </h3>
            </div>
            <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
              Deterministic D-13 Validation Rules
            </span>
          </div>

          <form onSubmit={handleApplySettings} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {/* Parameter 1 */}
              <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 space-y-1">
                <label className="text-[11px] font-display font-bold uppercase tracking-wider text-black dark:text-white block">
                  1. Driving Style
                </label>
                <p className="text-[10px] text-neutral-500 dark:text-neutral-400">
                  Pipeline A classification.
                </p>
                <select
                  value={manualForm.style_prediction}
                  onChange={(e) => setManualForm({ ...manualForm, style_prediction: e.target.value })}
                  className="w-full bg-white dark:bg-[#18181c] border border-[#dee2e6] dark:border-[#27272a] text-black dark:text-white rounded-lg px-2.5 py-1.5 text-xs font-display uppercase tracking-wider mt-1 focus:border-black dark:focus:border-white outline-none"
                >
                  <option value="0">Normal Cruising</option>
                  <option value="1">Aggressive Driving</option>
                  <option value="NONE">Sensor Offline</option>
                </select>
              </div>

              {/* Parameter 2 */}
              <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 space-y-1">
                <label className="text-[11px] font-display font-bold uppercase tracking-wider text-black dark:text-white block">
                  2. Kinematic Event
                </label>
                <p className="text-[10px] text-neutral-500 dark:text-neutral-400">
                  Pipeline B classification.
                </p>
                <select
                  value={manualForm.maneuver_prediction}
                  onChange={(e) => setManualForm({ ...manualForm, maneuver_prediction: e.target.value })}
                  className="w-full bg-white dark:bg-[#18181c] border border-[#dee2e6] dark:border-[#27272a] text-black dark:text-white rounded-lg px-2.5 py-1.5 text-xs font-display uppercase tracking-wider mt-1 focus:border-black dark:focus:border-white outline-none"
                >
                  <option value="0">Nominal Cruising</option>
                  <option value="1">Sudden Acceleration (SA)</option>
                  <option value="2">Sudden Deceleration (SD)</option>
                  <option value="3">Sharp Right Turn (SRT)</option>
                  <option value="4">Sharp Left Turn (SLT)</option>
                  <option value="NONE">Sensor Offline</option>
                </select>
              </div>

              {/* Parameter 3 */}
              <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 space-y-1">
                <label className="text-[11px] font-display font-bold uppercase tracking-wider text-black dark:text-white block">
                  3. Sensor Health
                </label>
                <p className="text-[10px] text-neutral-500 dark:text-neutral-400">
                  IMU bus availability.
                </p>
                <select
                  value={manualForm.sensor_health}
                  onChange={(e) => setManualForm({ ...manualForm, sensor_health: e.target.value })}
                  className="w-full bg-white dark:bg-[#18181c] border border-[#dee2e6] dark:border-[#27272a] text-black dark:text-white rounded-lg px-2.5 py-1.5 text-xs font-display uppercase tracking-wider mt-1 focus:border-black dark:focus:border-white outline-none"
                >
                  <option value="HEALTHY">Nominal (Healthy)</option>
                  <option value="DEGRADED">Degraded Telemetry</option>
                  <option value="MISSING">Hardware Offline</option>
                </select>
              </div>

              {/* Parameter 4 */}
              <div className="bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg p-3 space-y-1">
                <label className="text-[11px] font-display font-bold uppercase tracking-wider text-black dark:text-white flex justify-between">
                  <span>4. Confidence</span>
                  <span className="font-mono tabular-nums">{(manualForm.maneuver_confidence * 100).toFixed(0)}%</span>
                </label>
                <p className="text-[10px] text-neutral-500 dark:text-neutral-400">
                  Detection confidence probability.
                </p>
                <input
                  type="range"
                  min="0.4"
                  max="1.0"
                  step="0.05"
                  value={manualForm.maneuver_confidence}
                  onChange={(e) => setManualForm({ ...manualForm, maneuver_confidence: parseFloat(e.target.value) })}
                  className="w-full accent-black dark:accent-white mt-2"
                />
              </div>
            </div>

            {/* Actions Bar */}
            <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#dee2e6] dark:border-[#27272a]">
              <ParticleButton
                type="button"
                onClick={onResetAgent}
                variant="subtle"
                size="sm"
                pendingLabel="Resetting..."
                successLabel="Defaults Restored"
                errorLabel="Reset Failed"
              >
                Reset Default
              </ParticleButton>

              <ParticleButton
                type="submit"
                variant="solid"
                size="sm"
                pendingLabel="Applying..."
                successLabel="Override Active!"
                errorLabel="Apply Failed"
              >
                Apply Override
              </ParticleButton>
            </div>
          </form>
        </div>
      )}

      {/* Active Driving Alerts Feed */}
      {alerts.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-500" />
              <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                Active Hazard Advisories ({alerts.length})
              </h3>
            </div>
            <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
              Click advisory to inspect sensor evidence
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {alerts.map((alert, idx) => (
              <div
                key={alert.alert_id || idx}
                onClick={() => handleInspectAlert(alert)}
                className="cursor-pointer transition-transform hover:-translate-y-0.5"
                title="Click to view sensor evidence and advisory"
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

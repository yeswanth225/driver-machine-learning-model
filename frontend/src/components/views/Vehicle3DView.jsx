import React, { useState } from 'react';
import { CinematicVehicleScene } from '../three/CinematicVehicleScene';
import { ComponentInspectorModal } from '../common/ComponentInspectorModal';
import { Car, Activity, Zap, Compass, ShieldCheck } from 'lucide-react';
import { Badge } from '../common/Badge';
import { Button } from '../ui/button';

export function Vehicle3DView({
  assessment,
  telemetryState,
  onAssessCustom,
  vehicleColor,
  onColorChange,
}) {
  const [activeManeuver, setActiveManeuver] = useState(telemetryState?.maneuver_prediction ?? 0);
  const [activeStyle, setActiveStyle] = useState(telemetryState?.style_prediction ?? 0);
  const [inspectorTarget, setInspectorTarget] = useState(null);

  const currentStyle = assessment?.driving_style || 'Normal';
  const currentManeuver = assessment?.maneuver || 'None / Nominal';
  const safetyState = assessment?.safety_state || 'NORMAL';
  const riskScore = assessment?.risk_score ?? 0.0;

  // Real-time calculated kinematics based on active maneuver
  const getKinematicReadouts = () => {
    switch (activeManeuver) {
      case 1: // Sudden Acceleration
        return { accX: 0.54, accY: 0.04, accZ: -0.98, gyroZ: 0.02, speed: 74.2, steering: 1.2 };
      case 2: // Sudden Braking
        return { accX: -0.68, accY: -0.05, accZ: -0.99, gyroZ: -0.01, speed: 48.0, steering: -0.8 };
      case 3: // Sharp Right Turn
        return { accX: 0.12, accY: -0.46, accZ: -0.97, gyroZ: -0.38, speed: 61.5, steering: 18.5 };
      case 4: // Sharp Left Turn
        return { accX: 0.10, accY: 0.44, accZ: -0.97, gyroZ: 0.36, speed: 62.0, steering: -18.2 };
      default:
        return { accX: 0.02, accY: -0.01, accZ: -0.98, gyroZ: 0.01, speed: 65.0, steering: 0.2 };
    }
  };

  const kinematics = getKinematicReadouts();

  const handleTriggerEvent = (maneuverId, label) => {
    setActiveManeuver(maneuverId);
    onAssessCustom({
      timestamp: (assessment?.timestamp || 0) + 1.0,
      style_prediction: activeStyle,
      style_confidence: 0.94,
      maneuver_prediction: maneuverId,
      maneuver_confidence: 0.92,
      pipeline_a_health: 'HEALTHY',
      pipeline_b_health: 'HEALTHY',
    });
  };

  const handleToggleStyle = (styleId) => {
    setActiveStyle(styleId);
    onAssessCustom({
      timestamp: (assessment?.timestamp || 0) + 1.0,
      style_prediction: styleId,
      style_confidence: 0.94,
      maneuver_prediction: activeManeuver,
      maneuver_confidence: 0.90,
      pipeline_a_health: 'HEALTHY',
      pipeline_b_health: 'HEALTHY',
    });
  };

  const handleInspectSensor = (name, channel, val, unit, explanation, range) => {
    setInspectorTarget({
      type: 'sensor',
      data: {
        name,
        channel,
        currentValue: `${val} ${unit}`,
        nominalRange: range,
        rate: '10 Hz (100 ms intervals)',
        explanation,
      },
    });
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto bg-transparent text-black dark:text-white">
      {/* Component Inspector Modal */}
      <ComponentInspectorModal
        isOpen={!!inspectorTarget}
        onClose={() => setInspectorTarget(null)}
        target={inspectorTarget}
      />

      {/* Header Description */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#dee2e6] dark:border-[#27272a] gap-3">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center">
              <Car className="w-4 h-4" />
            </div>
            <h2 className="text-sm sm:text-base font-display font-bold uppercase tracking-spacex text-black dark:text-white">
              Driving Behavior &amp; 6-DOF Kinematics
            </h2>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1 font-sans">
            Real-time digital twin visualization reflecting chassis pitch, lateral roll, steering angle, and IMU forces.
          </p>
        </div>

        <div className="flex items-center gap-3 font-display text-xs uppercase tracking-wider">
          <Badge variant={safetyState} size="md">
            {safetyState}
          </Badge>
          <span className="text-neutral-400 dark:text-neutral-600">|</span>
          <span className="text-neutral-600 dark:text-neutral-400">
            Risk: <strong className="text-black dark:text-white font-mono">{riskScore.toFixed(1)}</strong>
          </span>
        </div>
      </div>

      {/* Main 3D Canvas */}
      <div className="h-[520px] rounded-sm overflow-hidden border border-[#dee2e6] dark:border-[#27272a] bg-black relative">
        <CinematicVehicleScene
          safetyState={safetyState}
          drivingStyle={currentStyle}
          maneuver={currentManeuver}
          riskScore={riskScore}
          confidence={assessment?.maneuver_confidence ?? 0.95}
          vehicleColor={vehicleColor}
          onColorChange={onColorChange}
          onInspectComponent={(target) => setInspectorTarget(target)}
          className="w-full h-full"
        />
      </div>

      {/* Interactive Controls & Kinematic Triggers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Style Selector */}
        <div className="spacex-card space-y-3">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
            <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
              Driving Behavior Style
            </span>
            <span className="text-xs font-display font-bold uppercase tracking-wider text-black dark:text-white">
              {currentStyle.toLowerCase().includes('aggressive') ? 'Aggressive' : 'Smooth Cruising'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-1">
            <Button
              variant={activeStyle === 0 ? "default" : "outline"}
              onClick={() => handleToggleStyle(0)}
              className="h-auto p-3.5 rounded-lg text-left flex-col items-start justify-start gap-1"
            >
              <div className="text-xs font-display font-bold uppercase tracking-spacex">
                Normal Cruising
              </div>
              <p className={`text-[11px] font-sans text-left normal-case ${activeStyle === 0 ? 'text-neutral-300 dark:text-neutral-600' : 'text-neutral-500 dark:text-neutral-400'}`}>
                Stable lane position, steady cruising suspension.
              </p>
            </Button>

            <Button
              variant={activeStyle === 1 ? "default" : "outline"}
              onClick={() => handleToggleStyle(1)}
              className={`h-auto p-3.5 rounded-lg text-left flex-col items-start justify-start gap-1 ${
                activeStyle === 1 ? 'bg-amber-600 text-white hover:bg-amber-700 dark:bg-amber-600 dark:text-white' : ''
              }`}
            >
              <div className="text-xs font-display font-bold uppercase tracking-spacex">
                Aggressive Driving
              </div>
              <p className={`text-[11px] font-sans text-left normal-case ${activeStyle === 1 ? 'text-amber-100' : 'text-neutral-500 dark:text-neutral-400'}`}>
                Dynamic chassis motion, heightened risk monitoring.
              </p>
            </Button>
          </div>
        </div>

        {/* Maneuver Triggers */}
        <div className="spacex-card space-y-3">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
            <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
              Trigger Driving Maneuver
            </span>
            <span className="text-xs font-display font-bold uppercase tracking-wider text-black dark:text-white truncate">
              {currentManeuver}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1 font-display text-xs uppercase tracking-wider">
            {[
              { id: 0, label: 'Stable Cruising' },
              { id: 1, label: 'Sudden Accel' },
              { id: 2, label: 'Sudden Braking' },
              { id: 4, label: 'Sharp Left' },
              { id: 3, label: 'Sharp Right' },
            ].map((m) => (
              <Button
                key={m.id}
                variant={activeManeuver === m.id ? "default" : "outline"}
                onClick={() => handleTriggerEvent(m.id, m.label)}
                className="py-2.5 px-3 h-auto rounded-lg text-center font-display text-xs uppercase tracking-wider"
              >
                {m.label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* 6-DOF IMU Sensor Telemetry Channels */}
      <div className="spacex-card space-y-3">
        <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-black dark:text-white" />
            <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
              6-DOF IMU Sensor Telemetry Channels
            </h3>
          </div>
          <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
            10 Hz Calibrated Stream // Click tile to inspect
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {[
            { label: 'Acc X (Longitudinal)', val: kinematics.accX, unit: 'g', range: '± 0.25 g', desc: 'Forward acceleration and braking force.' },
            { label: 'Acc Y (Lateral)', val: kinematics.accY, unit: 'g', range: '± 0.20 g', desc: 'Cornering centrifugal forces.' },
            { label: 'Acc Z (Vertical)', val: kinematics.accZ, unit: 'g', range: '-0.98 ± 0.05 g', desc: 'Gravity vector and road surface bumps.' },
            { label: 'Gyro Z (Yaw Rate)', val: kinematics.gyroZ, unit: 'rad/s', range: '± 0.05 rad/s', desc: 'Turn rotation angular velocity.' },
            { label: 'Vehicle Speed', val: kinematics.speed, unit: 'km/h', range: '0 - 120 km/h', desc: 'Derived ground travel velocity.' },
            { label: 'Steering Angle', val: kinematics.steering, unit: 'deg', range: '± 30 deg', desc: 'Front wheel steering displacement.' },
          ].map((s, idx) => (
            <div
              key={idx}
              onClick={() => handleInspectSensor(s.label, '6-DOF IMU', s.val, s.unit, s.desc, s.range)}
              className="p-3 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg cursor-pointer hover:border-black dark:hover:border-white transition-all"
              title="Click to view sensor channel diagnostics"
            >
              <span className="text-[9px] font-display font-bold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block truncate">
                {s.label}
              </span>
              <div className="text-sm font-mono font-bold text-black dark:text-white mt-1 tabular-nums">
                {s.val} <span className="text-[10px] font-normal text-neutral-500 dark:text-neutral-400">{s.unit}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

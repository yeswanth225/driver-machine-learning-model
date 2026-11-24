import React from 'react';
import { X, ShieldCheck, Activity, AlertTriangle, AlertOctagon, Car } from 'lucide-react';
import { Badge } from './Badge';
import { Button } from '../ui/button';

export function ComponentInspectorModal({ isOpen, onClose, target }) {
  if (!isOpen || !target) return null;

  const renderContent = () => {
    switch (target.type) {
      case 'vehicle_component': {
        const comp = target.data || {};
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-3 pb-3 border-b border-[#dee2e6] dark:border-[#27272a]">
              <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0">
                <Car className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                  {comp.title || 'Vehicle Component'}
                </h3>
                <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
                  {comp.subtitle || 'Kinematic Digital Twin'}
                </span>
              </div>
            </div>

            <p className="text-xs text-neutral-700 dark:text-neutral-300 font-sans leading-relaxed">
              {comp.description}
            </p>

            {comp.metrics && (
              <div className="grid grid-cols-2 gap-2 pt-1">
                {comp.metrics.map((m, idx) => (
                  <div key={idx} className="p-2.5 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg">
                    <span className="text-[9px] font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block">
                      {m.label}
                    </span>
                    <span className="text-xs font-mono font-bold text-black dark:text-white mt-0.5 block tabular-nums">
                      {m.value}
                    </span>
                    {m.hint && <span className="text-[10px] font-sans text-neutral-500 dark:text-neutral-400 mt-0.5 block">{m.hint}</span>}
                  </div>
                ))}
              </div>
            )}

            <div className="p-3 bg-neutral-100 dark:bg-neutral-900 border border-[#dee2e6] dark:border-[#27272a] rounded-lg text-xs font-sans text-neutral-800 dark:text-neutral-200 flex items-start gap-2.5">
              <ShieldCheck className="w-4 h-4 text-black dark:text-white shrink-0 mt-0.5" />
              <div>
                <span className="font-display font-bold uppercase tracking-wider text-black dark:text-white block text-[10px]">
                  Safety Protection Protocol
                </span>
                <span className="text-neutral-600 dark:text-neutral-300 mt-0.5 block">
                  {comp.safetyNote || 'Monitored continuously by the real-time vehicle safety agent.'}
                </span>
              </div>
            </div>
          </div>
        );
      }

      case 'sensor': {
        const sensor = target.data || {};
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-3 pb-3 border-b border-[#dee2e6] dark:border-[#27272a]">
              <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0">
                <Activity className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                  {sensor.name || 'Sensor Channel'}
                </h3>
                <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
                  {sensor.channel || '6-DOF IMU Telemetry'}
                </span>
              </div>
            </div>

            <p className="text-xs text-neutral-700 dark:text-neutral-300 font-sans leading-relaxed">
              {sensor.explanation || 'Measures physical kinematic forces acting on the vehicle chassis in real-time.'}
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <div className="p-2.5 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg">
                <span className="text-[9px] font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block">Current Value</span>
                <span className="text-xs font-mono font-bold text-black dark:text-white mt-0.5 block tabular-nums">{sensor.currentValue ?? '--'}</span>
              </div>
              <div className="p-2.5 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg">
                <span className="text-[9px] font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block">Nominal Range</span>
                <span className="text-xs font-mono font-medium text-neutral-700 dark:text-neutral-300 mt-0.5 block tabular-nums">{sensor.nominalRange || '± 0.25 g'}</span>
              </div>
              <div className="p-2.5 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg">
                <span className="text-[9px] font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block">Sampling Rate</span>
                <span className="text-xs font-mono font-bold text-black dark:text-white mt-0.5 block">{sensor.rate || '10 Hz (100 ms)'}</span>
              </div>
            </div>
          </div>
        );
      }

      case 'alert': {
        const alert = target.data || {};
        const severity = String(alert.severity || 'MEDIUM').toUpperCase();
        const isCritical = severity === 'CRITICAL' || severity === 'HIGH';

        return (
          <div className="space-y-4">
            <div className="flex items-center gap-3 pb-3 border-b border-[#dee2e6] dark:border-[#27272a]">
              <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0">
                {isCritical ? <AlertOctagon className="w-4 h-4 text-red-500" /> : <AlertTriangle className="w-4 h-4 text-amber-500" />}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                    {alert.title || 'Safety Warning'}
                  </h3>
                  <Badge variant={severity} size="sm">{severity}</Badge>
                </div>
                <span className="text-[10px] font-mono text-neutral-500 dark:text-neutral-400">
                  T+{Number(alert.timestamp || 0).toFixed(1)}s // {alert.category || 'VEHICLE DYNAMICS'}
                </span>
              </div>
            </div>

            <div className="p-3 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg space-y-1">
              <span className="text-[9px] font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block">
                Warning Cause
              </span>
              <p className="text-xs text-neutral-800 dark:text-neutral-200 font-sans leading-relaxed">
                {alert.explanation || 'Vehicle motion exceeded physical thresholds under active driving monitoring.'}
              </p>
            </div>

            {alert.evidence && (
              <div className="p-3 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg space-y-2">
                <span className="text-[9px] font-display uppercase tracking-wider text-neutral-500 dark:text-neutral-400 block">
                  Recorded Sensor Evidence
                </span>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  {Object.entries(alert.evidence).map(([k, v]) => (
                    <div key={k} className="p-2 bg-white dark:bg-[#070709] rounded-lg border border-[#dee2e6] dark:border-[#27272a]">
                      <span className="text-neutral-500 dark:text-neutral-400 text-[9px] block uppercase">{k.replace(/_/g, ' ')}</span>
                      <span className="text-black dark:text-white font-bold tabular-nums">{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {alert.safe_driving_advisory && (
              <div className="p-3 bg-neutral-100 dark:bg-neutral-900 border border-[#dee2e6] dark:border-[#27272a] rounded-lg text-xs font-sans text-neutral-800 dark:text-neutral-200 flex items-start gap-2.5">
                <ShieldCheck className="w-4 h-4 text-black dark:text-white shrink-0 mt-0.5" />
                <div>
                  <span className="font-display font-bold uppercase tracking-wider text-black dark:text-white block text-[10px]">
                    Recommended Driver Action
                  </span>
                  <p className="text-neutral-600 dark:text-neutral-300 mt-1 leading-relaxed">
                    {alert.safe_driving_advisory}
                  </p>
                </div>
              </div>
            )}
          </div>
        );
      }

      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-lg bg-white dark:bg-[#0c0c0e] border border-[#dee2e6] dark:border-[#27272a] rounded-xl p-6 overflow-hidden text-black dark:text-white shadow-2xl">
        {/* Close Button */}
        <Button
          variant="outline"
          size="icon"
          onClick={onClose}
          className="absolute top-4 right-4 h-7 w-7 rounded-md"
          title="Close inspection"
          aria-label="Close inspection"
        >
          <X className="w-4 h-4" />
        </Button>

        {renderContent()}

        <div className="mt-5 pt-3 border-t border-[#dee2e6] dark:border-[#27272a] flex items-center justify-between text-[10px] font-display uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
          <span>IDAMS // SAFETY INTELLIGENCE</span>
          <Button
            variant="outline"
            size="sm"
            onClick={onClose}
            className="text-[10px] h-7 px-3 rounded-md"
          >
            Dismiss
          </Button>
        </div>
      </div>
    </div>
  );
}

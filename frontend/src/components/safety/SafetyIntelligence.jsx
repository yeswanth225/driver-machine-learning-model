import React, { useState } from 'react';
import { ShieldAlert, Activity, GitBranch, CheckCircle2 } from 'lucide-react';
import { RiskGauge } from '../common/RiskGauge';
import { Badge } from '../common/Badge';
import { AlertCard } from '../common/AlertCard';
import { ComponentInspectorModal } from '../common/ComponentInspectorModal';

export function SafetyIntelligence({ assessment }) {
  const [inspectorTarget, setInspectorTarget] = useState(null);

  const safetyState = assessment?.safety_state || 'NORMAL';
  const riskScore = assessment?.risk_score ?? 0.0;
  const activeAlerts = assessment?.active_alerts || [];
  const activeIncidents = assessment?.active_incidents || [];
  const rules = assessment?.triggered_rules || [];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto bg-transparent text-black dark:text-white">
      {/* Component Inspector Modal */}
      <ComponentInspectorModal
        isOpen={!!inspectorTarget}
        onClose={() => setInspectorTarget(null)}
        target={inspectorTarget}
      />

      {/* Top Header Card */}
      <div className="spacex-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0">
              <ShieldAlert className="w-4 h-4" />
            </div>
            <h2 className="text-sm sm:text-base font-display font-bold uppercase tracking-spacex text-black dark:text-white">
              Safety Intelligence &amp; Risk Engine
            </h2>
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1 font-sans">
            How the autonomous safety agent evaluates real-time driving hazard risk, tracks active incidents, and ensures fail-safe passenger protection.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant={safetyState} size="lg">{safetyState}</Badge>
          <span className="font-mono text-sm text-black dark:text-white font-bold tabular-nums">
            {Number(riskScore).toFixed(1)} / 100.0 pts
          </span>
        </div>
      </div>

      {/* Risk Engine & Active Lifecycle Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: How Safety Risk is Evaluated (5 cols) */}
        <div className="lg:col-span-5 spacex-card space-y-4">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
            <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-black dark:text-white" />
              <span>Real-Time Risk Computation</span>
            </h3>
            <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">Deterministic Engine</span>
          </div>

          <div className="flex justify-center py-2">
            <RiskGauge score={riskScore} state={safetyState} size={150} strokeWidth={8} />
          </div>

          <div className="space-y-2 text-xs font-sans">
            <div className="p-3 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg">
              <span className="text-[9px] text-neutral-500 dark:text-neutral-400 uppercase font-display font-bold tracking-spacex block">
                Deterministic Policy D-13
              </span>
              <p className="text-neutral-700 dark:text-neutral-300 text-xs mt-1 leading-relaxed">
                Risk points accumulate on sudden kinematic triggers and multiply when compounding with an aggressive driving style.
              </p>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                <span className="text-neutral-600 dark:text-neutral-400">Sudden Braking:</span>
                <span className="text-black dark:text-white font-bold">50 pts (Primary Impact)</span>
              </div>
              <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                <span className="text-neutral-600 dark:text-neutral-400">Sharp Swerves:</span>
                <span className="text-black dark:text-white font-bold">35 pts (Cornering Force)</span>
              </div>
              <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                <span className="text-neutral-600 dark:text-neutral-400">Sudden Acceleration:</span>
                <span className="text-black dark:text-white font-bold">25 pts (Throttle Spike)</span>
              </div>
              <div className="flex justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-1">
                <span className="text-neutral-600 dark:text-neutral-400">Aggressive Style:</span>
                <span className="text-black dark:text-white font-bold">+25 pts Addition</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Compound Hazard:</span>
                <span className="text-black dark:text-white font-bold">1.5× Multiplier</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Active Incidents Lifecycle (7 cols) */}
        <div className="lg:col-span-7 spacex-card space-y-4">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
            <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white flex items-center gap-2">
              <GitBranch className="w-3.5 h-3.5 text-black dark:text-white" />
              <span>Active Driving Incidents Lifecycle</span>
            </h3>
            <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
              {activeIncidents.length} active incidents
            </span>
          </div>

          {activeIncidents.length === 0 ? (
            <div className="p-8 text-center bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg text-neutral-500 dark:text-neutral-400 font-display text-xs">
              <CheckCircle2 className="w-8 h-8 text-black dark:text-white mx-auto mb-2" />
              <div className="font-bold uppercase tracking-spacex text-black dark:text-white">
                No Active Safety Incidents
              </div>
              <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1 font-sans">
                The safety lifecycle verifies events over multiple frames. No persistent hazards are currently tracked.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {activeIncidents.map((inc, i) => (
                <div key={i} className="p-3 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg space-y-1 font-display text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold uppercase tracking-wider text-black dark:text-white">{inc.incident_type || inc.type}</span>
                    <Badge variant={inc.severity || 'WARNING'} size="sm">{inc.severity || 'WARNING'}</Badge>
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400 font-sans">
                    {inc.description || 'Elevated vehicle dynamics incident under active monitoring.'}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Triggered Guardrail Rules */}
          <div className="pt-2">
            <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400 block mb-2">
              Verified Guardrail Rules
            </span>
            <div className="flex flex-wrap gap-2">
              {rules.map((rule, idx) => (
                <span
                  key={idx}
                  className="px-2.5 py-1 bg-[#f8f9fa] dark:bg-[#121215] border border-[#dee2e6] dark:border-[#27272a] rounded-lg text-[10px] font-mono uppercase tracking-wider text-black dark:text-white"
                >
                  {rule}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Active Driving Alerts Feed */}
      {activeAlerts.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] pb-2">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-black dark:text-white" />
              <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
                Active Driving Alerts ({activeAlerts.length})
              </h3>
            </div>
            <span className="text-[10px] font-mono uppercase text-neutral-500 dark:text-neutral-400">
              Click alert card for evidence breakdown
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {activeAlerts.map((alert, idx) => (
              <div
                key={alert.alert_id || idx}
                onClick={() => setInspectorTarget({ type: 'alert', data: alert })}
                className="cursor-pointer group transition-transform hover:-translate-y-0.5"
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

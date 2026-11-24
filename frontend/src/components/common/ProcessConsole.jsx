import React, { useState, useEffect, useRef } from 'react';
import {
  Terminal,
  Play,
  Pause,
  Trash2,
} from 'lucide-react';
import { Badge } from './Badge';
import { Button } from '../ui/button';

export function ProcessConsole({
  safetyState = 'NORMAL',
  drivingStyle = 'Normal',
  maneuver = 'None / Nominal',
  riskScore = 0.0,
  sensorHealth = 'HEALTHY',
  isLive = false,
  className = '',
}) {
  const [isPaused, setIsPaused] = useState(false);
  const [filter, setFilter] = useState('ALL'); // ALL | EVENTS | ALERTS | SENSORS
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    if (isPaused) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
    const isAggressive = String(drivingStyle).toLowerCase().includes('aggressive');
    const isEventActive = maneuver && !maneuver.includes('None') && !maneuver.includes('Nominal');

    const newLogs = [];

    // Base telemetry reading
    newLogs.push({
      id: `${Date.now()}-1`,
      time: timeStr,
      category: 'SENSORS',
      message: `Sampling 6-DOF IMU motion channels: vehicle acceleration and turn rates within stable limits.`,
      type: 'info',
    });

    // Driving Behavior check
    newLogs.push({
      id: `${Date.now()}-2`,
      time: timeStr,
      category: 'EVENTS',
      message: isAggressive
        ? `Driving Behavior Analysis: Aggressive driving pattern identified. Heightened monitoring sensitivity engaged.`
        : `Driving Behavior Analysis: Smooth driving pattern confirmed. Quiescent driving stability maintained.`,
      type: isAggressive ? 'warning' : 'success',
    });

    // Sudden Event detection
    if (isEventActive) {
      newLogs.push({
        id: `${Date.now()}-3`,
        time: timeStr,
        category: 'EVENTS',
        message: `Sudden Vehicle Event: ${maneuver} detected by motion sensors. Evaluating passenger hazard risk.`,
        type: 'warning',
      });
    }

    // Safety Risk Engine assessment
    if (riskScore > 30.0 || safetyState !== 'NORMAL') {
      newLogs.push({
        id: `${Date.now()}-4`,
        time: timeStr,
        category: 'ALERTS',
        message: `Safety Risk Engine: Risk Index evaluated at ${Number(riskScore).toFixed(1)} / 100 pts. Safety state set to ${safetyState}.`,
        type: 'danger',
      });
    }

    setLogs((prev) => [...newLogs, ...prev].slice(0, 50));
  }, [safetyState, drivingStyle, maneuver, riskScore, isPaused]);

  const handleClearLogs = () => {
    setLogs([]);
  };

  const filteredLogs = logs.filter((log) => {
    if (filter === 'ALL') return true;
    return log.category === filter;
  });

  return (
    <div className={`bg-white dark:bg-[#0c0c0e] border border-[#dee2e6] dark:border-[#27272a] rounded-sm overflow-hidden font-sans ${className}`}>
      {/* Console Header Bar */}
      <div className="px-4 py-2.5 bg-[#f8f9fa] dark:bg-[#121215] border-b border-[#dee2e6] dark:border-[#27272a] flex flex-wrap items-center justify-between gap-3 select-none">
        <div className="flex items-center gap-2.5">
          <Terminal className="w-3.5 h-3.5 text-black dark:text-white" />
          <h3 className="text-xs font-display font-bold uppercase tracking-spacex text-black dark:text-white">
            Telemetry Event Console
          </h3>
          <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-900 text-neutral-800 dark:text-neutral-200">
            D-13 Guardrails Active
          </span>
        </div>

        {/* Console Controls */}
        <div className="flex items-center gap-2">
          {/* Filter Pills */}
          <div className="hidden sm:flex items-center gap-1 bg-neutral-200 dark:bg-neutral-800 p-0.5 rounded-lg text-[10px] font-display uppercase tracking-wider">
            {['ALL', 'EVENTS', 'ALERTS', 'SENSORS'].map((cat) => (
              <Button
                key={cat}
                variant={filter === cat ? "default" : "ghost"}
                size="sm"
                onClick={() => setFilter(cat)}
                className={`h-6 px-2 text-[10px] rounded-md transition-all ${
                  filter === cat ? 'font-bold shadow-xs' : 'text-neutral-600 dark:text-neutral-400'
                }`}
              >
                {cat}
              </Button>
            ))}
          </div>

          <Button
            variant="outline"
            size="icon"
            onClick={() => setIsPaused(!isPaused)}
            title={isPaused ? 'Resume live console feed' : 'Pause live console feed'}
            aria-label={isPaused ? 'Resume live console feed' : 'Pause live console feed'}
            className="h-7 w-7 rounded-md"
          >
            {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
          </Button>

          <Button
            variant="outline"
            size="icon"
            onClick={handleClearLogs}
            title="Clear console entries"
            aria-label="Clear console entries"
            className="h-7 w-7 rounded-md"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {/* Console Log Feed */}
      <div className="h-44 overflow-y-auto p-3 font-mono text-xs space-y-1.5 divide-y divide-[#f1f3f5] dark:divide-[#18181c]">
        {filteredLogs.length === 0 ? (
          <div className="text-neutral-400 dark:text-neutral-600 py-6 text-center text-xs uppercase tracking-wider">
            No telemetry log entries matching filter
          </div>
        ) : (
          filteredLogs.map((log) => (
            <div key={log.id} className="pt-1.5 flex items-start gap-2 text-[11px] leading-relaxed">
              <span className="text-neutral-500 dark:text-neutral-500 shrink-0 select-none">
                [{log.time}]
              </span>
              <span className={`px-1 py-0.2 rounded-lg font-mono uppercase text-[9px] shrink-0 border ${
                log.type === 'danger'
                  ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300'
                  : log.type === 'warning'
                  ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300'
                  : 'border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-900 text-neutral-700 dark:text-neutral-300'
              }`}>
                {log.category}
              </span>
              <span className="text-neutral-800 dark:text-neutral-200">
                {log.message}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

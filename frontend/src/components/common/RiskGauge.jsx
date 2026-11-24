import React from 'react';
import { Badge } from './Badge';

export function RiskGauge({ score = 0, state = 'NORMAL', size = 180, strokeWidth = 8 }) {
  const isInsufficient = state === 'INSUFFICIENT_DATA' || score === null || score === undefined;
  const clampedScore = isInsufficient ? 0 : Math.max(0, Math.min(100, Number(score) || 0));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  // 270 degree aerospace arc
  const arcLength = circumference * 0.75;
  const strokeDashoffset = isInsufficient ? arcLength : arcLength - (arcLength * (clampedScore / 100));

  // High-contrast aerospace stroke styling
  let strokeColor = 'currentColor';
  let strokeClass = 'text-black dark:text-white';
  if (isInsufficient) {
    strokeColor = 'currentColor';
    strokeClass = 'text-neutral-400 dark:text-neutral-600 opacity-30';
  } else if (clampedScore >= 75 || state === 'CRITICAL' || state === 'HIGH_RISK') {
    strokeColor = '#ef4444';
    strokeClass = '';
  } else if (clampedScore >= 50 || state === 'WARNING') {
    strokeColor = '#f97316';
    strokeClass = '';
  } else if (clampedScore >= 25 || state === 'CAUTION') {
    strokeColor = '#eab308';
    strokeClass = '';
  }

  return (
    <div className="relative flex flex-col items-center justify-center select-none" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        className="rotate-[135deg] transform origin-center"
      >
        {/* Aerospace subtle track background */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          className="text-neutral-200 dark:text-neutral-800"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${arcLength} ${circumference}`}
          fill="none"
        />
        {/* Active Telemetry Ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          className={strokeClass}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          fill="none"
          style={{
            transition: 'stroke-dashoffset 0.4s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.2s ease',
          }}
        />
      </svg>

      {/* Center Readout in Industrial D-DIN Monospace */}
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-4xl font-display font-bold tracking-tight text-black dark:text-white tabular-nums leading-none">
          {isInsufficient ? '--' : clampedScore.toFixed(1)}
        </span>
        <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400 mt-1">
          {isInsufficient ? 'No Telemetry' : 'Risk Index'}
        </span>
        <div className="mt-2">
          <Badge variant={state} size="sm">
            {state}
          </Badge>
        </div>
      </div>
    </div>
  );
}

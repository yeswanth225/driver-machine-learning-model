import React from 'react';
import { AlertTriangle, AlertOctagon, Info } from 'lucide-react';
import { Badge } from './Badge';

export function AlertCard({ alert }) {
  if (!alert) return null;

  const severity = String(alert.severity || 'MEDIUM').toUpperCase();
  const isCritical = severity === 'CRITICAL' || severity === 'HIGH';
  const isCaution = severity === 'MEDIUM';

  return (
    <div className="bg-white dark:bg-[#0c0c0e] border border-[#dee2e6] dark:border-[#27272a] rounded-xl p-4 transition-all">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 border border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white">
            {isCritical ? (
              <AlertOctagon className="w-4 h-4 text-red-600 dark:text-red-400" />
            ) : isCaution ? (
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            ) : (
              <Info className="w-4 h-4 text-neutral-600 dark:text-neutral-400" />
            )}
          </div>
          <div className="min-w-0">
            <h4 className="text-xs sm:text-sm font-display font-bold uppercase tracking-spacex-tight text-black dark:text-white leading-snug truncate">
              {alert.title}
            </h4>
            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
              <Badge variant={severity} size="sm">{severity}</Badge>
              <span className="text-[10px] font-display font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                {alert.category || 'SAFETY'}
              </span>
              <span className="text-[10px] font-mono text-neutral-500 dark:text-neutral-400 tabular-nums">
                T+{Number(alert.timestamp || 0).toFixed(1)}s
              </span>
            </div>
          </div>
        </div>
      </div>

      {alert.explanation && (
        <p className="mt-2.5 text-xs text-neutral-700 dark:text-neutral-300 leading-relaxed font-sans">
          {alert.explanation}
        </p>
      )}

      {alert.safe_driving_advisory && (
        <div className="mt-3 pt-2.5 border-t border-[#dee2e6] dark:border-[#27272a] bg-[#f8f9fa] dark:bg-[#121215] p-2.5 rounded-lg flex items-start gap-2 text-xs">
          <span className="font-display font-bold uppercase tracking-wider text-black dark:text-white shrink-0 text-[10px]">
            Advisory:
          </span>
          <p className="text-neutral-600 dark:text-neutral-300 text-xs">
            {alert.safe_driving_advisory}
          </p>
        </div>
      )}
    </div>
  );
}

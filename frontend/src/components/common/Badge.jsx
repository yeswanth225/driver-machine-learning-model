import React from 'react';

export function Badge({ variant = 'default', children, className = '', size = 'md' }) {
  const baseClasses = 'inline-flex items-center font-display font-bold uppercase tracking-wider transition-colors select-none rounded-md border';
  
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-[10px]',
    md: 'px-2.5 py-0.5 text-[11px]',
    lg: 'px-3 py-1 text-xs',
  }[size] || 'px-2.5 py-0.5 text-[11px]';

  const variantMap = {
    // Safety States
    NORMAL: 'bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white border-neutral-300 dark:border-neutral-700',
    CAUTION: 'bg-amber-50 dark:bg-amber-950/40 text-amber-900 dark:text-amber-300 border-amber-300 dark:border-amber-700/60',
    WARNING: 'bg-red-50 dark:bg-red-950/40 text-red-900 dark:text-red-300 border-red-300 dark:border-red-700/60',
    CRITICAL: 'bg-black dark:bg-white text-white dark:text-black border-black dark:border-white font-extrabold',
    HIGH_RISK: 'bg-black dark:bg-white text-white dark:text-black border-black dark:border-white font-extrabold',
    INSUFFICIENT_DATA: 'bg-neutral-100 dark:bg-neutral-900 text-neutral-500 dark:text-neutral-400 border-neutral-300 dark:border-neutral-700',

    // Pipeline Health States
    HEALTHY: 'bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white border-neutral-300 dark:border-neutral-700',
    DEGRADED: 'bg-amber-50 dark:bg-amber-950/40 text-amber-900 dark:text-amber-300 border-amber-300 dark:border-amber-700/60',
    MISSING: 'bg-red-50 dark:bg-red-950/40 text-red-900 dark:text-red-300 border-red-300 dark:border-red-700/60',
    INVALID: 'bg-neutral-100 dark:bg-neutral-900 text-neutral-500 dark:text-neutral-400 border-neutral-300 dark:border-neutral-700',
    QUIESCENT_BYPASS: 'bg-neutral-100 dark:bg-neutral-900 text-neutral-500 dark:text-neutral-400 border-neutral-300 dark:border-neutral-700',

    // Driving Styles & Maneuvers
    AGGRESSIVE: 'bg-amber-50 dark:bg-amber-950/40 text-amber-900 dark:text-amber-300 border-amber-300 dark:border-amber-700/60',
    SA: 'bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white border-neutral-300 dark:border-neutral-700',
    SD: 'bg-red-50 dark:bg-red-950/40 text-red-900 dark:text-red-300 border-red-300 dark:border-red-700/60',
    SRT: 'bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white border-neutral-300 dark:border-neutral-700',
    SLT: 'bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white border-neutral-300 dark:border-neutral-700',

    // Fallbacks
    default: 'bg-neutral-100 dark:bg-neutral-900 text-black dark:text-white border-neutral-300 dark:border-neutral-700',
  };

  const key = String(variant).toUpperCase();
  const classes = variantMap[key] || variantMap.default;

  return (
    <span className={`${baseClasses} ${sizeClasses} ${classes} ${className}`}>
      {children}
    </span>
  );
}

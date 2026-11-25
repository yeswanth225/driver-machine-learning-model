import React from 'react';
import { Cpu, RotateCcw, Menu } from 'lucide-react';
import { Badge } from '../common/Badge';
import { SwitchButton } from '../common/SwitchButton';
import { Button } from '@/components/ui/button';

export function Header({
  activeTab = 'dashboard',
  healthStatus = 'Connected',
  modelStatus = null,
  safetyState = 'NORMAL',
  onResetAgent,
  isResetting = false,
  onToggleSidebar,
}) {
  const isHealthy = healthStatus === 'Connected' || healthStatus === 'healthy';
  const allModelsVerified = modelStatus?.all_verified;

  const tabTitles = {
    dashboard: 'Dashboard',
    live_monitoring: 'Live Monitoring',
    driving_behaviour: 'Driving Behavior',
    driving_behavior: 'Driving Behavior',
    safety_alerts: 'Safety Alerts',
    trip_history: 'Trip History',
    driving_insights: 'Driving Insights',
    settings: 'System Settings',
  };

  const tabSubtitles = {
    dashboard: 'Mission Control & Real-Time Safety',
    live_monitoring: '6-DOF IMU Telemetry Stream',
    driving_behaviour: 'Kinematics & Maneuver Analysis',
    driving_behavior: 'Kinematics & Maneuver Analysis',
    safety_alerts: 'Deterministic Risk & Incident Lifecycle',
    trip_history: 'Multi-Driver Replay Studio',
    driving_insights: 'Model Benchmarks & Generalization',
    settings: 'Model Verification & Hardware Diagnostics',
  };

  const currentTitle = tabTitles[activeTab] || 'Dashboard';
  const currentSubtitle = tabSubtitles[activeTab] || 'Autonomous Safety System';

  return (
    <header className="sticky top-0 z-30 select-none bg-white/95 dark:bg-[#070709]/95 backdrop-blur-sm border-b border-[#dee2e6] dark:border-[#27272a] transition-colors">
      <div className="h-14 px-4 sm:px-6 flex items-center justify-between gap-4">
        {/* Left: Mobile Navigation Trigger, View Title & Context */}
        <div className="flex items-center gap-3 min-w-0">
          {onToggleSidebar && (
            <Button
              variant="outline"
              size="icon"
              onClick={onToggleSidebar}
              className="lg:hidden h-8 w-8 text-neutral-600 dark:text-neutral-300 border-[#dee2e6] dark:border-[#27272a] rounded-lg shrink-0"
              aria-label="Toggle Navigation"
            >
              <Menu className="w-4 h-4" />
            </Button>
          )}

          <div className="flex items-center gap-3 min-w-0">
            <h1 className="text-sm sm:text-base font-semibold text-black dark:text-white tracking-tight truncate">
              {currentTitle}
            </h1>
            <Badge variant={safetyState} size="sm">
              {safetyState}
            </Badge>
            <span className="hidden xl:inline text-neutral-300 dark:text-neutral-700">|</span>
            <span className="hidden xl:inline text-xs text-neutral-500 dark:text-neutral-400 font-normal truncate">
              {currentSubtitle}
            </span>
          </div>
        </div>

        {/* Right: Telemetry Health, Models Status, Theme Toggle & Actions */}
        <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
          {/* Models Status Pill */}
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-neutral-100 dark:bg-neutral-900 border border-[#dee2e6] dark:border-[#27272a] text-xs">
            <Cpu className="w-3.5 h-3.5 text-neutral-500 dark:text-neutral-400 shrink-0" />
            <span className="text-neutral-500 dark:text-neutral-400">Models:</span>
            <span className="font-mono font-medium text-black dark:text-white">
              {allModelsVerified ? '4/4 Verified' : 'Verifying...'}
            </span>
          </div>

          {/* System Online / Health Pill */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-neutral-100 dark:bg-neutral-900 border border-[#dee2e6] dark:border-[#27272a] text-xs font-mono">
            {isHealthy ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block animate-pulse" />
                <span className="font-medium text-emerald-700 dark:text-emerald-400">Online</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
                <span className="font-medium text-red-600 dark:text-red-400">Offline</span>
              </>
            )}
          </div>

          <div className="h-4 w-px bg-[#dee2e6] dark:bg-[#27272a] hidden sm:block" />

          {/* Theme Switcher Button */}
          <SwitchButton size="sm" showLabel={false} />

          {/* Reset Agent Button */}
          <Button
            variant="outline"
            onClick={onResetAgent}
            disabled={isResetting}
            className="text-xs h-8 px-3 rounded-lg gap-1.5 text-neutral-700 dark:text-neutral-200 border-[#dee2e6] dark:border-[#27272a] hover:bg-neutral-100 dark:hover:bg-neutral-900"
            title="Reset Agent state"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin' : ''}`} />
            <span className="hidden md:inline">Reset Agent</span>
          </Button>
        </div>
      </div>
    </header>
  );
}

export default Header;

import React from 'react';
import { Cpu, RotateCcw, Menu } from 'lucide-react';
import { Badge } from '../common/Badge';
import { SwitchButton } from '../common/SwitchButton';
import { Button } from '@/components/ui/button';

export function Header({
  activeTab = 'dashboard',
  healthStatus = 'Checking...',
  modelStatus = null,
  safetyState = 'NORMAL',
  onResetAgent,
  isResetting = false,
  onToggleSidebar,
}) {
  const isHealthy = healthStatus === 'Connected' || healthStatus === 'healthy';
  const allModelsVerified = modelStatus?.all_verified;

  const tabTitles = {
    dashboard: 'Dashboard // Mission Control',
    live_monitoring: 'Live Monitoring // Telemetry Stream',
    driving_behaviour: 'Driving Behavior // 6-DOF IMU & Kinematics',
    driving_behavior: 'Driving Behavior // 6-DOF IMU & Kinematics',
    safety_alerts: 'Safety Alerts // Risk & Incident Lifecycle',
    trip_history: 'Trip History // Multi-Driver Replay Studio',
    driving_insights: 'Driving Insights // Model Performance Benchmarks',
    settings: 'System Settings // Model Verification & Diagnostics',
  };

  const currentTitle = tabTitles[activeTab] || 'Driver Safety Agent';

  return (
    <header className="sticky top-0 z-30 select-none">
      {/* Tier 1: Aerospace Global Telemetry Strip (40px, pure black #000000, 11px uppercase tracking) */}
      <div className="h-10 px-6 bg-[#000000] text-white border-b border-[#27272a] flex items-center justify-between text-[11px] font-display uppercase tracking-spacex">
        <div className="flex items-center gap-4">
          {onToggleSidebar && (
            <Button
              variant="outline"
              size="icon"
              onClick={onToggleSidebar}
              className="lg:hidden h-7 w-7 text-neutral-400 hover:text-white border-neutral-700 hover:border-neutral-500 rounded-md"
              aria-label="Toggle Navigation"
            >
              <Menu className="w-4 h-4" />
            </Button>
          )}

          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-white rounded-full inline-block" />
            <span className="text-white font-bold tracking-spacex">IDAMS // Autonomous Safety</span>
            <span className="text-neutral-600 hidden sm:inline">|</span>
            <span className="text-neutral-400 hidden sm:inline">Edge ML &amp; 6-DOF Kinematics</span>
          </div>
        </div>

        {/* Global Telemetry Status Indicators */}
        <div className="flex items-center gap-4 text-[11px]">
          <div className="hidden md:flex items-center gap-1.5 text-neutral-400">
            <Cpu className="w-3.5 h-3.5 text-neutral-300" />
            <span>Models:</span>
            <span className="text-white font-mono font-medium">
              {allModelsVerified ? 'Verified [4/4]' : 'Verifying...'}
            </span>
          </div>

          <div className="flex items-center gap-1.5 text-neutral-300 font-mono">
            {isHealthy ? (
              <>
                <span className="w-2 h-2 rounded-full bg-[#10b981] inline-block animate-pulse" />
                <span className="text-white font-medium">Online</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-[#ef4444] inline-block" />
                <span className="text-[#ef4444] font-medium">Offline</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Tier 2: Sub-Nav Bar (48px, aerospace hairline border, pure theme-aware contrast) */}
      <div className="h-12 px-6 bg-white dark:bg-[#070709] border-b border-[#dee2e6] dark:border-[#27272a] flex items-center justify-between transition-colors">
        <div className="flex items-center gap-3 min-w-0">
          <h2 className="text-sm sm:text-base font-display font-bold uppercase tracking-spacex text-black dark:text-white leading-none truncate">
            {currentTitle}
          </h2>
          <Badge variant={safetyState} size="sm">
            {safetyState}
          </Badge>
        </div>

        <div className="flex items-center gap-3">
          {/* Top-Right Theme Switcher Button */}
          <SwitchButton size="sm" showLabel={true} />

          {/* Reset System Button */}
          <Button
            variant="outline"
            onClick={onResetAgent}
            disabled={isResetting}
            className="text-[11px] h-8 px-3 rounded-lg gap-1.5"
            title="Reset Agent state"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin' : ''}`} />
            <span>Reset Agent</span>
          </Button>
        </div>
      </div>
    </header>
  );
}

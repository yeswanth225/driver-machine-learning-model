import React from 'react';
import {
  LayoutDashboard,
  Radio,
  Car,
  ShieldAlert,
  Film,
  BarChart3,
  Settings,
  RotateCcw,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Sidebar({
  activeTab = 'dashboard',
  onSelectTab,
  scenarios = [],
  activeScenarioId = 'normal',
  onSelectScenarioPreset,
  onResetAgent,
  isResetting = false,
  isCollapsed = false,
  onToggleCollapse,
}) {
  // Navigation strictly matching Section 1 requirements
  const mainNavItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'live_monitoring', label: 'Live Monitoring', icon: Radio },
    { id: 'driving_behaviour', label: 'Driving Behavior', icon: Car },
    { id: 'safety_alerts', label: 'Safety Alerts', icon: ShieldAlert },
    { id: 'trip_history', label: 'Trip History', icon: Film },
    { id: 'driving_insights', label: 'Driving Insights', icon: BarChart3 },
  ];

  // Aerospace scenario presets for quick testing
  const scenarioPresets = [
    {
      id: 'normal',
      name: 'Normal Cruising',
      badge: 'NOMINAL',
      badgeColor: 'border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-900 text-neutral-800 dark:text-neutral-200',
      description: 'Quiescent vehicle dynamics within verified highway tolerances.',
      icon: CheckCircle2,
      payload: {
        timestamp: 1.0,
        style_prediction: 0,
        style_confidence: 0.96,
        maneuver_prediction: 0,
        maneuver_confidence: 0.98,
        pipeline_a_health: 'HEALTHY',
        pipeline_b_health: 'HEALTHY',
      },
    },
    {
      id: 'caution',
      name: 'Aggressive Weaving',
      badge: 'CAUTION',
      badgeColor: 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300',
      description: 'Elevated lateral acceleration and high steering variance.',
      icon: AlertTriangle,
      payload: {
        timestamp: 1.0,
        style_prediction: 1,
        style_confidence: 0.94,
        maneuver_prediction: 0,
        maneuver_confidence: 0.85,
        pipeline_a_health: 'HEALTHY',
        pipeline_b_health: 'HEALTHY',
      },
    },
    {
      id: 'warning',
      name: 'Sudden Deceleration',
      badge: 'WARNING',
      badgeColor: 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950/40 text-red-800 dark:text-red-300',
      description: 'Emergency braking exceeding -4.0 m/s² hard safety threshold.',
      icon: AlertOctagon,
      payload: {
        timestamp: 1.0,
        style_prediction: 0,
        style_confidence: 0.82,
        maneuver_prediction: 2,
        maneuver_confidence: 0.97,
        pipeline_a_health: 'HEALTHY',
        pipeline_b_health: 'HEALTHY',
      },
    },
  ];

  return (
    <aside
      className={`h-screen sticky top-0 bg-white dark:bg-[#070709] border-r border-[#dee2e6] dark:border-[#27272a] flex flex-col justify-between select-none transition-all duration-150 z-40 shrink-0 ${
        isCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Top Header & Navigation Items */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {/* Brand / Logo Header */}
        <div className="h-14 px-4 flex items-center justify-between border-b border-[#dee2e6] dark:border-[#27272a] bg-white dark:bg-[#070709] transition-colors">
          {!isCollapsed && (
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0 shadow-xs">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div className="min-w-0">
                <span className="text-sm font-semibold tracking-tight text-black dark:text-white truncate block leading-tight">
                  IDAMS
                </span>
                <span className="text-[11px] font-medium text-neutral-500 dark:text-neutral-400 truncate block leading-none">
                  Autonomous Safety
                </span>
              </div>
            </div>
          )}

          {isCollapsed && (
            <div className="mx-auto w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center shrink-0 shadow-xs">
              <ShieldCheck className="w-4 h-4" />
            </div>
          )}

          <Button
            variant="outline"
            size="icon"
            onClick={onToggleCollapse}
            className="hidden lg:inline-flex h-8 w-8 rounded-lg text-neutral-500 hover:text-black dark:hover:text-white border-[#dee2e6] dark:border-[#27272a]"
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </Button>
        </div>

        {/* Section 1: Main 6 Navigation Items */}
        <div className="p-3 space-y-1">
          {!isCollapsed && (
            <div className="px-3 py-1.5 text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
              Navigation
            </div>
          )}

          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <Button
                key={item.id}
                variant={isActive ? "default" : "outline"}
                onClick={() => onSelectTab(item.id)}
                title={isCollapsed ? item.label : undefined}
                className={`w-full justify-start gap-3 h-10 px-3 rounded-lg text-[12px] font-display uppercase tracking-spacex transition-all ${
                  isActive
                    ? 'font-bold shadow-xs'
                    : 'font-medium hover:border-neutral-300 dark:hover:border-neutral-700'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-white dark:text-black' : 'text-neutral-500 dark:text-neutral-400'}`} />
                {!isCollapsed && (
                  <span className="truncate">{item.label}</span>
                )}
              </Button>
            );
          })}
        </div>

        {/* Section 2: Test Profiles (Simulation Presets) */}
        <div className="px-3 pt-3 pb-2 border-t border-[#dee2e6] dark:border-[#27272a]">
          {!isCollapsed && (
            <div className="flex items-center justify-between px-3 mb-2">
              <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-neutral-500 dark:text-neutral-400">
                Simulation Profiles
              </span>
            </div>
          )}

          <div className="space-y-1.5">
            {scenarioPresets.map((preset) => {
              const Icon = preset.icon;
              const isSelected = activeScenarioId === preset.id;
              return (
                <Button
                  key={preset.id}
                  variant={isSelected ? "default" : "outline"}
                  onClick={() => onSelectScenarioPreset(preset)}
                  title={isCollapsed ? `${preset.name} (${preset.badge}): ${preset.description}` : preset.description}
                  className={`w-full text-left justify-between h-auto p-2.5 rounded-lg transition-all ${
                    isSelected ? 'shadow-xs' : ''
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-2 min-w-0">
                      <Icon className="w-3.5 h-3.5 shrink-0" />
                      {!isCollapsed && (
                        <span className={`text-[11px] font-display uppercase tracking-wider truncate ${isSelected ? 'font-bold' : 'font-medium'}`}>
                          {preset.name}
                        </span>
                      )}
                    </div>
                    {!isCollapsed && (
                      <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded-md border ${preset.badgeColor}`}>
                        {preset.badge}
                      </span>
                    )}
                  </div>
                </Button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Section 3: Placed separately at the very bottom of the sidebar */}
      <div className="p-3 border-t border-[#dee2e6] dark:border-[#27272a] bg-[#f8f9fa] dark:bg-[#0d0d10] space-y-2">
        {/* System Settings Button (Separate at Bottom) */}
        <Button
          variant={activeTab === 'settings' ? "default" : "outline"}
          onClick={() => onSelectTab('settings')}
          title={isCollapsed ? 'System Settings' : undefined}
          className={`w-full justify-start gap-3 h-10 px-3 rounded-lg text-[12px] font-display uppercase tracking-spacex transition-all ${
            activeTab === 'settings' ? 'font-bold shadow-xs' : 'font-medium'
          }`}
        >
          <Settings className="w-4 h-4 shrink-0" />
          {!isCollapsed && <span className="truncate">System Settings</span>}
        </Button>

        {/* Reset System Trigger */}
        <Button
          variant="outline"
          onClick={onResetAgent}
          disabled={isResetting}
          title={isCollapsed ? 'Reset Safety Agent State' : undefined}
          className="w-full justify-center gap-2 text-[11px] h-9 rounded-lg"
        >
          <RotateCcw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin' : ''}`} />
          {!isCollapsed && <span>Reset System</span>}
        </Button>
      </div>
    </aside>
  );
}

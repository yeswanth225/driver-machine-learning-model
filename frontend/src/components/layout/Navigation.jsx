import React from 'react';
import { LayoutDashboard, Radio, PlaySquare, ShieldAlert, BarChart3, Wrench } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Navigation({ activeTab = 'command', onSelectTab }) {
  const tabs = [
    { id: 'command', label: 'Command Center', icon: <LayoutDashboard className="w-3.5 h-3.5" /> },
    { id: 'live', label: 'Live Stream', icon: <Radio className="w-3.5 h-3.5" /> },
    { id: 'replay', label: 'Replay Studio', icon: <PlaySquare className="w-3.5 h-3.5" /> },
    { id: 'safety', label: 'Safety Intelligence', icon: <ShieldAlert className="w-3.5 h-3.5" /> },
    { id: 'analytics', label: 'Model Metrics', icon: <BarChart3 className="w-3.5 h-3.5" /> },
    { id: 'diagnostics', label: 'Diagnostics', icon: <Wrench className="w-3.5 h-3.5" /> },
  ];

  return (
    <nav className="bg-white dark:bg-black border-b border-[#dee2e6] dark:border-[#27272a] px-6 select-none font-mono">
      <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-2">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <Button
              key={tab.id}
              variant={isActive ? "default" : "outline"}
              onClick={() => onSelectTab(tab.id)}
              className={`flex items-center gap-2 px-3.5 h-8 text-xs uppercase tracking-wider transition-all whitespace-nowrap rounded-lg ${
                isActive ? 'font-semibold shadow-xs' : 'font-medium'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </Button>
          );
        })}
      </div>
    </nav>
  );
}

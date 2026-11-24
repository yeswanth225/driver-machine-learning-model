import React from 'react';
import { Play } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function QuickScenarioBar({ scenarios, activeScenarioId, onSelectScenario, isAssessing }) {
  if (!scenarios || scenarios.length === 0) return null;

  return (
    <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-2 px-4 border-b border-[#dee2e6] dark:border-[#27272a] bg-white dark:bg-[#070709] select-none">
      <span className="text-[11px] font-display uppercase tracking-spacex text-neutral-500 dark:text-neutral-400 shrink-0 flex items-center gap-1.5 mr-2">
        <Play className="w-3 h-3 text-black dark:text-white" />
        <span>Profiles</span>
      </span>

      <div className="flex items-center gap-1.5">
        {scenarios.map((scen) => {
          const isSelected = activeScenarioId === scen.id;
          return (
            <Button
              key={scen.id}
              variant={isSelected ? "default" : "outline"}
              size="sm"
              onClick={() => onSelectScenario(scen)}
              disabled={isAssessing}
              className="rounded-full px-3.5 h-7 text-xs font-display transition-all"
            >
              <span>{scen.name}</span>
            </Button>
          );
        })}
      </div>
    </div>
  );
}

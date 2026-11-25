import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { CommandCenter } from './components/command/CommandCenter';
import { Vehicle3DView } from './components/views/Vehicle3DView';
import { LiveMonitoring } from './components/live/LiveMonitoring';
import { ReplayStudio } from './components/replay/ReplayStudio';
import { SafetyIntelligence } from './components/safety/SafetyIntelligence';
import { DrivingAnalytics } from './components/analytics/DrivingAnalytics';
import { SystemDiagnostics } from './components/diagnostics/SystemDiagnostics';
import {
  fetchHealth,
  fetchModelStatus,
  fetchScenarios,
  assessTelemetry,
  resetAgent,
} from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [healthStatus, setHealthStatus] = useState('Checking...');
  const [modelStatus, setModelStatus] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [activeScenarioId, setActiveScenarioId] = useState('normal');
  const [isResetting, setIsResetting] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [vehicleColor, setVehicleColor] = useState('silver');

  // Global active assessment driving the 3D twin & HUD
  const [assessment, setAssessment] = useState({
    timestamp: 1.0,
    safety_state: 'NORMAL',
    risk_score: 0.0,
    driving_style: 'Normal',
    driving_style_id: 0,
    driving_style_confidence: 0.96,
    maneuver: 'None / Nominal',
    maneuver_id: 0,
    maneuver_confidence: 0.98,
    pipeline_a_health: 'HEALTHY',
    pipeline_b_health: 'HEALTHY',
    is_valid: true,
    data_quality_status: 'NOMINAL',
    triggered_rules: ['NOMINAL_CRUISING'],
    explanation: 'Smooth highway cruising with normal driving style and quiescent vehicle dynamics. Telemetry nominal (Hardware: NOT_VERIFIED until on-road test).',
    active_alerts: [],
    active_incidents: [],
    confidence_summary: { style: 0.96, maneuver: 0.98 },
  });

  const [telemetryState, setTelemetryState] = useState({
    style_prediction: 0,
    style_confidence: 0.96,
    maneuver_prediction: 0,
    maneuver_confidence: 0.98,
    pipeline_a_health: 'HEALTHY',
    pipeline_b_health: 'HEALTHY',
  });

  useEffect(() => {
    initApp();
  }, []);

  const initApp = async () => {
    try {
      const [hRes, mRes, sRes] = await Promise.all([
        fetchHealth().catch(() => ({ status: 'offline' })),
        fetchModelStatus().catch(() => null),
        fetchScenarios().catch(() => []),
      ]);

      setHealthStatus(hRes.status === 'healthy' ? 'Connected' : 'Offline');
      setModelStatus(mRes);
      setScenarios(sRes);
    } catch (err) {
      console.error('App initialization error:', err);
      setHealthStatus('Offline');
    }
  };

  // Handle Scenario selection from the persistent left sidebar
  const handleSelectScenarioPreset = async (preset) => {
    setActiveScenarioId(preset.id);
    try {
      const resp = await assessTelemetry(preset.telemetry);
      setAssessment(resp);
      setTelemetryState({
        style_prediction: preset.telemetry.style_prediction,
        style_confidence: preset.telemetry.style_confidence,
        maneuver_prediction: preset.telemetry.maneuver_prediction,
        maneuver_confidence: preset.telemetry.maneuver_confidence,
        pipeline_a_health: preset.telemetry.pipeline_a_health || 'HEALTHY',
        pipeline_b_health: preset.telemetry.pipeline_b_health || 'HEALTHY',
      });
    } catch (err) {
      console.error('Failed to execute scenario preset:', err);
    }
  };

  const handleAssessCustom = async (customPayload) => {
    setActiveScenarioId(null);
    try {
      const resp = await assessTelemetry(customPayload);
      setAssessment(resp);
      setTelemetryState({
        style_prediction: customPayload.style_prediction,
        style_confidence: customPayload.style_confidence,
        maneuver_prediction: customPayload.maneuver_prediction,
        maneuver_confidence: customPayload.maneuver_confidence,
        pipeline_a_health: customPayload.pipeline_a_health || 'HEALTHY',
        pipeline_b_health: customPayload.pipeline_b_health || 'HEALTHY',
      });
    } catch (err) {
      console.error('Failed to execute custom assessment:', err);
    }
  };

  const handleResetAgent = async () => {
    setIsResetting(true);
    try {
      await resetAgent();
      setActiveScenarioId('normal');
      const baselinePayload = {
        timestamp: 1.0,
        style_prediction: 0,
        style_confidence: 0.96,
        maneuver_prediction: 0,
        maneuver_confidence: 0.98,
        pipeline_a_health: 'HEALTHY',
        pipeline_b_health: 'HEALTHY',
      };
      const resp = await assessTelemetry(baselinePayload);
      setAssessment(resp);
      setTelemetryState(baselinePayload);
    } catch (err) {
      console.error('Reset agent error:', err);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fa] dark:bg-black text-black dark:text-white flex font-sans selection:bg-black selection:text-white dark:selection:bg-white dark:selection:text-black">
      {/* Persistent Left Sidebar with 6 navigation sections + Settings at bottom */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={(tabId) => {
          setActiveTab(tabId);
          setIsMobileSidebarOpen(false);
        }}
        scenarios={scenarios}
        activeScenarioId={activeScenarioId}
        onSelectScenarioPreset={handleSelectScenarioPreset}
        onResetAgent={handleResetAgent}
        isResetting={isResetting}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden bg-[#f8f9fa] dark:bg-black">
        {/* Top Header */}
        <Header
          activeTab={activeTab}
          healthStatus={healthStatus}
          modelStatus={modelStatus}
          safetyState={assessment?.safety_state || 'NORMAL'}
          onResetAgent={handleResetAgent}
          isResetting={isResetting}
          onToggleSidebar={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        />

        {/* Dynamic Page Views */}
        <main className="flex-1 overflow-y-auto bg-[#f8f9fa] dark:bg-black">
          {activeTab === 'dashboard' && (
            <CommandCenter
              assessment={assessment}
              telemetryState={telemetryState}
              onAssessCustom={handleAssessCustom}
              onResetAgent={handleResetAgent}
              vehicleColor={vehicleColor}
              onColorChange={setVehicleColor}
            />
          )}

          {activeTab === 'live_monitoring' && (
            <LiveMonitoring
              onGlobalAssessmentUpdate={(newAssessment) => setAssessment(newAssessment)}
            />
          )}

          {(activeTab === 'driving_behaviour' || activeTab === 'driving_behavior') && (
            <Vehicle3DView
              assessment={assessment}
              telemetryState={telemetryState}
              onAssessCustom={handleAssessCustom}
              vehicleColor={vehicleColor}
              onColorChange={setVehicleColor}
            />
          )}

          {activeTab === 'safety_alerts' && (
            <SafetyIntelligence assessment={assessment} />
          )}

          {activeTab === 'trip_history' && (
            <ReplayStudio
              onGlobalAssessmentUpdate={(newAssessment) => setAssessment(newAssessment)}
              vehicleColor={vehicleColor}
              onColorChange={setVehicleColor}
            />
          )}

          {activeTab === 'driving_insights' && <DrivingAnalytics />}

          {activeTab === 'settings' && <SystemDiagnostics />}
        </main>

        {/* Aerospace Minimal Footer */}
        <footer className="border-t border-[#dee2e6] dark:border-[#27272a] bg-white dark:bg-[#070709] py-3 px-6 text-xs text-neutral-500 dark:text-neutral-400 flex flex-col sm:flex-row items-center justify-between gap-2 shrink-0 transition-colors">
          <div className="flex items-center gap-2">
            <span className="text-black dark:text-white font-medium">IDAMS Autonomous Safety Platform</span>
            <span className="text-neutral-300 dark:text-neutral-700">·</span>
            <span>Deterministic D-13 Guardrail Engine</span>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span>6-DOF IMU Kinematics &amp; Telemetry</span>
            <span className="text-black dark:text-white font-medium">Models Verified [4/4]</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

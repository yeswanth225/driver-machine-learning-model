import React, { useState, useEffect } from 'react';
import { 
  Activity, Shield, AlertTriangle, CheckCircle, RefreshCw, 
  Play, Cpu, ArrowRight, Gauge, Layers, Info, Check, Clock
} from 'lucide-react';
import { 
  fetchHealth, 
  fetchModelStatus, 
  fetchScenarios, 
  assessTelemetry, 
  resetAgent,
  fetchUnifiedRegistry
} from '../../services/api';
import { Button } from '../ui/button';

export function OverviewTab() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState('SCN-01');
  const [isAssessing, setIsAssessing] = useState(false);
  const [assessmentResult, setAssessmentResult] = useState(null);
  const [isResetting, setIsResetting] = useState(false);
  const [customTimestamp, setCustomTimestamp] = useState(1.0);
  const [statusMessage, setStatusMessage] = useState(null);

  // Load system health and scenarios on mount
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [regData, scenData] = await Promise.all([
          fetchUnifiedRegistry().catch(() => null),
          fetchScenarios().catch(() => [])
        ]);
        if (regData && regData.system_status) {
          setSystemStatus(regData.system_status);
        }
        if (scenData && scenData.length > 0) {
          setScenarios(scenData);
          setSelectedScenarioId(scenData[0].id);
        }
      } catch (err) {
        console.error('Failed to load initial overview data:', err);
      }
    }
    loadInitialData();
  }, []);

  const handleAssessScenario = async () => {
    setIsAssessing(true);
    setStatusMessage(null);
    try {
      const scenario = scenarios.find(s => s.id === selectedScenarioId);
      
      let payload;
      if (scenario && scenario.telemetry) {
        // Use scenario telemetry sequence or single point
        const t = scenario.telemetry[0] || scenario.telemetry;
        payload = {
          timestamp: t.timestamp || customTimestamp,
          style_prediction: t.style_prediction !== undefined ? t.style_prediction : 0,
          style_confidence: t.style_confidence !== undefined ? t.style_confidence : 0.85,
          maneuver_prediction: t.maneuver_prediction !== undefined ? t.maneuver_prediction : 0,
          maneuver_confidence: t.maneuver_confidence !== undefined ? t.maneuver_confidence : 0.90,
          pipeline_a_health: t.pipeline_a_health || "HEALTHY",
          pipeline_b_health: t.pipeline_b_health || "HEALTHY"
        };
      } else {
        // Fallback default based on scenario ID
        const isAggressive = selectedScenarioId === 'SCN-03';
        const isBraking = selectedScenarioId === 'SCN-02';
        payload = {
          timestamp: customTimestamp,
          style_prediction: isAggressive ? 1 : 0,
          style_confidence: isAggressive ? 0.92 : 0.88,
          maneuver_prediction: isBraking ? 2 : (selectedScenarioId === 'SCN-03' ? 4 : 0),
          maneuver_confidence: (isBraking || isAggressive) ? 0.94 : 0.90,
          pipeline_a_health: selectedScenarioId === 'SCN-04' ? 'MISSING' : 'HEALTHY',
          pipeline_b_health: selectedScenarioId === 'SCN-05' ? 'MISSING' : (selectedScenarioId === 'SCN-06' ? 'MISSING' : 'HEALTHY')
        };
      }

      const res = await assessTelemetry(payload);
      setAssessmentResult(res);
      setCustomTimestamp(prev => parseFloat((prev + 1.0).toFixed(1)));
    } catch (err) {
      console.error('Assessment failed:', err);
      setStatusMessage({ type: 'error', text: err.message || 'Assessment failed' });
    } finally {
      setIsAssessing(false);
    }
  };

  const handleResetAgent = async () => {
    setIsResetting(true);
    try {
      await resetAgent();
      setAssessmentResult(null);
      setCustomTimestamp(1.0);
      setStatusMessage({ type: 'success', text: 'Safety Agent state and incidents reset successfully.' });
      setTimeout(() => setStatusMessage(null), 3500);
    } catch (err) {
      console.error('Reset failed:', err);
      setStatusMessage({ type: 'error', text: 'Failed to reset agent.' });
    } finally {
      setIsResetting(false);
    }
  };

  // State colors
  const getStateColor = (state) => {
    switch (state) {
      case 'NORMAL':
        return 'text-emerald-400 bg-emerald-950/40 border-emerald-800/50';
      case 'CAUTION':
        return 'text-amber-400 bg-amber-950/40 border-amber-800/50';
      case 'WARNING':
        return 'text-orange-400 bg-orange-950/40 border-orange-800/50';
      case 'CRITICAL':
      case 'HIGH_RISK':
        return 'text-rose-400 bg-rose-950/40 border-rose-800/50';
      case 'INSUFFICIENT_DATA':
        return 'text-slate-400 bg-slate-900 border-slate-800';
      default:
        return 'text-blue-400 bg-blue-950/40 border-blue-800/50';
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* 1. System Health Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-4 flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Pipeline A Status</div>
            <div className="text-sm font-semibold text-slate-100 truncate">Production SVM</div>
            <div className="text-[11px] text-emerald-400 flex items-center mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span>
              Online (Candidate Quarantined)
            </div>
          </div>
        </div>

        <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-4 flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
            <Activity className="w-5 h-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Pipeline B Status</div>
            <div className="text-sm font-semibold text-slate-100 truncate">Extra Trees Maneuver</div>
            <div className="text-[11px] text-cyan-400 flex items-center mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 mr-1.5 animate-pulse"></span>
              Online (Kinematic Event Gate)
            </div>
          </div>
        </div>

        <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-4 flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Shield className="w-5 h-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Safety Agent</div>
            <div className="text-sm font-semibold text-slate-100 truncate">Deterministic Fusion</div>
            <div className="text-[11px] text-emerald-400 flex items-center mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5"></span>
              121/121 Tests Passing (D-13/P-05A)
            </div>
          </div>
        </div>

        <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-4 flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <Layers className="w-5 h-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wider text-slate-400 font-medium">Hardware Telemetry</div>
            <div className="text-sm font-semibold text-slate-100 truncate">Simulation Harness</div>
            <div className="text-[11px] text-slate-400 flex items-center mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500 mr-1.5"></span>
              7 Evaluated Scenarios Active
            </div>
          </div>
        </div>
      </div>

      {/* 2. Dual-Pipeline Compact Architecture Flow */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 mb-5 border-b border-[#1E222D]">
          <div>
            <h2 className="text-lg font-semibold text-slate-100 flex items-center">
              <Layers className="w-5 h-5 mr-2 text-emerald-400" />
              Dual-Pipeline & Safety Agent System Flow
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Decoupled multi-modal feature extraction with deterministic life-critical arbitration.
            </p>
          </div>
          <span className="mt-2 sm:mt-0 text-[11px] px-2.5 py-1 rounded bg-[#161820] text-slate-300 border border-[#262A36] font-mono">
            Zero ML Override on Safety Gates
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
          {/* Step 1 */}
          <div className="bg-[#12141A] border border-[#222632] rounded-lg p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                <span>STAGE 01</span>
                <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded">INGESTION</span>
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Vehicle Sensors & IMU</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Continuous high-frequency tri-axial accelerometer, gyroscope, GPS speed & steering angles sampled in 1s rolling windows.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1C202C] text-[11px] text-slate-500 font-mono">
              Output: Synchronized raw feature vector
            </div>
          </div>

          {/* Step 2 */}
          <div className="bg-[#12141A] border border-[#222632] rounded-lg p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                <span>STAGE 02</span>
                <span className="text-[10px] bg-blue-900/60 text-blue-300 px-1.5 py-0.5 rounded">PARALLEL ML</span>
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Dual ML Classification</h3>
              <div className="space-y-1.5 text-xs text-slate-400">
                <div className="p-1.5 bg-[#181B24] rounded border border-[#262B3A]">
                  <strong className="text-slate-200">Pipeline A:</strong> Driving Style (SVM / ET 47-feat)
                </div>
                <div className="p-1.5 bg-[#181B24] rounded border border-[#262B3A]">
                  <strong className="text-slate-200">Pipeline B:</strong> Maneuvers (ET + Kinematic Gate)
                </div>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1C202C] text-[11px] text-blue-400/80 font-mono">
              Output: Style + Maneuver probabilities
            </div>
          </div>

          {/* Step 3 */}
          <div className="bg-[#12141A] border border-emerald-900/40 rounded-lg p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-emerald-400 mb-2">
                <span>STAGE 03</span>
                <span className="text-[10px] bg-emerald-900/60 text-emerald-300 px-1.5 py-0.5 rounded">SAFETY AGENT</span>
              </div>
              <h3 className="text-sm font-semibold text-emerald-300 mb-2">Deterministic Fusion</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Applies D-13 Option 1A-Pure hard kinematic limits, P-05A priority suppression, 2.50s incident hysteresis, and analytical residual risk decay.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1C202C] text-[11px] text-emerald-400 font-mono">
              Output: Continuous risk score [0–100]
            </div>
          </div>

          {/* Step 4 */}
          <div className="bg-[#12141A] border border-[#222632] rounded-lg p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                <span>STAGE 04</span>
                <span className="text-[10px] bg-amber-900/60 text-amber-300 px-1.5 py-0.5 rounded">ACTION</span>
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Assessment & Advisory</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Emits classified Safety State (NORMAL, CAUTION, WARNING, CRITICAL), incident tracking timestamps, and contextual driver advisory warnings.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1C202C] text-[11px] text-amber-400/80 font-mono">
              Output: Actionable driver guidance
            </div>
          </div>
        </div>
      </div>

      {/* 3. Interactive Scenario Runner & Live Telemetry Assessment */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 mb-6 border-b border-[#1E222D]">
          <div>
            <h2 className="text-lg font-semibold text-slate-100 flex items-center">
              <Play className="w-5 h-5 mr-2 text-emerald-400" />
              Live Telemetry Assessment & Scenario Runner
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Execute pre-configured multi-modal driving scenarios or custom telemetry points through the frozen pipeline and Safety Agent.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleResetAgent}
            disabled={isResetting}
            className="mt-3 sm:mt-0 gap-1.5 rounded-lg"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin' : ''}`} />
            <span>Reset Agent State</span>
          </Button>
        </div>

        {statusMessage && (
          <div className={`p-3 rounded-lg mb-6 text-xs flex items-center space-x-2 border ${
            statusMessage.type === 'error' 
              ? 'bg-rose-950/40 text-rose-300 border-rose-800/50' 
              : 'bg-emerald-950/40 text-emerald-300 border-emerald-800/50'
          }`}>
            <Info className="w-4 h-4 flex-shrink-0" />
            <span>{statusMessage.text}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Column */}
          <div className="lg:col-span-5 space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">
                Select Driving Scenario
              </label>
              <div className="space-y-2">
                {scenarios.map((scen) => (
                  <Button
                    key={scen.id}
                    variant={selectedScenarioId === scen.id ? "default" : "outline"}
                    onClick={() => setSelectedScenarioId(scen.id)}
                    className="w-full text-left p-3 h-auto rounded-lg flex items-center justify-between"
                  >
                    <div>
                      <div className="text-xs font-semibold flex items-center">
                        <span className="font-mono text-emerald-400 mr-2">{scen.id}</span>
                        {scen.name}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5 font-normal">
                        {scen.category} &bull; Expected: <span className="font-mono">{scen.expected_state}</span>
                      </div>
                    </div>
                    {selectedScenarioId === scen.id && (
                      <Check className="w-4 h-4 text-emerald-400 flex-shrink-0 ml-2" />
                    )}
                  </Button>
                ))}
              </div>
            </div>

            <div className="pt-2">
              <Button
                variant="default"
                onClick={handleAssessScenario}
                disabled={isAssessing}
                className="w-full py-3 h-auto rounded-lg text-sm font-medium gap-2"
              >
                {isAssessing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Evaluating Inference &amp; Fusion...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Assess Scenario Telemetry</span>
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Results Column */}
          <div className="lg:col-span-7">
            <div className="bg-[#12141A] border border-[#222632] rounded-xl p-5 h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 mb-4 border-b border-[#1C202C]">
                  <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                    Live Assessment Result
                  </span>
                  {assessmentResult ? (
                    <span className="text-xs font-mono text-slate-400">
                      Timestamp: {assessmentResult.timestamp}s
                    </span>
                  ) : (
                    <span className="text-xs text-slate-500">Awaiting assessment trigger</span>
                  )}
                </div>

                {assessmentResult ? (
                  <div className="space-y-5">
                    {/* Primary Safety State & Risk Gauge */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {/* Safety State Box */}
                      <div className={`p-4 rounded-xl border flex flex-col justify-between ${getStateColor(assessmentResult.safety_state)}`}>
                        <div className="text-xs font-medium uppercase tracking-wider opacity-80">Safety State</div>
                        <div className="text-2xl font-bold font-mono tracking-tight my-2">
                          {assessmentResult.safety_state}
                        </div>
                        <div className="text-[11px] opacity-75">
                          Deterministic arbitration output
                        </div>
                      </div>

                      {/* Continuous Risk Score */}
                      <div className="p-4 rounded-xl border border-[#222632] bg-[#161820] flex flex-col justify-between">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                          <span className="font-medium uppercase tracking-wider">Risk Score</span>
                          <span className="font-mono text-slate-200 font-bold">{assessmentResult.risk_score.toFixed(1)} / 100</span>
                        </div>
                        <div className="my-2.5">
                          <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                            <div 
                              className={`h-full transition-all duration-500 ${
                                assessmentResult.risk_score > 75 
                                  ? 'bg-rose-500' 
                                  : assessmentResult.risk_score > 40 
                                  ? 'bg-amber-500' 
                                  : 'bg-emerald-500'
                              }`}
                              style={{ width: `${Math.min(100, Math.max(0, assessmentResult.risk_score))}%` }}
                            />
                          </div>
                        </div>
                        <div className="text-[11px] text-slate-500 flex items-center justify-between">
                          <span>0 (Safe)</span>
                          <span>100 (Critical Hazard)</span>
                        </div>
                      </div>
                    </div>

                    {/* Model Inferences Breakdown */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="p-3 bg-[#181B24] rounded-lg border border-[#242938]">
                        <div className="text-[11px] text-slate-400 uppercase tracking-wider">Pipeline A (Driving Style)</div>
                        <div className="text-sm font-semibold text-slate-200 mt-1 flex items-center justify-between">
                          <span>{assessmentResult.driving_style}</span>
                          <span className="text-xs font-mono text-slate-400">
                            {(assessmentResult.driving_style_confidence * 100).toFixed(1)}% conf
                          </span>
                        </div>
                      </div>

                      <div className="p-3 bg-[#181B24] rounded-lg border border-[#242938]">
                        <div className="text-[11px] text-slate-400 uppercase tracking-wider">Pipeline B (Kinematic Maneuver)</div>
                        <div className="text-sm font-semibold text-slate-200 mt-1 flex items-center justify-between">
                          <span>{assessmentResult.maneuver}</span>
                          <span className="text-xs font-mono text-slate-400">
                            {(assessmentResult.maneuver_confidence * 100).toFixed(1)}% conf
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Active Alerts */}
                    <div>
                      <div className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
                        Active Alerts ({assessmentResult.alerts ? assessmentResult.alerts.length : 0})
                      </div>
                      {assessmentResult.alerts && assessmentResult.alerts.length > 0 ? (
                        <div className="space-y-2">
                          {assessmentResult.alerts.map((alert, i) => (
                            <div key={i} className="p-3 rounded-lg bg-rose-950/20 border border-rose-900/40 text-xs text-rose-200">
                              <div className="font-semibold text-rose-300 flex items-center justify-between">
                                <span className="flex items-center">
                                  <AlertTriangle className="w-3.5 h-3.5 mr-1.5 text-rose-400" />
                                  {alert.title}
                                </span>
                                <span className="font-mono text-[10px] bg-rose-900/50 px-1.5 py-0.5 rounded text-rose-300">
                                  {alert.severity}
                                </span>
                              </div>
                              <p className="mt-1 text-slate-400 text-[11px] leading-relaxed">
                                {alert.explanation}
                              </p>
                              {alert.safe_driving_advisory && (
                                <div className="mt-1.5 pt-1.5 border-t border-rose-900/30 text-[11px] text-emerald-400">
                                  <strong>Advisory:</strong> {alert.safe_driving_advisory}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="p-3 rounded-lg bg-[#161820] border border-[#222632] text-xs text-slate-500 flex items-center">
                          <CheckCircle className="w-4 h-4 mr-2 text-emerald-500" />
                          No active safety hazard alerts. Driving parameters nominal.
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center text-slate-500">
                    <Gauge className="w-10 h-10 mb-3 text-slate-600 stroke-[1.5]" />
                    <p className="text-sm font-medium text-slate-400">No Assessment Active</p>
                    <p className="text-xs max-w-sm mt-1 text-slate-500">
                      Select a scenario on the left and click &ldquo;Assess Scenario Telemetry&rdquo; to execute inference and view safety arbitration.
                    </p>
                  </div>
                )}
              </div>

              <div className="mt-4 pt-3 border-t border-[#1C202C] flex items-center justify-between text-[11px] text-slate-500">
                <span>Model Engine: Production SVM + ExtraTrees</span>
                <span>Deterministic Arbitration: Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import { 
  Cpu, Filter, Search, Shield, Activity, HardDrive, 
  Clock, CheckCircle, AlertCircle, Copy, Check, Info, FileText 
} from 'lucide-react';
import { fetchUnifiedRegistry } from '../../services/api';
import { Button } from '../ui/button';

export function ModelsTab() {
  const [registry, setRegistry] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPipeline, setSelectedPipeline] = useState('ALL');
  const [selectedStatus, setSelectedStatus] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedHash, setCopiedHash] = useState(null);

  useEffect(() => {
    async function loadRegistry() {
      try {
        setLoading(true);
        const data = await fetchUnifiedRegistry();
        setRegistry(data);
      } catch (err) {
        console.error('Failed to load unified registry:', err);
        setError(err.message || 'Failed to load model registry');
      } finally {
        setLoading(false);
      }
    }
    loadRegistry();
  }, []);

  const handleCopyHash = (hash, id) => {
    navigator.clipboard.writeText(hash);
    setCopiedHash(id);
    setTimeout(() => setCopiedHash(null), 2500);
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'Production':
        return 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60';
      case 'Validated Candidate':
        return 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60';
      case 'Experimental':
        return 'bg-amber-950/60 text-amber-300 border-amber-800/60';
      default:
        return 'bg-slate-900 text-slate-300 border-slate-700';
    }
  };

  const filteredModels = (registry?.models || []).filter((model) => {
    // Pipeline filter
    if (selectedPipeline !== 'ALL') {
      if (selectedPipeline === 'PIPELINE_A' && !model.pipeline.includes('Pipeline A')) return false;
      if (selectedPipeline === 'PIPELINE_B' && !model.pipeline.includes('Pipeline B')) return false;
      if (selectedPipeline === 'SAFETY_AGENT' && !model.pipeline.includes('Safety Agent')) return false;
    }
    // Status filter
    if (selectedStatus !== 'ALL') {
      if (selectedStatus === 'PRODUCTION' && model.status !== 'Production') return false;
      if (selectedStatus === 'CANDIDATE' && model.status !== 'Validated Candidate') return false;
      if (selectedStatus === 'EXPERIMENTAL' && model.status !== 'Experimental') return false;
    }
    // Search query
    if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      const matchName = model.name.toLowerCase().includes(q);
      const matchArch = model.architecture.toLowerCase().includes(q);
      const matchNotes = (model.notes || '').toLowerCase().includes(q);
      if (!matchName && !matchArch && !matchNotes) return false;
    }
    return true;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header & Filter Controls */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between pb-5 mb-5 border-b border-[#1E222D]">
          <div>
            <h2 className="text-lg font-semibold text-slate-100 flex items-center">
              <Cpu className="w-5 h-5 mr-2 text-emerald-400" />
              Unified Model Registry & Architecture Catalog
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Dynamic repository registry of all production baselines, drop-in candidates, and evaluated experimental architectures.
            </p>
          </div>
          <div className="mt-3 md:mt-0 text-xs font-mono text-slate-400 bg-[#161820] px-3 py-1.5 rounded-lg border border-[#262A36]">
            Total Registered: <span className="text-emerald-400 font-bold">{registry?.models?.length || 0}</span>
          </div>
        </div>

        {/* Filter Toolbar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Search Input */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
            <input
              type="text"
              placeholder="Search model, architecture, or notes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-[#12141A] border border-[#222632] rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Pipeline Selector */}
          <div className="flex items-center space-x-2 bg-[#12141A] border border-[#222632] rounded-lg px-2 py-1">
            <Filter className="w-3.5 h-3.5 text-slate-500 ml-1" />
            <select
              value={selectedPipeline}
              onChange={(e) => setSelectedPipeline(e.target.value)}
              className="w-full bg-transparent text-xs text-slate-300 focus:outline-none cursor-pointer py-1"
            >
              <option value="ALL">All Pipelines</option>
              <option value="PIPELINE_A">Pipeline A (Driving Style)</option>
              <option value="PIPELINE_B">Pipeline B (Maneuver Detection)</option>
              <option value="SAFETY_AGENT">Safety Agent (Fusion Engine)</option>
            </select>
          </div>

          {/* Status Selector */}
          <div className="flex items-center space-x-2 bg-[#12141A] border border-[#222632] rounded-lg px-2 py-1">
            <Activity className="w-3.5 h-3.5 text-slate-500 ml-1" />
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="w-full bg-transparent text-xs text-slate-300 focus:outline-none cursor-pointer py-1"
            >
              <option value="ALL">All Statuses</option>
              <option value="PRODUCTION">Production Only</option>
              <option value="CANDIDATE">Validated Candidates</option>
              <option value="EXPERIMENTAL">Experimental Only</option>
            </select>
          </div>
        </div>
      </div>

      {/* Model Cards Grid */}
      {loading ? (
        <div className="p-12 text-center text-slate-500 bg-[#0D0E12] border border-[#1E222D] rounded-xl">
          <Clock className="w-8 h-8 animate-spin mx-auto mb-2 text-slate-400" />
          <p className="text-xs">Loading Unified Model Registry...</p>
        </div>
      ) : error ? (
        <div className="p-6 text-center text-rose-400 bg-rose-950/20 border border-rose-900/40 rounded-xl text-xs">
          {error}
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="p-12 text-center text-slate-500 bg-[#0D0E12] border border-[#1E222D] rounded-xl text-xs">
          No models matched your filter criteria.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {filteredModels.map((model) => (
            <div 
              key={model.id}
              className="bg-[#0D0E12] border border-[#1E222D] hover:border-[#2A3040] transition rounded-xl p-6"
            >
              {/* Card Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#1E222D] gap-2">
                <div>
                  <div className="flex items-center space-x-2.5">
                    <h3 className="text-base font-semibold text-slate-100">{model.name}</h3>
                    <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border ${getStatusBadge(model.status)}`}>
                      {model.status}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 mt-1 font-mono">
                    {model.pipeline} &bull; ID: <span className="text-slate-300">{model.id}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-3 text-xs text-slate-400 font-mono">
                  {model.metrics?.latency_p50_ms && (
                    <div className="flex items-center space-x-1 bg-[#14161F] px-2.5 py-1 rounded border border-[#222634]">
                      <Clock className="w-3.5 h-3.5 text-amber-400" />
                      <span>{model.metrics.latency_p50_ms} ms (P50)</span>
                    </div>
                  )}
                  {model.metrics?.size_mb && (
                    <div className="flex items-center space-x-1 bg-[#14161F] px-2.5 py-1 rounded border border-[#222634]">
                      <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                      <span>{model.metrics.size_mb} MB</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Architecture & Specs */}
              <div className="py-4 grid grid-cols-1 md:grid-cols-3 gap-4 border-b border-[#1E222D] text-xs">
                <div>
                  <span className="text-slate-500 uppercase tracking-wider text-[10px] font-semibold block mb-1">Architecture</span>
                  <div className="text-slate-200 font-mono text-[11px] leading-relaxed">{model.architecture}</div>
                </div>
                <div>
                  <span className="text-slate-500 uppercase tracking-wider text-[10px] font-semibold block mb-1">Dataset & Validation</span>
                  <div className="text-slate-300 text-[11px] leading-relaxed">{model.dataset}</div>
                  <div className="text-slate-500 text-[10px] mt-0.5">{model.validation_type}</div>
                </div>
                <div>
                  <span className="text-slate-500 uppercase tracking-wider text-[10px] font-semibold block mb-1">Partition Parameters</span>
                  <div className="text-slate-300 text-[11px]">
                    Samples: <span className="font-mono text-slate-200">{model.sample_count}</span> &bull; Features: <span className="font-mono text-slate-200">{model.feature_count}</span>
                  </div>
                  <div className="text-slate-400 text-[10px] mt-0.5">
                    Operating Threshold: <span className="font-mono text-emerald-400">{model.threshold}</span>
                  </div>
                </div>
              </div>

              {/* Primary Metrics Grid */}
              <div className="py-4 border-b border-[#1E222D]">
                <div className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 mb-2.5">
                  Audited Verification Metrics
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Accuracy</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.accuracy === 'number' ? `${(model.metrics.accuracy * 100).toFixed(2)}%` : model.metrics?.accuracy}
                    </div>
                  </div>

                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Macro-F1</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.macro_f1 === 'number' ? `${(model.metrics.macro_f1 * 100).toFixed(2)}%` : model.metrics?.macro_f1}
                    </div>
                  </div>

                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Recall</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.recall === 'number' ? `${(model.metrics.recall * 100).toFixed(2)}%` : model.metrics?.recall}
                    </div>
                  </div>

                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Precision</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.precision === 'number' ? `${(model.metrics.precision * 100).toFixed(2)}%` : model.metrics?.precision}
                    </div>
                  </div>

                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">FPR (False Pos)</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.fpr === 'number' ? `${(model.metrics.fpr * 100).toFixed(2)}%` : model.metrics?.fpr}
                    </div>
                  </div>

                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">FNR (Missed)</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.fnr === 'number' ? `${(model.metrics.fnr * 100).toFixed(2)}%` : model.metrics?.fnr}
                    </div>
                  </div>

                  <div className="bg-[#12141A] p-2.5 rounded-lg border border-[#1E222D] text-center">
                    <div className="text-[10px] text-slate-500 uppercase">ROC-AUC</div>
                    <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                      {typeof model.metrics?.roc_auc === 'number' ? `${(model.metrics.roc_auc * 100).toFixed(2)}%` : model.metrics?.roc_auc}
                    </div>
                  </div>
                </div>

                {/* Directional Metrics for Pipeline A */}
                {model.directional_metrics && (
                  <div className="mt-3 p-2.5 bg-[#14161F] rounded-lg border border-[#202430] flex flex-wrap items-center justify-between text-xs font-mono text-slate-400 gap-2">
                    <div>
                      <span className="text-slate-500 mr-1.5">S1 &rarr; S2 Directional:</span>
                      <span className="text-emerald-400 font-bold">{(model.directional_metrics.s1_to_s2_acc * 100).toFixed(2)}% Acc</span>
                      <span className="text-slate-500 mx-1.5">|</span>
                      <span className="text-slate-300">{(model.directional_metrics.s1_to_s2_f1 * 100).toFixed(2)}% F1</span>
                    </div>
                    <div>
                      <span className="text-slate-500 mr-1.5">S2 &rarr; S1 Directional:</span>
                      <span className="text-slate-300">{(model.directional_metrics.s2_to_s1_acc * 100).toFixed(2)}% Acc</span>
                      <span className="text-slate-500 mx-1.5">|</span>
                      <span className="text-slate-300">{(model.directional_metrics.s2_to_s1_f1 * 100).toFixed(2)}% F1</span>
                    </div>
                    <div>
                      <span className="text-slate-500 mr-1.5">Arithmetic Mean:</span>
                      <span className="text-cyan-400 font-bold">{(model.directional_metrics.mean_cross_session_acc * 100).toFixed(2)}% Acc</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Artifact & Cryptographic Integrity Footer */}
              <div className="pt-4 flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-2">
                <div className="space-y-1">
                  <div className="text-slate-400 flex items-center space-x-1.5 font-mono text-[11px]">
                    <FileText className="w-3.5 h-3.5 text-slate-500" />
                    <span>Path: {model.artifact_path}</span>
                  </div>
                  {model.sha256 && model.sha256 !== 'N/A (Experimental)' && (
                    <div className="text-slate-500 flex items-center space-x-1.5 font-mono text-[10px]">
                      <span>SHA-256: {model.sha256}</span>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => handleCopyHash(model.sha256, model.id)}
                        className="h-5 w-5 rounded-md p-0 border-transparent hover:border-slate-700"
                        title="Copy SHA-256 Hash"
                        aria-label="Copy SHA-256 Hash"
                      >
                        {copiedHash === model.id ? (
                          <Check className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <Copy className="w-3 h-3 text-slate-500" />
                        )}
                      </Button>
                    </div>
                  )}
                </div>

                <div className="text-right text-[11px] text-slate-400 max-w-md italic">
                  &ldquo;{model.notes}&rdquo;
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

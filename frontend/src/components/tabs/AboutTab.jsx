import React from 'react';
import { 
  Info, Shield, Layers, Database, Compass, AlertTriangle, 
  Award, CheckCircle, FileText, ExternalLink 
} from 'lucide-react';

export function AboutTab() {
  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* 1. Project Overview & College Mini-Project Context */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 mb-5 border-b border-[#1E222D]">
          <div>
            <h2 className="text-lg font-semibold text-slate-100 flex items-center">
              <Info className="w-5 h-5 mr-2 text-emerald-400" />
              Project Scope & College Mini-Project Context
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Intelligent Driver Activity Monitoring & Safety Agent &bull; Final Engineering Submission
            </p>
          </div>
          <span className="mt-2 sm:mt-0 text-[11px] px-2.5 py-1 rounded bg-[#161820] text-emerald-400 border border-emerald-900/50 font-mono">
            Rigorous Empirical Standards
          </span>
        </div>

        <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
          <p>
            The <strong>Intelligent Driver Activity Monitoring & Safety Agent</strong> is an automotive Advanced Driver Assistance System (ADAS) software research platform developed as a college engineering mini-project. It combines high-frequency smartphone sensor telemetry, dual-pipeline machine learning models, and an ISO-compliant deterministic safety rule fusion engine.
          </p>
          <p>
            Unlike conventional academic machine learning demos that overfit on synthetic splits and fabricate inflated accuracy claims, this project adheres to strict industrial verification practices: 10-second guard band chronological holdouts, bidirectional leave-one-session-out (LODO) validation, cryptographic SHA-256 model freezing, and complete empirical transparency regarding single-driver dataset limitations.
          </p>
        </div>
      </div>

      {/* 2. Architectural Philosophy: Why Decoupled Fusion Outperforms Single Models */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="pb-5 mb-5 border-b border-[#1E222D]">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center">
            <Layers className="w-5 h-5 mr-2 text-emerald-400" />
            Architectural Rationale: Dual-Pipeline Decoupling + Deterministic Safety Agent
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Why automotive safety demands decoupled pipelines and deterministic arbitration over monolithic end-to-end deep learning.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300">
          <div className="p-4 bg-[#12141A] border border-[#222632] rounded-lg space-y-2.5">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center">
              <span className="w-2 h-2 rounded-full bg-cyan-400 mr-2"></span>
              Decoupled Multi-Modal Pipelines
            </h3>
            <p className="text-slate-400 leading-relaxed">
              Driving style (aggressive vs. normal) is an extended behavioral condition characterized by statistical variance over time, while maneuvers (sudden braking, sudden turns) are brief, transient kinematic impulses.
            </p>
            <p className="text-slate-400 leading-relaxed">
              Decoupling these into <strong>Pipeline A</strong> (Driving Style, 47 features, 1s rolling windows) and <strong>Pipeline B</strong> (Kinematic Maneuvers, ExtraTrees + KinematicEventGate) prevents temporal feature contamination and allows independent tuning, validation, and fail-safe handling.
            </p>
          </div>

          <div className="p-4 bg-[#12141A] border border-[#222632] rounded-lg space-y-2.5">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center">
              <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2"></span>
              Deterministic Safety Agent (Zero Hallucination)
            </h3>
            <p className="text-slate-400 leading-relaxed">
              In automotive safety, an ML model must never have unconstrained authority to suppress danger warnings or hallucinate safety states.
            </p>
            <p className="text-slate-400 leading-relaxed">
              The <strong>Safety Agent</strong> implements D-13 Option 1A-Pure: if kinematic sensors detect a physical hard brake or aggressive swerve, the safety alert is guaranteed deterministically regardless of what Pipeline A predicts. It includes P-05A alert priority suppression, 2.50s incident hysteresis, and analytical residual risk decay.
            </p>
          </div>
        </div>
      </div>

      {/* 3. Comprehensive Dataset Overview */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="pb-5 mb-5 border-b border-[#1E222D]">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center">
            <Database className="w-5 h-5 mr-2 text-cyan-400" />
            Evaluated Datasets & Modalities
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Overview of empirical datasets used across research, prototyping, and production verification.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 bg-[#12141A] border border-[#222632] rounded-lg flex flex-col justify-between">
            <div>
              <div className="font-mono text-cyan-400 font-bold mb-1">DATASET A</div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Driver Inattention & Distraction</h3>
              <p className="text-slate-400 leading-relaxed">
                Computer-vision and facial landmark telemetry capturing driver drowsiness, eye closure duration, head-pose deflection, and mobile phone usage.
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-[#1C202C] text-[11px] text-slate-500 font-mono">
              Status: Research Reference
            </div>
          </div>

          <div className="p-4 bg-[#12141A] border border-[#222632] rounded-lg flex flex-col justify-between">
            <div>
              <div className="font-mono text-cyan-400 font-bold mb-1">DATASET B</div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Smartphone IMU Maneuvers</h3>
              <p className="text-slate-400 leading-relaxed">
                Tri-axial accelerometer and gyroscope data collected across 3 drivers performing 4 specific maneuvers: Sudden Acceleration (SA), Sudden Deceleration (SD), Sudden Right Turn (SRT), Sudden Left Turn (SLT).
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-[#1C202C] text-[11px] text-emerald-400 font-mono">
              Status: Production Validated (Pipeline B)
            </div>
          </div>

          <div className="p-4 bg-[#12141A] border border-[#222632] rounded-lg flex flex-col justify-between">
            <div>
              <div className="font-mono text-cyan-400 font-bold mb-1">DATASET C</div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Naturalistic Driving Style</h3>
              <p className="text-slate-400 leading-relaxed">
                Smartphone IMU sensor recordings from 1 driver across 2 sessions (17 episodes, 6,441 1-second frames) labeled into Normal vs. Aggressive driving style.
              </p>
            </div>
            <div className="mt-3 pt-2.5 border-t border-[#1C202C] text-[11px] text-amber-400 font-mono">
              Status: Production Baseline (Pipeline A)
            </div>
          </div>
        </div>
      </div>

      {/* 4. Multi-Driver Collection Protocol Roadmap (8-Driver Plan) */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="pb-5 mb-5 border-b border-[#1E222D]">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center">
            <Compass className="w-5 h-5 mr-2 text-emerald-400" />
            Roadmap: 8-Driver Multi-Subject Collection Protocol
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Statistically grounded empirical protocol designed to achieve genuine, verifiable 90%+ multi-driver generalization.
          </p>
        </div>

        <div className="space-y-4 text-xs text-slate-300">
          <div className="p-4 bg-[#12141A] border border-[#222632] rounded-lg space-y-3">
            <h3 className="text-sm font-semibold text-slate-200">The 6 + 2 Partition Architecture</h3>
            <ul className="list-disc list-inside space-y-1.5 text-slate-400 leading-relaxed">
              <li>
                <strong className="text-slate-200">6 Development Drivers:</strong> Grouped nested cross-validation (4 training drivers, 1 validation driver for operating point calibration, 1 held-out test driver for out-of-fold generalization evaluation).
              </li>
              <li>
                <strong className="text-slate-200">2 Independent Lockbox Drivers:</strong> Sealed lockbox drivers completely isolated from hyperparameter tuning and model selection; evaluated exactly once on frozen final candidates.
              </li>
              <li>
                <strong className="text-slate-200">Rigid Mounting & Gravity Transformation:</strong> Standardized rigid vehicle cradles and virtual coordinate alignment (converting smartphone body axes to vehicle forward-lateral-vertical axes via low-pass gravitational projection) to eliminate the 31.36&deg; tilt drift.
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* 5. Limitations, Safety Disclaimers & Ethics */}
      <div className="bg-[#0D0E12] border border-[#1E222D] rounded-xl p-6">
        <div className="pb-4 mb-4 border-b border-[#1E222D]">
          <h2 className="text-base font-semibold text-slate-100 flex items-center">
            <Shield className="w-4 h-4 mr-2 text-amber-400" />
            System Boundaries, Ethics & Limitations
          </h2>
        </div>

        <div className="space-y-2 text-xs text-slate-400 leading-relaxed">
          <p>
            &bull; <strong>Advisory Nature:</strong> This software is strictly an advisory Advanced Driver Assistance System (ADAS) demonstration. It does not interface with or actuate vehicle steering, throttle, or braking controls.
          </p>
          <p>
            &bull; <strong>Single-Driver Boundary:</strong> Production Pipeline A is trained on Dataset C (single driver). While Candidate 1 demonstrates 91.43% directional transfer, Antigravity transparently states that multi-driver generalization across diverse vehicle dynamics requires multi-driver training data.
          </p>
          <p>
            &bull; <strong>Cryptographic Traceability:</strong> All production models, candidate weights, and test metrics are frozen with immutable SHA-256 checksums to ensure reproducibility.
          </p>
        </div>
      </div>
    </div>
  );
}

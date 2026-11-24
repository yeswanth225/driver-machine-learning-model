import React, { useState, useEffect } from 'react';
import { Activity, Radio, Cpu, RefreshCw, AlertOctagon, CheckCircle2, Play, Pause, Zap, Info, Gauge } from 'lucide-react';
import { fetchHardwareStatus } from '../../services/api';
import { Badge } from '../common/Badge';
import { ComponentInspectorModal } from '../common/ComponentInspectorModal';
import { Button } from '../ui/button';

export function VehicleSensorsView() {
  const [hwStatus, setHwStatus] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [inspectorTarget, setInspectorTarget] = useState(null);
  const [sensorData, setSensorData] = useState({
    accX: 0.02,
    accY: -0.01,
    accZ: -0.98,
    gyroX: 0.004,
    gyroY: -0.002,
    gyroZ: 0.001,
    speed: 68.4,
    steeringAngle: 0.5,
  });

  useEffect(() => {
    checkHardware();
  }, []);

  useEffect(() => {
    let interval;
    if (isSimulating) {
      interval = setInterval(() => {
        setSensorData({
          accX: Number(((Math.random() - 0.5) * 0.15).toFixed(3)),
          accY: Number(((Math.random() - 0.5) * 0.12).toFixed(3)),
          accZ: Number((-0.98 + (Math.random() - 0.5) * 0.05).toFixed(3)),
          gyroX: Number(((Math.random() - 0.5) * 0.02).toFixed(3)),
          gyroY: Number(((Math.random() - 0.5) * 0.02).toFixed(3)),
          gyroZ: Number(((Math.random() - 0.5) * 0.03).toFixed(3)),
          speed: Number((65 + Math.sin(Date.now() / 2000) * 8).toFixed(1)),
          steeringAngle: Number((Math.sin(Date.now() / 1500) * 12).toFixed(1)),
        });
      }, 200);
    }
    return () => clearInterval(interval);
  }, [isSimulating]);

  const checkHardware = async () => {
    setIsScanning(true);
    try {
      const data = await fetchHardwareStatus();
      setHwStatus(data);
    } catch (err) {
      console.error('Failed to query hardware status:', err);
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto bg-white dark:bg-black text-black dark:text-white select-none">
      {/* Header */}
      <div className="spacex-card flex flex-col md:flex-row md:items-center justify-between gap-4 p-5">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-black dark:bg-white text-white dark:text-black flex items-center justify-center">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#6c757d] dark:text-[#a1a1aa] block">
                Telemetry Diagnostics
              </span>
              <h2 className="text-xl font-bold tracking-tight text-black dark:text-white uppercase">
                Vehicle Sensor Telemetry
              </h2>
            </div>
          </div>
          <p className="text-xs text-[#6c757d] dark:text-[#a1a1aa] mt-2 font-mono">
            6-DOF IMU Kinematic Signals, High-Rate Polling Channels &amp; Bus Topology
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => setIsSimulating(!isSimulating)}
            className={`h-9 px-4 gap-2 text-xs font-mono rounded-lg transition-all ${
              isSimulating ? 'bg-black text-white dark:bg-white dark:text-black' : ''
            }`}
          >
            {isSimulating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isSimulating ? 'HALT SIMULATION' : 'SIMULATE STREAM'}</span>
          </Button>
          <Button
            variant="outline"
            onClick={checkHardware}
            disabled={isScanning}
            className="h-9 px-4 gap-2 text-xs font-mono rounded-lg"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isScanning ? 'animate-spin' : ''}`} />
            <span>POLL SENSORS</span>
          </Button>
        </div>
      </div>

      {/* Sensor Telemetry Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
        <div className="spacex-card p-4 space-y-1">
          <span className="text-[10px] uppercase text-[#6c757d] dark:text-[#a1a1aa] block">Acc Longitudinal (X)</span>
          <span className="text-2xl font-bold text-black dark:text-white">{sensorData.accX.toFixed(3)} g</span>
          <span className="text-[10px] text-[#6c757d] dark:text-[#a1a1aa] block">Acceleration / Deceleration</span>
        </div>

        <div className="spacex-card p-4 space-y-1">
          <span className="text-[10px] uppercase text-[#6c757d] dark:text-[#a1a1aa] block">Acc Lateral (Y)</span>
          <span className="text-2xl font-bold text-black dark:text-white">{sensorData.accY.toFixed(3)} g</span>
          <span className="text-[10px] text-[#6c757d] dark:text-[#a1a1aa] block">Cornering Force</span>
        </div>

        <div className="spacex-card p-4 space-y-1">
          <span className="text-[10px] uppercase text-[#6c757d] dark:text-[#a1a1aa] block">Acc Vertical (Z)</span>
          <span className="text-2xl font-bold text-black dark:text-white">{sensorData.accZ.toFixed(3)} g</span>
          <span className="text-[10px] text-[#6c757d] dark:text-[#a1a1aa] block">Gravity Vector Normal</span>
        </div>

        <div className="spacex-card p-4 space-y-1">
          <span className="text-[10px] uppercase text-[#6c757d] dark:text-[#a1a1aa] block">Angular Velocity (Yaw)</span>
          <span className="text-2xl font-bold text-black dark:text-white">{sensorData.gyroZ.toFixed(3)} rad/s</span>
          <span className="text-[10px] text-[#6c757d] dark:text-[#a1a1aa] block">Heading Rate</span>
        </div>
      </div>

      {/* Inspector Modal */}
      {inspectorTarget && (
        <ComponentInspectorModal
          target={inspectorTarget}
          onClose={() => setInspectorTarget(null)}
        />
      )}
    </div>
  );
}

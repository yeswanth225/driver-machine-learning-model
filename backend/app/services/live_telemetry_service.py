"""
Live Telemetry & Physical Device State Service
=============================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: backend/app/services/live_telemetry_service.py

Responsibilities:
- Tracks physical smartphone connection and heartbeat states.
- Maintains in-memory circular buffers for live streaming to frontend dashboards.
- Computes real-time sampling frequency.
- Broadcasts telemetry updates via WebSockets.
- Disconnection detection: transitions from STREAMING -> STANDBY -> DISCONNECTED.
"""

import time
from collections import deque
from typing import Dict, Any, List, Optional
from fastapi import WebSocket

from backend.app.schemas.sensor import (
    DeviceRegistrationRequest,
    DeviceRegistrationResponse,
    TelemetryFrame,
    LiveDeviceStatusResponse,
)


class LiveTelemetryService:
    """Manages active smartphone devices and real-time telemetry streaming."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "LiveTelemetryService":
        if cls._instance is None:
            cls._instance = LiveTelemetryService()
        return cls._instance

    def __init__(self):
        self._registered_devices: Dict[str, Dict[str, Any]] = {}
        self._active_session_id: Optional[str] = None
        self._active_driver_id: Optional[str] = None
        self._active_device_token: Optional[str] = None

        self._last_heartbeat_time = 0.0
        self._recent_frames: deque = deque(maxlen=200)  # ~10s buffer at 20Hz
        self._frame_times: deque = deque(maxlen=40)     # ~2s for rolling Hz calculation
        self._latest_frame: Optional[Dict[str, Any]] = None

        self._active_websockets: List[WebSocket] = []

    def register_device(self, req: DeviceRegistrationRequest) -> DeviceRegistrationResponse:
        """Registers a physical smartphone logger device."""
        token = f"dev_{int(time.time())}_{req.assigned_driver_id}"
        device_info = {
            "token": token,
            "device_name": req.device_name,
            "os_version": req.os_version,
            "app_version": req.app_version,
            "assigned_driver_id": req.assigned_driver_id,
            "registered_at": time.time(),
            "last_seen": time.time(),
        }
        self._registered_devices[token] = device_info
        self._active_device_token = token
        self._active_driver_id = req.assigned_driver_id

        return DeviceRegistrationResponse(
            device_token=token,
            assigned_driver_id=req.assigned_driver_id,
            status="REGISTERED",
            target_frequency_hz=20.0,
            message="Smartphone registered successfully for physical telemetry acquisition.",
        )

    def set_active_session(self, session_id: str, driver_id: str, device_token: Optional[str] = None):
        self._active_session_id = session_id
        self._active_driver_id = driver_id
        if device_token:
            self._active_device_token = device_token
        self._recent_frames.clear()
        self._frame_times.clear()

    def end_active_session(self):
        self._active_session_id = None

    def ingest_frame(self, frame: TelemetryFrame) -> None:
        """Processes an incoming real-time telemetry frame from the physical device."""
        now = time.time()
        self._last_heartbeat_time = now

        frame_dict = frame.model_dump(by_alias=True)
        self._latest_frame = frame_dict
        self._recent_frames.append(frame_dict)
        self._frame_times.append(now)

    async def broadcast_frame(self, frame: TelemetryFrame):
        """Asynchronously dispatches frame to connected WebSockets."""
        self.ingest_frame(frame)
        if not self._active_websockets:
            return

        payload = {
            "type": "TELEMETRY_FRAME",
            "data": self._latest_frame,
            "frequency_hz": self.get_measured_frequency(),
        }

        dead_sockets = []
        for ws in self._active_websockets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_sockets.append(ws)

        for dead in dead_sockets:
            if dead in self._active_websockets:
                self._active_websockets.remove(dead)

    def register_websocket(self, ws: WebSocket):
        self._active_websockets.append(ws)

    def unregister_websocket(self, ws: WebSocket):
        if ws in self._active_websockets:
            self._active_websockets.remove(ws)

    def get_measured_frequency(self) -> float:
        """Computes rolling instantaneous sampling rate from recent frame arrivals."""
        if len(self._frame_times) < 5:
            return 0.0
        times = list(self._frame_times)
        elapsed = times[-1] - times[0]
        if elapsed <= 0.001:
            return 0.0
        return round(float((len(times) - 1) / elapsed), 1)

    def get_live_status(self) -> LiveDeviceStatusResponse:
        """Returns dynamic device connection and telemetry status."""
        now = time.time()
        time_since_heartbeat = now - self._last_heartbeat_time if self._last_heartbeat_time > 0 else 9999.0

        if self._last_heartbeat_time == 0.0 or time_since_heartbeat > 30.0:
            connected = False
            status = "DISCONNECTED"
            quality_status = "NOT_AVAILABLE"
            message = "No physical smartphone currently connected or transmitting telemetry."
            freq = 0.0
        elif time_since_heartbeat <= 5.0:
            connected = True
            status = "STREAMING"
            quality_status = "ACTIVE"
            message = f"Physical telemetry streaming active at {self.get_measured_frequency():.1f} Hz."
            freq = self.get_measured_frequency()
        else:
            connected = True
            status = "STANDBY"
            quality_status = "PAUSED"
            message = f"Device standby (last heartbeat {time_since_heartbeat:.1f}s ago)."
            freq = 0.0

        return LiveDeviceStatusResponse(
            connected=connected,
            status=status,
            device_token=self._active_device_token,
            driver_id=self._active_driver_id,
            session_id=self._active_session_id,
            actual_frequency_hz=freq,
            latest_frame=self._latest_frame,
            last_heartbeat_epoch_ms=int(self._last_heartbeat_time * 1000) if self._last_heartbeat_time > 0 else None,
            quality_status=quality_status,
            message=message,
        )

    def get_recent_frames(self, count: int = 50) -> List[Dict[str, Any]]:
        """Retrieves the most recent N frames for frontend chart rendering."""
        return list(self._recent_frames)[-count:]

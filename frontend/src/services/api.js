/**
 * Centralized API Client for Intelligent Driver Activity Monitoring & Safety Agent.
 * Connects frontend modules to the FastAPI backend service.
 */

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelStatus() {
  const res = await fetch(`${API_BASE}/models/status`);
  if (!res.ok) throw new Error(`Model status check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchScenarios() {
  const res = await fetch(`${API_BASE}/simulated/scenarios`);
  if (!res.ok) throw new Error(`Scenarios fetch failed: ${res.statusText}`);
  return res.json();
}

export async function assessTelemetry(payload) {
  const res = await fetch(`${API_BASE}/telemetry/assess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Telemetry assessment error: ${res.statusText}`);
  }
  return res.json();
}

export async function assessRawFeatures(payload) {
  const res = await fetch(`${API_BASE}/telemetry/assess/raw`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Raw feature inference error: ${res.statusText}`);
  }
  return res.json();
}

export async function resetAgent() {
  const res = await fetch(`${API_BASE}/agent/reset`, { method: 'POST' });
  if (!res.ok) throw new Error(`Agent reset failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReplaySessions() {
  const res = await fetch(`${API_BASE}/replay/sessions`);
  if (!res.ok) throw new Error(`Replay sessions fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReplaySessionDetail(sessionId) {
  const res = await fetch(`${API_BASE}/replay/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(`Replay session detail fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchAnalyticsOverview() {
  const res = await fetch(`${API_BASE}/analytics/overview`);
  if (!res.ok) throw new Error(`Analytics overview fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchUnifiedRegistry() {
  const res = await fetch(`${API_BASE}/analytics/unified-registry`);
  if (!res.ok) throw new Error(`Unified registry fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHardwareStatus() {
  const res = await fetch(`${API_BASE}/system/hardware-status`);
  if (!res.ok) throw new Error(`Hardware status check failed: ${res.statusText}`);
  return res.json();
}

// --- Physical Sensor Acquisition & Live Stream APIs ---

export async function fetchLiveSensorStatus() {
  const res = await fetch(`${API_BASE}/sensor/live/status`);
  if (!res.ok) throw new Error(`Live sensor status check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchRecentTelemetryFrames(count = 50) {
  const res = await fetch(`${API_BASE}/sensor/live/frames?count=${count}`);
  if (!res.ok) throw new Error(`Recent frames fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSensorSessions() {
  const res = await fetch(`${API_BASE}/sensor/sessions`);
  if (!res.ok) throw new Error(`Sensor sessions fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSessionQualityReport(driverId, sessionId) {
  const res = await fetch(`${API_BASE}/sensor/sessions/${encodeURIComponent(driverId)}/${encodeURIComponent(sessionId)}/quality`);
  if (!res.ok) throw new Error(`Session quality report fetch failed: ${res.statusText}`);
  return res.json();
}

export async function uploadSensorCsv(formData) {
  const res = await fetch(`${API_BASE}/sensor/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Sensor CSV upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function createAnnotation(annotationData) {
  const res = await fetch(`${API_BASE}/sensor/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotationData),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Annotation creation failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchAnnotations(sessionId) {
  const res = await fetch(`${API_BASE}/sensor/annotations/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(`Annotations fetch failed: ${res.statusText}`);
  return res.json();
}

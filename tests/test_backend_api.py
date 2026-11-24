"""
Backend FastAPI Endpoint and Integration Tests
==============================================
Validates HTTP API behavior, Pydantic schema validation, model artifact status,
and end-to-end safety agent integration over HTTP.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.agent_service import FROZEN_ARTIFACTS


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestBackendAPI:
    """Test suite for FastAPI backend endpoints."""

    def test_health_check_endpoint(self, client):
        """GET /api/health returns 200 and healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Intelligent Driver Monitoring" in data["service"]

    def test_models_status_endpoint_and_frozen_hashes(self, client):
        """GET /api/models/status returns 200 and all 4 frozen model artifacts verified."""
        response = client.get("/api/models/status")
        assert response.status_code == 200
        data = response.json()
        assert data["all_verified"] is True
        assert len(data["artifacts"]) == 4

        expected_hashes = {a["path"]: a["expected_sha256"] for a in FROZEN_ARTIFACTS}
        for art in data["artifacts"]:
            assert art["exists"] is True
            assert art["is_verified"] is True
            assert art["actual_sha256"] == expected_hashes[art["path"]]

    def test_assess_nominal_highway_cruising(self, client):
        """POST /api/telemetry/assess with nominal inputs returns NORMAL state and 0.0 risk."""
        payload = {
            "timestamp": 1.0,
            "style_prediction": 0,
            "style_confidence": 0.95,
            "maneuver_prediction": 0,
            "maneuver_confidence": 0.98,
            "pipeline_a_health": "HEALTHY",
            "pipeline_b_health": "HEALTHY",
        }
        response = client.post("/api/telemetry/assess", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safety_state"] == "NORMAL"
        assert data["risk_score"] == 0.0
        assert data["pipeline_a_health"] == "HEALTHY"
        assert data["pipeline_b_health"] == "HEALTHY"
        assert len(data["active_alerts"]) == 0

    def test_assess_sudden_braking_triggers_warning(self, client):
        """POST /api/telemetry/assess with harsh braking returns WARNING state and 50.0 risk."""
        payload = {
            "timestamp": 2.0,
            "style_prediction": 0,
            "style_confidence": 0.90,
            "maneuver_prediction": 2,  # Sudden Deceleration
            "maneuver_confidence": 0.90,
        }
        response = client.post("/api/telemetry/assess", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safety_state"] == "WARNING"
        assert data["risk_score"] == 50.0
        assert any("Braking" in a["title"] for a in data["active_alerts"])

    def test_assess_compound_hazard_triggers_high_risk(self, client):
        """POST /api/telemetry/assess with aggressive style + harsh braking returns HIGH_RISK."""
        # 1. Establish aggressive style
        for i in range(5):
            client.post("/api/telemetry/assess", json={
                "timestamp": 10.0 + i * 0.2,
                "style_prediction": 1,
                "style_confidence": 0.95,
                "pipeline_b_health": "QUIESCENT_BYPASS",
            })
        # 2. Inject harsh braking simultaneously
        res = client.post("/api/telemetry/assess", json={
            "timestamp": 11.2,
            "style_prediction": 1,
            "style_confidence": 0.95,
            "maneuver_prediction": 2,
            "maneuver_confidence": 0.95,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["safety_state"] == "HIGH_RISK"
        assert data["risk_score"] >= 75.0
        assert "COMPOUND_HIGH_RISK_HAZARD" in data["triggered_rules"]

    def test_assess_pipeline_a_dropout_option_1a_pure(self, client):
        """POST /api/telemetry/assess with missing Pipeline A during turn preserves calculated state without artificial floor."""
        # Reset agent first
        client.post("/api/agent/reset")
        payload = {
            "timestamp": 1.0,
            "style_prediction": None,
            "style_confidence": None,
            "maneuver_prediction": 3,  # Sudden Right Turn
            "maneuver_confidence": 0.42,  # phi(0.42) = 0.70 -> 35.0 * 0.70 = 24.5 < 25.0 -> NORMAL
            "pipeline_a_health": "MISSING",
            "pipeline_b_health": "HEALTHY",
        }
        response = client.post("/api/telemetry/assess", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safety_state"] == "NORMAL"
        assert data["risk_score"] == pytest.approx(24.5, abs=0.1)
        assert data["pipeline_a_health"] == "MISSING"
        assert any(a["category"] == "DATA_QUALITY" for a in data["active_alerts"])

    def test_assess_pipeline_b_missing_p05a_clamps_to_caution(self, client):
        """POST /api/telemetry/assess with missing Pipeline B clamps nominal driving to CAUTION floor."""
        client.post("/api/agent/reset")
        payload = {
            "timestamp": 1.0,
            "style_prediction": 0,
            "style_confidence": 0.90,
            "maneuver_prediction": None,
            "maneuver_confidence": None,
            "pipeline_a_health": "HEALTHY",
            "pipeline_b_health": "MISSING",
        }
        response = client.post("/api/telemetry/assess", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safety_state"] == "CAUTION"
        assert data["pipeline_a_health"] == "HEALTHY"
        assert data["pipeline_b_health"] == "MISSING"
        assert any(a["category"] == "DATA_QUALITY" for a in data["active_alerts"])

    def test_get_simulated_scenarios(self, client):
        """GET /api/simulated/scenarios returns predefined educational demonstration scenarios."""
        response = client.get("/api/simulated/scenarios")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 6
        assert any(s["id"] == "SCN-01" for s in data)
        assert any(s["id"] == "SCN-02" for s in data)

    def test_agent_reset_endpoint(self, client):
        """POST /api/agent/reset resets state and returns success."""
        response = client.post("/api/agent/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_invalid_payload_returns_422(self, client):
        """Invalid types in payload trigger Pydantic 422 Unprocessable Entity."""
        response = client.post("/api/telemetry/assess", json={"timestamp": "not-a-number"})
        assert response.status_code == 422

    def test_replay_sessions_list_endpoint(self, client):
        """GET /api/replay/sessions returns verified session list."""
        response = client.get("/api/replay/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        session_ids = [s["id"] for s in data]
        assert "session-drv-1" in session_ids
        assert "session-multi-hazard" in session_ids

    def test_replay_session_detail_endpoint(self, client):
        """GET /api/replay/sessions/{session_id} returns frames and model telemetry."""
        response = client.get("/api/replay/sessions/session-drv-1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "session-drv-1"
        assert len(data["frames"]) > 0
        first_frame = data["frames"][0]
        assert "timestamp" in first_frame
        assert "acc_x" in first_frame
        assert "telemetry_request" in first_frame

    def test_analytics_overview_endpoint(self, client):
        """GET /api/analytics/overview returns verified historical metrics."""
        response = client.get("/api/analytics/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pipeline_a" in data
        assert "pipeline_b" in data
        assert data["pipeline_a"]["test_tuned"]["accuracy"] == pytest.approx(0.69925, abs=0.001)

    def test_hardware_status_endpoint(self, client):
        """GET /api/system/hardware-status returns honest disconnected status when hardware absent."""
        response = client.get("/api/system/hardware-status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["status"] == "DISCONNECTED"
        assert len(data["supported_sensors"]) > 0

    def test_root_serves_frontend_dashboard(self, client):
        """GET / serves the cinematic 3D frontend dashboard."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Intelligent Driver Activity Monitoring" in response.text



"""
Tests for Phase 3 Code Reconciliation:
- Task A: High-Risk State Hysteresis (DeterministicRiskEngine)
- Task B: Gate 1 Rate & Jitter Consistency (DataQualityService)
- Task C: Causal 21-tap FIR Decimator (CausalResampler20HzTo10Hz & PhysicalSensorInferenceAdapter)
- Task D: Standalone Vehicle-Frame Alignment (VehicleFrameAligner)
"""

import math
import numpy as np
import pandas as pd
import pytest

from ml.agent.config import (
    DEFAULT_CONFIG,
    SafetyState,
    SafetyAgentConfig,
)
from ml.agent.safety_agent import KinematicEventGate, GateStatus
from ml.agent.risk_engine import DeterministicRiskEngine
from backend.app.services.data_quality_service import DataQualityService
from ml.inference.sampling_compatibility_adapter import (
    FIR_21TAP_COEFFICIENTS,
    CausalResampler20HzTo10Hz,
)
from ml.inference.physical_sensor_adapter import PhysicalSensorInferenceAdapter
from ml.inference.vehicle_frame_alignment import VehicleFrameAligner


# =============================================================================
# TASK A: HIGH-RISK HYSTERESIS TESTS
# =============================================================================

class TestHighRiskHysteresis:
    """Tests for opt-in high-risk threshold hysteresis in DeterministicRiskEngine."""

    def test_default_config_instantaneous_switching(self):
        """Default config (enter=75, exit=75) preserves exact legacy instantaneous boundary."""
        config = SafetyAgentConfig(high_risk_enter_threshold=75.0, high_risk_exit_threshold=75.0)
        engine = DeterministicRiskEngine(config=config)

        # 74.9 is WARNING
        assert engine._map_score_to_state(74.9) == SafetyState.WARNING
        # 75.0 is HIGH_RISK
        assert engine._map_score_to_state(75.0) == SafetyState.HIGH_RISK
        # Dropping to 74.9 immediately returns to WARNING
        assert engine._map_score_to_state(74.9) == SafetyState.WARNING

    def test_opt_in_hysteresis_behavior(self):
        """When exit < enter (e.g. exit=72, enter=75), state holds HIGH_RISK until dropping below 72."""
        config = SafetyAgentConfig(high_risk_enter_threshold=75.0, high_risk_exit_threshold=72.0)
        engine = DeterministicRiskEngine(config=config)

        # Start at 70.0 -> WARNING
        state1 = engine._map_score_to_state(70.0)
        assert state1 == SafetyState.WARNING

        # Cross 75.0 to 76.0 -> Enters HIGH_RISK
        state2 = engine._map_score_to_state(76.0)
        assert state2 == SafetyState.HIGH_RISK

        # Drop to 74.0 -> Should hold HIGH_RISK (74.0 >= 72.0)
        state3 = engine._map_score_to_state(74.0)
        assert state3 == SafetyState.HIGH_RISK

        # Drop to 72.5 -> Still holds HIGH_RISK
        state4 = engine._map_score_to_state(72.5)
        assert state4 == SafetyState.HIGH_RISK

        # Drop to 71.5 -> Exits HIGH_RISK to WARNING (< 72.0)
        state5 = engine._map_score_to_state(71.5)
        assert state5 == SafetyState.WARNING

        # Rise back to 74.0 -> Must NOT re-enter HIGH_RISK (since 74.0 < enter 75.0)
        state6 = engine._map_score_to_state(74.0)
        assert state6 == SafetyState.WARNING

        # Cross 75.1 -> Re-enters HIGH_RISK
        state7 = engine._map_score_to_state(75.1)
        assert state7 == SafetyState.HIGH_RISK

    def test_reset_clears_hysteresis_state(self):
        """Engine reset clears the previous safety state."""
        config = SafetyAgentConfig(high_risk_enter_threshold=75.0, high_risk_exit_threshold=72.0)
        engine = DeterministicRiskEngine(config=config)

        engine._map_score_to_state(76.0)
        assert engine._last_safety_state == SafetyState.HIGH_RISK

        engine.reset()
        assert engine._last_safety_state == SafetyState.NORMAL


# =============================================================================
# TASK B: GATE 1 CONSISTENCY TESTS
# =============================================================================

class TestGate1Consistency:
    """Tests for DataQualityService Gate 1 rate and jitter consistency."""

    def _make_df(self, timestamps, segment_id="S0"):
        n = len(timestamps)
        return pd.DataFrame({
            "timestamp": timestamps,
            "timestamp_epoch_ms": [int(t * 1000) for t in timestamps],
            "Acc X": np.random.normal(0.0, 0.02, n),
            "Acc Y": np.random.normal(0.0, 0.02, n),
            "Acc Z": np.full(n, -1.0) + np.random.normal(0.0, 0.01, n),
            "gyro_x": np.random.normal(0.0, 0.005, n),
            "gyro_y": np.random.normal(0.0, 0.005, n),
            "gyro_z": np.random.normal(0.0, 0.005, n),
            "speed_mps": np.full(n, 15.0),
            "heading_deg": np.full(n, 90.0),
            "latitude": np.full(n, 37.77),
            "longitude": np.full(n, -122.42),
            "altitude": np.full(n, 50.0),
            "hdop": np.full(n, 1.2),
            "segment_id": np.full(n, segment_id),
        })

    def test_nominal_20hz_stream_passes(self):
        """Nominal 20.0 Hz stream (dt = 0.050s) passes Gate 1."""
        service = DataQualityService.get_instance()
        timestamps = [1000.0 + i * 0.050 for i in range(100)]
        df = self._make_df(timestamps)

        report = service.evaluate_session_df(df, driver_id="DRV_002", session_id="test_sess_nom")
        c1 = next(c for c in report.criteria_results if "Criterion 1" in c.criterion_name)
        assert c1.status == "PASS"
        assert 19.9 <= c1.measured_value["mean_freq_hz"] <= 20.1
        assert c1.measured_value["jitter_band_pct"] >= 95.0

    def test_frequency_below_18hz_triggers_warning(self):
        """Stream at 17.0 Hz (outside [18, 22] Hz) triggers WARNING in Gate 1."""
        service = DataQualityService.get_instance()
        dt = 1.0 / 17.0  # ~0.0588s
        timestamps = [1000.0 + i * dt for i in range(100)]
        df = self._make_df(timestamps)

        report = service.evaluate_session_df(df, driver_id="DRV_002", session_id="test_sess_low_hz")
        c1 = next(c for c in report.criteria_results if "Criterion 1" in c.criterion_name)
        assert c1.status == "WARNING"
        assert "Mean frequency" in c1.details

    def test_excessive_jitter_triggers_warning(self):
        """Alternating intervals 0.035s and 0.065s (< 95% in [40, 60] ms) trigger WARNING."""
        service = DataQualityService.get_instance()
        timestamps = [1000.0]
        for i in range(1, 100):
            dt = 0.035 if i % 2 == 0 else 0.065
            timestamps.append(timestamps[-1] + dt)
        df = self._make_df(timestamps)

        report = service.evaluate_session_df(df, driver_id="DRV_002", session_id="test_sess_jitter")
        c1 = next(c for c in report.criteria_results if "Criterion 1" in c.criterion_name)
        assert c1.status == "WARNING"
        assert "sampling jitter" in c1.details

    def test_large_gap_triggers_fail(self):
        """Timestamp gap >= 250 ms triggers hard FAIL."""
        service = DataQualityService.get_instance()
        timestamps = [1000.0 + i * 0.050 for i in range(30)]
        # Add 300 ms gap
        timestamps.append(timestamps[-1] + 0.300)
        for i in range(30):
            timestamps.append(timestamps[-1] + 0.050)
        df = self._make_df(timestamps)

        report = service.evaluate_session_df(df, driver_id="DRV_002", session_id="test_sess_gap")
        c1 = next(c for c in report.criteria_results if "Criterion 1" in c.criterion_name)
        assert c1.status == "FAIL"
        assert "gaps >= 250ms" in c1.details


# =============================================================================
# TASK C: CAUSAL 21-TAP FIR DECIMATOR TESTS
# =============================================================================

class TestCausalFIR21TapDecimator:
    """Tests for FIR 21-tap filter coefficients and causal resampler."""

    def test_fir_coefficients_properties(self):
        """Check 21 taps, symmetry, and unity DC gain."""
        assert len(FIR_21TAP_COEFFICIENTS) == 21

        # Symmetry test: c[i] == c[20-i]
        for i in range(10):
            assert math.isclose(FIR_21TAP_COEFFICIENTS[i], FIR_21TAP_COEFFICIENTS[20 - i], rel_tol=1e-6)

        # Unity DC gain: sum == 1.0
        assert math.isclose(sum(FIR_21TAP_COEFFICIENTS), 1.0, rel_tol=1e-5)

    def test_impulse_response_and_group_delay(self):
        """Impulse response peak occurs exactly at sample index 10 (group delay = 10 samples = 500 ms)."""
        max_coeff = max(FIR_21TAP_COEFFICIENTS)
        assert FIR_21TAP_COEFFICIENTS[10] == max_coeff
        resampler = CausalResampler20HzTo10Hz(filter_type="fir_21tap")
        assert resampler.group_delay_ms == 500.0
        assert resampler.is_research_filter is True

    def test_step_response_unity_gain(self):
        """Constant input reaches unity output once the 21-sample buffer fills."""
        resampler = CausalResampler20HzTo10Hz(filter_type="fir_21tap")

        outputs = []
        for i in range(50):
            sample = {
                "timestamp": 100.0 + i * 0.05,
                "timestamp_epoch_ms": int((100.0 + i * 0.05) * 1000),
                "Acc X": 1.0,
                "Acc Y": 2.0,
                "Acc Z": 3.0,
                "gyro_x": 0.1,
                "gyro_y": 0.2,
                "gyro_z": 0.3,
            }
            out = resampler.process_sample(sample)
            if out is not None:
                outputs.append(out)

        # 50 20Hz samples produce 25 10Hz samples
        assert len(outputs) == 25
        # Once buffer is fully populated (after sample 21, i.e., after output 11)
        for out in outputs[11:]:
            assert math.isclose(out["Acc X"], 1.0, abs_tol=1e-4)
            assert math.isclose(out["Acc Y"], 2.0, abs_tol=1e-4)
            assert math.isclose(out["Acc Z"], 3.0, abs_tol=1e-4)

    def test_physical_adapter_filter_selection(self):
        """PhysicalSensorInferenceAdapter accepts filter_type parameter."""
        adapter_2tap = PhysicalSensorInferenceAdapter(filter_type="2tap")
        assert adapter_2tap.resampler.filter_type == "2tap"
        assert adapter_2tap.resampler.is_research_filter is False

        adapter_fir = PhysicalSensorInferenceAdapter(filter_type="fir_21tap")
        assert adapter_fir.resampler.filter_type == "fir_21tap"
        assert adapter_fir.resampler.is_research_filter is True


# =============================================================================
# TASK D: VEHICLE-FRAME ALIGNMENT TESTS
# =============================================================================

class TestVehicleFrameAlignment:
    """Tests for standalone VehicleFrameAligner module."""

    def test_static_calibration_tilted_gravity(self):
        """Static calibration resolves down unit vector from tilted gravity."""
        aligner = VehicleFrameAligner()

        # Simulate stationary data with phone tilted: [0.2g, 0.4g, -0.8944g], norm = 1.0g
        gx = 0.2
        gy = 0.4
        gz = -math.sqrt(1.0 - gx ** 2 - gy ** 2)

        n = 50
        acc_stationary = np.tile([gx, gy, gz], (n, 1))

        success, conf, msg = aligner.calibrate_static(acc_stationary, gravity_tolerance=0.05)
        assert success is True
        assert aligner.is_static_calibrated is True
        assert conf >= 0.95

        # Down vector should match normalized mean acceleration
        down = aligner.down_vector
        assert math.isclose(down[0], gx, abs_tol=1e-3)
        assert math.isclose(down[1], gy, abs_tol=1e-3)
        assert math.isclose(down[2], gz, abs_tol=1e-3)

    def test_dynamic_calibration_and_orthonormality(self):
        """Dynamic calibration computes forward and right vectors forming orthonormal rotation matrix."""
        aligner = VehicleFrameAligner()

        # Static calibration: phone level (gravity purely in -Z)
        acc_stat = np.tile([0.0, 0.0, -1.0], (50, 1))
        success_stat, _, _ = aligner.calibrate_static(acc_stat)
        assert success_stat is True

        # Dynamic calibration: forward acceleration along phone's +Y axis
        # (e.g. phone mounted upright on windshield facing forward)
        n = 60
        acc_dyn = np.zeros((n, 3))
        acc_dyn[:, 1] = 0.3  # +0.3g along Y
        acc_dyn[:, 2] = -1.0  # gravity along Z
        speed_delta = 5.0  # accelerating forward (+5 m/s)

        success_dyn, conf_dyn, msg_dyn = aligner.calibrate_dynamic(acc_dyn, speed_delta_mps=speed_delta)
        assert success_dyn is True
        assert aligner.is_calibrated is True

        R = aligner.rotation_matrix
        assert R is not None

        # Check orthonormality: R * R^T = I
        identity = np.eye(3)
        assert np.allclose(R @ R.T, identity, atol=1e-5)
        assert np.allclose(R.T @ R, identity, atol=1e-5)
        # Check determinant is +1 (proper rotation, no reflection)
        assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-5)

    def test_coordinate_transformation(self):
        """Transformed acceleration projects gravity purely onto Z_veh (vertical)."""
        aligner = VehicleFrameAligner()

        # Phone mounted rotated: gravity tilted
        gx = 0.2
        gy = 0.4
        gz = -math.sqrt(1.0 - gx ** 2 - gy ** 2)

        acc_stat = np.tile([gx, gy, gz], (50, 1))
        success_stat, _, _ = aligner.calibrate_static(acc_stat)
        assert success_stat is True

        # Dynamic acceleration along horizontal plane orthogonal to down vector
        down = np.array([gx, gy, gz])
        arbitrary_h = np.array([1.0, 0.0, 0.0])
        fwd_unnorm = arbitrary_h - np.dot(arbitrary_h, down) * down
        fwd_dir = fwd_unnorm / np.linalg.norm(fwd_unnorm)

        acc_dyn = np.tile(down + 0.3 * fwd_dir, (50, 1))
        success_dyn, _, _ = aligner.calibrate_dynamic(acc_dyn, speed_delta_mps=3.0)
        assert success_dyn is True

        # Transform purely static gravity vector
        a_veh = aligner.transform_acceleration(gx, gy, gz)

        # In vehicle frame (SAE J670: X=fwd, Y=right, Z=down):
        # X (longitudinal) should be near 0
        # Y (lateral) should be near 0
        # Z (vertical down) should be near +1.0 along u_z
        assert math.isclose(a_veh[0], 0.0, abs_tol=1e-3)
        assert math.isclose(a_veh[1], 0.0, abs_tol=1e-3)
        assert math.isclose(a_veh[2], 1.0, abs_tol=1e-3)

    def test_portrait_flat_to_sae_j670_mapping(self):
        """Simulate phone flat on center console (screen up, top pointing forward)."""
        aligner = VehicleFrameAligner()
        # Stationary gravity purely in -Z (screen faces sky)
        acc_stat = np.tile([0.0, 0.0, -1.0], (50, 1))
        success_stat, _, _ = aligner.calibrate_static(acc_stat)
        assert success_stat is True

        # Dynamic acceleration along +Y (forward along phone screen top)
        acc_dyn = np.zeros((50, 3))
        acc_dyn[:, 1] = 0.35  # forward acceleration
        acc_dyn[:, 2] = -1.0  # gravity
        success_dyn, _, _ = aligner.calibrate_dynamic(acc_dyn, speed_delta_mps=4.0)
        assert success_dyn is True

        # Mapping verification:
        # Device +Y must map to Vehicle +X (Longitudinal)
        # Device +X must map to Vehicle +Y (Lateral)
        # Device -Z must map to Vehicle +Z (Down)
        a_fwd = aligner.transform_acceleration(0.0, 0.35, -1.0)
        assert math.isclose(a_fwd[0], 0.35, abs_tol=1e-3)
        assert math.isclose(a_fwd[1], 0.0, abs_tol=1e-3)
        assert math.isclose(a_fwd[2], 1.0, abs_tol=1e-3)

        a_lat = aligner.transform_acceleration(0.25, 0.0, -1.0)
        assert math.isclose(a_lat[0], 0.0, abs_tol=1e-3)
        assert math.isclose(a_lat[1], 0.25, abs_tol=1e-3)
        assert math.isclose(a_lat[2], 1.0, abs_tol=1e-3)

    def test_portrait_windshield_mount_to_sae_j670_mapping(self):
        """Simulate phone upright in portrait phone holder on windshield (screen facing cabin)."""
        aligner = VehicleFrameAligner()
        # Phone upright: top points skyward (+Y), so gravity is along -Y
        acc_stat = np.tile([0.0, -1.0, 0.0], (50, 1))
        success_stat, _, _ = aligner.calibrate_static(acc_stat)
        assert success_stat is True

        # Forward acceleration points through back of phone (-Z)
        acc_dyn = np.zeros((50, 3))
        acc_dyn[:, 1] = -1.0  # gravity
        acc_dyn[:, 2] = -0.40 # vehicle accelerates forward away from screen
        success_dyn, _, _ = aligner.calibrate_dynamic(acc_dyn, speed_delta_mps=4.0)
        assert success_dyn is True

        # Device -Z must map to Vehicle +X (Longitudinal)
        # Device +X must map to Vehicle +Y (Lateral)
        # Device -Y must map to Vehicle +Z (Down)
        a_fwd = aligner.transform_acceleration(0.0, -1.0, -0.40)
        assert math.isclose(a_fwd[0], 0.40, abs_tol=1e-3)
        assert math.isclose(a_fwd[1], 0.0, abs_tol=1e-3)
        assert math.isclose(a_fwd[2], 1.0, abs_tol=1e-3)


# =============================================================================
# 5. Configuration Fields & Documentation Reconciliation
# =============================================================================

class TestConfigAndDocumentationReconciliation:
    """Verifies Task 2 unused configuration fields and Task 3 factual definitions."""

    def test_unused_and_deprecated_config_fields_accessible(self):
        """All audited configuration fields remain accessible with exact defaults."""
        cfg = DEFAULT_CONFIG
        assert cfg.style_ema_alpha == 0.20
        assert cfg.low_confidence_discount == 0.70
        assert cfg.weight_compound_high_risk == 25.0
        assert cfg.weight_compound_hazard_bonus == 25.0
        assert cfg.threshold_caution == 25.0
        assert cfg.threshold_warning == 50.0
        assert cfg.threshold_high_risk == 75.0
        assert cfg.score_threshold_caution == 25.0
        assert cfg.score_threshold_warning == 50.0
        assert cfg.score_threshold_high_risk == 75.0
        assert cfg.high_risk_enter_threshold == 75.0
        assert cfg.high_risk_exit_threshold == 75.0

    def test_safety_state_thresholds_exact_boundaries(self):
        """Exact state boundaries: [0, 25) NORMAL, [25, 50) CAUTION, [50, 75) WARNING, [75, 100] HIGH_RISK."""
        engine = DeterministicRiskEngine()

        assert engine._map_score_to_state(0.0) == SafetyState.NORMAL
        assert engine._map_score_to_state(24.99) == SafetyState.NORMAL
        assert engine._map_score_to_state(25.00) == SafetyState.CAUTION
        assert engine._map_score_to_state(49.99) == SafetyState.CAUTION
        assert engine._map_score_to_state(50.00) == SafetyState.WARNING
        assert engine._map_score_to_state(74.99) == SafetyState.WARNING
        assert engine._map_score_to_state(75.00) == SafetyState.HIGH_RISK
        assert engine._map_score_to_state(100.00) == SafetyState.HIGH_RISK

    def test_pipeline_b_4_classes_and_gate_generation_of_none(self):
        """Pipeline B has exactly 4 trained classes (SA, SD, SRT, SLT); NONE (0) is gatekeeper-generated."""
        from ml.preprocessing.pipeline_dataset_b import LABEL_MAPPING, LABEL_ABBREVIATIONS
        assert len(LABEL_MAPPING) == 4
        assert set(LABEL_MAPPING.keys()) == {1, 2, 3, 4}
        assert LABEL_ABBREVIATIONS == {1: "SA", 2: "SD", 3: "SRT", 4: "SLT"}

        # Kinematic event gate generates class 0 (None) during quiet motion
        gate = KinematicEventGate(DEFAULT_CONFIG)
        quiet_obs = {
            "acc_horiz_std_3s": 0.05,
            "gyro_z_std_3s": 0.01,
            "acc_mag_ms2": 9.80665,
            "gyro_mag_rads": 0.02,
            "gyro_z_rads": 0.01,
        }
        gate_res = gate.evaluate(quiet_obs)
        assert gate_res.is_event_detected is False
        assert gate_res.status == GateStatus.VALID_QUIESCENT

    def test_physical_warmup_duration_is_5_seconds(self):
        """Physical sensor inference adapter enforces exactly 50 samples (5.0s at 10 Hz) warm-up."""
        from ml.inference.physical_sensor_adapter import PhysicalSensorInferenceAdapter
        adapter = PhysicalSensorInferenceAdapter(use_production_model=True)
        assert adapter.extractor.is_warmed_up is False

        # Feed 98 raw 20 Hz samples (yielding 49 decimated 10 Hz samples, < 50)
        dummy_row = {"timestamp": 0.0, "Acc X": 0.0, "Acc Y": 0.0, "Acc Z": -1.0, "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0}
        for _ in range(98):
            adapter.process_raw_20hz_sample(dummy_row)
        assert adapter.extractor.is_warmed_up is False

        # Feed 2 more raw samples (reaches 50 decimated samples = 5.0 seconds)
        adapter.process_raw_20hz_sample(dummy_row)
        res = adapter.process_raw_20hz_sample(dummy_row)
        assert adapter.extractor.is_warmed_up is True
        assert res["status"] != "WARMING_UP"


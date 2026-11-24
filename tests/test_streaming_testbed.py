"""
Unit and Integration Tests for Boundary-Independent Streaming Testbed
=====================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 7 - Phase 1: Boundary-Independent Streaming Testbed
"""

import math
import pytest
import numpy as np
import pandas as pd

from ml.agent.config import DEFAULT_CONFIG
from ml.agent.safety_agent import KinematicEventGate
from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier
from ml.evaluation.streaming_testbed import (
    StreamingDetectorConfig,
    StreamingObservation,
    GroundTruthEvent,
    PredictedEvent,
    StreamingManeuverDetector,
    StreamingReplayEngine,
    StreamingEventEvaluator,
    DetectorState,
)


@pytest.fixture(scope="module")
def classifier_b():
    """Load production Pipeline B Extra Trees classifier."""
    return DatasetBManeuverClassifier()


@pytest.fixture
def nominal_features(classifier_b):
    """Generate a baseline feature dictionary where all kinematic signals are quiescent (nominal cruising)."""
    feats = {}
    for name in classifier_b.feature_names:
        if "acc_mag" in name or name in ("AccZ", "acc_z_ms2"):
            feats[name] = 9.80665 if "_ms2" in name else (1.0 if "AccZ" in name or "_g" in name else 9.80665)
        elif "rms" in name and "acc" in name:
            feats[name] = 9.80665
        else:
            feats[name] = 0.0
    return feats


@pytest.fixture
def excited_features(nominal_features):
    """Generate a feature dictionary with significant lateral dynamic acceleration and yaw excursion."""
    feats = dict(nominal_features)
    feats["acc_horiz_ms2"] = 3.50
    feats["acc_horiz_std_3s"] = 0.85
    feats["gyro_z_rads"] = 0.45
    feats["gyro_z_std_3s"] = 0.18
    feats["acc_mag_ms2"] = 12.50
    feats["gyro_mag_rads"] = 0.50
    return feats


class TestStreamingReplayEngine:
    """Tests verifying continuous chronological replay, boundary isolation, and deterministic operation."""

    def test_continuous_chronological_replay(self):
        """Verify stream timestamps are monotonically strictly increasing at nominal 10 Hz."""
        replay = StreamingReplayEngine(sampling_rate_hz=10.0, quiescent_gap_seconds=1.0)
        observations, ground_truth = replay.generate_stream(driver_id="Driver-1")

        assert len(observations) > 0
        assert len(ground_truth) == 35  # Driver-1 has 35 events

        timestamps = [obs.timestamp for obs in observations]
        # Verify strict monotonicity
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            assert round(dt, 3) == 0.100, f"Expected 0.100s dt, got {dt} at step {i}"

    def test_no_future_sample_access_and_single_sample_emission(self):
        """Verify observations are emitted one sample at a time without future lookahead."""
        replay = StreamingReplayEngine(sampling_rate_hz=10.0)
        observations, _ = replay.generate_stream(driver_id="Driver-1")

        for obs in observations[:50]:
            assert isinstance(obs, StreamingObservation)
            assert isinstance(obs.timestamp, float)
            assert isinstance(obs.features, dict)
            # Ensure features is a flat dict, not containing sequence arrays or lists
            for val in obs.features.values():
                assert not isinstance(val, (list, tuple, np.ndarray, pd.Series))

    def test_no_ground_truth_boundary_access_during_prediction(self):
        """Verify ground-truth labels and event boundaries are strictly excluded from StreamingObservation."""
        replay = StreamingReplayEngine(sampling_rate_hz=10.0)
        observations, _ = replay.generate_stream(driver_id="Driver-1")

        forbidden_keys = {
            "Class", "event_id", "label_name", "label_abbr",
            "event_sample_idx", "event_duration_samples", "event_progress_ratio",
            "split"
        }

        for obs in observations[:50]:
            present_forbidden = set(obs.features.keys()) & forbidden_keys
            assert len(present_forbidden) == 0, f"Leaked ground truth in features: {present_forbidden}"


class TestStreamingManeuverDetector:
    """Tests verifying state transitions, onset confirmation, completion hysteresis, and fault recovery."""

    def test_event_onset_detection_and_transient_spike_rejection(self, classifier_b, nominal_features, excited_features):
        """Verify that a single-frame spike is rejected while 2 consecutive frames trigger candidate onset."""
        detector = StreamingManeuverDetector(
            classifier=classifier_b,
            config=StreamingDetectorConfig(min_onset_consecutive_samples=2)
        )

        # Frame 1: Nominal cruising
        obs1 = StreamingObservation(timestamp=0.1, features=nominal_features)
        assert detector.process_sample(obs1) is None
        assert detector.state == DetectorState.IDLE

        # Frame 2: Transient 1-frame excited spike
        obs2 = StreamingObservation(timestamp=0.2, features=excited_features)
        assert detector.process_sample(obs2) is None
        assert detector.state == DetectorState.CANDIDATE_ONSET

        # Frame 3: Immediate drop back to quiescent -> should revert to IDLE
        obs3 = StreamingObservation(timestamp=0.3, features=nominal_features)
        assert detector.process_sample(obs3) is None
        assert detector.state == DetectorState.IDLE

        # Frame 4 & 5: Sustained excitation -> should confirm onset and ACCUMULATE
        obs4 = StreamingObservation(timestamp=0.4, features=excited_features)
        detector.process_sample(obs4)
        obs5 = StreamingObservation(timestamp=0.5, features=excited_features)
        detector.process_sample(obs5)
        assert detector.state == DetectorState.ACCUMULATING

    def test_event_completion_hysteresis_and_emission(self, classifier_b, nominal_features, excited_features):
        """Verify that event accumulates and completes after completion_quiet_seconds hysteresis."""
        detector = StreamingManeuverDetector(
            classifier=classifier_b,
            config=StreamingDetectorConfig(
                min_onset_consecutive_samples=2,
                completion_quiet_seconds=0.30,
                min_event_duration_seconds=0.40,
                cooldown_seconds=0.50
            )
        )

        # 6 consecutive excited samples (0.6s)
        for i in range(6):
            obs = StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features)
            res = detector.process_sample(obs)
            assert res is None  # Still accumulating

        assert detector.state == DetectorState.ACCUMULATING

        # Quiet frame 1 (0.1s quiet)
        obs_q1 = StreamingObservation(timestamp=0.7, features=nominal_features)
        assert detector.process_sample(obs_q1) is None

        # Quiet frame 2 (0.2s quiet)
        obs_q2 = StreamingObservation(timestamp=0.8, features=nominal_features)
        assert detector.process_sample(obs_q2) is None

        # Quiet frame 3 (0.3s quiet -> reaches completion_quiet_seconds=0.30)
        obs_q3 = StreamingObservation(timestamp=0.9, features=nominal_features)
        emitted = detector.process_sample(obs_q3)

        assert emitted is not None
        assert isinstance(emitted, PredictedEvent)
        assert emitted.predicted_class_id in [1, 2, 3, 4]
        assert emitted.duration_seconds >= 0.40
        assert detector.state == DetectorState.COOLDOWN

    def test_duplicate_suppression_during_cooldown(self, classifier_b, nominal_features, excited_features):
        """Verify that excited frames occurring within the cooldown window do not trigger duplicate events."""
        detector = StreamingManeuverDetector(
            classifier=classifier_b,
            config=StreamingDetectorConfig(
                completion_quiet_seconds=0.20,
                cooldown_seconds=0.50
            )
        )

        # Accumulate and complete event
        for i in range(6):
            detector.process_sample(StreamingObservation(timestamp=0.1 + i * 0.1, features=excited_features))
        detector.process_sample(StreamingObservation(timestamp=0.7, features=nominal_features))
        emitted = detector.process_sample(StreamingObservation(timestamp=0.8, features=nominal_features))
        assert emitted is not None
        assert detector.state == DetectorState.COOLDOWN

        # Excitation within cooldown window (t = 1.0 < cooldown_until = 1.3)
        obs_in_cd = StreamingObservation(timestamp=1.0, features=excited_features)
        assert detector.process_sample(obs_in_cd) is None
        assert detector.state == DetectorState.COOLDOWN

        # Excitation after cooldown expires (t = 1.4 > 1.3)
        obs_after_cd = StreamingObservation(timestamp=1.4, features=excited_features)
        detector.process_sample(obs_after_cd)
        assert detector.state == DetectorState.CANDIDATE_ONSET

    def test_sensor_gap_and_dropout_recovery(self, classifier_b, nominal_features, excited_features):
        """Verify that a sudden timestamp leap (> 2.0s) safely finalizes in-flight event and resets state."""
        detector = StreamingManeuverDetector(
            classifier=classifier_b,
            config=StreamingDetectorConfig(min_event_duration_seconds=0.30)
        )

        # Accumulate 5 frames (0.5s duration)
        for i in range(5):
            detector.process_sample(StreamingObservation(timestamp=0.1 + i * 0.1, features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # Timestamp leap of 5.0 seconds (e.g. sensor disconnect/reconnect)
        obs_leap = StreamingObservation(timestamp=5.5, features=nominal_features)
        emitted = detector.process_sample(obs_leap)

        # In-flight event was finalized and emitted
        assert emitted is not None
        assert emitted.end_time <= 0.6
        # Detector state reset to IDLE
        assert detector.state == DetectorState.IDLE

    def test_invalid_observations_handling(self, classifier_b, nominal_features, excited_features):
        """Verify invalid or corrupted observations (NaN, inf, empty) do not crash the detector."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # Corrupted dict with NaN
        bad_nan = dict(nominal_features)
        bad_nan["acc_mag_ms2"] = float("nan")
        obs_nan = StreamingObservation(timestamp=0.1, features=bad_nan)
        assert detector.process_sample(obs_nan) is None
        assert detector.state == DetectorState.IDLE

        # Empty dict
        obs_empty = StreamingObservation(timestamp=0.2, features={})
        assert detector.process_sample(obs_empty) is None

        # None values
        bad_none = dict(nominal_features)
        bad_none["acc_horiz_std_3s"] = None
        obs_none = StreamingObservation(timestamp=0.3, features=bad_none)
        assert detector.process_sample(obs_none) is None

    def test_empty_stream_and_single_sample_stream(self, classifier_b, nominal_features, excited_features):
        """Verify edge cases: empty stream and single-sample stream."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # Flush empty detector
        assert detector.flush() is None

        # Single-sample stream
        obs = StreamingObservation(timestamp=0.1, features=excited_features)
        assert detector.process_sample(obs) is None
        # Flush single sample (below min duration 0.4s) -> discarded cleanly
        assert detector.flush() is None

    def test_deterministic_replay(self, classifier_b):
        """Verify two separate runs over the exact same stream produce identical predictions."""
        replay = StreamingReplayEngine(sampling_rate_hz=10.0, quiescent_gap_seconds=1.0)
        obs1, _ = replay.generate_stream(driver_id="Driver-1")
        obs2, _ = replay.generate_stream(driver_id="Driver-1")

        detector1 = StreamingManeuverDetector(classifier=classifier_b)
        detector2 = StreamingManeuverDetector(classifier=classifier_b)

        preds1 = []
        for o in obs1:
            p = detector1.process_sample(o)
            if p is not None:
                preds1.append(p)

        preds2 = []
        for o in obs2:
            p = detector2.process_sample(o)
            if p is not None:
                preds2.append(p)

        p1_flush = detector1.flush()
        if p1_flush:
            preds1.append(p1_flush)
        p2_flush = detector2.flush()
        if p2_flush:
            preds2.append(p2_flush)

        assert len(preds1) == len(preds2)
        for e1, e2 in zip(preds1, preds2):
            assert e1.start_time == e2.start_time
            assert e1.end_time == e2.end_time
            assert e1.predicted_class_id == e2.predicted_class_id
            assert math.isclose(e1.confidence, e2.confidence, abs_tol=1e-5)


class TestStreamingEventEvaluator:
    """Tests verifying IoU calculation, bipartite matching, latency extraction, and metric calculation."""

    def test_temporal_iou_calculation(self):
        """Verify IoU calculation on known intervals."""
        evaluator = StreamingEventEvaluator()

        # Identical intervals: IoU = 1.0
        assert math.isclose(evaluator.compute_temporal_iou(1.0, 2.0, 1.0, 2.0), 1.0)

        # Disjoint intervals: IoU = 0.0
        assert math.isclose(evaluator.compute_temporal_iou(1.0, 2.0, 3.0, 4.0), 0.0)

        # 50% overlap: [1.0, 3.0] and [2.0, 4.0] -> intersection=1.0, union=3.0 -> IoU = 1/3
        assert math.isclose(evaluator.compute_temporal_iou(1.0, 3.0, 2.0, 4.0), 1.0 / 3.0)

    def test_bipartite_matching_and_latency_measurement(self):
        """Verify evaluation matching, latency calculation, and false event rates on synthetic events."""
        evaluator = StreamingEventEvaluator(iou_threshold=0.40, onset_tolerance_seconds=0.50)

        gt_events = [
            GroundTruthEvent(
                event_id=1, driver_id="Driver-1", class_id=1, label_name="Sudden Acceleration",
                label_abbr="SA", start_time=1.0, end_time=2.0, duration_seconds=1.0, sample_count=10
            ),
            GroundTruthEvent(
                event_id=2, driver_id="Driver-1", class_id=3, label_name="Sudden Right Turn",
                label_abbr="SRT", start_time=5.0, end_time=6.0, duration_seconds=1.0, sample_count=10
            )
        ]

        pred_events = [
            # Matches GT 1 with onset latency = 0.2s, provisional latency = 0.4s, completion latency = 0.3s
            PredictedEvent(
                pred_id=1, start_time=1.1, end_time=2.1, duration_seconds=1.0, sample_count=10,
                predicted_class_id=1, label_name="Sudden Acceleration", label_abbr="SA",
                confidence=0.88, probabilities={"SA": 0.88, "SD": 0.04, "SRT": 0.04, "SLT": 0.04},
                onset_detected_at=1.2, first_provisional_at=1.4, first_provisional_class_id=1,
                first_provisional_confidence=0.85, final_decision_at=2.4, completion_detected_at=2.3
            ),
            # Spurious false event at t=3.5s
            PredictedEvent(
                pred_id=2, start_time=3.5, end_time=4.0, duration_seconds=0.5, sample_count=5,
                predicted_class_id=2, label_name="Sudden Deceleration", label_abbr="SD",
                confidence=0.75, probabilities={"SA": 0.05, "SD": 0.75, "SRT": 0.10, "SLT": 0.10},
                onset_detected_at=3.6, first_provisional_at=3.8, first_provisional_class_id=2,
                first_provisional_confidence=0.70, final_decision_at=4.3, completion_detected_at=4.3
            )
        ]

        res = evaluator.evaluate(pred_events, gt_events, total_duration_seconds=60.0)

        # GT 1 matched, GT 2 missed (FN=1), Pred 2 is spurious (FP=1), TP=1
        summary = res["summary"]
        assert summary["true_positives"] == 1
        assert summary["false_positives"] == 1
        assert summary["false_negatives"] == 1
        assert math.isclose(summary["detection_precision"], 0.50)
        assert math.isclose(summary["detection_recall"], 0.50)
        assert summary["false_events_per_minute"] == 1.0  # 1 FP in 1 minute

        # Verify latencies for matched event
        lat = res["latencies_seconds"]
        assert math.isclose(lat["candidate_onset_detection"]["mean"], 0.20, abs_tol=1e-3)
        assert math.isclose(lat["first_provisional_classification"]["mean"], 0.40, abs_tol=1e-3)
        assert math.isclose(lat["final_classification"]["mean"], 1.40, abs_tol=1e-3)
        assert math.isclose(lat["event_completion_detection"]["mean"], 0.30, abs_tol=1e-3)


class TestBoundaryCorrectionAndLifecycleInvariants:
    """Stage 7 Phase 1 Boundary Correction and Event Lifecycle Regression Suite."""

    def test_boundary_stub_rejection_with_quiet_hysteresis(self, classifier_b, nominal_features, excited_features):
        """
        Verify that an undersized active stub (< 4 samples, < 0.40s) followed by
        trailing quiet frames is correctly rejected, preventing boundary edge leakage.
        """
        detector = StreamingManeuverDetector(
            classifier=classifier_b,
            config=StreamingDetectorConfig(
                min_onset_consecutive_samples=2,
                completion_quiet_seconds=0.30,
                min_event_duration_seconds=0.40,
                cooldown_seconds=0.50
            )
        )

        # 2 excited frames (0.1s active duration, 2 active samples < 4)
        detector.process_sample(StreamingObservation(timestamp=0.1, features=excited_features))
        detector.process_sample(StreamingObservation(timestamp=0.2, features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # 3 trailing quiet frames (quiet hysteresis fired: 3 quiet samples)
        # Total buffer length is now 2 + 3 = 5 samples
        detector.process_sample(StreamingObservation(timestamp=0.3, features=nominal_features))
        detector.process_sample(StreamingObservation(timestamp=0.4, features=nominal_features))
        emitted = detector.process_sample(StreamingObservation(timestamp=0.5, features=nominal_features))

        # Because active_len is 2 (< 4) and duration is 0.1s (< 0.4s), it must be rejected!
        assert emitted is None, "Sub-threshold stub was erroneously emitted as an event!"
        assert detector.state == DetectorState.COOLDOWN

    def test_event_lifecycle_invariants_and_timestamp_duration_verification(self, classifier_b, nominal_features, excited_features):
        """
        Verify that a valid event satisfies all mathematical and lifecycle invariants:
        - duration_seconds == round(end_time - start_time, 3)
        - sample_count == active_len (excluding quiet frames)
        - end_time == start_time + (sample_count - 1) * dt
        - onset_detected_at <= final_decision_at
        - completion_detected_at >= end_time
        """
        detector = StreamingManeuverDetector(
            classifier=classifier_b,
            config=StreamingDetectorConfig(
                min_onset_consecutive_samples=2,
                completion_quiet_seconds=0.30,
                min_event_duration_seconds=0.40,
                cooldown_seconds=0.50
            )
        )

        # 6 excited frames at 10 Hz (t = 1.0 to 1.5, active_len = 6, duration = 0.5s)
        for i in range(6):
            detector.process_sample(StreamingObservation(timestamp=round(1.0 + i * 0.1, 2), features=excited_features))

        # 3 quiet frames (t = 1.6, 1.7, 1.8)
        detector.process_sample(StreamingObservation(timestamp=1.6, features=nominal_features))
        detector.process_sample(StreamingObservation(timestamp=1.7, features=nominal_features))
        event = detector.process_sample(StreamingObservation(timestamp=1.8, features=nominal_features))

        assert event is not None
        assert event.sample_count == 6
        assert event.start_time == 1.0
        assert event.end_time == 1.5
        assert event.duration_seconds == 0.5
        assert math.isclose(event.duration_seconds, round(event.end_time - event.start_time, 3))
        assert math.isclose(event.end_time, round(event.start_time + (event.sample_count - 1) * 0.1, 3))
        assert event.onset_detected_at <= event.final_decision_at
        assert event.completion_detected_at >= event.end_time


class TestFlushLifecycleAndDurationSemantics:
    """Stage 7 Phase 1 Flush Lifecycle and Minimum-Duration Semantics Test Suite."""

    def test_valid_active_event_finalized_at_end_of_stream(self, classifier_b, nominal_features, excited_features):
        """Verify that a valid active in-flight event (>= 4 samples) is emitted when stream terminates and flush() is called."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # 6 excited frames (0.5s duration, 6 active samples)
        for i in range(6):
            detector.process_sample(StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # Stream ends without quiet hysteresis -> flush() called
        emitted = detector.flush()

        assert emitted is not None
        assert isinstance(emitted, PredictedEvent)
        assert emitted.sample_count == 6
        assert emitted.duration_seconds == 0.5
        assert emitted.start_time == 0.1
        assert emitted.end_time == 0.6
        assert detector.state == DetectorState.IDLE

    def test_detector_state_is_idle_after_flush(self, classifier_b, nominal_features, excited_features):
        """Verify detector state transitions to IDLE and buffers are cleared after flush()."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        for i in range(5):
            detector.process_sample(StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        detector.flush()

        assert detector.state == DetectorState.IDLE
        assert len(detector._buffer) == 0
        assert len(detector._timestamps) == 0
        assert detector._onset_candidate_ts is None
        assert detector._consecutive_excited == 0
        assert detector._quiet_samples_count == 0

    def test_repeated_flush_calls_prevent_duplicate_emission(self, classifier_b, nominal_features, excited_features):
        """Verify that calling flush() multiple times in succession emits the event once and returns None thereafter."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        for i in range(6):
            detector.process_sample(StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features))

        first_flush = detector.flush()
        assert first_flush is not None
        assert detector.state == DetectorState.IDLE

        second_flush = detector.flush()
        assert second_flush is None
        assert detector.state == DetectorState.IDLE

        third_flush = detector.flush()
        assert third_flush is None
        assert detector.state == DetectorState.IDLE

    def test_observation_arrival_and_reuse_contract_after_flush(self, classifier_b, nominal_features, excited_features):
        """
        Verify the detector reuse contract: after flush(), new observations arrive cleanly.
        Nominal observations keep state IDLE; subsequent excited observations initiate a new maneuver lifecycle.
        """
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # Event 1
        for i in range(6):
            detector.process_sample(StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features))
        ev1 = detector.flush()
        assert ev1 is not None
        assert detector.state == DetectorState.IDLE

        # Feed nominal observations after flush -> processed safely without error, remains IDLE
        res_nom = detector.process_sample(StreamingObservation(timestamp=1.0, features=nominal_features))
        assert res_nom is None
        assert detector.state == DetectorState.IDLE

        # Feed new excited observations for Event 2 (t = 2.0 to 2.5)
        for i in range(6):
            detector.process_sample(StreamingObservation(timestamp=round(2.0 + i * 0.1, 2), features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # Quiet hysteresis to complete Event 2
        detector.process_sample(StreamingObservation(timestamp=2.6, features=nominal_features))
        detector.process_sample(StreamingObservation(timestamp=2.7, features=nominal_features))
        ev2 = detector.process_sample(StreamingObservation(timestamp=2.8, features=nominal_features))

        assert ev2 is not None
        assert ev2.pred_id == ev1.pred_id + 1
        assert ev2.start_time == 2.0
        assert ev2.end_time == 2.5
        assert ev2.sample_count == 6

    def test_minimum_duration_below_boundary_rejected(self, classifier_b, nominal_features, excited_features):
        """Below boundary: 3 active samples (duration = 0.20s < 0.40s, active_len = 3 < 4) must be REJECTED."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # 3 excited samples
        detector.process_sample(StreamingObservation(timestamp=0.1, features=excited_features))
        detector.process_sample(StreamingObservation(timestamp=0.2, features=excited_features))
        detector.process_sample(StreamingObservation(timestamp=0.3, features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # 3 quiet hysteresis samples
        detector.process_sample(StreamingObservation(timestamp=0.4, features=nominal_features))
        detector.process_sample(StreamingObservation(timestamp=0.5, features=nominal_features))
        emitted = detector.process_sample(StreamingObservation(timestamp=0.6, features=nominal_features))

        assert emitted is None, "3-sample sub-threshold burst was erroneously emitted!"

    def test_minimum_duration_at_boundary_accepted(self, classifier_b, nominal_features, excited_features):
        """At boundary: 4 active samples (duration = 0.30s, active_len = 4 samples == min_samples) must be ACCEPTED."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # 4 excited samples (t = 0.1, 0.2, 0.3, 0.4 -> duration = 0.30s, active_len = 4)
        for i in range(4):
            detector.process_sample(StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # 3 quiet hysteresis samples
        detector.process_sample(StreamingObservation(timestamp=0.5, features=nominal_features))
        detector.process_sample(StreamingObservation(timestamp=0.6, features=nominal_features))
        emitted = detector.process_sample(StreamingObservation(timestamp=0.7, features=nominal_features))

        assert emitted is not None, "4-sample boundary event was erroneously rejected!"
        assert emitted.sample_count == 4
        assert emitted.duration_seconds == 0.3

    def test_minimum_duration_above_boundary_accepted(self, classifier_b, nominal_features, excited_features):
        """Above boundary: 5 active samples (duration = 0.40s >= min_duration, active_len = 5 > 4) must be ACCEPTED."""
        detector = StreamingManeuverDetector(classifier=classifier_b)

        # 5 excited samples (t = 0.1 to 0.5 -> duration = 0.40s, active_len = 5)
        for i in range(5):
            detector.process_sample(StreamingObservation(timestamp=round(0.1 + i * 0.1, 2), features=excited_features))
        assert detector.state == DetectorState.ACCUMULATING

        # 3 quiet hysteresis samples
        detector.process_sample(StreamingObservation(timestamp=0.6, features=nominal_features))
        detector.process_sample(StreamingObservation(timestamp=0.7, features=nominal_features))
        emitted = detector.process_sample(StreamingObservation(timestamp=0.8, features=nominal_features))

        assert emitted is not None, "5-sample above-boundary event was erroneously rejected!"
        assert emitted.sample_count == 5
        assert emitted.duration_seconds == 0.4



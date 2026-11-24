"""
Boundary-Independent Streaming Evaluation Testbed for Pipeline B
================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 7 - Phase 1: Streaming Maneuver Classification Testbed

This module implements an isolated, boundary-independent streaming evaluation
framework for Pipeline B (Maneuver Classification). It simulates continuous
10 Hz telemetry streams, feeds individual observations sequentially without
exposing ground-truth event boundaries or future samples to the predictor,
detects and accumulates maneuver candidates via KinematicEventGate, and
rigorously evaluates event detection and classification latency and accuracy.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml.agent.config import (
    SafetyAgentConfig,
    DEFAULT_CONFIG,
    GateStatus,
    MANEUVER_LABEL_NAMES,
    MANEUVER_ABBREVIATIONS,
)
from ml.agent.safety_agent import KinematicEventGate, GateResult
from ml.inference.dataset_b_classifier import (
    DatasetBManeuverClassifier,
    LABEL_MAPPING,
    LABEL_ABBR,
    QUARANTINED_FEATURES,
)


class DetectorState(str, Enum):
    """Internal state machine states for streaming maneuver detection."""
    IDLE = "IDLE"
    CANDIDATE_ONSET = "CANDIDATE_ONSET"
    ACCUMULATING = "ACCUMULATING"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class StreamingDetectorConfig:
    """
    Configuration parameters for the streaming maneuver detector.
    All experimental thresholds are isolated here and do not modify production config.
    """
    sampling_rate_hz: float = 10.0
    # Minimum consecutive excited frames required to confirm candidate onset (suppresses single-frame spikes)
    min_onset_consecutive_samples: int = 2
    # Quiescent duration required to declare maneuver completed (reset hysteresis)
    completion_quiet_seconds: float = 0.30
    # Minimum duration required for an accumulated sequence to qualify as a maneuver
    min_event_duration_seconds: float = 0.40
    # Maximum allowable duration before forced event finalization
    max_event_duration_seconds: float = 2.50
    # Accumulated sample count required to emit first provisional classification
    provisional_window_samples: int = 3
    # Refractory period following event completion to prevent immediate duplicate re-triggering
    cooldown_seconds: float = 0.50
    # Base kinematic gate configuration (uses production defaults: acc_dev=1.47, gyro_mag=0.25, etc.)
    gate_config: SafetyAgentConfig = field(default_factory=lambda: DEFAULT_CONFIG)


@dataclass(frozen=True)
class StreamingObservation:
    """
    Single 10 Hz telemetry observation presented to the predictor.
    Ground-truth labels and event boundaries are strictly excluded.
    """
    timestamp: float
    features: Dict[str, Any]


@dataclass(frozen=True)
class GroundTruthEvent:
    """
    Isolated ground-truth maneuver annotation used exclusively by the evaluator.
    Never exposed to the detector during streaming replay.
    """
    event_id: int
    driver_id: str
    class_id: int
    label_name: str
    label_abbr: str
    start_time: float
    end_time: float
    duration_seconds: float
    sample_count: int


@dataclass
class PredictedEvent:
    """
    Independent maneuver event emitted by the streaming detector.
    Includes classification decision, confidence distribution, and latency timestamps.
    """
    pred_id: int
    start_time: float
    end_time: float
    duration_seconds: float
    sample_count: int
    predicted_class_id: int
    label_name: str
    label_abbr: str
    confidence: float
    probabilities: Dict[str, float]
    # Latency tracking timestamps
    onset_detected_at: float
    first_provisional_at: Optional[float]
    first_provisional_class_id: Optional[int]
    first_provisional_confidence: Optional[float]
    final_decision_at: float
    completion_detected_at: float


class StreamingManeuverDetector:
    """
    Boundary-independent streaming detector that processes one observation at a time,
    maintains an internal state machine (IDLE -> CANDIDATE_ONSET -> ACCUMULATING -> COOLDOWN),
    accumulates kinematic evidence via KinematicEventGate, generates provisional predictions,
    and finalizes maneuver events upon completion.
    """

    def __init__(
        self,
        classifier: DatasetBManeuverClassifier,
        config: Optional[StreamingDetectorConfig] = None
    ):
        self.classifier = classifier
        self.config = config or StreamingDetectorConfig()
        self.gate = KinematicEventGate(config=self.config.gate_config)
        self.reset()

    def reset(self):
        """Reset internal state machine, buffers, and tracking variables."""
        self.state = DetectorState.IDLE
        self._current_pred_id: int = 0
        self._buffer: List[Dict[str, Any]] = []
        self._timestamps: List[float] = []
        self._onset_candidate_ts: Optional[float] = None
        self._consecutive_excited: int = 0
        self._quiet_samples_count: int = 0
        self._onset_confirmed_ts: Optional[float] = None
        self._provisional_ts: Optional[float] = None
        self._provisional_class: Optional[int] = None
        self._provisional_conf: Optional[float] = None
        self._cooldown_until: float = -1.0
        self._last_processed_ts: Optional[float] = None

    def process_sample(self, obs: StreamingObservation) -> Optional[PredictedEvent]:
        """
        Process a single streaming observation.
        Returns a PredictedEvent if an event completed on this sample, otherwise None.
        """
        ts = obs.timestamp
        features = obs.features

        # Check for temporal backwards leap or severe gap
        if self._last_processed_ts is not None:
            dt = ts - self._last_processed_ts
            if dt < 0:
                # Monotonic clock violation; ignore or reset
                return None
            if dt > 2.0:
                # Discontinuity / sensor dropout: finalize in-flight event if valid, then reset
                emitted = self.flush()
                self.reset()
                self._last_processed_ts = ts
                return emitted

        self._last_processed_ts = ts

        # Check cooldown state
        if self.state == DetectorState.COOLDOWN:
            if ts >= self._cooldown_until:
                self.state = DetectorState.IDLE
            else:
                return None

        # Evaluate current observation with KinematicEventGate
        gate_res: GateResult = self.gate.evaluate(features)
        is_excited = (gate_res.status == GateStatus.VALID_EXCITED)
        is_valid = (gate_res.status != GateStatus.INVALID_OR_INSUFFICIENT)

        if not is_valid:
            # Corrupted observation: if accumulating, finalize or abort
            if self.state == DetectorState.ACCUMULATING:
                emitted = self._finalize_event(ts)
                self.state = DetectorState.IDLE
                return emitted
            return None

        if self.state == DetectorState.IDLE:
            if is_excited:
                self.state = DetectorState.CANDIDATE_ONSET
                self._onset_candidate_ts = ts
                self._consecutive_excited = 1
                self._buffer = [features]
                self._timestamps = [ts]
                self._quiet_samples_count = 0

                if self._consecutive_excited >= self.config.min_onset_consecutive_samples:
                    self.state = DetectorState.ACCUMULATING
                    self._onset_confirmed_ts = ts
            return None

        elif self.state == DetectorState.CANDIDATE_ONSET:
            if is_excited:
                self._consecutive_excited += 1
                self._buffer.append(features)
                self._timestamps.append(ts)
                if self._consecutive_excited >= self.config.min_onset_consecutive_samples:
                    self.state = DetectorState.ACCUMULATING
                    self._onset_confirmed_ts = ts
            else:
                # Transient 1-frame spike: revert to IDLE without emitting event
                self.state = DetectorState.IDLE
                self._buffer.clear()
                self._timestamps.clear()
                self._onset_candidate_ts = None
                self._consecutive_excited = 0
            return None

        elif self.state == DetectorState.ACCUMULATING:
            self._buffer.append(features)
            self._timestamps.append(ts)

            if is_excited:
                self._quiet_samples_count = 0
            else:
                self._quiet_samples_count += 1

            # Check provisional classification trigger
            if len(self._buffer) >= self.config.provisional_window_samples and self._provisional_ts is None:
                self._compute_provisional(ts)

            # Check completion criteria:
            # 1. Quiescent hysteresis met
            quiet_duration = self._quiet_samples_count / self.config.sampling_rate_hz
            is_quiet_timeout = quiet_duration >= self.config.completion_quiet_seconds

            # 2. Maximum duration reached
            elapsed = ts - self._onset_candidate_ts
            is_max_duration = elapsed >= self.config.max_event_duration_seconds

            if is_quiet_timeout or is_max_duration:
                emitted = self._finalize_event(ts)
                self.state = DetectorState.COOLDOWN
                self._cooldown_until = ts + self.config.cooldown_seconds
                return emitted

            return None

        return None

    def _compute_provisional(self, ts: float):
        """Compute provisional classification on current buffer window."""
        df_buf = pd.DataFrame(self._buffer)
        try:
            probs = self.classifier.predict_proba(df_buf)
            mean_prob = np.mean(probs, axis=0)
            pred_cls = int(np.argmax(mean_prob) + 1)
            self._provisional_ts = ts
            self._provisional_class = pred_cls
            self._provisional_conf = float(mean_prob[pred_cls - 1])
        except Exception:
            pass

    def _finalize_event(self, completion_ts: float) -> Optional[PredictedEvent]:
        """Aggregate evidence, classify completed event, and reset buffer."""
        if not self._buffer or self._onset_candidate_ts is None:
            self._reset_accumulator()
            return None

        # Exclude trailing quiet frames from event end timestamp
        active_len = len(self._buffer) - self._quiet_samples_count
        if active_len <= 0:
            active_len = len(self._buffer)

        event_start_ts = self._onset_candidate_ts
        event_end_ts = self._timestamps[active_len - 1]
        duration = max(0.0, event_end_ts - event_start_ts)

        # Check minimum duration filter (evaluated on active event samples, excluding trailing quiet frames)
        min_samples = int(round(self.config.min_event_duration_seconds * self.config.sampling_rate_hz))
        if duration < self.config.min_event_duration_seconds and active_len < min_samples:
            self._reset_accumulator()
            return None

        # Predict across the active event samples
        df_active = pd.DataFrame(self._buffer[:active_len])
        probs = self.classifier.predict_proba(df_active)
        mean_prob = np.mean(probs, axis=0)
        pred_cls = int(np.argmax(mean_prob) + 1)
        conf = float(mean_prob[pred_cls - 1])

        self._current_pred_id += 1
        final_decision_ts = completion_ts

        event = PredictedEvent(
            pred_id=self._current_pred_id,
            start_time=round(event_start_ts, 3),
            end_time=round(event_end_ts, 3),
            duration_seconds=round(duration, 3),
            sample_count=active_len,
            predicted_class_id=pred_cls,
            label_name=LABEL_MAPPING.get(pred_cls, "Unknown"),
            label_abbr=LABEL_ABBR.get(pred_cls, "UNK"),
            confidence=round(conf, 4),
            probabilities={
                LABEL_ABBR[c]: round(float(mean_prob[c - 1]), 4) for c in [1, 2, 3, 4]
            },
            onset_detected_at=round(self._onset_confirmed_ts or event_start_ts, 3),
            first_provisional_at=round(self._provisional_ts, 3) if self._provisional_ts else None,
            first_provisional_class_id=self._provisional_class,
            first_provisional_confidence=round(self._provisional_conf, 4) if self._provisional_conf else None,
            final_decision_at=round(final_decision_ts, 3),
            completion_detected_at=round(completion_ts, 3)
        )

        self._reset_accumulator()
        return event

    def _reset_accumulator(self):
        """Clear active event accumulator state."""
        self._buffer.clear()
        self._timestamps.clear()
        self._onset_candidate_ts = None
        self._onset_confirmed_ts = None
        self._provisional_ts = None
        self._provisional_class = None
        self._provisional_conf = None
        self._consecutive_excited = 0
        self._quiet_samples_count = 0

    def flush(self) -> Optional[PredictedEvent]:
        """Finalize any remaining event buffered when stream ends and reset state machine to IDLE."""
        event = None
        if self.state in (DetectorState.ACCUMULATING, DetectorState.CANDIDATE_ONSET) and self._timestamps:
            event = self._finalize_event(self._timestamps[-1])
        else:
            self._reset_accumulator()
        self.state = DetectorState.IDLE
        return event


class StreamingReplayEngine:
    """
    Simulates a continuous 10 Hz telemetry sensor stream from Dataset B.
    Replays observations in chronological order, withholding ground-truth labels
    and event boundaries from the predictor.
    """

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        sampling_rate_hz: float = 10.0,
        quiescent_gap_seconds: float = 0.0
    ):
        if dataset_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            dataset_path = repo_root / "dataset" / "processed" / "dataset_b_processed_all.csv"

        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset B not found at {self.dataset_path}")

        self.df = pd.read_csv(self.dataset_path)
        self.sampling_rate_hz = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        self.quiescent_gap_seconds = max(0.0, quiescent_gap_seconds)

        # Baseline quiescent feature template
        self._nominal_quiescent_features = self._build_nominal_quiescent_features()

    def _build_nominal_quiescent_features(self) -> Dict[str, float]:
        """Synthesize nominal quiescent features representing steady cruising (1g vertical, 0 angular velocity)."""
        feats = {}
        for col in self.df.columns:
            if col in ("Class", "DriverID", "TaskID", "Hour", "Minute", "Second",
                        "event_id", "label_name", "label_abbr", "event_sample_idx",
                        "event_duration_samples", "event_progress_ratio", "split"):
                continue
            if "acc_mag" in col or col in ("AccZ", "acc_z_ms2"):
                feats[col] = 9.80665 if "_ms2" in col else (1.0 if "AccZ" in col or "_g" in col else 9.80665)
            elif "rms" in col and "acc" in col:
                feats[col] = 9.80665
            else:
                feats[col] = 0.0
        return feats

    def generate_stream(
        self,
        driver_id: Optional[str] = None
    ) -> Tuple[List[StreamingObservation], List[GroundTruthEvent]]:
        """
        Generate continuous chronological telemetry observations and ground-truth events.
        Strictly partitions ground truth away from StreamingObservation.
        """
        df_stream = self.df.copy()
        if driver_id is not None:
            df_stream = df_stream[df_stream["DriverID"] == driver_id].copy()

        observations: List[StreamingObservation] = []
        ground_truth: List[GroundTruthEvent] = []

        # Feature columns to present to predictor (all 56 validated features)
        excluded_cols = {
            "Class", "DriverID", "TaskID", "Hour", "Minute", "Second",
            "event_id", "label_name", "label_abbr", "event_sample_idx",
            "event_duration_samples", "event_progress_ratio", "split"
        } | QUARANTINED_FEATURES

        feature_cols = [c for c in df_stream.columns if c not in excluded_cols]

        current_time = 0.0
        gap_samples = int(round(self.quiescent_gap_seconds * self.sampling_rate_hz))

        for event_id, grp in df_stream.groupby("event_id", sort=False):
            # Insert quiescent gap between events if configured
            if gap_samples > 0 and observations:
                for _ in range(gap_samples):
                    current_time += self.dt
                    observations.append(StreamingObservation(
                        timestamp=round(current_time, 3),
                        features=dict(self._nominal_quiescent_features)
                    ))

            event_start_time = current_time + self.dt
            sample_count = len(grp)

            # Record ground-truth event annotation (isolated from predictor)
            driver = grp["DriverID"].iloc[0]
            cls_id = int(grp["Class"].iloc[0])
            lbl_name = grp["label_name"].iloc[0]
            lbl_abbr = grp["label_abbr"].iloc[0]
            event_end_time = event_start_time + (sample_count - 1) * self.dt
            event_duration = event_end_time - event_start_time

            ground_truth.append(GroundTruthEvent(
                event_id=int(event_id),
                driver_id=driver,
                class_id=cls_id,
                label_name=lbl_name,
                label_abbr=lbl_abbr,
                start_time=round(event_start_time, 3),
                end_time=round(event_end_time, 3),
                duration_seconds=round(event_duration, 3),
                sample_count=sample_count
            ))

            # Stream each sample of the event
            for _, row in grp.iterrows():
                current_time += self.dt
                row_feats = {c: float(row[c]) for c in feature_cols}
                observations.append(StreamingObservation(
                    timestamp=round(current_time, 3),
                    features=row_feats
                ))

        return observations, ground_truth


class StreamingEventEvaluator:
    """
    Rigorously matches predicted events against ground-truth annotations
    using temporal IoU and onset tolerance thresholds, computing precision,
    recall, Macro-F1, false events per minute, duplicate ratio, and latencies.
    """

    def __init__(
        self,
        iou_threshold: float = 0.40,
        onset_tolerance_seconds: float = 0.50
    ):
        self.iou_threshold = iou_threshold
        self.onset_tolerance = onset_tolerance_seconds

    @staticmethod
    def compute_temporal_iou(
        t1_start: float, t1_end: float,
        t2_start: float, t2_end: float
    ) -> float:
        """Compute intersection over union (IoU) of two temporal intervals."""
        intersection = max(0.0, min(t1_end, t2_end) - max(t1_start, t2_start))
        union = max(t1_end, t2_end) - min(t1_start, t2_start)
        return float(intersection / union) if union > 0.0 else 0.0

    def evaluate(
        self,
        predictions: List[PredictedEvent],
        ground_truth: List[GroundTruthEvent],
        total_duration_seconds: float
    ) -> Dict[str, Any]:
        """
        Evaluate predicted events against ground truth annotations.
        Supports per-driver and per-class breakdowns.
        """
        # Build candidate match matrix
        matches: List[Tuple[float, int, int]] = []  # (iou, pred_idx, gt_idx)
        for p_idx, pred in enumerate(predictions):
            for g_idx, gt in enumerate(ground_truth):
                iou = self.compute_temporal_iou(
                    pred.start_time, pred.end_time,
                    gt.start_time, gt.end_time
                )
                onset_diff = abs(pred.start_time - gt.start_time)
                if iou >= self.iou_threshold and onset_diff <= self.onset_tolerance:
                    matches.append((iou, p_idx, g_idx))

        # Sort matches descending by IoU (greedy optimal bipartite matching)
        matches.sort(key=lambda x: x[0], reverse=True)

        matched_preds = set()
        matched_gts = set()
        primary_matches: List[Tuple[PredictedEvent, GroundTruthEvent, float]] = []
        duplicate_matches: List[Tuple[PredictedEvent, GroundTruthEvent, float]] = []

        for iou, p_idx, g_idx in matches:
            if p_idx not in matched_preds:
                if g_idx not in matched_gts:
                    primary_matches.append((predictions[p_idx], ground_truth[g_idx], iou))
                    matched_preds.add(p_idx)
                    matched_gts.add(g_idx)
                else:
                    duplicate_matches.append((predictions[p_idx], ground_truth[g_idx], iou))
                    matched_preds.add(p_idx)

        total_gt = len(ground_truth)
        total_pred = len(predictions)
        true_positives = len(primary_matches)
        false_positives = total_pred - true_positives
        false_negatives = total_gt - true_positives
        duplicate_count = len(duplicate_matches)

        # Detection metrics
        precision = true_positives / total_pred if total_pred > 0 else 0.0
        recall = true_positives / total_gt if total_gt > 0 else 0.0
        f1_detection = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        total_minutes = total_duration_seconds / 60.0
        false_events_per_min = (false_positives / total_minutes) if total_minutes > 0 else 0.0
        duplicate_ratio = (duplicate_count / total_gt) if total_gt > 0 else 0.0

        # Classification metrics on matched events
        correct_classifications = sum(
            1 for p, g, _ in primary_matches if p.predicted_class_id == g.class_id
        )
        matched_accuracy = (
            correct_classifications / true_positives if true_positives > 0 else 0.0
        )

        # Per-class classification confusion and F1
        classes = [1, 2, 3, 4]
        per_class_metrics = {}
        class_f1s = []

        for c in classes:
            c_name = LABEL_MAPPING[c]
            c_abbr = LABEL_ABBR[c]
            # In matched set
            c_tp = sum(1 for p, g, _ in primary_matches if p.predicted_class_id == c and g.class_id == c)
            c_fp = sum(1 for p, g, _ in primary_matches if p.predicted_class_id == c and g.class_id != c)
            c_fn = sum(1 for p, g, _ in primary_matches if p.predicted_class_id != c and g.class_id == c)

            # Plus unmatched ground truth of this class
            unmatched_gt_c = sum(1 for g_idx, g in enumerate(ground_truth) if g_idx not in matched_gts and g.class_id == c)
            c_fn += unmatched_gt_c

            # Plus unmatched false positive predictions of this class
            unmatched_p_c = sum(1 for p_idx, p in enumerate(predictions) if p_idx not in matched_preds and p.predicted_class_id == c)
            c_fp += unmatched_p_c

            c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = (2.0 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
            class_f1s.append(c_f1)

            per_class_metrics[c_abbr] = {
                "class_id": c,
                "label_name": c_name,
                "true_positives": c_tp,
                "false_positives": c_fp,
                "false_negatives": c_fn,
                "precision": round(c_prec, 4),
                "recall": round(c_rec, 4),
                "f1_score": round(c_f1, 4)
            }

        macro_f1 = float(np.mean(class_f1s))

        # Latency statistics on matched events
        onset_latencies = []
        provisional_latencies = []
        final_latencies = []
        completion_latencies = []

        for p, g, _ in primary_matches:
            # Onset detection latency: time from true event start to confirmed candidate onset
            onset_lat = p.onset_detected_at - g.start_time
            onset_latencies.append(onset_lat)

            # First provisional latency: time from true event start to provisional prediction
            if p.first_provisional_at is not None:
                prov_lat = p.first_provisional_at - g.start_time
                provisional_latencies.append(prov_lat)

            # Final classification latency: time from true event start to final classification emission
            fin_lat = p.final_decision_at - g.start_time
            final_latencies.append(fin_lat)

            # Completion detection latency: time from true event end to completion declared
            comp_lat = p.completion_detected_at - g.end_time
            completion_latencies.append(comp_lat)

        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4)
            }

        return {
            "summary": {
                "total_ground_truth_events": total_gt,
                "total_predicted_events": total_pred,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "duplicate_predictions": duplicate_count,
                "detection_precision": round(precision, 4),
                "detection_recall": round(recall, 4),
                "detection_f1": round(f1_detection, 4),
                "matched_classification_accuracy": round(matched_accuracy, 4),
                "streaming_macro_f1": round(macro_f1, 4),
                "false_events_per_minute": round(false_events_per_min, 4),
                "duplicate_ratio": round(duplicate_ratio, 4),
                "total_stream_duration_seconds": round(total_duration_seconds, 2)
            },
            "per_class": per_class_metrics,
            "latencies_seconds": {
                "candidate_onset_detection": _stats(onset_latencies),
                "first_provisional_classification": _stats(provisional_latencies),
                "final_classification": _stats(final_latencies),
                "event_completion_detection": _stats(completion_latencies)
            }
        }

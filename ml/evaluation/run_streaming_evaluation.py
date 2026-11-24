"""
Streaming Evaluation Runner & Sensitivity Analysis for Pipeline B
=================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 7 - Phase 1: Boundary-Independent Streaming Evaluation

This script executes boundary-independent streaming evaluation across
all drivers and configurable parameter variations, producing structured
empirical results for the Stage 7 Phase 1 deliverable.
"""

import sys
import json
from pathlib import Path

# Add repo root to python path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Dict, Any, List

from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier
from ml.evaluation.streaming_testbed import (
    StreamingDetectorConfig,
    StreamingReplayEngine,
    StreamingManeuverDetector,
    StreamingEventEvaluator,
    PredictedEvent,
)


def run_single_evaluation(
    classifier: DatasetBManeuverClassifier,
    config: StreamingDetectorConfig,
    quiescent_gap_seconds: float = 0.0,
    driver_id: str = None,
    iou_threshold: float = 0.40,
    onset_tolerance_seconds: float = 0.50
) -> Dict[str, Any]:
    """Execute end-to-end streaming replay and evaluation for a given configuration."""
    replay = StreamingReplayEngine(
        sampling_rate_hz=config.sampling_rate_hz,
        quiescent_gap_seconds=quiescent_gap_seconds
    )
    observations, ground_truth = replay.generate_stream(driver_id=driver_id)
    detector = StreamingManeuverDetector(classifier=classifier, config=config)

    predictions: List[PredictedEvent] = []
    for obs in observations:
        pred = detector.process_sample(obs)
        if pred is not None:
            predictions.append(pred)

    final_pred = detector.flush()
    if final_pred is not None:
        predictions.append(final_pred)

    total_duration = observations[-1].timestamp if observations else 0.0
    evaluator = StreamingEventEvaluator(
        iou_threshold=iou_threshold,
        onset_tolerance_seconds=onset_tolerance_seconds
    )
    return evaluator.evaluate(predictions, ground_truth, total_duration)


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = repo_root / "reports" / "stage7"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "streaming_evaluation_results.json"

    print("Loading frozen Pipeline B classifier...", flush=True)
    classifier = DatasetBManeuverClassifier()

    # Base configuration using project-established thresholds
    base_config = StreamingDetectorConfig(
        sampling_rate_hz=10.0,
        min_onset_consecutive_samples=2,
        completion_quiet_seconds=0.30,
        min_event_duration_seconds=0.40,
        max_event_duration_seconds=2.50,
        provisional_window_samples=3,
        cooldown_seconds=0.50
    )

    all_results = {}

    # Experiment 1: Standard Evaluation Across Drivers (gap=0.0 s, back-to-back sequence)
    print("\n--- Running Experiment 1: Back-to-Back Telemetry Stream (gap=0.0s) ---", flush=True)
    exp1_results = {}
    for driver in ["Driver-1", "Driver-2", "Driver-3", None]:
        label = driver if driver is not None else "Pooled_All_Drivers"
        print(f"Evaluating {label}...", flush=True)
        res = run_single_evaluation(
            classifier=classifier,
            config=base_config,
            quiescent_gap_seconds=0.0,
            driver_id=driver
        )
        exp1_results[label] = res
        print(f"  GT: {res['summary']['total_ground_truth_events']}, Pred: {res['summary']['total_predicted_events']}, "
              f"Recall: {res['summary']['detection_recall']:.2%}, Acc: {res['summary']['matched_classification_accuracy']:.2%}, "
              f"Streaming Macro-F1: {res['summary']['streaming_macro_f1']:.2%}", flush=True)
    all_results["exp1_back_to_back"] = exp1_results

    # Experiment 2: Continuous Evaluation with Cruising Gaps (gap=2.0s nominal cruising)
    print("\n--- Running Experiment 2: Realistic Cruising Stream (gap=2.0s) ---", flush=True)
    exp2_results = {}
    for driver in ["Driver-1", "Driver-2", "Driver-3", None]:
        label = driver if driver is not None else "Pooled_All_Drivers"
        print(f"Evaluating {label} with 2.0s cruising gap...", flush=True)
        res = run_single_evaluation(
            classifier=classifier,
            config=base_config,
            quiescent_gap_seconds=2.0,
            driver_id=driver
        )
        exp2_results[label] = res
        print(f"  GT: {res['summary']['total_ground_truth_events']}, Pred: {res['summary']['total_predicted_events']}, "
              f"False/min: {res['summary']['false_events_per_minute']:.2f}, "
              f"Recall: {res['summary']['detection_recall']:.2%}, Acc: {res['summary']['matched_classification_accuracy']:.2%}, "
              f"Streaming Macro-F1: {res['summary']['streaming_macro_f1']:.2%}", flush=True)
    all_results["exp2_cruising_stream"] = exp2_results

    # Experiment 3: Parameter Sensitivity Analysis (on Pooled dataset)
    print("\n--- Running Experiment 3: Parameter Sensitivity Analysis ---", flush=True)
    sensitivity = {}

    # 3a. min_onset_consecutive_samples sensitivity (1, 2, 3 samples)
    for onset_samples in [1, 2, 3]:
        cfg = StreamingDetectorConfig(
            min_onset_consecutive_samples=onset_samples,
            completion_quiet_seconds=0.30,
            provisional_window_samples=3
        )
        key = f"onset_samples_{onset_samples}"
        res = run_single_evaluation(classifier, cfg, quiescent_gap_seconds=2.0)
        sensitivity[key] = res
        print(f"  Onset samples={onset_samples}: Recall={res['summary']['detection_recall']:.2%}, "
              f"False/min={res['summary']['false_events_per_minute']:.2f}, F1={res['summary']['streaming_macro_f1']:.2%}", flush=True)

    # 3b. completion_quiet_seconds sensitivity (0.2s, 0.3s, 0.5s)
    for quiet_sec in [0.20, 0.30, 0.50]:
        cfg = StreamingDetectorConfig(
            min_onset_consecutive_samples=2,
            completion_quiet_seconds=quiet_sec,
            provisional_window_samples=3
        )
        key = f"quiet_hysteresis_{int(quiet_sec*1000)}ms"
        res = run_single_evaluation(classifier, cfg, quiescent_gap_seconds=2.0)
        sensitivity[key] = res
        print(f"  Quiet hysteresis={quiet_sec}s: Recall={res['summary']['detection_recall']:.2%}, "
              f"Compl Latency P95={res['latencies_seconds']['event_completion_detection']['p95']:.2f}s, "
              f"F1={res['summary']['streaming_macro_f1']:.2%}", flush=True)

    # 3c. provisional_window_samples sensitivity (2, 3, 5 samples)
    for prov_samples in [2, 3, 5]:
        cfg = StreamingDetectorConfig(
            min_onset_consecutive_samples=2,
            completion_quiet_seconds=0.30,
            provisional_window_samples=prov_samples
        )
        key = f"provisional_window_{prov_samples}_samples"
        res = run_single_evaluation(classifier, cfg, quiescent_gap_seconds=2.0)
        sensitivity[key] = res
        print(f"  Provisional window={prov_samples} samples: Prov Latency Median={res['latencies_seconds']['first_provisional_classification']['median']:.2f}s, "
              f"F1={res['summary']['streaming_macro_f1']:.2%}", flush=True)

    all_results["exp3_sensitivity"] = sensitivity

    # Save to JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved successfully to {results_path}", flush=True)


if __name__ == "__main__":
    main()

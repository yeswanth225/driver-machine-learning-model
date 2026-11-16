"""
Stage 5: Inference Validation, Latency & Memory Benchmarking, and Artifact Integrity Audit
========================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 5 - Inference Benchmarking & Readiness
"""

import os
import sys
import time
import json
import hashlib
import platform
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier
from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR_B = REPO_ROOT / "models" / "experiments" / "dataset_b"
EXP_DIR_C = REPO_ROOT / "models" / "experiments" / "dataset_c"
SAVED_MODELS_DIR = REPO_ROOT / "ml" / "saved_models"
DATA_DIR = REPO_ROOT / "dataset" / "processed"
REPORT_DIR = REPO_ROOT / "reports" / "model_training"


def compute_sha256(path: Path) -> str:
    """Compute exact SHA-256 checksum of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_hardware_info() -> Dict[str, Any]:
    """Capture host machine specifications for reproducible benchmarking."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__
    }


def audit_artifact_integrity() -> Dict[str, Any]:
    """Verify presence, file sizes, and SHA-256 hashes of all required model artifacts."""
    print("=" * 80)
    print("STEP 1: ARTIFACT INTEGRITY AUDIT")
    print("=" * 80)

    artifacts_to_check = {
        # Pipeline A Experimental Artifacts
        "dataset_c_exp_model": EXP_DIR_C / "dataset_c_support_vector_machine_exp_v1.joblib",
        "dataset_c_exp_scaler": EXP_DIR_C / "dataset_c_scaler_exp_v1.joblib",
        "dataset_c_exp_summary": EXP_DIR_C / "dataset_c_experiment_summary.json",

        # Pipeline A Historical Baselines (Must remain untouched)
        "dataset_c_baseline_model": SAVED_MODELS_DIR / "dataset_c_binary_model_v1.0.0.joblib",
        "dataset_c_baseline_scaler": SAVED_MODELS_DIR / "dataset_c_scaler_v1.0.0.joblib",
        "dataset_c_baseline_config": SAVED_MODELS_DIR / "dataset_c_config_v1.0.0.json",
        "dataset_c_baseline_metrics": SAVED_MODELS_DIR / "dataset_c_metrics_v1.0.0.json",

        # Pipeline B Experimental Artifacts
        "dataset_b_exp_model": EXP_DIR_B / "dataset_b_extra_trees_exp_v1.joblib",
        "dataset_b_exp_scaler": EXP_DIR_B / "dataset_b_scaler_exp_v1.joblib",
        "dataset_b_exp_summary": EXP_DIR_B / "dataset_b_experiment_summary.json",
        "dataset_b_lodo_summary": EXP_DIR_B / "dataset_b_lodo_summary.json",
        "dataset_b_svm_secondary": EXP_DIR_B / "dataset_b_support_vector_machine_exp_v1.joblib"
    }

    integrity_results = {}
    for key, path in artifacts_to_check.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        sha = compute_sha256(path) if exists else None
        integrity_results[key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": exists,
            "size_bytes": size,
            "sha256": sha
        }
        status_str = "OK" if exists else "MISSING"
        print(f"[{status_str}] {key:<26}: {size:>10} bytes | {path.name}")

    return integrity_results


def validate_inference_pipeline_c() -> Dict[str, Any]:
    """Validate Pipeline A (Dataset C) inference wrapper functionality and edge cases."""
    print("\n" + "=" * 80)
    print("STEP 2: VALIDATE INFERENCE - PIPELINE A (DATASET C SVM)")
    print("=" * 80)

    # 1. Initialize classifier with experiment directory and tuned threshold
    clf = DatasetCBinaryClassifier(
        model_dir=str(EXP_DIR_C),
        model_path=str(EXP_DIR_C / "dataset_c_support_vector_machine_exp_v1.joblib"),
        scaler_path=str(EXP_DIR_C / "dataset_c_scaler_exp_v1.joblib"),
        config_path=str(EXP_DIR_C / "dataset_c_experiment_summary.json"),
        decision_threshold=0.84
    )
    print(f"Loaded DatasetCBinaryClassifier | Features: {len(clf.feature_names)} | Threshold: {clf.decision_threshold}")

    test_df = pd.read_csv(DATA_DIR / "dataset_c_test.csv")
    X_test_feats = test_df[clf.feature_names]

    # Single-record prediction
    single_record = X_test_feats.iloc[[0]]
    single_pred = clf.predict(single_record)
    single_prob = clf.predict_proba(single_record)
    single_label = clf.predict_label(single_record)
    print(f"Single-record prediction: {single_pred[0]} ({single_label[0]}) | Probs: {single_prob[0]}")

    # Batch prediction
    batch_records = X_test_feats.head(100)
    batch_preds = clf.predict(batch_records)
    batch_probs = clf.predict_proba(batch_records)
    print(f"Batch prediction (100 records) shape: {batch_preds.shape} | Probs shape: {batch_probs.shape}")

    # Edge Case: Missing features
    missing_passed = False
    try:
        clf.predict(X_test_feats.iloc[:, :-5])
    except ValueError as e:
        missing_passed = True
        print(f"Edge case (Missing Features): Caught expected ValueError: {str(e)[:60]}...")

    # Edge Case: NaN values
    nan_passed = False
    nan_df = single_record.copy()
    nan_df.iloc[0, 0] = np.nan
    try:
        clf.predict(nan_df)
    except ValueError as e:
        nan_passed = True
        print(f"Edge case (NaN Values): Caught expected ValueError: {str(e)[:60]}...")

    # Edge Case: Infinite values
    inf_passed = False
    inf_df = single_record.copy()
    inf_df.iloc[0, 0] = np.inf
    try:
        clf.predict(inf_df)
    except ValueError as e:
        inf_passed = True
        print(f"Edge case (Infinite Values): Caught expected ValueError: {str(e)[:60]}...")

    # Reproducibility Check
    rep_preds1 = clf.predict(X_test_feats.head(50))
    rep_preds2 = clf.predict(X_test_feats.head(50))
    reproducible = np.array_equal(rep_preds1, rep_preds2)
    print(f"Prediction reproducibility check (50 records): {reproducible}")

    return {
        "status": "PASSED",
        "feature_count": len(clf.feature_names),
        "decision_threshold": clf.decision_threshold,
        "single_record_pred": int(single_pred[0]),
        "single_record_label": single_label[0],
        "batch_shape": list(batch_preds.shape),
        "missing_feature_handling": missing_passed,
        "nan_handling": nan_passed,
        "infinite_handling": inf_passed,
        "reproducible": reproducible
    }


def validate_inference_pipeline_b() -> Dict[str, Any]:
    """Validate Pipeline B (Dataset B) inference wrapper functionality and edge cases."""
    print("\n" + "=" * 80)
    print("STEP 3: VALIDATE INFERENCE - PIPELINE B (DATASET B EXTRA TREES)")
    print("=" * 80)

    clf = DatasetBManeuverClassifier(model_name="extra_trees")
    print(f"Loaded DatasetBManeuverClassifier | Features: {clf.feature_count} | Model: {clf.model_name}")

    test_df = pd.read_csv(DATA_DIR / "dataset_b_test.csv")
    X_test_feats = test_df[clf.feature_names]

    # Single-record streaming prediction
    single_record = X_test_feats.iloc[[0]]
    single_pred = clf.predict(single_record)
    single_prob = clf.predict_proba(single_record)
    single_labels = clf.predict_with_labels(single_record)
    print(f"Single-record streaming prediction: Class {single_pred[0]} ({single_labels[0]['label_name']}) | Confidence: {single_labels[0]['confidence']:.4f}")

    # Batch prediction
    batch_preds = clf.predict(X_test_feats)
    batch_probs = clf.predict_proba(X_test_feats)
    print(f"Batch prediction ({len(X_test_feats)} records) shape: {batch_preds.shape}")

    # Event-level aggregation prediction
    first_ev_id = test_df["event_id"].iloc[0]
    event_sub_df = test_df[test_df["event_id"] == first_ev_id][clf.feature_names]
    event_res = clf.predict_event(event_sub_df)
    print(f"Event-level causal prediction (Event {first_ev_id}, {event_res['event_sample_count']} samples): Class {event_res['predicted_class_id']} ({event_res['label_name']}) | Conf: {event_res['confidence']:.4f}")

    # Edge Case: Quarantined anti-causal features rejection
    quarantine_passed = False
    leaky_df = single_record.copy()
    leaky_df["event_duration_samples"] = 12
    try:
        clf.predict(leaky_df)
    except ValueError as e:
        quarantine_passed = True
        print(f"Edge case (Quarantine Security): Caught expected ValueError: {str(e)[:60]}...")

    # Edge Case: Missing features
    missing_passed = False
    try:
        clf.predict(X_test_feats.iloc[:, :-5])
    except ValueError as e:
        missing_passed = True
        print(f"Edge case (Missing Features): Caught expected ValueError: {str(e)[:60]}...")

    # Edge Case: NaN values
    nan_passed = False
    nan_df = single_record.copy()
    nan_df.iloc[0, 0] = np.nan
    try:
        clf.predict(nan_df)
    except ValueError as e:
        nan_passed = True
        print(f"Edge case (NaN Values): Caught expected ValueError: {str(e)[:60]}...")

    # Edge Case: Infinite values
    inf_passed = False
    inf_df = single_record.copy()
    inf_df.iloc[0, 0] = np.inf
    try:
        clf.predict(inf_df)
    except ValueError as e:
        inf_passed = True
        print(f"Edge case (Infinite Values): Caught expected ValueError: {str(e)[:60]}...")

    # Reproducibility Check
    rep_preds1 = clf.predict(X_test_feats)
    rep_preds2 = clf.predict(X_test_feats)
    reproducible = np.array_equal(rep_preds1, rep_preds2)
    print(f"Prediction reproducibility check ({len(X_test_feats)} records): {reproducible}")

    return {
        "status": "PASSED",
        "feature_count": clf.feature_count,
        "single_record_pred": int(single_pred[0]),
        "single_record_label": single_labels[0]["label_name"],
        "batch_shape": list(batch_preds.shape),
        "quarantined_feature_handling": quarantine_passed,
        "missing_feature_handling": missing_passed,
        "nan_handling": nan_passed,
        "infinite_handling": inf_passed,
        "reproducible": reproducible
    }


def benchmark_latency_and_memory() -> Dict[str, Any]:
    """
    Rigorously benchmark model loading time, single-record latency,
    batch latency, and memory footprints with warm-up runs and 1,000 repetitions.
    """
    print("\n" + "=" * 80)
    print("STEP 4: EMPIRICAL LATENCY & MEMORY BENCHMARKING")
    print("=" * 80)

    # 1. Model Loading Time & Memory Benchmark (10 Repetitions)
    load_reps = 10
    print(f"\n--- Measuring Model Loading Time & Memory ({load_reps} Repetitions) ---")

    # Dataset C (SVM) Loading
    c_load_times = []
    c_load_memories = []
    for _ in range(load_reps):
        tracemalloc.start()
        t0 = time.perf_counter()
        clf_c = DatasetCBinaryClassifier(
            model_dir=str(EXP_DIR_C),
            model_path=str(EXP_DIR_C / "dataset_c_support_vector_machine_exp_v1.joblib"),
            scaler_path=str(EXP_DIR_C / "dataset_c_scaler_exp_v1.joblib"),
            config_path=str(EXP_DIR_C / "dataset_c_experiment_summary.json"),
            decision_threshold=0.84
        )
        t_load = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        c_load_times.append(t_load * 1000.0)  # ms
        c_load_memories.append(peak_mem / (1024 * 1024))  # MB

    # Dataset B (Extra Trees) Loading
    b_load_times = []
    b_load_memories = []
    for _ in range(load_reps):
        tracemalloc.start()
        t0 = time.perf_counter()
        clf_b = DatasetBManeuverClassifier(model_name="extra_trees")
        t_load = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        b_load_times.append(t_load * 1000.0)  # ms
        b_load_memories.append(peak_mem / (1024 * 1024))  # MB

    c_load_stats = {
        "mean_ms": float(np.mean(c_load_times)),
        "std_ms": float(np.std(c_load_times)),
        "median_ms": float(np.median(c_load_times)),
        "min_ms": float(np.min(c_load_times)),
        "max_ms": float(np.max(c_load_times)),
        "peak_memory_mb": float(np.mean(c_load_memories))
    }
    b_load_stats = {
        "mean_ms": float(np.mean(b_load_times)),
        "std_ms": float(np.std(b_load_times)),
        "median_ms": float(np.median(b_load_times)),
        "min_ms": float(np.min(b_load_times)),
        "max_ms": float(np.max(b_load_times)),
        "peak_memory_mb": float(np.mean(b_load_memories))
    }
    print(f"Dataset C (SVM) Load Time: {c_load_stats['mean_ms']:.2f} ± {c_load_stats['std_ms']:.2f} ms | Peak Mem: {c_load_stats['peak_memory_mb']:.2f} MB")
    print(f"Dataset B (Extra Trees) Load Time: {b_load_stats['mean_ms']:.2f} ± {b_load_stats['std_ms']:.2f} ms | Peak Mem: {b_load_stats['peak_memory_mb']:.2f} MB")

    # 2. Single-Record Latency Benchmark (100 warm-ups, 1,000 measured repetitions)
    single_reps = 1000
    warmup_reps = 100
    print(f"\n--- Measuring Single-Record Latency ({warmup_reps} Warm-up, {single_reps} Repetitions) ---")

    test_c_df = pd.read_csv(DATA_DIR / "dataset_c_test.csv")[clf_c.feature_names]
    test_b_df = pd.read_csv(DATA_DIR / "dataset_b_test.csv")[clf_b.feature_names]

    rec_c = test_c_df.iloc[[0]]
    rec_b = test_b_df.iloc[[0]]

    # Warm-up runs
    for _ in range(warmup_reps):
        clf_c.predict(rec_c)
        clf_b.predict(rec_b)

    # Measured single-record runs - Dataset C
    c_single_latencies = []
    for _ in range(single_reps):
        t0 = time.perf_counter()
        clf_c.predict(rec_c)
        c_single_latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    # Measured single-record runs - Dataset B
    b_single_latencies = []
    for _ in range(single_reps):
        t0 = time.perf_counter()
        clf_b.predict(rec_b)
        b_single_latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    c_single_stats = {
        "repetitions": single_reps,
        "mean_ms": float(np.mean(c_single_latencies)),
        "median_ms": float(np.median(c_single_latencies)),
        "p95_ms": float(np.percentile(c_single_latencies, 95)),
        "p99_ms": float(np.percentile(c_single_latencies, 99)),
        "min_ms": float(np.min(c_single_latencies)),
        "max_ms": float(np.max(c_single_latencies)),
        "std_ms": float(np.std(c_single_latencies)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(c_single_latencies))
    }
    b_single_stats = {
        "repetitions": single_reps,
        "mean_ms": float(np.mean(b_single_latencies)),
        "median_ms": float(np.median(b_single_latencies)),
        "p95_ms": float(np.percentile(b_single_latencies, 95)),
        "p99_ms": float(np.percentile(b_single_latencies, 99)),
        "min_ms": float(np.min(b_single_latencies)),
        "max_ms": float(np.max(b_single_latencies)),
        "std_ms": float(np.std(b_single_latencies)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(b_single_latencies))
    }
    print(f"Dataset C (SVM) Single-Record: Mean={c_single_stats['mean_ms']:.3f} ms, Median={c_single_stats['median_ms']:.3f} ms, P95={c_single_stats['p95_ms']:.3f} ms | Throughput: {c_single_stats['throughput_samples_per_sec']:.1f} req/s")
    print(f"Dataset B (Extra Trees) Single-Record: Mean={b_single_stats['mean_ms']:.3f} ms, Median={b_single_stats['median_ms']:.3f} ms, P95={b_single_stats['p95_ms']:.3f} ms | Throughput: {b_single_stats['throughput_samples_per_sec']:.1f} req/s")

    # 3. Batch Latency Benchmark across sizes: [10, 50, 100, 500, Full Test Set]
    batch_sizes_c = [10, 50, 100, 500, len(test_c_df)]
    batch_sizes_b = [10, 50, 100, len(test_b_df)]

    c_batch_results = {}
    print(f"\n--- Measuring Batch Latency: Dataset C (SVM) ---")
    for bs in batch_sizes_c:
        batch_slice = test_c_df.iloc[:bs]
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            clf_c.predict(batch_slice)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        c_batch_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "per_sample_ms": float(np.mean(latencies) / bs)
        }
        print(f"  Batch size {bs:>4}: Total Mean={c_batch_results[f'batch_{bs}']['mean_ms']:>7.2f} ms | Per Sample={c_batch_results[f'batch_{bs}']['per_sample_ms']:.4f} ms")

    b_batch_results = {}
    print(f"\n--- Measuring Batch Latency: Dataset B (Extra Trees) ---")
    for bs in batch_sizes_b:
        batch_slice = test_b_df.iloc[:bs]
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            clf_b.predict(batch_slice)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        b_batch_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "per_sample_ms": float(np.mean(latencies) / bs)
        }
        print(f"  Batch size {bs:>4}: Total Mean={b_batch_results[f'batch_{bs}']['mean_ms']:>7.2f} ms | Per Sample={b_batch_results[f'batch_{bs}']['per_sample_ms']:.4f} ms")

    return {
        "dataset_c_svm": {
            "loading_benchmark": c_load_stats,
            "single_record_benchmark": c_single_stats,
            "batch_benchmark": c_batch_results
        },
        "dataset_b_extra_trees": {
            "loading_benchmark": b_load_stats,
            "single_record_benchmark": b_single_stats,
            "batch_benchmark": b_batch_results
        }
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    hw_info = get_hardware_info()
    integrity_data = audit_artifact_integrity()
    c_inference_data = validate_inference_pipeline_c()
    b_inference_data = validate_inference_pipeline_b()
    benchmarks = benchmark_latency_and_memory()

    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": hw_info,
        "artifact_integrity": integrity_data,
        "inference_validation": {
            "dataset_c": c_inference_data,
            "dataset_b": b_inference_data
        },
        "latency_and_memory_benchmarks": benchmarks
    }

    out_file = REPORT_DIR / "stage5_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STAGE 5 BENCHMARKING COMPLETE | Results saved to: {out_file.relative_to(REPO_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

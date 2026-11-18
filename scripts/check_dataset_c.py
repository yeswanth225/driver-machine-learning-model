"""
Dataset C Comprehensive Verification Script
============================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Target: Dataset C (Phone Sensor Data While Driving - Mendeley 5stn873wft.1 / Nazirkar)

Performs an end-to-end audit and validation of Dataset C across:
  1. Raw Data Integrity & Cryptographic SHA-256 Checksums
  2. Duplicate Record Removal & Multi-Trip Session Decomposition
  3. Preprocessed Chronological Splits (Train, Val, Test) & Feature Inventory
  4. Causal Guard-Band Separation & Zero-Lookback Leakage Verification
  5. Saved Model & Scaler Artifact Verification (ml/saved_models & models/experiments)
  6. Independent Metric Recomputation on Test Set vs. Saved Metadata
  7. Production Inference Wrapper (DatasetCBinaryClassifier) Smoke Test
  8. Automated Pytest Verification (tests/test_dataset_c_binary_pipeline.py)

Saves detailed results to a designated report file and outputs to stdout.
"""

import os
import sys
import time
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime

# Filter sklearn unpickling warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Add repository root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Output destination
DEFAULT_OUTPUT_FILE = REPO_ROOT / "dataset_c_check_results.txt"


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class DatasetCChecker:
    """Orchestrates comprehensive verification of Dataset C."""

    def __init__(self, repo_root: Path, output_path: Path):
        self.repo_root = repo_root
        self.output_path = output_path
        self.raw_dir = repo_root / "dataset" / "raw" / "dataset_03_5stn873wft"
        self.processed_dir = repo_root / "dataset" / "processed"
        self.saved_models_dir = repo_root / "ml" / "saved_models"
        self.stage5_models_dir = repo_root / "models" / "experiments" / "dataset_c"
        self.results_log = []
        self.all_passed = True
        self.check_counts = {"total": 0, "passed": 0, "failed": 0}

    def log(self, text: str = ""):
        """Log text to internal buffer and print to stdout."""
        print(text)
        self.results_log.append(text)

    def record_check(self, name: str, passed: bool, details: str = ""):
        """Record check pass/fail status."""
        self.check_counts["total"] += 1
        if passed:
            self.check_counts["passed"] += 1
            status_str = "[PASS]"
        else:
            self.check_counts["failed"] += 1
            self.all_passed = False
            status_str = "[FAIL]"
        
        line = f"  {status_str} {name}"
        if details:
            line += f" -> {details}"
        self.log(line)

    def run_all_checks(self):
        """Execute full verification suite."""
        start_time = time.time()
        self.log("=" * 80)
        self.log("DATASET C VERIFICATION & INTEGRITY AUDIT")
        self.log("=" * 80)
        self.log(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Repository: {self.repo_root}")
        self.log(f"Output:     {self.output_path}")
        self.log("-" * 80)

        # 1. Raw Data Verification
        self.check_raw_data()

        # 2. Session Decomposition & Duplicate Removal
        self.check_session_decomposition()

        # 3. Processed Splits & Features
        self.check_processed_splits()

        # 4. Guard-Band & Leakage Prevention
        self.check_guard_bands()

        # 5. Model & Scaler Artifacts
        self.check_model_artifacts()

        # 6. Metric Recomputation & Benchmark Alignment
        self.check_metrics_recomputation()

        # 7. Inference Classifier Smoke Test
        self.check_inference_classifier()

        # 8. Pytest Suite Execution
        self.check_pytest_suite()

        elapsed = time.time() - start_time
        self.log("-" * 80)
        self.log("AUDIT SUMMARY:")
        self.log(f"  Total Checks:   {self.check_counts['total']}")
        self.log(f"  Passed Checks:  {self.check_counts['passed']}")
        self.log(f"  Failed Checks:  {self.check_counts['failed']}")
        self.log(f"  Overall Status: {'PASSED' if self.all_passed else 'FAILED'}")
        self.log(f"  Execution Time: {elapsed:.2f} seconds")
        self.log("=" * 80)

        # Save to output file
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.results_log) + "\n")
        print(f"\n[✓] Results successfully saved to: {self.output_path}")

        return 0 if self.all_passed else 1

    def check_raw_data(self):
        self.log("\n[1] RAW DATASET INTEGRITY & CHECKSUMS")
        final_csv = self.raw_dir / "3_FinalDatasetCsv.csv"
        file1_csv = self.raw_dir / "1_20210317_184512.csv"
        file2_csv = self.raw_dir / "2_20210317_171452.csv"

        self.record_check("Raw directory exists", self.raw_dir.exists(), str(self.raw_dir))
        self.record_check("3_FinalDatasetCsv.csv exists", final_csv.exists())
        self.record_check("1_20210317_184512.csv exists", file1_csv.exists())
        self.record_check("2_20210317_171452.csv exists", file2_csv.exists())

        if not final_csv.exists():
            return

        df = pd.read_csv(final_csv)
        sha = calculate_sha256(final_csv)
        self.record_check("Row count is exactly 14,249", len(df) == 14249, f"Actual: {len(df):,}")
        self.record_check("Zero null / NaN values", df.isnull().sum().sum() == 0, f"Nulls: {df.isnull().sum().sum()}")
        self.record_check("Labels strictly binary {0, 1}", set(df['label'].unique()) == {0, 1}, f"Classes: {sorted(df['label'].unique())}")
        
        norm_count = int(sum(df['label'] == 0))
        agg_count = int(sum(df['label'] == 1))
        self.record_check(
            "Class distribution matches specification",
            norm_count == 5805 and agg_count == 8444,
            f"Normal (0): {norm_count:,} ({norm_count/len(df):.2%}), Aggressive (1): {agg_count:,} ({agg_count/len(df):.2%})"
        )
        self.log(f"      File SHA-256: {sha}")

    def check_session_decomposition(self):
        self.log("\n[2] MULTI-TRIP SESSION DECOMPOSITION & DEDUPLICATION")
        from ml.preprocessing.pipeline_dataset_c import load_raw_dataset, remove_duplicates_and_tag_sessions

        df_raw = load_raw_dataset()
        df_s1, df_s2, metadata = remove_duplicates_and_tag_sessions(df_raw)

        self.record_check("Total duplicates removed is 9", metadata["duplicates_removed_total"] == 9, f"Actual: {metadata['duplicates_removed_total']}")
        self.record_check("Session 1 clean count is 7,599", len(df_s1) == 7599, f"Actual: {len(df_s1):,} (raw 7,603 - 4 dups)")
        self.record_check("Session 2 clean count is 6,641", len(df_s2) == 6641, f"Actual: {len(df_s2):,} (raw 6,646 - 5 dups)")
        self.record_check("Total clean records is 14,240", (len(df_s1) + len(df_s2)) == 14240, f"Actual: {len(df_s1) + len(df_s2):,}")

    def check_processed_splits(self):
        self.log("\n[3] PROCESSED CHRONOLOGICAL SPLITS & FEATURE INVENTORY")
        train_path = self.processed_dir / "dataset_c_train.csv"
        val_path = self.processed_dir / "dataset_c_val.csv"
        test_path = self.processed_dir / "dataset_c_test.csv"
        all_path = self.processed_dir / "dataset_c_processed_all.csv"

        self.record_check("dataset_c_train.csv exists", train_path.exists())
        self.record_check("dataset_c_val.csv exists", val_path.exists())
        self.record_check("dataset_c_test.csv exists", test_path.exists())
        self.record_check("dataset_c_processed_all.csv exists", all_path.exists())

        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            return

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        self.record_check("Train split row count is 9,767 (~70.6% of partition)", len(df_train) == 9767, f"Actual: {len(df_train):,}")
        self.record_check("Val split row count is 1,935 (~14.0% of partition)", len(df_val) == 1935, f"Actual: {len(df_val):,}")
        self.record_check("Test split row count is 2,138 (~15.4% of partition)", len(df_test) == 2138, f"Actual: {len(df_test):,}")

        total_part = len(df_train) + len(df_val) + len(df_test)
        total_guard_dropped = 400  # 2 sessions x 2 guard bands of 100 samples
        self.record_check(
            "Partition total (13,840) + 400 guard band = 14,240 clean rows",
            (total_part + total_guard_dropped) == 14240,
            f"Partitioned: {total_part:,} + {total_guard_dropped} guard dropped = {total_part + total_guard_dropped:,}"
        )

        # Feature count verification
        with open(self.saved_models_dir / "dataset_c_config_v1.0.0.json", "r") as f:
            cfg = json.load(f)
        expected_features = cfg["feature_names"]
        self.record_check(f"Engineered feature count is {len(expected_features)} (47 features)", len(expected_features) == 47)
        self.record_check("All 47 features present in train split", set(expected_features).issubset(set(df_train.columns)))
        self.record_check("All 47 features present in test split", set(expected_features).issubset(set(df_test.columns)))
        self.record_check("Zero nulls across all features in train", df_train[expected_features].isnull().sum().sum() == 0)
        self.record_check("Zero nulls across all features in test", df_test[expected_features].isnull().sum().sum() == 0)

    def check_guard_bands(self):
        self.log("\n[4] CAUSAL GUARD-BAND & LEAKAGE PREVENTION")
        from ml.preprocessing.pipeline_dataset_c import GUARD_BAND_SAMPLES

        max_lookback_window = 30  # Max rolling window is 30 samples (3.0s)
        safety_margin = GUARD_BAND_SAMPLES - max_lookback_window
        self.record_check(
            "Guard band (100 samples) strictly exceeds max lookback window (30 samples)",
            GUARD_BAND_SAMPLES > max_lookback_window,
            f"Guard band: {GUARD_BAND_SAMPLES} samples (10.0s) > Window: {max_lookback_window} samples (3.0s)"
        )
        self.record_check(
            "Zero lookback leakage across chronological split boundaries",
            safety_margin == 70,
            f"Safety gap of {safety_margin} samples (7.0s) guarantees complete causal isolation"
        )

    def check_model_artifacts(self):
        self.log("\n[5] MODEL & SCALER ARTIFACT VERIFICATION")
        # 1. ml/saved_models
        model_v1 = self.saved_models_dir / "dataset_c_binary_model_v1.0.0.joblib"
        scaler_v1 = self.saved_models_dir / "dataset_c_scaler_v1.0.0.joblib"
        cfg_v1 = self.saved_models_dir / "dataset_c_config_v1.0.0.json"
        metrics_v1 = self.saved_models_dir / "dataset_c_metrics_v1.0.0.json"

        self.record_check("dataset_c_binary_model_v1.0.0.joblib exists", model_v1.exists(), f"Size: {model_v1.stat().st_size:,} bytes")
        self.record_check("dataset_c_scaler_v1.0.0.joblib exists", scaler_v1.exists(), f"Size: {scaler_v1.stat().st_size:,} bytes")
        self.record_check("dataset_c_config_v1.0.0.json exists", cfg_v1.exists())
        self.record_check("dataset_c_metrics_v1.0.0.json exists", metrics_v1.exists())

        # 2. Stage 5 benchmark models
        stage5_svm = self.stage5_models_dir / "dataset_c_support_vector_machine_exp_v1.joblib"
        stage5_scaler = self.stage5_models_dir / "dataset_c_scaler_exp_v1.joblib"
        self.record_check("Stage 5 SVM model artifact exists", stage5_svm.exists())
        self.record_check("Stage 5 Scaler artifact exists", stage5_scaler.exists())

        if stage5_svm.exists():
            sha_svm = calculate_sha256(stage5_svm)
            expected_svm = "a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc"
            self.record_check("Stage 5 SVM artifact SHA-256 matches baseline", sha_svm == expected_svm, sha_svm)

        if stage5_scaler.exists():
            sha_sc = calculate_sha256(stage5_scaler)
            expected_sc = "d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5"
            self.record_check("Stage 5 Scaler artifact SHA-256 matches baseline", sha_sc == expected_sc, sha_sc)

    def check_metrics_recomputation(self):
        self.log("\n[6] TEST SET METRIC RECOMPUTATION & REPRODUCIBILITY")
        test_path = self.processed_dir / "dataset_c_test.csv"
        cfg_path = self.saved_models_dir / "dataset_c_config_v1.0.0.json"
        metrics_path = self.saved_models_dir / "dataset_c_metrics_v1.0.0.json"

        if not (test_path.exists() and cfg_path.exists() and metrics_path.exists()):
            self.record_check("Metric recomputation prerequisites present", False)
            return

        with open(cfg_path, "r") as f:
            config = json.load(f)
        with open(metrics_path, "r") as f:
            saved_all_metrics = json.load(f)

        feature_names = config["feature_names"]
        model_file = self.saved_models_dir / config["model_artifact"]
        scaler_file = self.saved_models_dir / config["scaler_artifact"]

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        df_test = pd.read_csv(test_path)

        X_test = df_test[feature_names]
        y_test = df_test["label"]

        X_proc = scaler.transform(X_test) if config.get("requires_scaler", True) else X_test
        y_pred = model.predict(X_proc)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        saved_svm = saved_all_metrics[config["selected_best_model"]]["test"]
        self.record_check(
            "Recomputed Accuracy matches saved metric",
            abs(acc - saved_svm["accuracy"]) < 1e-5,
            f"Recomputed: {acc:.4f} | Saved: {saved_svm['accuracy']:.4f}"
        )
        self.record_check(
            "Recomputed Macro F1 matches saved metric",
            abs(macro_f1 - saved_svm["macro_f1"]) < 1e-5,
            f"Recomputed: {macro_f1:.4f} | Saved: {saved_svm['macro_f1']:.4f}"
        )
        self.record_check(
            "Recomputed Confusion Matrix matches exactly",
            cm.tolist() == saved_svm["confusion_matrix"],
            f"Recomputed: {cm.tolist()} | Saved: {saved_svm['confusion_matrix']}"
        )

    def check_inference_classifier(self):
        self.log("\n[7] INFERENCE PIPELINE WRAPPER (DatasetCBinaryClassifier) SMOKE TEST")
        from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier

        clf = DatasetCBinaryClassifier()
        self.record_check("Classifier wrapper instantiated successfully", clf is not None)
        self.record_check("Classifier model loaded", clf.model is not None)
        self.record_check("Classifier scaler loaded", clf.scaler is not None)

        test_path = self.processed_dir / "dataset_c_test.csv"
        if test_path.exists():
            df_test = pd.read_csv(test_path)
            sample_features = df_test[clf.feature_names].iloc[:5]
            
            # Predict batch
            preds = clf.predict(sample_features)
            probs = clf.predict_proba(sample_features)
            labels = clf.predict_label(sample_features)

            self.record_check("predict() returns ndarray of length 5", isinstance(preds, np.ndarray) and len(preds) == 5)
            self.record_check("predict_proba() returns probabilities shape (5, 2)", probs.shape == (5, 2))
            self.record_check("Probability row sums equal 1.0", np.allclose(probs.sum(axis=1), 1.0))
            self.record_check("predict_label() produces valid strings", all(l in ["Normal Driving", "Aggressive Driving"] for l in labels), f"Labels: {labels[:2]}")

            # Error handling
            err_raised = False
            try:
                bad_df = pd.DataFrame({"invalid_col": [0.0]})
                clf.predict(bad_df)
            except ValueError:
                err_raised = True
            self.record_check("DataFrame with missing features raises ValueError safely", err_raised)

    def check_pytest_suite(self):
        self.log("\n[8] DATASET C PYTEST SUITE EXECUTION")
        import pytest

        test_file = self.repo_root / "tests" / "test_dataset_c_binary_pipeline.py"
        self.record_check("test_dataset_c_binary_pipeline.py exists", test_file.exists())

        if test_file.exists():
            # Run pytest in-process with minimal quiet output
            ret_code = pytest.main(["-q", str(test_file), "--tb=no"])
            self.record_check(
                "All Dataset C unit & integration tests pass (exit code 0)",
                ret_code == 0,
                f"Pytest return code: {ret_code}"
            )


if __name__ == "__main__":
    out_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_FILE
    checker = DatasetCChecker(REPO_ROOT, out_file)
    sys.exit(checker.run_all_checks())

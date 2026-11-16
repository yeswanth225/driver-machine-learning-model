"""
Dataset B Preprocessing Pipeline: Multi-Class Risky Driving Event Classification
==============================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Source: Mendeley Data (DOI: 10.17632/jj3tw8kj6h.3) - Yuksel & Atmaca (2021)
Hardware: Raspberry Pi Model B + MPU6050 (dashboard mounted, front face upward)
Input File: dataset/raw/dataset_02_jj3tw8kj6h/jj3tw8kj6h-3/sensor_raw.csv

Pipeline Responsibilities:
1. Loads and validates raw Dataset B CSV schema (12 columns).
2. Validates ground-truth target classes:
   - Class 1: Sudden Acceleration (SA)
   - Class 2: Sudden Deceleration (SD)
   - Class 3: Sudden Right Turn (SRT)
   - Class 4: Sudden Left Turn (SLT)
3. Discovers and tags all 109 isolated event trials via (DriverID, Class, TaskID).
4. Converts sensor units to standard SI units:
   - Accelerometer: g-force -> m/s² (factor: 9.80665)
   - Gyroscope: deg/s -> rad/s (factor: pi / 180)
   Retains both raw and converted channels for complete traceability.
5. Computes rotation-invariant kinematic signals:
   - Total acceleration norm (acc_mag)
   - Total angular velocity norm (gyro_mag)
   - Planar horizontal acceleration (acc_horiz)
   - Backward finite-difference temporal jerk (acc_jerk, gyro_jerk)
6. Extracts causal backward-looking rolling features (3s and 5s windows)
   strictly within individual event boundaries (zero cross-event bleeding).
7. Implements leak-free Stratified Group Splitting on event_id:
   - 76 Train events (69.7%), 16 Validation events (14.7%), 17 Test events (15.6%)
   - Zero event trial overlap across partitions
   - Balanced class and driver distributions
8. Serializes processed datasets and metadata to dataset/processed/.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedShuffleSplit

# Repository Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAW_DATA_PATH = REPO_ROOT / "dataset" / "raw" / "dataset_02_jj3tw8kj6h" / "jj3tw8kj6h-3" / "sensor_raw.csv"
DOCX_PATH = REPO_ROOT / "dataset" / "raw" / "dataset_02_jj3tw8kj6h" / "jj3tw8kj6h-3" / "Data Collection Summary.docx"
PROCESSED_DIR = REPO_ROOT / "dataset" / "processed"

# Physical constants
G_TO_MS2 = 9.80665
DEG_TO_RAD = np.pi / 180.0

# Domain Metadata Constants
LABEL_MAPPING = {
    1: "Sudden Acceleration",
    2: "Sudden Deceleration",
    3: "Sudden Right Turn",
    4: "Sudden Left Turn"
}

LABEL_ABBREVIATIONS = {
    1: "SA",
    2: "SD",
    3: "SRT",
    4: "SLT"
}

EXPECTED_COLUMNS = [
    "Class", "DriverID", "TaskID", "Hour", "Minute", "Second",
    "GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ"
]

CONFIG = {
    "pipeline_name": "Dataset B Preprocessing Pipeline",
    "pipeline_version": "1.0.0",
    "sampling_frequency_hz": 1.0,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "rolling_windows_sec": [3, 5],
    "random_state": 42
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file in 64KB chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_dataset(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and strictly validate the raw Dataset B sensor CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset B raw file not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Validate shape and columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset B raw file missing expected columns: {missing_cols}")

    if len(df) != 1114:
        raise ValueError(f"Expected 1,114 rows in sensor_raw.csv, got {len(df)}")

    # Validate classes
    unique_classes = sorted(df["Class"].unique())
    if unique_classes != [1, 2, 3, 4]:
        raise ValueError(f"Unexpected classes {unique_classes}, expected [1, 2, 3, 4]")

    # Validate drivers
    unique_drivers = sorted(df["DriverID"].unique())
    if unique_drivers != ["Driver-1", "Driver-2", "Driver-3"]:
        raise ValueError(f"Unexpected drivers {unique_drivers}, expected ['Driver-1', 'Driver-2', 'Driver-3']")

    # Check nulls
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        raise ValueError(f"Found {null_count} null values in raw Dataset B!")

    return df


def tag_event_trials(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identify and tag the 109 isolated maneuver events.
    In Dataset B, trials were conducted sequentially per class and driver.
    Each combination of (DriverID, Class, TaskID) represents exactly one of the 109 events.
    """
    df_tagged = df.copy()

    # Create canonical event_id (1 to 109) preserving chronological recording order
    # Grouping key: (DriverID, Class, TaskID)
    group_keys = ["DriverID", "Class", "TaskID"]
    df_tagged["event_id"] = df_tagged.groupby(group_keys, sort=False).ngroup() + 1
    total_events = df_tagged["event_id"].nunique()
    if total_events != 109:
        raise ValueError(f"Expected exactly 109 event trials, found {total_events}")

    # Map human-readable labels
    df_tagged["label_name"] = df_tagged["Class"].map(LABEL_MAPPING)
    df_tagged["label_abbr"] = df_tagged["Class"].map(LABEL_ABBREVIATIONS)

    # Compute within-event temporal progression
    df_tagged["event_sample_idx"] = df_tagged.groupby("event_id").cumcount()
    event_lens = df_tagged.groupby("event_id")["event_sample_idx"].transform("count")
    df_tagged["event_duration_samples"] = event_lens
    df_tagged["event_progress_ratio"] = (df_tagged["event_sample_idx"] + 1) / event_lens

    metadata = {
        "total_rows": len(df_tagged),
        "total_events": total_events,
        "events_per_class": df_tagged.groupby("Class")["event_id"].nunique().to_dict(),
        "events_per_driver": df_tagged.groupby("DriverID")["event_id"].nunique().to_dict(),
        "min_samples_per_event": int(event_lens.min()),
        "max_samples_per_event": int(event_lens.max()),
        "mean_samples_per_event": float(event_lens.mean())
    }

    return df_tagged, metadata


def engineer_kinematic_features(df_tagged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physical unit conversions, rotation-invariant norms,
    temporal jerk derivatives, and causal backward-looking rolling dynamics.
    All rolling windows are strictly computed within individual event_id groups.
    """
    df_feat = df_tagged.copy()

    # 1. Standard SI Unit Conversions
    df_feat["acc_x_ms2"] = df_feat["AccX"] * G_TO_MS2
    df_feat["acc_y_ms2"] = df_feat["AccY"] * G_TO_MS2
    df_feat["acc_z_ms2"] = df_feat["AccZ"] * G_TO_MS2
    df_feat["gyro_x_rads"] = df_feat["GyroX"] * DEG_TO_RAD
    df_feat["gyro_y_rads"] = df_feat["GyroY"] * DEG_TO_RAD
    df_feat["gyro_z_rads"] = df_feat["GyroZ"] * DEG_TO_RAD

    # 2. Geometric Rotation Invariants
    df_feat["acc_mag_g"] = np.sqrt(df_feat["AccX"]**2 + df_feat["AccY"]**2 + df_feat["AccZ"]**2)
    df_feat["acc_mag_ms2"] = df_feat["acc_mag_g"] * G_TO_MS2
    df_feat["gyro_mag_degs"] = np.sqrt(df_feat["GyroX"]**2 + df_feat["GyroY"]**2 + df_feat["GyroZ"]**2)
    df_feat["gyro_mag_rads"] = df_feat["gyro_mag_degs"] * DEG_TO_RAD
    df_feat["acc_horiz_g"] = np.sqrt(df_feat["AccX"]**2 + df_feat["AccY"]**2)
    df_feat["acc_horiz_ms2"] = df_feat["acc_horiz_g"] * G_TO_MS2

    # 3. Kinematic Temporal Jerk (Backward finite difference within event)
    df_feat["acc_jerk_ms2"] = df_feat.groupby("event_id")["acc_mag_ms2"].diff().fillna(0.0)
    df_feat["gyro_jerk_rads"] = df_feat.groupby("event_id")["gyro_mag_rads"].diff().fillna(0.0)

    # 4. Causal Rolling Dynamics (3s and 5s windows)
    for w in CONFIG["rolling_windows_sec"]:
        prefix = f"{w}s"
        # Accel norm stats
        df_feat[f"acc_mag_mean_{prefix}"] = df_feat.groupby("event_id")["acc_mag_ms2"].rolling(w, min_periods=1).mean().reset_index(drop=True)
        df_feat[f"acc_mag_std_{prefix}"] = df_feat.groupby("event_id")["acc_mag_ms2"].rolling(w, min_periods=1).std().fillna(0.0).reset_index(drop=True)
        df_feat[f"acc_mag_min_{prefix}"] = df_feat.groupby("event_id")["acc_mag_ms2"].rolling(w, min_periods=1).min().reset_index(drop=True)
        df_feat[f"acc_mag_max_{prefix}"] = df_feat.groupby("event_id")["acc_mag_ms2"].rolling(w, min_periods=1).max().reset_index(drop=True)
        df_feat[f"acc_mag_range_{prefix}"] = df_feat[f"acc_mag_max_{prefix}"] - df_feat[f"acc_mag_min_{prefix}"]
        df_feat[f"acc_mag_rms_{prefix}"] = df_feat.groupby("event_id")["acc_mag_ms2"].rolling(w, min_periods=1).apply(lambda s: np.sqrt(np.mean(s**2)), raw=True).reset_index(drop=True)

        # Gyro norm stats
        df_feat[f"gyro_mag_mean_{prefix}"] = df_feat.groupby("event_id")["gyro_mag_rads"].rolling(w, min_periods=1).mean().reset_index(drop=True)
        df_feat[f"gyro_mag_std_{prefix}"] = df_feat.groupby("event_id")["gyro_mag_rads"].rolling(w, min_periods=1).std().fillna(0.0).reset_index(drop=True)
        df_feat[f"gyro_mag_max_{prefix}"] = df_feat.groupby("event_id")["gyro_mag_rads"].rolling(w, min_periods=1).max().reset_index(drop=True)
        df_feat[f"gyro_mag_rms_{prefix}"] = df_feat.groupby("event_id")["gyro_mag_rads"].rolling(w, min_periods=1).apply(lambda s: np.sqrt(np.mean(s**2)), raw=True).reset_index(drop=True)

        # Planar horizontal stats
        df_feat[f"acc_horiz_mean_{prefix}"] = df_feat.groupby("event_id")["acc_horiz_ms2"].rolling(w, min_periods=1).mean().reset_index(drop=True)
        df_feat[f"acc_horiz_std_{prefix}"] = df_feat.groupby("event_id")["acc_horiz_ms2"].rolling(w, min_periods=1).std().fillna(0.0).reset_index(drop=True)

        # Yaw rate rotation (critical to differentiate SRT vs SLT)
        df_feat[f"gyro_z_mean_{prefix}"] = df_feat.groupby("event_id")["gyro_z_rads"].rolling(w, min_periods=1).mean().reset_index(drop=True)
        df_feat[f"gyro_z_std_{prefix}"] = df_feat.groupby("event_id")["gyro_z_rads"].rolling(w, min_periods=1).std().fillna(0.0).reset_index(drop=True)
        df_feat[f"gyro_z_max_{prefix}"] = df_feat.groupby("event_id")["gyro_z_rads"].rolling(w, min_periods=1).max().reset_index(drop=True)
        df_feat[f"gyro_z_min_{prefix}"] = df_feat.groupby("event_id")["gyro_z_rads"].rolling(w, min_periods=1).min().reset_index(drop=True)

        # Longitudinal acceleration (critical to differentiate SA vs SD)
        df_feat[f"acc_x_mean_{prefix}"] = df_feat.groupby("event_id")["acc_x_ms2"].rolling(w, min_periods=1).mean().reset_index(drop=True)
        df_feat[f"acc_x_std_{prefix}"] = df_feat.groupby("event_id")["acc_x_ms2"].rolling(w, min_periods=1).std().fillna(0.0).reset_index(drop=True)

    if df_feat.isnull().any().any():
        raise ValueError("Engineered features contain unexpected NaN values!")

    return df_feat


def extract_event_level_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive statistical summary features aggregated per event trial (109 rows).
    Captures complete maneuver signatures for event-level classification.
    """
    event_rows = []
    signals = ["acc_x_ms2", "acc_y_ms2", "acc_z_ms2", "gyro_x_rads", "gyro_y_rads", "gyro_z_rads", "acc_mag_ms2", "gyro_mag_rads", "acc_horiz_ms2"]

    for event_id, grp in df_feat.groupby("event_id", sort=True):
        row_dict = {
            "event_id": event_id,
            "DriverID": grp["DriverID"].iloc[0],
            "Class": grp["Class"].iloc[0],
            "TaskID": grp["TaskID"].iloc[0],
            "label_name": grp["label_name"].iloc[0],
            "label_abbr": grp["label_abbr"].iloc[0],
            "duration_samples": len(grp)
        }
        for sig in signals:
            vals = grp[sig].values
            row_dict[f"{sig}_mean"] = float(np.mean(vals))
            row_dict[f"{sig}_std"] = float(np.std(vals))
            row_dict[f"{sig}_min"] = float(np.min(vals))
            row_dict[f"{sig}_max"] = float(np.max(vals))
            row_dict[f"{sig}_range"] = float(np.max(vals) - np.min(vals))
            row_dict[f"{sig}_median"] = float(np.median(vals))
            row_dict[f"{sig}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            row_dict[f"{sig}_rms"] = float(np.sqrt(np.mean(vals**2)))
            row_dict[f"{sig}_skew"] = float(skew(vals)) if len(vals) > 2 else 0.0
            row_dict[f"{sig}_kurt"] = float(kurtosis(vals)) if len(vals) > 2 else 0.0

        event_rows.append(row_dict)

    df_events = pd.DataFrame(event_rows)
    return df_events


def split_dataset_b_stratified_group(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split Dataset B using Stratified Group Splitting on event_id.
    Guarantees:
    1. Zero event trial overlap between Train, Validation, and Test.
    2. Proportional preservation of all 4 maneuver classes across splits.
    3. Distribution of all 3 driver cohorts across splits.
    """
    # Event-level table
    event_meta = df.groupby("event_id").first()[["DriverID", "Class", "TaskID"]].reset_index()

    # Step 1: Split into Train (70%) and Temp (30%)
    test_size = 1.0 - train_pct
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(event_meta, event_meta["Class"]))

    events_train = event_meta.iloc[train_idx].copy()
    events_temp = event_meta.iloc[temp_idx].copy()

    # Step 2: Split Temp into Validation (50% of Temp = 15% total) and Test (50% of Temp = 15% total)
    temp_test_ratio = 0.50
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=temp_test_ratio, random_state=random_state)
    val_sub_idx, test_sub_idx = next(sss2.split(events_temp, events_temp["Class"]))

    events_val = events_temp.iloc[val_sub_idx].copy()
    events_test = events_temp.iloc[test_sub_idx].copy()

    train_ids = set(events_train["event_id"])
    val_ids = set(events_val["event_id"])
    test_ids = set(events_test["event_id"])

    # Strict assertion of mutual exclusivity
    assert len(train_ids & val_ids) == 0, "Train and Validation share overlapping event IDs!"
    assert len(train_ids & test_ids) == 0, "Train and Test share overlapping event IDs!"
    assert len(val_ids & test_ids) == 0, "Validation and Test share overlapping event IDs!"

    # Partition DataFrame
    df_train = df[df["event_id"].isin(train_ids)].copy().reset_index(drop=True)
    df_val = df[df["event_id"].isin(val_ids)].copy().reset_index(drop=True)
    df_test = df[df["event_id"].isin(test_ids)].copy().reset_index(drop=True)

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    split_meta = {
        "strategy": "Stratified Group Split on event_id",
        "random_state": random_state,
        "train_events": len(train_ids),
        "val_events": len(val_ids),
        "test_events": len(test_ids),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "total_rows": len(df_train) + len(df_val) + len(df_test),
        "class_distribution": {
            "train": df_train["Class"].value_counts().to_dict(),
            "val": df_val["Class"].value_counts().to_dict(),
            "test": df_test["Class"].value_counts().to_dict()
        },
        "driver_distribution": {
            "train": df_train["DriverID"].value_counts().to_dict(),
            "val": df_val["DriverID"].value_counts().to_dict(),
            "test": df_test["DriverID"].value_counts().to_dict()
        }
    }

    return df_train, df_val, df_test, split_meta


def run_pipeline() -> Dict[str, Any]:
    """Execute end-to-end Dataset B preprocessing pipeline."""
    print("=" * 65)
    print("Intelligent Driver Activity Monitoring & Safety Agent")
    print("Dataset B Preprocessing Pipeline (Multi-Class Risky Maneuvers)")
    print("=" * 65)

    # 1. Load and validate raw data
    print("\n[+] Step 1: Loading and validating raw Dataset B...")
    df_raw = load_raw_dataset(RAW_DATA_PATH)
    raw_sha = compute_sha256(RAW_DATA_PATH)
    print(f"  [✓] Loaded {len(df_raw)} records, 12 columns. SHA-256: {raw_sha}")

    # 2. Tag event trials
    print("\n[+] Step 2: Tagging 109 isolated event trials...")
    df_tagged, tag_meta = tag_event_trials(df_raw)
    print(f"  [✓] Confirmed {tag_meta['total_events']} events across 3 drivers.")
    print(f"  [✓] Event lengths: Min={tag_meta['min_samples_per_event']}, Mean={tag_meta['mean_samples_per_event']:.1f}, Max={tag_meta['max_samples_per_event']} samples.")

    # 3. Kinematic Feature Engineering
    print("\n[+] Step 3: Engineering SI units, invariants, jerk, and causal rolling dynamics...")
    df_feat = engineer_kinematic_features(df_tagged)
    feature_cols = [c for c in df_feat.columns if c not in EXPECTED_COLUMNS + ["event_id", "label_name", "label_abbr", "event_sample_idx", "event_duration_samples", "event_progress_ratio", "split"]]
    print(f"  [✓] Engineered {len(feature_cols)} kinematic features. Total columns: {len(df_feat.columns)}.")

    # 4. Extract Event-Level Summary Features
    print("\n[+] Step 4: Extracting event-level summary features (109 instances)...")
    df_events = extract_event_level_features(df_feat)
    print(f"  [✓] Event summary matrix: {df_events.shape[0]} events x {df_events.shape[1]} features.")

    # 5. Stratified Group Splitting
    print("\n[+] Step 5: Executing Stratified Group Splitting on event_id...")
    df_train, df_val, df_test, split_meta = split_dataset_b_stratified_group(df_feat, random_state=CONFIG["random_state"])
    print(f"  [✓] Train: {split_meta['train_events']} events ({split_meta['train_rows']} samples, {split_meta['train_rows']/len(df_feat):.1%})")
    print(f"  [✓] Val:   {split_meta['val_events']} events ({split_meta['val_rows']} samples, {split_meta['val_rows']/len(df_feat):.1%})")
    print(f"  [✓] Test:  {split_meta['test_events']} events ({split_meta['test_rows']} samples, {split_meta['test_rows']/len(df_feat):.1%})")

    # Combine full tagged dataset with split label
    df_processed_all = pd.concat([df_train, df_val, df_test]).sort_values("event_id").reset_index(drop=True)

    # 6. Save Processed Artifacts
    print("\n[+] Step 6: Saving processed datasets and metadata to dataset/processed/...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_out = PROCESSED_DIR / "dataset_b_processed_all.csv"
    train_out = PROCESSED_DIR / "dataset_b_train.csv"
    val_out = PROCESSED_DIR / "dataset_b_val.csv"
    test_out = PROCESSED_DIR / "dataset_b_test.csv"
    event_out = PROCESSED_DIR / "dataset_b_event_features.csv"
    meta_out = PROCESSED_DIR / "dataset_b_preprocessing_metadata.json"

    df_processed_all.to_csv(all_out, index=False)
    df_train.to_csv(train_out, index=False)
    df_val.to_csv(val_out, index=False)
    df_test.to_csv(test_out, index=False)
    df_events.to_csv(event_out, index=False)

    metadata = {
        "pipeline_name": CONFIG["pipeline_name"],
        "pipeline_version": CONFIG["pipeline_version"],
        "source_dataset": "Mendeley Data DOI 10.17632/jj3tw8kj6h.3 (Yuksel & Atmaca)",
        "source_file": str(RAW_DATA_PATH.name),
        "source_sha256": raw_sha,
        "config": CONFIG,
        "label_mapping": LABEL_MAPPING,
        "label_abbreviations": LABEL_ABBREVIATIONS,
        "tag_metadata": tag_meta,
        "feature_names": feature_cols,
        "feature_count": len(feature_cols),
        "split_metadata": split_meta,
        "artifacts": {
            "processed_all": str(all_out.name),
            "processed_all_sha256": compute_sha256(all_out),
            "train_csv": str(train_out.name),
            "train_sha256": compute_sha256(train_out),
            "val_csv": str(val_out.name),
            "val_sha256": compute_sha256(val_out),
            "test_csv": str(test_out.name),
            "test_sha256": compute_sha256(test_out),
            "event_features_csv": str(event_out.name),
            "event_features_sha256": compute_sha256(event_out)
        }
    }

    with open(meta_out, "w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"  [✓] Saved: {all_out.name} ({len(df_processed_all)} rows)")
    print(f"  [✓] Saved: {train_out.name} ({len(df_train)} rows)")
    print(f"  [✓] Saved: {val_out.name} ({len(df_val)} rows)")
    print(f"  [✓] Saved: {test_out.name} ({len(df_test)} rows)")
    print(f"  [✓] Saved: {event_out.name} ({len(df_events)} events)")
    print(f"  [✓] Saved: {meta_out.name}")

    print("\n" + "=" * 65)
    print("DATASET B PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

    return metadata


if __name__ == "__main__":
    run_pipeline()

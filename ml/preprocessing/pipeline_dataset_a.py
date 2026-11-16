"""
Reproducible Preprocessing Pipeline for Dataset A (Smartphone Driving)
Project: Intelligent Driver Activity Monitoring & Safety Agent

Purpose:
- Synchronize Accelerometer and Gyroscope sensor streams via epoch timestamps.
- Resample to a uniform, calibrated sampling frequency (50 Hz).
- Calculate kinematic & derived features (Acceleration Magnitude, Linear Acceleration, Gyro Magnitude, Jerk).
- Segment into temporal sliding windows without crossing driver/recording boundaries.
- Compute statistical and frequency features per window.
- Apply a temporal split (Train 70% / Val 15% / Test 15%) per driver with a guard band buffer to strictly prevent data leakage.
- Save processed outputs into dataset/processed/ with full metadata documentation.
"""

import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Repository Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = REPO_ROOT / "dataset" / "raw" / "drivers"
PROCESSED_DIR = REPO_ROOT / "dataset" / "processed"

# Pipeline Configuration
CONFIG = {
    "target_sampling_rate_hz": 50,          # Uniform resampled frequency
    "sampling_interval_ms": 20,             # 1000 / 50 Hz = 20 ms
    "window_size_seconds": 5.0,             # 5.0 second window
    "window_size_samples": 250,             # 5.0s * 50Hz = 250 samples
    "window_stride_seconds": 2.5,           # 2.5 second stride (50% overlap for train/feature extraction)
    "window_stride_samples": 125,           # 2.5s * 50Hz = 125 samples
    "guard_band_seconds": 10.0,             # 10s separation buffer between splits to prevent window leakage
    "guard_band_samples": 500,              # 10s * 50Hz = 500 samples
    "train_ratio": 0.70,                    # Chronological first 70% of driving
    "val_ratio": 0.15,                      # Next 15%
    "test_ratio": 0.15,                     # Final 15%
    "excluded_drivers": ["Driver-2"],       # Excluded due to verified 12s recording duration anomaly
    "valid_drivers": [
        "Driver-1", "Driver-3", "Driver-4",
        "Driver-5", "Driver-6", "Driver-7"
    ]
}

def synchronize_sensors(driver_name: str) -> pd.DataFrame:
    """
    Load and synchronize Accelerometer and Gyroscope streams for a given driver.
    """
    acc_path = RAW_DIR / driver_name / "Accelerometer.csv"
    gyro_path = RAW_DIR / driver_name / "Gyroscope.csv"

    if not acc_path.exists() or not gyro_path.exists():
        raise FileNotFoundError(f"Missing sensor files for {driver_name}")

    df_acc = pd.read_csv(acc_path)
    df_gyro = pd.read_csv(gyro_path)

    # Calculate global epoch timestamp in milliseconds
    t0_acc_sec = df_acc["Unix Timestamp"].iloc[0]
    t0_gyro_sec = df_gyro["Unix Timestamp"].iloc[0]

    df_acc["epoch_ms"] = t0_acc_sec * 1000 + df_acc["Milliseconds"]
    df_gyro["epoch_ms"] = t0_gyro_sec * 1000 + df_gyro["Milliseconds"]

    # Determine overlapping temporal bounds
    t_start = max(df_acc["epoch_ms"].iloc[0], df_gyro["epoch_ms"].iloc[0])
    t_end = min(df_acc["epoch_ms"].iloc[-1], df_gyro["epoch_ms"].iloc[-1])

    if t_end <= t_start:
        raise ValueError(f"No temporal overlap for {driver_name}")

    # Create uniform resampling grid
    step_ms = CONFIG["sampling_interval_ms"]
    resampled_times = np.arange(t_start, t_end, step_ms)

    # Sort and drop exact duplicates
    acc_clean = df_acc.drop_duplicates(subset=["epoch_ms"]).sort_values("epoch_ms")
    gyro_clean = df_gyro.drop_duplicates(subset=["epoch_ms"]).sort_values("epoch_ms")

    # Interpolate Accelerometer to uniform grid
    acc_x = np.interp(resampled_times, acc_clean["epoch_ms"], acc_clean["X"])
    acc_y = np.interp(resampled_times, acc_clean["epoch_ms"], acc_clean["Y"])
    acc_z = np.interp(resampled_times, acc_clean["epoch_ms"], acc_clean["Z"])

    # Interpolate Gyroscope to uniform grid
    gyro_x = np.interp(resampled_times, gyro_clean["epoch_ms"], gyro_clean["X"])
    gyro_y = np.interp(resampled_times, gyro_clean["epoch_ms"], gyro_clean["Y"])
    gyro_z = np.interp(resampled_times, gyro_clean["epoch_ms"], gyro_clean["Z"])

    synced_df = pd.DataFrame({
        "epoch_ms": resampled_times,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "driver": driver_name
    })

    # Compute derived physical channels
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    synced_df["acc_mag"] = acc_mag
    synced_df["acc_linear_mag"] = np.abs(acc_mag - 9.80665)  # Gravity-compensated magnitude
    synced_df["gyro_mag"] = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    # Jerk (derivative of total acceleration)
    dt = step_ms / 1000.0
    synced_df["jerk"] = np.gradient(acc_mag, dt)

    return synced_df

def extract_window_features(window_data: pd.DataFrame, driver_id: str, split_label: str, window_idx: int) -> dict:
    """
    Extract comprehensive statistical and kinematic features from a single temporal window.
    """
    channels = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "acc_mag", "acc_linear_mag", "gyro_mag", "jerk"
    ]

    features = {
        "driver_id": driver_id,
        "window_idx": window_idx,
        "split": split_label,
        "start_epoch_ms": window_data["epoch_ms"].iloc[0],
        "end_epoch_ms": window_data["epoch_ms"].iloc[-1]
    }

    for col in channels:
        vals = window_data[col].values
        features[f"{col}_mean"] = float(np.mean(vals))
        features[f"{col}_std"] = float(np.std(vals))
        features[f"{col}_min"] = float(np.min(vals))
        features[f"{col}_max"] = float(np.max(vals))
        features[f"{col}_median"] = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        features[f"{col}_iqr"] = float(q75 - q25)
        features[f"{col}_skew"] = float(skew(vals))
        features[f"{col}_kurt"] = float(kurtosis(vals))
        features[f"{col}_energy"] = float(np.sum(vals**2) / len(vals))

    return features

def process_driver_windows(synced_df: pd.DataFrame, driver_name: str) -> list:
    """
    Segment synchronized time-series into Train, Validation, and Test partitions
    separated by explicit guard bands, and extract window features.
    """
    total_samples = len(synced_df)
    n_train = int(total_samples * CONFIG["train_ratio"])
    n_val = int(total_samples * CONFIG["val_ratio"])
    guard = CONFIG["guard_band_samples"]

    # Strict chronological partition boundaries
    train_end = n_train
    val_start = train_end + guard
    val_end = val_start + n_val
    test_start = val_end + guard
    test_end = total_samples

    partitions = [
        ("train", 0, train_end),
        ("val", val_start, val_end),
        ("test", test_start, test_end)
    ]

    win_size = CONFIG["window_size_samples"]
    win_stride = CONFIG["window_stride_samples"]

    window_records = []
    global_win_id = 0

    for split_name, start_idx, end_idx in partitions:
        if end_idx <= start_idx + win_size:
            continue

        part_df = synced_df.iloc[start_idx:end_idx].reset_index(drop=True)
        part_len = len(part_df)

        for w_start in range(0, part_len - win_size + 1, win_stride):
            w_df = part_df.iloc[w_start:w_start + win_size]
            feat = extract_window_features(w_df, driver_name, split_name, global_win_id)
            window_records.append(feat)
            global_win_id += 1

    return window_records

def run_pipeline():
    """Execute complete reproducible preprocessing pipeline for Dataset A."""
    print("=" * 65)
    print("EXECUTING REPRODUCIBLE PREPROCESSING PIPELINE (DATASET A)")
    print("=" * 65)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    driver_stats = {}

    for driver in CONFIG["valid_drivers"]:
        print(f"\n[+] Processing {driver}...")
        try:
            synced_df = synchronize_sensors(driver)
            dur_s = (synced_df["epoch_ms"].iloc[-1] - synced_df["epoch_ms"].iloc[0]) / 1000.0
            print(f"    Synchronized: {len(synced_df):,} samples ({dur_s:.1f} seconds / {dur_s/60:.1f} min)")

            windows = process_driver_windows(synced_df, driver)
            all_features.extend(windows)

            splits = [w["split"] for w in windows]
            driver_stats[driver] = {
                "total_resampled_samples": len(synced_df),
                "duration_seconds": round(dur_s, 2),
                "total_windows": len(windows),
                "train_windows": splits.count("train"),
                "val_windows": splits.count("val"),
                "test_windows": splits.count("test")
            }
            print(f"    Extracted windows: Total={len(windows)} (Train={splits.count('train')}, Val={splits.count('val')}, Test={splits.count('test')})")
        except Exception as e:
            print(f"    [-] Error processing {driver}: {e}")

    # Build final DataFrame
    df_features = pd.DataFrame(all_features)
    print(f"\n[✓] Total feature matrix shape: {df_features.shape}")

    # Save processed outputs
    csv_out = PROCESSED_DIR / "dataset_a_features.csv"
    df_features.to_csv(csv_out, index=False)
    print(f"[✓] Saved processed features to: {csv_out} ({csv_out.stat().st_size / (1024*1024):.2f} MB)")

    # Save preprocessing metadata
    metadata = {
        "pipeline_name": "Dataset A Preprocessing & Windowing Pipeline",
        "version": "1.0.0",
        "config": CONFIG,
        "driver_statistics": driver_stats,
        "total_windows": len(df_features),
        "feature_count": df_features.shape[1] - 5, # Exclude metadata columns
        "split_distribution": df_features["split"].value_counts().to_dict(),
        "driver_distribution": df_features["driver_id"].value_counts().to_dict(),
        "output_file": str(csv_out.name),
        "output_sha256": hashlib.sha256(open(csv_out, "rb").read()).hexdigest()
    }

    meta_out = PROCESSED_DIR / "preprocessing_metadata.json"
    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Saved preprocessing metadata to: {meta_out}")

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY (NO MODEL TRAINING PERFORMED)")
    print("=" * 65)

if __name__ == "__main__":
    run_pipeline()

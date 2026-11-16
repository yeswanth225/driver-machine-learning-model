"""
Dataset C Preprocessing Pipeline: Binary Driving Behaviour Classification
========================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Source: Mendeley Data (DOI: 10.17632/5stn873wft.1) - Shardul Nazirkar
Input File: dataset/raw/dataset_03_5stn873wft/3_FinalDatasetCsv.csv

Pipeline Responsibilities:
1. Validates schema, data types, and target labels (0=Normal, 1=Aggressive).
2. Investigates and removes 9 exact duplicate records.
3. Identifies the two underlying physical recording sessions:
   - Session 1: 7,599 clean rows (17:14:52 trip)
   - Session 2: 6,641 clean rows (18:45:12 trip)
4. Computes physical kinematic signals:
   - Accelerometer magnitude (acc_mag), Gyroscope magnitude (gyro_mag)
   - Horizontal planar acceleration (acc_horiz)
   - Temporal jerk derivatives (acc_jerk, gyro_jerk)
5. Extracts causal, backward-looking rolling statistics (1s, 2s, 3s):
   - Mean, std, min, max, range, and yaw rate dynamics (gyro_z std/max).
6. Applies strict chronological splitting within each session:
   - Train (70%), Validation (15%), Test (15%)
   - 10.0-second (100 samples) guard-band buffers dropped between splits
   - Zero future leakage or window overlap across split boundaries.
7. Saves processed splits and metadata to dataset/processed/.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "raw", "dataset_03_5stn873wft", "3_FinalDatasetCsv.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# Metadata constants
LABEL_MAPPING = {
    0: "Normal Driving",
    1: "Aggressive Driving"
}

SAMPLING_RATE_HZ = 10.0
GUARD_BAND_SECONDS = 10.0
GUARD_BAND_SAMPLES = int(GUARD_BAND_SECONDS * SAMPLING_RATE_HZ) # 100 samples


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_dataset(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and validate raw Dataset C CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    expected_cols = {'Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'label'}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Dataset C: {missing}")
    
    return df


def remove_duplicates_and_tag_sessions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Remove exact duplicate records and segment the dataset into its two physical sessions.
    Session 1 corresponds to original 2_20210317_171452.csv (7,603 raw rows).
    Session 2 corresponds to original 1_20210317_184512.csv (6,646 raw rows).
    """
    raw_count = len(df)
    s1_boundary_raw = 7603
    
    # Check duplicates per session
    df_s1_raw = df.iloc[:s1_boundary_raw]
    df_s2_raw = df.iloc[s1_boundary_raw:]
    
    dups_s1 = int(df_s1_raw.duplicated().sum())
    dups_s2 = int(df_s2_raw.duplicated().sum())
    
    # Global drop duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    clean_count = len(df_clean)
    
    # Session 1 length in cleaned data
    s1_clean_len = s1_boundary_raw - dups_s1
    df_s1 = df_clean.iloc[:s1_clean_len].copy().reset_index(drop=True)
    df_s2 = df_clean.iloc[s1_clean_len:].copy().reset_index(drop=True)
    
    df_s1['session'] = 'session_1'
    df_s2['session'] = 'session_2'
    
    metadata = {
        "raw_total_rows": raw_count,
        "clean_total_rows": clean_count,
        "duplicates_removed_total": raw_count - clean_count,
        "session_1_raw_rows": len(df_s1_raw),
        "session_1_clean_rows": len(df_s1),
        "session_1_duplicates": dups_s1,
        "session_2_raw_rows": len(df_s2_raw),
        "session_2_clean_rows": len(df_s2),
        "session_2_duplicates": dups_s2,
    }
    
    return df_s1, df_s2, metadata


def engineer_kinematic_features(df_session: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physical invariants, temporal derivatives, and causal backward rolling statistics.
    All rolling windows look strictly backward in time, ensuring zero future leakage.
    """
    df = df_session.copy()
    
    # 1. Physical Magnitudes
    df['acc_mag'] = np.sqrt(df['Acc X']**2 + df['Acc Y']**2 + df['Acc Z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    df['acc_horiz'] = np.sqrt(df['Acc X']**2 + df['Acc Y']**2)
    
    # 2. Temporal Jerk Derivatives (backward difference)
    df['acc_jerk'] = df['acc_mag'].diff().fillna(0.0)
    df['gyro_jerk'] = df['gyro_mag'].diff().fillna(0.0)
    
    # 3. Causal Rolling Temporal Windows (1s=10 samples, 2s=20 samples, 3s=30 samples)
    for w in [10, 20, 30]:
        sec = int(w / SAMPLING_RATE_HZ)
        prefix = f"{sec}s"
        
        # Accelerometer dynamics
        df[f'acc_mag_mean_{prefix}'] = df['acc_mag'].rolling(w, min_periods=1).mean()
        df[f'acc_mag_std_{prefix}'] = df['acc_mag'].rolling(w, min_periods=1).std().fillna(0.0)
        df[f'acc_mag_max_{prefix}'] = df['acc_mag'].rolling(w, min_periods=1).max()
        df[f'acc_mag_min_{prefix}'] = df['acc_mag'].rolling(w, min_periods=1).min()
        df[f'acc_mag_range_{prefix}'] = df[f'acc_mag_max_{prefix}'] - df[f'acc_mag_min_{prefix}']
        
        # Gyroscope dynamics (angular velocity)
        df[f'gyro_mag_mean_{prefix}'] = df['gyro_mag'].rolling(w, min_periods=1).mean()
        df[f'gyro_mag_std_{prefix}'] = df['gyro_mag'].rolling(w, min_periods=1).std().fillna(0.0)
        df[f'gyro_mag_max_{prefix}'] = df['gyro_mag'].rolling(w, min_periods=1).max()
        
        # Planar horizontal acceleration
        df[f'acc_horiz_mean_{prefix}'] = df['acc_horiz'].rolling(w, min_periods=1).mean()
        df[f'acc_horiz_std_{prefix}'] = df['acc_horiz'].rolling(w, min_periods=1).std().fillna(0.0)
        
        # Yaw rate rotation (sharp turn / swerve indicator)
        df[f'gyro_z_std_{prefix}'] = df['gyro_z'].rolling(w, min_periods=1).std().fillna(0.0)
        df[f'gyro_z_max_{prefix}'] = df['gyro_z'].abs().rolling(w, min_periods=1).max()
        
    return df


def split_session_chronological(df_session: pd.DataFrame, 
                                train_pct: float = 0.70, 
                                val_pct: float = 0.15,
                                guard_samples: int = GUARD_BAND_SAMPLES) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological partition of a session into Train, Validation, and Test sets
    with 10-second guard bands dropped between consecutive partitions to prevent temporal leakage.
    """
    n = len(df_session)
    n_train = int(train_pct * n)
    n_val = int(val_pct * n)
    
    # Train split: [0, n_train - guard_samples]
    train = df_session.iloc[:n_train - guard_samples].copy()
    train['split'] = 'train'
    
    # Validation split: [n_train, n_train + n_val - guard_samples]
    val = df_session.iloc[n_train:n_train + n_val - guard_samples].copy()
    val['split'] = 'val'
    
    # Test split: [n_train + n_val, n]
    test = df_session.iloc[n_train + n_val:].copy()
    test['split'] = 'test'
    
    return train, val, test


def run_pipeline() -> Dict:
    """Execute complete Dataset C preprocessing and export datasets."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("[1/5] Loading raw Dataset C...")
    df_raw = load_raw_dataset(RAW_DATA_PATH)
    raw_sha256 = compute_sha256(RAW_DATA_PATH)
    
    print("[2/5] Deduplicating and segmenting into physical sessions...")
    df_s1, df_s2, session_meta = remove_duplicates_and_tag_sessions(df_raw)
    
    print("[3/5] Engineering causal kinematic features...")
    df_s1_feat = engineer_kinematic_features(df_s1)
    df_s2_feat = engineer_kinematic_features(df_s2)
    
    print("[4/5] Applying chronological splits with 10s guard bands...")
    tr1, va1, te1 = split_session_chronological(df_s1_feat)
    tr2, va2, te2 = split_session_chronological(df_s2_feat)
    
    train_df = pd.concat([tr1, tr2]).reset_index(drop=True)
    val_df = pd.concat([va1, va2]).reset_index(drop=True)
    test_df = pd.concat([te1, te2]).reset_index(drop=True)
    
    # Define feature columns
    non_feature_cols = ['label', 'split', 'session']
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]
    
    # Save processed CSV files
    train_path = os.path.join(PROCESSED_DIR, "dataset_c_train.csv")
    val_path = os.path.join(PROCESSED_DIR, "dataset_c_val.csv")
    test_path = os.path.join(PROCESSED_DIR, "dataset_c_test.csv")
    full_path = os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv")
    
    all_processed = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    all_processed.to_csv(full_path, index=False)
    
    # Summary metadata
    metadata = {
        "pipeline_name": "Dataset C Binary Driving Classification Preprocessor",
        "pipeline_version": "1.0.0",
        "source_raw_file": os.path.basename(RAW_DATA_PATH),
        "source_raw_sha256": raw_sha256,
        "label_mapping": LABEL_MAPPING,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "guard_band_seconds": GUARD_BAND_SECONDS,
        "guard_band_samples_per_split": GUARD_BAND_SAMPLES,
        "session_summary": session_meta,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "guard_band_dropped": (session_meta["clean_total_rows"] - len(train_df) - len(val_df) - len(test_df)),
            "total_clean_rows": session_meta["clean_total_rows"]
        },
        "class_distributions": {
            "train": train_df['label'].value_counts().to_dict(),
            "validation": val_df['label'].value_counts().to_dict(),
            "test": test_df['label'].value_counts().to_dict(),
        },
        "output_files": {
            "train_csv": "dataset_c_train.csv",
            "val_csv": "dataset_c_val.csv",
            "test_csv": "dataset_c_test.csv",
            "full_csv": "dataset_c_processed_all.csv",
            "train_sha256": compute_sha256(train_path),
            "val_sha256": compute_sha256(val_path),
            "test_sha256": compute_sha256(test_path),
        }
    }
    
    meta_path = os.path.join(PROCESSED_DIR, "dataset_c_preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("[5/5] Preprocessing complete. Metadata saved to dataset_c_preprocessing_metadata.json.")
    return metadata


if __name__ == "__main__":
    meta = run_pipeline()
    print(f"Features: {meta['feature_count']}, Train: {meta['split_counts']['train']}, Val: {meta['split_counts']['validation']}, Test: {meta['split_counts']['test']}")

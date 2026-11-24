"""
Causal Feature Engineering Pipeline for Pipeline A (Driving Behaviour Classification)
=====================================================================================
Features Engineered:
1. Physical Directional Decomposition:
   - acc_planar = sqrt(Acc X^2 + Acc Y^2)  [Centrifugal & longitudinal force]
   - acc_vertical = abs(Acc Z)             [Vertical bump / gravity perturbation]
   - gyro_yaw = abs(gyro_z)                [Vehicle heading rotation]
   - gyro_wobble = sqrt(gyro_x^2 + gyro_y^2) [Mount pitch/roll vibration]
2. Road-Bump & Noise Decoupling Ratios:
   - bump_ratio = acc_vertical / (acc_planar + 0.1)
   - yaw_dominance = gyro_yaw / (gyro_wobble + 0.01)
   - planar_energy_ratio = (acc_planar^2) / (acc_mag^2 + 0.01)
3. Causal Exponential Moving Averages (EMA):
   - acc_planar_ema_fast (alpha=0.2, ~0.5-1s)
   - acc_planar_ema_slow (alpha=0.08, ~2-3s)
   - gyro_yaw_ema_fast (alpha=0.2)
   - gyro_yaw_ema_slow (alpha=0.08)
4. Multi-Scale Causal Rolling Kinematics (windows: 1s=10, 2s=20, 3s=30, 5s=50):
   - Mean, std, max, min, range of planar acceleration and yaw rotation
   - Kinetic energy integrals: mean(acc_planar^2), mean(gyro_yaw^2)
   - Causal Jerk: diff(acc_planar), diff(gyro_yaw), and rolling jerk std
5. Strict temporal causality:
   - All operations are strictly backward-looking (no forward shift, no centered windows)
   - Features computed independently per session (session buffer cleanly flushed)
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
EXP_DIR = os.path.join(BASE_DIR, "dataset", "experiments", "pipeline_a")

def compute_causal_features(df_session: pd.DataFrame) -> pd.DataFrame:
    """Compute causal kinematic features on a single session DataFrame."""
    df = df_session.copy().reset_index(drop=True)
    
    # 1. Raw Axes & Base Magnitudes
    ax = df['Acc X']
    ay = df['Acc Y']
    az = df['Acc Z']
    gx = df['gyro_x']
    gy = df['gyro_y']
    gz = df['gyro_z']
    
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    acc_planar = np.sqrt(ax**2 + ay**2)
    acc_vertical = np.abs(az)
    gyro_yaw = np.abs(gz)
    gyro_wobble = np.sqrt(gx**2 + gy**2)
    
    df['acc_mag'] = acc_mag
    df['gyro_mag'] = gyro_mag
    df['acc_planar'] = acc_planar
    df['acc_vertical'] = acc_vertical
    df['gyro_yaw'] = gyro_yaw
    df['gyro_wobble'] = gyro_wobble
    
    # 2. Decoupling Ratios (bump & wobble discriminators)
    df['bump_ratio'] = acc_vertical / (acc_planar + 0.1)
    df['yaw_dominance'] = gyro_yaw / (gyro_wobble + 0.01)
    df['planar_energy_ratio'] = (acc_planar**2) / (acc_mag**2 + 0.01)
    
    # 3. Causal Jerk Derivatives
    df['acc_jerk'] = acc_planar.diff().fillna(0.0)
    df['gyro_jerk'] = gyro_yaw.diff().fillna(0.0)
    
    # 4. Causal Exponential Moving Averages (low-pass filtering)
    df['acc_planar_ema_fast'] = acc_planar.ewm(alpha=0.20, adjust=False).mean()
    df['acc_planar_ema_slow'] = acc_planar.ewm(alpha=0.08, adjust=False).mean()
    df['gyro_yaw_ema_fast'] = gyro_yaw.ewm(alpha=0.20, adjust=False).mean()
    df['gyro_yaw_ema_slow'] = gyro_yaw.ewm(alpha=0.08, adjust=False).mean()
    
    # 5. Multi-Scale Causal Rolling Windows
    # 10 samples = 1s, 20 samples = 2s, 30 samples = 3s, 50 samples = 5s
    for w in [10, 20, 30, 50]:
        sec = int(w / 10.0)
        prefix = f"{sec}s"
        
        # Planar Acceleration Dynamics
        df[f'acc_planar_mean_{prefix}'] = acc_planar.rolling(w, min_periods=1).mean()
        df[f'acc_planar_std_{prefix}'] = acc_planar.rolling(w, min_periods=1).std().fillna(0.0)
        df[f'acc_planar_max_{prefix}'] = acc_planar.rolling(w, min_periods=1).max()
        df[f'acc_planar_min_{prefix}'] = acc_planar.rolling(w, min_periods=1).min()
        df[f'acc_planar_range_{prefix}'] = df[f'acc_planar_max_{prefix}'] - df[f'acc_planar_min_{prefix}']
        df[f'acc_planar_energy_{prefix}'] = (acc_planar**2).rolling(w, min_periods=1).mean()
        
        # Yaw Steering Dynamics
        df[f'gyro_yaw_mean_{prefix}'] = gyro_yaw.rolling(w, min_periods=1).mean()
        df[f'gyro_yaw_std_{prefix}'] = gyro_yaw.rolling(w, min_periods=1).std().fillna(0.0)
        df[f'gyro_yaw_max_{prefix}'] = gyro_yaw.rolling(w, min_periods=1).max()
        df[f'gyro_yaw_energy_{prefix}'] = (gyro_yaw**2).rolling(w, min_periods=1).mean()
        
        # Vertical Chatter Dynamics (Road Roughness indicator)
        df[f'acc_vertical_std_{prefix}'] = acc_vertical.rolling(w, min_periods=1).std().fillna(0.0)
        df[f'acc_vertical_mean_{prefix}'] = acc_vertical.rolling(w, min_periods=1).mean()
        
        # Jerk Dynamics
        df[f'acc_jerk_std_{prefix}'] = df['acc_jerk'].rolling(w, min_periods=1).std().fillna(0.0)
        
    return df

def generate_engineered_datasets():
    os.makedirs(EXP_DIR, exist_ok=True)
    
    # Load original processed_all
    df_all = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    
    # Process Session 1 and Session 2 separately to avoid boundary bleed
    s1 = df_all[df_all['session'] == 'session_1'].copy()
    s2 = df_all[df_all['session'] == 'session_2'].copy()
    
    print("Computing enhanced causal features for Session 1...")
    s1_feat = compute_causal_features(s1)
    print("Computing enhanced causal features for Session 2...")
    s2_feat = compute_causal_features(s2)
    
    all_enhanced = pd.concat([s1_feat, s2_feat]).reset_index(drop=True)
    
    # Partition according to original split column
    train_df = all_enhanced[all_enhanced['split'] == 'train'].reset_index(drop=True)
    val_df = all_enhanced[all_enhanced['split'] == 'val'].reset_index(drop=True)
    test_df = all_enhanced[all_enhanced['split'] == 'test'].reset_index(drop=True)
    
    # Identify non-feature columns
    meta_cols = ['label', 'split', 'session']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Save engineered feature sets
    train_path = os.path.join(EXP_DIR, "dataset_c_enhanced_train.csv")
    val_path = os.path.join(EXP_DIR, "dataset_c_enhanced_val.csv")
    test_path = os.path.join(EXP_DIR, "dataset_c_enhanced_test.csv")
    full_path = os.path.join(EXP_DIR, "dataset_c_enhanced_all.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    all_enhanced.to_csv(full_path, index=False)
    
    metadata = {
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "class_distributions": {
            "train": train_df['label'].value_counts().to_dict(),
            "val": val_df['label'].value_counts().to_dict(),
            "test": test_df['label'].value_counts().to_dict(),
        }
    }
    
    with open(os.path.join(EXP_DIR, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Engineered datasets saved in {EXP_DIR}")
    return metadata

if __name__ == "__main__":
    generate_engineered_datasets()

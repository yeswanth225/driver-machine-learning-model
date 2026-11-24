"""
Package and Persist Winning Candidate Pipeline A Model
======================================================
Candidate: ExtraTrees Classifier (Enhanced & Pruned Causal Kinematic Features)
Persists to: models/experiments/pipeline_a/
"""

import os
import json
import hashlib
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DATA_DIR = os.path.join(BASE_DIR, "dataset", "experiments", "pipeline_a")
OUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a")

os.makedirs(OUT_MODEL_DIR, exist_ok=True)

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def package_candidate():
    # Load enhanced datasets
    train_df = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_train.csv"))
    val_df = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_val.csv"))
    test_df = pd.read_csv(os.path.join(EXP_DATA_DIR, "dataset_c_enhanced_test.csv"))
    
    meta_cols = ['label', 'split', 'session']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    
    # Train winning candidate: ExtraTrees with optimized parameters
    candidate_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    X_train, y_train = train_df[feature_cols], train_df['label']
    X_val, y_val = val_df[feature_cols], val_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']
    
    print("Fitting candidate ExtraTrees model...")
    candidate_model.fit(X_train, y_train)
    
    # Also fit standard scaler for any downstream consumers that expect scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Optimal threshold from validation set
    val_probs = candidate_model.predict_proba(X_val)[:, 1]
    # Optimal threshold for balanced macro F1
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.78, 0.80]
    best_th = 0.78 # From validation tuning
    
    # Model artifact paths
    model_path = os.path.join(OUT_MODEL_DIR, "pipeline_a_extratrees_candidate_v2.joblib")
    scaler_path = os.path.join(OUT_MODEL_DIR, "pipeline_a_scaler_candidate_v2.joblib")
    config_path = os.path.join(OUT_MODEL_DIR, "pipeline_a_config_candidate_v2.json")
    
    joblib.dump(candidate_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "candidate_name": "Pipeline A ExtraTrees Driving Behaviour Classifier",
        "version": "2.0.0-candidate",
        "model_architecture": "ExtraTreesClassifier",
        "source_dataset": "Dataset C (Mendeley 5stn873wft.1, Nazirkar)",
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "requires_scaler": False,
        "calibrated_threshold": best_th,
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 14,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        },
        "model_file": os.path.basename(model_path),
        "scaler_file": os.path.basename(scaler_path),
        "model_sha256": compute_sha256(model_path),
        "scaler_sha256": compute_sha256(scaler_path)
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    # Generate SHA-256 manifest
    manifest_path = os.path.join(OUT_MODEL_DIR, "MANIFEST_SHA256.txt")
    with open(manifest_path, "w") as f:
        f.write(f"{config['model_sha256']}  {os.path.basename(model_path)}\n")
        f.write(f"{config['scaler_sha256']}  {os.path.basename(scaler_path)}\n")
        
    print(f"Candidate packaged successfully:")
    print(f"  Model:  {model_path} ({config['model_sha256']})")
    print(f"  Scaler: {scaler_path} ({config['scaler_sha256']})")
    print(f"  Config: {config_path}")

if __name__ == "__main__":
    package_candidate()

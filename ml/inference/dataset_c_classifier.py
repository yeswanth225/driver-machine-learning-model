"""
Inference Wrapper: Dataset C Binary Driving Behaviour Classifier
================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple


class DatasetCBinaryClassifier:
    """Production inference wrapper for Dataset C Binary Driving Behavior Classification."""
    
    def __init__(
        self,
        model_version: str = "1.0.0",
        model_dir: str = None,
        model_path: str = None,
        scaler_path: str = None,
        config_path: str = None,
        decision_threshold: float = None
    ):
        self.version = model_version
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if model_dir is None:
            model_dir = os.path.join(base_dir, "ml", "saved_models")
        self.model_dir = model_dir
        
        # Load artifacts
        self.config_path = config_path or os.path.join(model_dir, f"dataset_c_config_v{self.version}.json")
        self.model_path = model_path or os.path.join(model_dir, f"dataset_c_binary_model_v{self.version}.joblib")
        self.scaler_path = scaler_path or os.path.join(model_dir, f"dataset_c_scaler_v{self.version}.joblib")
        
        # Check if loading from experiment dir
        if not os.path.exists(self.config_path):
            exp_summary_path = os.path.join(model_dir, "dataset_c_experiment_summary.json")
            if os.path.exists(exp_summary_path):
                self.config_path = exp_summary_path
                if model_path is None:
                    self.model_path = os.path.join(model_dir, "dataset_c_support_vector_machine_exp_v1.joblib")
                if scaler_path is None:
                    self.scaler_path = os.path.join(model_dir, "dataset_c_scaler_exp_v1.joblib")
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            
        self.feature_names = self.config["feature_names"]
        
        # Handle label mapping
        raw_mapping = self.config.get("label_mapping", {"0": "Normal Driving", "1": "Aggressive Driving"})
        self.label_mapping = {int(k): v for k, v in raw_mapping.items()}
        self.requires_scaler = self.config.get("requires_scaler", True)
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path) if os.path.exists(self.scaler_path) else None

        # Decision threshold (e.g. 0.84 for tuned model, or default None -> model.predict)
        if decision_threshold is not None:
            self.decision_threshold = float(decision_threshold)
        elif "results" in self.config and "Support Vector Machine" in self.config.get("results", {}):
            svm_res = self.config["results"]["Support Vector Machine"]
            self.decision_threshold = float(svm_res.get("test_metrics", {}).get("decision_threshold", 0.5))
        else:
            self.decision_threshold = None

    def validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input features match required schema and handle missing values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
            
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")
            
        # Reorder to match exact training feature order
        X_ordered = X[self.feature_names].copy()
        
        # Check for NaN / infinite values
        if X_ordered.isnull().any().any():
            raise ValueError("Input features contain NaN values. Clean/impute before prediction.")
        if np.isinf(X_ordered.values).any():
            raise ValueError("Input features contain infinite values.")
            
        return X_ordered

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate binary class predictions (0=Normal, 1=Aggressive)."""
        if self.decision_threshold is not None:
            probs = self.predict_proba(X)[:, 1]
            return (probs >= self.decision_threshold).astype(int)
        
        X_valid = self.validate_features(X)
        if self.requires_scaler and self.scaler is not None:
            X_proc = self.scaler.transform(X_valid)
        else:
            X_proc = X_valid
        return self.model.predict(X_proc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probability estimates [P(Normal), P(Aggressive)]."""
        X_valid = self.validate_features(X)
        if self.requires_scaler and self.scaler is not None:
            X_proc = self.scaler.transform(X_valid)
        else:
            X_proc = X_valid
            
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_proc)
        elif hasattr(self.model, "decision_function"):
            df_vals = self.model.decision_function(X_proc)
            # Sigmoid approximation
            prob_pos = 1.0 / (1.0 + np.exp(-df_vals))
            return np.column_stack([1.0 - prob_pos, prob_pos])
        else:
            raise NotImplementedError("Model does not support probability estimation.")

    def predict_label(self, X: pd.DataFrame) -> List[str]:
        """Generate human-readable class labels ('Normal Driving' or 'Aggressive Driving')."""
        preds = self.predict(X)
        return [self.label_mapping[p] for p in preds]

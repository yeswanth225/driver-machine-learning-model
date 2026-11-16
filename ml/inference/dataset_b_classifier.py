"""
Inference Wrapper: Dataset B 4-Class Maneuver Classifier
========================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: 4B - Model Inference Wrapper
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
import joblib


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXP_DIR = REPO_ROOT / "models" / "experiments" / "dataset_b"

LABEL_MAPPING = {
    1: "Sudden Acceleration",
    2: "Sudden Deceleration",
    3: "Sudden Right Turn",
    4: "Sudden Left Turn"
}

LABEL_ABBR = {
    1: "SA",
    2: "SD",
    3: "SRT",
    4: "SLT"
}

QUARANTINED_FEATURES = {
    "event_duration_samples",
    "event_progress_ratio",
    "event_sample_idx"
}


class DatasetBManeuverClassifier:
    """Production inference wrapper for Dataset B 4-Class Maneuver Classification."""

    def __init__(
        self,
        model_name: str = "extra_trees",
        experiment_dir: Union[str, Path] = None
    ):
        if experiment_dir is None:
            self.experiment_dir = DEFAULT_EXP_DIR
        else:
            self.experiment_dir = Path(experiment_dir)

        self.summary_path = self.experiment_dir / "dataset_b_experiment_summary.json"
        if not self.summary_path.exists():
            raise FileNotFoundError(f"Experiment summary not found at: {self.summary_path}")

        with open(self.summary_path, "r") as f:
            self.summary = json.load(f)

        self.feature_names = self.summary["feature_names"]
        self.feature_count = len(self.feature_names)

        # Artifact path
        self.model_path = self.experiment_dir / f"dataset_b_{model_name}_exp_v1.joblib"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.model_name = model_name

        # Scaler path
        self.scaler_path = self.experiment_dir / "dataset_b_scaler_exp_v1.joblib"
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None

        # Check if model requires scaling
        model_meta = self.summary["results"].get(self._get_display_name(model_name), {})
        self.requires_scaling = model_meta.get("requires_scaling", False)

    def _get_display_name(self, safe_name: str) -> str:
        name_map = {
            "extra_trees": "Extra Trees",
            "random_forest": "Random Forest",
            "support_vector_machine": "Support Vector Machine",
            "logistic_regression": "Logistic Regression",
            "xgboost": "XGBoost",
            "dummy_stratified": "Dummy (Stratified)",
            "dummy_most_frequent": "Dummy (Most Frequent)"
        }
        return name_map.get(safe_name, safe_name)

    def _validate_and_extract_features(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Validate input schema and verify no quarantined features are used."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])

        if isinstance(X, pd.DataFrame):
            # Strict quarantine check: quarantined features must not be in input columns
            quarantined_present = set(X.columns) & QUARANTINED_FEATURES
            if quarantined_present:
                raise ValueError(
                    f"Security/Causality Violation: Quarantined anti-causal features detected: {quarantined_present}"
                )

            # Check if all required feature names are present
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Input DataFrame is missing {len(missing)} required features: {missing}")

            # Reorder columns strictly to match training order
            X_array = X[self.feature_names].values.astype(np.float64)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != self.feature_count:
                raise ValueError(
                    f"Expected {self.feature_count} features, received shape {X.shape}"
                )
            X_array = X.astype(np.float64)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        if np.isnan(X_array).any():
            raise ValueError("Input features contain NaN values. NaN imputation is not supported at inference.")
        if np.isinf(X_array).any():
            raise ValueError("Input features contain infinite values.")

        return X_array

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Predict integer class labels (1: SA, 2: SD, 3: SRT, 4: SLT)."""
        X_array = self._validate_and_extract_features(X)
        if self.requires_scaling and self.scaler is not None:
            X_array = self.scaler.transform(X_array)

        if "xgboost" in self.model_name.lower():
            probs = self.model.predict_proba(X_array)
            classes_order = np.array([1, 2, 3, 4])
            return classes_order[np.argmax(probs, axis=1)]

        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Predict class probability distribution over [SA, SD, SRT, SLT]."""
        X_array = self._validate_and_extract_features(X)
        if self.requires_scaling and self.scaler is not None:
            X_array = self.scaler.transform(X_array)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_array)
            return probs
        else:
            # Deterministic one-hot if no proba
            preds = self.predict(X)
            probs = np.zeros((len(preds), 4))
            for i, p in enumerate(preds):
                probs[i, p - 1] = 1.0
            return probs

    def predict_with_labels(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict with human-readable labels, abbreviations, and confidence."""
        preds = self.predict(X)
        probs = self.predict_proba(X)

        results = []
        for i, pred_cls in enumerate(preds):
            cls_idx = int(pred_cls)
            conf = float(probs[i, cls_idx - 1]) if probs.shape[1] == 4 else 1.0
            results.append({
                "class_id": cls_idx,
                "label_name": LABEL_MAPPING.get(cls_idx, "Unknown"),
                "label_abbr": LABEL_ABBR.get(cls_idx, "UNK"),
                "confidence": conf,
                "probabilities": {
                    LABEL_ABBR[c]: float(probs[i, c - 1]) for c in [1, 2, 3, 4]
                }
            })
        return results

    def predict_event(self, df_event: pd.DataFrame) -> Dict[str, Any]:
        """
        Causally aggregate sample predictions over a single event sequence.
        Returns single event-level classification.
        """
        if not isinstance(df_event, pd.DataFrame):
            raise TypeError("predict_event requires a pd.DataFrame containing event samples.")

        sample_probs = self.predict_proba(df_event)
        mean_prob = np.mean(sample_probs, axis=0)
        pred_cls = int(np.argmax(mean_prob) + 1)
        conf = float(mean_prob[pred_cls - 1])

        return {
            "event_sample_count": len(df_event),
            "predicted_class_id": pred_cls,
            "label_name": LABEL_MAPPING.get(pred_cls, "Unknown"),
            "label_abbr": LABEL_ABBR.get(pred_cls, "UNK"),
            "confidence": conf,
            "event_probabilities": {
                LABEL_ABBR[c]: float(mean_prob[c - 1]) for c in [1, 2, 3, 4]
            }
        }

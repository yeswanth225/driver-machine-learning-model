"""
Physical Sensor ML Inference Adapter
===================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: ml/inference/physical_sensor_adapter.py

Responsibilities:
- Bridges incoming physical 20 Hz smartphone sensor streams with machine learning models.
- Uses CausalResampler20HzTo10Hz to prevent temporal / frequency domain distortion.
- Extracts features online using StreamingPipelineAFeatureExtractor.
- Preserves the existing dataset-based inference pathway completely unmodified.
- Maintains strict buffer warm-up awareness (t < 50 samples = WARMING_UP).
- Evaluates inference safely and deterministically without modifying production model weights.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from ml.inference.sampling_compatibility_adapter import CausalResampler20HzTo10Hz
from ml.inference.streaming_pipeline_a_features import (
    StreamingPipelineAFeatureExtractor,
    CANDIDATE_V2_PRUNED_FEATURES
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDIDATE_MODEL_PATH = os.path.join(REPO_ROOT, "models", "experiments", "pipeline_a", "pipeline_a_extratrees_candidate_v2.joblib")
CANDIDATE_CONFIG_PATH = os.path.join(REPO_ROOT, "models", "experiments", "pipeline_a", "pipeline_a_config_candidate_v2.json")

PROD_MODEL_PATH = os.path.join(REPO_ROOT, "models", "experiments", "dataset_c", "dataset_c_support_vector_machine_exp_v1.joblib")
PROD_SCALER_PATH = os.path.join(REPO_ROOT, "models", "experiments", "dataset_c", "dataset_c_scaler_exp_v1.joblib")
PROD_CONFIG_PATH = os.path.join(REPO_ROOT, "ml", "saved_models", "dataset_c_config_v1.0.0.json")


class PhysicalSensorInferenceAdapter:
    """
    Isolated adapter that executes inference on physical sensor observations
    via causal decimation and online streaming feature extraction.
    """

    def __init__(self, use_production_model: bool = False, filter_type: str = "2tap"):
        self.use_production_model = use_production_model
        self.filter_type = filter_type
        self.resampler = CausalResampler20HzTo10Hz(filter_type=filter_type)

        if self.use_production_model:
            # Production 47-feature RBF SVM
            with open(PROD_CONFIG_PATH, "r") as f:
                self.config = json.load(f)
            self.feature_names = self.config["feature_names"]
            self.model = joblib.load(PROD_MODEL_PATH)
            self.scaler = joblib.load(PROD_SCALER_PATH)
            self.threshold = 0.84
        else:
            # Candidate V2 ExtraTrees (110 features in config, 102 pruned)
            if os.path.exists(CANDIDATE_CONFIG_PATH) and os.path.exists(CANDIDATE_MODEL_PATH):
                with open(CANDIDATE_CONFIG_PATH, "r") as f:
                    self.config = json.load(f)
                self.feature_names = self.config["feature_names"]
                self.model = joblib.load(CANDIDATE_MODEL_PATH)
                self.scaler = None
                self.threshold = float(self.config.get("calibrated_threshold", 0.50))
            else:
                self.config = {}
                self.feature_names = CANDIDATE_V2_PRUNED_FEATURES
                self.model = None
                self.scaler = None
                self.threshold = 0.50

        self.extractor = StreamingPipelineAFeatureExtractor(feature_names=self.feature_names)
        self.reset()

    def reset(self):
        """Flushes buffers upon session or trip boundary."""
        self.resampler.reset()
        self.extractor.reset()
        self._processed_20hz_samples = 0
        self._emitted_10hz_samples = 0

    def process_raw_20hz_sample(self, raw_sample: Dict[str, float]) -> Dict[str, Any]:
        """
        Ingests 1 raw observation at 20 Hz.
        Executes causal 2:1 resampling and inference on even steps.
        """
        self._processed_20hz_samples += 1
        resampled_10hz = self.resampler.process_sample(raw_sample)

        # On odd samples, resampler returns None
        if resampled_10hz is None:
            return {
                "status": "WAITING_DECIMATION",
                "sample_20hz_count": self._processed_20hz_samples,
                "is_inference_emitted": False,
            }

        self._emitted_10hz_samples += 1
        features = self.extractor.process_sample(resampled_10hz)

        # Warm-up check (needs 50 samples of 10 Hz = 5.0 seconds)
        if not self.extractor.is_warmed_up:
            return {
                "status": "WARMING_UP",
                "sample_20hz_count": self._processed_20hz_samples,
                "sample_10hz_count": self._emitted_10hz_samples,
                "is_inference_emitted": False,
                "warmup_progress_pct": round((self.extractor.sample_count / 50.0) * 100.0, 1),
                "message": f"Buffer warming up ({self.extractor.sample_count}/50 samples).",
            }

        # Model inference
        if self.model is None:
            return {
                "status": "MODEL_UNAVAILABLE",
                "sample_20hz_count": self._processed_20hz_samples,
                "is_inference_emitted": False,
            }

        feature_arr = np.array([[features[f] for f in self.feature_names]], dtype=np.float64)
        if self.scaler is not None:
            feature_arr = self.scaler.transform(feature_arr)

        prob_aggressive = float(self.model.predict_proba(feature_arr)[0, 1])
        prediction = 1 if prob_aggressive >= self.threshold else 0
        style_label = "Aggressive Driving" if prediction == 1 else "Normal Driving"

        return {
            "status": "ACTIVE_INFERENCE",
            "proven_compatibility": False,
            "disclaimer": "Physical-to-model inference is unproven on physical vehicular hardware. Retained for informational research telemetry only.",
            "sample_20hz_count": self._processed_20hz_samples,
            "sample_10hz_count": self._emitted_10hz_samples,
            "is_inference_emitted": True,
            "timestamp": resampled_10hz.get("timestamp", 0.0),
            "driving_style_id": prediction,
            "driving_style": style_label,
            "confidence": prob_aggressive if prediction == 1 else (1.0 - prob_aggressive),
            "probability_aggressive": prob_aggressive,
            "threshold": self.threshold,
        }

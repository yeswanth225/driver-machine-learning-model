"""
Unit & Integration Tests: Model Training, Persistence & Production Inference
=============================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Stage: Stage 4B - Model Testing
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier
from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR_B = REPO_ROOT / "models" / "experiments" / "dataset_b"
EXP_DIR_C = REPO_ROOT / "models" / "experiments" / "dataset_c"
DATA_DIR = REPO_ROOT / "dataset" / "processed"


class TestDatasetBModelArtifactsAndLoading:
    """Validate persistence, integrity, and schemas of Dataset B model artifacts."""

    def test_experiment_summary_exists_and_valid(self):
        summary_path = EXP_DIR_B / "dataset_b_experiment_summary.json"
        assert summary_path.exists(), f"Summary file missing at {summary_path}"
        with open(summary_path, "r") as f:
            summary = json.load(f)
        assert summary["pipeline"] == "Pipeline B (Dataset B)"
        assert summary["selected_best_model"] == "Extra Trees"
        assert len(summary["feature_names"]) == 56
        assert len(summary["results"]) >= 5

    def test_lodo_summary_exists_and_valid(self):
        lodo_path = EXP_DIR_B / "dataset_b_lodo_summary.json"
        assert lodo_path.exists(), f"LODO summary missing at {lodo_path}"
        with open(lodo_path, "r") as f:
            lodo = json.load(f)
        assert "models_evaluated" in lodo
        assert "Extra Trees" in lodo["models_evaluated"]
        assert len(lodo["models_evaluated"]["Extra Trees"]["folds"]) == 3

    def test_all_model_artifacts_exist(self):
        expected_models = [
            "extra_trees",
            "random_forest",
            "support_vector_machine",
            "logistic_regression",
            "xgboost"
        ]
        for m in expected_models:
            p = EXP_DIR_B / f"dataset_b_{m}_exp_v1.joblib"
            assert p.exists(), f"Missing artifact: {p}"
            loaded = joblib.load(p)
            assert loaded is not None

    def test_scaler_artifact_exists(self):
        scaler_p = EXP_DIR_B / "dataset_b_scaler_exp_v1.joblib"
        assert scaler_p.exists()
        scaler = joblib.load(scaler_p)
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == 56


class TestDatasetBManeuverClassifierInference:
    """Validate inference wrapper functionality, edge cases, and safety checks."""

    @pytest.fixture
    def clf(self):
        return DatasetBManeuverClassifier(model_name="extra_trees")

    @pytest.fixture
    def test_samples(self):
        df_test = pd.read_csv(DATA_DIR / "dataset_b_test.csv")
        # In real-time streaming, quarantined future features are absent
        streaming_df = df_test.drop(columns=[
            "event_duration_samples",
            "event_progress_ratio",
            "event_sample_idx"
        ])
        return streaming_df

    def test_initialization(self, clf):
        assert clf.feature_count == 56
        assert clf.model is not None

    def test_predict_dataframe(self, clf, test_samples):
        preds = clf.predict(test_samples)
        assert len(preds) == len(test_samples)
        assert set(preds).issubset({1, 2, 3, 4})

    def test_predict_proba(self, clf, test_samples):
        probs = clf.predict_proba(test_samples)
        assert probs.shape == (len(test_samples), 4)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    def test_predict_with_labels(self, clf, test_samples):
        results = clf.predict_with_labels(test_samples.head(5))
        assert len(results) == 5
        for r in results:
            assert "class_id" in r
            assert r["class_id"] in [1, 2, 3, 4]
            assert r["label_name"] in [
                "Sudden Acceleration", "Sudden Deceleration",
                "Sudden Right Turn", "Sudden Left Turn"
            ]
            assert r["label_abbr"] in ["SA", "SD", "SRT", "SLT"]
            assert 0.0 <= r["confidence"] <= 1.0

    def test_predict_event(self, clf, test_samples):
        first_event_id = test_samples["event_id"].iloc[0]
        event_df = test_samples[test_samples["event_id"] == first_event_id]
        res = clf.predict_event(event_df)
        assert res["event_sample_count"] == len(event_df)
        assert res["predicted_class_id"] in [1, 2, 3, 4]
        assert res["label_name"] in [
            "Sudden Acceleration", "Sudden Deceleration",
            "Sudden Right Turn", "Sudden Left Turn"
        ]

    def test_quarantined_feature_rejection(self, clf, test_samples):
        """Strict test: passing quarantined features must raise a ValueError!"""
        leaky_df = test_samples.copy()
        leaky_df["event_duration_samples"] = 10
        with pytest.raises(ValueError, match="Security/Causality Violation"):
            clf.predict(leaky_df)

    def test_missing_feature_rejection(self, clf, test_samples):
        incomplete_df = test_samples[clf.feature_names[:10]]
        with pytest.raises(ValueError, match="missing .* required features"):
            clf.predict(incomplete_df)

    def test_nan_feature_rejection(self, clf, test_samples):
        nan_df = test_samples[clf.feature_names].copy()
        nan_df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contain NaN values"):
            clf.predict(nan_df)

    def test_prediction_reproducibility(self, clf, test_samples):
        preds1 = clf.predict(test_samples.head(20))
        preds2 = clf.predict(test_samples.head(20))
        np.testing.assert_array_equal(preds1, preds2)


class TestDatasetCExperimentArtifactsAndLoading:
    """Validate persistence and integrity of Dataset C experimental models."""

    def test_experiment_summary_exists_and_valid(self):
        summary_path = EXP_DIR_C / "dataset_c_experiment_summary.json"
        assert summary_path.exists()
        with open(summary_path, "r") as f:
            summary = json.load(f)
        assert summary["pipeline"] == "Pipeline A (Dataset C)"
        assert summary["selected_best_model"] == "Support Vector Machine"
        assert len(summary["feature_names"]) == 47

    def test_all_model_artifacts_exist(self):
        expected_models = [
            "support_vector_machine",
            "random_forest",
            "xgboost",
            "logistic_regression",
            "multi-layer_perceptron"
        ]
        for m in expected_models:
            p = EXP_DIR_C / f"dataset_c_{m}_exp_v1.joblib"
            assert p.exists(), f"Missing artifact: {p}"
            loaded = joblib.load(p)
            assert loaded is not None

    def test_scaler_artifact_exists(self):
        scaler_p = EXP_DIR_C / "dataset_c_scaler_exp_v1.joblib"
        assert scaler_p.exists()
        scaler = joblib.load(scaler_p)
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == 47


class TestCrossPipelineDecoupling:
    """Ensure strict isolation between Dataset B and Dataset C."""

    def test_feature_count_difference(self):
        with open(EXP_DIR_B / "dataset_b_experiment_summary.json") as fb:
            sb = json.load(fb)
        with open(EXP_DIR_C / "dataset_c_experiment_summary.json") as fc:
            sc = json.load(fc)
        assert len(sb["feature_names"]) == 56
        assert len(sc["feature_names"]) == 47

    def test_dataset_b_classifier_rejects_dataset_c_data(self):
        clf_b = DatasetBManeuverClassifier()
        df_c = pd.read_csv(DATA_DIR / "dataset_c_test.csv")
        with pytest.raises(ValueError):
            clf_b.predict(df_c)

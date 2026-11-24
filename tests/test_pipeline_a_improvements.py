"""
Tests: tests/test_pipeline_a_improvements.py
Comprehensive regression & verification test suite for Pipeline A improvements,
feature contracts, scaling consistency, session isolation, and production freezing.
"""

import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent

# Frozen Production Hashes
FROZEN_MODEL_HASH = "a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc"
FROZEN_SCALER_HASH = "d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5"

PROD_MODEL_PATH = REPO_ROOT / "models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib"
PROD_SCALER_PATH = REPO_ROOT / "models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib"
PROD_SUMMARY_PATH = REPO_ROOT / "models/experiments/dataset_c/dataset_c_experiment_summary.json"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestPipelineAImprovementsAndIntegrity:
    """Rigorous verification suite for Pipeline A improvements and governance."""

    def test_feature_count_and_contract(self):
        """1. Verify that production Dataset C strictly enforces the 47-feature contract."""
        train_path = REPO_ROOT / "dataset/processed/dataset_c_train.csv"
        assert train_path.exists(), "dataset_c_train.csv must exist."
        
        df_train = pd.read_csv(train_path, nrows=10)
        feature_cols = [c for c in df_train.columns if c not in ["label", "session", "split"]]
        
        assert len(feature_cols) == 47, f"Expected 47 features, got {len(feature_cols)}."
        
        # Verify against production summary feature names
        with open(PROD_SUMMARY_PATH, "r") as f:
            summary = json.load(f)
        
        expected_features = summary["feature_names"]
        assert len(expected_features) == 47
        assert feature_cols == expected_features, "Feature order or names mismatch with production contract."

    def test_feature_ordering_consistency(self):
        """2. Verify that train, val, and test feature columns are identically ordered."""
        for split_name in ["train", "val", "test"]:
            path = REPO_ROOT / f"dataset/processed/dataset_c_{split_name}.csv"
            assert path.exists()
            df = pd.read_csv(path, nrows=5)
            cols = [c for c in df.columns if c not in ["label", "session", "split"]]
            assert len(cols) == 47
            assert cols[0] == "Acc X"
            assert cols[1] == "Acc Y"
            assert cols[2] == "Acc Z"
            assert cols[3] == "gyro_x"
            assert cols[-1] == "gyro_z_max_3s"

    def test_missing_values_and_completeness(self):
        """3. Verify zero missing or non-finite values in processed Dataset C."""
        all_path = REPO_ROOT / "dataset/processed/dataset_c_processed_all.csv"
        assert all_path.exists()
        
        df_all = pd.read_csv(all_path)
        assert len(df_all) == 13840, f"Expected 13,840 records, got {len(df_all)}."
        
        null_counts = df_all.isnull().sum().sum()
        assert null_counts == 0, f"Found {null_counts} null values in dataset_c_processed_all.csv."
        
        # Check non-finite in numeric columns
        numeric_cols = df_all.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df_all[numeric_cols].values).sum()
        assert inf_counts == 0, f"Found {inf_counts} infinite values."

    def test_scaling_consistency(self):
        """4. Verify that production scaler transforms 47 features with consistent scaling."""
        assert PROD_SCALER_PATH.exists()
        scaler = joblib.load(PROD_SCALER_PATH)
        
        # Test batch transform
        dummy_input = np.random.randn(20, 47)
        transformed = scaler.transform(dummy_input)
        assert transformed.shape == (20, 47)
        assert not np.isnan(transformed).any()
        assert not np.isinf(transformed).any()
        
        # Check scaler attributes
        assert scaler.n_features_in_ == 47
        assert len(scaler.mean_) == 47
        assert len(scaler.scale_) == 47

    def test_prediction_output_and_threshold(self):
        """5. Verify that production model outputs valid probabilities and thresholded predictions."""
        assert PROD_MODEL_PATH.exists()
        model = joblib.load(PROD_MODEL_PATH)
        scaler = joblib.load(PROD_SCALER_PATH)
        
        df_sample = pd.read_csv(REPO_ROOT / "dataset/processed/dataset_c_test.csv", nrows=15)
        feature_cols = [c for c in df_sample.columns if c not in ["label", "session", "split"]]
        X_sample = scaler.transform(df_sample[feature_cols].values)
        
        probs = model.predict_proba(X_sample)
        assert probs.shape == (15, 2), "Probabilities should have shape (N, 2)."
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]."
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(15), rtol=1e-5)
        
        # Decision threshold = 0.84
        preds_tau = (probs[:, 1] >= 0.84).astype(int)
        assert set(preds_tau).issubset({0, 1}), "Predictions must be binary {0, 1}."

    def test_production_model_artifact_loading_and_hashes(self):
        """6. Cryptographic verification: production model and scaler must match frozen SHA-256 hashes."""
        assert PROD_MODEL_PATH.exists(), "Production model file missing."
        assert PROD_SCALER_PATH.exists(), "Production scaler file missing."
        
        actual_model_hash = compute_file_sha256(PROD_MODEL_PATH)
        actual_scaler_hash = compute_file_sha256(PROD_SCALER_PATH)
        
        assert actual_model_hash == FROZEN_MODEL_HASH, (
            f"CRITICAL: Production model hash drift detected! "
            f"Expected {FROZEN_MODEL_HASH}, got {actual_model_hash}"
        )
        assert actual_scaler_hash == FROZEN_SCALER_HASH, (
            f"CRITICAL: Production scaler hash drift detected! "
            f"Expected {FROZEN_SCALER_HASH}, got {actual_scaler_hash}"
        )

    def test_session_separation(self):
        """7. Verify session partition boundaries and counts."""
        df_all = pd.read_csv(REPO_ROOT / "dataset/processed/dataset_c_processed_all.csv")
        s1 = df_all[df_all["session"] == "session_1"]
        s2 = df_all[df_all["session"] == "session_2"]
        
        assert len(s1) == 7399, f"Session 1 should have 7,399 samples, got {len(s1)}."
        assert len(s2) == 6441, f"Session 2 should have 6,441 samples, got {len(s2)}."
        assert len(s1) + len(s2) == 13840
        
        # Verify class counts
        assert (s1["label"] == 0).sum() == 3563
        assert (s1["label"] == 1).sum() == 3836
        assert (s2["label"] == 0).sum() == 2137
        assert (s2["label"] == 1).sum() == 4304

    def test_temporal_leakage_prevention(self):
        """8. Verify that enhanced feature engineering uses backward-looking causal operations only."""
        script_path = REPO_ROOT / "scripts/engineer_pipeline_a_features.py"
        assert script_path.exists()
        
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
            
        # Ensure no forward-looking shifts or centered windows
        assert "shift(-" not in code, "Forward shift detected in feature engineering script."
        assert "center=True" not in code, "Centered rolling window detected in feature engineering script."
        assert "ewm(alpha=" in code, "Causal exponential moving average should be used."

    def test_research_candidate_isolation(self):
        """9. Verify that all research candidate artifacts are stored outside the production directory."""
        prod_dir = REPO_ROOT / "models/experiments/dataset_c"
        research_dir = REPO_ROOT / "models/experiments/pipeline_a"
        
        assert research_dir.exists(), "Research candidate directory must exist."
        
        # Check that candidate v2 / 47feat joblib files are in research dir, not prod dir
        prod_files = [f.name for f in prod_dir.glob("*.joblib")]
        research_files = [f.name for f in research_dir.glob("*.joblib")]
        
        assert "dataset_c_support_vector_machine_exp_v1.joblib" in prod_files
        assert "pipeline_a_extratrees_47feat_candidate.joblib" not in prod_files
        assert "pipeline_a_extratrees_candidate_v2.joblib" not in prod_files
        assert len(research_files) >= 1, "Expected candidate joblib files in research directory."

    def test_safety_agent_inference_compatibility(self):
        """10. Verify that production Pipeline A features are fully compatible with SafetyAgent."""
        from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier
        from ml.inference.dataset_b_classifier import DatasetBManeuverClassifier
        from ml.agent.safety_agent import IntelligentSafetyAgent, SafetyState
        
        clf_c = DatasetCBinaryClassifier(
            model_dir=str(REPO_ROOT / "models" / "experiments" / "dataset_c"),
            model_path=str(PROD_MODEL_PATH),
            scaler_path=str(PROD_SCALER_PATH),
            config_path=str(PROD_SUMMARY_PATH),
            decision_threshold=0.84
        )
        clf_b = DatasetBManeuverClassifier(model_name="extra_trees")
        agent = IntelligentSafetyAgent(classifier_c=clf_c, classifier_b=clf_b)
        
        test_c_df = pd.read_csv(REPO_ROOT / "dataset/processed/dataset_c_test.csv")[clf_c.feature_names]
        test_b_df = pd.read_csv(REPO_ROOT / "dataset/processed/dataset_b_processed_all.csv")[clf_b.feature_names]
        
        assessment = agent.process_features(
            timestamp=1.0,
            features_c=test_c_df.iloc[[0]],
            features_b=test_b_df.iloc[[0]]
        )
        
        assert assessment.is_valid is True
        assert assessment.safety_state in [SafetyState.NORMAL, SafetyState.CAUTION, SafetyState.WARNING, SafetyState.HIGH_RISK]
        assert 0.0 <= assessment.risk_score <= 100.0
        assert assessment.driving_style in ["Normal Driving", "Aggressive Driving"]

    def test_causal_feature_all_102_features_audit(self):
        """11. Verify that all 102 Candidate V2 features are strictly causal (invariant to future data)."""
        from scripts.engineer_pipeline_a_features import compute_causal_features
        
        # Create a synthetic 50-sample telemetry segment
        np.random.seed(42)
        base_df = pd.DataFrame({
            'Acc X': np.random.randn(50) * 0.1,
            'Acc Y': np.random.randn(50) * 0.1,
            'Acc Z': 1.0 + np.random.randn(50) * 0.05,
            'gyro_x': np.random.randn(50) * 0.02,
            'gyro_y': np.random.randn(50) * 0.02,
            'gyro_z': np.random.randn(50) * 0.02
        })
        
        feat_orig = compute_causal_features(base_df)
        
        # Modify future samples (indices 30 to 49) with extreme values
        corrupted_df = base_df.copy()
        corrupted_df.loc[30:, 'Acc X'] = 999.0
        corrupted_df.loc[30:, 'gyro_z'] = 999.0
        
        feat_corrupted = compute_causal_features(corrupted_df)
        
        # Verify that for all past samples (indices 0 to 29), EVERY feature is bit-for-bit identical
        numeric_cols = [c for c in feat_orig.columns if c in feat_corrupted.columns]
        for col in numeric_cols:
            np.testing.assert_allclose(
                feat_orig.loc[:29, col].values,
                feat_corrupted.loc[:29, col].values,
                rtol=1e-5, atol=1e-5,
                err_msg=f"Feature '{col}' used future observations! Values at t < 30 changed after future modification."
            )

    def test_no_backward_filling_or_future_lookahead(self):
        """12. Audit code for absence of backward filling or forward shifts."""
        script_path = REPO_ROOT / "scripts/engineer_pipeline_a_features.py"
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        banned_patterns = ["bfill", "shift(-", "center=True", "fillna(method='bfill')"]
        for pattern in banned_patterns:
            assert pattern not in code, f"Banned forward-looking pattern '{pattern}' detected in feature pipeline."

    def test_session_boundary_buffer_flushing(self):
        """13. Verify that rolling window statistics do not bleed across session boundaries."""
        df_enh_path = REPO_ROOT / "dataset/experiments/pipeline_a/dataset_c_enhanced_all.csv"
        assert df_enh_path.exists()
        
        df_enh = pd.read_csv(df_enh_path)
        s2_start_idx = df_enh[df_enh["session"] == "session_2"].index.min()
        
        # At start of Session 2, jerk and rolling std should reset cleanly (min_periods=1)
        assert df_enh.loc[s2_start_idx, "acc_jerk"] == 0.0, "Session 2 initial jerk should be 0.0 (clean buffer reset)."
        assert df_enh.loc[s2_start_idx, "gyro_jerk"] == 0.0, "Session 2 initial gyro jerk should be 0.0 (clean buffer reset)."

    def test_recalculated_metrics_match_reports(self):
        """14. Recalculate S1 -> S2 ExtraTrees metrics and verify exact match with reports."""
        df_enh_path = REPO_ROOT / "dataset/experiments/pipeline_a/dataset_c_enhanced_all.csv"
        df_enh = pd.read_csv(df_enh_path)
        
        s1 = df_enh[df_enh["session"] == "session_1"].reset_index(drop=True)
        s2 = df_enh[df_enh["session"] == "session_2"].reset_index(drop=True)
        
        e_feats = [c for c in df_enh.columns if c not in ["label", "split", "session"]]
        high_jitter = ["gyro_mag_max_3s", "gyro_mag_max_2s", "acc_mag_std_3s", "acc_mag_std_2s"]
        v2_feats = [c for c in e_feats if c not in high_jitter and not ("min" in c and "acc_planar" in c)]
        
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        et = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight="balanced", random_state=42, n_jobs=-1)
        et.fit(s1[v2_feats].values, s1["label"].values)
        pred_s2 = et.predict(s2[v2_feats].values)
        
        recalc_acc = accuracy_score(s2["label"].values, pred_s2) * 100
        recalc_f1 = f1_score(s2["label"].values, pred_s2, average="macro") * 100
        
        # Read from cross_session_evaluation.csv
        cs_csv = REPO_ROOT / "reports/pipeline_a/improvement/cross_session_evaluation.csv"
        df_cs = pd.read_csv(cs_csv)
        row = df_cs[(df_cs["Model"] == "Extra_Trees") & (df_cs["Feature_Set"] == "Candidate_V2_102") & (df_cs["Direction"] == "Forward (S1->S2)")]
        
        assert len(row) == 1
        reported_acc = row["Accuracy"].values[0]
        reported_f1 = row["Macro_F1"].values[0]
        
        assert abs(recalc_acc - reported_acc) < 0.5, f"Recalculated acc {recalc_acc:.2f}% differs from reported {reported_acc:.2f}%"
        assert abs(recalc_f1 - reported_f1) < 0.5, f"Recalculated F1 {recalc_f1:.2f}% differs from reported {reported_f1:.2f}%"

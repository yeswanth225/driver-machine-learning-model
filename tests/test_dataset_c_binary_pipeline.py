"""
Automated Unit and Integration Tests: Dataset C Binary Driving Behaviour Pipeline
===============================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from ml.preprocessing.pipeline_dataset_c import (
    load_raw_dataset, remove_duplicates_and_tag_sessions,
    engineer_kinematic_features, split_session_chronological,
    LABEL_MAPPING, RAW_DATA_PATH, PROCESSED_DIR
)
from ml.inference.dataset_c_classifier import DatasetCBinaryClassifier

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "saved_models")


class TestDatasetCLoadingAndValidation:
    """Tests for raw dataset loading and structural integrity."""
    
    def test_raw_file_exists_and_loads(self):
        """Verify the raw dataset exists and loads with correct columns."""
        df = load_raw_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 14249
        expected_cols = {'Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'label'}
        assert expected_cols.issubset(set(df.columns))

    def test_label_values_valid(self):
        """Verify labels are strictly binary (0 and 1)."""
        df = load_raw_dataset()
        unique_labels = set(df['label'].unique())
        assert unique_labels == {0, 1}

    def test_missing_values_absent(self):
        """Verify raw dataset has zero missing / null values."""
        df = load_raw_dataset()
        assert df.isnull().sum().sum() == 0


class TestDatasetCPreprocessing:
    """Tests for deduplication, session splitting, feature engineering, and guard bands."""
    
    def test_duplicate_removal_and_session_tagging(self):
        """Verify exact duplicate removal and session breakdown."""
        df = load_raw_dataset()
        df_s1, df_s2, meta = remove_duplicates_and_tag_sessions(df)
        
        assert meta["duplicates_removed_total"] == 9
        assert meta["clean_total_rows"] == 14240
        assert len(df_s1) == 7599
        assert len(df_s2) == 6641
        assert (len(df_s1) + len(df_s2)) == 14240
        assert df_s1['session'].unique()[0] == 'session_1'
        assert df_s2['session'].unique()[0] == 'session_2'

    def test_kinematic_feature_engineering(self):
        """Verify physical invariants and rolling statistics computation."""
        df = load_raw_dataset()
        df_s1, _, _ = remove_duplicates_and_tag_sessions(df)
        df_feat = engineer_kinematic_features(df_s1)
        
        # Check derived physical channels
        assert 'acc_mag' in df_feat.columns
        assert 'gyro_mag' in df_feat.columns
        assert 'acc_horiz' in df_feat.columns
        assert 'acc_jerk' in df_feat.columns
        assert 'gyro_jerk' in df_feat.columns
        
        # Check causal rolling stats
        for prefix in ['1s', '2s', '3s']:
            assert f'acc_mag_mean_{prefix}' in df_feat.columns
            assert f'acc_mag_std_{prefix}' in df_feat.columns
            assert f'gyro_mag_mean_{prefix}' in df_feat.columns
            assert f'gyro_z_std_{prefix}' in df_feat.columns
            
        # Verify no NaNs generated
        assert df_feat.isnull().sum().sum() == 0

    def test_chronological_split_with_guard_bands(self):
        """Verify chronological split boundaries and guard-band buffer dropping."""
        df = load_raw_dataset()
        df_s1, _, _ = remove_duplicates_and_tag_sessions(df)
        df_feat = engineer_kinematic_features(df_s1)
        
        train, val, test = split_session_chronological(df_feat, train_pct=0.70, val_pct=0.15, guard_samples=100)
        
        # Train should end before val starts by guard_samples
        assert train['split'].iloc[0] == 'train'
        assert val['split'].iloc[0] == 'val'
        assert test['split'].iloc[0] == 'test'
        
        # Exact row count check: total rows - (2 * guard_samples)
        assert len(train) + len(val) + len(test) == len(df_feat) - 200


class TestModelArtifactsAndLoading:
    """Tests for model persistence, loading, and label mapping."""
    
    def test_saved_model_and_scaler_exist(self):
        """Verify model, scaler, and metadata artifacts exist on disk."""
        model_file = os.path.join(SAVED_MODELS_DIR, "dataset_c_binary_model_v1.0.0.joblib")
        scaler_file = os.path.join(SAVED_MODELS_DIR, "dataset_c_scaler_v1.0.0.joblib")
        config_file = os.path.join(SAVED_MODELS_DIR, "dataset_c_config_v1.0.0.json")
        metrics_file = os.path.join(SAVED_MODELS_DIR, "dataset_c_metrics_v1.0.0.json")
        
        assert os.path.exists(model_file)
        assert os.path.exists(scaler_file)
        assert os.path.exists(config_file)
        assert os.path.exists(metrics_file)

    def test_label_mapping(self):
        """Verify label mapping matches domain standards."""
        assert LABEL_MAPPING[0] == "Normal Driving"
        assert LABEL_MAPPING[1] == "Aggressive Driving"

    def test_classifier_wrapper_initialization(self):
        """Verify DatasetCBinaryClassifier loads correctly."""
        clf = DatasetCBinaryClassifier(model_version="1.0.0")
        assert clf.feature_names is not None
        assert len(clf.feature_names) == 47
        assert clf.label_mapping[0] == "Normal Driving"
        assert clf.label_mapping[1] == "Aggressive Driving"


class TestInferenceAndValidation:
    """Tests for prediction outputs, probability bounds, error handling, and reproducibility."""
    
    @pytest.fixture
    def test_data(self):
        test_path = os.path.join(PROCESSED_DIR, "dataset_c_test.csv")
        return pd.read_csv(test_path)

    def test_prediction_output_shape_and_values(self, test_data):
        """Verify predictions have matching length and only contain 0 and 1."""
        clf = DatasetCBinaryClassifier(model_version="1.0.0")
        preds = clf.predict(test_data)
        
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(test_data)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_prediction_probabilities(self, test_data):
        """Verify predicted probabilities sum to 1.0 and lie in [0, 1]."""
        clf = DatasetCBinaryClassifier(model_version="1.0.0")
        probs = clf.predict_proba(test_data)
        
        assert probs.shape == (len(test_data), 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_human_readable_labels(self, test_data):
        """Verify text label generation."""
        clf = DatasetCBinaryClassifier(model_version="1.0.0")
        sample = test_data.head(10)
        labels = clf.predict_label(sample)
        
        assert len(labels) == 10
        assert all(l in ["Normal Driving", "Aggressive Driving"] for l in labels)

    def test_missing_or_invalid_input_error(self, test_data):
        """Verify classifier raises ValueError on missing features or invalid input."""
        clf = DatasetCBinaryClassifier(model_version="1.0.0")
        
        # Missing feature column
        bad_df = test_data.drop(columns=['Acc X'])
        with pytest.raises(ValueError, match="missing required features"):
            clf.predict(bad_df)
            
        # NaN in input
        nan_df = test_data.head(5).copy()
        nan_df.iloc[0, nan_df.columns.get_loc('Acc X')] = np.nan
        with pytest.raises(ValueError, match="contain NaN values"):
            clf.predict(nan_df)

    def test_prediction_reproducibility(self, test_data):
        """Verify predictions from loaded model exactly match recorded test performance metrics."""
        clf = DatasetCBinaryClassifier(model_version="1.0.0")
        preds = clf.predict(test_data)
        
        metrics_file = os.path.join(SAVED_MODELS_DIR, "dataset_c_metrics_v1.0.0.json")
        with open(metrics_file, "r") as f:
            recorded_metrics = json.load(f)["Support Vector Machine"]["test"]
            
        y_true = test_data['label'].values
        accuracy = float(np.mean(preds == y_true))
        recorded_acc = recorded_metrics["accuracy"]
        
        # Must match exactly to 4 decimal places
        np.testing.assert_almost_equal(accuracy, recorded_acc, decimal=4)

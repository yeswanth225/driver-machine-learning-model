"""
Automated Unit and Integration Tests: Dataset B Preprocessing Pipeline
=====================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Target: ml/preprocessing/pipeline_dataset_b.py
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.preprocessing.pipeline_dataset_b import (
    load_raw_dataset, tag_event_trials, engineer_kinematic_features,
    extract_event_level_features, split_dataset_b_stratified_group,
    LABEL_MAPPING, LABEL_ABBREVIATIONS, EXPECTED_COLUMNS,
    G_TO_MS2, DEG_TO_RAD, RAW_DATA_PATH, PROCESSED_DIR
)


class TestDatasetBLoadingAndValidation:
    """Tests for raw Dataset B loading, schema validation, and integrity."""

    def test_raw_file_exists_and_loads(self):
        """Verify sensor_raw.csv exists and loads with exact 1,114 rows and 12 columns."""
        df = load_raw_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1114
        assert set(EXPECTED_COLUMNS).issubset(set(df.columns))

    def test_missing_values_absent(self):
        """Verify raw Dataset B has zero missing / null values."""
        df = load_raw_dataset()
        assert df.isnull().sum().sum() == 0

    def test_class_labels_valid(self):
        """Verify ground-truth classes are strictly {1, 2, 3, 4}."""
        df = load_raw_dataset()
        unique_classes = set(df["Class"].unique())
        assert unique_classes == {1, 2, 3, 4}

    def test_driver_ids_valid(self):
        """Verify driver identifiers match the three documented experimental subjects."""
        df = load_raw_dataset()
        unique_drivers = set(df["DriverID"].unique())
        assert unique_drivers == {"Driver-1", "Driver-2", "Driver-3"}

    def test_invalid_file_handling(self, tmp_path):
        """Verify ValueError is raised if raw file is missing columns or corrupted."""
        bad_file = tmp_path / "bad_sensor.csv"
        bad_df = pd.DataFrame({"AccX": [1.0], "AccY": [0.0]})
        bad_df.to_csv(bad_file, index=False)

        with pytest.raises(ValueError, match="missing expected columns"):
            load_raw_dataset(bad_file)


class TestDatasetBEventTaggingAndSemantics:
    """Tests for 109 isolated event trial discovery, semantics, and durations."""

    def test_exact_109_events_tagged(self):
        """Verify exactly 109 unique event trials are discovered."""
        df = load_raw_dataset()
        df_tagged, meta = tag_event_trials(df)
        assert meta["total_events"] == 109
        assert df_tagged["event_id"].nunique() == 109

    def test_human_readable_labels(self):
        """Verify correct mapping of classes to maneuver names and abbreviations."""
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)

        assert LABEL_MAPPING[1] == "Sudden Acceleration"
        assert LABEL_MAPPING[2] == "Sudden Deceleration"
        assert LABEL_MAPPING[3] == "Sudden Right Turn"
        assert LABEL_MAPPING[4] == "Sudden Left Turn"

        sample_sa = df_tagged[df_tagged["Class"] == 1].iloc[0]
        assert sample_sa["label_name"] == "Sudden Acceleration"
        assert sample_sa["label_abbr"] == "SA"

    def test_event_duration_bounds(self):
        """Verify event durations in rows fall within the verified 7-14 sample interval."""
        df = load_raw_dataset()
        df_tagged, meta = tag_event_trials(df)
        assert meta["min_samples_per_event"] >= 7
        assert meta["max_samples_per_event"] <= 14

    def test_temporal_progression_monotonic(self):
        """Verify event_sample_idx increments monotonically from 0 to N-1 per event."""
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)
        for _, grp in df_tagged.groupby("event_id"):
            indices = list(grp["event_sample_idx"])
            assert indices == list(range(len(grp)))
            assert grp["event_progress_ratio"].iloc[0] > 0.0
            assert np.isclose(grp["event_progress_ratio"].iloc[-1], 1.0)


class TestKinematicFeatureEngineering:
    """Tests for unit conversions, rotation invariants, temporal jerk, and rolling features."""

    @pytest.fixture
    def tagged_data(self):
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)
        return df_tagged

    def test_unit_conversions(self, tagged_data):
        """Verify mathematical accuracy of g -> m/s² and deg/s -> rad/s conversions."""
        df_feat = engineer_kinematic_features(tagged_data)

        # Accelerometer g to m/s²
        np.testing.assert_allclose(df_feat["acc_x_ms2"], df_feat["AccX"] * G_TO_MS2, rtol=1e-5)
        np.testing.assert_allclose(df_feat["acc_y_ms2"], df_feat["AccY"] * G_TO_MS2, rtol=1e-5)
        np.testing.assert_allclose(df_feat["acc_z_ms2"], df_feat["AccZ"] * G_TO_MS2, rtol=1e-5)

        # Gyroscope deg/s to rad/s
        np.testing.assert_allclose(df_feat["gyro_x_rads"], df_feat["GyroX"] * DEG_TO_RAD, rtol=1e-5)
        np.testing.assert_allclose(df_feat["gyro_y_rads"], df_feat["GyroY"] * DEG_TO_RAD, rtol=1e-5)
        np.testing.assert_allclose(df_feat["gyro_z_rads"], df_feat["GyroZ"] * DEG_TO_RAD, rtol=1e-5)

    def test_rotation_invariants(self, tagged_data):
        """Verify rotation invariant vector norms are strictly non-negative."""
        df_feat = engineer_kinematic_features(tagged_data)
        assert np.all(df_feat["acc_mag_ms2"] > 0.0)
        assert np.all(df_feat["gyro_mag_rads"] >= 0.0)
        assert np.all(df_feat["acc_horiz_ms2"] >= 0.0)

    def test_jerk_backward_causality(self, tagged_data):
        """Verify backward jerk derivatives start at 0.0 at event onset and contain no future leak."""
        df_feat = engineer_kinematic_features(tagged_data)
        for _, grp in df_feat.groupby("event_id"):
            assert grp["acc_jerk_ms2"].iloc[0] == 0.0
            assert grp["gyro_jerk_rads"].iloc[0] == 0.0

    def test_causal_rolling_features_exist_and_no_nan(self, tagged_data):
        """Verify rolling features for 3s and 5s exist and contain zero NaNs."""
        df_feat = engineer_kinematic_features(tagged_data)
        for w in ["3s", "5s"]:
            assert f"acc_mag_mean_{w}" in df_feat.columns
            assert f"acc_mag_std_{w}" in df_feat.columns
            assert f"acc_mag_rms_{w}" in df_feat.columns
            assert f"gyro_mag_mean_{w}" in df_feat.columns
            assert f"gyro_z_mean_{w}" in df_feat.columns
            assert f"acc_x_mean_{w}" in df_feat.columns

        assert df_feat.isnull().sum().sum() == 0

    def test_rolling_features_do_not_bleed_across_events(self, tagged_data):
        """Verify rolling window at sample 0 of any event draws strictly from that sample."""
        df_feat = engineer_kinematic_features(tagged_data)
        for _, grp in df_feat.groupby("event_id"):
            # At index 0, mean of window must equal the sample value itself
            assert np.isclose(grp["acc_mag_mean_3s"].iloc[0], grp["acc_mag_ms2"].iloc[0])
            assert np.isclose(grp["acc_mag_std_3s"].iloc[0], 0.0)


class TestEventLevelFeatures:
    """Tests for event-level summary feature matrix (109 rows)."""

    def test_event_level_shape_and_keys(self):
        """Verify event summary matrix has exactly 109 rows and retains identifiers."""
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)
        df_feat = engineer_kinematic_features(df_tagged)
        df_events = extract_event_level_features(df_feat)

        assert len(df_events) == 109
        assert "event_id" in df_events.columns
        assert "DriverID" in df_events.columns
        assert "Class" in df_events.columns
        assert "TaskID" in df_events.columns
        assert df_events["event_id"].nunique() == 109

    def test_event_level_statistical_consistency(self):
        """Verify min <= median <= max and RMS >= mean for non-negative signals."""
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)
        df_feat = engineer_kinematic_features(df_tagged)
        df_events = extract_event_level_features(df_feat)

        assert np.all(df_events["acc_mag_ms2_min"] <= df_events["acc_mag_ms2_mean"])
        assert np.all(df_events["acc_mag_ms2_mean"] <= df_events["acc_mag_ms2_max"])
        assert np.all(df_events["acc_mag_ms2_rms"] >= df_events["acc_mag_ms2_mean"])


class TestStratifiedGroupSplittingAndLeakagePrevention:
    """Tests for Stratified Group Splitting, zero-leakage isolation, and class balance."""

    @pytest.fixture
    def engineered_data(self):
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)
        return engineer_kinematic_features(df_tagged)

    def test_zero_event_leakage_across_splits(self, engineered_data):
        """Verify Train, Validation, and Test partitions share zero overlapping event IDs."""
        train, val, test, meta = split_dataset_b_stratified_group(engineered_data)

        train_events = set(train["event_id"])
        val_events = set(val["event_id"])
        test_events = set(test["event_id"])

        assert len(train_events & val_events) == 0
        assert len(train_events & test_events) == 0
        assert len(val_events & test_events) == 0

        assert len(train_events) == meta["train_events"]
        assert len(val_events) == meta["val_events"]
        assert len(test_events) == meta["test_events"]
        assert len(train_events) + len(val_events) + len(test_events) == 109

    def test_row_counts_sum_to_total(self, engineered_data):
        """Verify partitioned row counts sum exactly to 1,114."""
        train, val, test, _ = split_dataset_b_stratified_group(engineered_data)
        assert len(train) + len(val) + len(test) == len(engineered_data)
        assert len(train) + len(val) + len(test) == 1114

    def test_all_classes_represented_in_all_splits(self, engineered_data):
        """Verify all 4 maneuver classes are present in Train, Validation, and Test sets."""
        train, val, test, _ = split_dataset_b_stratified_group(engineered_data)
        assert set(train["Class"].unique()) == {1, 2, 3, 4}
        assert set(val["Class"].unique()) == {1, 2, 3, 4}
        assert set(test["Class"].unique()) == {1, 2, 3, 4}

    def test_all_drivers_represented_in_all_splits(self, engineered_data):
        """Verify all 3 drivers are represented in Train, Validation, and Test sets."""
        train, val, test, _ = split_dataset_b_stratified_group(engineered_data)
        assert set(train["DriverID"].unique()) == {"Driver-1", "Driver-2", "Driver-3"}
        assert set(val["DriverID"].unique()) == {"Driver-1", "Driver-2", "Driver-3"}
        assert set(test["DriverID"].unique()) == {"Driver-1", "Driver-2", "Driver-3"}


class TestArtifactPersistenceAndReproducibility:
    """Tests for disk artifact existence, hashes, and deterministic reproducibility."""

    def test_saved_artifacts_exist(self):
        """Verify all 6 generated artifacts exist on disk."""
        all_csv = PROCESSED_DIR / "dataset_b_processed_all.csv"
        train_csv = PROCESSED_DIR / "dataset_b_train.csv"
        val_csv = PROCESSED_DIR / "dataset_b_val.csv"
        test_csv = PROCESSED_DIR / "dataset_b_test.csv"
        event_csv = PROCESSED_DIR / "dataset_b_event_features.csv"
        meta_json = PROCESSED_DIR / "dataset_b_preprocessing_metadata.json"

        assert all_csv.exists()
        assert train_csv.exists()
        assert val_csv.exists()
        assert test_csv.exists()
        assert event_csv.exists()
        assert meta_json.exists()

    def test_split_reproducibility(self):
        """Verify splitting with random_state=42 is 100% deterministic."""
        df = load_raw_dataset()
        df_tagged, _ = tag_event_trials(df)
        df_feat = engineer_kinematic_features(df_tagged)

        train1, val1, test1, _ = split_dataset_b_stratified_group(df_feat, random_state=42)
        train2, val2, test2, _ = split_dataset_b_stratified_group(df_feat, random_state=42)

        assert list(train1["event_id"]) == list(train2["event_id"])
        assert list(val1["event_id"]) == list(val2["event_id"])
        assert list(test1["event_id"]) == list(test2["event_id"])

"""
Automated Pytest Suite: Offline vs Streaming Feature Parity for Pipeline A
==========================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Validates bit-for-bit parity (< 1e-5) between StreamingPipelineAFeatureExtractor
and offline batch compute_causal_features().
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.inference.streaming_pipeline_a_features import (
    StreamingPipelineAFeatureExtractor,
    CANDIDATE_V2_PRUNED_FEATURES
)
from scripts.engineer_pipeline_a_features import compute_causal_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed", "dataset_c_processed_all.csv")
CANDIDATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a", "pipeline_a_extratrees_candidate_v2.joblib")


@pytest.fixture(scope="module")
def dataset_c_raw():
    assert os.path.exists(PROCESSED_DATA_PATH), f"Dataset C missing at {PROCESSED_DATA_PATH}"
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


def test_streaming_extractor_init_and_reset():
    extractor = StreamingPipelineAFeatureExtractor()
    assert extractor.sample_count == 0
    assert not extractor.is_warmed_up
    assert len(extractor.feature_names) == 102
    
    # Process 1 dummy sample
    dummy = {'Acc X': 0.1, 'Acc Y': 0.2, 'Acc Z': -0.98, 'gyro_x': 0.01, 'gyro_y': 0.02, 'gyro_z': 0.03}
    feats = extractor.process_sample(dummy)
    assert extractor.sample_count == 1
    assert not extractor.is_warmed_up
    assert len(feats) == 102
    
    # Reset
    extractor.reset()
    assert extractor.sample_count == 0
    assert not extractor.is_warmed_up
    assert extractor._prev_acc_planar is None
    assert extractor._acc_planar_ema_fast is None


def test_streaming_extractor_warmup_transition():
    extractor = StreamingPipelineAFeatureExtractor()
    dummy = {'Acc X': 0.05, 'Acc Y': 0.15, 'Acc Z': -0.99, 'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0}
    
    for i in range(1, 60):
        extractor.process_sample(dummy)
        if i < 50:
            assert not extractor.is_warmed_up, f"Expected cold start at step {i}"
        else:
            assert extractor.is_warmed_up, f"Expected warmed up at step {i}"
    assert extractor.sample_count == 59


def test_streaming_vs_offline_batch_parity_session_1(dataset_c_raw):
    s1 = dataset_c_raw[dataset_c_raw['session'] == 'session_1'].head(250).copy().reset_index(drop=True)
    assert len(s1) == 250
    
    # 1. Offline Batch Computation
    df_batch = compute_causal_features(s1)
    
    # 2. Online Streaming Computation
    extractor = StreamingPipelineAFeatureExtractor()
    streaming_records = []
    for _, row in s1.iterrows():
        feat = extractor.process_sample(row)
        streaming_records.append(feat)
    df_stream = pd.DataFrame(streaming_records)
    
    # 3. Check every feature
    for feat_name in CANDIDATE_V2_PRUNED_FEATURES:
        assert feat_name in df_batch.columns, f"Missing {feat_name} in batch"
        assert feat_name in df_stream.columns, f"Missing {feat_name} in stream"
        
        b_vals = df_batch[feat_name].values
        s_vals = df_stream[feat_name].values
        
        diff = np.max(np.abs(b_vals - s_vals))
        assert diff < 1e-5, f"Feature {feat_name} exceeded 1e-5 tolerance: max diff = {diff:.2e}"


def test_streaming_vs_offline_batch_parity_session_2(dataset_c_raw):
    s2 = dataset_c_raw[dataset_c_raw['session'] == 'session_2'].head(250).copy().reset_index(drop=True)
    assert len(s2) == 250
    
    df_batch = compute_causal_features(s2)
    extractor = StreamingPipelineAFeatureExtractor()
    streaming_records = [extractor.process_sample(row) for _, row in s2.iterrows()]
    df_stream = pd.DataFrame(streaming_records)
    
    for feat_name in CANDIDATE_V2_PRUNED_FEATURES:
        b_vals = df_batch[feat_name].values
        s_vals = df_stream[feat_name].values
        diff = np.max(np.abs(b_vals - s_vals))
        assert diff < 1e-5, f"Feature {feat_name} exceeded 1e-5 tolerance on S2: max diff = {diff:.2e}"


def test_streaming_session_reset_cleanliness(dataset_c_raw):
    s1 = dataset_c_raw[dataset_c_raw['session'] == 'session_1'].head(100).copy().reset_index(drop=True)
    
    extractor_1 = StreamingPipelineAFeatureExtractor()
    res_run1 = [extractor_1.process_sample(row) for _, row in s1.iterrows()]
    
    # Reset and rerun identical sequence
    extractor_1.reset()
    res_run2 = [extractor_1.process_sample(row) for _, row in s1.iterrows()]
    
    df1 = pd.DataFrame(res_run1)
    df2 = pd.DataFrame(res_run2)
    
    for feat_name in CANDIDATE_V2_PRUNED_FEATURES:
        diff = np.max(np.abs(df1[feat_name].values - df2[feat_name].values))
        assert diff == 0.0, f"Post-reset run produced non-identical output for {feat_name}"


def test_process_sample_to_array(dataset_c_raw):
    sample = dataset_c_raw.iloc[0]
    extractor = StreamingPipelineAFeatureExtractor()
    arr = extractor.process_sample_to_array(sample)
    
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (102,)
    assert arr.dtype == np.float64
    assert not np.isnan(arr).any(), "NaN found in feature array"
    assert not np.isinf(arr).any(), "Inf found in feature array"


def test_streaming_inference_with_candidate_model(dataset_c_raw):
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        pytest.skip("Candidate model artifact not found, skipping inference test")
        
    cfg_path = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a", "pipeline_a_config_candidate_v2.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    model = joblib.load(CANDIDATE_MODEL_PATH)
    extractor = StreamingPipelineAFeatureExtractor(feature_names=cfg["feature_names"])
    
    s1_slice = dataset_c_raw[dataset_c_raw['session'] == 'session_1'].head(60).copy().reset_index(drop=True)
    
    probs = []
    preds = []
    for _, row in s1_slice.iterrows():
        feat_arr = extractor.process_sample_to_array(row).reshape(1, -1)
        prob = model.predict_proba(feat_arr)[0, 1]
        pred = int(prob >= cfg.get("calibrated_threshold", 0.50))
        probs.append(prob)
        preds.append(pred)
        
    assert len(probs) == 60
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(pr in [0, 1] for pr in preds)

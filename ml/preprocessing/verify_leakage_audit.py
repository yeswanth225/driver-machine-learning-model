import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Verify exact indices and boundaries
from ml.preprocessing.pipeline_dataset_c import (
    load_raw_dataset, remove_duplicates_and_tag_sessions,
    engineer_kinematic_features, split_session_chronological,
    GUARD_BAND_SAMPLES
)

df = load_raw_dataset()
df_s1, df_s2, meta = remove_duplicates_and_tag_sessions(df)

def verify_session_boundaries(df_s, sess_name):
    print(f"\n=== VERIFYING BOUNDARIES FOR {sess_name} (Length: {len(df_s)}) ===")
    n = len(df_s)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    tr_end_idx = n_train - GUARD_BAND_SAMPLES - 1
    val_start_idx = n_train
    val_end_idx = n_train + n_val - GUARD_BAND_SAMPLES - 1
    test_start_idx = n_train + n_val
    test_end_idx = n - 1
    
    print(f"Train indices: [0, {tr_end_idx}] (Count: {tr_end_idx + 1})")
    print(f"Guard band 1: [{tr_end_idx + 1}, {val_start_idx - 1}] (Count: {val_start_idx - tr_end_idx - 1})")
    print(f"Val indices:   [{val_start_idx}, {val_end_idx}] (Count: {val_end_idx - val_start_idx + 1})")
    print(f"Guard band 2: [{val_end_idx + 1}, {test_start_idx - 1}] (Count: {test_start_idx - val_end_idx - 1})")
    print(f"Test indices:  [{test_start_idx}, {test_end_idx}] (Count: {test_end_idx - test_start_idx + 1})")
    
    # Check max window size
    max_w = 30
    val_earliest_lookback = val_start_idx - max_w + 1
    print(f"Val start: {val_start_idx}, Val earliest lookback index: {val_earliest_lookback}")
    print(f"Is Val earliest lookback > Train end ({tr_end_idx})? {val_earliest_lookback > tr_end_idx} (Safety margin: {val_earliest_lookback - tr_end_idx} samples)")
    
    test_earliest_lookback = test_start_idx - max_w + 1
    print(f"Test start: {test_start_idx}, Test earliest lookback index: {test_earliest_lookback}")
    print(f"Is Test earliest lookback > Val end ({val_end_idx})? {test_earliest_lookback > val_end_idx} (Safety margin: {test_earliest_lookback - val_end_idx} samples)")

verify_session_boundaries(df_s1, "Session 1")
verify_session_boundaries(df_s2, "Session 2")

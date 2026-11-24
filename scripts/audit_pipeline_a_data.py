"""
Comprehensive Dataset C Quality Audit and Statistical Profiling
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_ALL = os.path.join(BASE_DIR, "dataset", "processed", "dataset_c_processed_all.csv")
RAW_PATH = os.path.join(BASE_DIR, "dataset", "raw", "dataset_03_5stn873wft", "3_FinalDatasetCsv.csv")
RAW_S1 = os.path.join(BASE_DIR, "dataset", "raw", "dataset_03_5stn873wft", "2_20210317_171452.csv")
RAW_S2 = os.path.join(BASE_DIR, "dataset", "raw", "dataset_03_5stn873wft", "1_20210317_184512.csv")

def run_audit():
    df_all = pd.read_csv(PROCESSED_ALL)
    df_raw = pd.read_csv(RAW_PATH)
    df_raw_s1 = pd.read_csv(RAW_S1)
    df_raw_s2 = pd.read_csv(RAW_S2)
    
    s1 = df_all[df_all['session'] == 'session_1'].copy()
    s2 = df_all[df_all['session'] == 'session_2'].copy()
    
    # 1. Row counts & Duplicates
    raw_total = len(df_raw)
    raw_s1_len = len(df_raw_s1)
    raw_s2_len = len(df_raw_s2)
    clean_total = len(df_all)
    dups_total = int(df_raw.duplicated().sum())
    
    # 2. Label block distribution
    blocks_s1 = (s1['label'] != s1['label'].shift()).cumsum()
    s1_block_summary = s1.groupby(blocks_s1)['label'].agg(['first', 'count']).rename(columns={'first': 'label', 'count': 'samples'})
    
    blocks_s2 = (s2['label'] != s2['label'].shift()).cumsum()
    s2_block_summary = s2.groupby(blocks_s2)['label'].agg(['first', 'count']).rename(columns={'first': 'label', 'count': 'samples'})
    
    # 3. Autocorrelation
    lags = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    autocorrs_s1 = {lag: float(s1['label'].autocorr(lag)) for lag in lags}
    autocorrs_s2 = {lag: float(s2['label'].autocorr(lag)) for lag in lags}
    
    # 4. Sensor baseline differences between sessions
    sensors = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_mag', 'gyro_mag', 'acc_horiz']
    sensor_stats = {}
    for s in sensors:
        s1_vals = s1[s]
        s2_vals = s2[s]
        t_stat, p_val = stats.ttest_ind(s1_vals, s2_vals, equal_var=False)
        ks_stat, ks_pval = stats.ks_2samp(s1_vals, s2_vals)
        sensor_stats[s] = {
            "s1_mean": float(s1_vals.mean()),
            "s1_std": float(s1_vals.std()),
            "s2_mean": float(s2_vals.mean()),
            "s2_std": float(s2_vals.std()),
            "mean_shift": float(abs(s1_vals.mean() - s2_vals.mean())),
            "std_ratio": float(s1_vals.std() / (s2_vals.std() + 1e-8)),
            "welch_t_stat": float(t_stat),
            "welch_p_val": float(p_val),
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval)
        }
        
    # 5. Split breakdown
    split_info = {}
    for split_name in ['train', 'val', 'test']:
        sp_df = df_all[df_all['split'] == split_name]
        ct = pd.crosstab(sp_df['session'], sp_df['label']).to_dict()
        split_info[split_name] = {
            "total": len(sp_df),
            "crosstab": ct
        }
        
    audit_results = {
        "raw_total": raw_total,
        "clean_total": clean_total,
        "duplicates_removed": dups_total,
        "s1_clean_len": len(s1),
        "s2_clean_len": len(s2),
        "s1_blocks": s1_block_summary.to_dict(orient="records"),
        "s2_blocks": s2_block_summary.to_dict(orient="records"),
        "autocorrelation_s1": autocorrs_s1,
        "autocorrelation_s2": autocorrs_s2,
        "sensor_session_shift": sensor_stats,
        "split_info": split_info
    }
    
    out_path = os.path.join(BASE_DIR, "reports", "pipeline_a", "dataset_c_audit_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
        
    print(f"Audit completed. Metrics saved to {out_path}")
    return audit_results

if __name__ == "__main__":
    res = run_audit()
    print("S1 blocks:", len(res["s1_blocks"]))
    print("S2 blocks:", len(res["s2_blocks"]))
    print("S1 Autocorr lag 10:", res["autocorrelation_s1"][10])
    print("Split breakdown:", res["split_info"])

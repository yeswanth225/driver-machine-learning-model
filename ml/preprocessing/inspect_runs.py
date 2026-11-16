import os
import pandas as pd

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df_final = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))

runs = (df_final['label'] != df_final['label'].shift()).cumsum()
run_df = df_final.groupby(runs).agg(
    label=('label', 'first'),
    count=('label', 'count'),
    start_idx=('label', lambda x: x.index[0]),
    end_idx=('label', lambda x: x.index[-1])
)
print("Label blocks across 3_FinalDatasetCsv.csv:")
for idx, row in run_df.iterrows():
    session = "Session 1 (File 2)" if row['start_idx'] < 7603 else "Session 2 (File 1)"
    duration_s = row['count'] / 10.0
    print(f"Block {idx:2d} | Label: {row['label']} | Rows: {row['count']:4d} ({duration_s:5.1f}s) | Range: [{row['start_idx']:5d}, {row['end_idx']:5d}] | {session}")

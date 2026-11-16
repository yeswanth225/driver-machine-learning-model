import os
import pandas as pd
import numpy as np

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df_final = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))
df1 = pd.read_csv(os.path.join(DATASET_C_DIR, "1_20210317_184512.csv"))
df2 = pd.read_csv(os.path.join(DATASET_C_DIR, "2_20210317_171452.csv"))

print("=== INSPECTING TIME, SAMPLING RATE, AND SENSOR CHARACTERISTICS ===")

# In df2: 7603 rows over 17:14:53 to 17:27:35
# Duration = 12 min 42 sec = 762 seconds!
# 7603 rows / 762 seconds = ~9.98 Hz (~10 Hz)!
t2_start = pd.to_datetime("2021-03-17 " + df2['Time'].iloc[0].replace("-", ":"))
t2_end = pd.to_datetime("2021-03-17 " + df2['Time'].iloc[-1].replace("-", ":"))
dur2_s = (t2_end - t2_start).total_seconds()
print(f"File 2 (Session 1): start {t2_start}, end {t2_end}, duration {dur2_s}s, rows {len(df2)}, freq ~{len(df2)/dur2_s:.2f} Hz")

t1_start = pd.to_datetime("2021-03-17 " + df1['Time'].iloc[0].replace("-", ":"))
t1_end = pd.to_datetime("2021-03-17 " + df1['Time'].iloc[-1].replace("-", ":"))
dur1_s = (t1_end - t1_start).total_seconds()
print(f"File 1 (Session 2): start {t1_start}, end {t1_end}, duration {dur1_s}s, rows {len(df1)}, freq ~{len(df1)/dur1_s:.2f} Hz")

# Check time distribution per second (samples per second)
vc_time2 = df2['Time'].value_counts()
print(f"Samples per second in File 2: min={vc_time2.min()}, max={vc_time2.max()}, mean={vc_time2.mean():.2f}, median={vc_time2.median()}")

vc_time1 = df1['Time'].value_counts()
print(f"Samples per second in File 1: min={vc_time1.min()}, max={vc_time1.max()}, mean={vc_time1.mean():.2f}, median={vc_time1.median()}")

# Check summary statistics of 3_FinalDatasetCsv.csv
print("\nSummary statistics of 3_FinalDatasetCsv.csv:")
print(df_final.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

# Check correlations with label
print("\nCorrelation with label:")
print(df_final.corr()['label'])

# Check if there are consecutive runs of label 0 and label 1
runs = (df_final['label'] != df_final['label'].shift()).cumsum()
run_lengths = df_final.groupby(runs)['label'].agg(['first', 'count'])
print(f"\nLabel runs count: {len(run_lengths)}")
print("Run lengths summary for label 0:\n", run_lengths[run_lengths['first'] == 0]['count'].describe())
print("Run lengths summary for label 1:\n", run_lengths[run_lengths['first'] == 1]['count'].describe())

import os
import pandas as pd
import numpy as np

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df_raw = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))
df1 = pd.read_csv(os.path.join(DATASET_C_DIR, "1_20210317_184512.csv"))
df2 = pd.read_csv(os.path.join(DATASET_C_DIR, "2_20210317_171452.csv"))

# Find exact duplicate rows
dup_mask = df_raw.duplicated(keep=False)
dup_rows = df_raw[dup_mask].copy()
dup_rows['original_index'] = dup_rows.index

print(f"Total duplicate-involved rows: {len(dup_rows)} (which yields 9 redundant rows)")
print(dup_rows)

# Group by identical rows to see clusters
grouped = dup_rows.groupby(list(df_raw.columns)).apply(lambda x: list(x.index)).reset_index(name='matching_indices')
print("\nDuplicate clusters:")
for idx, row in grouped.iterrows():
    indices = row['matching_indices']
    # Check if indices are in Session 1 or Session 2
    sess = ["S1" if i < 7603 else "S2" for i in indices]
    # Check timestamps from raw files
    times = []
    for i in indices:
        if i < 7603:
            times.append(f"Trip2 Time: {df2.iloc[i]['Time']}")
        else:
            times.append(f"Trip1 Time: {df1.iloc[i - 7603]['Time']}")
    print(f"\nCluster {idx+1}: Indices {indices} ({sess}) | Times: {times}")
    print(f"Sensor values: AccX={row['Acc X']}, AccY={row['Acc Y']}, AccZ={row['Acc Z']}, gyro_x={row['gyro_x']}, gyro_y={row['gyro_y']}, gyro_z={row['gyro_z']}, label={row['label']}")

import os
import pandas as pd
import numpy as np

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"

df1 = pd.read_csv(os.path.join(DATASET_C_DIR, "1_20210317_184512.csv"))
df2 = pd.read_csv(os.path.join(DATASET_C_DIR, "2_20210317_171452.csv"))
df3 = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))

print(f"df1: len {len(df1)}, df2: len {len(df2)}, df3: len {len(df3)}")

# Test concatenation order: is df3 == [df2, df1] or [df1, df2]?
# Check df3 rows 0:7603 vs df2
df3_part1 = df3.iloc[:len(df2)].reset_index(drop=True)
df3_part2 = df3.iloc[len(df2):].reset_index(drop=True)

# Compare sensor values
cols_sensor = ['gyro_x', 'gyro_y', 'gyro_z']
# Note: In df1 & df2, columns are 'Acc X', 'Acc Y', 'Acc Z'
diff_part1_df2 = (df3_part1[cols_sensor] - df2[cols_sensor]).abs().max()
print("Diff between df3[:7603] and df2 for gyro:\n", diff_part1_df2)

diff_acc_df2 = (df3_part1[['Acc X', 'Acc Y', 'Acc Z']] - df2[['Acc X', 'Acc Y', 'Acc Z']]).abs().max()
print("Diff between df3[:7603] and df2 for Acc:\n", diff_acc_df2)

diff_part2_df1 = (df3_part2[cols_sensor] - df1[cols_sensor]).abs().max()
print("Diff between df3[7603:] and df1 for gyro:\n", diff_part2_df1)

diff_acc_df1 = (df3_part2[['Acc X', 'Acc Y', 'Acc Z']] - df1[['Acc X', 'Acc Y', 'Acc Z']]).abs().max()
print("Diff between df3[7603:] and df1 for Acc:\n", diff_acc_df1)

# Check labels in part 1 and part 2!
print("\nLabels in df3[:7603] (File 2 part):")
print(df3_part1['label'].value_counts())

print("\nLabels in df3[7603:] (File 1 part):")
print(df3_part2['label'].value_counts())

# Check timestamps in df1 and df2
print("\nTime in df2 (first 5, last 5):")
print(df2['Time'].head())
print(df2['Time'].tail())

print("\nTime in df1 (first 5, last 5):")
print(df1['Time'].head())
print(df1['Time'].tail())

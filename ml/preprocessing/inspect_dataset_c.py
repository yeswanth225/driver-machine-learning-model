import os
import pandas as pd
import numpy as np

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"

files = ["1_20210317_184512.csv", "2_20210317_171452.csv", "3_FinalDatasetCsv.csv"]

print("=== DATASET C DETAILED INSPECTION ===")
for fname in files:
    fpath = os.path.join(DATASET_C_DIR, fname)
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
    
    print(f"\n" + "="*50)
    print(f"FILE: {fname} (Size: {os.path.getsize(fpath)} bytes)")
    print("="*50)
    
    df = pd.read_csv(fpath)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nHead (3 rows):\n{df.head(3)}")
    print(f"\nTail (3 rows):\n{df.tail(3)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows count: {df.duplicated().sum()}")
    
    # Check for target or label columns
    for col in df.columns:
        if any(term in col.lower() for term in ['class', 'label', 'target', 'state', 'driving', 'style']):
            print(f"\nValue counts for column '{col}':\n{df[col].value_counts(dropna=False)}")
            
    # Check timestamps or time columns
    time_cols = [c for c in df.columns if any(t in c.lower() for t in ['time', 'date', 'sec', 'ms', 'index'])]
    print(f"\nPotential time/index columns: {time_cols}")
    for tc in time_cols:
        print(f"  Col '{tc}' - min: {df[tc].min()}, max: {df[tc].max()}, nunique: {df[tc].nunique()}")
        if pd.api.types.is_numeric_dtype(df[tc]):
            diffs = df[tc].diff().dropna()
            print(f"  Col '{tc}' diff stats - min: {diffs.min()}, max: {diffs.max()}, mean: {diffs.mean()}, median: {diffs.median()}")

# Now investigate relationship between 1, 2, and 3
df1 = pd.read_csv(os.path.join(DATASET_C_DIR, files[0]))
df2 = pd.read_csv(os.path.join(DATASET_C_DIR, files[1]))
df3 = pd.read_csv(os.path.join(DATASET_C_DIR, files[2]))

print("\n" + "="*50)
print("RELATIONSHIP ANALYSIS: File 1 vs File 2 vs File 3")
print("="*50)
print(f"File 1 shape: {df1.shape}")
print(f"File 2 shape: {df2.shape}")
print(f"File 3 shape: {df3.shape}")
print(f"File 1 + File 2 rows: {len(df1) + len(df2)} vs File 3 rows: {len(df3)}")

# Check if columns match
print(f"File 1 columns == File 2 columns: {list(df1.columns) == list(df2.columns)}")
print(f"File 1 columns == File 3 columns: {list(df1.columns) == list(df3.columns)}")

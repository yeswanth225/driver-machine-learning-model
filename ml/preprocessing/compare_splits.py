import os
import pandas as pd
import numpy as np

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df_final = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))

print("=== SPLIT STRATEGY ANALYSIS ===")
# Total rows: 14249
# Session 1 (Trip 2): [0, 7602] (7603 rows)
# Session 2 (Trip 1): [7603, 14248] (6646 rows)

print("\n--- OPTION 1: Strict Session-Wise Split ---")
# Session 1 is Train + Val, Session 2 is Test
# Within Session 1 (7603 rows): Train = 75% (~5700 rows), Val = 25% (~1903 rows)
train_s1 = df_final.iloc[:5700]
val_s1 = df_final.iloc[5700:7603]
test_s2 = df_final.iloc[7603:]

print(f"Train: len={len(train_s1)} ({len(train_s1)/len(df_final):.1%}), class distribution:\n{train_s1['label'].value_counts(normalize=True).to_dict()}")
print(f"Val:   len={len(val_s1)} ({len(val_s1)/len(df_final):.1%}), class distribution:\n{val_s1['label'].value_counts(normalize=True).to_dict()}")
print(f"Test:  len={len(test_s2)} ({len(test_s2)/len(df_final):.1%}), class distribution:\n{test_s2['label'].value_counts(normalize=True).to_dict()}")

print("\n--- OPTION 2: Chronological Split within EACH Session (Stratified Sessions) ---")
# Split Session 1: 70% Train, 15% Val, 15% Test
# Split Session 2: 70% Train, 15% Val, 15% Test
# Combined Train = Train_S1 + Train_S2, etc.
# With guard bands (e.g. 10s = 100 rows) dropped at boundary
def split_session(df_sub, prefix=""):
    n = len(df_sub)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = df_sub.iloc[:n_train]
    val = df_sub.iloc[n_train:n_train + n_val]
    test = df_sub.iloc[n_train + n_val:]
    return train, val, test

t1, v1, te1 = split_session(df_final.iloc[:7603])
t2, v2, te2 = split_session(df_final.iloc[7603:])
t_comb = pd.concat([t1, t2])
v_comb = pd.concat([v1, v2])
te_comb = pd.concat([te1, te2])
print(f"Combined Train: len={len(t_comb)} ({len(t_comb)/len(df_final):.1%}), classes: {t_comb['label'].value_counts(normalize=True).to_dict()}")
print(f"Combined Val:   len={len(v_comb)} ({len(v_comb)/len(df_final):.1%}), classes: {v_comb['label'].value_counts(normalize=True).to_dict()}")
print(f"Combined Test:  len={len(te_comb)} ({len(te_comb)/len(df_final):.1%}), classes: {te_comb['label'].value_counts(normalize=True).to_dict()}")

print("\n--- OPTION 3: Pure Chronological Split of 3_FinalDatasetCsv.csv (70/15/15) ---")
n_tot = len(df_final)
n_train_tot = int(0.70 * n_tot) # 9974
n_val_tot = int(0.15 * n_tot)   # 2137
# test = 2138
train_c = df_final.iloc[:n_train_tot]
val_c = df_final.iloc[n_train_tot:n_train_tot + n_val_tot]
test_c = df_final.iloc[n_train_tot + n_val_tot:]
print(f"Train_C: len={len(train_c)} ({len(train_c)/len(df_final):.1%}), classes: {train_c['label'].value_counts(normalize=True).to_dict()}")
print(f"Val_C:   len={len(val_c)} ({len(val_c)/len(df_final):.1%}), classes: {val_c['label'].value_counts(normalize=True).to_dict()}")
print(f"Test_C:  len={len(test_c)} ({len(test_c)/len(df_final):.1%}), classes: {test_c['label'].value_counts(normalize=True).to_dict()}")

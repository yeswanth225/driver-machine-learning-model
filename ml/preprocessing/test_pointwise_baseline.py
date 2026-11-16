import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))

# Drop the 9 exact duplicate rows (out of 14,249)
df_clean = df.drop_duplicates().reset_index(drop=True)
print(f"Clean df shape: {df_clean.shape} (dropped {len(df) - len(df_clean)} duplicates)")

# Add kinematic signals
df_clean['acc_mag'] = np.sqrt(df_clean['Acc X']**2 + df_clean['Acc Y']**2 + df_clean['Acc Z']**2)
df_clean['gyro_mag'] = np.sqrt(df_clean['gyro_x']**2 + df_clean['gyro_y']**2 + df_clean['gyro_z']**2)
df_clean['acc_horiz'] = np.sqrt(df_clean['Acc X']**2 + df_clean['Acc Y']**2)

feature_cols = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_mag', 'gyro_mag', 'acc_horiz']

# Test chronological split on df_clean (70% train, 15% val, 15% test) with Session preservation or chronological split
# Let's test Session 1 (0 to ~7600) vs Session 2 (~7600 to end)
split_idx_session = 7598 # approx session boundary after dropping duplicates
df_s1 = df_clean.iloc[:split_idx_session].copy()
df_s2 = df_clean.iloc[split_idx_session:].copy()

print(f"Session 1: {len(df_s1)}, Session 2: {len(df_s2)}")

# Let's do chronological split within each session with guard-bands:
# 10s guard band = 100 samples at 10 Hz
guard = 100

def get_session_splits(df_sub, guard_samples=100):
    n = len(df_sub)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train = df_sub.iloc[:n_train - guard_samples]
    val = df_sub.iloc[n_train:n_train + n_val - guard_samples]
    test = df_sub.iloc[n_train + n_val:]
    return train, val, test

tr1, va1, te1 = get_session_splits(df_s1, guard)
tr2, va2, te2 = get_session_splits(df_s2, guard)

train = pd.concat([tr1, tr2]).reset_index(drop=True)
val = pd.concat([va1, va2]).reset_index(drop=True)
test = pd.concat([te1, te2]).reset_index(drop=True)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Train labels:\n{train['label'].value_counts(normalize=True)}")
print(f"Val labels:\n{val['label'].value_counts(normalize=True)}")
print(f"Test labels:\n{test['label'].value_counts(normalize=True)}")

X_tr, y_tr = train[feature_cols], train['label']
X_va, y_va = val[feature_cols], val['label']
X_te, y_te = test[feature_cols], test['label']

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

pred_va = rf.predict(X_va)
pred_te = rf.predict(X_te)
prob_te = rf.predict_proba(X_te)[:, 1]

print("\n--- POINT-WISE RF RESULTS ---")
print("Validation Report:\n", classification_report(y_va, pred_va, digits=4))
print("Test Report:\n", classification_report(y_te, pred_te, digits=4))
print("Test ROC-AUC:", roc_auc_score(y_te, prob_te))

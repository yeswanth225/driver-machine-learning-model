import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))

# Drop exact duplicates
df_clean = df.drop_duplicates().reset_index(drop=True)
split_idx_session = 7598
df_s1 = df_clean.iloc[:split_idx_session].copy().reset_index(drop=True)
df_s2 = df_clean.iloc[split_idx_session:].copy().reset_index(drop=True)

def add_temporal_features(df_sess):
    df_s = df_sess.copy()
    # Instantaneous physical features
    df_s['acc_mag'] = np.sqrt(df_s['Acc X']**2 + df_s['Acc Y']**2 + df_s['Acc Z']**2)
    df_s['gyro_mag'] = np.sqrt(df_s['gyro_x']**2 + df_s['gyro_y']**2 + df_s['gyro_z']**2)
    df_s['acc_horiz'] = np.sqrt(df_s['Acc X']**2 + df_s['Acc Y']**2)
    df_s['acc_jerk'] = df_s['acc_mag'].diff().fillna(0)
    
    # Causal backward rolling windows (1s = 10 samples, 3s = 30 samples)
    for w in [10, 30]:
        sec = w // 10
        # Rolling means
        df_s[f'acc_mag_mean_{sec}s'] = df_s['acc_mag'].rolling(w, min_periods=1).mean()
        df_s[f'acc_mag_std_{sec}s'] = df_s['acc_mag'].rolling(w, min_periods=1).std().fillna(0)
        df_s[f'acc_mag_max_{sec}s'] = df_s['acc_mag'].rolling(w, min_periods=1).max()
        df_s[f'gyro_mag_mean_{sec}s'] = df_s['gyro_mag'].rolling(w, min_periods=1).mean()
        df_s[f'gyro_mag_std_{sec}s'] = df_s['gyro_mag'].rolling(w, min_periods=1).std().fillna(0)
        df_s[f'gyro_mag_max_{sec}s'] = df_s['gyro_mag'].rolling(w, min_periods=1).max()
        df_s[f'acc_horiz_mean_{sec}s'] = df_s['acc_horiz'].rolling(w, min_periods=1).mean()
        df_s[f'acc_horiz_std_{sec}s'] = df_s['acc_horiz'].rolling(w, min_periods=1).std().fillna(0)
    return df_s

df_feat1 = add_temporal_features(df_s1)
df_feat2 = add_temporal_features(df_s2)

guard = 100 # 10s guard band between chronological splits
def split_sess(df_sub):
    n = len(df_sub)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    tr = df_sub.iloc[:n_tr - guard]
    va = df_sub.iloc[n_tr:n_tr + n_va - guard]
    te = df_sub.iloc[n_tr + n_va:]
    return tr, va, te

tr1, va1, te1 = split_sess(df_feat1)
tr2, va2, te2 = split_sess(df_feat2)

train = pd.concat([tr1, tr2]).reset_index(drop=True)
val = pd.concat([va1, va2]).reset_index(drop=True)
test = pd.concat([te1, te2]).reset_index(drop=True)

drop_cols = ['label']
feat_cols = [c for c in train.columns if c not in drop_cols]
print(f"Feature count: {len(feat_cols)}")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train[feat_cols], train['label'])

p_va = rf.predict(val[feat_cols])
p_te = rf.predict(test[feat_cols])
prob_te = rf.predict_proba(test[feat_cols])[:, 1]

print("\n--- CAUSAL ROLLING TEMPORAL RF RESULTS ---")
print("Validation Report:\n", classification_report(val['label'], p_va, digits=4))
print("Test Report:\n", classification_report(test['label'], p_te, digits=4))
print("Test ROC-AUC:", roc_auc_score(test['label'], prob_te))

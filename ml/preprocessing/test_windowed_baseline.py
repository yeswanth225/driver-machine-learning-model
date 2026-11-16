import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))

# Drop the 9 exact duplicate rows
df_clean = df.drop_duplicates().reset_index(drop=True)
split_idx_session = 7598
df_s1 = df_clean.iloc[:split_idx_session].copy()
df_s2 = df_clean.iloc[split_idx_session:].copy()

def extract_window_features(df_session, window_size=30, stride=15): # 3s window, 1.5s stride
    # Compute signals
    acc_mag = np.sqrt(df_session['Acc X']**2 + df_session['Acc Y']**2 + df_session['Acc Z']**2)
    gyro_mag = np.sqrt(df_session['gyro_x']**2 + df_session['gyro_y']**2 + df_session['gyro_z']**2)
    acc_horiz = np.sqrt(df_session['Acc X']**2 + df_session['Acc Y']**2)
    
    signals = {
        'acc_x': df_session['Acc X'].values,
        'acc_y': df_session['Acc Y'].values,
        'acc_z': df_session['Acc Z'].values,
        'gyro_x': df_session['gyro_x'].values,
        'gyro_y': df_session['gyro_y'].values,
        'gyro_z': df_session['gyro_z'].values,
        'acc_mag': acc_mag.values,
        'gyro_mag': gyro_mag.values,
        'acc_horiz': acc_horiz.values,
    }
    labels = df_session['label'].values
    
    rows = []
    n_samples = len(df_session)
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        win_label = int(np.round(np.mean(labels[start:end]))) # majority vote
        
        feat_dict = {'window_start': start, 'window_end': end, 'label': win_label}
        for name, sig in signals.items():
            w = sig[start:end]
            feat_dict[f'{name}_mean'] = np.mean(w)
            feat_dict[f'{name}_std'] = np.std(w)
            feat_dict[f'{name}_min'] = np.min(w)
            feat_dict[f'{name}_max'] = np.max(w)
            feat_dict[f'{name}_range'] = np.ptp(w)
            feat_dict[f'{name}_iqr'] = np.percentile(w, 75) - np.percentile(w, 25)
        rows.append(feat_dict)
    return pd.DataFrame(rows)

wdf1 = extract_window_features(df_s1, window_size=30, stride=15)
wdf2 = extract_window_features(df_s2, window_size=30, stride=15)
print(f"Windowed S1 shape: {wdf1.shape}, S2 shape: {wdf2.shape}")

# Split chronologically with guard band (e.g. 5 windows = 7.5s)
guard_w = 4
def split_window_df(wdf):
    n = len(wdf)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    tr = wdf.iloc[:n_tr - guard_w]
    va = wdf.iloc[n_tr:n_tr + n_va - guard_w]
    te = wdf.iloc[n_tr + n_va:]
    return tr, va, te

tr1, va1, te1 = split_window_df(wdf1)
tr2, va2, te2 = split_window_df(wdf2)

w_train = pd.concat([tr1, tr2]).reset_index(drop=True)
w_val = pd.concat([va1, va2]).reset_index(drop=True)
w_test = pd.concat([te1, te2]).reset_index(drop=True)

feat_cols = [c for c in w_train.columns if c not in ['window_start', 'window_end', 'label']]
print(f"Num features: {len(feat_cols)}")
print(f"Window train: {len(w_train)}, val: {len(w_val)}, test: {len(w_test)}")
print("Train label distribution:\n", w_train['label'].value_counts(normalize=True))
print("Val label distribution:\n", w_val['label'].value_counts(normalize=True))
print("Test label distribution:\n", w_test['label'].value_counts(normalize=True))

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(w_train[feat_cols], w_train['label'])

pred_va = rf.predict(w_val[feat_cols])
pred_te = rf.predict(w_test[feat_cols])
prob_te = rf.predict_proba(w_test[feat_cols])[:, 1]

print("\n--- WINDOW-BASED RF RESULTS (3s window, 1.5s stride) ---")
print("Validation Report:\n", classification_report(w_val['label'], pred_va, digits=4))
print("Test Report:\n", classification_report(w_test['label'], pred_te, digits=4))
print("Test ROC-AUC:", roc_auc_score(w_test['label'], prob_te))

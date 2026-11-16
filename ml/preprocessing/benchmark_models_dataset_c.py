import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
raw_file = os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv")
df_raw = pd.read_csv(raw_file)

# 1. Deduplication
df_clean = df_raw.drop_duplicates().reset_index(drop=True)
print(f"Total raw: {len(df_raw)}, Clean: {len(df_clean)}, Duplicates: {len(df_raw) - len(df_clean)}")

# 2. Session identification
# Session 1 corresponds to rows originally from 2_20210317_171452.csv (7603 rows)
# In df_raw, indices 0..7602 were Session 1; in df_clean, 3 duplicates were in Session 1, 6 in Session 2
# Let's verify exact split point
s1_raw_len = 7603
df_s1_raw = df_raw.iloc[:s1_raw_len]
df_s2_raw = df_raw.iloc[s1_raw_len:]
dups_s1 = df_s1_raw.duplicated().sum() # 3
dups_s2 = df_s2_raw.duplicated().sum() # 6

s1_clean_len = s1_raw_len - dups_s1 # 7600
df_s1 = df_clean.iloc[:s1_clean_len].copy().reset_index(drop=True)
df_s2 = df_clean.iloc[s1_clean_len:].copy().reset_index(drop=True)

print(f"Clean Session 1: {len(df_s1)} rows, Session 2: {len(df_s2)} rows")

def engineer_features(df_sess):
    df_feat = df_sess.copy()
    # Kinematic signals
    df_feat['acc_mag'] = np.sqrt(df_feat['Acc X']**2 + df_feat['Acc Y']**2 + df_feat['Acc Z']**2)
    df_feat['gyro_mag'] = np.sqrt(df_feat['gyro_x']**2 + df_feat['gyro_y']**2 + df_feat['gyro_z']**2)
    df_feat['acc_horiz'] = np.sqrt(df_feat['Acc X']**2 + df_feat['Acc Y']**2)
    
    # Jerk / temporal rates of change (causal backward diff)
    df_feat['acc_jerk'] = df_feat['acc_mag'].diff().fillna(0)
    df_feat['gyro_jerk'] = df_feat['gyro_mag'].diff().fillna(0)
    
    # Causal backward rolling statistics: 1s (10 samples) and 3s (30 samples)
    for w in [10, 30]:
        sec = w // 10
        df_feat[f'acc_mag_mean_{sec}s'] = df_feat['acc_mag'].rolling(w, min_periods=1).mean()
        df_feat[f'acc_mag_std_{sec}s'] = df_feat['acc_mag'].rolling(w, min_periods=1).std().fillna(0)
        df_feat[f'acc_mag_max_{sec}s'] = df_feat['acc_mag'].rolling(w, min_periods=1).max()
        df_feat[f'gyro_mag_mean_{sec}s'] = df_feat['gyro_mag'].rolling(w, min_periods=1).mean()
        df_feat[f'gyro_mag_std_{sec}s'] = df_feat['gyro_mag'].rolling(w, min_periods=1).std().fillna(0)
        df_feat[f'gyro_mag_max_{sec}s'] = df_feat['gyro_mag'].rolling(w, min_periods=1).max()
        df_feat[f'acc_horiz_mean_{sec}s'] = df_feat['acc_horiz'].rolling(w, min_periods=1).mean()
        df_feat[f'acc_horiz_std_{sec}s'] = df_feat['acc_horiz'].rolling(w, min_periods=1).std().fillna(0)
    return df_feat

df_s1_feat = engineer_features(df_s1)
df_s2_feat = engineer_features(df_s2)

# Chronological split with 10s (100 samples) guard-band buffer
guard = 100
def split_session(df_sub, sess_id):
    n = len(df_sub)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    
    tr = df_sub.iloc[:n_tr - guard].copy()
    tr['split'] = 'train'
    
    va = df_sub.iloc[n_tr:n_tr + n_va - guard].copy()
    va['split'] = 'val'
    
    te = df_sub.iloc[n_tr + n_va:].copy()
    te['split'] = 'test'
    
    tr['session'] = sess_id
    va['session'] = sess_id
    te['session'] = sess_id
    return tr, va, te

tr1, va1, te1 = split_session(df_s1_feat, 'session_1')
tr2, va2, te2 = split_session(df_s2_feat, 'session_2')

train = pd.concat([tr1, tr2]).reset_index(drop=True)
val = pd.concat([va1, va2]).reset_index(drop=True)
test = pd.concat([te1, te2]).reset_index(drop=True)

feature_cols = [c for c in train.columns if c not in ['label', 'split', 'session']]
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

X_train, y_train = train[feature_cols], train['label']
X_val, y_val = val[feature_cols], val['label']
X_test, y_test = test[feature_cols], test['label']

# Scaler fitted ONLY on train data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        "use_scaled": False
    },
    "Extra Trees": {
        "model": ExtraTreesClassifier(n_estimators=150, max_depth=12, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),
        "use_scaled": False
    },
    "XGBoost": {
        "model": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.62, random_state=42, eval_metric="logloss"),
        "use_scaled": False
    },
    "Support Vector Machine": {
        "model": SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        "use_scaled": True
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
        "use_scaled": True
    }
}

print("\n" + "="*70)
print(f"{'Model':<25} | {'Val Acc':<8} | {'Val F1-Macro':<12} | {'Val Rec-Agg':<11} | {'Test Acc':<8} | {'Test F1-Macro':<13} | {'Test Rec-Agg':<12} | {'Test AUC':<8}")
print("="*70)

results = {}
for name, cfg in models.items():
    clf = cfg['model']
    use_scaled = cfg['use_scaled']
    X_tr_m = X_train_scaled if use_scaled else X_train
    X_va_m = X_val_scaled if use_scaled else X_val
    X_te_m = X_test_scaled if use_scaled else X_test
    
    clf.fit(X_tr_m, y_train)
    
    pred_va = clf.predict(X_va_m)
    prob_va = clf.predict_proba(X_va_m)[:, 1] if hasattr(clf, "predict_proba") else None
    
    pred_te = clf.predict(X_te_m)
    prob_te = clf.predict_proba(X_te_m)[:, 1] if hasattr(clf, "predict_proba") else None
    
    val_acc = accuracy_score(y_val, pred_va)
    val_f1_macro = f1_score(y_val, pred_va, average='macro')
    val_rec_agg = recall_score(y_val, pred_va, pos_label=1)
    
    test_acc = accuracy_score(y_test, pred_te)
    test_prec_agg = precision_score(y_test, pred_te, pos_label=1)
    test_rec_agg = recall_score(y_test, pred_te, pos_label=1)
    test_f1_agg = f1_score(y_test, pred_te, pos_label=1)
    test_f1_macro = f1_score(y_test, pred_te, average='macro')
    test_auc = roc_auc_score(y_test, prob_te) if prob_te is not None else 0.0
    cm = confusion_matrix(y_test, pred_te)
    
    results[name] = {
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1_macro,
        "val_recall_aggressive": val_rec_agg,
        "test_accuracy": test_acc,
        "test_precision_aggressive": test_prec_agg,
        "test_recall_aggressive": test_rec_agg,
        "test_f1_aggressive": test_f1_agg,
        "test_f1_macro": test_f1_macro,
        "test_roc_auc": test_auc,
        "confusion_matrix": cm.tolist()
    }
    
    print(f"{name:<25} | {val_acc:<8.4f} | {val_f1_macro:<12.4f} | {val_rec_agg:<11.4f} | {test_acc:<8.4f} | {test_f1_macro:<13.4f} | {test_rec_agg:<12.4f} | {test_auc:<8.4f}")

print("="*70)

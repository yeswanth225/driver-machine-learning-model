import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score
)

# Load data and setup splits as before
DATASET_C_DIR = r"d:\driver ml model\driver-monitoring-agent\dataset\raw\dataset_03_5stn873wft"
df_raw = pd.read_csv(os.path.join(DATASET_C_DIR, "3_FinalDatasetCsv.csv"))
df_clean = df_raw.drop_duplicates().reset_index(drop=True)

s1_raw_len = 7603
dups_s1 = df_raw.iloc[:s1_raw_len].duplicated().sum()
s1_clean_len = s1_raw_len - dups_s1

df_s1 = df_clean.iloc[:s1_clean_len].copy().reset_index(drop=True)
df_s2 = df_clean.iloc[s1_clean_len:].copy().reset_index(drop=True)

def engineer_features(df_sess):
    df_feat = df_sess.copy()
    # Physical signals
    df_feat['acc_mag'] = np.sqrt(df_feat['Acc X']**2 + df_feat['Acc Y']**2 + df_feat['Acc Z']**2)
    df_feat['gyro_mag'] = np.sqrt(df_feat['gyro_x']**2 + df_feat['gyro_y']**2 + df_feat['gyro_z']**2)
    df_feat['acc_horiz'] = np.sqrt(df_feat['Acc X']**2 + df_feat['Acc Y']**2)
    
    # Derivatives (jerk)
    df_feat['acc_jerk'] = df_feat['acc_mag'].diff().fillna(0)
    df_feat['gyro_jerk'] = df_feat['gyro_mag'].diff().fillna(0)
    
    # Causal rolling windows (1s = 10 samples, 2s = 20 samples, 3s = 30 samples)
    for w in [10, 20, 30]:
        sec = w / 10.0
        sec_str = f"{int(sec)}s"
        df_feat[f'acc_mag_mean_{sec_str}'] = df_feat['acc_mag'].rolling(w, min_periods=1).mean()
        df_feat[f'acc_mag_std_{sec_str}'] = df_feat['acc_mag'].rolling(w, min_periods=1).std().fillna(0)
        df_feat[f'acc_mag_max_{sec_str}'] = df_feat['acc_mag'].rolling(w, min_periods=1).max()
        df_feat[f'acc_mag_min_{sec_str}'] = df_feat['acc_mag'].rolling(w, min_periods=1).min()
        
        df_feat[f'gyro_mag_mean_{sec_str}'] = df_feat['gyro_mag'].rolling(w, min_periods=1).mean()
        df_feat[f'gyro_mag_std_{sec_str}'] = df_feat['gyro_mag'].rolling(w, min_periods=1).std().fillna(0)
        df_feat[f'gyro_mag_max_{sec_str}'] = df_feat['gyro_mag'].rolling(w, min_periods=1).max()
        
        df_feat[f'acc_horiz_mean_{sec_str}'] = df_feat['acc_horiz'].rolling(w, min_periods=1).mean()
        df_feat[f'acc_horiz_std_{sec_str}'] = df_feat['acc_horiz'].rolling(w, min_periods=1).std().fillna(0)
        
        # Yaw rate rolling (swerving indicator)
        df_feat[f'gyro_z_std_{sec_str}'] = df_feat['gyro_z'].rolling(w, min_periods=1).std().fillna(0)
        df_feat[f'gyro_z_max_{sec_str}'] = df_feat['gyro_z'].abs().rolling(w, min_periods=1).max()

    return df_feat

df_s1_feat = engineer_features(df_s1)
df_s2_feat = engineer_features(df_s2)

guard = 100 # 10s guard band
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
print(f"Total features: {len(feature_cols)}")

X_train, y_train = train[feature_cols], train['label']
X_val, y_val = val[feature_cols], val['label']
X_test, y_test = test[feature_cols], test['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Test hyperparameter tuned ExtraTrees vs RandomForest vs XGBoost vs SVM vs KNN
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=6, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_split=6, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.65, random_state=42, eval_metric="logloss"),
    "Support Vector Machine": SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=25, weights='distance', n_jobs=-1)
}

scaled_needed = {"Support Vector Machine", "K-Nearest Neighbors"}

for name, model in models.items():
    is_scaled = name in scaled_needed
    X_tr = X_train_scaled if is_scaled else X_train
    X_va = X_val_scaled if is_scaled else X_val
    X_te = X_test_scaled if is_scaled else X_test
    
    model.fit(X_tr, y_train)
    
    p_va = model.predict(X_va)
    p_te = model.predict(X_te)
    prob_te = model.predict_proba(X_te)[:, 1]
    
    print(f"\n=================== {name} ===================")
    print("VAL REPORT:")
    print(classification_report(y_val, p_va, digits=4))
    print("TEST REPORT:")
    print(classification_report(y_test, p_te, digits=4))
    print("CONFUSION MATRIX (Test):")
    cm = confusion_matrix(y_test, p_te)
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],\n [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print(f"Accuracy: {accuracy_score(y_test, p_te):.4f}")
    print(f"Macro F1: {f1_score(y_test, p_te, average='macro'):.4f}")
    print(f"Aggressive Recall: {recall_score(y_test, p_te, pos_label=1):.4f}")
    print(f"Aggressive Precision: {precision_score(y_test, p_te, pos_label=1):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, prob_te):.4f}")
    print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, prob_te):.4f}")

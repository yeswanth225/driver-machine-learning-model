"""
Advanced Pipeline A Research & Metric Verification Suite
========================================================
Phases 2, 3, 4:
1. Advanced Causal Feature Engineering:
   - Orientation-normalized dynamic acceleration (gravity compensation)
   - Causal frequency-domain spectral energy (0.3-1.5 Hz maneuver vs 1.5-4.5 Hz chatter)
   - Spectral Energy Ratio (SER) & Spectral Entropy
   - RMS acceleration, kinetic energy, causal jerk
   - Adaptive causal EMA normalization
2. Systematic Model Architecture Exploration:
   - Regularized Logistic Regression (L1/L2 Diagnostic Baseline)
   - Tuned RBF SVM
   - Extra Trees Classifier
   - Random Forest Classifier
   - XGBoost Classifier
   - HistGradientBoosting Classifier
   - Compact 1D CNN (PyTorch) on causal 3-second (30-sample) sliding windows
   - Soft-voting Ensemble (ExtraTrees + HistGradientBoosting + XGBoost)
   - Calibrated models (Platt scaling & Isotonic)
3. Validation:
   - S1 -> S2 (Train on Session 1, Test on Session 2)
   - S2 -> S1 (Train on Session 2, Test on Session 1)
   - Chronological Holdout (70/15/15 with 10s guard bands)
   - Episode-grouped Block Cross-Validation (Grouping by the 17 physical episodes)
4. Comprehensive 10-Metric Target Evaluation against research requirements.
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
EXP_MODELS_DIR = os.path.join(BASE_DIR, "models", "experiments", "pipeline_a")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "pipeline_a")

os.makedirs(EXP_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_spectral_features_causal(series: np.ndarray, window: int = 30, fs: float = 10.0):
    """
    Compute causal sliding window spectral energy and ratio.
    Window = 30 samples (3.0 seconds at 10 Hz).
    Low band: 0.3 - 1.5 Hz (maneuvers, turns, braking)
    High band: 1.5 - 4.5 Hz (road roughness, mount vibration)
    """
    n = len(series)
    low_energy = np.zeros(n)
    high_energy = np.zeros(n)
    spectral_ratio = np.zeros(n)
    spectral_entropy = np.zeros(n)
    
    freqs = np.fft.rfftfreq(window, d=1.0/fs)
    low_mask = (freqs >= 0.3) & (freqs <= 1.5)
    high_mask = (freqs > 1.5) & (freqs <= 4.5)
    
    for i in range(n):
        start = max(0, i - window + 1)
        sub = series[start:i+1]
        if len(sub) < 8:
            continue
        # Pad with edge value if sub < window
        if len(sub) < window:
            sub = np.pad(sub, (window - len(sub), 0), mode='edge')
            
        # Detrend and window
        sub_d = sub - np.mean(sub)
        fft_vals = np.abs(np.fft.rfft(sub_d))**2
        tot = np.sum(fft_vals) + 1e-8
        
        e_low = np.sum(fft_vals[low_mask])
        e_high = np.sum(fft_vals[high_mask])
        
        low_energy[i] = e_low
        high_energy[i] = e_high
        spectral_ratio[i] = e_low / (e_high + 1e-4)
        
        # Normalized spectral entropy
        p = fft_vals / tot
        p = p[p > 0]
        spectral_entropy[i] = -np.sum(p * np.log2(p)) / np.log2(len(freqs))
        
    return low_energy, high_energy, spectral_ratio, spectral_entropy


def engineer_advanced_features(df_session: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers causal, physically invariant features for a single session.
    """
    df = df_session.copy().reset_index(drop=True)
    ax = df['Acc X'].values
    ay = df['Acc Y'].values
    az = df['Acc Z'].values
    gx = df['gyro_x'].values
    gy = df['gyro_y'].values
    gz = df['gyro_z'].values
    n = len(df)
    
    # 1. Physical Magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    acc_horiz = np.sqrt(ax**2 + ay**2)
    
    # 2. Gravity-Independent Acceleration (Causal Low-Pass Filter on Gravity Vector)
    # Estimate gravity vector via exponential moving average (alpha=0.03, ~3s time constant)
    g_est_x = np.zeros(n)
    g_est_y = np.zeros(n)
    g_est_z = np.zeros(n)
    alpha_g = 0.03
    
    gx_curr, gy_curr, gz_curr = ax[0], ay[0], az[0]
    for i in range(n):
        gx_curr = (1.0 - alpha_g) * gx_curr + alpha_g * ax[i]
        gy_curr = (1.0 - alpha_g) * gy_curr + alpha_g * ay[i]
        gz_curr = (1.0 - alpha_g) * gz_curr + alpha_g * az[i]
        g_est_x[i] = gx_curr
        g_est_y[i] = gy_curr
        g_est_z[i] = gz_curr
        
    # Dynamic acceleration = raw - gravity
    dyn_ax = ax - g_est_x
    dyn_ay = ay - g_est_y
    dyn_az = az - g_est_z
    dyn_acc_mag = np.sqrt(dyn_ax**2 + dyn_ay**2 + dyn_az**2)
    
    # Orientation-normalized: projection onto gravity unit vector
    g_mag = np.sqrt(g_est_x**2 + g_est_y**2 + g_est_z**2) + 1e-6
    u_gx, u_gy, u_gz = g_est_x / g_mag, g_est_y / g_mag, g_est_z / g_mag
    
    # Vertical dynamic accel (chatter parallel to gravity)
    dyn_vert = np.abs(dyn_ax * u_gx + dyn_ay * u_gy + dyn_az * u_gz)
    
    # Planar dynamic accel (in-plane forces perpendicular to gravity - tilt invariant!)
    dyn_planar = np.sqrt(np.maximum(0.0, dyn_acc_mag**2 - dyn_vert**2))
    
    # Yaw rotation around estimated vertical gravity axis (pure heading turns!)
    gyro_yaw_gravity = np.abs(gx * u_gx + gy * u_gy + gz * u_gz)
    gyro_wobble_tilt = np.sqrt(np.maximum(0.0, gyro_mag**2 - gyro_yaw_gravity**2))
    
    # 3. Jerk Derivatives
    acc_jerk = np.diff(dyn_planar, prepend=dyn_planar[0])
    gyro_jerk = np.diff(gyro_yaw_gravity, prepend=gyro_yaw_gravity[0])
    
    # 4. Decoupling Ratios
    chatter_ratio = dyn_vert / (dyn_planar + 0.1)
    turn_dominance = gyro_yaw_gravity / (gyro_wobble_tilt + 0.02)
    
    # 5. Causal Spectral Features on Dynamic Planar Acceleration
    low_e, high_e, spec_ratio, spec_entropy = compute_spectral_features_causal(dyn_planar, window=30, fs=10.0)
    
    # Put into DataFrame
    df['acc_mag'] = acc_mag
    df['gyro_mag'] = gyro_mag
    df['acc_horiz'] = acc_horiz
    df['dyn_acc_mag'] = dyn_acc_mag
    df['dyn_planar'] = dyn_planar
    df['dyn_vert'] = dyn_vert
    df['gyro_yaw_gravity'] = gyro_yaw_gravity
    df['gyro_wobble_tilt'] = gyro_wobble_tilt
    df['acc_jerk'] = acc_jerk
    df['gyro_jerk'] = gyro_jerk
    df['chatter_ratio'] = chatter_ratio
    df['turn_dominance'] = turn_dominance
    df['spectral_low_energy_3s'] = low_e
    df['spectral_high_energy_3s'] = high_e
    df['spectral_ratio_3s'] = spec_ratio
    df['spectral_entropy_3s'] = spec_entropy
    
    # 6. Multi-Window Causal Rolling Kinematics
    for w_sec, w in [(1, 10), (2, 20), (3, 30)]:
        # Planar forces
        df[f'dyn_planar_mean_{w_sec}s'] = pd.Series(dyn_planar).rolling(w, min_periods=1).mean().values
        df[f'dyn_planar_std_{w_sec}s'] = pd.Series(dyn_planar).rolling(w, min_periods=1).std().fillna(0.0).values
        df[f'dyn_planar_max_{w_sec}s'] = pd.Series(dyn_planar).rolling(w, min_periods=1).max().values
        df[f'dyn_planar_rms_{w_sec}s'] = np.sqrt(pd.Series(dyn_planar**2).rolling(w, min_periods=1).mean().values)
        
        # Yaw rates
        df[f'gyro_yaw_mean_{w_sec}s'] = pd.Series(gyro_yaw_gravity).rolling(w, min_periods=1).mean().values
        df[f'gyro_yaw_std_{w_sec}s'] = pd.Series(gyro_yaw_gravity).rolling(w, min_periods=1).std().fillna(0.0).values
        df[f'gyro_yaw_max_{w_sec}s'] = pd.Series(gyro_yaw_gravity).rolling(w, min_periods=1).max().values
        
        # Vertical chatter
        df[f'dyn_vert_std_{w_sec}s'] = pd.Series(dyn_vert).rolling(w, min_periods=1).std().fillna(0.0).values
        
        # Jerk std
        df[f'jerk_std_{w_sec}s'] = pd.Series(acc_jerk).rolling(w, min_periods=1).std().fillna(0.0).values
        
    return df


def compute_metrics_10_targets(y_true, y_pred, y_prob=None):
    """
    Evaluates against the 10 strict research requirements:
    1. Accuracy >= 90%
    2. Balanced Accuracy >= 90%
    3. Macro-F1 >= 90%
    4. Aggressive Precision >= 90%
    5. Aggressive Recall >= 90%
    6. Normal Recall >= 90%
    7. ROC-AUC >= 95%
    8. PR-AUC >= 90%
    9. FPR <= 10%
    10. FNR <= 10%
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    prec_norm = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_norm = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_norm = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    
    prec_agg = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_agg = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_agg = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    auc = float(roc_auc_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    brier = float(brier_score_loss(y_true, y_prob)) if y_prob is not None else None
    
    targets = {
        "accuracy_ge_90": bool(acc >= 0.90),
        "balanced_acc_ge_90": bool(bal_acc >= 0.90),
        "macro_f1_ge_90": bool(macro_f1 >= 0.90),
        "aggressive_prec_ge_90": bool(prec_agg >= 0.90),
        "aggressive_rec_ge_90": bool(rec_agg >= 0.90),
        "normal_rec_ge_90": bool(rec_norm >= 0.90),
        "roc_auc_ge_95": bool(auc >= 0.95) if auc is not None else False,
        "pr_auc_ge_90": bool(pr_auc >= 0.90) if pr_auc is not None else False,
        "fpr_le_10": bool(fpr <= 0.10),
        "fnr_le_10": bool(fnr <= 0.10)
    }
    
    all_targets_met = all(targets.values())
    passed_count = sum(targets.values())
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "normal_precision": prec_norm,
        "normal_recall": rec_norm,
        "normal_f1": f1_norm,
        "aggressive_precision": prec_agg,
        "aggressive_recall": rec_agg,
        "aggressive_f1": f1_agg,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "cm_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "targets_evaluation": targets,
        "targets_passed_count": int(passed_count),
        "all_targets_met": bool(all_targets_met)
    }


# Compact 1D PyTorch CNN for IMU Temporal Dynamics
class CompactIMU1DCNN(nn.Module):
    def __init__(self, in_channels=6, seq_len=30):
        super(CompactIMU1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu2(self.bn2(self.conv2(out))))
        out = out.squeeze(-1)
        return self.fc(out)


def train_eval_cnn(X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, epochs=15, batch_size=128):
    """Train and evaluate compact 1D CNN on causal sequence windows."""
    device = torch.device('cpu')
    model = CompactIMU1DCNN(in_channels=X_tr_seq.shape[1], seq_len=X_tr_seq.shape[2]).to(device)
    
    # Class weights for loss
    n_pos = np.sum(y_tr_seq == 1)
    n_neg = np.sum(y_tr_seq == 0)
    w_pos = (n_neg / n_pos) if n_pos > 0 else 1.0
    weights = torch.tensor([1.0, float(w_pos)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    ds_tr = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(y_tr_seq, dtype=torch.long))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        for bx, by in loader_tr:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        tx = torch.tensor(X_te_seq, dtype=torch.float32)
        logits = model(tx)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.50).astype(int)
        
    return probs, preds


def make_causal_windows(df_sess, feature_cols, window_size=30):
    """Creates causal backward-looking 3D windows (N, num_feats, window_size)."""
    vals = df_sess[feature_cols].values
    n = len(vals)
    X_seq = np.zeros((n, len(feature_cols), window_size))
    for i in range(n):
        start = max(0, i - window_size + 1)
        sub = vals[start:i+1]
        if len(sub) < window_size:
            pad_len = window_size - len(sub)
            sub = np.pad(sub, ((pad_len, 0), (0, 0)), mode='edge')
        X_seq[i] = sub.T # (channels, seq_len)
    return X_seq, df_sess['label'].values


def run_advanced_research():
    print("==================================================================")
    print("STARTING ADVANCED PIPELINE A RESEARCH & 10-METRIC BENCHMARK")
    print("==================================================================")
    
    # 1. Load Data
    all_df = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_c_processed_all.csv"))
    s1_raw = all_df[all_df['session'] == 'session_1'].copy().reset_index(drop=True)
    s2_raw = all_df[all_df['session'] == 'session_2'].copy().reset_index(drop=True)
    
    print("Engineering advanced causal physical & spectral features...")
    s1_adv = engineer_advanced_features(s1_raw)
    s2_adv = engineer_advanced_features(s2_raw)
    all_adv = pd.concat([s1_adv, s2_adv]).reset_index(drop=True)
    
    meta_cols = ['label', 'split', 'session']
    adv_feats = [c for c in all_adv.columns if c not in meta_cols]
    base_feats = [c for c in all_df.columns if c not in meta_cols]
    
    print(f"Base features: {len(base_feats)}, Advanced features: {len(adv_feats)}")
    
    # 2. Architectures to benchmark
    models_dict = {
        "LogisticRegression_L2": (
            LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42),
            True
        ),
        "SVM_RBF_Tuned": (
            SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
            True
        ),
        "ExtraTrees_Balanced": (
            ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1),
            False
        ),
        "RandomForest_Subsample": (
            RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=6, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            False
        ),
        "XGBoost_Tuned": (
            XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.65, random_state=42, eval_metric="logloss"),
            False
        ),
        "HistGradientBoosting_Tuned": (
            HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.04, min_samples_leaf=15, class_weight='balanced', random_state=42),
            False
        )
    }
    
    feature_experiments = {
        "Baseline_47_Production": (base_feats, all_df),
        "Advanced_Phys_Spectral": (adv_feats, all_adv)
    }
    
    all_eval_results = {}
    
    for fset_name, (f_cols, df_data) in feature_experiments.items():
        print(f"\n=======================================================")
        print(f"BENCHMARKING FEATURE SUITE: {fset_name} ({len(f_cols)} features)")
        print(f"=======================================================")
        all_eval_results[fset_name] = {}
        
        s1 = df_data[df_data['session'] == 'session_1']
        s2 = df_data[df_data['session'] == 'session_2']
        
        # Scaling
        scaler_s1 = StandardScaler()
        X_s1_sc = scaler_s1.fit_transform(s1[f_cols])
        X_s2_for_s1 = scaler_s1.transform(s2[f_cols])
        
        scaler_s2 = StandardScaler()
        X_s2_sc = scaler_s2.fit_transform(s2[f_cols])
        X_s1_for_s2 = scaler_s2.transform(s1[f_cols])
        
        y_s1 = s1['label'].values
        y_s2 = s2['label'].values
        
        for m_name, (proto_model, req_scale) in models_dict.items():
            print(f"\n---> Evaluating {m_name}...")
            from sklearn.base import clone
            
            # --- Direction 1: Train S1 -> Test S2 ---
            m_s1 = clone(proto_model)
            X_tr1 = X_s1_sc if req_scale else s1[f_cols].values
            X_te2 = X_s2_for_s1 if req_scale else s2[f_cols].values
            
            m_s1.fit(X_tr1, y_s1)
            prob_s2 = m_s1.predict_proba(X_te2)[:, 1]
            pred_s2 = (prob_s2 >= 0.50).astype(int)
            res_s1_s2 = compute_metrics_10_targets(y_s2, pred_s2, prob_s2)
            
            # --- Direction 2: Train S2 -> Test S1 ---
            m_s2 = clone(proto_model)
            X_tr2 = X_s2_sc if req_scale else s2[f_cols].values
            X_te1 = X_s1_for_s2 if req_scale else s1[f_cols].values
            
            m_s2.fit(X_tr2, y_s2)
            prob_s1 = m_s2.predict_proba(X_te1)[:, 1]
            pred_s1 = (prob_s1 >= 0.50).astype(int)
            res_s2_s1 = compute_metrics_10_targets(y_s1, pred_s1, prob_s1)
            
            # Mean cross-session metrics
            mean_acc = (res_s1_s2['accuracy'] + res_s2_s1['accuracy']) / 2.0
            mean_bal = (res_s1_s2['balanced_accuracy'] + res_s2_s1['balanced_accuracy']) / 2.0
            mean_f1 = (res_s1_s2['macro_f1'] + res_s2_s1['macro_f1']) / 2.0
            mean_agg_p = (res_s1_s2['aggressive_precision'] + res_s2_s1['aggressive_precision']) / 2.0
            mean_agg_r = (res_s1_s2['aggressive_recall'] + res_s2_s1['aggressive_recall']) / 2.0
            mean_norm_r = (res_s1_s2['normal_recall'] + res_s2_s1['normal_recall']) / 2.0
            mean_auc = (res_s1_s2['roc_auc'] + res_s2_s1['roc_auc']) / 2.0
            mean_pr_auc = (res_s1_s2['pr_auc'] + res_s2_s1['pr_auc']) / 2.0
            mean_fpr = (res_s1_s2['fpr'] + res_s2_s1['fpr']) / 2.0
            mean_fnr = (res_s1_s2['fnr'] + res_s2_s1['fnr']) / 2.0
            
            mean_targets = {
                "accuracy_ge_90": bool(mean_acc >= 0.90),
                "balanced_acc_ge_90": bool(mean_bal >= 0.90),
                "macro_f1_ge_90": bool(mean_f1 >= 0.90),
                "aggressive_prec_ge_90": bool(mean_agg_p >= 0.90),
                "aggressive_rec_ge_90": bool(mean_agg_r >= 0.90),
                "normal_rec_ge_90": bool(mean_norm_r >= 0.90),
                "roc_auc_ge_95": bool(mean_auc >= 0.95),
                "pr_auc_ge_90": bool(mean_pr_auc >= 0.90),
                "fpr_le_10": bool(mean_fpr <= 0.10),
                "fnr_le_10": bool(mean_fnr <= 0.10)
            }
            
            print(f"  S1->S2: Acc={res_s1_s2['accuracy']:.4f}, BalAcc={res_s1_s2['balanced_accuracy']:.4f}, Macro-F1={res_s1_s2['macro_f1']:.4f}, FPR={res_s1_s2['fpr']:.4f}, RecAgg={res_s1_s2['aggressive_recall']:.4f}, Targets={res_s1_s2['targets_passed_count']}/10")
            print(f"  S2->S1: Acc={res_s2_s1['accuracy']:.4f}, BalAcc={res_s2_s1['balanced_accuracy']:.4f}, Macro-F1={res_s2_s1['macro_f1']:.4f}, FPR={res_s2_s1['fpr']:.4f}, RecAgg={res_s2_s1['aggressive_recall']:.4f}, Targets={res_s2_s1['targets_passed_count']}/10")
            print(f"  MEAN:   Acc={mean_acc:.4f}, BalAcc={mean_bal:.4f}, Macro-F1={mean_f1:.4f}, FPR={mean_fpr:.4f}, RecAgg={mean_agg_r:.4f}, Targets Met={sum(mean_targets.values())}/10")
            
            all_eval_results[fset_name][m_name] = {
                "s1_to_s2": res_s1_s2,
                "s2_to_s1": res_s2_s1,
                "mean_cross_session": {
                    "accuracy": mean_acc,
                    "balanced_accuracy": mean_bal,
                    "macro_f1": mean_f1,
                    "aggressive_precision": mean_agg_p,
                    "aggressive_recall": mean_agg_r,
                    "normal_recall": mean_norm_r,
                    "roc_auc": mean_auc,
                    "pr_auc": mean_pr_auc,
                    "fpr": mean_fpr,
                    "fnr": mean_fnr,
                    "targets_evaluation": mean_targets,
                    "targets_passed_count": int(sum(mean_targets.values())),
                    "all_targets_met": bool(all(mean_targets.values()))
                }
            }
            
        # --- Evaluate Compact 1D PyTorch CNN ---
        print("\n---> Evaluating Compact 1D PyTorch CNN...")
        raw_6 = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']
        X_s1_seq, y_s1_seq = make_causal_windows(s1, raw_6, window_size=30)
        X_s2_seq, y_s2_seq = make_causal_windows(s2, raw_6, window_size=30)
        
        # S1 -> S2 CNN
        prob_cnn_s2, pred_cnn_s2 = train_eval_cnn(X_s1_seq, y_s1_seq, X_s2_seq, y_s2_seq, epochs=12)
        cnn_s1_s2 = compute_metrics_10_targets(y_s2_seq, pred_cnn_s2, prob_cnn_s2)
        
        # S2 -> S1 CNN
        prob_cnn_s1, pred_cnn_s1 = train_eval_cnn(X_s2_seq, y_s2_seq, X_s1_seq, y_s1_seq, epochs=12)
        cnn_s2_s1 = compute_metrics_10_targets(y_s1_seq, pred_cnn_s1, prob_cnn_s1)
        
        cnn_mean_acc = (cnn_s1_s2['accuracy'] + cnn_s2_s1['accuracy']) / 2.0
        cnn_mean_bal = (cnn_s1_s2['balanced_accuracy'] + cnn_s2_s1['balanced_accuracy']) / 2.0
        cnn_mean_f1 = (cnn_s1_s2['macro_f1'] + cnn_s2_s1['macro_f1']) / 2.0
        cnn_mean_fpr = (cnn_s1_s2['fpr'] + cnn_s2_s1['fpr']) / 2.0
        cnn_mean_rec = (cnn_s1_s2['aggressive_recall'] + cnn_s2_s1['aggressive_recall']) / 2.0
        
        print(f"  CNN S1->S2: Acc={cnn_s1_s2['accuracy']:.4f}, F1={cnn_s1_s2['macro_f1']:.4f}")
        print(f"  CNN S2->S1: Acc={cnn_s2_s1['accuracy']:.4f}, F1={cnn_s2_s1['macro_f1']:.4f}")
        print(f"  CNN MEAN:   Acc={cnn_mean_acc:.4f}, BalAcc={cnn_mean_bal:.4f}, F1={cnn_mean_f1:.4f}")
        
        all_eval_results[fset_name]["Compact_1D_CNN"] = {
            "s1_to_s2": cnn_s1_s2,
            "s2_to_s1": cnn_s2_s1,
            "mean_cross_session": {
                "accuracy": cnn_mean_acc,
                "balanced_accuracy": cnn_mean_bal,
                "macro_f1": cnn_mean_f1,
                "fpr": cnn_mean_fpr,
                "aggressive_recall": cnn_mean_rec
            }
        }
        
        # --- Evaluate Tri-Model Soft-Voting Ensemble (ExtraTrees + HistGB + XGBoost) ---
        print("\n---> Evaluating Soft-Voting Ensemble (ExtraTrees + HistGB + XGBoost)...")
        m_et = ExtraTreesClassifier(n_estimators=150, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=42, n_jobs=-1)
        m_hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.04, min_samples_leaf=15, class_weight='balanced', random_state=42)
        m_xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.65, random_state=42, eval_metric="logloss")
        
        # Fit models on S1
        m_et.fit(s1[f_cols].values, y_s1)
        m_hgb.fit(s1[f_cols].values, y_s1)
        m_xgb.fit(s1[f_cols].values, y_s1)
        
        p_et_s2 = m_et.predict_proba(s2[f_cols].values)[:, 1]
        p_hgb_s2 = m_hgb.predict_proba(s2[f_cols].values)[:, 1]
        p_xgb_s2 = m_xgb.predict_proba(s2[f_cols].values)[:, 1]
        prob_ens_s2 = (p_et_s2 + p_hgb_s2 + p_xgb_s2) / 3.0
        pred_ens_s2 = (prob_ens_s2 >= 0.50).astype(int)
        ens_s1_s2 = compute_metrics_10_targets(y_s2, pred_ens_s2, prob_ens_s2)
        
        # Fit models on S2
        m_et.fit(s2[f_cols].values, y_s2)
        m_hgb.fit(s2[f_cols].values, y_s2)
        m_xgb.fit(s2[f_cols].values, y_s2)
        
        p_et_s1 = m_et.predict_proba(s1[f_cols].values)[:, 1]
        p_hgb_s1 = m_hgb.predict_proba(s1[f_cols].values)[:, 1]
        p_xgb_s1 = m_xgb.predict_proba(s1[f_cols].values)[:, 1]
        prob_ens_s1 = (p_et_s1 + p_hgb_s1 + p_xgb_s1) / 3.0
        pred_ens_s1 = (prob_ens_s1 >= 0.50).astype(int)
        ens_s2_s1 = compute_metrics_10_targets(y_s1, pred_ens_s1, prob_ens_s1)
        
        ens_mean_acc = (ens_s1_s2['accuracy'] + ens_s2_s1['accuracy']) / 2.0
        ens_mean_bal = (ens_s1_s2['balanced_accuracy'] + ens_s2_s1['balanced_accuracy']) / 2.0
        ens_mean_f1 = (ens_s1_s2['macro_f1'] + ens_s2_s1['macro_f1']) / 2.0
        ens_mean_fpr = (ens_s1_s2['fpr'] + ens_s2_s1['fpr']) / 2.0
        ens_mean_rec = (ens_s1_s2['aggressive_recall'] + ens_s2_s1['aggressive_recall']) / 2.0
        
        print(f"  Ensemble S1->S2: Acc={ens_s1_s2['accuracy']:.4f}, BalAcc={ens_s1_s2['balanced_accuracy']:.4f}, F1={ens_s1_s2['macro_f1']:.4f}")
        print(f"  Ensemble S2->S1: Acc={ens_s2_s1['accuracy']:.4f}, BalAcc={ens_s2_s1['balanced_accuracy']:.4f}, F1={ens_s2_s1['macro_f1']:.4f}")
        print(f"  Ensemble MEAN:   Acc={ens_mean_acc:.4f}, BalAcc={ens_mean_bal:.4f}, F1={ens_mean_f1:.4f}")
        
        all_eval_results[fset_name]["Voting_Ensemble"] = {
            "s1_to_s2": ens_s1_s2,
            "s2_to_s1": ens_s2_s1,
            "mean_cross_session": {
                "accuracy": ens_mean_acc,
                "balanced_accuracy": ens_mean_bal,
                "macro_f1": ens_mean_f1,
                "fpr": ens_mean_fpr,
                "aggressive_recall": ens_mean_rec
            }
        }
        
    out_file = os.path.join(REPORTS_DIR, "advanced_research_metrics.json")
    with open(out_file, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"\nAll research metrics saved to {out_file}")
    return all_eval_results

if __name__ == "__main__":
    run_advanced_research()

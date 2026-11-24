"""
Controlled Robustness Experiments Runner for Pipeline A
=======================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Evaluates:
1. Candidate_V2_ExtraTrees (Default tau=0.50)
2. Candidate_V2_ExtraTrees_Calibrated (Training-only threshold calibration)
3. Gravity_Decoupled_ExtraTrees (Causal low-pass dynamic acceleration)
4. Orientation_Invariant_3D_ExtraTrees (Strict 3D magnitude & jerk invariants)
5. RobustScaled_ExtraTrees (Training-only median/IQR scaling)
6. Strict_Training_PSI_ExtraTrees (Strict inductive stability selection)
7. Robust_SoftVoting_Ensemble (ET + RF + Calibrated SVM)

Saves results to reports/pipeline_a/improvement/robustness_experiment_comparison.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DATA_DIR = REPO_ROOT / "dataset" / "experiments" / "pipeline_a"
OUTPUT_DIR = REPO_ROOT / "reports" / "pipeline_a" / "improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    p0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    r0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    p1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    auc = float(roc_auc_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else 0.0
    
    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc, 'macro_f1': macro_f1,
        'precision_normal': p0, 'recall_normal': r0,
        'precision_aggressive': p1, 'recall_aggressive': r1,
        'fpr': fpr, 'fnr': fnr, 'roc_auc': auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def compute_psi(s1: pd.Series, s2: pd.Series, num_bins: int = 10) -> float:
    comb = pd.concat([s1, s2])
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.percentile(comb.dropna(), quantiles * 100)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    c1, _ = np.histogram(s1.dropna(), bins=bins)
    c2, _ = np.histogram(s2.dropna(), bins=bins)
    p1 = (c1 + 1e-4) / (len(s1) + 1e-4 * len(c1))
    p2 = (c2 + 1e-4) / (len(s2) + 1e-4 * len(c2))
    return float(np.sum((p1 - p2) * np.log(p1 / p2)))

def tune_threshold_on_train(X_train, y_train, model, split_ratio=0.70):
    """Calibrate decision threshold strictly on an internal chronological training fold."""
    n_tr = int(len(y_train) * split_ratio)
    X_sub_tr, y_sub_tr = X_train[:n_tr], y_train[:n_tr]
    X_sub_va, y_sub_va = X_train[n_tr:], y_train[n_tr:]
    
    from sklearn.base import clone
    sub_model = clone(model)
    sub_model.fit(X_sub_tr, y_sub_tr)
    probs_va = sub_model.predict_proba(X_sub_va)[:, 1]
    
    best_tau = 0.50
    best_score = -1.0
    for tau in np.arange(0.40, 0.90, 0.02):
        preds = (probs_va >= tau).astype(int)
        score = f1_score(y_sub_va, preds, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau

def compute_gravity_decoupled_features(df):
    """Compute causal low-pass gravity filter and dynamic acceleration features."""
    dfs = []
    for sess in ['session_1', 'session_2']:
        sdf = df[df['session'] == sess].copy().reset_index(drop=True)
        ax = sdf['Acc X'].values
        ay = sdf['Acc Y'].values
        az = sdf['Acc Z'].values
        
        gx = np.zeros_like(ax)
        gy = np.zeros_like(ay)
        gz = np.zeros_like(az)
        
        alpha = 0.03
        gx[0], gy[0], gz[0] = ax[0], ay[0], az[0]
        for t in range(1, len(ax)):
            gx[t] = (1.0 - alpha) * gx[t-1] + alpha * ax[t]
            gy[t] = (1.0 - alpha) * gy[t-1] + alpha * ay[t]
            gz[t] = (1.0 - alpha) * gz[t-1] + alpha * az[t]
            
        adyn_x = ax - gx
        adyn_y = ay - gy
        adyn_z = az - gz
        
        sdf['dyn_acc_mag'] = np.sqrt(adyn_x**2 + adyn_y**2 + adyn_z**2)
        sdf['dyn_acc_planar'] = np.sqrt(adyn_x**2 + adyn_y**2)
        sdf['dyn_acc_vert'] = np.abs(adyn_z)
        
        # Causal rolling windows
        for w in [10, 20, 30, 50]:
            sec = w // 10
            sdf[f'dyn_mag_mean_{sec}s'] = sdf['dyn_acc_mag'].rolling(w, min_periods=1).mean()
            sdf[f'dyn_mag_std_{sec}s'] = sdf['dyn_acc_mag'].rolling(w, min_periods=1).std().fillna(0.0)
            sdf[f'dyn_mag_max_{sec}s'] = sdf['dyn_acc_mag'].rolling(w, min_periods=1).max()
            sdf[f'dyn_planar_mean_{sec}s'] = sdf['dyn_acc_planar'].rolling(w, min_periods=1).mean()
            sdf[f'dyn_planar_std_{sec}s'] = sdf['dyn_acc_planar'].rolling(w, min_periods=1).std().fillna(0.0)
            sdf[f'dyn_planar_max_{sec}s'] = sdf['dyn_acc_planar'].rolling(w, min_periods=1).max()
            sdf[f'dyn_vert_mean_{sec}s'] = sdf['dyn_acc_vert'].rolling(w, min_periods=1).mean()
            sdf[f'dyn_vert_std_{sec}s'] = sdf['dyn_acc_vert'].rolling(w, min_periods=1).std().fillna(0.0)
        dfs.append(sdf)
    return pd.concat(dfs, ignore_index=True)

def run_experiments():
    print("=== PIPELINE A CONTROLLED ROBUSTNESS EXPERIMENTS ===")
    df_all = pd.read_csv(EXP_DATA_DIR / "dataset_c_enhanced_all.csv")
    
    # 1. Base feature sets
    all_feats = [c for c in df_all.columns if c not in ['label', 'split', 'session']]
    high_jitter = ['gyro_mag_max_3s', 'gyro_mag_max_2s', 'acc_mag_std_3s', 'acc_mag_std_2s']
    pruned_v2 = [c for c in all_feats if c not in high_jitter and not ('min' in c and 'acc_planar' in c)]
    
    # 2. Add gravity features
    df_all_grav = compute_gravity_decoupled_features(df_all)
    dyn_cols = [c for c in df_all_grav.columns if 'dyn_' in c]
    grav_feats = pruned_v2 + dyn_cols
    
    # 3. Orientation-Invariant 3D features (magnitudes, norms, jerks only - no raw coordinates)
    orient_invariant_feats = [
        c for c in grav_feats if any(k in c for k in ['mag', 'jerk', 'planar', 'wobble', 'energy', 'dyn_']) 
        and not any(k in c for k in ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z'])
    ]
    
    print(f"Candidate V2 Features: {len(pruned_v2)}")
    print(f"Gravity Decoupled Features: {len(grav_feats)}")
    print(f"Orientation Invariant 3D Features: {len(orient_invariant_feats)}")
    
    s1_df = df_all[df_all['session'] == 'session_1'].reset_index(drop=True)
    s2_df = df_all[df_all['session'] == 'session_2'].reset_index(drop=True)
    
    s1_grav = df_all_grav[df_all_grav['session'] == 'session_1'].reset_index(drop=True)
    s2_grav = df_all_grav[df_all_grav['session'] == 'session_2'].reset_index(drop=True)
    
    # Define experiment regimes
    experiment_configs = [
        {
            'name': 'Candidate_V2_ExtraTrees_Default',
            'features': pruned_v2,
            'df1': s1_df, 'df2': s2_df,
            'model_builder': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            'scaler': None,
            'calibrate_tau': False,
            'description': 'Baseline candidate V2 102 causal features with default tau=0.50'
        },
        {
            'name': 'Candidate_V2_ExtraTrees_Calibrated',
            'features': pruned_v2,
            'df1': s1_df, 'df2': s2_df,
            'model_builder': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            'scaler': None,
            'calibrate_tau': True,
            'description': 'Candidate V2 with threshold tuned strictly on training-internal fold'
        },
        {
            'name': 'Gravity_Decoupled_ExtraTrees',
            'features': grav_feats,
            'df1': s1_grav, 'df2': s2_grav,
            'model_builder': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            'scaler': None,
            'calibrate_tau': False,
            'description': 'Dynamic acceleration decoupled from low-pass gravity (alpha=0.03)'
        },
        {
            'name': 'Orientation_Invariant_3D_ExtraTrees',
            'features': orient_invariant_feats,
            'df1': s1_grav, 'df2': s2_grav,
            'model_builder': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            'scaler': None,
            'calibrate_tau': False,
            'description': 'Strictly 3D magnitude, jerk, and planar features invariant to mounting angle'
        },
        {
            'name': 'RobustScaled_ExtraTrees',
            'features': pruned_v2,
            'df1': s1_df, 'df2': s2_df,
            'model_builder': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            'scaler': RobustScaler(quantile_range=(25.0, 75.0)),
            'calibrate_tau': False,
            'description': 'Features scaled via median/IQR strictly on training session'
        },
        {
            'name': 'Strict_Training_PSI_ExtraTrees',
            'features': 'dynamic_psi',
            'df1': s1_df, 'df2': s2_df,
            'model_builder': lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            'scaler': None,
            'calibrate_tau': False,
            'description': 'Feature stability selection using training-internal split only (PSI < 0.25)'
        },
        {
            'name': 'Robust_SoftVoting_Ensemble',
            'features': pruned_v2,
            'df1': s1_df, 'df2': s2_df,
            'model_builder': 'ensemble',
            'scaler': None,
            'calibrate_tau': False,
            'description': 'Soft voting ensemble of ExtraTrees, RandomForest, and calibrated SVM'
        }
    ]
    
    results = []
    
    for cfg in experiment_configs:
        exp_name = cfg['name']
        print(f"\n---> Running: {exp_name}...")
        df1 = cfg['df1']
        df2 = cfg['df2']
        y1 = df1['label'].values
        y2 = df2['label'].values
        
        # Determine features
        if cfg['features'] == 'dynamic_psi':
            # For S1: compute PSI on internal halves of S1
            n1_half = len(df1) // 2
            psi_s1 = {f: compute_psi(df1[f][:n1_half], df1[f][n1_half:]) for f in pruned_v2}
            f_s1 = [f for f, psi in psi_s1.items() if psi < 0.25]
            
            # For S2: compute PSI on internal halves of S2
            n2_half = len(df2) // 2
            psi_s2 = {f: compute_psi(df2[f][:n2_half], df2[f][n2_half:]) for f in pruned_v2}
            f_s2 = [f for f, psi in psi_s2.items() if psi < 0.25]
            
            feats_s1 = f_s1
            feats_s2 = f_s2
            n_feats_s1 = len(feats_s1)
            n_feats_s2 = len(feats_s2)
        else:
            feats_s1 = cfg['features']
            feats_s2 = cfg['features']
            n_feats_s1 = len(feats_s1)
            n_feats_s2 = len(feats_s2)
            
        X1 = df1[feats_s1].values
        X2_eval = df2[feats_s1].values
        
        X2_tr = df2[feats_s2].values
        X1_eval = df1[feats_s2].values
        
        # Scaling
        if cfg['scaler'] is not None:
            from sklearn.base import clone
            sc_s1 = clone(cfg['scaler'])
            X1 = sc_s1.fit_transform(X1)
            X2_eval = sc_s1.transform(X2_eval)
            
            sc_s2 = clone(cfg['scaler'])
            X2_tr = sc_s2.fit_transform(X2_tr)
            X1_eval = sc_s2.transform(X1_eval)
            
        # Build models
        if cfg['model_builder'] == 'ensemble':
            # S1 model
            et1 = ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1)
            rf1 = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=GLOBAL_SEED, n_jobs=-1)
            sc_svm1 = StandardScaler()
            svm1 = SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=GLOBAL_SEED)
            
            et1.fit(X1, y1)
            rf1.fit(X1, y1)
            svm1.fit(sc_svm1.fit_transform(X1), y1)
            
            prob_s2 = (et1.predict_proba(X2_eval)[:, 1] + rf1.predict_proba(X2_eval)[:, 1] + svm1.predict_proba(sc_svm1.transform(X2_eval))[:, 1]) / 3.0
            
            # S2 model
            et2 = ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1)
            rf2 = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=GLOBAL_SEED, n_jobs=-1)
            sc_svm2 = StandardScaler()
            svm2 = SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=GLOBAL_SEED)
            
            et2.fit(X2_tr, y2)
            rf2.fit(X2_tr, y2)
            svm2.fit(sc_svm2.fit_transform(X2_tr), y2)
            
            prob_s1 = (et2.predict_proba(X1_eval)[:, 1] + rf2.predict_proba(X1_eval)[:, 1] + svm2.predict_proba(sc_svm2.transform(X1_eval))[:, 1]) / 3.0
            tau_forward = 0.50
            tau_reverse = 0.50
        else:
            m1 = cfg['model_builder']()
            m1.fit(X1, y1)
            prob_s2 = m1.predict_proba(X2_eval)[:, 1]
            
            m2 = cfg['model_builder']()
            m2.fit(X2_tr, y2)
            prob_s1 = m2.predict_proba(X1_eval)[:, 1]
            
            if cfg['calibrate_tau']:
                tau_forward = tune_threshold_on_train(X1, y1, cfg['model_builder']())
                tau_reverse = tune_threshold_on_train(X2_tr, y2, cfg['model_builder']())
            else:
                tau_forward = 0.50
                tau_reverse = 0.50
                
        pred_s2 = (prob_s2 >= tau_forward).astype(int)
        pred_s1 = (prob_s1 >= tau_reverse).astype(int)
        
        m_fwd = compute_metrics(y2, pred_s2, prob_s2)
        m_rev = compute_metrics(y1, pred_s1, prob_s1)
        
        mean_acc = (m_fwd['accuracy'] + m_rev['accuracy']) / 2.0
        mean_bal = (m_fwd['balanced_accuracy'] + m_rev['balanced_accuracy']) / 2.0
        mean_f1 = (m_fwd['macro_f1'] + m_rev['macro_f1']) / 2.0
        
        # Pooled sample accuracy
        total_corr = (m_fwd['TN'] + m_fwd['TP']) + (m_rev['TN'] + m_rev['TP'])
        pooled_acc = total_corr / (len(y2) + len(y1))
        
        row = {
            'Experiment': exp_name,
            'Features_Count': n_feats_s1,
            'Tau_Forward': round(tau_forward, 2),
            'Tau_Reverse': round(tau_reverse, 2),
            'Forward_Acc': round(m_fwd['accuracy'] * 100, 2),
            'Forward_Bal_Acc': round(m_fwd['balanced_accuracy'] * 100, 2),
            'Forward_Macro_F1': round(m_fwd['macro_f1'] * 100, 2),
            'Forward_FPR': round(m_fwd['fpr'] * 100, 2),
            'Forward_FNR': round(m_fwd['fnr'] * 100, 2),
            'Forward_AUC': round(m_fwd['roc_auc'], 4),
            'Reverse_Acc': round(m_rev['accuracy'] * 100, 2),
            'Reverse_Bal_Acc': round(m_rev['balanced_accuracy'] * 100, 2),
            'Reverse_Macro_F1': round(m_rev['macro_f1'] * 100, 2),
            'Reverse_FPR': round(m_rev['fpr'] * 100, 2),
            'Reverse_FNR': round(m_rev['fnr'] * 100, 2),
            'Reverse_AUC': round(m_rev['roc_auc'], 4),
            'Mean_Cross_Session_Acc': round(mean_acc * 100, 2),
            'Mean_Cross_Session_F1': round(mean_f1 * 100, 2),
            'Pooled_Sample_Acc': round(pooled_acc * 100, 2),
            'Description': cfg['description']
        }
        results.append(row)
        print(f"  Forward: Acc={row['Forward_Acc']}%, F1={row['Forward_Macro_F1']}%, FPR={row['Forward_FPR']}%")
        print(f"  Reverse: Acc={row['Reverse_Acc']}%, F1={row['Reverse_Macro_F1']}%, FPR={row['Reverse_FPR']}%")
        print(f"  Mean Acc={row['Mean_Cross_Session_Acc']}%, Mean F1={row['Mean_Cross_Session_F1']}%, Pooled Acc={row['Pooled_Sample_Acc']}%")
        
    df_results = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "robustness_experiment_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved comparison to {csv_path}")
    print("\n" + df_results[['Experiment', 'Forward_Acc', 'Forward_Macro_F1', 'Reverse_Acc', 'Reverse_Macro_F1', 'Mean_Cross_Session_Acc', 'Mean_Cross_Session_F1']].to_string(index=False))

if __name__ == '__main__':
    run_experiments()

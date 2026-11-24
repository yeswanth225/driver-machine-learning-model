"""
Script: scripts/run_pipeline_a_improvement_experiments.py
Master experimental runner for Pipeline A improvements across all validation regimes.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "pipeline_a" / "improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42

def evaluate_predictions(y_true, y_pred, y_prob=None):
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    p0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    r0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    p1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    auc = float(roc_auc_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if (y_prob is not None and len(np.unique(y_true)) > 1) else None

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision_normal': p0,
        'recall_normal': r0,
        'f1_normal': f1_0,
        'precision_aggressive': p1,
        'recall_aggressive': r1,
        'f1_aggressive': f1_1,
        'fpr': fpr,
        'fnr': fnr,
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist()
    }

def benchmark_latency(model, X_sample, n_runs=300):
    """Measure single-sample CPU inference latency with warm-up."""
    # Warmup
    for _ in range(20):
        _ = model.predict_proba(X_sample[:1])
    
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_sample[:1])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # in ms
    
    return {
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'mean_ms': float(np.mean(times))
    }

class SoftVotingEnsemble:
    _estimator_type = "classifier"
    def __init__(self, estimators=None):
        self.estimators = estimators or []
        self.classes_ = np.array([0, 1])

    def get_params(self, deep=True):
        return {'estimators': self.estimators}

    def set_params(self, **params):
        if 'estimators' in params:
            self.estimators = params['estimators']
        return self

    def fit(self, X, y):
        from sklearn.base import clone
        self.fitted_estimators_ = [(name, clone(est).fit(X, y)) for name, est in self.estimators]
        return self

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for name, est in self.fitted_estimators_]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.50).astype(int)

def get_candidate_models():
    return {
        'Logistic_Regression': (
            LogisticRegression(max_iter=500, class_weight='balanced', random_state=GLOBAL_SEED),
            True
        ),
        'SVM_RBF_Tuned': (
            SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=GLOBAL_SEED),
            True
        ),
        'Random_Forest': (
            RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced_subsample', random_state=GLOBAL_SEED, n_jobs=-1),
            False
        ),
        'Extra_Trees': (
            ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_split=6, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1),
            False
        ),
        'XGBoost': (
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=GLOBAL_SEED, eval_metric='logloss'),
            False
        ),
        'HistGradientBoosting': (
            HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=GLOBAL_SEED),
            False
        ),
        'Voting_Ensemble': (
            SoftVotingEnsemble(
                estimators=[
                    ('svm', SVC(C=1.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=GLOBAL_SEED)),
                    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=GLOBAL_SEED, n_jobs=-1)),
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=GLOBAL_SEED, n_jobs=-1))
                ]
            ),
            True # Standardize for SVM component
        )
    }

def run_all_experiments():
    print("Loading datasets...")
    # Baseline 47 Features
    df_base_all = pd.read_csv(REPO_ROOT / 'dataset/processed/dataset_c_processed_all.csv')
    base_feats = [c for c in df_base_all.columns if c not in ['label', 'session', 'split']]
    
    # Enhanced Features (Candidate V2 & Domain Stable)
    enh_all_path = REPO_ROOT / 'dataset/experiments/pipeline_a/dataset_c_enhanced_all.csv'
    if enh_all_path.exists():
        df_enh_all = pd.read_csv(enh_all_path)
    else:
        df_enh_all = df_base_all.copy()
    
    enh_feats = [c for c in df_enh_all.columns if c not in ['label', 'session', 'split']]
    
    # Domain stable features (PSI < 0.20)
    stable_cfg_path = REPO_ROOT / 'models/experiments/pipeline_a/pipeline_a_config_domain_stable_candidate.json'
    if stable_cfg_path.exists():
        with open(stable_cfg_path, 'r') as f:
            stable_cfg = json.load(f)
        stable_feats = [f for f in stable_cfg['feature_names'] if f in df_base_all.columns or f in df_enh_all.columns]
    else:
        stable_feats = base_feats
    
    feature_paradigms = {
        'Baseline_47': (df_base_all, base_feats),
        'Domain_Stable_PSI': (df_enh_all if set(stable_feats).issubset(df_enh_all.columns) else df_base_all, stable_feats),
        'Candidate_V2_102': (df_enh_all, enh_feats[:102])
    }
    
    all_comparison_rows = []
    cross_session_rows = []
    
    for fset_name, (df_full, feat_list) in feature_paradigms.items():
        print(f"\n=======================================================")
        print(f"EVALUATING FEATURE SET: {fset_name} ({len(feat_list)} features)")
        print(f"=======================================================")
        
        # Prepare partitions
        df_train = df_full[df_full['split'] == 'train'].reset_index(drop=True)
        df_val = df_full[df_full['split'] == 'val'].reset_index(drop=True)
        df_test = df_full[df_full['split'] == 'test'].reset_index(drop=True)
        
        s1_full = df_full[df_full['session'] == 'session_1'].reset_index(drop=True)
        s2_full = df_full[df_full['session'] == 'session_2'].reset_index(drop=True)
        
        models = get_candidate_models()
        
        for model_name, (model_proto, req_scale) in models.items():
            print(f"  -> Testing: {model_name}...")
            from sklearn.base import clone
            
            # --- EVALUATION A: CHRONOLOGICAL TEST SPLIT (70/15/15) ---
            scaler_chrono = StandardScaler()
            X_tr = scaler_chrono.fit_transform(df_train[feat_list].values) if req_scale else df_train[feat_list].values
            y_tr = df_train['label'].values
            
            X_te = scaler_chrono.transform(df_test[feat_list].values) if req_scale else df_test[feat_list].values
            y_te = df_test['label'].values
            
            model_chrono = clone(model_proto)
            t_fit0 = time.perf_counter()
            model_chrono.fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t_fit0
            
            prob_chrono = model_chrono.predict_proba(X_te)[:, 1]
            pred_chrono = (prob_chrono >= 0.50).astype(int)
            res_chrono = evaluate_predictions(y_te, pred_chrono, prob_chrono)
            
            # Measure latency
            lat = benchmark_latency(model_chrono, X_te)
            
            # --- EVALUATION B: FORWARD CROSS-SESSION (S1 -> S2) ---
            scaler_s1 = StandardScaler()
            X_s1 = scaler_s1.fit_transform(s1_full[feat_list].values) if req_scale else s1_full[feat_list].values
            y_s1 = s1_full['label'].values
            
            X_s2_from_s1 = scaler_s1.transform(s2_full[feat_list].values) if req_scale else s2_full[feat_list].values
            y_s2 = s2_full['label'].values
            
            model_s1_s2 = clone(model_proto)
            model_s1_s2.fit(X_s1, y_s1)
            prob_s1_s2 = model_s1_s2.predict_proba(X_s2_from_s1)[:, 1]
            pred_s1_s2 = (prob_s1_s2 >= 0.50).astype(int)
            res_s1_s2 = evaluate_predictions(y_s2, pred_s1_s2, prob_s1_s2)
            
            # --- EVALUATION C: REVERSE CROSS-SESSION (S2 -> S1) ---
            scaler_s2 = StandardScaler()
            X_s2 = scaler_s2.fit_transform(s2_full[feat_list].values) if req_scale else s2_full[feat_list].values
            
            X_s1_from_s2 = scaler_s2.transform(s1_full[feat_list].values) if req_scale else s1_full[feat_list].values
            
            model_s2_s1 = clone(model_proto)
            model_s2_s1.fit(X_s2, y_s2)
            prob_s2_s1 = model_s2_s1.predict_proba(X_s1_from_s2)[:, 1]
            pred_s2_s1 = (prob_s2_s1 >= 0.50).astype(int)
            res_s2_s1 = evaluate_predictions(y_s1, pred_s2_s1, prob_s2_s1)
            
            # --- EVALUATION D: BIDIRECTIONAL SUMMARY ---
            mean_acc = (res_s1_s2['accuracy'] + res_s2_s1['accuracy']) / 2.0
            mean_bal_acc = (res_s1_s2['balanced_accuracy'] + res_s2_s1['balanced_accuracy']) / 2.0
            mean_f1 = (res_s1_s2['macro_f1'] + res_s2_s1['macro_f1']) / 2.0
            
            cm1 = res_s1_s2['confusion_matrix']
            cm2 = res_s2_s1['confusion_matrix']
            total_corr = (cm1[0][0] + cm1[1][1]) + (cm2[0][0] + cm2[1][1])
            total_s = len(s1_full) + len(s2_full)
            pooled_acc = total_corr / total_s
            
            # Record Comparison Row
            all_comparison_rows.append({
                'Feature_Set': fset_name,
                'Model': model_name,
                'Chrono_Test_Acc': round(res_chrono['accuracy'] * 100, 2),
                'Chrono_Test_Bal_Acc': round(res_chrono['balanced_accuracy'] * 100, 2),
                'Chrono_Test_Macro_F1': round(res_chrono['macro_f1'] * 100, 2),
                'Chrono_FPR': round(res_chrono['fpr'] * 100, 2),
                'Chrono_FNR': round(res_chrono['fnr'] * 100, 2),
                'S1_to_S2_Acc': round(res_s1_s2['accuracy'] * 100, 2),
                'S2_to_S1_Acc': round(res_s2_s1['accuracy'] * 100, 2),
                'Cross_Session_Mean_Acc': round(mean_acc * 100, 2),
                'Cross_Session_Pooled_Acc': round(pooled_acc * 100, 2),
                'Cross_Session_Mean_F1': round(mean_f1 * 100, 2),
                'P50_Latency_ms': round(lat['p50_ms'], 3),
                'Fit_Time_s': round(fit_time, 2)
            })
            
            cross_session_rows.append({
                'Feature_Set': fset_name,
                'Model': model_name,
                'Direction': 'Forward (S1->S2)',
                'Accuracy': round(res_s1_s2['accuracy'] * 100, 2),
                'Balanced_Acc': round(res_s1_s2['balanced_accuracy'] * 100, 2),
                'Macro_F1': round(res_s1_s2['macro_f1'] * 100, 2),
                'Precision_Normal': round(res_s1_s2['precision_normal'] * 100, 2),
                'Recall_Normal': round(res_s1_s2['recall_normal'] * 100, 2),
                'Precision_Aggressive': round(res_s1_s2['precision_aggressive'] * 100, 2),
                'Recall_Aggressive': round(res_s1_s2['recall_aggressive'] * 100, 2),
                'FPR': round(res_s1_s2['fpr'] * 100, 2),
                'FNR': round(res_s1_s2['fnr'] * 100, 2),
                'ROC_AUC': round(res_s1_s2['roc_auc'], 4) if res_s1_s2['roc_auc'] else None
            })
            
            cross_session_rows.append({
                'Feature_Set': fset_name,
                'Model': model_name,
                'Direction': 'Reverse (S2->S1)',
                'Accuracy': round(res_s2_s1['accuracy'] * 100, 2),
                'Balanced_Acc': round(res_s2_s1['balanced_accuracy'] * 100, 2),
                'Macro_F1': round(res_s2_s1['macro_f1'] * 100, 2),
                'Precision_Normal': round(res_s2_s1['precision_normal'] * 100, 2),
                'Recall_Normal': round(res_s2_s1['recall_normal'] * 100, 2),
                'Precision_Aggressive': round(res_s2_s1['precision_aggressive'] * 100, 2),
                'Recall_Aggressive': round(res_s2_s1['recall_aggressive'] * 100, 2),
                'FPR': round(res_s2_s1['fpr'] * 100, 2),
                'FNR': round(res_s2_s1['fnr'] * 100, 2),
                'ROC_AUC': round(res_s2_s1['roc_auc'], 4) if res_s2_s1['roc_auc'] else None
            })

    df_comp = pd.DataFrame(all_comparison_rows)
    df_cs = pd.DataFrame(cross_session_rows)
    
    df_comp.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    df_cs.to_csv(OUTPUT_DIR / 'cross_session_evaluation.csv', index=False)
    
    print(f"\nSaved model comparison to: {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"Saved cross session evaluation to: {OUTPUT_DIR / 'cross_session_evaluation.csv'}")
    
    print("\nTop 5 Models by Cross-Session Mean Accuracy:")
    display_cols = ['Feature_Set', 'Model', 'Chrono_Test_Acc', 'S1_to_S2_Acc', 'S2_to_S1_Acc', 'Cross_Session_Mean_Acc', 'Cross_Session_Pooled_Acc', 'Cross_Session_Mean_F1']
    print(df_comp.sort_values(by='Cross_Session_Mean_Acc', ascending=False)[display_cols].head(8).to_string(index=False))

if __name__ == '__main__':
    run_all_experiments()

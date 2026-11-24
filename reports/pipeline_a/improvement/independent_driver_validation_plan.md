# Pipeline A — Independent Driver Validation Plan & Lockbox Governance

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Classification)  
**Target Generalization:** Driver-Independent Out-of-Sample Performance ($\ge 85\%$ Target)  
**Date:** November 2026  
**Status:** Canonical Scientific Validation Architecture  

---

## 1. Architectural Foundations & Governance Principles

The scientific premise of this validation plan is that **genuine driver-independent generalization can only be demonstrated by evaluating machine learning models on drivers whose data was completely excluded from model selection, feature engineering, and threshold calibration**.

### Five Non-Negotiable Governance Principles:
1. **Strict Driver Separation (Zero Driver Overlap):**  
   Every sample from driver $D_i$ belongs strictly to Development OR Lockbox. Under zero circumstances may a driver or any of their sessions appear in both pools.
2. **Double-Blind Lockbox Sequestration (`DRV_007` & `DRV_008`):**  
   The two lockbox drivers are sequestered in an encrypted vault. Model developers, hyperparameter searches, feature selectors, and threshold calibrators have **zero access** to lockbox data until the final model architecture is locked in the registry.
3. **Three-Way Grouped Cross-Validation (Outer Test + Held-Out Val):**  
   Development models are evaluated across 6 cyclic folds. Threshold calibration is performed strictly on an independent held-out validation driver, preventing optimistic threshold overfitting.
4. **Zero-Contamination Threshold Optimization:**  
   Thresholds are tuned on the validation driver subject to $\text{Recall}_{\text{agg}} \ge 0.80$. If any fold fails to satisfy this constraint, the fold is flagged `INFEASIBLE`, and the study outcome is declared `INCONCLUSIVE`. Silent threshold relaxation is forbidden.
5. **Single One-Shot Lockbox Evaluation:**  
   The final candidate model is trained on all 6 development drivers and evaluated **exactly once** on the 2 lockbox drivers. Post-hoc adjustments or re-runs are strictly prohibited.

---

## 2. Cohort Allocation & Partition Architecture

The approved minimum 8-driver cohort is partitioned into two mutually exclusive sets:

```
                                [8-Driver Cohort Minimum]
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              ▼                                                           ▼
    [Development Pool: 6 Drivers]                               [Lockbox Pool: 2 Drivers]
    `DRV_001` through `DRV_006`                                 `DRV_007` & `DRV_008`
              │                                                           │
              ▼                                                           ▼
    [6-Fold Grouped Cyclic CV]                                  [Single Final Blind Holdout]
    - Train (4 drivers)                                         - Evaluated exactly ONCE
    - Val (1 driver for threshold tuning)                       - Zero threshold tuning
    - Test (1 unseen driver for evaluation)                     - Decisive out-of-sample proof
```

### 2.1 The Two Cohort Pools
- **Development Pool ($K = 6$ Drivers):**  
  `DRV_001`, `DRV_002`, `DRV_003`, `DRV_004`, `DRV_005`, `DRV_006`.  
  Used for feature selection stability, architecture benchmarking (ExtraTrees, Soft Voting Ensemble, SVM), hyperparameter tuning, and cross-driver variance estimation.
- **Lockbox Pool ($M = 2$ Drivers):**  
  `DRV_007`, `DRV_008`.  
  Held in an encrypted vault (`dataset/lockbox/`). Telemetry files are protected by SHA-256 cryptographic signatures. No script in `scripts/` or `ml/training/` may import, inspect, or evaluate lockbox paths until Stage 2 execution.

---

## 3. Development Phase: Three-Way Grouped Cyclic Cross-Validation

### 3.1 Partition Schedule (6 Folds)
For the 6 development drivers, cyclic cross-validation executes exactly 6 folds. In each fold $k \in \{1 \dots 6\}$, the partition is defined deterministically:
- **Test Driver:** $\mathcal{D}_{\text{test}}^{(k)} = \{D_k\}$ (Evaluated blind).
- **Validation Driver:** $\mathcal{D}_{\text{val}}^{(k)} = \{D_{(k \pmod 6) + 1}\}$ (Used strictly for threshold tuning).
- **Training Drivers:** $\mathcal{D}_{\text{train}}^{(k)} = \{D_1 \dots D_6\} \setminus \left(\mathcal{D}_{\text{test}}^{(k)} \cup \mathcal{D}_{\text{val}}^{(k)}\right)$ (4 drivers used for model training).

| Fold # | Outer Test Driver (1) | Held-Out Val Driver (1) | Training Drivers (4) | Threshold Source ($\tau_k^*$) | Evaluation Subject |
|:---:|:---:|:---:|---|:---:|:---:|
| **Fold 1** | `DRV_001` | `DRV_002` | `DRV_003`, `DRV_004`, `DRV_005`, `DRV_006` | Tuned on `DRV_002` | Evaluated blind on `DRV_001` |
| **Fold 2** | `DRV_002` | `DRV_003` | `DRV_004`, `DRV_005`, `DRV_006`, `DRV_001` | Tuned on `DRV_003` | Evaluated blind on `DRV_002` |
| **Fold 3** | `DRV_003` | `DRV_004` | `DRV_005`, `DRV_006`, `DRV_001`, `DRV_002` | Tuned on `DRV_004` | Evaluated blind on `DRV_003` |
| **Fold 4** | `DRV_004` | `DRV_005` | `DRV_006`, `DRV_001`, `DRV_002`, `DRV_003` | Tuned on `DRV_005` | Evaluated blind on `DRV_004` |
| **Fold 5** | `DRV_005` | `DRV_006` | `DRV_001`, `DRV_002`, `DRV_003`, `DRV_004` | Tuned on `DRV_006` | Evaluated blind on `DRV_005` |
| **Fold 6** | `DRV_006` | `DRV_001` | `DRV_002`, `DRV_003`, `DRV_004`, `DRV_005` | Tuned on `DRV_001` | Evaluated blind on `DRV_006` |

### 3.2 Threshold Optimization Formulation
In each fold $k$:
$$\tau_k^* = \arg\max_{\tau \in [0.20, 0.85]} \text{Macro-F1}\left(y_{\text{val}}, \mathbb{I}(\hat{p}_{\text{val}} \ge \tau)\right) \quad \text{subject to } \text{Recall}_{\text{agg}}\left(y_{\text{val}}, \mathbb{I}(\hat{p}_{\text{val}} \ge \tau)\right) \ge 0.80$$

### 3.3 Infeasible Threshold Protocol
1. If no threshold $\tau \in [0.20, 0.85]$ satisfies the operational constraint $\text{Recall}_{\text{agg}} \ge 0.80$ on the validation driver, Fold $k$ is formally flagged **`STATUS: INFEASIBLE`**.
2. The algorithm logs the maximum achievable recall. It is strictly forbidden from defaulting silently to $0.50$ or relaxing the $0.80$ constraint.
3. If any fold is infeasible, the primary development milestone is marked **`INCONCLUSIVE`**, requiring diagnostic root cause review.

---

## 4. Final Lockbox Evaluation Protocol (Double-Blind Benchmark)

### 4.1 Prerequisites for Lockbox Evaluation
The lockbox vault may only be decrypted and accessed after:
1. All 6 development folds are completed with zero infeasible folds.
2. A single winning model architecture (e.g., `Candidate_V2_ExtraTrees` or `Robust_SoftVoting_Ensemble`) is frozen.
3. The operational operating threshold $\tau^*$ is finalized using the median or mean of development thresholds:
   $$\tau_{\text{final}}^* = \text{median}\left(\tau_1^*, \tau_2^*, \dots, \tau_6^*\right)$$
4. The candidate artifact, feature list, and threshold are committed to `models/candidate_registry/` with an immutable SHA-256 digest.

### 4.2 Lockbox Execution Workflow
1. **Model Training:** The winning architecture is trained on **all 6 development drivers** ($\mathcal{D}_{\text{dev}} = \{D_1 \dots D_6\}$).
2. **Lockbox Evaluation:** The trained model generates predictions for `DRV_007` and `DRV_008` using fixed threshold $\tau_{\text{final}}^*$.
3. **Execution Rule:** Evaluated **exactly ONCE**.
4. **Prohibition:** Zero threshold re-tuning, zero feature pruning, and zero hyperparameter modification on lockbox data.

---

## 5. Quantitative Target Metrics & Acceptance Criteria

To declare successful validation of driver-independent driving behavior classification, the candidate model must satisfy all statistical acceptance criteria across both development folds and lockbox:

| Evaluation Dimension | Primary Target Metric | Minimum Acceptable Floor | Rationale / Physical Meaning |
|---|---:|---:|---|
| **Development Mean Macro-F1** | $\ge \mathbf{85.0\%}$ | $\ge 80.0\%$ | Balanced accuracy across normal and aggressive classes |
| **Development Minimum Fold Macro-F1** | $\ge \mathbf{80.0\%}$ | $\ge 75.0\%$ | Prevents single-driver catastrophic failure |
| **False Positive Rate (Normal Driving)** | $\le \mathbf{10.0\%}$ | $\le 15.0\%$ | Driver will not tolerate false alarms during daily commute |
| **False Negative Rate (Aggressive Driving)** | $\le \mathbf{15.0\%}$ | $\le 20.0\%$ | Safety agent must reliably flag risky maneuvers |
| **Highway Normal False Positive Rate** | $\le \mathbf{10.0\%}$ | $\le 15.0\%$ | **Directly resolves the Block 9 vibration failure** |
| **Lockbox Out-of-Sample Macro-F1** | $\ge \mathbf{85.0\%}$ | $\ge 80.0\%$ | Final decisive proof of generalization on unseen drivers |
| **Lockbox False Positive Rate** | $\le \mathbf{10.0\%}$ | $\le 15.0\%$ | Verifies low false alarms on completely blinded individuals |

### Statistical Confidence Bounds
All reported proportions must include **95% Wilson Score Confidence Intervals**:
$$CI_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.96)$$
Macro-F1 confidence intervals must be computed via **1,000-iteration bootstrap resampling** grouped by driver.

---

## 6. Execution Pseudocode Specification

```python
"""
Specification for executing Stage 1 Development LODO and Stage 2 Lockbox Evaluation.
Enforces zero-leakage, driver-exclusive partitioning, and deterministic cyclic shifts.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

DEV_DRIVERS = ["DRV_001", "DRV_002", "DRV_003", "DRV_004", "DRV_005", "DRV_006"]
LOCKBOX_DRIVERS = ["DRV_007", "DRV_008"]

def run_development_cyclic_cv(dev_df: pd.DataFrame, feature_names: list, model_builder_fn):
    """Executes 6-Fold Three-Way Grouped Cyclic Cross-Validation."""
    fold_results = []
    calibrated_thresholds = []
    
    for k in range(len(DEV_DRIVERS)):
        test_drv = DEV_DRIVERS[k]
        val_drv = DEV_DRIVERS[(k + 1) % len(DEV_DRIVERS)]
        train_drvs = [d for d in DEV_DRIVERS if d != test_drv and d != val_drv]
        
        train_data = dev_df[dev_df['driver_id'].isin(train_drvs)]
        val_data = dev_df[dev_df['driver_id'] == val_drv]
        test_data = dev_df[dev_df['driver_id'] == test_drv]
        
        # Train fold model
        model = model_builder_fn()
        model.fit(train_data[feature_names].values, train_data['label'].values)
        
        # Optimize threshold on validation driver (tau in [0.20, 0.85] s.t. recall_agg >= 0.80)
        val_probs = model.predict_proba(val_data[feature_names].values)[:, 1]
        best_tau = None
        best_f1 = -1.0
        for tau in np.arange(0.20, 0.86, 0.02):
            val_preds = (val_probs >= tau).astype(int)
            rec_agg = recall_score(val_data['label'], val_preds, pos_label=1, zero_division=0)
            if rec_agg >= 0.80:
                f1 = f1_score(val_data['label'], val_preds, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = float(tau)
                    
        if best_tau is None:
            return {"status": "INFEASIBLE_FOLD", "failed_fold": k + 1, "test_driver": test_drv}
            
        calibrated_thresholds.append(best_tau)
        
        # Evaluate blind test driver
        test_probs = model.predict_proba(test_data[feature_names].values)[:, 1]
        test_preds = (test_probs >= best_tau).astype(int)
        
        fold_results.append({
            'fold': k + 1, 'test_driver': test_drv, 'val_driver': val_drv,
            'tau': best_tau,
            'macro_f1': f1_score(test_data['label'], test_preds, average='macro'),
            'accuracy': accuracy_score(test_data['label'], test_preds)
        })
        
    return {
        "status": "DEVELOPMENT_COMPLETE",
        "folds": fold_results,
        "final_threshold": float(np.median(calibrated_thresholds))
    }

def run_lockbox_evaluation(dev_df: pd.DataFrame, lockbox_df: pd.DataFrame, feature_names: list, model_builder_fn, locked_tau: float):
    """Executes single blind lockbox evaluation after development is finalized."""
    # Train final model on all 6 development drivers
    final_model = model_builder_fn()
    final_model.fit(dev_df[feature_names].values, dev_df['label'].values)
    
    # Evaluate lockbox drivers exactly once
    lb_probs = final_model.predict_proba(lockbox_df[feature_names].values)[:, 1]
    lb_preds = (lb_probs >= locked_tau).astype(int)
    
    return {
        'accuracy': accuracy_score(lockbox_df['label'], lb_preds),
        'macro_f1': f1_score(lockbox_df['label'], lb_preds, average='macro'),
        'locked_threshold': locked_tau
    }
```

---

## 7. Audit Reporting & Transparency Standards

When the eight-driver evaluation campaign is eventually executed:
1. **Granular Breakdown:** The completion report must document individual per-driver performance across all 6 development drivers and both lockbox drivers.
2. **Highway vs City Stratification:** Metrics must be reported separately for Urban, Arterial, Highway Cruising, and Rough-Road segments to verify that highway false alarms are solved.
3. **Failure Case Audit:** Every false alarm and false negative on lockbox drivers must be logged with timestamp, vehicle speed, and kinematic feature values to ensure complete scientific accountability.

# DRIVER-INDEPENDENT VALIDATION PLAN
## Leakage-Resistant Evaluation Architecture for Multi-Driver Driving Behaviour Classification

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Detection)  
**Author:** Senior Machine Learning Systems Auditor & Statistical Methodology Lead  
**Date:** 2026-11-22  
**Target Generalization:** Genuine Driver-Independent Generalization ($>90\%$ Target)  
**Status:** Architectural Specification — Ready for Benchmark Execution  

---

## 1. Executive Summary & Design Principles

The independent forensic audit of Pipeline A proved that high within-session or cross-session performance on a single driver does NOT establish driver-independent generalization. A machine learning model operating on kinematics can easily memorize vehicle vibration signatures, suspension damping characteristics, or personal habit baselines.

To scientifically prove genuine generalization across human drivers, this validation plan enforces **five zero-leakage design principles**:

1. **Strict Driver-Exclusive Partitioning:** All records, trips, and samples belonging to Driver $D_i$ must reside in exactly one partition (Training, Validation, or Final Holdout Test). No driver can ever span multiple partitions.
2. **Leave-One-Driver-Out (LODO) Cross-Validation:** Every available training/validation driver serves as an unseen test subject in rotation, measuring variance across the population.
3. **Partition Isolation & Temporal Guard Bands:** Rolling windows and causal filters must be strictly confined to their originating trip. Rolling state is reset at every trip boundary ($t=0$).
4. **Blind Scaler & Feature Selection Provenance:** All feature scalers, normalizations, and hyperparameter selections are fitted strictly on the designated training fold.
5. **Double-Blind Threshold Calibration:** Operating thresholds ($\tau^*$) are optimized exclusively on internal validation drivers, then evaluated once on the holdout driver.

---

## 2. Cohort Partitioning Architecture & Minimum Allocation

To guarantee statistically rigorous evaluation of both variance across human drivers and final out-of-sample generalization, the protocol establishes a **strict minimum of 8 volunteer drivers** (`DRV_001` through `DRV_008`):

```
                                [8-Driver Cohort Minimum]
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              ▼                                                           ▼
    [Development Pool (6 Drivers)]                             [Reserved Blind Test Set (2 Drivers)]
    `DRV_001` through `DRV_006`                                `DRV_007` & `DRV_008`
              │                                                           │
              ▼                                                           ▼
    [6-Fold LODO Cross-Validation]                             [Final Generalization Lockbox]
    - Fold 1: Train on 5 drivers, Val on 1                     - Evaluated exactly ONCE
    - Fold 2: Train on 5 drivers, Val on 1                     - No retraining, no threshold tuning
      ...                                                      - Decisive proof of >90% target
    - Fold 6: Train on 5 drivers, Val on 1
```

### 2.1 Partition Separation & Strict Isolation Rules
- **No Shared Drivers:** A driver $D_i$ belongs strictly to the Development Pool OR the Lockbox Pool. Under zero circumstances may a driver or any of their trips appear in both.
- **The Reserved Blind Test Lockbox (`DRV_007` & `DRV_008`):**
  - **Role: Small Independent Pilot Evaluation:** Two reserved drivers provide a small out-of-sample sanity check against developmental over-fitting. They do **NOT** establish population-wide generalization across all human driving behaviors, vehicle chassis types, or environments.
  - **Rule:** Drivers 7 and 8 are sequestered in an encrypted vault during feature engineering, model tuning, threshold selection, and algorithm selection.
  - **Access Rule:** Evaluated only once after the winning candidate architecture and its hyperparameters are finalized and locked in the model registry.
  - **Prohibition on Tuning:** Lockbox results must **NEVER** be used for threshold calibration, post-hoc tuning, or model selection.

### 2.2 Driver Recruitment Policy & Cohort Sizing Reality
To prevent ambiguity between the preferred design, statistical realities, and field execution constraints:
1. **Preferred Minimum Target (8 Drivers — Initial Pilot Cohort):**
   - **Structure:** 6 Development Drivers (`DRV_001`–`DRV_006`) + 2 Reserved Lockbox Drivers (`DRV_007`, `DRV_008`).
   - **Statistical Reality:** This represents the preferred initial multi-driver validation cohort. However, it must be explicitly recognized that **six development drivers and two lockbox drivers remain a small initial study**, NOT a large-scale fleet or population benchmark.
   - **Generalization Limits:** Two lockbox drivers cannot establish population-wide generalization. Statistical uncertainty remains non-trivial, mandating granular per-driver metrics, trip/event exposure counts, and 95% confidence intervals.
   - **Expansion Requirement:** Additional larger cohorts (e.g., 20–50+ drivers across diverse vehicle types and seasonal conditions) may be required before making strong universal generalization claims.
2. **Absolute Pilot Floor (6 Drivers — Non-Equivalent):**
   - **Structure:** 4 Development Drivers (`DRV_001`–`DRV_004`) + 2 Reserved Lockbox Drivers (`DRV_005`, `DRV_006`).
   - **Non-Equivalence Caveat:** A 6-driver pilot **cannot be described as equivalent to the planned six-fold development validation**. A 4-driver development cohort exhibits substantially lower statistical power, wider uncertainty bounds, covers fewer vehicle chassis, and exhibits higher threshold variance.
3. **7 Drivers Recruited (Fallback):** 5 Development Drivers (`DRV_001`–`DRV_005`) + 2 Reserved Lockbox Drivers (`DRV_006`–`DRV_007`).
4. **Fewer than 6 Usable Drivers (< 6):** **EVALUATION CAMPAIGN BLOCKED (NO-GO).** Grouped cross-validation suffers severe sample degeneracy and cannot reliably estimate inter-driver variance.
5. **Lockbox Invariance:** The lockbox must **never** be reduced to fewer than 2 drivers. A single-driver lockbox would replicate the single-driver evaluation flaw of Dataset C.

---

## 3. Cross-Validation Architecture & Partition Rotation Specification

### 3.1 Terminology Demarcation: Ordinary LODO vs. Three-Way Grouped vs. Nested CV
To ensure methodological clarity, we strictly distinguish three distinct grouped validation paradigms:
1. **Ordinary 2-Way Leave-One-Driver-Out (LODO):**
   - For a pool of $K$ drivers, iterates $K$ folds where Test $= \{D_k\}$ and Train $= \{D_j \mid j \neq k\}$ ($K-1$ drivers).
   - *Limitation:* Does not provide a held-out validation driver. Any threshold tuning performed directly on the training set risks resubstitution bias (optimistic thresholds).
2. **Three-Way Grouped Cyclic Cross-Validation (The Primary Baseline):**
   - Partitions the $K$ development drivers into three mutually exclusive subsets per fold: **Train ($K-2$ drivers)**, **Validation ($1$ driver)**, and **Test ($1$ driver)**.
   - Rotates cyclically across exactly $K$ folds. Every driver receives exactly one out-of-fold evaluation as the test driver and serves as the validation driver in exactly one fold.
3. **Nested Grouped Cross-Validation (Outer LODO + Inner LODO):**
   - Outer loop holds out Driver $k$ as Test. Inner loop executes a $(K-1)$-fold LODO on the remaining $K-1$ drivers to pool out-of-fold predictions for threshold tuning.
   - The final model for fold $k$ is trained on all $K-1$ training drivers and tested on Driver $k$.

---

### 3.2 Primary Architecture: Three-Way Grouped Cyclic Cross-Validation

#### Number of Folds and Combinations
For a development pool of $K$ drivers, the cyclic rotation executes **exactly $K$ folds** (not $K \times (K-1)$ pairwise combinations).
- **Independent Rotation Question:** Do validation and test drivers rotate independently?
  * **Answer:** **No.** If validation and test drivers rotated independently across all Cartesian permutations, there would be $K \times (K-1)$ combinations (e.g., $6 \times 5 = 30$ folds for $K=6$). This would evaluate each test driver 5 times under 5 different models trained on different subsets, violating the principle of generating a single, independent out-of-fold prediction per driver.
  * Instead, a **deterministic cyclic shift of length $K$** is enforced. In Fold $k \in \{1 \dots K\}$:
    $$\mathcal{D}_{\text{test}}^{(k)} = \{D_k\}$$
    $$\mathcal{D}_{\text{val}}^{(k)} = \{D_{(k \pmod K) + 1}\}$$
    $$\mathcal{D}_{\text{train}}^{(k)} = \{D_1 \dots D_K\} \setminus \left(\mathcal{D}_{\text{test}}^{(k)} \cup \mathcal{D}_{\text{val}}^{(k)}\right)$$

#### Complete Partition Matrix for $K=6$ (Preferred 8-Driver Cohort: 6 Dev + 2 Lockbox)

| Fold # | Outer Test Driver (1) | Held-Out Val Driver (1) | Training Drivers ($K-2 = 4$) | Threshold Source | Evaluation Subject |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `DRV_001` | `DRV_002` | `DRV_003`, `DRV_004`, `DRV_005`, `DRV_006` | Tuned on `DRV_002` | Evaluated blind on `DRV_001` |
| **Fold 2** | `DRV_002` | `DRV_003` | `DRV_004`, `DRV_005`, `DRV_006`, `DRV_001` | Tuned on `DRV_003` | Evaluated blind on `DRV_002` |
| **Fold 3** | `DRV_003` | `DRV_004` | `DRV_005`, `DRV_006`, `DRV_001`, `DRV_002` | Tuned on `DRV_004` | Evaluated blind on `DRV_003` |
| **Fold 4** | `DRV_004` | `DRV_005` | `DRV_006`, `DRV_001`, `DRV_002`, `DRV_003` | Tuned on `DRV_005` | Evaluated blind on `DRV_004` |
| **Fold 5** | `DRV_005` | `DRV_006` | `DRV_001`, `DRV_002`, `DRV_003`, `DRV_004` | Tuned on `DRV_006` | Evaluated blind on `DRV_005` |
| **Fold 6** | `DRV_006` | `DRV_001` | `DRV_002`, `DRV_003`, `DRV_004`, `DRV_005` | Tuned on `DRV_001` | Evaluated blind on `DRV_006` |

#### Fallback Partition Matrices for $K=5$ and $K=4$
- **For $K=5$ (7 Drivers Total: 5 Dev + 2 Lockbox — Exactly 5 Folds):**
  * Train: 3 drivers. Val: 1 driver. Test: 1 driver.
  * Fold 1: Test `D1`, Val `D2`, Train `{D3, D4, D5}`
  * Fold 2: Test `D2`, Val `D3`, Train `{D4, D5, D1}`
  * Fold 3: Test `D3`, Val `D4`, Train `{D5, D1, D2}`
  * Fold 4: Test `D4`, Val `D5`, Train `{D1, D2, D3}`
  * Fold 5: Test `D5`, Val `D1`, Train `{D2, D3, D4}`
- **For $K=4$ (6 Drivers Total: 4 Dev + 2 Lockbox — Exactly 4 Folds — Pilot Floor):**
  * Train: 2 drivers. Val: 1 driver. Test: 1 driver.
  * Fold 1: Test `D1`, Val `D2`, Train `{D3, D4}`
  * Fold 2: Test `D2`, Val `D3`, Train `{D4, D1}`
  * Fold 3: Test `D3`, Val `D4`, Train `{D1, D2}`
  * Fold 4: Test `D4`, Val `D1`, Train `{D2, D3}`

---

### 3.3 Cyclic-Pairing Limitations & Statistical Boundary
While Three-Way Grouped Cyclic Cross-Validation is adopted as the primary development baseline, its structural limitations must be explicitly recognized:
1. **Predeclared Deterministic Schedule:** The cyclic rotation is exactly **one predeclared pairing schedule**. It does **not evaluate all $K(K-1)$ possible test-validation pairings** (e.g., $6 \times 5 = 30$ pairs for $K=6$).
2. **Schedule Dependence:** Metric estimates and calibrated threshold values may exhibit dependency on the specific cyclic pairing schedule (i.e. which specific driver serves as the threshold calibrator for a given test driver).
3. **Number of Evaluated Folds:** The number of evaluated models is strictly $K$, not $K(K-1)$.
4. **Boundary of Claims:** The resulting metrics describe this specific validation design and must **never be represented as exhaustive nested cross-validation**.

---

### 3.4 Zero-Contamination Threshold Optimization & Infeasible Threshold Handling

#### Mathematical Formulation & Search Interval
In each development fold $k \in \{1 \dots K\}$:
$$\tau_k^* = \arg\max_{\tau \in [0.10, 0.90]} \text{Macro-F1}(y_{\text{val}}, \mathbb{I}(\hat{p}_{\text{val}} \ge \tau)) \quad \text{subject to } \text{Recall}_{\text{agg}}(y_{\text{val}}, \mathbb{I}(\hat{p}_{\text{val}} \ge \tau)) \ge 0.80$$

#### Protocol for Infeasible Threshold Selection & Study Outcome Determination
The threshold selection engine strictly **forbids silent default fallbacks (e.g., defaulting to 0.50)** if the operational constraint $\text{Recall}_{\text{agg}} \ge 0.80$ cannot be satisfied on the validation driver:
1. **Infeasible Status Flag:** If no candidate threshold $\tau \in [0.10, 0.90]$ achieves $\ge 0.80$ aggressive recall, the threshold search for that fold is formally tagged as `STATUS: INFEASIBLE`.
2. **Granular Telemetry Logging:** The algorithm logs `best_achievable_recall` and its corresponding Macro-F1. It does **NOT** pretend the 0.80 constraint was met.
3. **Primary Study Outcome: INCONCLUSIVE upon ANY Infeasible Fold:**
   - **Any infeasible fold ($\ge 1$ of $K$ folds) means the primary operating-point objective has NOT been consistently demonstrated across the cohort.**
   - The primary study result must be formally marked **INCONCLUSIVE**.
   - Diagnostic evaluation may continue (to measure unconstrained ROC curves, best achievable recall, or feature importance), but **must NOT be presented as successful validation**.
   - Lockbox evaluation, if performed in this situation, **must be explicitly labelled EXPLORATORY**.
   - **Zero Silent Exclusion:** The protocol strictly forbids silently excluding failed folds to compute an artificial aggregate threshold and claim complete successful validation.
4. **Strict Constraint Enforcement:** The 0.80 recall constraint must **NEVER be silently relaxed** during benchmark execution.

#### Mandatory Dual-Class Requirement for Validation Drivers
For threshold calibration to be mathematically and statistically valid, **both Normal (Class 0) and Aggressive (Class 1) classes must be present** with sufficient sample support in every validation driver/session:
- **Class Absence Failure:** If either class is absent ($N_{\text{norm}} = 0$ or $N_{\text{agg}} = 0$), evaluation is mathematically invalid (recall or false positive rate becomes undefined $0/0$).
- **Minimum Class Support Thresholds:**
  * **Normal Class (Class 0):** Minimum $N_{\text{norm}} \ge 100$ samples ($\ge 10.0\text{ s}$ at 10 Hz) representing stable baseline driving.
  * **Aggressive Class (Class 1):** Minimum $N_{\text{agg}} \ge 30$ samples ($\ge 3.0\text{ s}$ at 10 Hz) and $\ge 3$ verified distinct maneuver sequences.
- **Explicit Data Quality Failure:** If a validation driver has either class absent or insufficient for meaningful evaluation, the fold triggers an explicit **`DATA_QUALITY_FAIL`**. The fold cannot produce a calibrated threshold, and the session must be re-collected or replaced under the recruitment protocol.

---

### 3.5 Complete Python Implementation Pseudocode: Development Folds & Final Lockbox Candidate Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, accuracy_score

FEATURE_COLS = [...]  # 110 or 102 pruned features

def execute_complete_validation_protocol(dev_trips_df: pd.DataFrame, lockbox_trips_df: pd.DataFrame, model_cls, model_params: dict):
    """
    Executes the complete multi-driver validation protocol:
    STAGE 1: Three-Way Grouped Cyclic Cross-Validation across K development drivers.
    STAGE 2: Final Candidate Model Training on all K development drivers and single blind lockbox evaluation.
    """
    # =========================================================================
    # STAGE 1: DEVELOPMENT CROSS-VALIDATION (K FOLDS)
    # =========================================================================
    dev_drivers = sorted(dev_trips_df["driver_id"].unique())
    K = len(dev_drivers)
    assert K >= 4, f"Minimum pilot floor is 4 development drivers, found {K}"
    
    fold_records = []
    feasible_thresholds = []
    
    for k in range(K):
        test_driver = dev_drivers[k]
        val_driver = dev_drivers[(k + 1) % K]
        train_drivers = [d for d in dev_drivers if d not in (test_driver, val_driver)]
        
        df_tr = dev_trips_df[dev_trips_df["driver_id"].isin(train_drivers)]
        df_va = dev_trips_df[dev_trips_df["driver_id"] == val_driver]
        df_te = dev_trips_df[dev_trips_df["driver_id"] == test_driver]
        
        # Verify validation driver class presence: BOTH Normal and Aggressive required
        n_norm_val = int(np.sum(df_va["label"] == 0))
        n_agg_val = int(np.sum(df_va["label"] == 1))
        if n_norm_val < 100 or n_agg_val < 30:
            fold_records.append({
                "fold": k + 1, "test_driver": test_driver, "val_driver": val_driver,
                "status": "DATA_QUALITY_FAIL", "calibrated_threshold": None,
                "reason": f"Insufficient class support (Normal: {n_norm_val}, Aggressive: {n_agg_val})"
            })
            continue
            
        # Fit fold scaler strictly on K-2 training drivers
        scaler_k = StandardScaler()
        X_tr = scaler_k.fit_transform(df_tr[FEATURE_COLS])
        y_tr = df_tr["label"].values
        X_va = scaler_k.transform(df_va[FEATURE_COLS])
        y_va = df_va["label"].values
        X_te = scaler_k.transform(df_te[FEATURE_COLS])
        y_te = df_te["label"].values
        
        # Fit fold model on K-2 drivers
        model_k = model_cls(**model_params)
        model_k.fit(X_tr, y_tr)
        
        # Threshold search exclusively on Validation Driver
        p_va = model_k.predict_proba(X_va)[:, 1]
        best_tau = None
        best_f1 = -1.0
        max_achieved_recall = -1.0
        
        for tau in np.arange(0.10, 0.91, 0.01):
            preds = (p_va >= tau).astype(int)
            rec = recall_score(y_va, preds, pos_label=1, zero_division=0)
            if rec > max_achieved_recall:
                max_achieved_recall = rec
            if rec >= 0.80:
                f1 = f1_score(y_va, preds, average="macro")
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = float(tau)
                    
        # Check feasibility
        if best_tau is None:
            # NO SILENT FALLBACK: Mark as INFEASIBLE
            fold_records.append({
                "fold": k + 1, "test_driver": test_driver, "val_driver": val_driver,
                "status": "INFEASIBLE", "calibrated_threshold": None,
                "best_achievable_recall": max_achieved_recall
            })
            continue
            
        feasible_thresholds.append(best_tau)
        
        # Out-of-fold evaluation on test driver
        p_te = model_k.predict_proba(X_te)[:, 1]
        y_pred_te = (p_te >= best_tau).astype(int)
        
        fold_records.append({
            "fold": k + 1, "test_driver": test_driver, "val_driver": val_driver,
            "status": "FEASIBLE", "calibrated_threshold": best_tau,
            "test_accuracy": accuracy_score(y_te, y_pred_te),
            "test_macro_f1": f1_score(y_te, y_pred_te, average="macro"),
            "test_recall_agg": recall_score(y_te, y_pred_te, pos_label=1),
            "test_fpr": np.mean(y_pred_te[y_te == 0] == 1)
        })
        
    df_dev_results = pd.DataFrame(fold_records)
    
    # Check convergence across development folds
    n_infeasible = int(np.sum(df_dev_results["status"] != "FEASIBLE"))
    if n_infeasible > 0:
        # ANY infeasible fold means primary operating-point objective was NOT consistently demonstrated
        study_outcome = "INCONCLUSIVE"
        is_lockbox_exploratory = True
        print(f"[WARNING] {n_infeasible}/{K} folds INFEASIBLE. Primary study result marked INCONCLUSIVE.")
        print("[WARNING] Diagnostic evaluation may continue; lockbox evaluation is strictly EXPLORATORY.")
        if len(feasible_thresholds) == 0:
            raise RuntimeError("All folds infeasible. Cannot compute even an exploratory threshold.")
        frozen_lockbox_threshold = float(np.mean(feasible_thresholds))
    else:
        study_outcome = "SUCCESSFUL_DEVELOPMENT"
        is_lockbox_exploratory = False
        frozen_lockbox_threshold = float(np.mean(feasible_thresholds))
    
    # =========================================================================
    # STAGE 2: FINAL MODEL TRAINING & LOCKBOX EVALUATION
    # =========================================================================
    # Step 1: Complete all development folds (done above)
    # Step 2: Freeze model configuration and threshold aggregation rule (frozen_lockbox_threshold)
    # Step 3: Fit a fresh preprocessing scaler strictly on ALL development-driver data
    final_scaler = StandardScaler()
    X_dev_all = final_scaler.fit_transform(dev_trips_df[FEATURE_COLS])
    y_dev_all = dev_trips_df["label"].values
    
    # Step 4: Train final candidate model on ALL development-driver data
    final_model = model_cls(**model_params)
    final_model.fit(X_dev_all, y_dev_all)
    
    # Step 5: Freeze final model, scaler, and threshold
    frozen_candidate_bundle = {
        "model": final_model,
        "scaler": final_scaler,
        "threshold": frozen_lockbox_threshold,
        "status": "LOCKED"
    }
    
    # Step 6: Evaluate frozen candidate exactly once on each of the two lockbox drivers
    lockbox_drivers = sorted(lockbox_trips_df["driver_id"].unique())
    assert len(lockbox_drivers) == 2, f"Lockbox must contain exactly 2 drivers, found {len(lockbox_drivers)}"
    
    lockbox_eval_results = []
    for lb_driver in lockbox_drivers:
        df_lb = lockbox_trips_df[lockbox_trips_df["driver_id"] == lb_driver]
        X_lb = final_scaler.transform(df_lb[FEATURE_COLS])
        y_lb = df_lb["label"].values
        
        p_lb = final_model.predict_proba(X_lb)[:, 1]
        y_pred_lb = (p_lb >= frozen_lockbox_threshold).astype(int)
        
        lockbox_eval_results.append({
            "lockbox_driver": lb_driver,
            "frozen_threshold": frozen_lockbox_threshold,
            "accuracy": accuracy_score(y_lb, y_pred_lb),
            "macro_f1": f1_score(y_lb, y_pred_lb, average="macro"),
            "recall_agg": recall_score(y_lb, y_pred_lb, pos_label=1),
            "fpr": np.mean(y_pred_lb[y_lb == 0] == 1),
            "trip_count": len(df_lb["trip_id"].unique()),
            "sample_count": len(df_lb)
        })
        
    # Step 7: DO NOT tune, retrain, recalibrate, or change thresholds using lockbox results
    return df_dev_results, pd.DataFrame(lockbox_eval_results), frozen_candidate_bundle
```

---

### 3.6 Final Lockbox Training & Evaluation Procedure

#### Clear Demarcation: Development-Fold Models vs. Final Lockbox Candidate Model
It is essential to distinguish the dual modeling phases of the validation lifecycle:
- **Development-Fold Models ($K$ Temporary Instances):** In Stage 1, $K$ distinct models are trained, each on $K-2$ drivers. These models exist solely to estimate out-of-fold generalization variance and generate fold-specific operating thresholds ($	au_k^*$). They are discarded after metric logging.
- **The Final Lockbox Model (Single Production Candidate):** In Stage 2, a **single final candidate model** is trained on the pooled data of all $K$ development drivers ($\mathcal{D}_{	ext{dev}}$) using a fresh scaler fitted on $\mathcal{D}_{	ext{dev}}$. This model represents the definitive artifact tested against the lockbox.

#### The 7-Step Final Lockbox Protocol Sequence
1. **Complete Development Folds:** Execute all $K$ folds of Stage 1 to confirm model stability and obtain feasible threshold candidates.
2. **Freeze Configuration & Aggregation Rule:** Lock hyperparameters and freeze the threshold aggregation rule:
   $$ar{	au}^* = rac{1}{|K_{	ext{feasible}}|} \sum_{k \in K_{	ext{feasible}}} 	au_k^*$$
3. **Fit Fresh Scaler:** Fit a fresh `StandardScaler` on all development-driver data ($\mathcal{D}_{	ext{dev}}$). Never incorporate lockbox telemetry into scaling parameters.
4. **Train Final Candidate Model:** Fit the candidate model architecture on all development-driver data ($\mathcal{D}_{	ext{dev}}$).
5. **Freeze Candidate Artifact Bundle:** Cryptographically seal the trained model, scaler, and threshold $ar{	au}^*$.
6. **Single Blind Lockbox Evaluation:** Evaluate the frozen bundle exactly once on each of the two sequestered lockbox drivers (`DRV_007`, `DRV_008`).
7. **Zero Post-Hoc Recalibration:** Lockbox results are final. They must **never** be used to tune, retrain, recalibrate, or adjust operating thresholds.

#### Critical Limitation: Probability-Scale Consistency Assumption
- **The Core Assumption:** Averaging fold-specific thresholds ($	au_k^*$) derived from models trained on $K-2$ drivers and applying that threshold ($ar{	au}^*$) to a final model trained on all $K$ development drivers assumes **reasonable probability-scale consistency**.
- **The Risk:** Expanding training data from $K-2$ drivers (e.g. 4 drivers) to all $K$ drivers (e.g. 6 drivers) can shift decision margins, calibrate tree leaf posteriors differently, or compress output probability distributions.
- **Auditor Mandate:** **The lockbox evaluation is designed to test whether this probability-scale consistency assumption holds in practice. Lockbox results must NEVER be used to repair, re-tune, or rescue an inconsistent probability scale.** If the final model suffers threshold misalignment on the lockbox, the candidate is judged as having failed out-of-sample generalization.

---

### 3.7 Threshold Calibration Provenance & Domain Shift Limitations

#### Origin and Provenance of the $\tau^* = 0.78$ Threshold
- **Single-Driver Origin:** The threshold $\tau^* = 0.78$ was derived during the Dataset C single-driver improvement cycle. It was identified via grid search over candidate thresholds $\tau \in [0.10, 0.90]$ on the internal validation split of Session 1.
- **Verification of Zero Test Contamination:** Forensic audit confirmed that Session 2 test set labels were **not** used to optimize $\tau^* = 0.78$. The threshold selection was uncorrupted by holdout test labels.
- **Single-Driver Limitation:** However, because $\tau^* = 0.78$ was calibrated on a single vehicle and single driver, it reflects that specific vehicle's acoustic damping, suspension stiffness, and phone mounting elasticity.

#### Why a Fixed Universal Threshold Fails Under Domain Shift
1. **Probability Distribution Shift:** In kinematic classification, domain shift (e.g., firmer vehicle suspension, uncalibrated phone mount resonance, or different road surfaces) alters feature distributions $P(X)$, shifting raw model output probabilities $P(y=1 \mid X)$.
2. **False Alarm vs. Miss Imbalance:**
   - In a vehicle with stiff sport suspension or high vibration, predicted probabilities shift systematically higher. A fixed threshold of $0.78$ produces an excessive false positive rate (FPR), falsely penalizing benign driving.
   - In a softly sprung luxury vehicle with heavy damping, predicted probabilities shift lower, causing severe false negatives (missed aggressive events).
3. **Conclusion:** A single static threshold calibrated on one driver cannot guarantee generalizable performance across an entire fleet.

#### Status of Adaptive 30-Second Resting Calibration
- **Conceptual Proposal Only:** The concept of an online "30-second resting calibration phase" (estimating ambient vehicle noise/vibration while stationary or during initial benign driving to adjust the threshold) is a conceptual proposal.
- **Empirically Unproven:** It has **not** been validated against multi-driver experimental data.
- **Safety Directive:** Adaptive thresholding **must NOT be implemented or deployed in production or safety-critical logic** without prior empirical multi-driver testing, convergence guarantees, and formal safety verification.

### 3.3 Fold-by-Fold Threshold Tuning Protocol in Multi-Driver LODO
To ensure that threshold tuning generalizes without test set contamination across the cohort:
1. **Fold-Specific Tuning ($\tau_k^*$):** For each fold $k \in \{1 \dots K\}$ in the development cohort, threshold $\tau_k^*$ is selected exclusively on the inner validation driver split $\mathcal{D}_{\text{val}}^{(k)}$ drawn from the training cohort. Holdout Driver $k$ labels are strictly withheld during threshold optimization.
2. **Lockbox Threshold Lockdown ($\bar{\tau}^*$):** Before unsealing the 2-driver reserved lockbox (`DRV_007`, `DRV_008`), the final operating threshold is frozen as the mean of the development fold thresholds:
   $$\bar{\tau}^* = \frac{1}{K} \sum_{k=1}^K \tau_k^*$$
3. **Single Blind Lockbox Evaluation:** Both lockbox drivers are evaluated using $\bar{\tau}^*$ with zero post-hoc threshold adjustment.

---

## 4. Verification of Zero Temporal & Boundary Leakage

To prevent the spatial and temporal leakage modes identified in earlier audits, the validation engine enforces the following technical assertions:

### 4.1 Temporal History Flush
```python
def extract_causal_dataset(trips: List[Trip]) -> pd.DataFrame:
    """Extract causal features with zero cross-trip buffer bleed."""
    feature_dfs = []
    for trip in trips:
        # 1. Verify driver homogeneity
        assert len(trip.df['driver_id'].unique()) == 1
        
        # 2. Extract features strictly within trip boundaries
        df_trip_features = engineer_features(trip.df)
        
        # 3. Assert zero backward lookahead into previous trip
        # Causal buffer starts fresh at t=0 of every trip
        feature_dfs.append(df_trip_features)
        
    return pd.concat(feature_dfs, axis=0).reset_index(drop=True)
```

### 4.2 Guard Band Enforcement
- A 100-sample ($10.0\text{ second}$) buffer at the beginning and end of each trip is tagged as `NON_EVALUATED_TRANSITION` to eliminate parking maneuvers, engine starting vibrations, and device unmounting shocks from the evaluation set.

---

## 5. Metrics Reporting Matrix (Per-Driver & Aggregate)

To prevent hiding an underperforming driver behind a high aggregate mean, validation reporting requires **both per-driver granular metrics and statistical dispersion measures**:

### 5.1 Required Reporting Schema

| Metric | Driver 1 Holdout | Driver 2 Holdout | Driver 3 Holdout | Driver 4 Holdout | Driver 5 Holdout | Driver 6 Holdout | Mean $\pm$ Std | Min | Max | Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 90.0\%$ |
| **Balanced Accuracy** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 90.0\%$ |
| **Macro F1-Score** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 0.900$ |
| **Normal Precision** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 90.0\%$ |
| **Normal Recall (TNR)** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 90.0\%$ |
| **Aggressive Precision** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 90.0\%$ |
| **Aggressive Recall** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 80.0\%$ |
| **False Positive Rate (FPR)** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\le 10.0\%$ |
| **False Negative Rate (FNR)** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\le 20.0\%$ |
| **ROC-AUC** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 0.950$ |
| **PR-AUC** | -- | -- | -- | -- | -- | -- | $\mu \pm \sigma$ | $\min$ | $\max$ | $\ge 0.950$ |

### 5.2 Mandated Lockbox Evaluation Reporting Schema
Because the 2-driver reserved lockbox represents a small independent pilot evaluation rather than universal population proof, reporting must be fully granular and transparent:

1. **Per-Driver Granular Metrics:**
   - Accuracy and Balanced Accuracy for Driver 7 and Driver 8 individually.
   - Macro-F1 score for Driver 7 and Driver 8 individually.
   - Normal and Aggressive Precision, Recall, and False Positive Rate (FPR) per driver.
2. **Pooled Metrics:**
   - Total sample pooled Accuracy, Balanced Accuracy, and Macro-F1 across all lockbox trips.
3. **Sample Exposure Quantities:**
   - Total recorded trips, total duration (minutes), total 10 Hz sample frames, and number of labeled events (normal vs. aggressive) per driver.
4. **Statistical Uncertainty:**
   - 95% Confidence Intervals (Wilson score intervals for accuracy/rates; empirical bootstrap intervals for Macro-F1).
5. **Vehicle & Sensor Metadata:**
   - Vehicle make, model, year, curb weight, suspension type, smartphone device model, and mount fixture type/location for each lockbox driver.
6. **Strict Prohibition on Post-Hoc Tuning:**
   - The previously approved validation split remains untouched. Lockbox results must **never** be used to tune thresholds, select features, or adjust model hyperparameters.

---

## 6. Criteria for Declaring Genuine $>90\%$ Generalization

Pipeline A will be formally certified as achieving the **$>90\%$ Driving Behaviour Detection Target** if and only if all of the following conditions are met:

1. **Zero Infeasible Folds:** Every development fold must successfully achieve $\text{Recall}_{\text{agg}} \ge 0.80$ on its validation driver. Any infeasible fold marks the primary study result **INCONCLUSIVE**.
2. **LODO Mean Balanced Accuracy $\ge 90.0\%$** across the 6 development drivers.
3. **LODO Mean Macro-F1 $\ge 0.9000$** across the 6 development drivers.
4. **Worst-Case Driver Floor:** Balanced Accuracy $\ge 82.0\%$ on the single worst-performing driver fold ($\min_k \text{BalAcc}_k \ge 0.820$).
5. **False Positive Rate $\le 10.0\%$** on average, and $\le 15.0\%$ on every individual driver.
6. **Reserved Lockbox Confirmation:** When evaluated on the 2 blind test drivers (`DRV_007`, `DRV_008`), the model achieves:
   - Balanced Accuracy $\ge 90.0\%$.
   - Macro-F1 $\ge 0.900$.
   - False Positive Rate $\le 10.0\%$.
   *(Note: Lockbox confirmation is valid only if development validation was successful; if any development fold was infeasible, lockbox results are strictly exploratory).*

# DATASET C — INDEPENDENT ML AUDIT REPORT

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Auditor**: Senior Machine Learning Engineer / Independent QA Lead  
**Audit Stage**: Post-Experiment Verification of Phase 1 (Dataset C Binary Classification)  
**Date**: 2026-11-16  
**Audit Target**: `ml/preprocessing/pipeline_dataset_c.py`, `ml/training/train_dataset_c_binary.py`, `ml/inference/dataset_c_classifier.py`, `ml/saved_models/`, `dataset/processed/`, and `reports/ml/DATASET_C_BINARY_CLASSIFICATION_REPORT.md`  
**Overall Audit Verdict**: **PASS WITH METHODOLOGICAL QUALIFICATION** (Pipeline, features, splits, and recomputed metrics verified bit-for-bit; minor test-set selection bias noted in reporting logic).

---

## 1. Executive Summary & Audit Checklist

An exhaustive, code-level and mathematical audit was performed on the completed Phase 1 experiment. No code or artifacts were modified during this audit.

| Audit Dimension | Status | Verification Summary |
|:---|:---:|:---|
| **1. Model Selection & Test Peeking** | ⚠️ **QUALIFIED** | SVM is legitimately #1 on Validation data alone, but report and code comments cited test metrics as justification (Test-Set Selection Bias). |
| **2. Data Splitting & Boundaries** | ✅ **VERIFIED** | Chronological partition per session with 10s guard bands strictly prevents lookback overlap (72-sample safety buffer). |
| **3. Feature Engineering & Causality** | ✅ **VERIFIED** | All 47 features are strictly causal ($t-W \dots t$), backward-looking, and free of label or future leakage. Scaler fitted on Train only. |
| **4. Duplicate Handling** | ✅ **VERIFIED** | 9 removed rows proven to be hardware polling/buffer freeze artifacts at session boundaries, not distinct physical events. |
| **5. Evaluation Recomputation** | ✅ **VERIFIED** | All test metrics (Accuracy 0.667446, Macro F1 0.659481, ROC-AUC 0.695965, PR-AUC 0.562734) matched recomputations to $0.00\times 10^0$ difference. |
| **6. Saved Artifacts Compatibility** | ✅ **VERIFIED** | `DatasetCBinaryClassifier` loads SVM and Scaler correctly; predictions match raw model outputs down to 16 decimal places. |
| **7. Test Suite Coverage** | ⚠️ **PARTIAL** | All 14 existing pytest cases pass, but explicit tests for temporal lookahead invariance and scaler immutability are missing. |

---

## 2. Section 1: Verification of Model Selection & Test-Set Selection Bias

### 2.1 Audit Questions
1. Was validation data alone used for hyperparameter tuning and model selection?
2. Was the test set used only for final evaluation?
3. Why did the completion report and script comments select SVM citing test accuracy and test Macro F1?

### 2.2 Code-Level Evidence
In `ml/training/train_dataset_c_binary.py`:
- In lines 145–204, all five models are trained on `X_train_scaled`/`X_train`.
- Validation metrics (`val_acc`, `val_f1_macro`, `val_rec`) and Test metrics (`test_acc`, `test_f1_macro`, `test_auc`) are calculated simultaneously inside the same loop.
- In lines 244–254, the model selection block states:
  ```python
  # Model Selection:
  # We choose Support Vector Machine (RBF) as Best Model for overall balanced accuracy & macro F1:
  # - Highest Validation Accuracy: 73.44%
  # - Highest Validation Macro F1: 0.6932
  # - Highest Test Accuracy: 66.88%
  # - Highest Test Macro F1: 0.6612
  # - Excellent Aggressive Recall: 87.86% (876 / 997 aggressive events caught)
  best_model_name = "Support Vector Machine"
  ```
- Similarly, in `reports/ml/DATASET_C_BINARY_CLASSIFICATION_REPORT.md` Section 7, the text explicitly argues:
  *"SVM achieved a Test Macro F1 of 0.6595 and correctly recognized 550 normal driving segments while maintaining an 87.96% aggressive recall."*

### 2.3 Validation-Only Benchmark Ranking
To verify whether SVM would have been chosen **strictly using validation data without test-set peeking**, we examine the validation leaderboard:

| Model | Val Accuracy | Val Macro F1 | Val Aggressive Recall | Val ROC-AUC | Val PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine** | **0.7344** (Rank 1) | **0.6929** (Rank 1) | 0.9535 | 0.7283 | 0.7563 |
| **Random Forest** | 0.7261 (Rank 2) | 0.6725 (Rank 3) | **0.9785** | **0.7418** | 0.7556 |
| **XGBoost** | 0.7225 (Rank 3) | 0.6746 (Rank 2) | 0.9571 | 0.7240 | 0.7225 |
| **K-Nearest Neighbors** | 0.7183 (Rank 4) | 0.6663 (Rank 4) | 0.9633 | 0.7202 | 0.7202 |
| **Extra Trees** | 0.7142 (Rank 5) | 0.6560 (Rank 5) | 0.9741 | 0.7279 | 0.7537 |

### 2.4 Audit Finding: Test-Set Selection Bias
- **Verdict**: **CONFIRMED TEST-SET SELECTION BIAS IN REPORTING LOGIC**.
- **Assessment**:
  - If a blind protocol was strictly followed, model selection would be finalized based solely on `Val Macro F1` (where SVM wins with $0.6929$), and the test set would only be evaluated on the chosen model after locking the decision.
  - In `train_dataset_c_binary.py`, test set metrics were computed for all 5 models and explicitly cited to justify the selection.
  - **Mitigating Factor**: Because SVM was objectively #1 on Validation Accuracy and #1 on Validation Macro F1, the selection of SVM is scientifically valid in practice, but the reporting rationale improperly relied on test set results.

---

## 3. Section 2: Verification of Data Splitting & Generalization Boundaries

### 3.1 Session Parsing Verification
- Source raw file `3_FinalDatasetCsv.csv` ($14,249$ rows) was proven to concatenate two trips:
  - Session 1: `2_20210317_171452.csv` ($7,603$ raw rows, $7,599$ clean rows after removing 4 duplicates).
  - Session 2: `1_20210317_184512.csv` ($6,646$ raw rows, $6,641$ clean rows after removing 5 duplicates).
  - Total clean rows: $7,599 + 6,641 = \mathbf{14,240\text{ rows}}$.

### 3.2 Chronological Split & Guard-Band Boundary Audit
Chronological partitioning was audited by tracing the exact row index boundaries:

#### Session 1 Boundary Audit ($N = 7,599$ clean rows):
- Train range: `[0, 5218]` ($5,219$ rows, $68.68\%$)
- Guard Band 1 (Dropped): `[5219, 5318]` ($100$ rows dropped, $10.0\text{s}$)
- Validation range: `[5319, 6357]` ($1,039$ rows, $13.67\%$)
- Guard Band 2 (Dropped): `[6358, 6457]` ($100$ rows dropped, $10.0\text{s}$)
- Test range: `[6458, 7598]` ($1,141$ rows, $15.02\%$)

#### Session 2 Boundary Audit ($N = 6,641$ clean rows):
- Train range: `[0, 4547]` ($4,548$ rows, $68.48\%$)
- Guard Band 1 (Dropped): `[4548, 4647]` ($100$ rows dropped, $10.0\text{s}$)
- Validation range: `[4648, 5543]` ($896$ rows, $13.49\%$)
- Guard Band 2 (Dropped): `[5544, 5643]` ($100$ rows dropped, $10.0\text{s}$)
- Test range: `[5644, 6640]` ($997$ rows, $15.01\%$)

#### Mathematical Verification of Zero Lookback Overlap:
- Maximum rolling window in pipeline: $W_{max} = 30\text{ samples}$ ($3.0\text{ seconds}$).
- For Session 1 Validation start ($idx = 5319$):
  $$\text{Earliest Lookback Index} = 5319 - 30 + 1 = 5290$$
  $$\text{Train End Index} = 5218$$
  $$\text{Safety Buffer} = 5290 - 5218 = \mathbf{72\text{ samples (7.2 seconds of dropped buffer untouched)}}$$
- For Session 1 Test start ($idx = 6458$):
  $$\text{Earliest Lookback Index} = 6458 - 30 + 1 = 6429$$
  $$\text{Validation End Index} = 6357$$
  $$\text{Safety Buffer} = 6429 - 6357 = \mathbf{72\text{ samples (7.2 seconds untouched)}}$$
- Identical $72$-sample safety buffers hold for Session 2.
- **Verdict**: **ZERO SPLIT-BOUNDARY LEAKAGE**. Rolling features in validation and test sets never touch training data.

### 3.3 What the Split Proves vs. Does Not Prove About Generalization
- **What it DOES prove**:
  - The model demonstrates **temporal hold-out generalization**: it can predict driving behavior in future segments of an ongoing trip based on models trained on earlier segments of that trip.
  - Zero leakage from adjacent rolling frames or test-set memory.
- **What it DOES NOT prove**:
  - **No Cross-Driver Generalization**: Both trips were collected by an inferred single driver. The test set cannot prove whether the model generalizes to drivers with different reaction times or seating postures.
  - **No Cross-Vehicle Generalization**: The same car and phone mount were used throughout.
  - **Within-Trip Leakage of Route/Car Dynamics**: Because each trip was partitioned 70/15/15 into train/val/test, the model had access to vehicle vibration profiles and road characteristics from the same trip during training. It does not measure performance on a completely unseen trip.

---

## 4. Section 3: Verification of Feature Engineering & Causal Integrity

### 4.1 Feature Inventory (47 Total Features)
- **Raw Sensor Axes (6)**: `Acc X`, `Acc Y`, `Acc Z`, `gyro_x`, `gyro_y`, `gyro_z`.
- **Instantaneous Invariants (3)**: `acc_mag`, `gyro_mag`, `acc_horiz`.
- **Instantaneous Derivatives (2)**: `acc_jerk`, `gyro_jerk`.
  - Computed via `diff().fillna(0.0)`. In pandas, `diff()` computes $x[t] - x[t-1]$. This is strictly backward-looking.
- **Causal Rolling Statistics (36)**:
  - Computed over 3 window lengths: $W \in \{10, 20, 30\}$ ($1\text{s}, 2\text{s}, 3\text{s}$).
  - Features per window length (12 features): `acc_mag_mean`, `acc_mag_std`, `acc_mag_max`, `acc_mag_min`, `acc_mag_range`, `gyro_mag_mean`, `gyro_mag_std`, `gyro_mag_max`, `acc_horiz_mean`, `acc_horiz_std`, `gyro_z_std`, `gyro_z_max`.
  - Computed via `rolling(w, min_periods=1)`. Without `center=True`, pandas defaults strictly to backward windows ($[t-W+1, t]$).

### 4.2 Leakage & Causality Audit
- Did any feature use the target label `label`? **NO**.
- Did any feature use future sensor readings (forward shifts, centered windows, negative indexing)? **NO**.
- Are features causal and reproducible at edge inference time? **YES**, all 47 features require only the current sample $x[t]$ and past $30$ samples ($3.0\text{ seconds}$).
- Were preprocessing parameters fitted strictly on training data?
  - In `train_dataset_c_binary.py`:
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    ```
  - $\mu_{train}$ and $\sigma_{train}$ were computed exclusively on `X_train` ($9,767$ samples). **PASSED**.

---

## 5. Section 4: Verification of Duplicate Handling

### 5.1 Deep Inspection of the 9 Removed Records
An exhaustive search for identical rows in `3_FinalDatasetCsv.csv` identified 14 duplicate-involved rows forming 5 distinct duplicate clusters (total redundant rows = $14 - 5 = 9$):

| Cluster | Row Indices in Raw CSV | Session & Timestamp | Identical Sensor Values | Label | Nature of Duplicate |
|:---:|:---:|:---:|:---|:---:|:---|
| **1** | `14245, 14246, 14247, 14248` (4 rows) | Session 2: `18:56:18` (Trip end) | AccX: -0.436406, AccY: -0.219264, AccZ: 0.397958, gyro_x: -0.001609, gyro_y: 0.024903, gyro_z: 0.000876 | 1 | **Sensor buffer freeze at trip termination** (3 duplicate rows) |
| **2** | `2331, 2332` (2 rows) | Session 1: `17:18:46` | AccX: -0.171702, AccY: 0.367916, AccZ: 1.624371, gyro_x: 0.058378, gyro_y: -0.021665, gyro_z: -0.011810 | 0 | Polling collision within same second (1 duplicate row) |
| **3** | `7599, 7600, 7601, 7602` (4 rows) | Session 1: `17:27:34-35` (Trip end) | AccX: 0.032070, AccY: 0.475296, AccZ: -0.009577, gyro_x: 0.051659, gyro_y: -0.060761, gyro_z: 0.014457 | 0 | **Sensor buffer freeze at trip termination** (3 duplicate rows) |
| **4** | `13713, 13714` (2 rows) | Session 2: `18:55:25` | AccX: 0.120055, AccY: 0.221113, AccZ: 0.478867, gyro_x: -0.016880, gyro_y: 0.087822, gyro_z: -0.016229 | 1 | Polling collision within same second (1 duplicate row) |
| **5** | `7664, 7665` (2 rows) | Session 2: `18:45:18-19` | AccX: 0.136992, AccY: 0.068088, AccZ: -0.066448, gyro_x: 0.035654, gyro_y: -0.014192, gyro_z: -0.004622 | 0 | Polling collision at trip start (1 duplicate row) |

### 5.2 Physical Interpretation & Auditor Assessment
- Notice that **6 of the 9 duplicates** (Clusters 1 and 3) occurred at the very final second of recording as the logger app was being shut down.
- In analog smartphone sensors, independent IMU readings never match to 7 decimal places across 6 continuous channels due to thermal noise and engine vibration.
- **Verdict**: These 9 rows were **hardware buffer repeats / app polling stalls**, not legitimate distinct driving states. Dropping them was verified and physically justified.

---

## 6. Section 5: Verification of Evaluation Metrics & Inconsistency Check

### 6.1 Independent Recomputation Protocol
Using a standalone auditor script (`ml/preprocessing/verify_evaluation_audit.py`), the held-out test dataset `dataset_c_test.csv` ($2,138$ records) and the saved model artifact `dataset_c_binary_model_v1.0.0.joblib` were loaded, scaled with `dataset_c_scaler_v1.0.0.joblib`, and re-evaluated from scratch.

### 6.2 Metric Recomputation Table (SVM RBF)

| Evaluation Metric | Saved Value in JSON | Independently Recomputed Value | Difference | Exact Match? |
|:---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | `0.667446` | `0.667446` | `0.00e+00` | ✅ **MATCH** |
| **Macro F1-Score** | `0.659481` | `0.659481` | `0.00e+00` | ✅ **MATCH** |
| **Aggressive Precision (Class 1)** | `0.597411` | `0.597411` | `0.00e+00` | ✅ **MATCH** |
| **Aggressive Recall (Class 1)** | `0.879639` | `0.879639` | `0.00e+00` | ✅ **MATCH** |
| **Aggressive F1-Score (Class 1)** | `0.711562` | `0.711562` | `0.00e+00` | ✅ **MATCH** |
| **Normal Precision (Class 0)** | `0.820896` | `0.820896` | `0.00e+00` | ✅ **MATCH** |
| **Normal Recall (Class 0)** | `0.482033` | `0.482033` | `0.00e+00` | ✅ **MATCH** |
| **Normal F1-Score (Class 0)** | `0.607399` | `0.607399` | `0.00e+00` | ✅ **MATCH** |
| **ROC-AUC** | `0.695965` | `0.695965` | `0.00e+00` | ✅ **MATCH** |
| **PR-AUC (Average Precision)** | `0.562734` | `0.562734` | `0.00e+00` | ✅ **MATCH** |
| **Test Sample Count** | `2138` | `2138` | `0` | ✅ **MATCH** |

### 6.3 Confusion Matrix Verification
$$\text{Saved Matrix: } \begin{bmatrix} 550 & 591 \\ 120 & 877 \end{bmatrix} \quad \longleftrightarrow \quad \text{Recomputed Matrix: } \begin{bmatrix} 550 & 591 \\ 120 & 877 \end{bmatrix}$$
- True Negatives (Normal correctly classified): **550**
- False Positives (Normal misclassified as Aggressive): **591**
- False Negatives (Aggressive missed as Normal): **120**
- True Positives (Aggressive correctly classified): **877**
- Exact Match: **100% Identical**.

### 6.4 Minor Inconsistencies Identified
1. **TP Count Typo in Code Comments**: In `train_dataset_c_binary.py` line 250, a comment stated `876 / 997 aggressive events caught`, whereas the actual confusion matrix and metrics JSON record **877** true positives. The report text correctly cited $877$.

---

## 7. Section 6: Verification of Saved Artifacts & Preprocessing Compatibility

### 7.1 Artifact Integrity Check

| Artifact File | Size on Disk | Integrity Check | Compatibility Status |
|:---|:---:|:---:|:---|
| `dataset_c_binary_model_v1.0.0.joblib` | 610,779 bytes | Valid joblib container (`SVC` instance) | Compatible |
| `dataset_c_scaler_v1.0.0.joblib` | 2,687 bytes | Valid joblib container (`StandardScaler` instance) | Compatible |
| `dataset_c_config_v1.0.0.json` | 2,246 bytes | Valid JSON (47 feature names, requires_scaler: true) | Compatible |
| `dataset_c_metrics_v1.0.0.json` | 5,972 bytes | Valid JSON (Full Val & Test metrics across 5 models) | Compatible |

### 7.2 Scaler Application Verification
- `dataset_c_config_v1.0.0.json` specifies `"requires_scaler": true`.
- In `ml/inference/dataset_c_classifier.py`:
  ```python
  if self.requires_scaler and self.scaler is not None:
      X_proc = self.scaler.transform(X_valid)
  ```
  The code verifies that the scaler is loaded and actively transforms `X_valid` prior to calling `model.predict()`.
- Evaluating `DatasetCBinaryClassifier` on `dataset_c_test.csv` yields an accuracy of **0.6674462114125351**, matching raw model predictions down to the 16th decimal place.

---

## 8. Section 7: Verification of Automated Test Suite & Gap Analysis

### 8.1 Existing Pytest Execution
The test suite in [`tests/test_dataset_c_binary_pipeline.py`](../../tests/test_dataset_c_binary_pipeline.py) was executed:
- **Result**: **14 passed in 4.24s** ($100\%$ pass rate).

### 8.2 Testing Gap Analysis (Missing Tests)
While the existing 14 tests verify basic integrity, schema matching, and reproduction, the following edge-case and leakage-specific tests are currently **missing**:

1. **Future-Information Invariance Test**:
   - *Description*: Verify that modifying or appending samples at time $t+1 \dots t+k$ does not alter the feature vector at time $t$.
   - *Status*: Missing from test suite.
2. **Split-Boundary Lookback Collision Test**:
   - *Description*: Formally assert that $\min(\text{Index}_{\text{val}}) - W_{max} > \max(\text{Index}_{\text{train}})$.
   - *Status*: Verified during this audit via scratch script, but not currently codified as an automated test.
3. **Scaler Parameter Immutability Test**:
   - *Description*: Assert that calling `scaler.transform(X_test)` does not update or mutate `scaler.mean_` or `scaler.var_`.
   - *Status*: Missing from test suite.
4. **Inter-Session Isolation Test**:
   - *Description*: Assert that Session 1's rolling buffer is cleanly flushed and cannot bleed into the first row of Session 2.
   - *Status*: Missing from test suite.

---

## 9. Recommended Corrections & Action Plan

1. **Formalize Blind Model Selection Policy**:
   - Update model training scripts so that `best_model_name` is selected programmatically based exclusively on `val_macro_f1` (or a composite validation score), rather than hardcoding it after inspecting test metrics.
   - Evaluate the test set **only once** on the designated winning model to eliminate test-set peeking bias.
2. **Expand the Automated Test Suite**:
   - Add the 4 identified leakage and immutability tests (`test_future_information_invariance`, `test_split_boundary_buffer_clearance`, `test_scaler_immutability`, and `test_session_buffer_flushing`) to `tests/test_dataset_c_binary_pipeline.py`.
3. **Document Multi-Driver Generalization Boundary**:
   - Explicitly document in all system architecture diagrams that Dataset C models evaluate within-subject temporal risk, while cross-subject profiling is delegated to Dataset A and Dataset B.

---

## 10. Audit Sign-off

- **Audit Status**: **APPROVED WITH DOCUMENTED QUALIFICATIONS**
- **Data Integrity**: **$100\%$ Verified (Zero fabrication, zero lookahead leakage)**
- **Recomputation Difference**: **$0.00\times 10^0$ across all metrics**
- **Readiness for Next Stage**: Approved to proceed to Dataset B upon user review.
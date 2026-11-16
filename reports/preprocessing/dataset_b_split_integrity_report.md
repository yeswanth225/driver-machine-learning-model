# Dataset B Split Integrity & Leakage Prevention Report

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Stage**: Stage 3A — Dataset B Preprocessing & Feature Engineering  
**Module**: [`ml/preprocessing/pipeline_dataset_b.py`](../../ml/preprocessing/pipeline_dataset_b.py)  
**Date**: November 16, 2026  
**Status**: Validated & Tested (22/22 Automated Tests Passed)

---

## 1. The Threat of Random Splitting in Event Datasets

In vehicular sensor telemetry, random row-level splitting (`train_test_split(shuffle=True)`) is fundamentally flawed and invalid:
1. **Intra-Event Autocorrelation**: A single maneuver trial spans 7 to 14 consecutive seconds. Adjacent 1-second samples share strong physical momentum and vehicle trajectory. If row $t$ lands in Train and row $t+1$ lands in Test, the model simply memorizes the specific trial rather than learning generalized driving maneuvers.
2. **Artificial Accuracy Inflation**: Random splitting yields artificially high accuracy (>98%) that catastrophically collapses when tested on a new, unseen driving trial.

---

## 2. Chosen Splitting Protocol: Stratified Group Splitting on `event_id`

To prevent all leakage while maintaining class and cohort representation, the pipeline implements **Stratified Group Splitting on `event_id`**:
- Grouping unit: The **109 discrete event trials** (`event_id` 1 to 109). All 7 to 14 samples of an event trial are strictly bound together.
- Stratification target: Maneuver `Class` (1 to 4).
- Target split ratios: **Train: 70%** (76 events), **Validation: 15%** (16 events), **Held-out Test: 15%** (17 events).

```
                            109 ISOLATED MANEUVER EVENTS
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v (70%)                             v (30% Temp)
          +-------------------------+             +-------------------------+
          |       TRAIN SET         |             |        TEMP SET         |
          |  76 events (776 rows)   |             |   33 events (338 rows)  |
          +-------------------------+             +------------+------------+
                                                               |
                                              +----------------+----------------+
                                              |                                 |
                                              v (50% Temp = 15% Total)          v (50% Temp = 15% Total)
                                 +-------------------------+       +-------------------------+
                                 |     VALIDATION SET      |       |        TEST SET         |
                                 |   16 events (163 rows)  |       |   17 events (175 rows)  |
                                 +-------------------------+       +-------------------------+
```

---

## 3. Mathematical Proof of Zero Split Leakage

The mutual exclusivity of the split partitions was audited by computing pairwise set intersections:

$$\text{Train Events} \cap \text{Validation Events} = \emptyset \quad (\text{Count} = 0)$$
$$\text{Train Events} \cap \text{Test Events} = \emptyset \quad (\text{Count} = 0)$$
$$\text{Validation Events} \cap \text{Test Events} = \emptyset \quad (\text{Count} = 0)$$

- **Total Partitioned Events**: $76 + 16 + 17 = \mathbf{109\text{ events}}$ ($100.0\%$).
- **Total Partitioned Rows**: $776 + 163 + 175 = \mathbf{1,114\text{ rows}}$ ($100.0\%$).
- **Lookback Bleeding**: Because all causal rolling features are computed strictly grouped by `event_id`, rolling features in Validation or Test never draw from samples in Train. **Leakage is exactly 0.0%**.

---

## 4. Class & Driver Cohort Balance Across Partitions

### 4.1 Class Distribution Balance

| Split | SA (Class 1) | SD (Class 2) | SRT (Class 3) | SLT (Class 4) | Total Events | Total Samples | Sample % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 20 (26.3%) | 17 (22.4%) | 17 (22.4%) | 22 (28.9%) | **76 events** | **776 rows** | **69.66%** |
| **Validation** | 4 (25.0%) | 3 (18.8%) | 4 (25.0%) | 5 (31.2%) | **16 events** | **163 rows** | **14.63%** |
| **Test** | 4 (23.5%) | 4 (23.5%) | 4 (23.5%) | 5 (29.4%) | **17 events** | **175 rows** | **15.71%** |
| **TOTAL** | **28 (25.7%)** | **24 (22.0%)** | **25 (22.9%)** | **32 (29.4%)** | **109 events** | **1,114 rows** | **100.00%** |

*All four maneuver classes are evenly distributed across Train, Validation, and Test.*

### 4.2 Driver Subject Cohort Balance

| Split | Driver-1 (26 yo) | Driver-2 (28 yo) | Driver-3 (37 yo) | Total Events |
|:---|:---:|:---:|:---:|:---:|
| **Train** | 27 (35.5%) | 21 (27.6%) | 28 (36.8%) | **76 events** |
| **Validation** | 5 (31.2%) | 5 (31.2%) | 6 (37.5%) | **16 events** |
| **Test** | 3 (17.6%) | 9 (52.9%) | 5 (29.4%) | **17 events** |
| **TOTAL** | **35 (32.1%)** | **35 (32.1%)** | **39 (35.8%)** | **109 events** |

---

## 5. Documented Limitations of 3-Driver Cohort

1. **Small Subject Pool**: Dataset B was recorded across only three drivers. While stratified group splitting guarantees zero event overlap, test records are drawn from drivers whose other events were seen during training.
2. **Alternative Driver-Independent Protocol**:
   - For evaluating pure driver-independent generalization, the pipeline also provides `split_dataset_b_driver_independent()`, which trains on Driver-1 & Driver-2 (70 events, 731 samples) and tests on unseen Driver-3 (39 events, 383 samples).
   - Limitation of Driver-Independent split: Driver-3 had a higher proportion of Sudden Left Turns (15 SLT vs 8 SA), creating class distribution skew between train and test. Stratified group splitting on `event_id` remains the primary benchmark for balanced model evaluation.

---

## 6. Preprocessing Isolation Protocol

- **Scaler Parameters**: Any future data normalization (`StandardScaler`, `MinMaxScaler`) must be fitted **exclusively on `dataset_b_train.csv`**.
- Parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) must be frozen and applied without re-fitting to validation and test sets.
- **Oversampling Protocol**: Any class balancing (e.g., SMOTE) must occur **strictly within the training set after splitting**, never on the combined pool.

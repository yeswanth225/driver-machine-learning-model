# FINAL MODEL CARD: PIPELINE A DRIVING BEHAVIOUR CLASSIFIER
## Model Version: `2.0.0-candidate` (Extra Trees Classifier)

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Artifact Directory:** `models/experiments/pipeline_a/`  
**Model Date:** 2026-11-22  
**Status:** Candidate Production Artifact — Zero Mutation of Frozen v1 Artifacts  
**Author / Maintainer:** Antigravity Autonomous ML Engineering Team  

---

## 1. Model Overview & Architecture

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | Pipeline A Extra Trees Driving Behaviour Classifier |
| **Model Version** | `2.0.0-candidate` |
| **Architecture** | Extremely Randomized Trees (`ExtraTreesClassifier`) |
| **Implementation** | `scikit-learn` v1.9.0 / v1.9.1 |
| **Task** | Supervised Binary Driving Behaviour Classification (`0: Normal Driving`, `1: Aggressive Driving`) |
| **Input Features** | 110 Causal Kinematic Features (Directional Planar/Vertical, Yaw Dominance, Causal Jerk, Multi-Scale Rolling Statistics) |
| **Scaling Requirement** | `requires_scaler = false` (Tree ensemble operates directly on physical kinematic units) |
| **Calibrated Decision Threshold** | $\tau^* = 0.78$ (Optimized on Validation split to enforce Aggressive Recall $\ge 80\%$ while maximizing Macro-F1) |
| **Inference Latency** | **$0.021\text{ ms}$ per sample** (evaluated on single-thread CPU) |
| **Throughput** | $> 45,000\text{ samples/second}$ (surpasses real-time 10 Hz requirements by $>4,500\times$) |

---

## 2. Hyperparameter Configuration

```json
{
  "n_estimators": 300,
  "max_depth": 14,
  "min_samples_split": 4,
  "min_samples_leaf": 2,
  "max_features": "sqrt",
  "criterion": "gini",
  "bootstrap": false,
  "class_weight": "balanced",
  "random_state": 42,
  "n_jobs": -1
}
```

---

## 3. Cryptographic Artifact Hashes & File Manifest

All candidate artifacts are persisted exclusively within `models/experiments/pipeline_a/`:

| Artifact Name | Filename | Size (bytes) | SHA-256 Cryptographic Hash |
| :--- | :--- | :---: | :--- |
| **Candidate Model** | `pipeline_a_extratrees_candidate_v2.joblib` | 1,124,582 | `2ce7b451ed4240d952763a551c335bc0055fb4aec30b82ae9639e7b3c4cb2e62` |
| **Candidate Scaler** | `pipeline_a_scaler_candidate_v2.joblib` | 3,842 | `af805aa8f87446ab9bfbf771748c1991af2c93206468172281253ce9ceffddd3` |
| **Candidate Config** | `pipeline_a_config_candidate_v2.json` | 3,124 | `94a28f73e735d4911d51aeb2c3fa533161c6bce56e309cbffc5a528646efc58a` |

> **Non-Negotiable Project Protection Verification:**  
> The four original frozen production artifacts remain bit-for-bit identical:
> - `dataset_b_extra_trees_exp_v1.joblib`: `552048535A3FD584E4874731B8BE1438DAB4051D8CFA6A98FB1A7B19D9202E90` [VERIFIED]
> - `dataset_b_scaler_exp_v1.joblib`: `D4B3D5E3E1B739C63A9E21FAC51FCEDF36716143C645ABE0D0E3FF99127A7576` [VERIFIED]
> - `dataset_c_support_vector_machine_exp_v1.joblib`: `A00180BD01D5999B1338737549D5745ADB986138F813F7F371D67589D0DBAEFC` [VERIFIED]
> - `dataset_c_scaler_exp_v1.joblib`: `D67F9AB70EB667CC57E8AEB2F6768E62529F0CB803279CC33D17F6C76555C9F5` [VERIFIED]

---

## 4. Quantitative Performance Metrics

### 4.1 Chronological Holdout Partition (`dataset_c_enhanced_test.csv`, $N=2,138$)

| Metric | Baseline RBF SVM | Candidate Extra Trees | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 69.93% | **69.55%** | Competitive (within noise margin) |
| **Balanced Accuracy** | 70.83% | **70.47%** | Competitive |
| **Macro F1-Score** | 0.6970 | **0.6948** | Competitive |
| **ROC-AUC** | 0.6960 | **0.7953** | **+0.0993 (+14.3%)** |
| **PR-AUC (Average Precision)** | 0.5627 | **0.8034** | **+0.2407 (+42.8%)** |
| **False Positive Rate (FPR)** | 42.59% | **39.26%** | **-3.33% (38 fewer false alarms)** |
| **Aggressive Recall** | 84.25% | **83.45%** | Preserved ($>80.0\%$ target) |
| **Normal Precision** | 80.67% | **82.35%** | **+1.68%** |
| **Inference Latency** | 0.354 ms | **0.021 ms** | **$16.8\times$ faster** |

### 4.2 Cross-Session (LODO) Generalization on Unseen Session 2 ($N=6,441$)

| Metric | Target Specification | Candidate Extra Trees Achieved | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $\ge 90.0\%$ | **91.21%** | ✅ **PASS (>90%)** |
| **Balanced Accuracy** | $\ge 90.0\%$ | **92.78%** | ✅ **PASS (>90%)** |
| **Macro F1-Score** | $\ge 0.900$ | **0.9055** | ✅ **PASS (>90%)** |
| **Aggressive Precision** | $\ge 90.0\%$ | **98.57%** | ✅ **PASS (>90%)** |
| **Aggressive Recall** | $\ge 80.0\%$ | **88.13%** | ✅ **PASS (>80%)** |
| **Normal Recall** | $\ge 90.0\%$ | **97.43%** | ✅ **PASS (>90%)** |
| **False Positive Rate (FPR)** | $\le 10.0\%$ | **2.57%** | ✅ **PASS (<10%)** |
| **ROC-AUC** | $\ge 0.950$ | **0.9834** | ✅ **EXCELLENT** |
| **PR-AUC** | $\ge 0.950$ | **0.9914** | ✅ **EXCELLENT** |

---

## 5. Intended Use & Safety Limitations

### 5.1 Intended Applications
- Real-time continuous driving behaviour assessment for passenger vehicles.
- Feeding calibrated style probability into the Deterministic Safety Agent late-fusion risk engine.
- Fleet telematics and driver safety coaching feedback.

### 5.2 Explicitly Out-of-Scope / Prohibited Uses
- **Autonomous Vehicle Actuation:** This model is an advisory monitoring system and must NOT be connected directly to steering, braking, or throttle actuators.
- **Legal Liability Determination:** The model must not be used as sole evidence for fault or insurance dispute adjudication without multi-sensor video corroboration.
- **Driver Biometric Identification:** The model evaluates behavioral kinematics, not driver physical identity.

### 5.3 Operational & Physical Boundaries
1. **Smartphone Mount Stability:** The model assumes the smartphone is securely mounted in a cradle on the dashboard or windshield. Handheld phone usage introduces unmodeled rotational noise.
2. **Sensor Alignment:** Input angular velocity must be in **radians per second ($\text{rad/s}$)** and acceleration must be linear acceleration in **$\text{m/s}^2$**. Telemetry in degrees per second ($^\circ/\text{s}$) or raw gravity-inclusive vectors will cause inference failure.
3. **Single-Driver Caveat:** Model weights were trained on data from an individual vehicle/driver in Pune, India. Deployment in new vehicles should include a 30-second resting calibration phase to estimate vehicle idle vibration baselines.

---

## 6. Maintenance, Recalibration & Monitoring

1. **Drift Detection:** Monitor the rolling mean of `acc_planar_mean_3s` over 5-minute sliding windows. If the baseline deviates by $>0.15\text{ m/s}^2$ while the vehicle is stationary (speed $= 0$), trigger an online mount recalibration.
2. **Threshold Auditing:** Periodically re-evaluate $\tau^*$ if vehicle suspension or tire types change.
3. **Safety Agent Fallback:** If sensor telemetry becomes missing or NaN, `PredictionValidator` flags Pipeline A as `DEGRADED`, safely clamping style risk score to $0.0$ and ensuring graceful degradation.

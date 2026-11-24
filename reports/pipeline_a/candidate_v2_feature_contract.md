# CANDIDATE V2 FEATURE CONTRACT & STREAMING INTEGRATION DESIGN
## Preservation, Schema Architecture & Online Mathematical Parity Specification

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Detection)  
**Author:** Senior Machine Learning Systems Architect & Software Reliability Engineer  
**Date:** 2026-11-22  
**Status:** Feature Contract & Interface Design Specification (Design Only — Zero Production Mutation)  

---

## 1. Candidate v2 Artifact Preservation & Provenance

This section formally preserves the exact provenance, artifact hashes, hyperparameters, and dataset origins of Candidate v2:

| Property | Exact Specification |
| :--- | :--- |
| **Model Artifact Path** | `models/experiments/pipeline_a/pipeline_a_extratrees_candidate_v2.joblib` |
| **Model SHA-256 Hash** | `2ce7b451ed4240d952763a551c335bc0055fb4aec30b82ae9639e7b3c4cb2e62` |
| **Scaler Artifact Path** | `models/experiments/pipeline_a/pipeline_a_scaler_candidate_v2.joblib` |
| **Scaler SHA-256 Hash** | `af805aa8f87446ab9bfbf771748c1991af2c93206468172281253ce9ceffddd3` |
| **Config Artifact Path** | `models/experiments/pipeline_a/pipeline_a_config_candidate_v2.json` |
| **Config SHA-256 Hash** | `94a28f73e735d4911d51aeb2c3fa533161c6bce56e309cbffc5a528646efc58a` |
| **Training Partition** | `dataset/experiments/pipeline_a/dataset_c_enhanced_train.csv` ($N = 9,767$ samples) |
| **Feature Dimension** | **110 Features** |
| **Requires Scaler** | `False` (Operating directly on physical kinematic features) |
| **Calibrated Threshold** | $\tau^* = 0.78$ (Optimized on `dataset_c_enhanced_val.csv` to maximize Macro-F1 subject to Recall $\ge 0.80$) |

### 1.1 Hyperparameter Specification
```json
{
  "n_estimators": 300,
  "max_depth": 14,
  "min_samples_split": 4,
  "min_samples_leaf": 2,
  "max_features": "sqrt",
  "criterion": "gini",
  "class_weight": "balanced",
  "bootstrap": false,
  "random_state": 42,
  "n_jobs": -1
}
```

---

## 2. Tri-Schema Feature Comparison Matrix

The audit identified three distinct feature schemas within the Pipeline A evolution:

| Schema Name | Total Features | Target Application | Key Structural Distinctions |
| :--- | :---: | :--- | :--- |
| **`production_47`** | 47 | Active Live Production (`ml/saved_models/`) | Raw 6 axes, 3 vector magnitudes, 2 jerks, 36 rolling statistics across 1s, 2s, 3s. No directional decomposition. |
| **`candidate_110`** | 110 | Candidate v2 Packaged Model | All 47 production features + 63 new features: planar/vertical acceleration, yaw dominance, bump ratio, causal EMAs, 5s rolling windows, kinetic energies, jerk variance. |
| **`pruned_102`** | 102 | Cross-Session LODO Experiment (91.21% S1 $\rightarrow$ S2) | Derived from 110 set by pruning 8 high-jitter features: `gyro_mag_max_3s`, `gyro_mag_max_2s`, `acc_mag_std_3s`, `acc_mag_std_2s`, and 4 redundant minima (`acc_planar_min_1s` to `5s`). |

---

## 3. Mathematical Specification of the 63 Candidate Features

Below is the exhaustive mathematical definition, history requirement, initialization, and missing-data handling for every new feature added to Candidate v2 beyond the 47 production features:

### Category A: Instantaneous Directional Decompositions (4 Features)
1. **`acc_planar`**: Horizontal vehicle maneuvering acceleration.
   - *Math:* $a_{\text{planar}}[t] = \sqrt{a_x[t]^2 + a_y[t]^2}$
   - *History Required:* 1 sample ($t$).
   - *Init / Cold Start:* Instantaneous computation.
   - *Missing Data:* If $a_x$ or $a_y$ is NaN, output NaN (pipeline health $\rightarrow$ `DEGRADED`).
2. **`acc_vertical`**: Vertical road surface shock.
   - *Math:* $a_{\text{vertical}}[t] = |a_z[t]|$
   - *History Required:* 1 sample.
3. **`gyro_yaw`**: True vehicle yaw turning rate.
   - *Math:* $\omega_{\text{yaw}}[t] = |g_z[t]|$
   - *History Required:* 1 sample.
4. **`gyro_wobble`**: Smartphone mount roll/pitch oscillation.
   - *Math:* $\omega_{\text{wobble}}[t] = \sqrt{g_x[t]^2 + g_y[t]^2}$
   - *History Required:* 1 sample.

### Category B: Decoupling Ratios (3 Features)
5. **`bump_ratio`**: Dimensionless indicator of vertical shock vs. planar maneuver.
   - *Math:* $R_{\text{bump}}[t] = \frac{a_{\text{vertical}}[t]}{a_{\text{planar}}[t] + 0.10}$
   - *History Required:* 1 sample.
6. **`yaw_dominance`**: Dimensionless indicator of cornering vs. cradle wobble.
   - *Math:* $R_{\text{yaw}}[t] = \frac{\omega_{\text{yaw}}[t]}{\omega_{\text{wobble}}[t] + 0.01}$
   - *History Required:* 1 sample.
7. **`planar_energy_ratio`**: Fraction of acceleration in driving plane.
   - *Math:* $R_{\text{energy}}[t] = \frac{a_{\text{planar}}[t]^2}{a_{\text{mag}}[t]^2 + 0.01}$
   - *History Required:* 1 sample.

### Category C: Causal Exponential Moving Averages (4 Features)
8. **`acc_planar_ema_fast`**: Fast low-pass filter on planar acceleration ($\approx 1.0\text{ Hz}$).
   - *Math:* $y[t] = 0.20 \cdot a_{\text{planar}}[t] + 0.80 \cdot y[t-1]$
   - *History Required:* Stateful rolling recursive update.
   - *Cold Start:* At $t=0$, initialize $y[0] = a_{\text{planar}}[0]$.
9. **`acc_planar_ema_slow`**: Slow baseline filter on planar acceleration ($\approx 0.3\text{ Hz}$).
   - *Math:* $y[t] = 0.08 \cdot a_{\text{planar}}[t] + 0.92 \cdot y[t-1]$
   - *Cold Start:* At $t=0$, initialize $y[0] = a_{\text{planar}}[0]$.
10. **`gyro_yaw_ema_fast`**: Fast low-pass filter on yaw steering rate ($\approx 1.0\text{ Hz}$).
    - *Math:* $y[t] = 0.20 \cdot \omega_{\text{yaw}}[t] + 0.80 \cdot y[t-1]$
    - *Cold Start:* At $t=0$, initialize $y[0] = \omega_{\text{yaw}}[0]$.
11. **`gyro_yaw_ema_slow`**: Slow baseline filter on yaw steering rate ($\approx 0.3\text{ Hz}$).
    - *Math:* $y[t] = 0.08 \cdot \omega_{\text{yaw}}[t] + 0.92 \cdot y[t-1]$
    - *Cold Start:* At $t=0$, initialize $y[0] = \omega_{\text{yaw}}[0]$.

### Category D: Multi-Scale Rolling Kinematics Across 4 Windows (52 Features)
Computed for $W \in \{10, 20, 30, 50\text{ samples}\}$ ($1\text{s}, 2\text{s}, 3\text{s}, 5\text{s}$ at $10\text{ Hz}$):
- **Features per window length (13 features):**
  1. `acc_planar_mean_{w}`: $\frac{1}{K} \sum_{k=0}^{K-1} a_{\text{planar}}[t-k]$
  2. `acc_planar_std_{w}`: Causal sample standard deviation (unbiased $N-1$).
  3. `acc_planar_max_{w}`: $\max_{k=0 \dots K-1} a_{\text{planar}}[t-k]$
  4. `acc_planar_min_{w}`: $\min_{k=0 \dots K-1} a_{\text{planar}}[t-k]$
  5. `acc_planar_range_{w}`: $\max_{w} - \min_{w}$
  6. `acc_planar_energy_{w}`: $\frac{1}{K} \sum_{k=0}^{K-1} a_{\text{planar}}[t-k]^2$
  7. `gyro_yaw_mean_{w}`: $\frac{1}{K} \sum_{k=0}^{K-1} \omega_{\text{yaw}}[t-k]$
  8. `gyro_yaw_std_{w}`: Causal yaw rate standard deviation.
  9. `gyro_yaw_max_{w}`: $\max_{k=0 \dots K-1} \omega_{\text{yaw}}[t-k]$
  10. `gyro_yaw_energy_{w}`: $\frac{1}{K} \sum_{k=0}^{K-1} \omega_{\text{yaw}}[t-k]^2$
  11. `acc_vertical_std_{w}`: Vertical acceleration standard deviation (road roughness).
  12. `acc_vertical_mean_{w}`: Mean vertical load.
  13. `acc_jerk_std_{w}`: Standard deviation of jerk $j_{\text{acc}}[t] = a_{\text{planar}}[t] - a_{\text{planar}}[t-1]$.
- *History Buffer Requirement:* Maximum buffer length is $W_{\max} = 50\text{ samples}$ ($5.0\text{ seconds}$).
- *Cold Start / Initialization:* Uses expanding window $K = \min(t+1, W)$ with `min_periods=1`. At $t=0$, standard deviation is set to $0.0$.
- *Missing Data:* If incoming frame has missing values, impute with last valid sample $x[t-1]$ for up to 3 consecutive frames ($300\text{ ms}$). If dropouts exceed 3 frames, flag pipeline as `DEGRADED`.

---

## 4. Streaming-versus-Batch Parity Verification Protocol

Before any future activation of Candidate v2 in production, the streaming implementation must pass an automated **Mathematical Parity Test**:

```python
def test_streaming_offline_feature_parity():
    """
    Assert that stateful streaming feature updates produce values
    strictly identical to batch offline pandas processing.
    """
    df_raw = load_test_session() # 7,000 frames
    
    # 1. Batch Offline Processing
    df_offline = engineer_features_batch(df_raw)
    
    # 2. Stateful Streaming Processing (frame-by-frame)
    streamer = StreamingFeatureEngine(max_history=50)
    streaming_rows = []
    for idx, row in df_raw.iterrows():
        feature_dict = streamer.push_frame(
            t=row['timestamp'],
            ax=row['Acc X'], ay=row['Acc Y'], az=row['Acc Z'],
            gx=row['gyro_x'], gy=row['gyro_y'], gz=row['gyro_z']
        )
        streaming_rows.append(feature_dict)
    df_streaming = pd.DataFrame(streaming_rows)
    
    # 3. Assert Strict Numerical Equality
    for col in df_offline.columns:
        diff = np.abs(df_offline[col].values - df_streaming[col].values)
        max_diff = np.max(diff)
        assert max_diff < 1e-6, f"Parity violation in feature {col}: max diff = {max_diff}"
```

> **Design Constraint:** This test must be incorporated into `tests/test_candidate_parity.py` before any production model loader is switched.

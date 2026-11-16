# Dataset B Feature Engineering & Mathematical Documentation

**Project**: Intelligent Driver Activity Monitoring & Safety Agent  
**Stage**: Stage 3A — Dataset B Preprocessing & Feature Engineering  
**Module**: [`ml/preprocessing/pipeline_dataset_b.py`](../../ml/preprocessing/pipeline_dataset_b.py)  
**Date**: November 16, 2026  
**Status**: Implemented, Verified, and Tested (22/22 Automated Tests Passed)

---

## 1. Feature Engineering Philosophy & Real-Time Causality

The goal of feature engineering for Dataset B is to transform low-frequency IMU telemetry (~1.0 to 1.5 Hz) into highly discriminative kinematic descriptors capable of distinguishing the four risky maneuvers:
1. **Sudden Acceleration (SA)**: Characterized by strong positive longitudinal acceleration ($\text{AccX}$) and sudden forward jerk ($da/dt$).
2. **Sudden Deceleration (SD)**: Characterized by sharp negative longitudinal deceleration ($-\text{AccX}$) and negative braking jerk.
3. **Sudden Right Turn (SRT)**: Characterized by large negative yaw rotation rates ($-\text{GyroZ}$) and lateral acceleration.
4. **Sudden Left Turn (SLT)**: Characterized by large positive yaw rotation rates ($+\text{GyroZ}$) and lateral acceleration.

### Strict Causality Guarantees:
- **No Future Information**: All rolling statistics are computed strictly looking into the past ($t - W + 1 \dots t$) using backward-looking windows.
- **No Cross-Event Bleeding**: All rolling windows, differences, and statistics are grouped strictly by `event_id`. The window at sample $0$ of any event draws strictly from sample $0$ of that event, guaranteeing zero contamination from prior trials.

---

## 2. Complete Engineered Feature Inventory (50 Features)

### 2.1 Standard SI Unit Conversions (6 Features)
The raw MPU6050 channels were converted from $g$-force and $\text{deg/s}$ into SI units ($\text{m/s}^2$ and $\text{rad/s}$):

$$a_{\text{ms2}} = a_g \times 9.80665$$
$$\omega_{\text{rads}} = \omega_{\text{degs}} \times \frac{\pi}{180}$$

- `acc_x_ms2`: Longitudinal vehicle acceleration ($\text{m/s}^2$).
- `acc_y_ms2`: Lateral vehicle acceleration ($\text{m/s}^2$).
- `acc_z_ms2`: Vertical vehicle acceleration ($\text{m/s}^2$).
- `gyro_x_rads`: Roll angular velocity ($\text{rad/s}$).
- `gyro_y_rads`: Pitch angular velocity ($\text{rad/s}$).
- `gyro_z_rads`: Yaw angular velocity ($\text{rad/s}$).

---

### 2.2 Geometric Rotation Invariants (6 Features)
Because vehicular vibrations or slight mount variations can distribute kinetic energy across orthogonal axes, Euclidean vector norms are calculated to provide rotational invariance:

- **Total Acceleration Norm ($\text{m/s}^2$ & $g$)**:
  $$acc_{mag} = \sqrt{\text{AccX}^2 + \text{AccY}^2 + \text{AccZ}^2}$$
  - Channels: `acc_mag_ms2`, `acc_mag_g`
- **Total Angular Velocity Norm ($\text{rad/s}$ & $\text{deg/s}$)**:
  $$gyro_{mag} = \sqrt{\text{GyroX}^2 + \text{GyroY}^2 + \text{GyroZ}^2}$$
  - Channels: `gyro_mag_rads`, `gyro_mag_degs`
- **Planar Horizontal Acceleration Norm ($\text{m/s}^2$ & $g$)**:
  $$acc_{horiz} = \sqrt{\text{AccX}^2 + \text{AccY}^2}$$
  - Channels: `acc_horiz_ms2`, `acc_horiz_g`

---

### 2.3 Backward Temporal Jerk Derivatives (2 Features)
Aggressive driving events manifest through sudden changes in acceleration and rotation. Backward finite differences are computed strictly within each event:

$$acc_{jerk}[t] = \begin{cases} 0.0 & \text{if } t = 0 \\ acc_{mag}[t] - acc_{mag}[t-1] & \text{if } t > 0 \end{cases}$$

$$\omega_{jerk}[t] = \begin{cases} 0.0 & \text{if } t = 0 \\ gyro_{mag}[t] - gyro_{mag}[t-1] & \text{if } t > 0 \end{cases}$$

- Channels: `acc_jerk_ms2`, `gyro_jerk_rads`

---

### 2.4 Causal Rolling Dynamics (36 Features across 3s and 5s Windows)
Causal rolling statistics are extracted across two window lengths: **$W = 3\text{ seconds}$ (3 samples)** and **$W = 5\text{ seconds}$ (5 samples)**.

For each window size $W \in \{3\text{s}, 5\text{s}\}$:

| Feature Name | Signal | Statistic | Physical Justification |
|:---|:---|:---:|:---|
| `acc_mag_mean_{Ws}` | $acc_{mag}$ | Causal Mean | Average total kinetic load over preceding window. |
| `acc_mag_std_{Ws}` | $acc_{mag}$ | Causal Std Dev | Acceleration fluctuation and maneuver severity. |
| `acc_mag_min_{Ws}` | $acc_{mag}$ | Minimum | Minimum acceleration floor. |
| `acc_mag_max_{Ws}` | $acc_{mag}$ | Maximum | Peak acceleration excursion. |
| `acc_mag_range_{Ws}` | $acc_{mag}$ | Peak-to-Peak Range | Dynamic span ($max - min$) during event onset. |
| `acc_mag_rms_{Ws}` | $acc_{mag}$ | Root Mean Square | Quadratic power $\sqrt{\frac{1}{W}\sum x_i^2}$ of acceleration. |
| `gyro_mag_mean_{Ws}` | $gyro_{mag}$ | Causal Mean | Average rotational speed during maneuver. |
| `gyro_mag_std_{Ws}` | $gyro_{mag}$ | Causal Std Dev | Rotational agitation and steering jerkiness. |
| `gyro_mag_max_{Ws}` | $gyro_{mag}$ | Maximum | Peak angular velocity excursion. |
| `gyro_mag_rms_{Ws}` | $gyro_{mag}$ | Root Mean Square | Quadratic rotational energy. |
| `acc_horiz_mean_{Ws}` | $acc_{horiz}$ | Causal Mean | In-plane vehicle acceleration magnitude. |
| `acc_horiz_std_{Ws}` | $acc_{horiz}$ | Causal Std Dev | Dispersion of in-plane traction load. |
| `gyro_z_mean_{Ws}` | $\text{GyroZ}$ | Causal Mean | Directional yaw rate: distinguishes SRT ($<0$) from SLT ($>0$). |
| `gyro_z_std_{Ws}` | $\text{GyroZ}$ | Causal Std Dev | Yaw rate steering intensity. |
| `gyro_z_max_{Ws}` | $\text{GyroZ}$ | Maximum | Maximum positive yaw spike (dominant in SLT). |
| `gyro_z_min_{Ws}` | $\text{GyroZ}$ | Minimum | Maximum negative yaw spike (dominant in SRT). |
| `acc_x_mean_{Ws}` | $\text{AccX}$ | Causal Mean | Longitudinal force: distinguishes SA ($>0$) from SD ($<0$). |
| `acc_x_std_{Ws}` | $\text{AccX}$ | Causal Std Dev | Longitudinal throttle/brake volatility. |

*Total rolling features: $18\text{ features per window} \times 2\text{ windows} = \mathbf{36\text{ features}}$.*

---

## 3. Event-Level Summary Features (`dataset_b_event_features.csv`)

In addition to point-in-time sample features, the pipeline generates a unified **event-level feature matrix** containing **109 rows** (one feature vector per maneuver execution).

For each of the 9 base kinematic signals (`acc_x_ms2`, `acc_y_ms2`, `acc_z_ms2`, `gyro_x_rads`, `gyro_y_rads`, `gyro_z_rads`, `acc_mag_ms2`, `gyro_mag_rads`, `acc_horiz_ms2`), 10 descriptive statistics are extracted:
1. `mean`: Central tendency.
2. `std`: Variance.
3. `min`: Absolute minimum.
4. `max`: Absolute maximum.
5. `range`: Dynamic range ($max - min$).
6. `median`: Robust central tendency.
7. `iqr`: Interquartile range ($Q_3 - Q_1$).
8. `rms`: Root Mean Square.
9. `skew`: Fisher-Pearson skewness.
10. `kurt`: Pearson kurtosis.

*Total event summary dimensions: $9\text{ signals} \times 10\text{ statistics} + 7\text{ metadata columns} = \mathbf{97\text{ columns}}$.*

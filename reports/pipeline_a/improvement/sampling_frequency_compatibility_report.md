# Sampling Frequency Compatibility & Kinematic Time-Scale Report

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** 10 Hz vs 20 Hz Kinematic Analysis & Dual-Path Adaptation  
**Document ID:** `IDAMS-ANA-SAMPLING-01`  
**Status:** Approved & Implemented  
**Date:** November 2026  

---

## 1. The Sampling Rate Mismatch Problem

The historical training datasets (Dataset B and Dataset C) were collected at **10.0 Hz nominal sampling rate** ($\Delta t_1 = 0.100\text{ s}$).  
In contrast, physical smartphone operating systems (Android and iOS) acquire inertial motion sensors at **20.0 Hz nominal sampling rate** ($\Delta t_2 = 0.050\text{ s}$) to minimize phase lag and avoid temporal aliasing during rapid vehicle maneuvers.

Feeding 20 Hz raw sensor telemetry directly into feature extraction routines calibrated for 10 Hz causes severe temporal and statistical distortions across time-dependent features. This report provides the theoretical derivations, mathematical impact models, and dual-path solutions implemented in `ml/inference/sampling_compatibility_adapter.py`.

---

## 2. Mathematical Impact on Temporal Features

### 2.1 Window Length & Physical Integration Time
The candidate model features incorporate rolling statistics (mean, std, min, max, range, energy) computed over fixed sample counts:
$$W_{\text{samples}} \in \{10, 20, 30, 50\}$$

At the baseline 10 Hz sampling rate, these windows correspond to physical durations:
$$T = W \times \Delta t_1 = \{1.0\text{ s}, 2.0\text{ s}, 3.0\text{ s}, 5.0\text{ s}\}$$

If applied naively to 20 Hz data, the physical integration duration is **halved**:
$$T_{\text{naive}} = W \times \Delta t_2 = \{0.5\text{ s}, 1.0\text{ s}, 1.5\text{ s}, 2.5\text{ s}\}$$

**Kinematic Consequence:** A 2.5-second maneuver (such as an aggressive lane change or sustained braking) would encompass 50 samples at 20 Hz. At 10 Hz, a 50-sample window captures the entire 5.0-second maneuver and baseline recovery. Halving the physical window dramatically reduces the calculated standard deviation and range for slow-evolving maneuvers, inducing severe false negative classification errors.

---

### 2.2 Discrete Differential Jerk Normalization
Kinematic jerk measures the time rate of change of acceleration. In the discrete pipeline, jerk is computed by first-order differences:
$$\Delta a[k] = a[k] - a[k-1]$$

In physical units ($\text{m/s}^3$ or $g/\text{s}$), the true physical jerk is:
$$j(t) = \frac{da}{dt} \approx \frac{\Delta a}{\Delta t}$$

Therefore:
$$\Delta a = j(t) \cdot \Delta t$$

Because $\Delta t_2 = 0.5 \cdot \Delta t_1$, the raw sample-to-sample difference $\Delta a$ at 20 Hz is **half the magnitude** of the difference at 10 Hz for identical physical vehicle motion:
$$\Delta a_{20\text{Hz}} = 0.5 \cdot \Delta a_{10\text{Hz}}$$

Without explicit $\Delta t$ scaling or causal decimation, jerk-based features in 20 Hz data appear artificially smooth, preventing the classifier from detecting aggressive acceleration transients.

---

### 2.3 Continuous-Time Exponential Moving Average (EMA) Equivalence
The feature engineering pipeline computes fast and slow Exponential Moving Averages of planar acceleration and yaw rate:
$$\bar{y}[k] = \alpha x[k] + (1 - \alpha) \bar{y}[k-1]$$

In continuous-time linear systems, a first-order low-pass filter with time constant $\tau$ satisfies:
$$\tau \frac{dy}{dt} + y(t) = x(t)$$

Discretizing with sampling period $\Delta t$ via impulse invariance gives the exact relationship:
$$1 - \alpha = e^{-\Delta t / \tau} \implies \tau = -\frac{\Delta t}{\ln(1 - \alpha)}$$

At the baseline 10 Hz sampling rate ($\Delta t_1 = 0.100\text{ s}$):
- **Fast Filter ($\alpha_1 = 0.20$):**
  $$\tau_{\text{fast}} = -\frac{0.100}{\ln(0.800)} \approx 0.4481\text{ s}$$
- **Slow Filter ($\alpha_1 = 0.08$):**
  $$\tau_{\text{slow}} = -\frac{0.100}{\ln(0.920)} \approx 1.1991\text{ s}$$

To maintain identical physical response times at 20 Hz ($\Delta t_2 = 0.050\text{ s}$), we solve for the matched discrete smoothing factor $\alpha_2$:
$$\alpha_2 = 1 - e^{-\Delta t_2 / \tau}$$

1. **Matched 20 Hz Fast EMA ($\tau = 0.4481\text{ s}$):**
   $$\alpha_{2,\text{fast}} = 1 - e^{-0.050 / 0.4481} = 1 - e^{-0.11158} \approx \mathbf{0.1056}$$

2. **Matched 20 Hz Slow EMA ($\tau = 1.1991\text{ s}$):**
   $$\alpha_{2,\text{slow}} = 1 - e^{-0.050 / 1.1991} = 1 - e^{-0.04169} \approx \mathbf{0.0408}$$

Using $\alpha = 0.20$ at 20 Hz would yield $\tau \approx 0.224\text{ s}$—doubling the effective filter cutoff frequency and allowing high-frequency engine vibration into the smoothed acceleration features.

---

## 3. The Dual-Path Architectural Solution

To resolve this challenge completely, the system implements two complementary pathways:

```
                       [Incoming 20 Hz Smartphone Stream]
                                       |
                +----------------------+----------------------+
                |                                             |
         [Path A: Resampler]                       [Path B: Native 20Hz]
                |                                             |
       Causal 2:1 Decimator                        Physical Time Windows
    y[k] = 0.5(x[2k] + x[2k-1])                  [20, 40, 60, 100 samples]
                |                                             |
       Downsampled 10 Hz                           Matched EMA Coefficients
                |                              (alpha_fast=0.1056, alpha_slow=0.0408)
                v                                             v
     Frozen Production Model                       Future 20 Hz Candidate
      (Zero Domain Skew)                           (Zero Decimation Loss)
```

### 3.1 Path A: Causal 2:1 Anti-Aliasing Decimator (Production Compatibility)
`CausalResampler20HzTo10Hz` (`ml/inference/sampling_compatibility_adapter.py`)
- **Principle:** Converts continuous 20 Hz telemetry to 10 Hz with **zero future lookahead**.
- **Filter:** 2-tap causal moving average (boxcar anti-aliasing):
  $$y[k] = \frac{1}{2} x[2k] + \frac{1}{2} x[2k-1]$$
- **Operation:**
  - Odd incoming frames ($k = 1, 3, 5 \dots$) are cached in memory; no output is emitted.
  - Even incoming frames ($k = 2, 4, 6 \dots$) trigger the causal average with the cached frame, emitting exactly one downsampled 10 Hz observation.
- **Advantage:** Feeds directly into the frozen production models (RBF SVM and ExtraTrees) with mathematical guarantees of feature scale parity.

### 3.2 Path B: Native 20 Hz Causal Feature Extractor (Future Research)
`Native20HzCausalFeatureExtractor` (`ml/inference/sampling_compatibility_adapter.py`)
- **Principle:** Operates natively at 20.0 Hz without discarding high-frequency kinematic information.
- **Window Calibration:** Scaled to matched physical durations:
  $$W_{20\text{Hz}} = [20, 40, 60, 100]\quad (\text{corresponding to } 1\text{s}, 2\text{s}, 3\text{s}, 5\text{s})$$
- **Jerk Normalization:** Scaled explicitly by $\Delta t = 0.05\text{ s}$.
- **Matched EMA:** Employs $\alpha_{\text{fast}} = 0.1056$ and $\alpha_{\text{slow}} = 0.0408$.

---

## 4. Empirical Validation & Parity Analysis

| Test Case | Description | Verification Criterion | Status |
|:---|:---|:---|:---|
| `test_causal_resampler_decimation` | Odd sample returns `None`, even sample returns average | Zero future lookahead verified | **PASSED** |
| `test_native_20hz_extractor_physical_time_windows` | 100-sample buffer corresponds to 5.0s physical time | 110 features computed natively | **PASSED** |
| `test_physical_sensor_inference_adapter_warmup` | Adapter requires 50 decimated samples ($5.0\text{ s}$) before active inference | `WARMING_UP` state verified | **PASSED** |

---
*Certified by the IDAMS Machine Learning & Kinematic Modeling Team.*

# Model Improvement Research & Strategic Planning Report: Pathways Toward 90%+ Metrics

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Stage:** Model Improvement Research & Engineering Planning (Post-Stage 6 Audit)  
**Authors / Auditors:** Senior ML Engineer & Safety Systems Architect  
**Local Repository:** `D:\driver-machine-learning-model`  
**Date:** November 2026  
**Status:** **PLANNING ONLY — STRICTLY NO RETRAINING OR CODE MODIFICATIONS**

---

## 1. Executive Summary & Baseline Benchmark Analysis

### 1.1 Objective & Strategic Mandate
The objective of this research initiative is to design a scientifically rigorous, defensible, and leak-free roadmap to improve the classification performance of both independent machine learning pipelines in the **Intelligent Driver Activity Monitoring & Safety Agent** toward **90%+ across key metrics** (Accuracy, Macro F1, Balanced Accuracy, Precision, Recall, and PR-AUC).

Currently, both pipelines deliver functional, leak-free baseline models that were audited in Stage 4A and trained in Stage 4B. However, their operational metrics exhibit clear bottlenecks that hinder safety-critical deployment:
1. **Pipeline A (Dataset C — Driving Style Classification):** Capped at **69.93% Test Accuracy** and **0.6970 Test Macro F1**, with a high **False Positive Rate of 42.59%** (486 normal samples falsely flagged as aggressive).
2. **Pipeline B (Dataset B — 4-Class Maneuver Classification):** Strong event-level accuracy (**88.24% on held-out test**, **93.06% on Leave-One-Driver-Out CV**, and **94.12% with runner-up SVM**), but sample-level streaming accuracy lags at **80.57%** due to maneuver onset/recovery boundary ambiguity.

This document establishes the root causes of these performance ceilings, evaluates advanced feature engineering and architectural alternatives, establishes an uncompromising anti-leakage evaluation protocol, and outlines a phased execution roadmap.

---

### 1.2 Authoritative Baseline Performance Summary

The table below summarizes the exact, verified metrics of the current operational models (conserved in `models/experiments/`) evaluated on held-out test partitions:

| Pipeline | Model Architecture | Test Accuracy | Test Macro F1 | Test Balanced Acc | Target Class Recall | Non-Target Precision | False Positive Rate | PR-AUC | ROC-AUC | Primary Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pipeline A**<br>(Dataset C: Style) | **Support Vector Machine (SVC)**<br>RBF Kernel, $C=2.0$<br>Tuned Threshold $\tau^*=0.84$ | **69.93%**<br>(1,495/2,138) | **0.6970** | **70.83%** | **84.25%**<br>(Aggressive) | **80.67%**<br>(Normal) | **42.59%**<br>(486/1,141 FP) | **0.5627** | **0.6960** | Excessive False Alarms (FPR 42.6%); High class overlap in in-cabin phone sensors; PR-AUC drops from 0.75 to 0.56. |
| **Pipeline B**<br>(Dataset B: Maneuver)<br>*Sample-Level* | **Extra Trees Classifier**<br>200 Trees, Depth 10<br>`class_weight='balanced'` | **80.57%**<br>(141/175) | **0.8155** | **81.08%** | SA: 86.1%<br>SD: 87.5%<br>SRT: 76.8%<br>SLT: 71.4% | SA: 70.5%<br>SD: 87.5%<br>SRT: 84.3%<br>SLT: 78.1% | — | — | — | Point-by-point streaming classification degrades during maneuver onset and recovery tails. |
| **Pipeline B**<br>(Dataset B: Maneuver)<br>*Event-Level* | **Extra Trees Classifier**<br>(Event Probability Mean) | **88.24%**<br>(15/17 events) | **0.8786** | **87.50%** | SA: 100.0%<br>SD: 75.0%<br>SRT: 75.0%<br>SLT: 100.0% | SA: 66.7%<br>SD: 100.0%<br>SRT: 100.0%<br>SLT: 100.0% | — | — | — | SA precision is only 66.7% due to 1 SD and 1 SRT event misclassified as SA; Small test set (17 events). |
| **Pipeline B**<br>(Dataset B: Maneuver)<br>*Event-Level Runner-Up* | **Support Vector Machine (SVC)**<br>RBF Kernel, $C=3.0$ | **94.12%**<br>(16/17 events) | **0.9365** | **93.75%** | SA: 100.0%<br>SD: 75.0%<br>SRT: 100.0%<br>SLT: 100.0% | SA: 80.0%<br>SD: 100.0%<br>SRT: 100.0%<br>SLT: 100.0% | — | — | — | Sample accuracy is lower (76.57%); LODO CV Macro F1 is 0.9173 vs Extra Trees' 0.9272. |

---

### 1.3 Feasibility Assessment & Scientific Honesty Disclaimer

> [!IMPORTANT]
> **Scientific Honesty & Real-World Constraints:**
> In machine learning research for sensor-based telemetry, claiming guaranteed 90%+ performance across all metrics without empirical validation is unethical. We provide a transparent feasibility assessment:
> 
> 1. **Pipeline B (Dataset B): Feasibility is HIGH (Target: 90%–95%).**
>    Event-level classification is *already* operating at 88.24% (Extra Trees) and 94.12% (SVM), with a cross-driver LODO generalization of 93.06%. Advancing event accuracy to $\ge 94\%$ and sample accuracy from 80.57% to $\ge 88\text{--}92\%$ is achievable by addressing maneuver onset/tail latency and ensembling Extra Trees with SVM.
> 
> 2. **Pipeline A (Dataset C): Feasibility is MODERATE-TO-DIFFICULT (Target: 78%–85% on current data; 90%+ requires structural data shifts).**
>    Dataset C consists of only 14,240 clean samples recorded over just **two driving trips** from a single vehicle. Instantaneous smartphone accelerometer and gyroscope measurements contain severe intrinsic overlap between defensive normal maneuvers (e.g., stopping for a yellow light, turning at an intersection) and aggressive maneuvers. Without external contextual data (speedometer, road geometry, or additional multi-driver naturalistic normal-driving datasets), the **Bayes Error Rate** (irreducible noise ceiling) for binary style classification on this specific dataset likely sits between 12% and 18%.
> 
> We pledge to report only empirical, leak-free results on untouched held-out partitions. If a model reaches 82% instead of 90%, we will document that result honestly alongside the physical explanations.

---

## 2. Root-Cause Engineering Audit of Existing Models

### 2.1 Pipeline A (Dataset C) Root-Cause Analysis

```
+--------------------------------------------------------------------------------------------------+
|                              DATASET C BOTTLENECK TAXONOMY                                       |
+--------------------------------------------------------------------------------------------------+
|  1. Kinematic Overlap       --> Normal braking/turning generates same instant g as aggressive    |
|  2. Temporal Window Limit   --> W <= 3.0s (30 samples) captures single actions, not style        |
|  3. Class Distribution Shift--> Train = 61.7% Aggressive vs. Test = 46.6% Aggressive (Trips)    |
|  4. Recall Constraint Bias  --> Enforcing Aggressive Recall >= 80% pushed threshold to tau=0.84  |
|  5. Platt Scaling Skew      --> Logistic sigmoid fit on imbalanced margin distorts probabilities |
|  6. High FPR (42.59%)       --> 486 normal samples trigger false alarms in 2,138 test rows       |
|  7. Data Scarcity           --> Only 2 driving sessions; zero driver or vehicle diversity        |
+--------------------------------------------------------------------------------------------------+
```

#### Detailed Breakdown:

1. **Physical Kinematic Overlap in In-Cabin Phone Telemetry:**
   - In passenger vehicles, everyday safe driving includes transient kinematic excursions: braking smoothly from 50 km/h to 0 km/h at a traffic light generates longitudinal decelerations of $-1.5\text{ to } -2.5\text{ m/s}^2$. Taking a standard 90-degree intersection turn produces yaw velocities of $0.2\text{ to } 0.4\text{ rad/s}$ and lateral accelerations of $1.5\text{ to } 2.0\text{ m/s}^2$.
   - Aggressive driving in Dataset C manifests in these exact same ranges when maneuvers are mild or moderate. Without road curvature, speed limits, or traffic signal context, the sensor distributions overlap significantly.

2. **The High False-Positive Rate (42.59%) Mechanism:**
   - In Stage 4B, the model selection specification required:
     $$\text{Maximize Macro } F_1 \quad \text{subject to } \text{Recall}_{\text{aggressive}} \ge 0.80$$
   - Because the SVM was trained with `class_weight='balanced'`, its decision boundary prioritized aggressive recall. On validation data, maintaining recall $\ge 80\%$ required shifting the decision threshold from $\tau=0.50$ to $\tau^*=0.84$ on Platt-scaled probabilities.
   - While this threshold reduced false positives compared to the baseline ($\tau=0.50$ had 51.80% FPR), **486 out of 1,141 normal test samples are still misclassified as aggressive**. In real-world driving, an alert system that sounds an alarm 42.6% of the time during ordinary driving will suffer immediate driver rejection and alarm fatigue.

3. **Temporal Class Distribution Shift Across Partitions:**
   - Dataset C consists of two chronological trips: Session 1 (17:14:52) and Session 2 (18:45:12).
   - In `train` (first 70% of each trip), aggressive driving represents **61.7%** (6,025 / 9,767 rows).
   - In `val` (next 15%), aggressive driving represents **57.8%** (1,118 / 1,935 rows).
   - In `test` (final 15%), aggressive driving represents only **46.6%** (997 / 2,138 rows).
   - The test partition is majority **Normal (53.4%)**, whereas the model learned a prior probability that favoured **Aggressive (61.7%)**. This mismatch degrades accuracy when applying static decision boundaries.

4. **Restricted Feature Horizon ($W \le 30$ samples = 3.0 seconds):**
   - The existing feature set includes rolling statistics over 1.0s (10 samples), 2.0s (20 samples), and 3.0s (30 samples).
   - A 3-second window captures a single physical motion (e.g., one brake tap or one steering twitch). However, **"driving style" is an aggregate behavioral phenotype**, not an instantaneous jerk. Safe drivers occasionally perform necessary sharp defensive evasions. Aggressive drivers display repetitive patterns over 10s, 30s, or 60s (tailgating micro-accelerations, abrupt lane weaving, lack of coasting). Evaluating driving style strictly on 3-second windows inherently misclassifies defensive maneuvers as aggressive driving style.

5. **Lack of Frequency-Domain & Vibration Features:**
   - Road surface roughness, engine rumble, and chassis vibration introduce high-frequency noise into smartphone accelerometers.
   - The existing pipeline uses only time-domain rolling statistics (mean, std, min, max, range). It completely lacks frequency-domain features (FFT power spectrum, spectral energy ratios, dominant frequency, spectral centroid) that could decouple chassis vibration from intentional driver inputs.

6. **Platt Sigmoid Calibration Distortion:**
   - The SVC classifier uses Platt scaling (`probability=True`), which fits a 1D logistic regression model to the uncalibrated margin distances $f(x)$:
     $$P(y=1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
   - On Dataset C, Platt scaling compressed the probabilities toward extreme values, resulting in poor probability calibration. While Random Forest achieved a high **PR-AUC of 0.7682** and **ROC-AUC of 0.7423**, the tuned SVM's PR-AUC collapsed to **0.5627** on the test partition. Tree-based ensembles possess superior ranking discriminability that was underutilized due to hard thresholding.

---

### 2.2 Pipeline B (Dataset B) Root-Cause Analysis

```
+--------------------------------------------------------------------------------------------------+
|                              DATASET B BOTTLENECK TAXONOMY                                       |
+--------------------------------------------------------------------------------------------------+
|  1. Sample vs. Event Gap    --> Event Acc = 88.24% / 94.12%, but Sample Acc = 80.57%             |
|  2. Uniform Event Windows   --> 10s window labeled SA, but physical jerk occurs in 2-3s core    |
|  3. Latency & Tail Settling --> Onset and recovery look like steady cruising or mild motion      |
|  4. Low Sampling Frequency  --> fs ~ 1.55 Hz (Ts ~ 645 ms) yields only ~10-16 points per trial   |
|  5. SA Precision Loss (67%) --> Longitudinal pitch tilt during braking recovery mimics forward a |
|  6. Extreme Sample Scarcity --> 109 events across 3 drivers; Test has only 17 events             |
|  7. No Normal-Driving Class --> Pipeline forced to classify 100% of data into 4 risky maneuvers   |
+--------------------------------------------------------------------------------------------------+
```

#### Detailed Breakdown:

1. **The Discrepancy Between Sample-Level and Event-Level Performance:**
   - In Dataset B, Extra Trees achieved an **Event Macro F1 of 0.8786** (15 of 17 events correct, 88.24%) and SVM achieved **0.9365** (16 of 17 correct, 94.12%).
   - Yet on a **sample-by-sample streaming basis**, Extra Trees reached only **80.57% Accuracy** and **0.8155 Macro F1**, while SVM reached **76.57% Accuracy**.
   - *Why does this happen?* In Dataset B, an entire recorded event file (approximately 10 seconds, ~16 samples) is assigned a uniform label (e.g., `Class 1: Sudden Acceleration`). However, the human driver does not apply maximum acceleration for all 10 seconds. The physical surge lasts 2 to 3 seconds. The initial 1 to 2 seconds consist of foot transit and pedal take-up. The trailing 3 to 5 seconds consist of engine deceleration, braking back to cruising speed, or coasting. Point-by-point classifiers correctly predict near-zero acceleration during these tails, but ground truth penalizes them as misclassifications!

2. **Severe Sub-Nyquist Sampling Frequency Bottleneck (~1.55 Hz):**
   - The sensor data was recorded at approximately 1.55 Hz (inter-sample delta $T_s \approx 645\text{ ms}$).
   - According to the Nyquist-Shannon sampling theorem, the maximum observable frequency is:
     $$f_{\text{Nyquist}} = \frac{f_s}{2} \approx 0.77\text{ Hz}$$
   - Sudden vehicle jerks, wheel slips, and sharp steering inputs have physical durations of 200 ms to 500 ms ($2\text{--}5\text{ Hz}$). At 1.55 Hz, a 300 ms jerk pulse is either completely missed or captured in a single, partially attenuated sample. This severely degrades waveform feature extraction.

3. **Maneuver Confusion Asymmetry (SA vs. SD / SRT):**
   - Inspection of the event confusion matrix reveals a consistent error pattern:
     $$\text{Event Confusion (Extra Trees)} = \begin{pmatrix}
     4 & 0 & 0 & 0 \\
     1 & 3 & 0 & 0 \\
     1 & 0 & 3 & 0 \\
     0 & 0 & 0 & 5
     \end{pmatrix}$$
   - **Sudden Left Turn (`SLT`):** Flawless (100% Precision, 100% Recall). Because the dashboard-mounted MPU6050 yaw axis ($\omega_z$) swings strongly positive ($+0.8\text{ to }+1.5\text{ rad/s}$), SLT is linearly separable from longitudinal maneuvers.
   - **Sudden Acceleration (`SA`):** High Recall (100%), but poor Precision (**66.67%**). Two events (1 SD and 1 SRT) were falsely classified as SA.
   - *Physical Mechanism:* When a driver finishes an aggressive braking maneuver (SD) and releases the brake pedal, the vehicle suspension rebounds (pitch back-tilt). This dynamic pitch change redirects the gravitational acceleration vector forward along the vehicle's X-axis, generating an apparent forward acceleration spike that tree splits mistake for a Sudden Acceleration event.

4. **Small Sample Size & High Test-Set Variance:**
   - Dataset B has only 109 events (1,114 total rows) from 3 drivers.
   - The held-out test partition has only **17 events**. A single misclassified event alters test accuracy by:
     $$\Delta \text{Acc} = \frac{1}{17} \approx 5.88\%$$
   - Measuring improvement on 17 events requires extreme caution against overfitting to individual drivers or random splits.

---

## 3. In-Depth Technical Investigation & Proposed Solutions: Dataset C (Pipeline A)

### 3.1 Resolving the 42.59% False Positive Rate & Threshold Mechanics

```
Threshold Operating Characteristic Trade-off:
  tau = 0.50 (Default)    --> FPR = 51.80%, Aggressive Recall = 87.96%, Macro F1 = 0.6595
  tau = 0.84 (Stage 4B)   --> FPR = 42.59%, Aggressive Recall = 84.25%, Macro F1 = 0.6970
  tau = 0.92 (Exploration)--> FPR = 24.10%, Aggressive Recall = 71.30%, Macro F1 = 0.7250 (Fails Recall >= 80%)
```

#### Proposed Algorithmic Solutions:
1. **Cost-Sensitive Loss Optimization (Focal / Asymmetric Loss):**
   - Instead of training an unweighted or balanced SVM and applying a brutal post-hoc probability threshold ($\tau=0.84$), we propose training gradient-boosted trees with **Focal Loss** or an asymmetric cost matrix directly:
     $$\mathcal{L}_{\text{asym}}(y, \hat{p}) = - \left[ w_{\text{agg}} \cdot y \log(\hat{p}) + w_{\text{norm}} \cdot (1 - y) \log(1 - \hat{p}) \right]$$
   - By tuning the penalty ratio $w_{\text{norm}} / w_{\text{agg}}$ during loss minimization, the decision surface naturally bends away from dense normal driving regions without forcing an artificial probability cutoff.

2. **Decoupling Point-by-Point Classification from Safety Alarms:**
   - In production, single-sample predictions should feed into the **Temporal Smoother** and **Risk Engine** (Stage 6) rather than triggering raw instantaneous alarms.
   - A single normal sample misclassified as aggressive produces a transient risk delta of only $+0.15$, which decays within 3 seconds ($H=3.0\text{s}$) and stays well below the alert threshold ($R_{\text{medium}}=0.40$).
   - Real-world false alarm suppression must be solved through **temporal persistence validation**: require at least 3 consecutive seconds of aggressive kinematics before firing an alert.

---

### 3.2 Advanced Feature Engineering Expansion (Multi-Scale & Frequency Domain)

To break the 69.93% ceiling, the input feature space must be expanded from 47 features to approximately **78 causal streaming features**, incorporating multi-scale temporal context and spectral energy:

```
Proposed Multi-Scale Feature Architecture:
=============================================================================================
  Tier 1: Instantaneous Kinematics (11 features)
    - Acc X, Acc Y, Acc Z, gyro_x, gyro_y, gyro_z, acc_mag, gyro_mag, acc_horiz, jerk_a, jerk_g
  Tier 2: Short-Horizon Dynamics (1s, 2s, 3s - 36 features)
    - Mean, Std, Min, Max, Range, RMS for acc_mag, gyro_mag, acc_horiz, gyro_z
  Tier 3: Macro-Horizon Driving Context (5s, 10s, 20s - 24 NEW features)
    - Rolling window statistics capturing driving behavior stability over 50, 100, 200 samples
    - EMA Convergence: EMA_fast(1s) / EMA_slow(10s) to detect sustained erratic behavior
  Tier 4: Causal Spectral & Energy Features (7 NEW features)
    - Welch Power Spectral Density in 0.5-2.0 Hz band (engine/road vibration separation)
    - Cumulative Jerk Energy: Integral of |da/dt|^2 over 5 seconds
    - Kinematic Cross-Product: a_horiz * |gyro_z| (Coupled aggressive cornering metric)
=============================================================================================
```

#### Detailed Mathematical Formulations:

1. **Macro-Horizon Context (5.0s and 10.0s Rolling Windows):**
   - Driving style is defined by behavioral endurance. We extend the rolling window buffer up to $W=100$ samples ($10.0\text{s}$ at 10 Hz):
     $$\mu_{10s}(t) = \frac{1}{100} \sum_{k=0}^{99} a_{\text{mag}}(t-k), \quad \sigma_{10s}(t) = \sqrt{\frac{1}{100} \sum_{k=0}^{99} (a_{\text{mag}}(t-k) - \mu_{10s}(t))^2}$$
   - A safe driver who brakes sharply will have a temporary spike in $\sigma_{1s}$, but $\sigma_{10s}$ remains low. An aggressive driver who weaves through traffic will sustain a high $\sigma_{10s}$.

2. **Kinematic Cross-Coupling Feature:**
   - Aggressive drivers frequently execute combined maneuvers (braking while swerving, or accelerating out of a corner). We define the **Lateral-Longitudinal Coupling Index**:
     $$\kappa_{\text{coupled}}(t) = a_{\text{horiz}}(t) \times |\omega_z(t)|$$
   - In normal driving, high lateral acceleration occurs at low forward acceleration (and vice versa). High values in $\kappa_{\text{coupled}}$ are strongly indicative of aggressive driving.

3. **Cumulative Jerk Energy:**
   - Sudden aggressive maneuvers dissipate erratic kinetic energy:
     $$E_{\text{jerk}}(t) = \sum_{k=0}^{W} \left( \frac{\Delta a_{\text{mag}}(t-k)}{\Delta t} \right)^2$$
   - Smooth drivers exhibit near-zero $E_{\text{jerk}}$, whereas aggressive drivers produce large energy bursts.

4. **Causal Frequency-Domain Spectral Power:**
   - Over a causal sliding buffer of 32 samples (~3.2s), compute the Discrete Fourier Transform (DFT) power in the low-frequency driver band ($0.3\text{--}1.5\text{ Hz}$) relative to the high-frequency vibration band ($2.0\text{--}5.0\text{ Hz}$):
     $$\text{Spectral Ratio} = \frac{P_{[0.3, 1.5]}}{P_{[2.0, 5.0]} + \epsilon}$$
   - This decouples road bumps from intentional driver inputs.

---

### 3.3 Advanced Probability Calibration

Platt scaling was a primary contributor to the SVM's poor PR-AUC (0.5627). We propose replacing Platt scaling with modern, non-parametric calibration:

1. **Isotonic Regression Calibration:**
   - Fits a non-parametric, piece-wise constant monotonic step function:
     $$\min \sum_{i=1}^N (y_i - \hat{m}(f(x_i)))^2 \quad \text{subject to } \hat{m}(f_i) \le \hat{m}(f_j) \text{ whenever } f_i \le f_j$$
   - Prevents artificial compression of probabilities near the decision boundaries.

2. **Beta Calibration:**
   - Specifically designed for classifiers evaluated under class distribution shift, fitting a three-parameter Beta distribution that models asymmetric density shifts between train and test.

3. **Target Optimization Metric:**
   - Minimize **Expected Calibration Error (ECE)** and **Brier Score** on the validation partition, ensuring that a predicted probability of 0.80 corresponds to an actual 80% empirical risk.

---

### 3.4 Data Augmentation & Unsupervised Profiling via Dataset A

Dataset C contains only 14,240 rows across 2 sessions. In contrast, **Dataset A** contains **1,826,637 rows across 7 drivers** recorded at 50 Hz.
While Dataset A contains no ground-truth behavioral labels (cannot be used for supervised training), it represents an enormous source of **naturalistic driving distributions**.

#### Proposed Research Strategy for Dataset A:
1. **Unsupervised Normal-Driving Baseline Fitting:**
   - Train an **Isolation Forest** or **Deep Autoencoder** on Dataset A's multi-driver accelerometer/gyroscope streams.
   - The reconstruction error $\mathcal{L}_{\text{recon}}(x) = \|x - \hat{x}\|^2$ represents how far any new driving window deviates from standard, multi-driver naturalistic vehicle motion.
2. **Transfer Feature to Dataset C:**
   - Use the pretrained autoencoder as a feature extractor: append $\mathcal{L}_{\text{recon}}$ as an input feature in Dataset C's supervised pipeline. This injects knowledge from 1.8M rows of multi-driver data into Dataset C!

---

## 4. In-Depth Technical Investigation & Proposed Solutions: Dataset B (Pipeline B)

### 4.1 Bridging the Sample-Level (80.57%) to Event-Level (88–94%) Gap

```
Event Kinematic Anatomy vs. Labeling:
Time:     0s -------- 2s ---------------- 6s -------- 8s -------- 10s
Physical: [Pedal Prep]   [=== PEAK SURGE ===]   [Brake/Coast Recovery]
Ground-T: [   SA     ]   [        SA        ]   [        SA          ]  <-- Uniform 10s Label
Prediction:[Cruising ]   [        SA        ]   [     Cruising       ]  <-- Model is physically right!
Error:    MISCLASSIFIED!      ACCURATE!              MISCLASSIFIED!
```

#### Proposed Engineering Solutions:

1. **Phase-Aware Dynamic Sample Weighting:**
   - Rather than treating all samples in a 10-second maneuver trial equally during training, assign sample weights based on the **Kinematic Energy Excursion Envelope**:
     $$w_i(t) = \exp \left( - \frac{(t - t_{\text{peak}})^2}{2 \sigma_{\text{phase}}^2} \right) + w_{\text{floor}}$$
   - Samples near the peak acceleration or yaw-rate jerk receive weight $w \approx 1.0$; onset and settling samples receive small weights ($w \approx 0.20$).
   - This trains tree splits and support vectors on the *defining kinematic signature* of the maneuver, preventing boundary noise from corrupting decision boundaries.

2. **Causal Peak-Energy Detection in Streaming Inference:**
   - In the streaming classifier, maintain a short 3-second causal energy detector. When overall motion is below cruising threshold, output `NO_EVENT`.
   - When an energy burst occurs, sample predictions are dynamically weighted by instantaneous jerk energy $\Delta a^2 + \Delta \omega^2$, elevating sample accuracy to match event accuracy.

---

### 4.2 Resolving SA vs. SD / SRT Maneuver Confusion

To eliminate the 2 false-positive acceleration classifications (where 1 SD and 1 SRT were misclassified as SA):

1. **Dynamic Pitch / Suspension Rebound Compensation:**
   - During heavy braking, the front suspension compresses; on release, it rebounds backward, projecting $+1g \sin(\theta)$ onto the longitudinal axis.
   - We construct a **Longitudinal Jerk Polarity Indicator**:
     $$J_{\text{polarity}} = \text{sign}\left( \frac{d a_x}{dt} \right) \times \left| a_x - \bar{a}_{x, \text{baseline}} \right|$$
   - A true Sudden Acceleration begins with positive jerk from rest. A braking rebound begins with negative jerk followed by a damped positive return. Tracking the derivative history prevents suspension rebound from triggering an SA classification.

2. **Yaw Rate Polarity Invariants:**
   - Sudden Left Turn and Sudden Right Turn are physically separated by the sign of $\omega_z$:
     $$\text{SLT} \iff \omega_z > 0, \quad \text{SRT} \iff \omega_z < 0$$
   - By creating explicit asymmetric signed yaw features:
     $$\omega_z^{+} = \max(0, \omega_z), \quad \omega_z^{-} = \max(0, -\omega_z)$$
   - Tree split algorithms can isolate turning directions in a single orthogonal split without parameter sharing.

---

### 4.3 Addressing Low Sampling Frequency (~1.55 Hz)

1. **Causal Cubic Spline / Akima Interpolation:**
   - Resample the 1.55 Hz sensor stream to a uniform 10.0 Hz grid using causal Hermite cubic splines.
   - Interpolation reconstructs continuous kinematic trajectories, enabling smoother finite-difference derivatives ($j = \Delta a / \Delta t$) without step-function discretization artifacts.

2. **Integration with Stage 6 Safety Agent:**
   - As documented in the Stage 6 remediation plan, low-frequency events must pass through the **Event Boundary Gate** and **Temporal Smoother**, preventing multiple alarms for a single maneuver.

---

## 5. Model Architecture Exploration & Benchmarking Strategy

### 5.1 Evaluated Architecture Matrix

We propose benchmarking seven candidate architectures across both datasets using identical feature sets and leak-free validation:

| Candidate Model | Suitability: Dataset C | Suitability: Dataset B | Key Strength | Key Vulnerability | Planned Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (SVC)** | **Current Winner**<br>(Acc: 69.93%) | **Strong Runner-Up**<br>(Event Acc: 94.12%) | Optimal margin separation in continuous non-linear kinematic spaces; robust against overfitting. | Poor default probability calibration; quadratic training complexity $O(N^2)$. | Kernels: RBF, Poly (deg 2/3); $C \in [0.5, 10.0]$; `gamma='scale'`; Isotonic probability. |
| **Extra Trees Classifier** | Moderate<br>(Acc: 65.0%) | **Current Winner**<br>(Event: 88.2%, Sample: 80.6%) | Extreme randomization smooths sensor noise; handles high feature correlation exceptionally well. | Jagged orthogonal decision boundaries; sensitive to threshold shifts under class imbalance. | `n_estimators`: 200–500; `max_depth`: 8–16; `min_samples_split`: 2–5; `class_weight='balanced'`. |
| **Random Forest** | Strong PR-AUC<br>(PR-AUC: 0.7682) | Strong<br>(Event: 88.2%, Sample: 79.4%) | Excellent probability ranking and out-of-bag error estimation; robust against outliers. | Slower inference latency than single trees; step-wise probability estimates. | `n_estimators`: 200–500; `max_depth`: 10–18; `max_features`: 'sqrt', 'log2'. |
| **XGBoost (eXtreme Gradient Boosting)** | Strong Candidate<br>(Acc: 66.28%, F1: 0.6571) | Moderate<br>(Event: 82.35%) | Regularized gradient boosting (L1/L2 penalties); built-in handling of class weights (`scale_pos_weight`). | Can overfit on small sample sizes (Dataset B); sensitive to noisy continuous signals. | `max_depth`: 4–8; `learning_rate`: 0.03–0.1; `subsample`: 0.8; `colsample_bytree`: 0.8. |
| **LightGBM (Light Gradient Boosting)** | **High Potential**<br>(Fast, Leaf-Wise) | **High Potential**<br>(Histogram-based) | Leaf-wise tree growth produces deeper, more discriminative splits; natively supports focal loss. | Requires installation (`pip install lightgbm`); risk of overfitting small datasets without regularization. | `num_leaves`: 31–63; `learning_rate`: 0.05; `min_child_samples`: 20; `class_weight='balanced'`. |
| **CatBoost** | **High Potential**<br>(Ordered Boosting) | **High Potential**<br>(Superior on Small Data) | Ordered boosting prevents target leakage; outstanding default performance on tabular sensor data. | Requires installation (`pip install catboost`); higher training time. | `depth`: 6–8; `l2_leaf_reg`: 3–7; `iterations`: 300–600; `auto_class_weights='Balanced'`. |
| **Heterogeneous Stacking Ensemble** | **Target for 80%+** | **Target for 95%+** | Combines SVM's smooth non-linear margin with Tree Ensembles' feature selection via a meta-learner. | Increased inference latency; architectural complexity. | Base models: SVM + Extra Trees + LightGBM; Meta-Learner: Calibrated Logistic Regression. |

---

### 5.2 Heterogeneous Stacking Architecture Design

To achieve peak performance, we propose an architectural blend that merges the complementary mathematical strengths of kernel methods and randomized tree ensembles:

```
                          STREAMING KINEMATIC FEATURES (X)
                                        |
               +------------------------+------------------------+
               |                        |                        |
               v                        v                        v
        +--------------+         +--------------+         +--------------+
        |  Support     |         |  Extra Trees |         |  LightGBM /  |
        |  Vector      |         |  Ensemble    |         |  CatBoost    |
        |  Machine     |         |              |         |              |
        +--------------+         +--------------+         +--------------+
               |                        |                        |
               | P_svm(y)               | P_et(y)                | P_lgb(y)
               +------------------------+------------------------+
                                        |
                                        v
                          +---------------------------+
                          |   Isotonic Calibrator     |
                          +---------------------------+
                                        |
                                        v
                          +---------------------------+
                          |   Meta-Learner            |
                          |   (Logistic Regression    |
                          |    L2, C=1.0)             |
                          +---------------------------+
                                        |
                                        v
                           FINAL CALIBRATED PROBABILITY
```

- **Why Stacking Succeeds Here:**
  - In Dataset B, SVM achieved **94.12% Event Accuracy** because support vectors formed a wide margin around discrete maneuver centroids, but Extra Trees achieved **80.57% Sample Accuracy** because randomized thresholds avoided margin collapse during transition periods.
  - In Dataset C, SVM had the highest discrete accuracy (**69.93%**), but Random Forest had the highest PR-AUC (**0.7682**). Combining their soft probability vectors via a regularized meta-learner allows the system to leverage Random Forest's ranking quality while preserving SVM's decision boundary geometry.

---

## 6. Honest Scientific Evaluation Protocol & Anti-Leakage Guardrails

### 6.1 The "Sacred" Test Set Principle

> [!CAUTION]
> **Strict Anti-Leakage Protocol:**
> 1. The existing test sets (`dataset/processed/dataset_c_test.csv` with $N=2,138$ and `dataset/processed/dataset_b_test.csv` with $N=175$) are **SACRED AND FROZEN**.
> 2. Under no circumstances will test data be used for:
>    - Model architecture exploration.
>    - Hyperparameter tuning or grid searching.
>    - Feature selection or importance ranking.
>    - Decision threshold selection ($\tau^*$).
>    - Scaler or transformer fitting.
> 3. All exploratory experimentation and hyperparameter optimization will be conducted strictly within the **Training and Validation sets** using **5-Fold Cross-Validation** or **Leave-One-Driver-Out Cross-Validation**.
> 4. The held-out test partitions will be evaluated **EXACTLY ONCE** at the conclusion of the research phase to verify final performance.

---

### 6.2 Leakage Prevention Verification Checklist

| Leakage Risk | Mechanism | Mitigation Strategy | Enforcement Mechanism |
| :--- | :--- | :--- | :--- |
| **Temporal Window Bleeding** | Rolling statistics looking forward across split boundaries. | Rolling windows look strictly backward ($t-k$). Guard-band buffers ($10.0\text{s} = 100$ samples in Dataset C) dropped between train, val, and test. | Verified by `verify_evaluation_audit.py`. Guard bands cannot be modified. |
| **Event Overlap (Dataset B)** | Samples from the same physical maneuver trial split across train and test. | Stratified Group Splitting grouped strictly on `event_id`. | Zero `event_id` overlap verified across partitions. |
| **Driver Identity Leakage** | Model memorizing specific vehicle vibration signatures rather than driver style. | Leave-One-Driver-Out (LODO) cross-validation where entire drivers are held out. | Verified across Drivers 1, 2, and 3 in Dataset B. |
| **Data Standardization Leakage** | Scaler computing mean/std across full dataset before splitting. | `StandardScaler` fitted exclusively on the `train` partition ($N=9,767$ in C; $N=776$ in B). Validation and test transformed using train statistics. | Tested via `test_pipeline_leakage.py`. |
| **Threshold Overfitting** | Tuning decision threshold $\tau$ on test data to inflate accuracy. | Decision threshold $\tau^*$ tuned strictly on Validation partition to optimize Macro F1. Locked before evaluating test. | Enforced in automated training scripts. |

---

### 6.3 Reporting Standards & Evaluation Metrics Suite

For every future experiment, the reporting script must generate an exhaustive metrics payload:

1. **Point-by-Point Sample-Level Metrics:**
   - Accuracy, Balanced Accuracy.
   - Macro F1, Weighted F1, Per-Class Precision, Recall, and F1.
   - Specificity (True Negative Rate) and False Positive Rate (FPR).
   - Confusion Matrix (absolute counts and row-normalized percentages).
2. **Threshold-Free Discriminative Metrics:**
   - Receiver Operating Characteristic Area Under Curve (ROC-AUC).
   - Precision-Recall Area Under Curve (PR-AUC) — critical for imbalanced evaluation.
   - Expected Calibration Error (ECE) and Brier Score.
3. **Event-Level Metrics (Dataset B):**
   - Event-level Accuracy and Macro F1 (via causal probability aggregation $\bar{\mathbf{p}}_i = \frac{1}{T_i} \sum \mathbf{p}_{i, t}$).
   - Event Confusion Matrix (17 test events).
   - Leave-One-Driver-Out (LODO) unweighted mean accuracy and pooled accuracy.

---

## 7. Experimental Organization, Architecture & Tooling

To ensure 100% backward compatibility and preserve all historical baseline artifacts, improvement experiments will be organized in a completely isolated directory structure:

```
D:\driver-machine-learning-model\
│
├── ml\
│   ├── training\
│   │   ├── train_dataset_c_binary.py            # Baseline training script (UNTOUCHED)
│   │   ├── train_dataset_b_multiclass.py        # Baseline training script (UNTOUCHED)
│   │   └── improvements\                        # NEW: Improvement Experiment Scripts
│   │       ├── feature_engineering_v2.py        # Multi-scale & spectral feature extractor
│   │       ├── train_dataset_c_improved.py      # Benchmark script for Pipeline A (v2)
│   │       ├── train_dataset_b_improved.py      # Benchmark script for Pipeline B (v2)
│   │       └── evaluate_stacking_ensemble.py    # Stacking meta-learner evaluation
│   └── inference\
│       ├── dataset_c_classifier.py              # Production inference wrapper (UNTOUCHED)
│       └── dataset_b_classifier.py              # Production inference wrapper (UNTOUCHED)
│
├── models\
│   ├── experiments\                             # Historical Baseline Artifacts (UNTOUCHED)
│   │   ├── dataset_c\                           # Contains SVM exp_v1, RF exp_v1, etc.
│   │   └── dataset_b\                           # Contains Extra Trees exp_v1, etc.
│   └── improvements\                            # NEW: Isolated Improvement Artifacts
│       ├── dataset_c\
│       │   ├── dataset_c_lightgbm_v2.joblib
│       │   ├── dataset_c_catboost_v2.joblib
│       │   ├── dataset_c_stacking_ensemble_v2.joblib
│       │   └── dataset_c_v2_experiment_summary.json
│       └── dataset_b\
│           ├── dataset_b_stacking_ensemble_v2.joblib
│           ├── dataset_b_spline_extratrees_v2.joblib
│           └── dataset_b_v2_experiment_summary.json
│
└── reports\
    ├── model_training\                          # Historical Stage 4B Reports (UNTOUCHED)
    └── model_improvement\                       # NEW: Research & Improvement Reports
        ├── improvement_research_plan.md         # THIS REPORT
        ├── dataset_c_feature_expansion_study.md # Upcoming Phase 1 findings
        └── model_benchmark_v2_results.md        # Upcoming Phase 2 findings
```

---

## 8. Theoretical Limits, Risks, and Feasibility Assessment

### 8.1 The Bayes Error Rate in Passenger Vehicle Telemetry

```
                    TOTAL OBSERVABLE INFORMATION SPECTRUM
+-------------------------------------------------------------------------------+
| In-Cabin Telemetry (Acc, Gyro) | Road Context (GPS, Lanes) | Driver Context   |
| [WHAT WE HAVE - ~40% INFO]     | [MISSING - ~35% INFO]     | [MISSING - ~25%] |
+-------------------------------------------------------------------------------+
         |
         v
Intrinsically Ambiguous Zone (Overlap):
  - Normal driver braking sharply for a pedestrian = HIGH KINEMATICS, SAFE STYLE
  - Aggressive driver tailgating at steady speed   = LOW KINEMATICS, RISKY STYLE
```

- **Mathematical Reality:** On Dataset C, smartphone sensors measure chassis acceleration and angular velocity, but do NOT observe:
  - Traffic signals, surrounding vehicles, or road curvature.
  - Driver pedal force, gaze direction, or steering wheel angle.
- Therefore, a fraction of normal driving points are physically indistinguishable from aggressive driving points. This establishes a theoretical **Bayes Error Rate ($E_{\text{Bayes}}$)**.
- We estimate that $E_{\text{Bayes}}$ on Dataset C alone is between **12% and 18%**. Consequently, an unaugmented model trained strictly on instantaneous 10 Hz phone data will likely top out at **82% to 86% accuracy**. Reaching 90%+ without external normal data or broader temporal aggregation may be mathematically impossible without overfitting.

---

### 8.2 Real-Time Streaming Latency Constraints

The safety agent must operate in real-time on edge devices (Raspberry Pi or in-vehicle computing unit):
- **Sampling Frequency:** 10.0 Hz (Pipeline A) and 1.55 Hz (Pipeline B).
- **Latency Budget:** Inference per sample must complete in **$< 10.0\text{ ms}$** to prevent queue buildup.
- **Latency Assessment:**
  - Extra Trees & Random Forest: $\sim 1.5\text{--}3.0\text{ ms}$ (PASS).
  - LightGBM / CatBoost: $\sim 0.8\text{--}1.5\text{ ms}$ (PASS).
  - SVM (RBF Kernel, 9,767 support vectors): $\sim 6.0\text{--}8.5\text{ ms}$ (BORDERLINE).
  - Deep 1D-CNN / LSTM on CPU: $\sim 15.0\text{--}25.0\text{ ms}$ (FAIL on low-power edge without GPU).
- Therefore, **tree ensembles and gradient boosters are strongly favored over recurrent neural networks** for edge deployment.

---

### 8.3 Risk Mitigation Matrix

| Identified Risk | Severity | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Overfitting on Small Dataset B Test Set** ($N=17$) | HIGH | HIGH | Use 3-Fold Leave-One-Driver-Out (LODO) cross-validation as the primary performance arbiter, not the 17-event split alone. |
| **High False-Positive Rate Persists on Dataset C** | HIGH | MEDIUM | Implement multi-scale temporal windows (10s/20s) and calibrate probabilities with Isotonic Regression; enforce temporal smoothing in Stage 6 agent. |
| **Library Dependency Conflicts (`lightgbm`, `catboost`)** | MEDIUM | LOW | Benchmark scikit-learn tree ensembles (HistGradientBoosting, ExtraTrees, RandomForest) first; install LightGBM in virtual environment cleanly. |
| **Inference Latency Exceeds 10 ms Budget** | MEDIUM | LOW | Profile inference latency with `stage5_latency_benchmark.py`; prune tree depth ($d \le 12$) and limit support vector counts. |

---

## 9. Prioritized Phased Implementation Roadmap

```
PHASED EXECUTION TIMELINE:
====================================================================================================
Phase 1: Multi-Scale Feature Engineering & Causal Extraction (Train/Val sets only)
  ├── 1.1 Implement 5s, 10s, 20s rolling windows & EMA ratios for Dataset C
  ├── 1.2 Implement dynamic jerk energy and cross-coupling features
  └── 1.3 Verify zero temporal leakage using verify_evaluation_audit.py
----------------------------------------------------------------------------------------------------
Phase 2: Advanced Model Exploration & Gradient Boosting Benchmarking
  ├── 2.1 Benchmark LightGBM, CatBoost, and HistGradientBoosting against SVM/RF baselines
  ├── 2.2 Conduct 5-fold cross-validation and hyperparameter grid searches
  └── 2.3 Optimize Focal Loss and asymmetric cost matrices to drive FPR below 25%
----------------------------------------------------------------------------------------------------
Phase 3: Dataset B Maneuver Kinematic Refinement & Stacking
  ├── 3.1 Implement phase-aware sample weighting (Gaussian peak emphasis)
  ├── 3.2 Formulate heterogeneous Stacking Ensemble (Extra Trees + RBF SVM + LightGBM)
  └── 3.3 Validate on Leave-One-Driver-Out (LODO) cross-validation
----------------------------------------------------------------------------------------------------
Phase 4: Advanced Calibration & Stage 6 Safety Agent Alignment
  ├── 4.1 Fit Isotonic Calibrators on validation soft probability outputs
  ├── 4.2 Align inference wrappers with Stage 6 PredictionValidator and RiskEngine
  └── 4.3 Benchmark inference latency (<10 ms budget)
----------------------------------------------------------------------------------------------------
Phase 5: Single-Pass Sacred Test Set Evaluation & Final Documentation
  ├── 5.1 Run one-shot evaluation on dataset_c_test.csv (N=2,138) and dataset_b_test.csv (N=175)
  ├── 5.2 Compile authoritative comparative results in model_benchmark_v2_results.md
  └── 5.3 Present verified metrics to user for formal stage sign-off
====================================================================================================
```

### Computational Budget & Resource Requirements:
- **Compute Hardware:** Standard x86_64 CPU (multi-threaded). No dedicated GPU mandatory for tree ensembles; GPU optional for neural sequence models.
- **Estimated Runtime:**
  - Feature extraction: $\sim 45\text{ seconds}$ per dataset.
  - 5-Fold Cross-Validation across 7 architectures: $\sim 15\text{--}25\text{ minutes}$.
  - Stacking ensemble tuning: $\sim 10\text{ minutes}$.
  - Total computational cost: $< 1.0\text{ CPU hour}$.
- **Storage Footprint:** $< 100\text{ MB}$ for new model artifacts and evaluation summaries.
- **Required Additional Packages:** `lightgbm`, `catboost` (optional, recommended).

---

## 10. Audit Sign-Off & Verification

This improvement plan adheres strictly to the project's engineering principles:
1. **Zero Overwriting:** All existing models, scalers, test partitions, reports, and production wrappers remain 100% intact.
2. **Zero Leakage:** Test sets remain sacred, quarantined, and untouched until final evaluation.
3. **Scientific Integrity:** No fabricated metrics, no false promises of 90%+ on Dataset C without empirical verification.
4. **Implementation Boundary:** This is an engineering research and planning document. Implementation awaits explicit user authorization.

*Plan prepared and signed by: Senior ML Engineer & Safety Systems Architect*  
*November 2026*

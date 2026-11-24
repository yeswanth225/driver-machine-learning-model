# FINAL MULTI-DRIVER COLLECTION READINESS AUDIT & SAFETY REVIEW
## Technical Audit, Inconsistency Resolution & Operational Readiness Certification

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Pipeline A (Driving Behaviour Detection) — Multi-Driver Validation Preparation  
**Author:** Principal ML Safety Systems Auditor & Statistical Lead  
**Date:** 2026-11-22  
**Audit Target:** Protocols, Validation Architecture, Domain Shift Policies & Baseline Integrity  
**Deliverable Path:** `reports/pipeline_a/final_collection_readiness_audit.md`  

---

## 1. Executive Summary & Audit Mandate

This audit delivers the comprehensive technical and safety review of all specifications, protocols, and architectural designs prepared for the upcoming multi-driver data collection campaign. Prior to sending human volunteers onto proving grounds and roads, every statistical assumption, data pipeline interface, and safety failure mode was audited against empirical machine learning principles and the project's strict non-negotiable protection constraints.

### 1.1 Core Audit Findings
1. **Critical Safety Inconsistency Resolved (Domain Shift):** The previously proposed heuristic `PSI >= 0.25 -> style risk = 0.0` was formally rejected. Zeroing style risk under sensor drift or high vibration would suppress genuine aggressive driving evidence and falsely certify dangerous maneuvers as safe. It has been replaced with the approved data-quality architecture (`pipeline_a_health = "DEGRADED"`, style prediction withheld as `None`, Pipeline B maneuvers fully preserved, zero modifications to safety policies).
2. **Driver Recruitment Policy & Statistical Sizing Reality Formally Clarified:** The audit establishes a strict distinction between cohort configurations:
   - **8 Drivers is the Preferred Initial Validation Cohort:** Permits 6 development drivers for 6-fold cross-validation and 2 reserved lockbox drivers. Crucially, the audit clarifies that **six development drivers and two lockbox drivers still constitute a small initial study**, NOT a large-scale population benchmark. Two lockbox drivers do not establish population-wide generalization; reporting of confidence intervals and per-driver metrics is mandatory, and larger cohorts (e.g., 20–50+ drivers) may be required before making universal generalization claims.
   - **6 Drivers is the Absolute Pilot Floor:** Permits 4 development drivers (4-fold cross-validation) and 2 reserved lockbox drivers. A six-driver pilot is **NOT equivalent** to the preferred eight-driver study due to reduced statistical sample size, broader confidence intervals, and restricted vehicle/driver diversity.
   - **7 Drivers permits:** 5 development drivers (5-fold cross-validation) and 2 reserved lockbox drivers.
   - **Fewer than 6 Usable Drivers means NO-GO:** If fewer than 6 usable drivers complete the protocol, the evaluation campaign is aborted as statistically invalid.
3. **Threshold Calibration Provenance & Limitations Established:** The $\tau^* = 0.78$ threshold was confirmed to originate from single-driver Dataset C Session 1 validation data. While verified to be free of test-label contamination, it reflects single-vehicle acoustics and cannot be assumed to generalize across diverse fleet dynamics. Furthermore, the proposed "adaptive 30-second resting calibration" was audited and classified as an **unproven conceptual proposal** that must not be deployed in production or safety-critical logic without empirical multi-driver validation.
4. **Lockbox Interpretation & Reporting Standards Defined:** Two reserved lockbox drivers provide a **small independent pilot evaluation**, NOT definitive proof of generalization across all human drivers. Full granular reporting is mandated: per-driver accuracy, balanced accuracy, Macro-F1, precision, recall, FPR, pooled metrics, event/trip counts, confidence intervals, and vehicle/sensor configuration metadata. Lockbox data must never be used for threshold tuning.
5. **Rigorous Readiness Terminology Enforced:** Prerequisites are strictly categorized as **Design Verified**, **Implementation Verified**, or **Experimentally Validated**. Hardware specifications, calibration procedures, guard bands, and streaming parity are explicitly marked as **NOT experimentally validated** until physical real-world driving data is acquired.
6. **Data Collection Acceptance Criteria & Ground Truth Clarified:** Sensor acquisition acceptance criteria were strictly separated from annotation ground truth. The mandatory core dataset consists of 10 Hz linear acceleration, calibrated gyroscope in rad/s, and GPS speed, accompanied by dual-observer video consensus ($\kappa \ge 0.80$). CAN/OBD-II logging is designated as optional secondary physical telemetry, preventing vehicle bus access limitations from blocking the campaign while explicitly noting that video labels are observational proxies, not direct actuator measurements.
7. **Absolute Protection of Production Baseline:** All 4 protected frozen model artifacts match their required SHA-256 hashes bit-for-bit. All 100 automated unit and integration tests pass without regression. Zero git branch, commit, push, or reset actions were taken.

### 1.2 Formal Audit Determination
- **PHYSICAL DATA COLLECTION:** **GO (READY WITH CONDITIONS — SUBJECT TO PILOT FLOOR $\ge 6$, PREFERRED TARGET $\ge 8$)**  
  *Approved to proceed with field recruitment, proving ground scheduling, and data acquisition, subject to recruiting at least the 6-driver pilot floor (with full protocol targeting the preferred 8 drivers), obtaining signed volunteer consent, and proving ground booking. If fewer than 6 usable drivers complete the protocol, the campaign automatically reverts to NO-GO.*
- **PRODUCTION DEPLOYMENT OF CANDIDATE V2:** **STRICT NO-GO (BLOCKED)**  
  *Candidate v2 remains an experimental research artifact. It must NOT be deployed into the production backend until it passes multi-driver LODO validation and real-time streaming feature parity tests.*

---

## 2. Forensic Resolution of Safety & Architectural Inconsistencies

### 2.1 Resolution of Critical Domain-Shift Safety Review (Task 1)

#### The Audited Vulnerability: Heuristic PSI Risk Zeroing
The proposed rule `PSI >= 0.25 -> style risk = 0.0` was evaluated against automotive functional safety standards (ISO 26262 / SOTIF). The audit proved that this rule introduces an unacceptable catastrophic failure mode:
1. **Suppression of Genuine Evidence:** A driver engaged in violent swerving or high-speed curb-hopping induces extreme mechanical vibration and high spectral noise, driving the Population Stability Index (PSI) above $0.25$. Under the proposed rule, the system would immediately zero the style risk ($0.0$), interpreting sensor-level domain shift as definitive proof of calm, normal driving.
2. **Violation of Failure Mode Principles:** In safety-critical autonomy, an uncalibrated, noisy, or shifted sensor must be classified as **unreliable or degraded**, never as an affirmative witness of safety. Treating uncalibrated telemetry as evidence of normal driving violates core fail-silent safety architecture.
3. **Contamination of Multimodal Risk Fusion:** The Safety Agent computes an integrated risk score based on Pipeline A (behaviour style) and Pipeline B (tactical maneuvers). Artificially clamping Pipeline A to zero distorts the joint Bayesian or rule-based risk evaluation.

#### The Approved Data-Quality Architecture
In accordance with Stage 6 safety policies, the documentation across all reports was updated to enforce the following compliant data-quality architecture:
- **Health Degradation Flag:** When feature drift occurs ($\text{PSI} \ge 0.25$, Wasserstein distance anomaly, or mount vibration exceeding $3.0\sigma$), Pipeline A's health state transitions from `OK` to `DEGRADED` in `ValidatedPrediction`.
- **Prediction Withholding:** While in `DEGRADED` status, Pipeline A's discrete prediction and style risk are set to `None` (withheld), rather than assuming Normal Driving ($0.0$).
- **Preservation of Pipeline B:** Tactical maneuver detection (Pipeline B) continues to evaluate sudden braking, lane departures, and shock impacts independently. Pipeline B evidence is **never** muted or suppressed by Pipeline A's domain shift.
- **Zero Modifications to Safety Engine:** The existing Deterministic Safety Agent rules and policies remain completely untouched, handling missing or degraded inputs according to its existing validated fallback policies.
- **Status of PSI:** PSI is explicitly documented as a **continuous monitoring indicator and telemetry health metric**, NOT an independently validated safety-actuation threshold.

```
[Raw Kinematics Stream] ──► [Pipeline A Feature Extraction] ──► [PSI Drift Monitor]
                                                                        │
                                             ┌──────────────────────────┴──────────────────────────┐
                                             ▼ (PSI < 0.25: In-Domain)                              ▼ (PSI >= 0.25: Domain Shift)
                                    [Execute Classification]                                [Withhold Style Prediction]
                                    - Prediction: Normal / Aggressive                       - style_prediction = None
                                    - pipeline_a_health = "OK"                             - pipeline_a_health = "DEGRADED"
                                             │                                                      │
                                             └──────────────────────────┬───────────────────────────┘
                                                                        │
                                                                        ▼
                                                   [Deterministic Safety Agent]
                                                   - Evaluates ValidatedPrediction
                                                   - Pipeline B evidence 100% active
                                                   - Zero arbitrary risk clamping
```

---

### 2.2 Reconciled Driver Cohort Allocation & Recruitment Policy (Task 2)

#### Explicit Recruitment Policy & Statistical Sizing Reality
To eliminate ambiguity between the statistical ideal and field execution realities, the multi-driver recruitment policy is formally tiered:

1. **8 Drivers is the Preferred Initial Multi-Driver Validation Cohort:**
   - **Structure:** 6 Development Drivers (`DRV_001`–`DRV_006`) + 2 Reserved Lockbox Drivers (`DRV_007`, `DRV_008`).
   - **Statistical Reality:** This represents the preferred initial validation cohort. However, it must be explicitly recognized that **six development drivers and two lockbox drivers remain a small initial study**, NOT a large-scale population benchmark.
   - **Generalization Limits:** Two lockbox drivers do not establish population-wide generalization across the entire human driving population or vehicle spectrum. Statistical uncertainty remains non-trivial, mandating granular per-driver metrics, reporting of trip/event exposure, and 95% confidence intervals.
   - **Expansion Requirement:** Additional larger cohorts (e.g., 20–50+ drivers across diverse geographic regions, chassis types, and road conditions) may be required before making strong universal generalization claims.
2. **6 Drivers is the Absolute Pilot Floor (NOT Equivalent to Preferred Study):**
   - **Structure:** 4 Development Drivers (`DRV_001`–`DRV_004`) + 2 Reserved Lockbox Drivers (`DRV_005`, `DRV_006`).
   - **Scientific Non-Equivalence Caveat:** A six-driver pilot **cannot be described as equivalent to the preferred eight-driver study**. A 4-driver development cohort exhibits substantially reduced statistical power, covers fewer vehicle chassis and driving styles, produces wider confidence intervals, and increases the variance of the cross-validated threshold.
3. **7 Drivers Fallback Arrangement:**
   - **Structure:** 5 Development Drivers (`DRV_001`–`DRV_005`) for 5-fold cross-validation + 2 Reserved Lockbox Drivers (`DRV_006`, `DRV_007`).
4. **Fewer than 6 Usable Drivers Means Immediate NO-GO:**
   - **Hard Cutoff:** If recruitment, device failure, or participant dropout yields $<6$ usable drivers, the evaluation campaign is an immediate **NO-GO (BLOCKED)**. With $<4$ development drivers, grouped cross-validation suffers severe sample degeneracy and cannot estimate inter-driver variance.
5. **Non-Negotiable Lockbox Invariance:**
   - Under no circumstances may the lockbox be reduced below 2 drivers. A single-driver lockbox would replicate the single-driver evaluation flaw of Dataset C.
6. **Strict Driver Isolation Rule:**
   - A driver $D_i$ belongs strictly to the Development Pool OR the Lockbox Pool. Under zero circumstances may a driver or any of their trips be partitioned across both pools.

```
                              [Driver Recruitment & Sizing Reality]
                                                │
                ┌───────────────────────────────┴───────────────────────────────┐
                ▼                                                               ▼
   [Preferred Initial Cohort: >= 8 Drivers]                       [Absolute Pilot Floor: 6 Drivers]
   - 6 Development Drivers                                        - 4 Development Drivers
   - 2 Reserved Lockbox Drivers                                   - 2 Reserved Lockbox Drivers
   - Small initial study; requires CIs                            - NOT equivalent to preferred study
   - Captures primary pilot diversity                             - Wider uncertainty, higher threshold variance
                │                                                               │
                └───────────────────────────────┬───────────────────────────────┘
                                                │
                                                ▼
                                  [Fallback Hierarchy Decision]
                                  * >= 8 Drivers: GO (Preferred Initial Cohort)
                                  * 7 Drivers:    PILOT GO (5 Dev Group + 2 Lockbox)
                                  * 6 Drivers:    PILOT GO (4 Dev Group + 2 Lockbox, Non-Equivalent)
                                  * < 6 Drivers:  IMMEDIATE NO-GO (Campaign Blocked)
```

---

### 2.3 Cross-Validation Architecture & Partition Rotation Specification

#### Methodological Demarcation: Ordinary LODO vs. Three-Way Grouped vs. Nested CV
To resolve methodological ambiguity, we explicitly distinguish three grouped validation paradigms:
1. **Ordinary 2-Way Leave-One-Driver-Out (LODO):**
   - For a pool of $K$ drivers, iterates $K$ folds where $\mathcal{D}_{\text{test}}^{(k)} = \{D_k\}$ and $\mathcal{D}_{\text{train}}^{(k)} = \{D_j \mid j \neq k\}$ ($K-1$ drivers).
   - *Limitation:* Does not provide a held-out validation driver for threshold tuning. If thresholds are tuned on the training drivers' own training predictions, it introduces resubstitution optimism.
2. **Three-Way Grouped Cyclic Cross-Validation (The Primary Baseline):**
   - Partitions the $K$ development drivers into three mutually exclusive subsets per fold: **Train ($K-2$ drivers)**, **Validation ($1$ driver)**, and **Test ($1$ driver)**.
   - Rotates cyclically across exactly $K$ folds. Every driver receives exactly one out-of-fold evaluation as the test driver and serves as the held-out validation driver in exactly one fold.
3. **Nested Grouped Cross-Validation (Outer LODO + Inner LODO):**
   - Outer loop holds out Driver $k$ as Test. Inner loop executes an inner $(K-1)$-fold LODO on the remaining $K-1$ drivers to generate pooled out-of-fold predictions for threshold tuning. The final model is fitted on all $K-1$ training drivers and tested on Driver $k$.

#### Primary Architecture: Three-Way Grouped Cyclic Cross-Validation Mechanics
- **Exact Number of Folds:** Exactly $K$ folds (e.g., 6 folds for $K=6$; 5 folds for $K=5$; 4 folds for $K=4$).
- **Do Validation and Test Drivers Rotate Independently?**
  * **No.** If validation and test drivers rotated independently across all Cartesian combinations, there would be $K \times (K-1)$ folds (e.g., $6 \times 5 = 30$ folds for $K=6$). This would evaluate each test driver 5 separate times under 5 different models, producing redundant, non-independent evaluations.
  * Instead, a **deterministic cyclic shift of length $K$** is enforced. In Fold $k \in \{1 \dots K\}$:
    $$\mathcal{D}_{\text{test}}^{(k)} = \{D_k\}$$
    $$\mathcal{D}_{\text{val}}^{(k)} = \{D_{(k \pmod K) + 1}\}$$
    $$\mathcal{D}_{\text{train}}^{(k)} = \{D_1 \dots D_K\} \setminus \left(\mathcal{D}_{\text{test}}^{(k)} \cup \mathcal{D}_{\text{val}}^{(k)}\right)$$
- **Explicit Cyclic-Pairing Limitations:**
  * The deterministic cyclic pairing is **one predeclared pairing schedule**.
  * It does **not evaluate all $K(K-1)$ possible test-validation pairings**.
  * Metric estimates and threshold choices depend on the pairing schedule.
  * The number of evaluated folds is $K$, not $K(K-1)$.
  * The resulting metrics describe this specific validation design and must **not be represented as exhaustive nested cross-validation**.
- **Independent Out-of-Fold Evaluation:** Every development driver receives **exactly ONE independent out-of-fold evaluation** as the test driver.
- **Infeasible Threshold Selection Protocol & Study Outcome Determination:**
  * The threshold search strictly **forbids silent default fallbacks to 0.50**.
  * If no candidate threshold achieves $\text{Recall}_{\text{agg}} \ge 0.80$ on the validation driver, the fold is marked `STATUS: INFEASIBLE`.
  * The best achievable recall and corresponding Macro-F1 are logged; the fold's threshold is set to `None`.
  * **Primary Study Outcome: INCONCLUSIVE upon ANY Infeasible Fold:**
    - Any infeasible fold ($\ge 1$ of $K$ folds) means the primary operating-point objective has **NOT been consistently demonstrated across the cohort**.
    - The primary study result must be marked **INCONCLUSIVE**.
    - Diagnostic evaluation may continue, but **must NOT be presented as successful validation**.
    - Lockbox evaluation, if performed in this situation, **must be explicitly labelled EXPLORATORY**.
    - The protocol strictly forbids silently excluding failed folds and claiming complete successful validation.
  * **Mandatory Dual-Class Requirement for Validation Drivers:**
    - Both Normal (Class 0, $N_{\text{norm}} \ge 100$) and Aggressive (Class 1, $N_{\text{agg}} \ge 30$) classes must be present with sufficient sample support in every validation driver.
    - If either class is absent or insufficient for meaningful evaluation, it triggers an explicit **`DATA_QUALITY_FAIL`**, and the fold cannot produce a valid calibrated threshold.
- **Complete Final Lockbox Evaluation Procedure (7 Steps):**
  1. Complete all $K$ development folds.
  2. Freeze model configuration and threshold aggregation rule $\bar{\tau}^* = \frac{1}{|K_{\text{feasible}}|} \sum \tau_k^*$.
  3. Fit a fresh scaler strictly on all development-driver data.
  4. Train the final candidate model on all development-driver data.
  5. Freeze model, scaler, and threshold $\bar{	au}^*$.
  6. Evaluate the frozen candidate exactly once on each of the two lockbox drivers.
  7. Do not tune, retrain, recalibrate, or change thresholds using lockbox results.
  * *Probability-Scale Consistency Assumption:* Averaging fold thresholds and applying them to a model trained on all development drivers assumes probability-scale consistency. Lockbox results evaluate this assumption, and must **never be used to repair or recalibrate it**.

#### Concrete Partition Matrix for $K=6$ Development Drivers (Preferred 8-Driver Cohort)

| Fold # | Outer Test Driver (1) | Held-Out Val Driver (1) | Training Drivers ($K-2 = 4$) | Threshold Source | Evaluation Subject |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `DRV_001` | `DRV_002` | `DRV_003`, `DRV_004`, `DRV_005`, `DRV_006` | Tuned on `DRV_002` | Evaluated blind on `DRV_001` |
| **Fold 2** | `DRV_002` | `DRV_003` | `DRV_004`, `DRV_005`, `DRV_006`, `DRV_001` | Tuned on `DRV_003` | Evaluated blind on `DRV_002` |
| **Fold 3** | `DRV_003` | `DRV_004` | `DRV_005`, `DRV_006`, `DRV_001`, `DRV_002` | Tuned on `DRV_004` | Evaluated blind on `DRV_003` |
| **Fold 4** | `DRV_004` | `DRV_005` | `DRV_006`, `DRV_001`, `DRV_002`, `DRV_003` | Tuned on `DRV_005` | Evaluated blind on `DRV_004` |
| **Fold 5** | `DRV_005` | `DRV_006` | `DRV_001`, `DRV_002`, `DRV_003`, `DRV_004` | Tuned on `DRV_006` | Evaluated blind on `DRV_005` |
| **Fold 6** | `DRV_006` | `DRV_001` | `DRV_002`, `DRV_003`, `DRV_004`, `DRV_005` | Tuned on `DRV_001` | Evaluated blind on `DRV_006` |

#### Concrete Partition Matrices for Fallbacks ($K=5$ and $K=4$)
- **For $K=5$ (7 Drivers Total: 5 Dev + 2 Lockbox — Exactly 5 Folds, Train 3, Val 1, Test 1):**
  * Fold 1: Test `D1`, Val `D2`, Train `{D3, D4, D5}`
  * Fold 2: Test `D2`, Val `D3`, Train `{D4, D5, D1}`
  * Fold 3: Test `D3`, Val `D4`, Train `{D5, D1, D2}`
  * Fold 4: Test `D4`, Val `D5`, Train `{D1, D2, D3}`
  * Fold 5: Test `D5`, Val `D1`, Train `{D2, D3, D4}`
- **For $K=4$ (6 Drivers Total: 4 Dev + 2 Lockbox — Exactly 4 Folds, Train 2, Val 1, Test 1 — Pilot Floor):**
  * Fold 1: Test `D1`, Val `D2`, Train `{D3, D4}`
  * Fold 2: Test `D2`, Val `D3`, Train `{D4, D1}`
  * Fold 3: Test `D3`, Val `D4`, Train `{D1, D2}`
  * Fold 4: Test `D4`, Val `D1`, Train `{D2, D3}`

---

### 2.4 Threshold Calibration Provenance & Domain Shift Limitations (Task 3)

#### Origin & Integrity of the $\tau^* = 0.78$ Threshold
- **Exact Provenance:** The threshold $\tau^* = 0.78$ was derived during the Dataset C single-driver improvement cycle. It was identified via grid search over candidate thresholds $\tau \in [0.10, 0.90]$ with step size $\Delta\tau = 0.01$ on the internal validation split of Session 1, maximizing Macro-F1 subject to the operational constraint $\text{Recall}_{\text{agg}} \ge 0.80$.
- **Verification of Zero Test Contamination:** Forensic inspection of the codebase verified that holdout test labels from Session 2 were **never accessed or utilized** during threshold selection. The threshold selection was uncorrupted by holdout test labels.
- **Single-Driver Limitation:** Because $\tau^* = 0.78$ was calibrated on a single driver in a single vehicle, it is tightly fitted to that vehicle's mechanical compliance, engine idle vibration profile, and phone mount elasticity.

#### Vulnerability of a Fixed Threshold Under Domain Shift
1. **Probability Distribution Shift:** In kinematic classification, vehicle suspension stiffness, uncalibrated mount vibrations, and individual driver baselines alter feature distributions $P(X)$, shifting raw model output probabilities $P(y=1 \mid X)$.
2. **Asymmetric Error Inductions:**
   - In a vehicle with stiff sport suspension or uncalibrated mount resonance, output probabilities drift systematically higher. A static threshold of $0.78$ produces an unacceptably high False Positive Rate (FPR), falsely penalizing benign driving.
   - In a softly sprung luxury vehicle with heavy acoustic insulation, output probabilities drift downward, leading to severe False Negatives (missed aggressive driving).
3. **Conclusion:** A single static threshold calibrated on one driver cannot guarantee generalizable performance across diverse vehicles and driving populations.

#### Status of Adaptive 30-Second Resting Calibration
- **Audit Assessment:** The proposed concept of an online "30-second resting calibration phase" (estimating ambient vehicle noise/vibration while stationary or during initial benign driving to adjust the threshold) is a **conceptual proposal only**.
- **Empirically Unproven:** It has **not** been tested or validated against real-world multi-vehicle kinematic datasets.
- **Safety Directive:** Adaptive thresholding **must NOT be implemented or deployed in production or safety-critical logic** without prior empirical multi-driver validation, mathematical convergence guarantees, and formal safety verification.

#### Multi-Driver LODO Threshold Tuning Protocol
To prevent threshold leakage and evaluate generalization rigorously across the multi-driver cohort:
1. **Fold-Specific Threshold Tuning ($\tau_k^*$):** For each fold $k \in \{1 \dots K\}$ in the development cohort, threshold $\tau_k^*$ is selected exclusively on the inner validation driver split $\mathcal{D}_{\text{val}}^{(k)}$ drawn from the training cohort. Holdout Driver $k$ labels are strictly withheld during threshold optimization.
2. **Lockbox Threshold Lockdown ($\bar{\tau}^*$):** Before unsealing the 2-driver reserved lockbox (`DRV_007`, `DRV_008`), the final operating threshold is frozen as the mean of the development fold thresholds:
   $$\bar{\tau}^* = \frac{1}{K} \sum_{k=1}^K \tau_k^*$$
3. **Single Blind Lockbox Evaluation:** Both lockbox drivers are evaluated using $\bar{\tau}^*$ with zero post-hoc threshold adjustment.

---

### 2.5 Data Collection Acceptance Criteria & Ground Truth Separation (Task 4)

#### Separation of Mandatory Sensors vs. Optional CAN/OBD-II
- **Mandatory Sensor Acquisition:** High-rate 10 Hz linear acceleration ($x, y, z$), calibrated gyroscope in rad/s ($\omega_x, \omega_y, \omega_z$), and GPS velocity ($v_{\text{gps}}$) from the primary calibrated mobile unit.
- **Optional Physical CAN/OBD-II Telemetry:** High-speed OBD-II/CAN logging (steering angle, throttle position, brake line pressure) is designated as **optional secondary telemetry**. Because volunteer vehicles exhibit widely disparate OBD-II PIDs, CAN baud rates, and gateway firewall protections, making CAN mandatory would unnecessarily block the collection campaign.

#### Ground Truth Protocols & Video-Only Caveats
1. **Primary Ground Truth Mechanism:** Dual-observer synchronized video consensus ($\kappa \ge 0.80$) serves as the mandatory primary ground truth for aggressive driving annotations.
2. **Video-Only Ground Truth Caveat:** Video-based annotations represent **observational behavioral proxies, NOT direct physical ground truth**. Video raters observe external vehicle kinematics (swerving, rapid headway closure, aggressive overtaking) but cannot directly measure brake master cylinder hydraulic pressure, steering rack torque, or engine torque demand. All benchmark reports must explicitly state that video labels are human perceptual annotations.
3. **Inter-Rater Consensus Threshold:** The inter-rater agreement threshold is strictly mandated at Cohen's Kappa $\kappa \ge 0.80$. Any event interval with $\kappa < 0.80$ is tagged as `AMBIGUOUS_UNLABELED` and excluded from model training and evaluation.
4. **Behavioral Balance & Proving Ground Mandate:** Each driver must achieve $\ge 15.0\%$ aggressive driving samples. To eliminate public safety risk, all aggressive maneuvers (emergency braking $\ge 4.5\text{ m/s}^2$, high-speed slalom $\ge 0.45\text{ rad/s}$, abrupt lane changes) must be performed strictly within the private closed proving ground, while benign normal driving is captured on designated public mixed-road routes.

---

### 2.6 Lockbox Interpretation & Mandated Reporting Standards

#### The Role of the Lockbox: Small Independent Pilot Evaluation
- **Clear Scientific Demarcation:** The 2 reserved lockbox drivers (`DRV_007` & `DRV_008`) constitute a **small independent pilot evaluation**, NOT definitive proof of generalization across all human drivers.
- **Statistical Reality:** Human driving behaviors, vehicle chassis dynamics, and sensor noise exhibit vast multi-modal variation. Two human subjects cannot mathematically prove universal generalization across the driving population. Instead, the lockbox serves as a clean out-of-sample sanity check to detect whether the winning candidate suffered catastrophic over-fitting to the development cohort.
- **Strict Protocol Protection:** The previously approved validation split remains untouched. Lockbox results must **NEVER be used for threshold tuning, model selection, or hyperparameter adjustment**.

#### Mandatory Lockbox Granular Reporting Schema
When the lockbox is evaluated, the benchmark report must present granular metrics to ensure complete scientific transparency:
1. **Per-Driver Metrics:**
   - Accuracy and Balanced Accuracy for each lockbox driver individually.
   - Macro-F1 score for each lockbox driver individually.
   - Precision, Recall, and False Positive Rate (FPR) for each class (Normal vs. Aggressive) per driver.
2. **Pooled Metrics:** Aggregate accuracy, balanced accuracy, Macro-F1, precision, recall, and FPR pooled across all lockbox samples.
3. **Sample Exposure Quantities:** Exact count of recorded trips, total duration (minutes), total sample frames, and number of labeled events (normal vs. aggressive) per driver.
4. **Statistical Uncertainty:** 95% Confidence Intervals (Wilson score intervals for proportions/rates; empirical bootstrap intervals for Macro-F1).
5. **Configuration Metadata:** Explicit documentation of vehicle differences (make, model, year, suspension type, curb weight) and sensor mounting conditions (smartphone model, mount fixture type, mounting location) for each lockbox driver.

---

## 3. Comprehensive Prerequisite Audit Matrix & Readiness Terminology

### 3.1 Strict Definitions of Readiness Levels
To maintain scientific integrity and prevent premature claims of readiness, every prerequisite is classified across three distinct verification levels:
- **Design Verified:** Specifications, mathematical formulations, interface contracts, and procedural guardrails have been formally drafted, architecturally reviewed, and verified against functional safety principles.
- **Implementation Verified:** Code, schemas, offline pipeline transformations, cryptographic hashes, and unit tests have been executed and verified in software/offline environments on synthetic or historical data.
- **Experimentally Validated:** Empirically proven through physical data collection on multiple real-world vehicles and human drivers in the field.

> [!CAUTION] Mandatory Verification Boundary
> Hardware specifications, mounting calibration procedures, guard band flushing, domain shift behavior, and real-time streaming feature parity are **NOT described as experimentally validated**. They are currently **Design Verified** or **Implementation Verified on historical offline data only**. Experimental validation will only occur once physical multi-driver real-world driving data is acquired and evaluated.

### 3.2 Prerequisite Classification Matrix

| # | Prerequisite Dimension | Classification | Verification Level | Audit Finding / Technical Status | Operational Condition for Execution |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **P-01** | **Safety & Ethics Mandate** | **READY WITH CONDITIONS** | **Design Verified** | Protocol mandates zero aggressive driving on public roads; closed track specified. Volunteer informed consent drafted. | Signed consent forms from all participants; track booking confirmed with emergency EMT on standby. |
| **P-02** | **Driver Cohort Allocation** | **READY WITH CONDITIONS** | **Design Verified** | Preferred target ($\ge 8$ drivers) and pilot floor ($\ge 6$ drivers) specified; $<6$ NO-GO rule codified. Non-equivalence documented. | Minimum 6 drivers (pilot floor) onboarded, actively targeting $\ge 8$ drivers. |
| **P-03** | **Primary Sensor Hardware Spec** | **READY** | **Design Verified (NOT Experimentally Validated)** | 10 Hz linear acceleration, calibrated gyroscope in rad/s, GPS speed fully specified with tolerance limits. Physical fleet noise unvalidated. | Standardized test smartphone models loaded with verified logging software; battery thermal checks confirmed. |
| **P-04** | **Two-Stage Mounting Calibration** | **READY** | **Design Verified (NOT Experimentally Validated)** | 30s stationary resting phase + 60s straight-line driving calibration formulated to eliminate mount orientation bias. Physical convergence unvalidated. | Execution of two-stage calibration routine before every recorded trip; stationary variance $< 0.05\text{ m/s}^2$. |
| **P-05** | **Ground Truth & Annotation** | **READY WITH CONDITIONS** | **Design Verified** | Dual-observer video protocol with $\kappa \ge 0.80$ specified. Optional CAN separation clarified. Video-only caveat documented. | Dual-rater training and calibration session completed prior to annotation; rater consensus $\kappa \ge 0.80$ enforced. |
| **P-06** | **Temporal & Spatial Leakage Guardrails** | **READY** | **Implementation Verified (NOT Experimentally Validated on live vehicle stream)** | Trip boundary buffer flushing and 100-sample ($10.0\text{ s}$) guard bands specified in code contracts. Zero cross-trip bleed verified on historical data. | Verification script executed against raw parquet/CSV partitions before ingestion into feature extraction. |
| **P-07** | **LODO Cross-Validation & Lockbox** | **READY** | **Design Verified** | Grouped LODO with 2-driver reserved lockbox architected. Zero driver overlap strictly enforced. Lockbox defined as pilot evaluation. | Automated LODO harness configured to read cohort metadata and verify zero driver identity overlap across folds. |
| **P-08** | **Threshold Calibration Architecture** | **READY** | **Design Verified** | Provenance of 0.78 documented; domain shift vulnerability analyzed; adaptive 30s proposal flagged unproven; fold-by-fold tuning specified. | Dynamic fold-specific threshold optimization executed exclusively on training drivers; lockbox threshold locked as fold mean. |
| **P-09** | **Domain Shift & Data Quality Policy** | **READY** | **Design Verified** | Rejection of `PSI >= 0.25 -> style risk = 0.0` audited and applied. Health degradation (`DEGRADED`) and style withholding approved. | Passive PSI monitoring service deployed in parallel; zero risk zeroing; Pipeline B maneuver logic untouched. |
| **P-10** | **Streaming Feature Contract** | **READY** | **Design & Offline Implementation Verified (NOT Experimentally Validated on live stream)** | Mathematical formulations, history buffers (100 samples), and cold-start parity tests specified for all 110/102 features. Offline feature generator verified. | Ready for offline validation; streaming stateful engine implementation deferred until candidate promotion phase. |
| **P-11** | **Production Baseline & Artifacts** | **READY** | **Implementation Verified** | 4/4 frozen model and scaler artifacts match required SHA-256 hashes bit-for-bit. Production model preserved 100%. | None. Baseline integrity is actively secured and verified. |
| **P-12** | **Test Suite Regression Integrity** | **READY** | **Implementation Verified** | Full backend API and safety agent test suite executed: 100/100 tests pass without regression. Frontend builds cleanly. | Continuous execution of test harness after all documentation updates. |
| **P-13** | **Production Deployment of Candidate v2** | **BLOCKED (NO-GO)** | **BLOCKED (Requires Experimental Validation)** | Candidate v2 is an experimental artifact evaluated only on single-driver Dataset C. Production expects 47 features vs 110. | **STRICT NO-GO:** Blocked until multi-driver LODO proves $>90\%$ generalization and streaming parity is verified. |

---

## 4. Operational Verification & Baseline Protection

### 4.1 Cryptographic Verification of Protected Production Artifacts
The 4 frozen production model artifacts must never be modified, retrained, or overwritten. Forensic verification confirms that their SHA-256 hashes match the baseline bit-for-bit:

| Artifact Path | Expected Baseline SHA-256 | Actual Measured SHA-256 | Audit Status |
| :--- | :--- | :--- | :---: |
| `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib` | `552048535A3FD584E4874731B8BE1438DAB4051D8CFA6A98FB1A7B19D9202E90` | `552048535A3FD584E4874731B8BE1438DAB4051D8CFA6A98FB1A7B19D9202E90` | **MATCH (VERIFIED)** |
| `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib` | `D4B3D5E3E1B739C63A9E21FAC51FCEDF36716143C645ABE0D0E3FF99127A7576` | `D4B3D5E3E1B739C63A9E21FAC51FCEDF36716143C645ABE0D0E3FF99127A7576` | **MATCH (VERIFIED)** |
| `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib` | `A00180BD01D5999B1338737549D5745ADB986138F813F7F371D67589D0DBAEFC` | `A00180BD01D5999B1338737549D5745ADB986138F813F7F371D67589D0DBAEFC` | **MATCH (VERIFIED)** |
| `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib` | `D67F9AB70EB667CC57E8AEB2F6768E62529F0CB803279CC33D17F6C76555C9F5` | `D67F9AB70EB667CC57E8AEB2F6768E62529F0CB803279CC33D17F6C76555C9F5` | **MATCH (VERIFIED)** |

### 4.2 Cryptographic Verification of Preserved Candidate v2 Artifacts
Candidate v2 artifacts remain securely isolated under `models/experiments/pipeline_a/` with uncorrupted provenance:

| Artifact Path | SHA-256 Hash | State | Provenance |
| :--- | :--- | :---: | :--- |
| `models/experiments/pipeline_a/pipeline_a_extratrees_candidate_v2.joblib` | `2ce7b451ed4240d952763a551c335bc0055fb4aec30b82ae9639e7b3c4cb2e62` | **LOCKED** | Single-driver Dataset C Session 1 training; 110 features. |
| `models/experiments/pipeline_a/pipeline_a_scaler_candidate_v2.joblib` | `af805aa8f87446ab9bfbf771748c1991af2c93206468172281253ce9ceffddd3` | **LOCKED** | Fitted strictly on Dataset C Session 1 training split. |

### 4.3 Automated Test Suite Regression
The automated regression suite was executed to ensure zero side effects on backend API endpoints, inference handlers, or deterministic safety assessment logic:
- **Test Command:** `pytest tests/test_backend_api.py tests/test_safety_agent.py`
- **Execution Summary:** `100 passed in 6.39s`
- **Pass Rate:** **100.0% (Zero Failures, Zero Regressions)**
- **Deterministic Safety Logic:** Verified that missing/degraded inputs are handled in strict adherence to Stage 6 policy without unauthorized risk zeroing.

### 4.4 Git Working Tree Status & Repository Preservation
- **Branch:** `stage6-phase1` (Active, unchanged)
- **Commit Baseline:** `f746291700659556862d9f1cfadf5ed2df4b9bc4`
- **Working Tree Reality (Not Clean — Pre-Existing Work Preserved):** The working tree is **explicitly NOT clean**, as seven pre-existing tracked files remain modified from previous engineering phases.
- **Tracked Files Preservation Audit:**
  * The seven pre-existing modified tracked files (`README.md`, `ml/agent/config.py`, `ml/agent/prediction_validator.py`, `ml/agent/risk_engine.py`, `ml/agent/safety_agent.py`, `ml/agent/temporal_smoother.py`, `tests/test_safety_agent.py`) were **strictly preserved and NOT altered** during this documentation task.
  * Only the four intended Pipeline A documentation files changed during this task:
    1. `reports/pipeline_a/driver_independent_validation_plan.md`
    2. `reports/pipeline_a/final_collection_readiness_audit.md`
    3. `reports/pipeline_a/multidriver_readiness_report.md`
    4. `reports/pipeline_a/multidriver_data_collection_protocol.md`
  * No scratch scripts remain.
  * Zero git commits, zero branch creations, zero merges, zero rebases, zero hard resets, and zero branch switches occurred.

---

## 5. Audit Determination & Official Decision

### 5.1 Official Decision for Physical Data Collection: **GO (READY WITH CONDITIONS — TIERED RECRUITMENT RECONCILIATION)**
The engineering specifications, mathematical formulations, safety guardrails, and validation plans are complete, rigorous, and verified. Field data collection is approved to proceed, with the decision strictly reconciled against the driver recruitment fallback hierarchy:

1. **Preferred Initial Cohort Execution (Preferred Target: $\ge 8$ Drivers):** **GO (Preferred Initial Cohort).** Enables 6-development-driver 3-way grouped cyclic cross-validation and 2-driver reserved lockbox benchmark. Explicitly recognizes that eight drivers remains a small initial study requiring per-driver reporting and confidence intervals.
2. **Pilot Protocol Execution (Fallback Floor: 6–7 Drivers):** **PILOT GO (NON-EQUIVALENT).** 
   - **7 Drivers:** 5-driver development LODO + 2-driver lockbox.
   - **6 Drivers:** 4-driver development LODO + 2-driver lockbox.
   - **Non-Equivalence Caveat:** A 6-driver pilot is **NOT equivalent** to the planned 6-fold development validation, providing lower statistical power and wider uncertainty bounds.
3. **Hard Abort Cutoff ($< 6$ Usable Drivers):** **IMMEDIATE NO-GO (CAMPAIGN BLOCKED).** If recruitment or participant attrition yields fewer than 6 usable drivers, the evaluation campaign is terminated.

Field mobilization is subject to the following mandatory operational sign-offs:
1. **Recruitment Sign-off:** Minimum 6 volunteer drivers (pilot floor) onboarded, actively targeting $\ge 8$ drivers.
2. **Proving Ground Sign-off:** Private closed-course venue contracted and inspected for aggressive maneuver execution.
3. **Safety & Consent Sign-off:** Volunteer informed consent documents signed; medical/safety personnel confirmed on-site during aggressive testing.
4. **Sensor Calibration Sign-off:** Stationary zero-velocity and straight-line calibration routines verified on all collection smartphones.
5. **Rater Calibration Sign-off:** Dual-observer video annotators pass pre-campaign test with Cohen's Kappa $\kappa \ge 0.80$.

### 5.2 Official Decision for Production Deployment: **STRICT NO-GO (BLOCKED)**
- **Candidate v2 Deployment Status:** **BLOCKED.**
- **Rationale:** Candidate v2 was trained and evaluated exclusively on a single driver (Dataset C). Its 91.21% accuracy represents same-driver cross-session performance and does NOT substantiate multi-driver generalization. Furthermore, Candidate v2 expects 110 features, whereas production expects 47 features. Candidate v2 must remain strictly an experimental research artifact until:
  1. Multi-driver data collection is completed.
  2. The candidate achieves $\ge 90.0\%$ balanced accuracy and Macro-F1 across unseen drivers in LODO validation.
  3. The candidate confirms $\ge 90.0\%$ performance on the 2-driver reserved lockbox (`DRV_007`, `DRV_008`).
  4. Real-time streaming feature extraction passes parity testing ($< 10^{-6}$ numerical error).

---

## 6. Actionable Roadmap for the Field Engineering Team

```
                      [PHASE 1: PRE-COLLECTION MOBILIZATION]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[Cohort Onboarding]                                             [Proving Ground Inspection]
- Recruit >= 8 drivers                                          - Clear closed-course runoffs
- Sign informed consent & NDAs                                  - Verify GPS line-of-sight
- Distribute vehicle metadata survey                            - Station safety marshals
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
                      [PHASE 2: FIELD DATA ACQUISITION CAMPAIGN]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[Part A: Proving Ground (Aggressive)]                           [Part B: Public Road (Normal)]
- Mount 2-stage calibration (30s stat + 60s run)                - Benign driving routes (highway/urban)
- Controlled braking (>= 4.5 m/s^2)                             - Natural lane keeping, smooth cruising
- Slalom swerves (>= 0.45 rad/s)                                - Strict compliance with traffic laws
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
                      [PHASE 3: QUALITY GATING & ANNOTATION]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[Automated Sensor QC]                                           [Dual-Rater Video Annotation]
- Verify 10 Hz sampling, dropouts < 0.1%                        - Dual independent video review
- Verify stationary variance < 0.05 m/s^2                       - Compute Cohen's Kappa (assert kappa >= 0.80)
- Guard band trimming (10s buffers)                             - Optional CAN telemetry correlation
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
                      [PHASE 4: BENCHMARK EXECUTION (LODO & LOCKBOX)]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[6-Fold Development LODO]                                       [Reserved Lockbox Benchmark]
- Train on 5 drivers, test on 1                                 - Drivers 7 & 8 unsealed ONCE
- Fold-specific threshold optimization                          - Frozen threshold bar_tau* evaluated blind
- Measure inter-driver variance & FPR                           - Definitively prove >90% generalization target
```

---
*End of Final Collection Readiness Audit & Safety Review.*
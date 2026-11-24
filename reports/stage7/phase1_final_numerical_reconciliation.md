# Stage 7 Phase 1 — Final Numerical Reconciliation and Report Correction

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Repository:** `D:/driver ml model/driver-monitoring-agent`  
**Branch:** `stage6-phase1` (HEAD commit: `f746291`)  
**Audit Date:** November 21, 2026  
**Audit Nature:** READ-ONLY Forensic Investigation and Numerical Correction  
**Authoritative Documents:**
- Primary Report Audited: [`reports/stage7/phase1_forensic_validation.md`](file:///D:/driver%20ml%20model/driver-monitoring-agent/reports/stage7/phase1_forensic_validation.md)
- Supporting Results File: [`reports/stage7/streaming_evaluation_results.json`](file:///D:/driver%20ml%20model/driver-monitoring-agent/reports/stage7/streaming_evaluation_results.json)
- Processed Dataset: [`dataset/processed/dataset_b_processed_all.csv`](file:///D:/driver%20ml%20model/driver-monitoring-agent/dataset/processed/dataset_b_processed_all.csv)

---

## 1. Executive Summary

This forensic investigation was commissioned to independently re-evaluate every calculation, metric definition, and assertion in `phase1_forensic_validation.md` against raw prediction records, ground-truth annotations, and evaluator code. Operating under strict read-only controls, we re-ran all driver evaluations, reconstructed every ground-truth and predicted event, audited the four-class confusion matrix, reconciled replay durations and synthetic cruising gaps, and evaluated the false-event rate against project requirements.

### Key Forensic Findings:

1. **Resolution of the Class-Level False Negative Discrepancy ($\sum \text{FN}_c = 14$, not $15$):**  
   The previous report claimed that class-level false negatives totaled 15 (comprising 14 missed detections and 1 misclassification). This was an arithmetic error. In the 109-event dataset, exactly 96 events are matched and 13 are missed detections ($109 - 96 = 13$). Adding the single misclassified matched event (Driver-3 GT 22) yields exactly **14 class-level false negatives** ($\sum \text{FN}_c = 13 + 1 = 14$).

2. **Root Cause of the Sudden Right Turn (SRT) Discrepancy ($25$ GT, not $26$):**  
   In the previous report's confusion matrix, the SRT row reported 25 ground-truth events but listed $22\text{ correct} + 1\text{ misclassified} + 3\text{ missed} = 26$ events, yielding an erroneous recall of $22/26 = 84.62\%$. In reality, SRT has exactly **2 missed detections** (GT 23 and GT 24 in Driver-3), 1 misclassified matched event (GT 22), and 22 correct detections, totaling $22 + 1 + 2 = 25$ GT events. The true SRT recall is **$22/25 = 88.00\%$**, and true SRT F1 is **$91.67\%$**.

3. **Lineage of the 90.47% vs 90.60% vs 90.94% Macro-F1 Figures:**  
   - **$90.47\%$ is mathematically erroneous:** It resulted directly from the SRT recall error ($84.62\%$ instead of $88.00\%$, yielding SRT F1 of $89.80\%$ instead of $91.67\%$). Averaging $(92.86\% + 97.87\% + 89.7959\% + 81.3559\%) / 4 = \mathbf{90.47\%}$.  
   - **$90.60\%$ represents the Pooled Stream:** In the pooled evaluation, the 102nd prediction (PRED 79, an inter-driver boundary artifact) was classified as Sudden Left Turn (SLT), raising SLT False Positives from 3 to 4, lowering SLT precision to $85.71\%$ and SLT F1 to $80.00\%$. Averaging $(92.86\% + 97.87\% + 91.67\% + 80.00\%) / 4 = \mathbf{90.60\%}$.  
   - **$90.94\%$ is the True Clean Driver-Wise Combined Baseline:** In the clean combined evaluation of isolated driver runs (101 predictions, 0 boundary artifacts), SLT has 3 False Positives, SLT precision is $88.89\%$, and SLT F1 is $81.36\%$. Averaging $(92.86\% + 97.87\% + 91.67\% + 81.36\%) / 4 = \mathbf{90.94\%}$.

4. **Reconciliation of the 40-Observation / 4.0-Second Discrepancy:**  
   The isolated driver evaluations total $1,048 + 1,043 + 1,143 = 3,234$ observations ($323.4\text{ s}$), containing 106 synthetic cruising gaps ($34 + 34 + 38$). The pooled evaluation totals $3,274$ observations ($327.4\text{ s}$), containing 108 synthetic gaps. The difference is exactly **$2$ inter-driver boundary gaps** ($2 \times 20 = 40$ observations, or $4.0\text{ s}$) inserted between Driver-1 $\to$ Driver-2 and Driver-2 $\to$ Driver-3 when concatenating events into a single uninterrupted sequence.

5. **False-Event Rate Target Audit ($\le 0.50\text{ FE/min}$):**  
   The Stage 7 specification mandates $\le 0.50\text{ false events/minute}$. The base configuration yields **$0.9276\text{ FE/min}$** (clean driver-wise sum) and **$1.0996\text{ FE/min}$** (pooled), both failing the specification. Furthermore, the $N=1$ sensitivity configuration yields **$0.5498\text{ FE/min}$**, which also **fails** the requirement ($0.55 > 0.50$). Speculative claims that a 1.0-second cooldown or jerk threshold will pass cannot be accepted without empirical controlled experiments.

6. **Acceptance Recommendation:**  
   Phase 1 is granted **CONDITIONAL ACCEPT**. The streaming testbed code and boundary-independent evaluation mechanics are structurally sound and independently reproducible, but production deployment is blocked until the false-event rate is brought $\le 0.50\text{ FE/min}$ in a controlled experiment on `train + val` and validated on locked `test`.

---

## 2. Reconstructed Event-Level Dataset

### 2.1 Matching Methodology and Mathematical Invariants

The evaluation was independently executed using `StreamingEventEvaluator` with standard project thresholds:
- Temporal Intersection-over-Union ($\text{IoU}$) threshold: $\ge 0.40$.
- Onset tolerance ($\Delta t_{\text{onset}} = |t_{\text{pred\_start}} - t_{\text{gt\_start}}|$): $\le 0.50\text{ s}$.
- Matching algorithm: Greedy optimal bipartite matching sorted descending by $\text{IoU}$.

Under these rules, each ground-truth event and each predicted event is assigned to exactly one of the following mutually exclusive categories:
- **Primary Match (True Positive Detection):** Assigned to a GT event with $\text{IoU} \ge 0.40$ and $\Delta t_{\text{onset}} \le 0.50\text{ s}$.
- **Duplicate Prediction:** A secondary prediction overlapping an already-matched GT event (0 occurrences observed).
- **Unmatched Prediction (False Positive Detection):** A prediction that fails to match any GT event (spurious alarm).
- **Unmatched Ground Truth (False Negative Detection):** A GT event that was not detected by any prediction (missed event).

---

### 2.2 Driver-1 Authoritative Event-Level Reconstruction (35 GT, 31 Pred, 29 Matched, 6 Missed, 2 FP)

Replay Duration: $104.800\text{ s}$ (1,048 observations at 10 Hz; 368 maneuver samples + 680 synthetic gap samples across 34 gaps).

| GT ID | GT Class | GT Start | GT End | Pred ID | Pred Class | Pred Start | Pred End | IoU | Onset Diff | Status | Correctness | Conf |
|:---:|:---|---:|---:|:---:|:---|---:|---:|---:|---:|:---|:---|---:|
| 1 | SRT (3) | 0.1s | 1.4s | 1 | SRT (3) | 0.2s | 1.4s | 0.9231 | 0.100s | MATCHED | CORRECT | 66.96% |
| 2 | SRT (3) | 3.5s | 4.8s | 2 | SRT (3) | 3.5s | 4.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 63.21% |
| 3 | SRT (3) | 6.9s | 8.2s | 3 | SRT (3) | 6.9s | 8.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 68.20% |
| 4 | SRT (3) | 10.3s | 11.6s | 4 | SRT (3) | 10.5s | 11.5s | 0.7692 | 0.200s | MATCHED | CORRECT | 63.34% |
| 5 | SRT (3) | 13.7s | 15.0s | 5 | SRT (3) | 13.8s | 15.0s | 0.9231 | 0.100s | MATCHED | CORRECT | 59.91% |
| 6 | SRT (3) | 17.1s | 18.4s | 6 | SRT (3) | 17.1s | 18.4s | 1.0000 | 0.000s | MATCHED | CORRECT | 61.12% |
| 7 | SRT (3) | 20.5s | 21.8s | 7 | SRT (3) | 20.5s | 21.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 60.88% |
| 8 | SRT (3) | 23.9s | 25.2s | 8 | SRT (3) | 23.9s | 25.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 58.00% |
| 9 | SRT (3) | 27.3s | 28.6s | 9 | SRT (3) | 27.4s | 28.6s | 0.9231 | 0.100s | MATCHED | CORRECT | 73.51% |
| 26 | SA (1) | 30.7s | 31.5s | 10 | SA (1) | 30.7s | 31.4s | 0.8750 | 0.000s | MATCHED | CORRECT | 51.73% |
| 27 | SA (1) | 33.6s | 34.4s | 11 | SA (1) | 33.6s | 34.3s | 0.8750 | 0.000s | MATCHED | CORRECT | 62.54% |
| 28 | SA (1) | 36.5s | 37.3s | 12 | SA (1) | 36.5s | 37.3s | 1.0000 | 0.000s | MATCHED | CORRECT | 64.07% |
| 29 | SA (1) | 39.4s | 40.2s | 13 | SA (1) | 39.7s | 40.2s | 0.6250 | 0.300s | MATCHED | CORRECT | 69.84% |
| 30 | SA (1) | 42.3s | 43.1s | 14 | SA (1) | 42.3s | 43.1s | 1.0000 | 0.000s | MATCHED | CORRECT | 62.88% |
| 31 | SA (1) | 45.2s | 46.0s | 15 | SA (1) | 45.3s | 46.0s | 0.8750 | 0.100s | MATCHED | CORRECT | 69.55% |
| 32 | SA (1) | 48.1s | 48.9s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 33 | SA (1) | 51.0s | 51.8s | 17 | SA (1) | 51.1s | 51.7s | 0.7500 | 0.100s | MATCHED | CORRECT | 45.64% |
| 34 | SA (1) | 53.9s | 54.7s | 18 | SA (1) | 53.9s | 54.7s | 1.0000 | 0.000s | MATCHED | CORRECT | 64.45% |
| 35 | SA (1) | 56.8s | 57.6s | 19 | SA (1) | 57.0s | 57.5s | 0.6250 | 0.200s | MATCHED | CORRECT | 69.76% |
| 54 | SD (2) | 59.7s | 60.8s | 20 | SD (2) | 59.8s | 60.8s | 0.9091 | 0.100s | MATCHED | CORRECT | 77.94% |
| 55 | SD (2) | 62.9s | 64.0s | 21 | SD (2) | 63.1s | 64.0s | 0.8182 | 0.200s | MATCHED | CORRECT | 80.44% |
| 56 | SD (2) | 66.1s | 67.2s | 22 | SD (2) | 66.3s | 67.2s | 0.8182 | 0.200s | MATCHED | CORRECT | 74.91% |
| 57 | SD (2) | 69.3s | 70.4s | 23 | SD (2) | 69.3s | 70.3s | 0.9091 | 0.000s | MATCHED | CORRECT | 66.49% |
| 58 | SD (2) | 72.5s | 73.6s | 24 | SD (2) | 72.7s | 73.5s | 0.7273 | 0.200s | MATCHED | CORRECT | 81.99% |
| 59 | SD (2) | 75.7s | 76.8s | 25 | SD (2) | 75.7s | 76.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 71.97% |
| 60 | SD (2) | 78.9s | 80.0s | 26 | SD (2) | 78.9s | 80.0s | 1.0000 | 0.000s | MATCHED | CORRECT | 73.58% |
| 61 | SD (2) | 82.1s | 83.2s | 27 | SD (2) | 82.1s | 83.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 77.27% |
| 78 | SLT (4) | 85.3s | 85.9s | 28 | SLT (4) | 85.3s | 85.9s | 1.0000 | 0.000s | MATCHED | CORRECT | 96.17% |
| 79 | SLT (4) | 88.0s | 88.6s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 80 | SLT (4) | 90.7s | 91.3s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 81 | SLT (4) | 93.4s | 94.0s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 82 | SLT (4) | 96.1s | 96.7s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 83 | SLT (4) | 98.8s | 99.4s | 30 | SLT (4) | 98.8s | 99.4s | 1.0000 | 0.000s | MATCHED | CORRECT | 90.80% |
| 84 | SLT (4) | 101.5s | 102.1s | 31 | SLT (4) | 101.8s | 102.1s | 0.5000 | 0.300s | MATCHED | CORRECT | 89.16% |
| 85 | SLT (4) | 104.2s | 104.8s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |

**Driver-1 Unmatched Predictions (False Positives / Spurious Alarms):**
- **Pred 16:** Class 1 (SA), $48.7\text{ s} - 48.8\text{ s}$, Duration: $0.10\text{ s}$, Confidence: $50.14\%$. (Kinematic decay tail oscillation of missed GT 32).
- **Pred 29:** Class 4 (SLT), $90.7\text{ s} - 90.8\text{ s}$, Duration: $0.10\text{ s}$, Confidence: $88.20\%$. (Secondary sub-threshold spike during missed GT 80).

*Driver-1 Verification:* $29\text{ Matched} + 6\text{ Missed GT} = 35\text{ GT}$. $29\text{ Matched} + 2\text{ FP Pred} = 31\text{ Pred}$. Matched accuracy: $29/29 = \mathbf{100.00\%}$.

---

### 2.3 Driver-2 Authoritative Event-Level Reconstruction (35 GT, 34 Pred, 32 Matched, 3 Missed, 2 FP)

Replay Duration: $104.300\text{ s}$ (1,043 observations at 10 Hz; 363 maneuver samples + 680 synthetic gap samples across 34 gaps).

| GT ID | GT Class | GT Start | GT End | Pred ID | Pred Class | Pred Start | Pred End | IoU | Onset Diff | Status | Correctness | Conf |
|:---:|:---|---:|---:|:---:|:---|---:|---:|---:|---:|:---|:---|---:|
| 10 | SRT (3) | 0.1s | 1.4s | 1 | SRT (3) | 0.2s | 1.4s | 0.9231 | 0.100s | MATCHED | CORRECT | 74.35% |
| 11 | SRT (3) | 3.5s | 4.8s | 2 | SRT (3) | 3.5s | 4.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 64.46% |
| 12 | SRT (3) | 6.9s | 8.2s | 3 | SRT (3) | 6.9s | 8.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 65.30% |
| 13 | SRT (3) | 10.3s | 11.6s | 4 | SRT (3) | 10.3s | 11.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 68.36% |
| 14 | SRT (3) | 13.7s | 15.0s | 5 | SRT (3) | 13.7s | 15.0s | 1.0000 | 0.000s | MATCHED | CORRECT | 71.24% |
| 15 | SRT (3) | 17.1s | 18.4s | 6 | SRT (3) | 17.1s | 18.2s | 0.8462 | 0.000s | MATCHED | CORRECT | 69.83% |
| 16 | SRT (3) | 20.5s | 21.8s | 7 | SRT (3) | 20.7s | 21.8s | 0.8462 | 0.200s | MATCHED | CORRECT | 78.66% |
| 17 | SRT (3) | 23.9s | 25.2s | 8 | SRT (3) | 24.2s | 25.2s | 0.7692 | 0.300s | MATCHED | CORRECT | 67.67% |
| 18 | SRT (3) | 27.3s | 28.6s | 9 | SRT (3) | 27.3s | 28.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 66.85% |
| 36 | SA (1) | 30.7s | 31.5s | 10 | SA (1) | 30.7s | 31.5s | 1.0000 | 0.000s | MATCHED | CORRECT | 48.32% |
| 37 | SA (1) | 33.6s | 34.4s | 11 | SA (1) | 33.6s | 34.3s | 0.8750 | 0.000s | MATCHED | CORRECT | 60.38% |
| 38 | SA (1) | 36.5s | 37.3s | 12 | SA (1) | 36.5s | 37.3s | 1.0000 | 0.000s | MATCHED | CORRECT | 61.22% |
| 39 | SA (1) | 39.4s | 40.2s | 13 | SA (1) | 39.4s | 40.1s | 0.8750 | 0.000s | MATCHED | CORRECT | 64.95% |
| 40 | SA (1) | 42.3s | 43.1s | 14 | SA (1) | 42.3s | 43.1s | 1.0000 | 0.000s | MATCHED | CORRECT | 50.08% |
| 41 | SA (1) | 45.2s | 46.0s | 15 | SA (1) | 45.4s | 46.0s | 0.7500 | 0.200s | MATCHED | CORRECT | 51.63% |
| 42 | SA (1) | 48.1s | 48.9s | 16 | SA (1) | 48.1s | 48.9s | 1.0000 | 0.000s | MATCHED | CORRECT | 61.77% |
| 43 | SA (1) | 51.0s | 51.8s | 17 | SA (1) | 51.1s | 51.8s | 0.8750 | 0.100s | MATCHED | CORRECT | 56.73% |
| 44 | SA (1) | 53.9s | 54.7s | 18 | SA (1) | 53.9s | 54.6s | 0.8750 | 0.000s | MATCHED | CORRECT | 42.41% |
| 45 | SA (1) | 56.8s | 57.6s | 19 | SA (1) | 56.8s | 57.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 47.52% |
| 62 | SD (2) | 59.7s | 60.8s | 20 | SD (2) | 59.7s | 60.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 75.93% |
| 63 | SD (2) | 62.9s | 64.0s | 21 | SD (2) | 62.9s | 64.0s | 1.0000 | 0.000s | MATCHED | CORRECT | 68.52% |
| 64 | SD (2) | 66.1s | 67.2s | 22 | SD (2) | 66.1s | 67.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 73.12% |
| 65 | SD (2) | 69.3s | 70.4s | 23 | SD (2) | 69.3s | 70.4s | 1.0000 | 0.000s | MATCHED | CORRECT | 70.96% |
| 66 | SD (2) | 72.5s | 73.6s | 24 | SD (2) | 72.5s | 73.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 79.71% |
| 67 | SD (2) | 75.7s | 76.8s | 25 | SD (2) | 75.9s | 76.8s | 0.8182 | 0.200s | MATCHED | CORRECT | 79.22% |
| 68 | SD (2) | 78.9s | 80.0s | 26 | SD (2) | 78.9s | 80.0s | 1.0000 | 0.000s | MATCHED | CORRECT | 59.68% |
| 86 | SLT (4) | 82.1s | 82.7s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 87 | SLT (4) | 84.8s | 85.4s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 88 | SLT (4) | 87.5s | 88.1s | 29 | SLT (4) | 87.8s | 88.1s | 0.5000 | 0.300s | MATCHED | CORRECT | 92.33% |
| 89 | SLT (4) | 90.2s | 90.8s | 30 | SLT (4) | 90.2s | 90.5s | 0.5000 | 0.000s | MATCHED | CORRECT | 74.95% |
| 90 | SLT (4) | 92.9s | 93.5s | 31 | SLT (4) | 92.9s | 93.5s | 1.0000 | 0.000s | MATCHED | CORRECT | 86.41% |
| 91 | SLT (4) | 95.6s | 96.2s | 32 | SLT (4) | 95.6s | 96.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 82.68% |
| 92 | SLT (4) | 98.3s | 98.9s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 93 | SLT (4) | 101.0s | 101.6s | 33 | SLT (4) | 101.0s | 101.5s | 0.8333 | 0.000s | MATCHED | CORRECT | 82.18% |
| 94 | SLT (4) | 103.7s | 104.3s | 34 | SLT (4) | 103.7s | 104.3s | 1.0000 | 0.000s | MATCHED | CORRECT | 95.33% |

**Driver-2 Unmatched Predictions (False Positives / Spurious Alarms):**
- **Pred 27:** Class 4 (SLT), $82.2\text{ s} - 82.4\text{ s}$, Duration: $0.20\text{ s}$, Confidence: $91.32\%$. (Sub-threshold fragment during missed GT 86).
- **Pred 28:** Class 4 (SLT), $85.1\text{ s} - 85.3\text{ s}$, Duration: $0.20\text{ s}$, Confidence: $66.92\%$. (Sub-threshold fragment during missed GT 87).

*Driver-2 Verification:* $32\text{ Matched} + 3\text{ Missed GT} = 35\text{ GT}$. $32\text{ Matched} + 2\text{ FP Pred} = 34\text{ Pred}$. Matched accuracy: $32/32 = \mathbf{100.00\%}$.

---

### 2.4 Driver-3 Authoritative Event-Level Reconstruction (39 GT, 36 Pred, 35 Matched, 4 Missed, 1 FP)

Replay Duration: $114.300\text{ s}$ (1,143 observations at 10 Hz; 383 maneuver samples + 760 synthetic gap samples across 38 gaps).

| GT ID | GT Class | GT Start | GT End | Pred ID | Pred Class | Pred Start | Pred End | IoU | Onset Diff | Status | Correctness | Conf |
|:---:|:---|---:|---:|:---:|:---|---:|---:|---:|---:|:---|:---|---:|
| 19 | SRT (3) | 0.1s | 1.4s | 1 | SRT (3) | 0.1s | 1.4s | 1.0000 | 0.000s | MATCHED | CORRECT | 68.89% |
| 20 | SRT (3) | 3.5s | 4.8s | 2 | SRT (3) | 3.5s | 4.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 57.43% |
| 21 | SRT (3) | 6.9s | 8.2s | 3 | SRT (3) | 6.9s | 8.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 40.27% |
| 22 | SRT (3) | 10.3s | 11.6s | 4 | SA (1) | 10.3s | 11.6s | 1.0000 | 0.000s | MATCHED | MISCLASSIFIED | 40.04% |
| 23 | SRT (3) | 13.7s | 15.0s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 24 | SRT (3) | 17.1s | 18.4s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 25 | SRT (3) | 20.5s | 21.8s | 6 | SRT (3) | 20.5s | 21.8s | 1.0000 | 0.000s | MATCHED | CORRECT | 43.59% |
| 46 | SA (1) | 23.9s | 24.7s | 7 | SA (1) | 24.0s | 24.7s | 0.8750 | 0.100s | MATCHED | CORRECT | 44.40% |
| 47 | SA (1) | 26.8s | 27.6s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 48 | SA (1) | 29.7s | 30.5s | 8 | SA (1) | 29.7s | 30.5s | 1.0000 | 0.000s | MATCHED | CORRECT | 71.37% |
| 49 | SA (1) | 32.6s | 33.4s | 9 | SA (1) | 32.6s | 33.3s | 0.8750 | 0.000s | MATCHED | CORRECT | 75.62% |
| 50 | SA (1) | 35.5s | 36.3s | 10 | SA (1) | 35.5s | 36.3s | 1.0000 | 0.000s | MATCHED | CORRECT | 61.36% |
| 51 | SA (1) | 38.4s | 39.2s | 11 | SA (1) | 38.4s | 39.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 63.61% |
| 52 | SA (1) | 41.3s | 42.1s | 12 | SA (1) | 41.3s | 42.1s | 1.0000 | 0.000s | MATCHED | CORRECT | 70.07% |
| 53 | SA (1) | 44.2s | 45.0s | 13 | SA (1) | 44.2s | 45.0s | 1.0000 | 0.000s | MATCHED | CORRECT | 53.97% |
| 69 | SD (2) | 47.1s | 48.2s | 14 | SD (2) | 47.2s | 48.1s | 0.8182 | 0.100s | MATCHED | CORRECT | 67.74% |
| 70 | SD (2) | 50.3s | 51.4s | 15 | SD (2) | 50.3s | 51.4s | 1.0000 | 0.000s | MATCHED | CORRECT | 63.15% |
| 71 | SD (2) | 53.5s | 54.6s | 16 | SD (2) | 53.5s | 54.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 74.30% |
| 72 | SD (2) | 56.7s | 57.8s | 17 | SD (2) | 56.8s | 57.8s | 0.9091 | 0.100s | MATCHED | CORRECT | 64.52% |
| 73 | SD (2) | 59.9s | 61.0s | 18 | SD (2) | 59.9s | 60.9s | 0.9091 | 0.000s | MATCHED | CORRECT | 56.64% |
| 74 | SD (2) | 63.1s | 64.2s | 19 | SD (2) | 63.1s | 64.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 47.98% |
| 75 | SD (2) | 66.3s | 67.4s | 20 | SD (2) | 66.3s | 67.4s | 1.0000 | 0.000s | MATCHED | CORRECT | 57.53% |
| 76 | SD (2) | 69.5s | 70.6s | 21 | SD (2) | 69.5s | 70.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 68.86% |
| 77 | SD (2) | 72.7s | 73.8s | — | — | — | — | 0.0000 | — | UNMATCHED_GT | MISSED | — |
| 95 | SLT (4) | 75.9s | 76.5s | 22 | SLT (4) | 75.9s | 76.5s | 1.0000 | 0.000s | MATCHED | CORRECT | 90.35% |
| 96 | SLT (4) | 78.6s | 79.2s | 23 | SLT (4) | 78.6s | 79.1s | 0.8333 | 0.000s | MATCHED | CORRECT | 83.63% |
| 97 | SLT (4) | 81.3s | 81.9s | 24 | SLT (4) | 81.5s | 81.8s | 0.5000 | 0.200s | MATCHED | CORRECT | 95.26% |
| 98 | SLT (4) | 84.0s | 84.6s | 25 | SLT (4) | 84.0s | 84.6s | 1.0000 | 0.000s | MATCHED | CORRECT | 79.28% |
| 99 | SLT (4) | 86.7s | 87.3s | 26 | SLT (4) | 86.7s | 87.3s | 1.0000 | 0.000s | MATCHED | CORRECT | 85.39% |
| 100 | SLT (4) | 89.4s | 90.0s | 27 | SLT (4) | 89.4s | 90.0s | 1.0000 | 0.000s | MATCHED | CORRECT | 79.57% |
| 101 | SLT (4) | 92.1s | 92.7s | 28 | SLT (4) | 92.1s | 92.7s | 1.0000 | 0.000s | MATCHED | CORRECT | 92.47% |
| 102 | SLT (4) | 94.8s | 95.4s | 29 | SLT (4) | 94.9s | 95.4s | 0.8333 | 0.100s | MATCHED | CORRECT | 78.38% |
| 103 | SLT (4) | 97.5s | 98.1s | 30 | SLT (4) | 97.5s | 98.1s | 1.0000 | 0.000s | MATCHED | CORRECT | 69.67% |
| 104 | SLT (4) | 100.2s | 100.8s | 31 | SLT (4) | 100.3s | 100.8s | 0.8333 | 0.100s | MATCHED | CORRECT | 94.04% |
| 105 | SLT (4) | 102.9s | 103.5s | 32 | SLT (4) | 102.9s | 103.5s | 1.0000 | 0.000s | MATCHED | CORRECT | 85.70% |
| 106 | SLT (4) | 105.6s | 106.2s | 33 | SLT (4) | 105.6s | 106.2s | 1.0000 | 0.000s | MATCHED | CORRECT | 84.31% |
| 107 | SLT (4) | 108.3s | 108.9s | 34 | SLT (4) | 108.3s | 108.9s | 1.0000 | 0.000s | MATCHED | CORRECT | 89.27% |
| 108 | SLT (4) | 111.0s | 111.6s | 35 | SLT (4) | 111.0s | 111.4s | 0.6667 | 0.000s | MATCHED | CORRECT | 93.43% |
| 109 | SLT (4) | 113.7s | 114.3s | 36 | SLT (4) | 113.7s | 114.3s | 1.0000 | 0.000s | MATCHED | CORRECT | 59.89% |

**Driver-3 Unmatched Predictions (False Positives / Spurious Alarms):**
- **Pred 5:** Class 3 (SRT), $14.3\text{ s} - 15.0\text{ s}$, Duration: $0.70\text{ s}$, Confidence: $53.19\%$. (Overlaps missed GT 23 [$13.7\text{ s}-15.0\text{ s}$], but onset difference was $\Delta t = 0.60\text{ s} > 0.50\text{ s}$, failing the onset tolerance gate).

*Driver-3 Verification:* $35\text{ Matched} + 4\text{ Missed GT} = 39\text{ GT}$. $35\text{ Matched} + 1\text{ FP Pred} = 36\text{ Pred}$. Matched accuracy: $34/35 = \mathbf{97.14\%}$. Exactly 1 misclassification (GT 22, True SRT $\to$ Pred SA).

---

## 3. Corrected Class-Level Confusion Matrix

### 3.1 Mathematical Definitions and Accounting Rules

To prevent ambiguity, the four-class confusion matrix strictly follows canonical multi-class event evaluation definitions:

1. **Class True Positive ($\text{TP}_c$):** A ground-truth event of class $c$ successfully matched to a predicted event whose predicted class is also $c$.
   $$\sum_{c=1}^4 \text{TP}_c = \text{Total correctly classified matched events} = 95$$

2. **Class False Positive ($\text{FP}_c$):**
   $$\text{FP}_c = \text{FP}_c^{\text{misclass}} + \text{FP}_c^{\text{unmatched}}$$
   - $\text{FP}_c^{\text{misclass}}$: A matched event of true class $k \neq c$ incorrectly predicted as class $c$.
   - $\text{FP}_c^{\text{unmatched}}$: An unmatched prediction whose emitted class is $c$.
   $$\sum_{c=1}^4 \text{FP}_c = \text{Total misclassified matched events} (1) + \text{Total unmatched predictions} (5 \text{ or } 6)$$

3. **Class False Negative ($\text{FN}_c$):**
   $$\text{FN}_c = \text{FN}_c^{\text{misclass}} + \text{FN}_c^{\text{missed}}$$
   - $\text{FN}_c^{\text{misclass}}$: A matched event of true class $c$ incorrectly predicted as class $k \neq c$.
   - $\text{FN}_c^{\text{missed}}$: An unmatched ground-truth event of class $c$ (completely missed detection).
   $$\sum_{c=1}^4 \text{FN}_c = \text{Total misclassified matched events} (1) + \text{Total unmatched GT events} (13) = \mathbf{14}$$

4. **Class Ground Truth Invariant:**
   $$\text{GT}_c = \text{TP}_c + \text{FN}_c \quad \implies \quad \sum_{c=1}^4 \text{GT}_c = 95 + 14 = \mathbf{109}$$

5. **Class Prediction Invariant:**
   $$\text{Pred}_c = \text{TP}_c + \text{FP}_c \quad \implies \quad \sum_{c=1}^4 \text{Pred}_c = 95 + 6 = \mathbf{101} \quad (\text{Clean Combined})$$

---

### 3.2 Audit of the Previous Forensic Report's Discrepancies

In `phase1_forensic_validation.md`, Table Section 2 claimed:
$$\text{Sum of } \text{FN}_c = 15 \quad (\text{Reported as: } 14 \text{ missed} + 1 \text{ misclass})$$
$$\text{SRT GT Count} = 25, \quad \text{SRT Matched} = 22, \quad \text{SRT Missed} = \text{"3 (+1 misclass)"} \implies \text{Total } \text{FN}_{\text{SRT}} = 4$$

**Forensic Audit of the Errors:**
1. **Total Missed Detections are $13$, not $14$:**  
   Total ground-truth events = $109$. Total matched events = $96$. By definition, total missed detections = $109 - 96 = \mathbf{13}$.  
   The previous report incorrectly asserted that there were 14 missed detections.
2. **SRT Missed Detections are $2$, not $3$:**  
   Examining Driver-3 annotations:
   - GT 22 was matched to Pred 4 (misclassified as SA). It is NOT missed; it is matched.
   - GT 23 was missed (unmatched GT).
   - GT 24 was missed (unmatched GT).  
   There are only **2 missed detections** in SRT across the entire dataset. The previous report mistakenly counted 3 missed detections AND added 1 misclassification, arriving at $\text{FN}_{\text{SRT}} = 4$.  
   With $\text{FN}_{\text{SRT}} = 4$, $\text{TP} + \text{FN} = 22 + 4 = 26$, violating the verified ground-truth total of $25$ SRT events.
3. **Corrected SRT Accounting:**  
   $\text{GT}_{\text{SRT}} = 25$. Correct detections $\text{TP}_{\text{SRT}} = 22$. Misclassified $\text{FN}_{\text{SRT}}^{\text{misclass}} = 1$. Missed $\text{FN}_{\text{SRT}}^{\text{missed}} = 2$.  
   $$\text{Total } \text{FN}_{\text{SRT}} = 1 + 2 = \mathbf{3}$$
   $$\text{Recall}_{\text{SRT}} = \frac{22}{22 + 3} = \frac{22}{25} = \mathbf{88.00\%} \quad (\text{not } \frac{22}{26} = 84.62\%)$$
   $$\text{F1}_{\text{SRT}} = 2 \times \frac{0.956522 \times 0.880000}{0.956522 + 0.880000} = \mathbf{91.67\%} \quad (\text{not } 89.80\%)$$

---

### 3.3 Authoritative Reconstructed Four-Class Confusion Matrix (Clean Driver-Wise Combined, 101 Predictions)

This table represents the true boundary-independent baseline aggregated across Driver-1, Driver-2, and Driver-3 without synthetic inter-driver boundary artifacts.

| Class ID | Maneuver Name | True GT | Matched Correct ($\text{TP}_c$) | Misclass Error ($\text{FP}_c^{\text{err}}$) | Spurious Detection ($\text{FP}_c^{\text{spur}}$) | Total $\text{FP}_c$ | Missed Detection ($\text{FN}_c^{\text{miss}}$) | Misclass Loss ($\text{FN}_c^{\text{err}}$) | Total $\text{FN}_c$ | Precision ($\%$) | Recall ($\%$) | Class F1 ($\%$) |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | Sudden Acceleration (SA) | 28 | 26 | 1 (GT 22) | 1 (D1 P16) | 2 | 2 (D1 G32, D3 G47) | 0 | 2 | 92.86% | 92.86% | **92.86%** |
| **2** | Sudden Deceleration (SD) | 24 | 23 | 0 | 0 | 0 | 1 (D3 G77) | 0 | 1 | 100.00% | 95.83% | **97.87%** |
| **3** | Sudden Right Turn (SRT) | 25 | 22 | 0 | 1 (D3 P5) | 1 | 2 (D3 G23, D3 G24) | 1 (GT 22) | 3 | 95.65% | 88.00% | **91.67%** |
| **4** | Sudden Left Turn (SLT) | 32 | 24 | 0 | 3 (D1 P29, D2 P27, P28) | 3 | 8 (D1: 5, D2: 3) | 0 | 8 | 88.89% | 75.00% | **81.36%** |
| **Total / Macro** | **Clean Combined Baseline** | **109** | **95** | **1** | **5** | **6** | **13** | **1** | **14** | **94.35%** | **87.92%** | **90.94%** |

*Exact Verification Checks (Clean Combined):*
- $\sum \text{TP}_c = 26 + 23 + 22 + 24 = 95$ (Matches correctly classified matched events).
- $\sum \text{FP}_c = 2 + 0 + 1 + 3 = 6 = 1\text{ (misclass)} + 5\text{ (unmatched predictions)}$.
- $\sum \text{FN}_c = 2 + 1 + 3 + 8 = 14 = 1\text{ (misclass)} + 13\text{ (unmatched GT)}$.
- $\sum \text{GT}_c = 28 + 24 + 25 + 32 = 109$.
- $\text{Macro-F1} = (92.8571\% + 97.8723\% + 91.6667\% + 81.3559\%) / 4 = \mathbf{90.9380\%} \approx \mathbf{90.94\%}$.

---

### 3.4 Pooled Replay Confusion Matrix (102 Predictions, Includes PRED 79 Boundary Artifact)

For complete mathematical disclosure, the table below reflects the raw pooled continuous stream evaluated in `streaming_evaluation_results.json`:

| Class ID | Maneuver Name | True GT | Matched Correct ($\text{TP}_c$) | Misclass Error ($\text{FP}_c^{\text{err}}$) | Spurious Detection ($\text{FP}_c^{\text{spur}}$) | Total $\text{FP}_c$ | Missed Detection ($\text{FN}_c^{\text{miss}}$) | Misclass Loss ($\text{FN}_c^{\text{err}}$) | Total $\text{FN}_c$ | Precision ($\%$) | Recall ($\%$) | Class F1 ($\%$) |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | Sudden Acceleration (SA) | 28 | 26 | 1 | 1 (P31) | 2 | 2 | 0 | 2 | 92.86% | 92.86% | **92.86%** |
| **2** | Sudden Deceleration (SD) | 24 | 23 | 0 | 0 | 0 | 1 | 0 | 1 | 100.00% | 95.83% | **97.87%** |
| **3** | Sudden Right Turn (SRT) | 25 | 22 | 0 | 1 (P23) | 1 | 2 | 1 | 3 | 95.65% | 88.00% | **91.67%** |
| **4** | Sudden Left Turn (SLT) | 32 | 24 | 0 | 4 (P76, P79*, P80, P81) | 4 | 8 | 0 | 8 | 85.71% | 75.00% | **80.00%** |
| **Total / Macro** | **Pooled Replay Benchmark** | **109** | **95** | **1** | **6** | **7** | **13** | **1** | **14** | **93.56%** | **87.92%** | **90.60%** |

*\*Note on PRED 79:* PRED 79 ($262.5\text{ s} - 262.6\text{ s}$, duration $0.10\text{ s}$, confidence $90.40\%$, class SLT) is an inter-driver boundary edge artifact generated when Driver-1 telemetry transitioned through a synthetic cruising gap into Driver-2. It adds exactly $+1\text{ FP}$ to SLT, reducing SLT precision from $88.89\%$ to $85.71\%$, SLT F1 from $81.36\%$ to $80.00\%$, and overall Macro-F1 from $90.94\%$ to **$90.60\%$**.

---

## 4. Per-Driver and Combined Detection Metrics

### 4.1 Metric Formulas

- **Detection Precision:** $\text{Precision}_{\text{det}} = \frac{\text{TP}_{\text{det}}}{\text{Total Predicted Events}} = \frac{\text{TP}_{\text{det}}}{\text{TP}_{\text{det}} + \text{FP}_{\text{det}}}$
- **Detection Recall:** $\text{Recall}_{\text{det}} = \frac{\text{TP}_{\text{det}}}{\text{Total GT Events}} = \frac{\text{TP}_{\text{det}}}{\text{TP}_{\text{det}} + \text{FN}_{\text{det}}}$
- **Detection F1:** $\text{F1}_{\text{det}} = \frac{2 \times \text{Precision}_{\text{det}} \times \text{Recall}_{\text{det}}}{\text{Precision}_{\text{det}} + \text{Recall}_{\text{det}}}$
- **Matched Classification Accuracy:** $\text{Accuracy}_{\text{matched}} = \frac{\sum_{i \in \text{matched}} \mathbb{I}(\hat{y}_i = y_i)}{\text{TP}_{\text{det}}}$
- **False Events per Minute:** $\text{FE/min} = \frac{\text{FP}_{\text{det}}}{\text{Duration (seconds)} / 60.0}$

---

### 4.2 Comprehensive Event Detection Performance Summary

| Cohort | GT Count | Pred Count | TP | FP | FN | Duplicate Preds | Precision ($\%$) | Recall ($\%$) | Detection F1 ($\%$) | Matched Acc ($\%$) | Streaming Macro-F1 ($\%$) | Stream Duration (s) | False Events / min |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Driver-1** | 35 | 31 | 29 | 2 | 6 | 0 | 93.55% | 82.86% | 87.88% | 100.00% | 85.00% | 104.8s | **1.1450** |
| **Driver-2** | 35 | 34 | 32 | 2 | 3 | 0 | 94.12% | 91.43% | 92.75% | 100.00% | 92.65% | 104.3s | **1.1505** |
| **Driver-3** | 39 | 36 | 35 | 1 | 4 | 0 | 97.22% | 89.74% | 93.33% | 97.14% | 87.07% | 114.3s | **0.5249** |
| **Driver-Wise Combined** | **109** | **101** | **96** | **5** | **13** | **0** | **95.05%** | **88.07%** | **91.43%** | **98.96%** | **90.94%** | **323.4s** | **0.9276** |
| **Pooled Stream (Artifact)** | 109 | 102 | 96 | 6 | 13 | 0 | 94.12% | 88.07% | 91.00% | 98.96% | 90.60% | 327.4s | 1.0996 |

### 4.3 Why Pooled and Driver-Wise Sum Results Are Not Interchangeable

The pooled evaluation is not simply the sum of isolated driver evaluations because of state machine boundary semantics:
1. **Accumulator Buffer Trimming at Stream Termination:** In isolated Driver-1 evaluation, the stream terminates abruptly after observation 1,048 (row 945). The detector's `flush()` method checks `active_len < 4` and discards the sub-threshold 2-sample stub ($0.10\text{ s}$).
2. **Hysteresis Continuation into Inter-Driver Gaps:** In pooled evaluation, row 945 is immediately followed by a 2.0-second synthetic cruising gap into Driver-2. The accumulator ingested 3 quiet frames before the quiet hysteresis timer fired. At finalization, the check `len(self._buffer) < 4` evaluated $5 < 4$ as False (because it counted trailing quiet frames before trimming). Consequently, an undersized 2-sample prediction (PRED 79, duration $0.10\text{ s}$) was emitted.
3. **Scientific Implication:** Evaluating across synthetic boundaries without resetting internal accumulator states introduces non-physical boundary artifacts. The **Driver-Wise Combined Cohort (101 predictions, 90.94% Macro-F1)** is the authoritative, clean representation of boundary-independent detector performance.

---

## 5. Replay Duration and Synthetic Gap Reconciliation

### 5.1 Physical Accounting of Replay Telemetry

The repository dataset [`dataset/processed/dataset_b_processed_all.csv`](file:///D:/driver%20ml%20model/driver-monitoring-agent/dataset/processed/dataset_b_processed_all.csv) contains **1,114 maneuver rows** at 10 Hz across 109 events. In natural telemetry, maneuvers are separated by ordinary cruising. Because Dataset B contains zero natural ordinary driving, `StreamingReplayEngine` inserts configurable synthetic cruising gaps ($2.0\text{ s} = 20\text{ samples}$ of nominal 1g cruising) between consecutive events.

| Cohort | Maneuver Rows | Event Count | Number of Gaps | Gap Rows (20/gap) | Total Observations | Total Duration (s) | Duration (min) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Driver-1** | 368 | 35 | 34 | 680 | 1,048 | 104.800 s | 1.7467 min |
| **Driver-2** | 363 | 35 | 34 | 680 | 1,043 | 104.300 s | 1.7383 min |
| **Driver-3** | 383 | 39 | 38 | 760 | 1,143 | 114.300 s | 1.9050 min |
| **Driver-Wise Sum** | **1,114** | **109** | **106** | **2,120** | **3,234** | **323.400 s** | **5.3900 min** |
| **Pooled Stream** | **1,114** | **109** | **108** | **2,160** | **3,274** | **327.400 s** | **5.4567 min** |
| **Discrepancy (Pooled - Sum)** | **0** | **0** | **+2** | **+40** | **+40** | **+4.000 s** | **+0.0667 min** |

### 5.2 Explanation of the 40-Observation / 4.0-Second Discrepancy

The 40-observation / 4.0-second discrepancy is explained by the loop structure in `StreamingReplayEngine.generate_stream()`:
- When a driver is isolated, gaps are inserted between consecutive events within that driver. For $N$ events, there are $N - 1$ gaps.
  - Driver-1: $35 - 1 = 34\text{ gaps}$.
  - Driver-2: $35 - 1 = 34\text{ gaps}$.
  - Driver-3: $39 - 1 = 38\text{ gaps}$.
  - Driver Sum: $34 + 34 + 38 = \mathbf{106\text{ gaps}}$ ($2,120\text{ gap rows}$).
- When all drivers are pooled into one stream, all 109 events are replayed sequentially in a single pass. For 109 events, there are $109 - 1 = \mathbf{108\text{ gaps}}$ ($2,160\text{ gap rows}$).
- The two additional gaps ($2 \times 20 = 40\text{ observations} = 4.0\text{ s}$) are:
  1. **Gap #35:** Between Driver-1 GT 85 and Driver-2 GT 10.
  2. **Gap #70:** Between Driver-2 GT 94 and Driver-3 GT 19.

### 5.3 Denominator Reconciliation for False Events Per Minute

Using the wrong duration denominator produces incorrect false alarm rates:
- **Correct Driver-Wise Combined Calculation:**
  $$\text{FE/min} = \frac{5\text{ FP}}{323.4\text{ s} / 60.0} = \frac{5}{5.3900\text{ min}} = \mathbf{0.9276\text{ FE/min}}$$
- **Correct Pooled Calculation:**
  $$\text{FE/min} = \frac{6\text{ FP}}{327.4\text{ s} / 60.0} = \frac{6}{5.4567\text{ min}} = \mathbf{1.0996\text{ FE/min}}$$
- **Erroneous Hybrid Mix (Pooled Duration with Driver FP):**
  $$\frac{5\text{ FP}}{327.4\text{ s} / 60.0} = 0.9163\text{ FE/min} \quad (\text{Invalid})$$

---

## 6. Audit of the 90.47% Macro-F1 Claim

### 6.1 Derivation of the Erroneous 90.47% Claim

The figure $90.47\%$ in `phase1_forensic_validation.md` was traced to the following intermediate step:
$$\text{Recall}_{\text{SRT}} = \frac{22}{\mathbf{26}} = 84.6154\%$$
$$\text{F1}_{\text{SRT}} = 2 \times \frac{0.956522 \times 0.846154}{0.956522 + 0.846154} = \mathbf{89.7959\%}$$
$$\text{Macro-F1} = \frac{92.8571\% + 97.8723\% + \mathbf{89.7959\%} + 81.3559\%}{4} = \frac{361.8812\%}{4} = \mathbf{90.4703\%} \approx \mathbf{90.47\%}$$

Because the ground truth contains only 25 SRT events, the denominator $26$ was mathematically invalid. The claim that Macro-F1 is $90.47\%$ is **REFUTED**.

---

### 6.2 Derivation of the Pooled 90.60% Benchmark

In `streaming_evaluation_results.json`, `exp2_cruising_stream["Pooled_All_Drivers"]` reports Macro-F1 = $0.9060$ ($90.60\%$). This calculation is reproduced exactly:
- SA: $\text{F1} = 92.8571\%$
- SD: $\text{F1} = 97.8723\%$
- SRT: $\text{F1} = 91.6667\%$ (correct denominator $25$, recall $88.00\%$)
- SLT: $\text{F1} = 80.0000\%$ (contaminated by PRED 79, precision $24/28 = 85.71\%$)
$$\text{Macro-F1}_{\text{pooled}} = \frac{92.8571\% + 97.8723\% + 91.6667\% + 80.0000\%}{4} = \frac{362.3961\%}{4} = \mathbf{90.5990\%} \approx \mathbf{90.60\%}$$

This confirms that $90.60\%$ was directly calculated on the pooled stream containing the inter-driver boundary artifact.

---

### 6.3 Authoritative Performance Baselines

| Metric | Clean Driver-Wise Combined (Authoritative Baseline) | Pooled Continuous Stream (Benchmark with Artifact) | Erroneous Previous Claim |
|:---|---:|---:|---:|
| **Evaluated Predictions** | 101 | 102 | 101 |
| **Matched True Positives** | 96 | 96 | 96 |
| **False Positives (Detection)** | 5 | 6 | 5 |
| **False Negatives (Detection)** | 13 | 13 | 13 |
| **Matched Accuracy** | 98.96% (95/96) | 98.96% (95/96) | 98.96% |
| **SA F1-Score** | 92.86% | 92.86% | 92.86% |
| **SD F1-Score** | 97.87% | 97.87% | 97.87% |
| **SRT F1-Score** | **91.67%** | **91.67%** | 89.80% (arithmetic error) |
| **SLT F1-Score** | **81.36%** | **80.00%** (boundary artifact) | 81.36% |
| **Macro-F1** | **90.94%** | **90.60%** | **90.47%** (arithmetic error) |
| **False Events / min** | **0.93** (0.9276) | **1.10** (1.0996) | 0.93 |

---

## 7. False-Event Target Assessment

### 7.1 Compliance Audit Against Stage 7 Requirement

The approved Stage 7 Phase 1 specification defines the acceptance criterion:
$$\text{False Events per Minute} \le \mathbf{0.50}$$

The measured empirical results across all configurations are:

| Configuration | Testbed Stream | Stream Duration | False Positives | Measured FE / min | Target ($\le 0.50$) | Compliance Status |
|:---|:---|---:|---:|---:|:---:|:---:|
| **Base Configuration** | Clean Driver-Wise Combined | 323.4 s (5.39 min) | 5 | **0.9276** | $\le 0.50$ | **FAILS** (+85.5% exceedance) |
| **Base Configuration** | Pooled Stream | 327.4 s (5.46 min) | 6 | **1.0996** | $\le 0.50$ | **FAILS** (+119.9% exceedance) |
| **Sensitivity: Onset $N=1$** | Pooled Stream | 327.4 s (5.46 min) | 3 | **0.5498** | $\le 0.50$ | **FAILS** (+10.0% exceedance) |
| **Sensitivity: Onset $N=3$** | Pooled Stream | 327.4 s (5.46 min) | 5 | **0.9163** | $\le 0.50$ | **FAILS** (+83.3% exceedance) |
| **Sensitivity: Quiet $200\text{ ms}$** | Pooled Stream | 327.4 s (5.46 min) | 15 | **2.7489** | $\le 0.50$ | **FAILS** (+449.8% exceedance) |
| **Sensitivity: Quiet $500\text{ ms}$** | Pooled Stream | 327.4 s (5.46 min) | 5 | **0.9163** | $\le 0.50$ | **FAILS** (+83.3% exceedance) |

**Audit Determination:**  
The base configuration **FAILS** the false-event rate requirement. Even the most aggressive sensitivity configuration ($N=1$ onset samples) achieved $0.5498\text{ FE/min}$, which also **FAILS** the requirement ($0.55 > 0.50$).

---

### 7.2 Forensic Critique of Proposed Mitigations

The previous report proposed that increasing cooldown from $0.5\text{ s}$ to $1.0\text{ s}$ or introducing a jerk zero-crossing threshold would bring false events $\le 0.50\text{ FE/min}$.

Under strict scientific engineering principles:
1. **Hypotheses Are Not Proof:** A proposed engineering mitigation cannot be reported as effective without empirical validation.
2. **Trade-off Risks:** Increasing cooldown to $1.0\text{ s}$ risks blinding the detector to rapid consecutive emergency maneuvers (e.g., an evasive swerve immediately followed by emergency braking), increasing False Negatives.
3. **No Retraining or Parameter Mutation in Audit:** As required by governance, no production parameters were modified during this audit. Stage 7 Phase 2 must execute a controlled parameter sweep on `train + val` to evaluate tail-decay suppression before evaluating on locked test data.

---

## 8. Remaining Methodological Limitations

The following scientific limitations must remain explicitly documented in all project artifacts:

1. **Dataset B Lacks Genuine Ordinary Driving Telemetry:**  
   Dataset B consists exclusively of isolated maneuver recordings ($1,114$ rows across 109 events). There are zero continuous hours of natural commuting, highway cruising, or city traffic. Replay continuity was achieved by inserting synthetic 1g quiescent frames. Consequently, the detector's resistance to road surface bumps, potholes, railway crossings, engine vibrations, and mild highway bends remains **unverified on natural telemetry**.

2. **Synthetic Cruising Gaps Are Not Road Telemetry:**  
   The synthetic cruising gaps assume mathematically constant acceleration ($9.80665\text{ m/s}^2$ downward) and zero angular velocity. Real sensor streams exhibit persistent low-frequency road noise, chassis pitch/roll, and sensor drift.

3. **Parameter Sensitivity Sweep Selection Bias:**  
   The parameter sensitivity sweep in `exp3_sensitivity` evaluated variations against the pooled 109-event dataset, which includes the 17 locked test events. This creates selection bias if parameter choices are guided by this sweep. In Stage 7 Phase 2, detector hyperparameters must be tuned strictly on `train + val` (92 events), locked, and evaluated exactly once on `test` (17 events).

4. **Preservation of Frozen Model Artifacts:**  
   The underlying Extra Trees classifier (`dataset_b_extra_trees_exp_v1.joblib`) and StandardScaler (`dataset_b_scaler_exp_v1.joblib`) were fitted on isolated sample windows and cannot be modified.

5. **Research Classification:**  
   This streaming testbed is an academic demonstration and experimental evaluation tool. It does not constitute proof of automotive-grade safety or production road-readiness.

---

## 9. Final Acceptance Recommendation

### Acceptance Decision: CONDITIONAL ACCEPT

Phase 1 cannot be granted full, unconditional acceptance because the detector fails the mandatory false-event target ($\le 0.50\text{ FE/min}$) and replay relies on synthetic cruising gaps.

However, Phase 1 has successfully accomplished its core engineering mission:
- Created an isolated, boundary-independent streaming evaluation testbed.
- Proved that the frozen Extra Trees classifier and KinematicEventGate can detect maneuvers from continuous streaming telemetry without ground-truth boundary exposure.
- Achieved **$88.07\%$ event detection recall**, **$98.96\%$ matched classification accuracy**, and **$90.94\%$ streaming Macro-F1** on clean continuous replay.
- Demonstrated sub-second early provisional warning latency (median $200\text{ ms}$).
- Reconciled all arithmetic and metric discrepancies with $100\%$ mathematical transparency.

### Mandatory Conditions for Stage 7 Phase 2 Authorization:
1. **Partition Isolation:** Hyperparameter tuning (cooldown duration, onset samples, tail-decay hysteresis) must be performed exclusively on `train + val` (92 events).
2. **Tail-Decay Suppression Experiment:** Implement and empirically validate tail-decay suppression on `train + val` to demonstrate $\le 0.50\text{ FE/min}$ prior to test set evaluation.
3. **Unbiased Test Set Evaluation:** Execute a single, final evaluation run on the 17 locked test events with fixed hyperparameters to report unbiased generalization.
4. **Boundary Edge Fix in Testbed Code:** Update `_finalize_event()` to evaluate `active_len` rather than raw `len(self._buffer)` when checking minimum duration, eliminating boundary stub artifacts.
5. **Frozen Artifact Immutability:** Maintain all four frozen production model artifacts without modification or retraining.

---

## 10. Audit Verification and System Integrity Sign-Off

### 10.1 Cryptographic Hash Verification (Bit-for-Bit Match)

All four production model artifacts were cryptographically audited before and after this reconciliation:

| Artifact Path | Expected SHA-256 Hash | Post-Audit Hash | Status |
|:---|:---|:---|:---:|
| `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib` | `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` | `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90` | **MATCH** |
| `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib` | `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` | `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576` | **MATCH** |
| `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib` | `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` | `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc` | **MATCH** |
| `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib` | `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` | `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5` | **MATCH** |

### 10.2 Regression Test Suite Verification

Full test suite execution: **161 passed**, 0 failed, 0 regressions across:
- `tests/test_backend_api.py` (10 tests)
- `tests/test_model_training_and_inference.py` (53 tests)
- `tests/test_safety_agent.py` (86 tests)
- `tests/test_streaming_testbed.py` (12 tests)

### 10.3 Git Status and Change Control

- **Branch:** `stage6-phase1`
- **Head Commit:** `f746291`
- **Tracked Code Modified:** 0 files modified by this audit.
- **New Deliverables Created:**
  - `reports/stage7/phase1_final_numerical_reconciliation.md`
- **Git Commit/Push:** None performed.

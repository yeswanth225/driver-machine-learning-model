# Ground-Truth Annotation Guide & Standard Operating Procedure

**Project:** Intelligent Driver Activity Monitoring & Safety Agent  
**Module:** Independent Human Observer Ground-Truth Labeling Protocol  
**Document ID:** `IDAMS-SOP-ANNOTATION-01`  
**Status:** Approved Standard Operating Procedure  
**Date:** November 2026  

---

## 1. Objectives & Ground-Truth Rationale

Supervised evaluation of machine learning models requires rigorously validated, independent ground-truth annotations that are completely decoupled from model predictions.

To establish genuine ground truth without circular bias:
1. Ground-truth labels are created solely by human observers based on physical driving observations and verified vehicle kinematics.
2. Annotations are stored immutably in `dataset/annotations/ground_truth_manifest.json` with unique UUIDs, timestamp bounds, observer identifiers, and observational evidence notes.
3. Machine learning models never access annotations during feature extraction or online inference.

---

## 2. Standard Label Taxonomy & Kinematic Criteria

The annotation taxonomy defines seven mutually exclusive behavioral classes:

| Class Label | Maneuver / Style Category | Primary Visual & Behavioral Indicators | Mandatory Kinematic Thresholds |
|:---|:---|:---|:---|
| `NORMAL` | Quiescent Driving | Steady lane keeping, gradual speed changes, compliance with speed limits and lane markings. | $|a_x| < 0.25g$, $|a_y| < 0.25g$, $\|\vec{\omega}\| < 0.20\text{ rad/s}$ |
| `AGGRESSIVE` | Hostile Driving Style | Tailgating, rapid repeated lane changes without signaling, abrupt weaving, high engine throttle transitions. | Sustained elevated variance, high jerk density ($> 3$ events/min) |
| `SUDDEN_ACCELERATION` | Hard Acceleration Stomp | Driver floors the accelerator pedal; rapid longitudinal weight transfer to rear axle. | $a_x \ge +0.35g$, Jerk $\frac{\Delta a_x}{\Delta t} \ge +1.0g/\text{s}$ |
| `SUDDEN_DECELERATION` | Panic / Emergency Braking | Driver slams brake pedal; rapid nose dive and seatbelt pre-tensioning lock. | $a_x \le -0.45g$, Jerk $\frac{\Delta a_x}{\Delta t} \le -1.2g/\text{s}$ |
| `SUDDEN_RIGHT_TURN` | Sharp / Evasive Right Turn | Rapid steering wheel rotation to the right at speed; high leftward centrifugal force. | $a_y \le -0.35g$, Yaw rate $\omega_z \ge +0.30\text{ rad/s}$ |
| `SUDDEN_LEFT_TURN` | Sharp / Evasive Left Turn | Rapid steering wheel rotation to the left at speed; high rightward centrifugal force. | $a_y \ge +0.35g$, Yaw rate $\omega_z \le -0.30\text{ rad/s}$ |
| `UNCERTAIN` | Ambiguous / Pothole Event | Road surface shock, speed breaker impact, or obstructed view where intent is indeterminable. | Discarded from supervised model benchmarks |

---

## 3. Observation & Bounding Methodology

### 3.1 Real-Time In-Car Tagging
During the drive, the in-car observer uses the mobile cockpit HUD or the web dashboard to mark events in real time:
1. When a maneuver begins, the observer taps the corresponding event button (e.g., `SUDDEN_ACCEL`, `SUDDEN_BRAKE`, `SHARP_TURN`).
2. The system automatically registers the event timestamp $t_{\text{mark}}$ and creates an initial bounded window $[t_{\text{mark}} - 2.5\text{ s}, t_{\text{mark}}]$.

### 3.2 Post-Drive Kinematic Window Refinement
During post-drive data review:
1. The lead researcher plots the 20 Hz longitudinal and lateral acceleration waveforms around $t_{\text{mark}}$.
2. **Onset Timestamp ($t_{\text{start}}$):** Set at the exact inflection point where $|a(t)|$ departs the baseline noise band ($> 3\sigma$).
3. **Termination Timestamp ($t_{\text{end}}$):** Set at the instant when vehicle kinematics return to within $10\%$ of quiescent cruising baseline.
4. The refined time window $[t_{\text{start}}, t_{\text{end}}]$ is updated with verification status `VERIFIED`.

---

## 4. Storage & Manifest Schema

All annotations are stored in `dataset/annotations/ground_truth_manifest.json`:

```json
[
  {
    "annotation_id": "ann_1700000045000_1",
    "session_id": "sess_1700000000_DRV_001",
    "driver_id": "DRV_001",
    "start_timestamp": 45.20,
    "end_timestamp": 47.80,
    "label": "SUDDEN_DECELERATION",
    "annotator_id": "OBS_01",
    "evidence": "Vehicle braked suddenly from 68 km/h to 15 km/h to avoid merging truck. Nose pitch visible.",
    "verification_status": "VERIFIED",
    "created_at": "2026-11-24T18:30:45Z"
  }
]
```

---

## 5. Inter-Annotator Reliability & Conflict Resolution

To maintain academic and industrial rigor:
1. **Dual Independent Review:** $20\%$ of all recorded sessions are independently annotated by a second researcher from synchronized dashcam video.
2. **Statistical Agreement Metric:** Inter-annotator agreement is computed using Cohen's Kappa coefficient:
   $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
3. **Acceptance Threshold:** Campaigns must achieve $\kappa \ge 0.85$. Any event with discrepant categorization is marked `CONTESTED` and reviewed by the lead safety engineer before benchmark inclusion.

---
*Certified by the IDAMS Annotation & Data Governance Committee.*

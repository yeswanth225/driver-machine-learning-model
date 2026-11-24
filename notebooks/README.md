# Intelligent Driver Activity Monitoring & Safety Agent — Jupyter Notebooks

This directory contains two standalone, fully executable, and scientifically rigorous Jupyter Notebooks accompanying the **Intelligent Driver Activity Monitoring & Safety Agent** project:

1. [`Driver_Activity_EDA.ipynb`](file:///notebooks/Driver_Activity_EDA.ipynb) — Multi-Dataset Exploratory Data Analysis & Kinematic Telemetry Audit
2. [`Driver_Activity_ML_Models.ipynb`](file:///notebooks/Driver_Activity_ML_Models.ipynb) — Machine Learning Models, Cross-Validation & Production Governance

All notebooks are designed to execute sequentially without external dependencies beyond standard scientific Python packages (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`).

---

## 1. Notebook Descriptions

### `Driver_Activity_EDA.ipynb`
A comprehensive 9-section data audit and exploratory kinematic analysis covering:
- **Multi-Dataset Inventory:** Systematic auditing of primary Datasets C (Pipeline A) and B (Pipeline B), alongside auxiliary naturalistic Datasets A1, A2, F, and G.
- **Physical Sensor Tilt Analysis:** Mathematical decomposition of accelerometer gravity vectors between Session 1 ($\vec{a}_1 = [-0.048, 0.177, -0.031]$) and Session 2 ($\vec{a}_2 = [0.055, 0.183, -0.028]$), establishing an uncompensated $31.36^\circ$ mounting pitch rotation.
- **Temporal Autocorrelation & Effective Sample Size:** Sokal integrated autocorrelation time calculation demonstrating that Dataset C's 13,840 nominal frames correspond to an effective independent sample size of **$N_{\text{eff}} \approx 707$ observations**.
- **Contiguous Driving Episode Segmentation:** Segmentation into 17 contiguous behavioral episodes (7 in Session 1, 10 in Session 2).
- **Dataset B Maneuver Kinematics:** Analysis of 109 isolated maneuver trials across 3 human drivers, showing clean kinematic separation between longitudinal maneuvers (AccX for SA/SD) and yaw maneuvers (GyroZ for SRT/SLT).

### `Driver_Activity_ML_Models.ipynb`
A complete 9-section machine learning engineering and governance notebook covering:
- **Pipeline Architecture:** Dual late-fusion topology decoupling continuous driving style classification (Pipeline A) from acute sudden maneuver detection (Pipeline B) and rule-based safety evaluation.
- **Production Baseline Evaluation:** Evaluation of the frozen production RBF SVM ($C=2.0$) on the chronological test split (69.93% accuracy, 70.83% balanced accuracy).
- **Bidirectional Cross-Session Validation:** Rigorous evaluation of forward ($S1 \rightarrow S2$: 91.43%) vs. reverse ($S2 \rightarrow S1$: 74.71%) session holdout, yielding an arithmetic mean of 83.07% and pooled sample accuracy of 82.49%.
- **Multi-Model Architectural Benchmark:** Comparison of Logistic Regression, RBF SVM, Random Forest, ExtraTrees, and XGBoost across physical driving sessions.
- **Pipeline B & 3-Fold LODO Validation:** Leave-One-Driver-Out cross-validation across Driver-1 (97.14%), Driver-2 (100.0%), and Driver-3 (82.05%), achieving a mean event accuracy of **$93.06\% \pm 7.87\%$**.
- **KinematicEventGate Integration:** Streaming telemetry evaluation demonstrating rejection of quiescent cruise frames and excitation during acute braking maneuvers.
- **Production Model Governance:** Cryptographic SHA-256 verification of frozen production model artifacts and export of standardized summaries to `notebooks/outputs/`.

---

## 2. Directory Structure & Outputs

```
notebooks/
├── Driver_Activity_EDA.ipynb            # Complete exploratory data analysis
├── Driver_Activity_ML_Models.ipynb      # Complete ML training & evaluation
├── README.md                            # Documentation & execution guide
└── outputs/                             # Generated benchmark artifacts
    ├── model_comparison_matrix.csv      # Unified comparison across all models
    ├── pipeline_a_comparison.png        # Bar chart comparing Pipeline A candidates
    ├── pipeline_a_evaluation_summary.json # Pipeline A diagnostic metrics
    └── pipeline_b_evaluation_summary.json # Pipeline B diagnostic metrics & LODO
```

---

## 3. Execution Instructions

### Prerequisites
Ensure your active Python environment has the project dependencies installed:
```powershell
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost joblib jupyter nbconvert
```

### Running the Notebooks

Both notebooks feature self-resolving repository path logic: they can be run from the repository root or from within the `notebooks/` directory.

#### Option A: Interactive Execution
Open the notebooks using Jupyter Lab or VS Code:
```powershell
jupyter lab notebooks/Driver_Activity_EDA.ipynb
jupyter lab notebooks/Driver_Activity_ML_Models.ipynb
```

#### Option B: Headless Command-Line Execution
To execute all cells and update the notebooks with rendered outputs in place:
```powershell
# Execute EDA notebook
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Driver_Activity_EDA.ipynb --ExecutePreprocessor.timeout=300

# Execute ML Models notebook
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Driver_Activity_ML_Models.ipynb --ExecutePreprocessor.timeout=600
```

---

## 4. Academic Findings & Performance Governance

| Pipeline | Model | Evaluation Method | Primary Accuracy | Macro F1 | Academic Classification |
|---|---|---|---|---|---|
| **Pipeline A** | Production RBF SVM | Chronological Test Split | 69.93% | 0.6970 | Single-Driver Session Split |
| **Pipeline A** | ExtraTrees (Candidate 1) | Forward Session ($S1 \rightarrow S2$) | 91.43% | 0.9079 | Cross-Session Generalization |
| **Pipeline A** | ExtraTrees (Candidate 1) | Reverse Session ($S2 \rightarrow S1$) | 74.71% | 0.7280 | Cross-Session Generalization |
| **Pipeline A** | ExtraTrees (Candidate 1) | Cross-Session Mean / Pooled | 83.07% / 82.49% | 0.8179 | Bidirectional Session Generalization |
| **Pipeline B** | Production ExtraTrees | Discrete Event Test Split | 88.24% | 0.8786 | Event-Level Generalization |
| **Pipeline B** | Production ExtraTrees | 3-Fold LODO (3 Drivers) | **93.06%** $\pm$ 7.87% | **0.9272** | **Genuine Multi-Driver Generalization** |

### Critical Academic Honesty Notice
- **Pipeline A (Dataset C):** Dataset C was collected from a **single driver** across two physical trips. Therefore, the 91.43% result reflects forward-session adaptation, **not unseen-driver generalization**. Achieving genuine $\ge 90\%$ unseen-driver generalization requires multi-subject naturalistic telemetry collected under the approved 8-Driver Protocol.
- **Pipeline B (Dataset B):** Dataset B was collected from **three distinct drivers**, confirming genuine multi-driver generalization under Leave-One-Driver-Out cross-validation (93.06%).

---

## 5. Production Model Cryptographic Hashes

The production model artifacts in `models/experiments/` are frozen and verified against the following SHA-256 digests:

- `models/experiments/dataset_b/dataset_b_extra_trees_exp_v1.joblib`:  
  `552048535a3fd584e4874731b8be1438dab4051d8cfa6a98fb1a7b19d9202e90`
- `models/experiments/dataset_b/dataset_b_scaler_exp_v1.joblib`:  
  `d4b3d5e3e1b739c63a9e21fac51fcedf36716143c645abe0d0e3ff99127a7576`
- `models/experiments/dataset_c/dataset_c_support_vector_machine_exp_v1.joblib`:  
  `a00180bd01d5999b1338737549d5745adb986138f813f7f371d67589d0dbaefc`
- `models/experiments/dataset_c/dataset_c_scaler_exp_v1.joblib`:  
  `d67f9ab70eb667cc57e8aeb2f6768e62529f0cb803279cc33d17f6c76555c9f5`

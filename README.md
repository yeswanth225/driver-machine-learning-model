# Intelligent Driver Activity Monitoring & Safety Agent

An AI-powered, real-time safety system designed to monitor driver behavioral patterns, detect distractions, fatigue, and anomalous driving actions, and trigger safety interventions through an autonomous safety agent loop.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Machine Learning** | Python, Scikit-learn, XGBoost, PyTorch (optional DL/CV), Pandas, NumPy |
| **Backend & APIs** | FastAPI, Uvicorn, WebSockets, SQLAlchemy, Pydantic |
| **Database** | SQLite (`aiosqlite`) for rapid local iteration; migration-ready for PostgreSQL |
| **Frontend UI** | React, TypeScript, Tailwind CSS, Vite |
| **Real-time Pipeline** | Full-duplex WebSockets for telemetry stream & low-latency alert broadcast |
| **Testing & CI** | Pytest, Pytest-Asyncio |

---

## 📁 Project Architecture & Directory Structure

```text
driver-monitoring-agent/
├── dataset/                     # Raw, processed, and external driver activity datasets
├── notebooks/                   # Jupyter notebooks for EDA, feature exploration & prototyping
├── ml/                          # Core Machine Learning pipeline
│   ├── preprocessing/           # Data cleaning, normalization, and feature extraction
│   ├── training/                # Model training scripts, hyperparameter tuning, cross-validation
│   ├── evaluation/              # Model metrics (accuracy, F1, confusion matrix, ROC-AUC)
│   ├── inference/               # Optimized runtime prediction pipelines and feature transformers
│   └── saved_models/            # Serialized model weights, scalers, and artifacts (.joblib, .pt)
├── backend/                     # FastAPI backend application
│   ├── app/
│   │   ├── api/                 # REST endpoints and WebSocket route handlers
│   │   ├── core/                # Core configuration, application settings, and security
│   │   ├── database/            # Database engine, session lifecycle, and migrations
│   │   ├── models/              # SQLAlchemy ORM database models
│   │   ├── schemas/             # Pydantic schemas for request/response validation
│   │   ├── services/            # Business logic (telemetry processing, ML inference bridge)
│   │   └── agents/              # Autonomous safety agent: decision logic, alert triggers & rules
│   └── requirements.txt         # Python dependencies for backend and ML services
├── frontend/                    # React + TypeScript single-page application with Tailwind CSS
├── reports/                     # Model evaluation reports, performance graphs, and safety summaries
├── docs/                        # Architecture documentation, API specifications, and design guidelines
├── tests/                       # Unit, integration, and end-to-end test suites
├── .env.example                 # Template for local environment variables
├── .gitignore                   # Git ignore patterns for Python, JS, models, and datasets
└── README.md                    # Project documentation and quickstart guide
```

---

## 🚀 Environment Initialization (Python & Backend)

Follow these steps to set up the development environment on your local machine:

### 1. Navigate to the project root
```bash
cd "driver-monitoring-agent"
```

### 2. Create and activate a Python virtual environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip and install backend & ML dependencies
```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

> **Note on PyTorch:** The `requirements.txt` specifies standard PyTorch. If you intend to utilize NVIDIA GPU acceleration with CUDA, install the PyTorch wheel variant tailored to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Configure environment variables
Copy `.env.example` to `.env`:
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Bash / Zsh:
cp .env.example .env
```

---

## 📦 Dataset Verification & GitHub Portability

The project is designed to be fully self-contained and reproducible when cloned on any machine without requiring external platform logins or Kaggle authentication.

### Verify Dataset Integrity
Run the built-in integrity check from the repository root:
```bash
python ml/preprocessing/verify_dataset_integrity.py
```
This script computes SHA-256 checksums, validates column structures, and confirms record counts using repository-relative paths.

### Cloning on Another Machine
When cloning this repository to a new machine:
```bash
git clone <YOUR_REPOSITORY_URL>
cd driver-monitoring-agent
python ml/preprocessing/verify_dataset_integrity.py
```
All base datasets (**Dataset A1/A2, Dataset B, and Dataset C**) are pre-packaged directly in the repository with verified checksums.

### Dataset Sources & Legal Redistribution:
- **Dataset A (Mendeley DOI: 10.17632/9vr83n7z5j.2)**: Licensed under CC BY 4.0. Author: Wawage et al. / Eishkaran Singh.
- **Dataset B (Mendeley DOI: 10.17632/jj3tw8kj6h.3)**: Licensed under CC BY 4.0. Author: Yuksel & Atmaca.
- **Dataset C (Mendeley DOI: 10.17632/5stn873wft.1)**: Licensed under CC BY 4.0. Author: Shardul Nazirkar.
- **Datasets 4–7 (Kaggle)**: Open-source driving datasets (CC BY 4.0 / CC BY-SA 4.0 / ODbL).

---

## 🗺️ Next Steps

1. **Dataset Selection & Exploration:**
   - Define data modalities (e.g., driver telematics, IMU sensor data, face/eye landmark metrics, or vision feeds).
   - Place raw data in `dataset/` and explore in `notebooks/`.
2. **Feature Engineering & ML Baseline:**
   - Implement preprocessing in `ml/preprocessing/`.
   - Train baseline models (Random Forest, XGBoost) in `ml/training/` and serialize to `ml/saved_models/`.
3. **Backend Service & Inference Pipeline:**
   - Connect FastAPI service (`backend/app/services/`) to ML inference (`ml/inference/`).
   - Define the Safety Agent decision loop in `backend/app/agents/`.
4. **Frontend Dashboard Initialization:**
   - Initialize the React + TypeScript frontend scaffold using Vite and Tailwind CSS inside `frontend/`.

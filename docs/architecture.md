# System Architecture: Intelligent Driver Activity Monitoring & Safety Agent

## 1. Overview
The **Intelligent Driver Activity Monitoring & Safety Agent** is an end-to-end AI safety system. It ingests driver sensor telemetry and/or visual facial features, classifies driver states (e.g., attentive, drowsy, texting, phone usage, distracted), evaluates risk levels in real time, and triggers automated safety interventions.

```
+--------------------+       WebSocket Telemetry       +----------------------------+
|  Driver / Vehicle  | ----------------------------->  |   FastAPI Ingestion Node   |
|   Telemetry / UI   | <-----------------------------  |   (backend/app/api)        |
+--------------------+         Alert Dispatch          +--------------+-------------+
                                                                      |
                                                                      v
                                                       +----------------------------+
                                                       |   Feature Pipeline &       |
                                                       |   ML Inference Engine      |
                                                       |   (ml/inference)           |
                                                       +--------------+-------------+
                                                                      |
                                                                      v
                                                       +----------------------------+
                                                       |   Autonomous Safety Agent  |
                                                       |   (backend/app/agents)     |
                                                       +--------------+-------------+
                                                                      |
                                         +----------------------------+---------------------------+
                                         |                                                        |
                                         v                                                        v
                          +------------------------------+                         +------------------------------+
                          |   Database State & Logs      |                         |   Frontend Operator UI       |
                          |   (SQLite / SQLAlchemy)      |                         |   (React + TS + Tailwind)    |
                          +------------------------------+                         +------------------------------+
```

---

## 2. Directory Responsibilities

### `dataset/`
- **Purpose**: Repository for dataset artifacts, including raw sensor time-series, landmark points, or behavioral annotations.
- **Rules**: Datasets are git-ignored by default to prevent bloated repository sizes.

### `notebooks/`
- **Purpose**: Sandbox environment for exploratory data analysis (EDA), data cleaning trials, feature correlation studies, and model prototyping before operationalizing code into modules.

### `ml/` (Machine Learning Engine)
- **`preprocessing/`**: Scalers, categorical encoders, windowing transforms, and missing value imputers.
- **`training/`**: Cross-validation, hyperparameter grid/random searches, model comparison (Scikit-learn vs XGBoost vs PyTorch), and training job scripts.
- **`evaluation/`**: Classification metrics, ROC-AUC, PR curves, confusion matrices, latency benchmarks, and safety threshold analysis.
- **`inference/`**: Stateless, production-grade predictor classes capable of loading models from disk and executing sub-millisecond predictions.
- **`saved_models/`**: Versioned, serialized model artifacts (`.joblib`, `.pt`, `.onnx`) accompanied by feature metadata.

### `backend/` (FastAPI Core)
- **`app/api/`**: REST endpoints (CRUD for sessions, driver logs) and WebSocket connections for high-frequency telemetry and alerting.
- **`app/core/`**: Configuration management via Pydantic Settings, global error handlers, and security middleware.
- **`app/database/`**: Async engine configuration (`aiosqlite`), database session dependency injection, and schema management.
- **`app/models/`**: Declarative ORM models representing drivers, driving sessions, telemetry events, and safety alerts.
- **`app/schemas/`**: Strongly typed Pydantic models for incoming data payload validation and outgoing response serialization.
- **`app/services/`**: Integration layer orchestrating data storage, feature transformation, and calling the ML inference engine.
- **`app/agents/`**: Autonomous Safety Agent logic:
  - Temporal smoothing (e.g., 3 consecutive drowsy frames before alerting).
  - Risk categorization (Low, Medium, Critical).
  - Escalation protocol (Audio chime, visual popup, emergency contact notification).

### `frontend/`
- **Purpose**: React + TypeScript client with Tailwind CSS.
- **Responsibilities**:
  - Real-time driver telemetry visualization (gauges, activity status, time in state).
  - Live alert panel receiving updates via WebSockets.
  - Driver history and session playback.

### `reports/`
- **Purpose**: Output directory for generated model evaluation sheets, audit logs, confusion matrix plots, and safety compliance reports.

### `docs/`
- **Purpose**: Technical documentation, system architecture specifications, API contracts, and deployment runbooks.

### `tests/`
- **Purpose**: Pytest test suites covering ML pipeline transforms, API endpoints, agent safety state transitions, and database operations.

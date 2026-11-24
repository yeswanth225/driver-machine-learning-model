"""
Replay Studio Service: Historical Driving Session Playback & Telemetry Streaming.
================================================================================
Provides pre-indexed recorded driving episodes from the audited test evaluation set.
Supplies actual kinematic sensor streams and pre-calculated model predictions
without exposing evaluation-only ground-truth boundaries to the inference path.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from ml.inference.dataset_b_classifier import (
    DatasetBManeuverClassifier,
    QUARANTINED_FEATURES,
    LABEL_MAPPING,
)
from backend.app.schemas.telemetry import (
    ReplaySessionSummary,
    ReplayFrame,
    ReplaySessionDetail,
    TelemetryPredictionRequest,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET_B_TEST_PATH = ROOT_DIR / "dataset" / "processed" / "dataset_b_test.csv"


class ReplayService:
    """Singleton service for loading and serving recorded driving sessions."""

    _instance: Optional["ReplayService"] = None

    def __init__(self):
        self._classifier: Optional[DatasetBManeuverClassifier] = None
        self._sessions_cache: Dict[str, ReplaySessionDetail] = {}
        self._summaries_cache: Optional[List[ReplaySessionSummary]] = None

    @classmethod
    def get_instance(cls) -> "ReplayService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_classifier(self) -> DatasetBManeuverClassifier:
        if self._classifier is None:
            self._classifier = DatasetBManeuverClassifier()
        return self._classifier

    def _load_data(self) -> pd.DataFrame:
        if not DATASET_B_TEST_PATH.is_file():
            raise FileNotFoundError(f"Dataset B test file not found at {DATASET_B_TEST_PATH}")
        return pd.read_csv(DATASET_B_TEST_PATH)

    def _build_sessions(self):
        """Construct available replay sessions from test partition."""
        if self._summaries_cache is not None:
            return

        df = self._load_data()
        clf = self._get_classifier()

        # Clean anti-causal features
        clean_df = df.drop(columns=[c for c in QUARANTINED_FEATURES if c in df.columns])
        preds = clf.predict_with_labels(clean_df)

        # Map predictions to df
        df["pred_class_id"] = [p["class_id"] for p in preds]
        df["pred_confidence"] = [p["confidence"] for p in preds]
        df["pred_label_abbr"] = [p["label_abbr"] for p in preds]
        df["pred_label_name"] = [p["label_name"] for p in preds]

        numeric_feature_cols = [c for c in clf.feature_names if c in clean_df.columns]

        sessions_def = [
            ("session-drv-1", "Driver 1 — Urban Route (Curated Testbed)", "Driver-1",
             "Real 10 Hz IMU telemetry captured during urban driving tasks with sharp turns and deceleration."),
            ("session-drv-2", "Driver 2 — Highway Transition & Maneuvers", "Driver-2",
             "Extended driving sequence highlighting highway merges, sudden braking events, and lateral corrections."),
            ("session-drv-3", "Driver 3 — Aggressive Acceleration & Braking", "Driver-3",
             "High-dynamic vehicle kinematic session with repeated sudden accelerations and sharp braking."),
            ("session-multi-hazard", "Complete 175-Frame Test Benchmark", "Pooled",
             "Full sequential replay of all 175 test evaluation frames spanning all four active maneuver classes."),
        ]

        summaries: List[ReplaySessionSummary] = []

        for s_id, title, driver_filter, desc in sessions_def:
            if driver_filter == "Pooled":
                sub_df = df.copy()
            else:
                sub_df = df[df["DriverID"] == driver_filter].copy()

            if len(sub_df) == 0:
                continue

            frames: List[ReplayFrame] = []
            event_counts: Dict[str, int] = {}

            # Construct frames
            current_speed = 65.0  # km/h nominal
            t = 0.0

            for idx, (_, row) in enumerate(sub_df.iterrows()):
                acc_x = float(row.get("AccX", 0.0))
                acc_y = float(row.get("AccY", 0.0))
                acc_z = float(row.get("AccZ", 0.0))
                gyro_x = float(row.get("GyroX", 0.0))
                gyro_y = float(row.get("GyroY", 0.0))
                gyro_z = float(row.get("GyroZ", 0.0))

                # Speed representation with realistic physical modulation
                current_speed = max(15.0, min(130.0, current_speed + acc_x * 0.4))
                # Steering representation from lateral and yaw dynamics
                steering_angle = float(np.clip(gyro_z * 45.0 + acc_y * 15.0, -35.0, 35.0))

                pred_class_id = int(row["pred_class_id"])
                pred_conf = float(row["pred_confidence"])
                pred_abbr = str(row["pred_label_abbr"])
                gt_label = str(row.get("label_abbr", "None"))

                event_counts[pred_abbr] = event_counts.get(pred_abbr, 0) + 1

                # Pipeline A style inference heuristic: high gyro/acc std -> Aggressive, else Normal
                is_aggressive = abs(acc_x) > 0.35 or abs(acc_y) > 0.35
                style_pred = 1 if is_aggressive else 0
                style_conf = 0.88 if is_aggressive else 0.94

                feat_dict = {col: float(row[col]) for col in numeric_feature_cols if col in row}

                telem_req = TelemetryPredictionRequest(
                    timestamp=round(t, 2),
                    style_prediction=style_pred,
                    style_confidence=style_conf,
                    maneuver_prediction=pred_class_id,
                    maneuver_confidence=round(pred_conf, 4),
                    pipeline_a_health="HEALTHY",
                    pipeline_b_health="HEALTHY",
                )

                frames.append(
                    ReplayFrame(
                        frame_index=idx,
                        timestamp=round(t, 2),
                        acc_x=round(acc_x, 4),
                        acc_y=round(acc_y, 4),
                        acc_z=round(acc_z, 4),
                        gyro_x=round(gyro_x, 4),
                        gyro_y=round(gyro_y, 4),
                        gyro_z=round(gyro_z, 4),
                        speed=round(current_speed, 1),
                        steering_angle=round(steering_angle, 1),
                        ground_truth_label=gt_label,
                        telemetry_request=telem_req,
                        features_b=feat_dict,
                    )
                )
                t += 0.2  # 5 Hz nominal replay timestep

            duration = round(t, 2)
            total_frames = len(frames)

            detail = ReplaySessionDetail(
                id=s_id,
                title=title,
                driver_id=driver_filter,
                duration_seconds=duration,
                total_frames=total_frames,
                event_counts=event_counts,
                description=desc,
                frames=frames,
            )
            self._sessions_cache[s_id] = detail

            summaries.append(
                ReplaySessionSummary(
                    id=s_id,
                    title=title,
                    driver_id=driver_filter,
                    duration_seconds=duration,
                    total_frames=total_frames,
                    event_counts=event_counts,
                    description=desc,
                )
            )

        self._summaries_cache = summaries

    def get_sessions(self) -> List[ReplaySessionSummary]:
        self._build_sessions()
        return self._summaries_cache or []

    def get_session_detail(self, session_id: str) -> Optional[ReplaySessionDetail]:
        self._build_sessions()
        return self._sessions_cache.get(session_id)

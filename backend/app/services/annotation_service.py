"""
Ground-Truth Annotation Management Service
==========================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: backend/app/services/annotation_service.py

Implements storage, validation, and retrieval of independent human observer annotations.
Supported ground-truth labels:
- NORMAL
- AGGRESSIVE
- SUDDEN_ACCELERATION
- SUDDEN_DECELERATION
- SUDDEN_RIGHT_TURN
- SUDDEN_LEFT_TURN
- UNCERTAIN
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANNOTATIONS_DIR = REPO_ROOT / "dataset" / "annotations"

VALID_LABELS = {
    "NORMAL",
    "AGGRESSIVE",
    "SUDDEN_ACCELERATION",
    "SUDDEN_DECELERATION",
    "SUDDEN_RIGHT_TURN",
    "SUDDEN_LEFT_TURN",
    "UNCERTAIN"
}


class AnnotationRecord(BaseModel):
    annotation_id: Optional[str] = None
    session_id: str
    driver_id: str
    start_timestamp: float = Field(..., description="Start timestamp in seconds from session start")
    end_timestamp: float = Field(..., description="End timestamp in seconds from session start")
    label: str = Field(..., description="Maneuver / Style label")
    annotator_id: str = Field(..., description="Anonymized observer token, e.g. OBS_01")
    evidence: str = Field(default="", description="Visual or telemetry evidence justification")
    verification_status: str = Field(default="VERIFIED", description="VERIFIED, UNVERIFIED, or CONTESTED")
    created_at: Optional[str] = None


class AnnotationService:
    """Manages ground-truth annotation storage and retrieval."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "AnnotationService":
        if cls._instance is None:
            cls._instance = AnnotationService()
        return cls._instance

    def __init__(self):
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_session_annotations_path(self, session_id: str) -> Path:
        return ANNOTATIONS_DIR / f"{session_id}_annotations.json"

    def create_annotation(self, record: AnnotationRecord) -> Dict[str, Any]:
        """Creates and persists an independent human ground-truth annotation."""
        clean_label = record.label.strip().upper()
        if clean_label not in VALID_LABELS:
            raise ValueError(f"Invalid label '{record.label}'. Must be one of: {sorted(list(VALID_LABELS))}")

        if record.end_timestamp <= record.start_timestamp:
            raise ValueError(f"end_timestamp ({record.end_timestamp}) must be strictly greater than start_timestamp ({record.start_timestamp}).")

        path = self._get_session_annotations_path(record.session_id)
        existing = []
        if path.exists():
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        ann_id = record.annotation_id or f"ann_{int(time.time() * 1000)}_{len(existing) + 1}"
        record.annotation_id = ann_id
        record.label = clean_label
        record.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        existing.append(record.model_dump())

        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

        return record.model_dump()

    def get_session_annotations(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all annotations associated with a specific session."""
        path = self._get_session_annotations_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def delete_annotation(self, session_id: str, annotation_id: str) -> bool:
        """Removes a specific annotation by ID."""
        path = self._get_session_annotations_path(session_id)
        if not path.exists():
            return False

        with open(path, "r") as f:
            existing = json.load(f)

        filtered = [a for a in existing if a.get("annotation_id") != annotation_id]
        if len(filtered) == len(existing):
            return False

        with open(path, "w") as f:
            json.dump(filtered, f, indent=2)

        return True

"""
Data Ingestion and Raw Storage Infrastructure for Eight-Driver Physical Telemetry
=================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: backend/app/services/sensor_storage_service.py

Responsibilities:
- Ingestion of raw CSV recordings and metadata.
- Cryptographic SHA-256 checksum generation and verification.
- Detection of duplicate recordings.
- Enforcing immutable storage for raw recordings.
- Development-driver pool segregation (DRV_001 - DRV_006).
- Strict rejection of lockbox drivers (DRV_007, DRV_008) through developmental endpoints.
- Derived data separation.
"""

import os
import re
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASE_RAW_DIR = REPO_ROOT / "dataset" / "raw" / "eight_driver_campaign"
BASE_DERIVED_DIR = REPO_ROOT / "dataset" / "derived" / "eight_driver_campaign"
METADATA_DIR = REPO_ROOT / "dataset" / "metadata"
ANNOTATIONS_DIR = REPO_ROOT / "dataset" / "annotations"

DEVELOPMENT_DRIVERS = {"DRV_001", "DRV_002", "DRV_003", "DRV_004", "DRV_005", "DRV_006"}
RESERVED_LOCKBOX_DRIVERS = {"DRV_007", "DRV_008"}

# Strict pattern for session identifiers: alphanumeric, hyphens, and underscores only
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

MANDATORY_CSV_COLUMNS = [
    "timestamp", "timestamp_epoch_ms",
    "Acc X", "Acc Y", "Acc Z",
    "gyro_x", "gyro_y", "gyro_z",
    "speed_mps", "heading_deg",
    "latitude", "longitude", "altitude", "hdop",
    "segment_id"
]


class SensorStorageService:
    """Manages raw recording storage, integrity verification, and directory layout."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "SensorStorageService":
        if cls._instance is None:
            cls._instance = SensorStorageService()
        return cls._instance

    def __init__(self):
        BASE_RAW_DIR.mkdir(parents=True, exist_ok=True)
        BASE_DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

        for drv in DEVELOPMENT_DRIVERS:
            (BASE_RAW_DIR / drv).mkdir(exist_ok=True)
            (BASE_DERIVED_DIR / drv).mkdir(exist_ok=True)

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()

    def validate_driver_id(self, driver_id: str) -> None:
        """Enforces development driver segregation and rejects lockbox drivers."""
        clean_id = driver_id.strip().upper()
        if clean_id in RESERVED_LOCKBOX_DRIVERS:
            raise PermissionError(
                f"ACCESS DENIED: Driver '{clean_id}' belongs to the sequestered Blind Lockbox. "
                "Lockbox data cannot be ingested through development endpoints."
            )
        if clean_id not in DEVELOPMENT_DRIVERS:
            raise ValueError(
                f"Invalid driver token '{driver_id}'. Must be an approved development driver (DRV_001 to DRV_006)."
            )

    def validate_session_id(self, session_id: str) -> None:
        """Enforces path traversal safety and syntax on session identifiers."""
        clean_sess = session_id.strip()
        if not clean_sess or len(clean_sess) > 128:
            raise ValueError("Session identifier must be between 1 and 128 characters.")
        if ".." in clean_sess or "/" in clean_sess or "\\" in clean_sess:
            raise PermissionError("Path traversal violation detected in session identifier.")
        if not SESSION_ID_PATTERN.match(clean_sess):
            raise ValueError(f"Invalid characters in session identifier '{session_id}'. Allowed: alphanumeric, hyphens, and underscores.")

    def validate_csv_schema(self, content_str: str) -> Tuple[bool, str, pd.DataFrame]:
        """Validates CSV format, required columns, and non-emptiness."""
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(content_str))
        except Exception as e:
            return False, f"Malformed CSV content: {str(e)}", pd.DataFrame()

        if len(df) == 0:
            return False, "CSV contains 0 data rows.", df

        missing_cols = [c for c in MANDATORY_CSV_COLUMNS if c not in df.columns]
        if missing_cols:
            return False, f"Missing required columns in CSV header: {missing_cols}", df

        return True, "Valid CSV schema.", df

    def check_duplicate_recording(self, driver_id: str, sha256_hash: str) -> Optional[str]:
        """Checks if a recording with identical SHA-256 hash already exists."""
        drv_dir = BASE_RAW_DIR / driver_id
        if not drv_dir.exists():
            return None

        for sess_dir in drv_dir.iterdir():
            if sess_dir.is_dir():
                hash_file = sess_dir / "sha256_checksum.txt"
                if hash_file.exists():
                    existing_hash = hash_file.read_text().strip()
                    if existing_hash.lower() == sha256_hash.lower():
                        return sess_dir.name
        return None

    def store_raw_recording(
        self,
        driver_id: str,
        session_id: str,
        csv_bytes: bytes,
        client_sha256: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Stores raw telemetry CSV immutably, writes metadata and checksum.
        """
        self.validate_driver_id(driver_id)
        self.validate_session_id(session_id)
        calculated_sha256 = self.compute_sha256(csv_bytes)

        if client_sha256 and client_sha256.strip().lower() != calculated_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch! Client provided '{client_sha256}', server calculated '{calculated_sha256}'."
            )

        # Check duplicate
        existing_sess = self.check_duplicate_recording(driver_id, calculated_sha256)
        if existing_sess:
            return {
                "session_id": existing_sess,
                "driver_id": driver_id,
                "is_duplicate": True,
                "sha256": calculated_sha256,
                "message": f"Duplicate recording detected. Identical telemetry already stored in session '{existing_sess}'."
            }

        # Validate CSV structure
        content_str = csv_bytes.decode("utf-8", errors="replace")
        valid, msg, df = self.validate_csv_schema(content_str)
        if not valid:
            raise ValueError(f"CSV Validation Failed: {msg}")

        # Create session storage directory
        sess_dir = BASE_RAW_DIR / driver_id / session_id
        sess_dir.mkdir(parents=True, exist_ok=True)

        # Write immutable raw CSV
        csv_path = sess_dir / f"raw_telemetry_{session_id}.csv"
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)

        # Write SHA-256 checksum file
        hash_path = sess_dir / "sha256_checksum.txt"
        hash_path.write_text(calculated_sha256)

        # Compute basic session statistics
        sample_count = len(df)
        duration_sec = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if sample_count > 1 else 0.0

        # Build and write metadata manifest
        meta = {
            "session_id": session_id,
            "driver_id": driver_id,
            "sha256": calculated_sha256,
            "file_name": csv_path.name,
            "file_size_bytes": len(csv_bytes),
            "sample_count": sample_count,
            "duration_seconds": duration_sec,
            "sampling_frequency_nominal_hz": 20.0,
            "columns": list(df.columns),
            "custom_metadata": additional_metadata or {}
        }
        meta_path = sess_dir / "session_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "session_id": session_id,
            "driver_id": driver_id,
            "file_name": csv_path.name,
            "file_size_bytes": len(csv_bytes),
            "sample_count": sample_count,
            "duration_seconds": duration_sec,
            "sha256": calculated_sha256,
            "is_duplicate": False,
            "message": "Raw telemetry recording stored immutably."
        }

    def get_session_telemetry_df(self, driver_id: str, session_id: str) -> Optional[pd.DataFrame]:
        """Loads stored raw telemetry DataFrame for a session."""
        self.validate_driver_id(driver_id)
        self.validate_session_id(session_id)
        csv_path = BASE_RAW_DIR / driver_id / session_id / f"raw_telemetry_{session_id}.csv"
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)

    def list_all_sessions(self) -> List[Dict[str, Any]]:
        """Lists all ingested physical recording sessions."""
        sessions = []
        for drv_dir in BASE_RAW_DIR.iterdir():
            if drv_dir.is_dir() and drv_dir.name in DEVELOPMENT_DRIVERS:
                for sess_dir in drv_dir.iterdir():
                    if sess_dir.is_dir():
                        meta_file = sess_dir / "session_metadata.json"
                        if meta_file.exists():
                            with open(meta_file, "r") as f:
                                meta = json.load(f)
                            sessions.append(meta)
        return sessions

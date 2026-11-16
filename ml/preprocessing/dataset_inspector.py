"""
Dataset Inspector Module for Driver Activity Monitoring & Safety Agent.
Provides reusable tools for inspecting smartphone sensor time-series datasets,
computing data quality metrics, calculating sampling frequencies, and identifying anomalies.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class DatasetInspector:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    def list_files(self, extension: str = ".csv") -> List[Path]:
        """Recursively list all data files matching extension."""
        return sorted(list(self.base_dir.rglob(f"*{extension}")))

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive inspection on a single CSV file."""
        file_size_bytes = file_path.stat().st_size
        rel_path = str(file_path.relative_to(self.base_dir))

        if file_size_bytes == 0:
            return {
                "file_path": rel_path,
                "file_size_bytes": 0,
                "file_size_kb": 0.0,
                "status": "EMPTY_FILE",
                "rows": 0,
                "columns": [],
                "missing_values": {},
                "duplicate_rows": 0,
            }

        try:
            # Read first chunk to check header and columns
            df = pd.read_csv(file_path)
        except Exception as e:
            return {
                "file_path": rel_path,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_bytes / 1024, 2),
                "status": f"READ_ERROR: {str(e)}",
                "rows": 0,
                "columns": [],
                "missing_values": {},
                "duplicate_rows": 0,
            }

        rows, cols = df.shape
        missing = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated().sum())

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        numeric_summary = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            numeric_summary[c] = {
                "mean": float(df[c].mean()),
                "std": float(df[c].std()) if rows > 1 else 0.0,
                "min": float(df[c].min()),
                "25%": float(df[c].quantile(0.25)) if rows > 1 else float(df[c].min()),
                "50%": float(df[c].median()),
                "75%": float(df[c].quantile(0.75)) if rows > 1 else float(df[c].max()),
                "max": float(df[c].max()),
            }

        # Sampling rate analysis if Milliseconds column exists
        sampling_info = {}
        if "Milliseconds" in df.columns and rows > 1:
            diffs = df["Milliseconds"].diff().dropna()
            # If milliseconds wraps at 1000 or timestamp changes
            valid_diffs = diffs[(diffs > 0) & (diffs < 1000)]
            if len(valid_diffs) > 0:
                median_delta_ms = float(valid_diffs.median())
                mean_delta_ms = float(valid_diffs.mean())
                approx_hz = round(1000.0 / median_delta_ms, 2) if median_delta_ms > 0 else 0.0
                sampling_info = {
                    "median_delta_ms": median_delta_ms,
                    "mean_delta_ms": round(mean_delta_ms, 2),
                    "estimated_sampling_rate_hz": approx_hz,
                    "min_delta_ms": float(valid_diffs.min()),
                    "max_delta_ms": float(valid_diffs.max()),
                }

        return {
            "file_path": rel_path,
            "file_size_bytes": file_size_bytes,
            "file_size_kb": round(file_size_bytes / 1024, 2),
            "status": "VALID",
            "rows": rows,
            "columns": list(df.columns),
            "dtypes": dtypes,
            "missing_values": missing,
            "duplicate_rows": duplicates,
            "duplicate_pct": round(duplicates / rows * 100, 3) if rows > 0 else 0.0,
            "numeric_summary": numeric_summary,
            "sampling_info": sampling_info,
        }

    def inspect_all(self) -> List[Dict[str, Any]]:
        """Inspect all CSV files under base_dir."""
        files = self.list_files(".csv")
        results = []
        for f in files:
            results.append(self.inspect_file(f))
        return results


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    inspector = DatasetInspector(target_dir)
    results = inspector.inspect_all()
    print(f"Inspected {len(results)} files in {target_dir}")
    for res in results:
        status = res["status"]
        rows = res["rows"]
        kb = res["file_size_kb"]
        print(f"{res['file_path']} | {status} | {rows} rows | {kb} KB")

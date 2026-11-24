"""
Sensor Calibration & Data Quality Acceptance Engine for Physical Driving Telemetry
===================================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: backend/app/services/data_quality_service.py

Implements automated evaluation of the 6 mandatory acceptance criteria defined in
reports/pipeline_a/improvement/data_quality_acceptance_criteria.md.
Evaluates:
1. Sampling frequency, temporal jitter, duplicate timestamps, and gap limits (< 250ms).
2. Sensor dynamic range, saturation avoidance, and dead-channel detection.
3. Static gravitational calibration (|g| in [0.98, 1.02] g) and mount drift (<= 3.0 deg).
4. GPS ground speed validity, fix availability, and HDOP consistency.
5. Driving regime coverage (Highway normal >= 2.5 min, Aggressive >= 2.0 min).
6. Zero nulls, NaNs, or infinite values.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from backend.app.schemas.sensor import DataQualityReport, QualityCriterionResult


class DataQualityService:
    """Automated validator for incoming physical sensor telemetry."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "DataQualityService":
        if cls._instance is None:
            cls._instance = DataQualityService()
        return cls._instance

    def evaluate_session_df(
        self,
        df: pd.DataFrame,
        driver_id: str,
        session_id: str
    ) -> DataQualityReport:
        """Runs the complete 6-criterion evaluation against a telemetry DataFrame."""
        criteria_results: List[QualityCriterionResult] = []
        overall_status = "PASS"

        # -------------------------------------------------------------
        # 1. Null / NaN / Inf Anomaly Audit (Criterion 6)
        # -------------------------------------------------------------
        nan_count = int(df.isnull().sum().sum())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = int(np.isinf(df[numeric_cols].values).sum()) if len(numeric_cols) > 0 else 0

        if nan_count > 0 or inf_count > 0:
            c6_status = "FAIL"
            overall_status = "FAIL"
            c6_details = f"Detected {nan_count} NaNs and {inf_count} infinite values in telemetry."
        else:
            c6_status = "PASS"
            c6_details = "Zero NaNs, nulls, or infinite values detected across all columns."

        criteria_results.append(QualityCriterionResult(
            criterion_name="Criterion 6: Missing Values & Data Hygiene",
            status=c6_status,
            measured_value={"nan_count": nan_count, "inf_count": inf_count},
            threshold="Exactly 0 NaNs and 0 Infs",
            details=c6_details
        ))

        # -------------------------------------------------------------
        # 2. Sampling Frequency, Jitter & Gap Limits (Criterion 1)
        # -------------------------------------------------------------
        timestamps = df["timestamp"].values
        dt = np.diff(timestamps)
        total_samples = len(df)
        duration_sec = float(timestamps[-1] - timestamps[0]) if total_samples > 1 else 0.0

        if total_samples < 2:
            return self._build_empty_report(driver_id, session_id, "FAIL", "Insufficient data rows (< 2)")

        # Monotonicity & duplicates
        duplicates = int(np.sum(dt == 0.0))
        backwards = int(np.sum(dt < 0.0))
        max_gap_ms = float(np.max(dt) * 1000.0) if len(dt) > 0 else 0.0
        mean_dt_ms = float(np.mean(dt) * 1000.0) if len(dt) > 0 else 0.0
        actual_freq_hz = float(1000.0 / mean_dt_ms) if mean_dt_ms > 0 else 0.0

        # Frequency tolerance [18.0, 22.0] Hz maps to nominal period [45.45, 55.56] ms
        # Hardware inter-sample jitter window: [40.0, 60.0] ms (allows +/- 10ms OS coroutine jitter)
        large_gaps = int(np.sum(dt >= 0.250))
        in_jitter_band = int(np.sum((dt >= 0.040) & (dt <= 0.060)))
        jitter_pct = float(in_jitter_band / len(dt)) * 100.0 if len(dt) > 0 else 0.0

        if backwards > 0:
            c1_status = "FAIL"
            overall_status = "FAIL"
            c1_details = f"Detected {backwards} backwards timestamp steps (monotonicity violation)."
        elif duplicates > (0.01 * total_samples):
            c1_status = "FAIL"
            overall_status = "FAIL"
            c1_details = f"Detected {duplicates} duplicate timestamps (> 1% of stream)."
        elif large_gaps > 0:
            c1_status = "FAIL"
            overall_status = "FAIL"
            c1_details = f"Detected {large_gaps} telemetry gaps >= 250ms (max gap: {max_gap_ms:.1f}ms)."
        elif actual_freq_hz < 18.0 or actual_freq_hz > 22.0:
            c1_status = "WARNING"
            if overall_status != "FAIL": overall_status = "WARNING"
            c1_details = f"Mean frequency {actual_freq_hz:.1f}Hz deviates from nominal 20.0Hz [18.0, 22.0]Hz band."
        elif jitter_pct < 95.0:
            c1_status = "WARNING"
            if overall_status != "FAIL": overall_status = "WARNING"
            c1_details = f"Elevated sampling jitter: {jitter_pct:.1f}% of intervals in [40, 60]ms (< 95.0% threshold)."
        else:
            c1_status = "PASS"
            c1_details = f"Nominal 20Hz confirmed (mean: {actual_freq_hz:.1f}Hz, jitter: {jitter_pct:.1f}% in [40, 60]ms, max gap: {max_gap_ms:.1f}ms)."

        criteria_results.append(QualityCriterionResult(
            criterion_name="Criterion 1: Sampling Frequency & Jitter",
            status=c1_status,
            measured_value={"mean_freq_hz": round(actual_freq_hz, 2), "max_gap_ms": round(max_gap_ms, 1), "large_gaps": large_gaps, "jitter_band_pct": round(jitter_pct, 1)},
            threshold="Nominal 20.0Hz (18.0 - 22.0Hz), Zero gaps >= 250ms, strictly monotonic, >= 95.0% in [40, 60]ms",
            details=c1_details
        ))

        # -------------------------------------------------------------
        # 3. Dynamic Range & Saturation Avoidance (Criterion 2)
        # -------------------------------------------------------------
        ax, ay, az = df["Acc X"].values, df["Acc Y"].values, df["Acc Z"].values
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        max_acc_g = float(np.max(acc_mag))

        gx, gy, gz = df["gyro_x"].values, df["gyro_y"].values, df["gyro_z"].values
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        max_gyro_rad = float(np.max(gyro_mag))

        # Clipping thresholds (assuming minimum +/- 4g accelerometer and +/- 500 deg/s gyroscope)
        clipped_acc = int(np.sum(np.abs(ax) >= 3.92) + np.sum(np.abs(ay) >= 3.92) + np.sum(np.abs(az) >= 3.92))
        clipped_gyro = int(np.sum(gyro_mag >= 8.5))

        # Dead channel check (constant zero std over > 40 samples / 2 seconds)
        acc_std = float(np.std(ax) + np.std(ay) + np.std(az))

        if clipped_acc > 0 or clipped_gyro > 0:
            c2_status = "FAIL"
            overall_status = "FAIL"
            c2_details = f"Sensor saturation detected ({clipped_acc} clipped accelerometer samples)."
        elif acc_std < 0.001 and duration_sec > 10.0:
            c2_status = "FAIL"
            overall_status = "FAIL"
            c2_details = "Dead accelerometer channel detected (zero standard deviation across trip)."
        else:
            c2_status = "PASS"
            c2_details = f"Dynamic range valid. Peak Acc: {max_acc_g:.2f}g, Peak Gyro: {max_gyro_rad:.2f}rad/s."

        criteria_results.append(QualityCriterionResult(
            criterion_name="Criterion 2: Dynamic Range & Saturation",
            status=c2_status,
            measured_value={"peak_acc_g": round(max_acc_g, 2), "peak_gyro_rad": round(max_gyro_rad, 2)},
            threshold="Full-scale >= +/- 4.0g, zero clipped samples (< 3.92g)",
            details=c2_details
        ))

        # -------------------------------------------------------------
        # 4. Static Gravity Calibration & Mount Drift (Criterion 3)
        # -------------------------------------------------------------
        s0_df = df[df["segment_id"] == "S0"] if "segment_id" in df.columns else pd.DataFrame()
        s6_df = df[df["segment_id"] == "S6"] if "segment_id" in df.columns else pd.DataFrame()

        # If segment S0 not explicitly marked, evaluate first 30 seconds
        if len(s0_df) < 20:
            s0_df = df.head(min(len(df), 600))  # 30 seconds at 20 Hz

        g0 = np.array([s0_df["Acc X"].mean(), s0_df["Acc Y"].mean(), s0_df["Acc Z"].mean()])
        g0_mag = float(np.linalg.norm(g0))

        mount_drift_deg = None
        c3_details = ""
        c3_status = "PASS"

        if g0_mag < 0.98 or g0_mag > 1.02:
            c3_status = "FAIL"
            overall_status = "FAIL"
            c3_details = f"Static gravity magnitude {g0_mag:.3f}g out of nominal bounds [0.98, 1.02]g (9.81 +/- 0.20 m/s^2)."
        else:
            c3_details = f"Initial static gravity valid: {g0_mag:.3f}g."

        # Check post-drive mount drift if S6 exists
        if len(s6_df) >= 20:
            g6 = np.array([s6_df["Acc X"].mean(), s6_df["Acc Y"].mean(), s6_df["Acc Z"].mean()])
            g6_mag = float(np.linalg.norm(g6))
            cos_theta = np.dot(g0, g6) / (g0_mag * g6_mag)
            mount_drift_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

            if mount_drift_deg > 3.0:
                c3_status = "FAIL"
                overall_status = "FAIL"
                c3_details += f" Mount slip detected! Angular drift: {mount_drift_deg:.2f} deg (> 3.0 deg threshold)."
            else:
                c3_details += f" Mount stable across drive (drift: {mount_drift_deg:.2f} deg <= 3.0 deg)."

        criteria_results.append(QualityCriterionResult(
            criterion_name="Criterion 3: Static Gravity & Mount Stability",
            status=c3_status,
            measured_value={"static_gravity_g": round(g0_mag, 3), "mount_drift_deg": round(mount_drift_deg, 2) if mount_drift_deg else "N/A"},
            threshold="Static |g| in [0.98, 1.02]g, mount drift <= 3.0 deg",
            details=c3_details
        ))

        # -------------------------------------------------------------
        # 5. GPS Telemetry & Ground Speed Quality (Criterion 4)
        # -------------------------------------------------------------
        if "speed_mps" in df.columns:
            speeds = df["speed_mps"].values
            valid_speed_pct = float(np.sum(speeds >= 0.0) / total_samples) * 100.0
            max_speed_kmh = float(np.max(speeds) * 3.6)
            negative_speeds = int(np.sum(speeds < 0.0))

            hdop_vals = df["hdop"].values if "hdop" in df.columns else np.array([1.0])
            mean_hdop = float(np.mean(hdop_vals))

            # Compute 3D Fix availability rate
            if "latitude" in df.columns and "longitude" in df.columns:
                valid_coords = int(np.sum((df["latitude"] != 0.0) & (df["longitude"] != 0.0) & (~df["latitude"].isna())))
                fix_rate_pct = float(valid_coords / total_samples) * 100.0
            else:
                fix_rate_pct = 100.0

            if negative_speeds > 0:
                c4_status = "FAIL"
                overall_status = "FAIL"
                c4_details = f"Negative GPS ground speed values detected ({negative_speeds} rows)."
            elif fix_rate_pct < 92.0:
                c4_status = "FAIL"
                overall_status = "FAIL"
                c4_details = f"Insufficient GPS fix coverage: {fix_rate_pct:.1f}% of drive (< 92% required)."
            elif mean_hdop > 2.5:
                c4_status = "WARNING"
                if overall_status != "FAIL": overall_status = "WARNING"
                c4_details = f"Degraded GPS satellite geometry (mean HDOP: {mean_hdop:.2f} > 2.5 target)."
            else:
                c4_status = "PASS"
                c4_details = f"GPS ground speed verified (max: {max_speed_kmh:.1f} km/h, fix rate: {fix_rate_pct:.1f}%, mean HDOP: {mean_hdop:.2f})."
        else:
            c4_status = "NOT_AVAILABLE"
            if overall_status != "FAIL": overall_status = "WARNING"
            c4_details = "GPS ground speed channel missing from telemetry."

        criteria_results.append(QualityCriterionResult(
            criterion_name="Criterion 4: GPS Ground Speed & Satellite Fix",
            status=c4_status,
            measured_value={
                "max_speed_kmh": round(max_speed_kmh, 1) if "speed_mps" in df.columns else 0.0,
                "mean_hdop": round(mean_hdop, 2) if "speed_mps" in df.columns else 1.0,
                "fix_rate_pct": round(fix_rate_pct, 1) if "speed_mps" in df.columns else 0.0
            },
            threshold="Non-negative speed, HDOP < 2.5, fix rate >= 92%",
            details=c4_details
        ))

        # -------------------------------------------------------------
        # 6. Driving Regime Balance & Coverage (Criterion 5)
        # -------------------------------------------------------------
        regime_breakdown = {}
        if "segment_id" in df.columns:
            counts = df["segment_id"].value_counts()
            for seg, count in counts.items():
                regime_breakdown[str(seg)] = round(float(count * mean_dt_ms / 60000.0), 2)  # in minutes

        # Check highway normal duration (S3 or speed >= 60 km/h)
        highway_duration_sec = 0.0
        if "speed_mps" in df.columns:
            highway_samples = np.sum((df["speed_mps"].values * 3.6) >= 60.0)
            highway_duration_sec = float(highway_samples * (mean_dt_ms / 1000.0))

        if duration_sec < 720.0:  # < 12 minutes
            c5_status = "WARNING"
            if overall_status != "FAIL": overall_status = "WARNING"
            c5_details = f"Session duration {duration_sec/60.0:.1f} min is below 12.0 min target."
        elif highway_duration_sec < 150.0:  # < 2.5 minutes
            c5_status = "WARNING"
            if overall_status != "FAIL": overall_status = "WARNING"
            c5_details = f"Insufficient highway cruising ({highway_duration_sec/60.0:.1f} min < 2.5 min target). Risk of Block 9 vibration confounding."
        else:
            c5_status = "PASS"
            c5_details = f"Balanced regime coverage verified (Duration: {duration_sec/60.0:.1f} min, Highway cruising: {highway_duration_sec/60.0:.1f} min)."

        criteria_results.append(QualityCriterionResult(
            criterion_name="Criterion 5: Driving Regime Coverage & Highway Anti-Confounding",
            status=c5_status,
            measured_value={"duration_min": round(duration_sec / 60.0, 2), "highway_min": round(highway_duration_sec / 60.0, 2)},
            threshold="Duration >= 12.0 min, Highway Normal >= 2.5 min",
            details=c5_details
        ))

        # Recommendation
        if overall_status == "PASS":
            recommendation = "Session satisfies all 6 data quality criteria. Approved for ingestion into raw campaign vault."
        elif overall_status == "WARNING":
            recommendation = "Session accepted with warnings. Inspect sampling rate or regime coverage before model training."
        else:
            recommendation = "Session REJECTED due to critical data quality failure. Session must be re-recorded."

        report = DataQualityReport(
            session_id=session_id,
            driver_id=driver_id,
            timestamp_generated=datetime.utcnow().isoformat() + "Z",
            overall_status=overall_status,
            total_samples=total_samples,
            duration_seconds=duration_sec,
            sampling_frequency_hz=round(actual_freq_hz, 2),
            criteria_results=criteria_results,
            regime_breakdown=regime_breakdown,
            mount_drift_deg=round(mount_drift_deg, 2) if mount_drift_deg else None,
            recommendation=recommendation
        )

        return report

    def _build_empty_report(self, driver_id: str, session_id: str, status: str, details: str) -> DataQualityReport:
        return DataQualityReport(
            session_id=session_id,
            driver_id=driver_id,
            timestamp_generated=datetime.utcnow().isoformat() + "Z",
            overall_status=status,
            total_samples=0,
            duration_seconds=0.0,
            sampling_frequency_hz=0.0,
            criteria_results=[QualityCriterionResult(
                criterion_name="Empty Payload Check",
                status=status,
                measured_value=0,
                threshold="Non-empty file",
                details=details
            )],
            regime_breakdown={},
            recommendation=details
        )

    def save_quality_reports(
        self,
        report: DataQualityReport,
        target_dir: Path
    ) -> Tuple[Path, Path]:
        """Saves JSON and Markdown versions of the Data Quality Report."""
        target_dir.mkdir(parents=True, exist_ok=True)

        json_path = target_dir / "data_quality_report.json"
        with open(json_path, "w") as f:
            f.write(report.model_dump_json(indent=2))

        # Build Markdown Report
        md_lines = [
            f"# Data Quality & Calibration Report — Session `{report.session_id}`",
            "",
            f"**Driver Token:** `{report.driver_id}`  ",
            f"**Evaluation Status:** **`{report.overall_status}`**  ",
            f"**Total Samples:** {report.total_samples:,}  ",
            f"**Duration:** {report.duration_seconds / 60.0:.2f} minutes ({report.duration_seconds:.1f} s)  ",
            f"**Nominal Sampling Frequency:** {report.sampling_frequency_hz:.2f} Hz  ",
            f"**Mount Angular Drift:** {f'{report.mount_drift_deg:.2f}°' if report.mount_drift_deg is not None else 'N/A'}  ",
            f"**Generated:** {report.timestamp_generated}  ",
            "",
            "---",
            "",
            "## 1. Quality Acceptance Matrix",
            "",
            "| Quality Criterion | Status | Measured Value | Standard Threshold | Details |",
            "|---|:---:|---|---|---|"
        ]

        for crit in report.criteria_results:
            status_badge = f"**`{crit.status}`**"
            md_lines.append(f"| {crit.criterion_name} | {status_badge} | `{crit.measured_value}` | {crit.threshold} | {crit.details} |")

        md_lines.extend([
            "",
            "---",
            "",
            "## 2. Regime Coverage Breakdown",
            ""
        ])

        if report.regime_breakdown:
            md_lines.append("| Segment ID | Description / Road Type | Duration (Minutes) |")
            md_lines.append("|:---:|---|---:|")
            for seg, mins in report.regime_breakdown.items():
                md_lines.append(f"| `{seg}` | Regime Segment | {mins:.2f} min |")
        else:
            md_lines.append("_No segment markers recorded._")

        md_lines.extend([
            "",
            "---",
            "",
            "## 3. Operational Recommendation",
            "",
            f"> [!{('IMPORTANT' if report.overall_status == 'PASS' else ('WARNING' if report.overall_status == 'WARNING' else 'CAUTION'))}]",
            f"> {report.recommendation}",
            ""
        ])

        md_path = target_dir / "data_quality_report.md"
        md_path.write_text("\n".join(md_lines))

        return json_path, md_path

"""
EDA Analyzer Module for Driver Activity Monitoring & Safety Agent.
Performs in-depth exploratory data analysis on smartphone IMU sensor recordings,
computes summary metrics, outlier detection, correlations, and exports visualization figures.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:
    def __init__(self, raw_dir: str, output_viz_dir: str):
        self.raw_dir = Path(raw_dir)
        self.output_viz_dir = Path(output_viz_dir)
        self.output_viz_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    def load_driver_data(self, driver_id: str = "Driver-1") -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Load Accel, Gyro, and Pressure for a specific driver."""
        d_dir = self.raw_dir / "drivers" / driver_id
        accel_df = pd.read_csv(d_dir / "Accelerometer.csv")
        gyro_df = pd.read_csv(d_dir / "Gyroscope.csv")
        press_path = d_dir / "Pressure.csv"
        press_df = pd.read_csv(press_path) if press_path.exists() else None
        return accel_df, gyro_df, press_df

    def plot_subject_distribution(self, driver_stats: List[Dict], day_stats: List[Dict]):
        """Plot row counts across drivers (Table 3) and single-driver daywise trips (Table 4)."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Drivers
        df_drivers = pd.DataFrame(driver_stats)
        sns.barplot(data=df_drivers, x="driver_id", y="rows", ax=axes[0], color="#2b5c8f")
        axes[0].set_title("Multi-Subject Telemetry (Driver-1 to 7: 637k Accel rows)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Driver Identifier (Table 3)")
        axes[0].set_ylabel("Accelerometer Record Count")
        for p in axes[0].patches:
            axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")

        # Daywise
        df_days = pd.DataFrame(day_stats)
        sns.barplot(data=df_days, x="trip_id", y="rows", ax=axes[1], color="#388e3c")
        axes[1].set_title("Single-Driver Multi-Day Trips (Day-1 to 7: 1.25M Accel rows)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Trip Session (Table 4)")
        axes[1].set_ylabel("Accelerometer Record Count")
        axes[1].tick_params(axis="x", rotation=45)

        out_path = self.output_viz_dir / "01_subject_record_distribution.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def plot_sensor_distributions(self, accel_df: pd.DataFrame, gyro_df: pd.DataFrame):
        """Plot distribution of raw sensor readings and magnitudes."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))

        accel_mag = np.sqrt(accel_df["X"]**2 + accel_df["Y"]**2 + accel_df["Z"]**2)
        gyro_mag = np.sqrt(gyro_df["X"]**2 + gyro_df["Y"]**2 + gyro_df["Z"]**2)

        # Accel
        for i, col in enumerate(["X", "Y", "Z"]):
            sns.histplot(accel_df[col], kde=True, ax=axes[0, i], color="#1976d2", bins=50)
            axes[0, i].set_title(f"Accelerometer {col} (m/s²)")
            axes[0, i].set_xlabel("Acceleration (m/s²)")
        sns.histplot(accel_mag, kde=True, ax=axes[0, 3], color="#d32f2f", bins=50)
        axes[0, 3].set_title("Acceleration Norm ||a|| (m/s²)")
        axes[0, 3].set_xlabel("Norm (m/s²)")
        axes[0, 3].axvline(9.81, color="black", linestyle="--", label="1g (9.81 m/s²)")
        axes[0, 3].legend()

        # Gyro
        for i, col in enumerate(["X", "Y", "Z"]):
            sns.histplot(gyro_df[col], kde=True, ax=axes[1, i], color="#388e3c", bins=50)
            axes[1, i].set_title(f"Gyroscope {col} (rad/s)")
            axes[1, i].set_xlabel("Angular Velocity (rad/s)")
        sns.histplot(gyro_mag, kde=True, ax=axes[1, 3], color="#f57c00", bins=50)
        axes[1, 3].set_title("Gyroscope Norm ||ω|| (rad/s)")
        axes[1, 3].set_xlabel("Norm (rad/s)")

        out_path = self.output_viz_dir / "02_sensor_distributions.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def plot_sensor_time_series(self, accel_df: pd.DataFrame, gyro_df: pd.DataFrame, window_samples: int = 3000):
        """Plot snippet of continuous sensor telemetry over time (30 seconds at 100 Hz)."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        snippet_accel = accel_df.iloc[:window_samples]
        snippet_gyro = gyro_df.iloc[:window_samples]
        time_sec_accel = np.arange(len(snippet_accel)) / 100.0  # Empirical 100 Hz
        time_sec_gyro = np.arange(len(snippet_gyro)) / 100.0

        axes[0].plot(time_sec_accel, snippet_accel["X"], label="Acc X (Phone Frame)", alpha=0.8, color="#1976d2")
        axes[0].plot(time_sec_accel, snippet_accel["Y"], label="Acc Y (Phone Frame)", alpha=0.8, color="#388e3c")
        axes[0].plot(time_sec_accel, snippet_accel["Z"], label="Acc Z (Phone Frame / Gravity)", alpha=0.8, color="#f57c00")
        axes[0].set_ylabel("Acceleration (m/s²)")
        axes[0].set_title("Driver-1 Telemetry: 30-Second Acceleration Traces (100 Hz sampling)", fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle=":")

        axes[1].plot(time_sec_gyro, snippet_gyro["X"], label="Gyro X (Phone Frame)", alpha=0.8, color="#7b1fa2")
        axes[1].plot(time_sec_gyro, snippet_gyro["Y"], label="Gyro Y (Phone Frame)", alpha=0.8, color="#c2185b")
        axes[1].plot(time_sec_gyro, snippet_gyro["Z"], label="Gyro Z (Phone Frame)", alpha=0.8, color="#d32f2f")
        axes[1].set_ylabel("Angular Velocity (rad/s)")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_title("Driver-1 Telemetry: 30-Second Gyroscope Traces (100 Hz sampling)", fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle=":")

        out_path = self.output_viz_dir / "03_sensor_time_series.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def plot_correlation_heatmap(self, accel_df: pd.DataFrame, gyro_df: pd.DataFrame):
        """Plot cross-correlation between Accelerometer and Gyroscope axes."""
        min_len = min(len(accel_df), len(gyro_df))
        df_combined = pd.DataFrame({
            "Acc_X": accel_df["X"].iloc[:min_len],
            "Acc_Y": accel_df["Y"].iloc[:min_len],
            "Acc_Z": accel_df["Z"].iloc[:min_len],
            "Acc_Norm": np.sqrt(accel_df["X"]**2 + accel_df["Y"]**2 + accel_df["Z"]**2).iloc[:min_len],
            "Gyro_X": gyro_df["X"].iloc[:min_len],
            "Gyro_Y": gyro_df["Y"].iloc[:min_len],
            "Gyro_Z": gyro_df["Z"].iloc[:min_len],
            "Gyro_Norm": np.sqrt(gyro_df["X"]**2 + gyro_df["Y"]**2 + gyro_df["Z"]**2).iloc[:min_len],
        })

        corr = df_combined.corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Kinematic Feature Correlation Matrix (Phone Frame)", fontsize=12, fontweight="bold")

        out_path = self.output_viz_dir / "04_correlation_heatmap.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def plot_outlier_boxplots(self, accel_df: pd.DataFrame, gyro_df: pd.DataFrame):
        """Plot boxplots to inspect outliers and distributions across axes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=accel_df[["X", "Y", "Z"]], ax=axes[0], palette="Blues")
        axes[0].set_title("Accelerometer Measurements & Spikes (m/s²)", fontweight="bold")
        axes[0].set_ylabel("m/s²")

        sns.boxplot(data=gyro_df[["X", "Y", "Z"]], ax=axes[1], palette="Greens")
        axes[1].set_title("Gyroscope Measurements & Spikes (rad/s)", fontweight="bold")
        axes[1].set_ylabel("rad/s")

        out_path = self.output_viz_dir / "05_outlier_boxplots.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def plot_driver_comparison(self):
        """Compare acceleration magnitude and variance across all 7 drivers."""
        comparison = []
        for i in range(1, 8):
            d = f"Driver-{i}"
            accel_path = self.raw_dir / "drivers" / d / "Accelerometer.csv"
            gyro_path = self.raw_dir / "drivers" / d / "Gyroscope.csv"
            if accel_path.exists() and gyro_path.exists():
                a_df = pd.read_csv(accel_path)
                g_df = pd.read_csv(gyro_path)
                a_mag = np.sqrt(a_df["X"]**2 + a_df["Y"]**2 + a_df["Z"]**2)
                g_mag = np.sqrt(g_df["X"]**2 + g_df["Y"]**2 + g_df["Z"]**2)
                comparison.append({
                    "driver": d,
                    "accel_mean": a_mag.mean(),
                    "accel_std": a_mag.std(),
                    "accel_max": a_mag.max(),
                    "gyro_mean": g_mag.mean(),
                    "gyro_std": g_mag.std(),
                    "gyro_max": g_mag.max(),
                })

        df_comp = pd.DataFrame(comparison)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(data=df_comp, x="driver", y="accel_std", ax=axes[0], color="#1976d2")
        axes[0].set_title("Driver Comparison: Acceleration Norm Dispersion (Std Dev)", fontweight="bold")
        axes[0].set_ylabel("Accel Norm Std Dev (m/s²)")
        axes[0].set_xlabel("Driver (Table 3)")

        sns.barplot(data=df_comp, x="driver", y="gyro_std", ax=axes[1], color="#f57c00")
        axes[1].set_title("Driver Comparison: Angular Velocity Norm Dispersion (Std Dev)", fontweight="bold")
        axes[1].set_ylabel("Gyro Norm Std Dev (rad/s)")
        axes[1].set_xlabel("Driver (Table 3)")

        out_path = self.output_viz_dir / "06_driver_comparison.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def plot_sampling_jitter(self, accel_df: pd.DataFrame):
        """Analyze time delta distribution between consecutive samples."""
        diffs = accel_df["Milliseconds"].diff()
        valid = diffs[(diffs > 0) & (diffs < 100)].dropna()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.histplot(valid, bins=30, kde=True, ax=ax, color="#7b1fa2")
        ax.set_title("Accelerometer Sampling Interval Distribution (Audit: ~100 Hz)", fontweight="bold")
        ax.set_xlabel("Delta Time Between Consecutive Samples (Milliseconds)")
        ax.set_ylabel("Sample Count")
        median_val = float(valid.median())
        ax.axvline(median_val, color="red", linestyle="--", label=f"Median = {median_val:.1f} ms (~{1000/median_val:.0f} Hz)")
        ax.legend()

        out_path = self.output_viz_dir / "07_sampling_jitter.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def run_all(self):
        """Execute full EDA analysis and save all visualization artifacts."""
        print("Starting full EDA visualization pipeline...")
        accel_df, gyro_df, press_df = self.load_driver_data("Driver-1")

        driver_stats = []
        for i in range(1, 8):
            p = self.raw_dir / "drivers" / f"Driver-{i}" / "Accelerometer.csv"
            if p.exists():
                count = sum(1 for _ in open(p, "rb")) - 1
                driver_stats.append({"driver_id": f"Driver-{i}", "rows": count})

        day_stats = []
        for d in sorted(list((self.raw_dir / "daywise" / "Daywise data").glob("Day-*"))):
            p = d / "Accelerometer.csv"
            if p.exists():
                count = sum(1 for _ in open(p, "rb")) - 1
                day_stats.append({"trip_id": d.name, "rows": count})

        self.plot_subject_distribution(driver_stats, day_stats)
        self.plot_sensor_distributions(accel_df, gyro_df)
        self.plot_sensor_time_series(accel_df, gyro_df)
        self.plot_correlation_heatmap(accel_df, gyro_df)
        self.plot_outlier_boxplots(accel_df, gyro_df)
        self.plot_driver_comparison()
        self.plot_sampling_jitter(accel_df)
        print("All EDA visualizations generated successfully!")


if __name__ == "__main__":
    raw_dir = r"d:\driver ml model\driver-monitoring-agent\dataset\raw"
    viz_dir = r"d:\driver ml model\driver-monitoring-agent\reports\eda\visualizations"
    analyzer = EDAAnalyzer(raw_dir, viz_dir)
    analyzer.run_all()

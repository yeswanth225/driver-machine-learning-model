"""
Vehicle-Frame Coordinate Alignment Engine (Research Module)
============================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: ml/inference/vehicle_frame_alignment.py

Estimates coordinate transformation between an arbitrarily oriented smartphone
windshield/dash clamp and standard vehicle body axes (SAE J670):
- X_veh: Longitudinal axis (Forward +, Braking -)
- Y_veh: Lateral axis (Right +, Left -)
- Z_veh: Vertical axis (Downwards / Gravity +, Upwards -)

Calibration sequence:
1. Static Window (S0): Estimates gravity vector g_0 to derive the down unit vector u_down.
2. Dynamic Window (Straight acceleration): Projects dynamic acceleration onto horizontal plane
   to derive the forward unit vector u_fwd.
3. Constructs direction cosine rotation matrix R = [u_x, u_y, u_z]^T.

Notice: This module is an opt-in research tool. It is not wired into the production
inference path by default, preserving frozen model invariants.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class VehicleFrameAligner:
    """
    Computes and applies dynamic 3D coordinate rotation from device frame to vehicle body frame.
    """

    def __init__(self, target_gravity_g: float = 1.0):
        self.target_gravity_g = target_gravity_g
        self.reset()

    def reset(self):
        """Reset calibration states."""
        self._down_vector: Optional[np.ndarray] = None  # unit vector in device frame
        self._fwd_vector: Optional[np.ndarray] = None   # unit vector in device frame
        self._rotation_matrix: Optional[np.ndarray] = None  # 3x3 R: v_veh = R @ v_dev
        self._static_confidence: float = 0.0
        self._dynamic_confidence: float = 0.0
        self._mean_gravity_norm_g: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        """True if both down and forward axes have been resolved."""
        return self._rotation_matrix is not None

    @property
    def is_static_calibrated(self) -> bool:
        """True if static gravity down-vector has been resolved."""
        return self._down_vector is not None

    @property
    def alignment_confidence(self) -> float:
        """Composite alignment quality score [0.0, 1.0]."""
        if not self.is_calibrated:
            return 0.5 * self._static_confidence
        return float(np.clip(0.5 * self._static_confidence + 0.5 * self._dynamic_confidence, 0.0, 1.0))

    @property
    def down_vector(self) -> Optional[np.ndarray]:
        """Resolved down unit vector in device coordinates, or None if uncalibrated."""
        return self._down_vector.copy() if self._down_vector is not None else None

    @property
    def forward_vector(self) -> Optional[np.ndarray]:
        """Resolved forward unit vector in device coordinates, or None if uncalibrated."""
        return self._fwd_vector.copy() if self._fwd_vector is not None else None

    @property
    def rotation_matrix(self) -> Optional[np.ndarray]:
        """Current 3x3 orthonormal rotation matrix, or None if uncalibrated."""
        return self._rotation_matrix.copy() if self._rotation_matrix is not None else None

    def calibrate_static(
        self,
        samples: Union[List[Dict[str, float]], pd.DataFrame, np.ndarray],
        gravity_tolerance: float = 0.05,
    ) -> Tuple[bool, float, str]:
        """
        Calibrate the vehicle down-vector from stationary observations.
        
        Parameters:
            samples: Observations during engine-idle / parked period (S0).
            gravity_tolerance: Allowable deviation from 1.0g (default 0.05g -> [0.95, 1.05]g).
        Returns:
            (success: bool, confidence: float, message: str)
        """
        if isinstance(samples, pd.DataFrame):
            acc = samples[["Acc X", "Acc Y", "Acc Z"]].values
        elif isinstance(samples, list) and len(samples) > 0 and isinstance(samples[0], dict):
            acc = np.array([[s["Acc X"], s["Acc Y"], s["Acc Z"]] for s in samples], dtype=np.float64)
        elif isinstance(samples, np.ndarray) and samples.ndim == 2 and samples.shape[1] >= 3:
            acc = samples[:, :3]
        else:
            return False, 0.0, "Invalid input sample format. Expected DataFrame, list of dicts, or (N,3) array."

        if len(acc) < 5:
            return False, 0.0, f"Insufficient static samples ({len(acc)} < 5 minimum)."

        # Check for NaNs
        if np.isnan(acc).any() or np.isinf(acc).any():
            return False, 0.0, "Non-finite values detected in static calibration window."

        mean_acc = np.mean(acc, axis=0)
        norm_g = float(np.linalg.norm(mean_acc))
        std_g = float(np.mean(np.std(acc, axis=0)))

        self._mean_gravity_norm_g = norm_g

        # Evaluate gravity norm against canonical bounds [0.98, 1.02]g
        norm_error = abs(norm_g - self.target_gravity_g)
        if norm_error > gravity_tolerance:
            self._static_confidence = 0.0
            return False, 0.0, f"Gravity norm {norm_g:.3f}g deviates from {self.target_gravity_g}g by {norm_error:.3f}g (> {gravity_tolerance}g allowable)."

        # Down vector in device coordinates
        self._down_vector = mean_acc / norm_g

        # Confidence: high when variance is low and norm error is near zero
        variance_penalty = min(1.0, std_g / 0.05)
        norm_score = max(0.0, 1.0 - (norm_error / gravity_tolerance))
        self._static_confidence = float(np.clip(norm_score * (1.0 - 0.5 * variance_penalty), 0.0, 1.0))

        return True, self._static_confidence, f"Static calibration successful (|g0| = {norm_g:.3f}g, std = {std_g:.4f}g, confidence = {self._static_confidence:.2f})."

    def calibrate_dynamic(
        self,
        samples: Union[List[Dict[str, float]], pd.DataFrame, np.ndarray],
        speed_delta_mps: Optional[float] = None,
        min_excitation_g: float = 0.08,
    ) -> Tuple[bool, float, str]:
        """
        Calibrate the vehicle forward-vector from straight-line driving observations.
        Requires static calibration to have completed first.
        
        Parameters:
            samples: Observations during straight-line acceleration or braking.
            speed_delta_mps: Optional change in speed over the window (positive = acceleration, negative = braking).
            min_excitation_g: Minimum horizontal acceleration needed for robust orientation (default 0.08g).
        Returns:
            (success: bool, confidence: float, message: str)
        """
        if self._down_vector is None:
            return False, 0.0, "Static calibration must be completed before dynamic forward calibration."

        if isinstance(samples, pd.DataFrame):
            acc = samples[["Acc X", "Acc Y", "Acc Z"]].values
            gyro = samples[["gyro_x", "gyro_y", "gyro_z"]].values if "gyro_z" in samples.columns else None
        elif isinstance(samples, list) and len(samples) > 0 and isinstance(samples[0], dict):
            acc = np.array([[s["Acc X"], s["Acc Y"], s["Acc Z"]] for s in samples], dtype=np.float64)
            gyro = np.array([[s.get("gyro_x", 0.0), s.get("gyro_y", 0.0), s.get("gyro_z", 0.0)] for s in samples]) if "gyro_z" in samples[0] else None
        elif isinstance(samples, np.ndarray) and samples.ndim == 2 and samples.shape[1] >= 3:
            acc = samples[:, :3]
            gyro = samples[:, 3:6] if samples.shape[1] >= 6 else None
        else:
            return False, 0.0, "Invalid dynamic sample format."

        if len(acc) < 10:
            return False, 0.0, f"Insufficient dynamic samples ({len(acc)} < 10 minimum)."

        # If gyro is provided, verify driving is straight (|gyro_z| < 0.10 rad/s)
        if gyro is not None:
            mean_yaw_rate = float(np.mean(np.abs(gyro[:, 2])))
            if mean_yaw_rate > 0.15:
                return False, 0.0, f"Straight-line requirement violated: mean yaw rate {mean_yaw_rate:.3f} rad/s exceeds 0.15 rad/s threshold."

        u_down = self._down_vector

        # Remove gravity component to isolate dynamic acceleration: a_dyn = a - (a . u_down) * u_down
        projections_down = np.dot(acc, u_down)[:, np.newaxis]
        acc_horiz = acc - projections_down * u_down[np.newaxis, :]

        # Find principal horizontal component via SVD / PCA
        horiz_magnitudes = np.linalg.norm(acc_horiz, axis=1)
        mean_horiz_g = float(np.mean(horiz_magnitudes))
        max_horiz_g = float(np.max(horiz_magnitudes))

        if max_horiz_g < min_excitation_g:
            return False, 0.0, f"Insufficient longitudinal excitation: peak dynamic horizontal acceleration {max_horiz_g:.3f}g < {min_excitation_g:.3f}g minimum."

        # Compute dominant axis from covariance
        cov = np.cov(acc_horiz.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_vector = eigvecs[:, -1]

        # Ensure principal vector is orthogonal to down
        principal_vector = principal_vector - np.dot(principal_vector, u_down) * u_down
        norm_p = np.linalg.norm(principal_vector)
        if norm_p < 1e-6:
            return False, 0.0, "Failed to resolve orthogonal horizontal axis."
        principal_vector /= norm_p

        # Determine forward direction sign
        # If speed_delta_mps is provided:
        # positive speed delta implies positive acceleration along forward direction
        if speed_delta_mps is not None and abs(speed_delta_mps) > 0.5:
            # Average dynamic acceleration in direction of principal vector
            dot_products = np.dot(acc_horiz, principal_vector)
            mean_dot = float(np.mean(dot_products))
            if speed_delta_mps > 0:
                # Accelerating: forward vector aligns with positive acceleration
                if mean_dot < 0:
                    principal_vector = -principal_vector
            else:
                # Braking: forward vector opposes braking acceleration
                if mean_dot > 0:
                    principal_vector = -principal_vector

        self._fwd_vector = principal_vector

        # Construct right-handed vehicle body coordinate basis (SAE J670):
        # u_x = forward
        # u_z = down
        # u_y = down x forward (points right)
        u_x = self._fwd_vector
        u_z = self._down_vector
        u_y = np.cross(u_z, u_x)
        norm_y = np.linalg.norm(u_y)
        if norm_y < 1e-6:
            return False, 0.0, "Singularity: forward vector parallel to down vector."
        u_y /= norm_y

        # Re-orthogonalize u_x = u_y x u_z
        u_x = np.cross(u_y, u_z)
        u_x /= np.linalg.norm(u_x)
        self._fwd_vector = u_x

        # Rotation matrix: v_veh = R @ v_dev
        self._rotation_matrix = np.vstack([u_x, u_y, u_z])

        # Dynamic confidence based on excitation SNR
        snr = min(1.0, max_horiz_g / 0.30)
        self._dynamic_confidence = float(np.clip(snr, 0.1, 1.0))

        return True, self._dynamic_confidence, f"Dynamic alignment successful (peak horiz: {max_horiz_g:.3f}g, confidence: {self._dynamic_confidence:.2f})."

    def transform_acceleration(self, ax: float, ay: float, az: float) -> Tuple[float, float, float]:
        """
        Transforms 3D acceleration from device frame to vehicle body frame.
        If uncalibrated, returns original readings.
        """
        if self._rotation_matrix is None:
            return float(ax), float(ay), float(az)
        v_dev = np.array([ax, ay, az], dtype=np.float64)
        v_veh = self._rotation_matrix @ v_dev
        return float(v_veh[0]), float(v_veh[1]), float(v_veh[2])

    def transform_angular_velocity(self, gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
        """
        Transforms 3D angular rate from device frame to vehicle body frame.
        If uncalibrated, returns original readings.
        """
        if self._rotation_matrix is None:
            return float(gx), float(gy), float(gz)
        v_dev = np.array([gx, gy, gz], dtype=np.float64)
        v_veh = self._rotation_matrix @ v_dev
        return float(v_veh[0]), float(v_veh[1]), float(v_veh[2])

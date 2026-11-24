"""
Streaming Feature Extractor for Pipeline A (Candidate V2 102 Causal Features)
=============================================================================
Project: Intelligent Driver Activity Monitoring & Safety Agent
Location: ml/inference/streaming_pipeline_a_features.py

Provides bit-for-bit numerical parity with batch `compute_causal_features()`
while processing telemetry observations sample-by-sample in real time.
Features:
- Fixed-size circular buffers (O(1) memory per stream, 50 samples = 5.0 s at 10 Hz)
- Zero lookahead, right-aligned causal windows
- Recursive EMA tracking (alpha=0.20, alpha=0.08)
- Numerical parity verification: diff < 1e-5 vs offline batch
"""

from collections import deque
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


# Canonical 102 Pruned Features for Candidate V2
CANDIDATE_V2_PRUNED_FEATURES = [
    'Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z',
    'acc_mag', 'gyro_mag', 'acc_horiz', 'acc_jerk', 'gyro_jerk',
    'acc_mag_mean_1s', 'acc_mag_std_1s', 'acc_mag_max_1s', 'acc_mag_min_1s', 'acc_mag_range_1s',
    'gyro_mag_mean_1s', 'gyro_mag_std_1s', 'gyro_mag_max_1s',
    'acc_horiz_mean_1s', 'acc_horiz_std_1s',
    'gyro_z_std_1s', 'gyro_z_max_1s',
    'acc_mag_mean_2s', 'acc_mag_max_2s', 'acc_mag_min_2s', 'acc_mag_range_2s',
    'gyro_mag_mean_2s', 'gyro_mag_std_2s',
    'acc_horiz_mean_2s', 'acc_horiz_std_2s',
    'gyro_z_std_2s', 'gyro_z_max_2s',
    'acc_mag_mean_3s', 'acc_mag_max_3s', 'acc_mag_min_3s', 'acc_mag_range_3s',
    'gyro_mag_mean_3s', 'gyro_mag_std_3s',
    'acc_horiz_mean_3s', 'acc_horiz_std_3s',
    'gyro_z_std_3s', 'gyro_z_max_3s',
    'acc_planar', 'acc_vertical', 'gyro_yaw', 'gyro_wobble',
    'bump_ratio', 'yaw_dominance', 'planar_energy_ratio',
    'acc_planar_ema_fast', 'acc_planar_ema_slow',
    'gyro_yaw_ema_fast', 'gyro_yaw_ema_slow',
    'acc_planar_mean_1s', 'acc_planar_std_1s', 'acc_planar_max_1s', 'acc_planar_range_1s', 'acc_planar_energy_1s',
    'gyro_yaw_mean_1s', 'gyro_yaw_std_1s', 'gyro_yaw_max_1s', 'gyro_yaw_energy_1s',
    'acc_vertical_std_1s', 'acc_vertical_mean_1s',
    'acc_jerk_std_1s',
    'acc_planar_mean_2s', 'acc_planar_std_2s', 'acc_planar_max_2s', 'acc_planar_range_2s', 'acc_planar_energy_2s',
    'gyro_yaw_mean_2s', 'gyro_yaw_std_2s', 'gyro_yaw_max_2s', 'gyro_yaw_energy_2s',
    'acc_vertical_std_2s', 'acc_vertical_mean_2s',
    'acc_jerk_std_2s',
    'acc_planar_mean_3s', 'acc_planar_std_3s', 'acc_planar_max_3s', 'acc_planar_range_3s', 'acc_planar_energy_3s',
    'gyro_yaw_mean_3s', 'gyro_yaw_std_3s', 'gyro_yaw_max_3s', 'gyro_yaw_energy_3s',
    'acc_vertical_std_3s', 'acc_vertical_mean_3s',
    'acc_jerk_std_3s',
    'acc_planar_mean_5s', 'acc_planar_std_5s', 'acc_planar_max_5s', 'acc_planar_range_5s', 'acc_planar_energy_5s',
    'gyro_yaw_mean_5s', 'gyro_yaw_std_5s', 'gyro_yaw_max_5s', 'gyro_yaw_energy_5s',
    'acc_vertical_std_5s', 'acc_vertical_mean_5s',
    'acc_jerk_std_5s'
]


class StreamingPipelineAFeatureExtractor:
    """
    Online feature extractor for Pipeline A Candidate V2.
    Accepts 1 raw sample (Acc X, Acc Y, Acc Z, gyro_x, gyro_y, gyro_z) at a time
    and emits the complete 102-feature vector with exact parity to offline batch.
    """

    MAX_WINDOW_SAMPLES = 50  # 5.0 seconds at 10 Hz
    WINDOW_SIZES = [10, 20, 30, 50]  # 1s, 2s, 3s, 5s

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or CANDIDATE_V2_PRUNED_FEATURES
        self.reset()

    def reset(self) -> None:
        """Reset internal buffers and filter states to cold start."""
        self._sample_count = 0
        
        # Circular buffers for rolling window operations
        self._buf_acc_mag = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_gyro_mag = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_acc_horiz = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_gyro_z = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_gyro_z_abs = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_acc_planar = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_gyro_yaw = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_acc_vertical = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        self._buf_acc_jerk = deque(maxlen=self.MAX_WINDOW_SAMPLES)
        
        # Jerk state
        self._prev_acc_planar: Optional[float] = None
        self._prev_gyro_yaw: Optional[float] = None
        
        # EMA states
        self._acc_planar_ema_fast: Optional[float] = None
        self._acc_planar_ema_slow: Optional[float] = None
        self._gyro_yaw_ema_fast: Optional[float] = None
        self._gyro_yaw_ema_slow: Optional[float] = None

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def is_warmed_up(self) -> bool:
        """True when buffer has reached maximum window length (50 samples / 5.0s)."""
        return self._sample_count >= self.MAX_WINDOW_SAMPLES

    def _calc_rolling_stats(self, buf: deque, window_size: int):
        """
        Compute mean, sample std (ddof=1, fillna 0), min, max, energy for a window slice.
        Matches pandas rolling(window, min_periods=1) behavior.
        """
        n = len(buf)
        w = min(n, window_size)
        # Sliced items from tail
        items = [buf[n - w + i] for i in range(w)]
        m = float(np.mean(items))
        if w > 1:
            s = float(np.std(items, ddof=1))
        else:
            s = 0.0
        mn = float(np.min(items))
        mx = float(np.max(items))
        rng = mx - mn
        eng = float(np.mean([x**2 for x in items]))
        return m, s, mn, mx, rng, eng

    def process_sample(self, raw_sample: Union[Dict[str, float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Process a single raw IMU observation and return the engineered feature dictionary.
        
        Parameters:
            raw_sample: Dict or Series with keys ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']
                        or 6-element array/list [ax, ay, az, gx, gy, gz].
        Returns:
            Dict[str, float] of all 102 engineered features.
        """
        if isinstance(raw_sample, (list, tuple, np.ndarray)):
            ax, ay, az = float(raw_sample[0]), float(raw_sample[1]), float(raw_sample[2])
            gx, gy, gz = float(raw_sample[3]), float(raw_sample[4]), float(raw_sample[5])
        elif isinstance(raw_sample, (dict, pd.Series)):
            ax = float(raw_sample['Acc X'])
            ay = float(raw_sample['Acc Y'])
            az = float(raw_sample['Acc Z'])
            gx = float(raw_sample['gyro_x'])
            gy = float(raw_sample['gyro_y'])
            gz = float(raw_sample['gyro_z'])
        else:
            raise TypeError(f"Unsupported sample type: {type(raw_sample)}")

        self._sample_count += 1

        # 1. Base Magnitudes and Physics
        acc_mag = float(np.sqrt(ax**2 + ay**2 + az**2))
        gyro_mag = float(np.sqrt(gx**2 + gy**2 + gz**2))
        acc_planar = float(np.sqrt(ax**2 + ay**2))
        acc_horiz = acc_planar  # By definition identical
        acc_vertical = float(np.abs(az))
        gyro_yaw = float(np.abs(gz))
        gyro_wobble = float(np.sqrt(gx**2 + gy**2))

        # 2. Causal Jerk Derivatives
        if self._prev_acc_planar is None:
            acc_jerk = 0.0
            gyro_jerk = 0.0
        else:
            acc_jerk = acc_planar - self._prev_acc_planar
            gyro_jerk = gyro_yaw - self._prev_gyro_yaw
        self._prev_acc_planar = acc_planar
        self._prev_gyro_yaw = gyro_yaw

        # 3. Decoupling Ratios
        bump_ratio = float(acc_vertical / (acc_planar + 0.1))
        yaw_dominance = float(gyro_yaw / (gyro_wobble + 0.01))
        planar_energy_ratio = float((acc_planar**2) / (acc_mag**2 + 0.01))

        # 4. Recursive Exponential Moving Averages (adjust=False)
        alpha_fast = 0.20
        alpha_slow = 0.08
        if self._acc_planar_ema_fast is None:
            self._acc_planar_ema_fast = acc_planar
            self._acc_planar_ema_slow = acc_planar
            self._gyro_yaw_ema_fast = gyro_yaw
            self._gyro_yaw_ema_slow = gyro_yaw
        else:
            self._acc_planar_ema_fast = (1.0 - alpha_fast) * self._acc_planar_ema_fast + alpha_fast * acc_planar
            self._acc_planar_ema_slow = (1.0 - alpha_slow) * self._acc_planar_ema_slow + alpha_slow * acc_planar
            self._gyro_yaw_ema_fast = (1.0 - alpha_fast) * self._gyro_yaw_ema_fast + alpha_fast * gyro_yaw
            self._gyro_yaw_ema_slow = (1.0 - alpha_slow) * self._gyro_yaw_ema_slow + alpha_slow * gyro_yaw

        # 5. Append to Circular Buffers
        self._buf_acc_mag.append(acc_mag)
        self._buf_gyro_mag.append(gyro_mag)
        self._buf_acc_horiz.append(acc_horiz)
        self._buf_gyro_z.append(gz)
        self._buf_gyro_z_abs = deque(maxlen=self.MAX_WINDOW_SAMPLES) if not hasattr(self, '_buf_gyro_z_abs') else self._buf_gyro_z_abs
        self._buf_gyro_z_abs.append(abs(gz))
        self._buf_acc_planar.append(acc_planar)
        self._buf_gyro_yaw.append(gyro_yaw)
        self._buf_acc_vertical.append(acc_vertical)
        self._buf_acc_jerk.append(acc_jerk)

        # 6. Assemble Features
        feat_dict = {
            'Acc X': ax, 'Acc Y': ay, 'Acc Z': az,
            'gyro_x': gx, 'gyro_y': gy, 'gyro_z': gz,
            'acc_mag': acc_mag, 'gyro_mag': gyro_mag,
            'acc_horiz': acc_horiz,
            'acc_jerk': acc_jerk, 'gyro_jerk': gyro_jerk,
            'acc_planar': acc_planar, 'acc_vertical': acc_vertical,
            'gyro_yaw': gyro_yaw, 'gyro_wobble': gyro_wobble,
            'bump_ratio': bump_ratio,
            'yaw_dominance': yaw_dominance,
            'planar_energy_ratio': planar_energy_ratio,
            'acc_planar_ema_fast': self._acc_planar_ema_fast,
            'acc_planar_ema_slow': self._acc_planar_ema_slow,
            'gyro_yaw_ema_fast': self._gyro_yaw_ema_fast,
            'gyro_yaw_ema_slow': self._gyro_yaw_ema_slow,
        }

        # 7. Compute Rolling Features for Windows (1s=10, 2s=20, 3s=30, 5s=50)
        for w in self.WINDOW_SIZES:
            sec = w // 10
            prefix = f"{sec}s"

            # Planar stats
            p_m, p_s, p_mn, p_mx, p_rng, p_eng = self._calc_rolling_stats(self._buf_acc_planar, w)
            feat_dict[f'acc_planar_mean_{prefix}'] = p_m
            feat_dict[f'acc_planar_std_{prefix}'] = p_s
            feat_dict[f'acc_planar_max_{prefix}'] = p_mx
            feat_dict[f'acc_planar_min_{prefix}'] = p_mn
            feat_dict[f'acc_planar_range_{prefix}'] = p_rng
            feat_dict[f'acc_planar_energy_{prefix}'] = p_eng

            # Yaw stats
            y_m, y_s, _, y_mx, _, y_eng = self._calc_rolling_stats(self._buf_gyro_yaw, w)
            feat_dict[f'gyro_yaw_mean_{prefix}'] = y_m
            feat_dict[f'gyro_yaw_std_{prefix}'] = y_s
            feat_dict[f'gyro_yaw_max_{prefix}'] = y_mx
            feat_dict[f'gyro_yaw_energy_{prefix}'] = y_eng

            # Vertical stats
            v_m, v_s, _, _, _, _ = self._calc_rolling_stats(self._buf_acc_vertical, w)
            feat_dict[f'acc_vertical_mean_{prefix}'] = v_m
            feat_dict[f'acc_vertical_std_{prefix}'] = v_s

            # Jerk stats
            _, j_s, _, _, _, _ = self._calc_rolling_stats(self._buf_acc_jerk, w)
            feat_dict[f'acc_jerk_std_{prefix}'] = j_s

            # Baseline window features (1s, 2s, 3s)
            if w <= 30:
                am_m, am_s, am_mn, am_mx, am_rng, _ = self._calc_rolling_stats(self._buf_acc_mag, w)
                feat_dict[f'acc_mag_mean_{prefix}'] = am_m
                feat_dict[f'acc_mag_std_{prefix}'] = am_s
                feat_dict[f'acc_mag_max_{prefix}'] = am_mx
                feat_dict[f'acc_mag_min_{prefix}'] = am_mn
                feat_dict[f'acc_mag_range_{prefix}'] = am_rng

                gm_m, gm_s, _, gm_mx, _, _ = self._calc_rolling_stats(self._buf_gyro_mag, w)
                feat_dict[f'gyro_mag_mean_{prefix}'] = gm_m
                feat_dict[f'gyro_mag_std_{prefix}'] = gm_s
                feat_dict[f'gyro_mag_max_{prefix}'] = gm_mx

                ah_m, ah_s, _, _, _, _ = self._calc_rolling_stats(self._buf_acc_horiz, w)
                feat_dict[f'acc_horiz_mean_{prefix}'] = ah_m
                feat_dict[f'acc_horiz_std_{prefix}'] = ah_s

                _, gz_s, _, _, _, _ = self._calc_rolling_stats(self._buf_gyro_z, w)
                _, _, _, gz_mx, _, _ = self._calc_rolling_stats(self._buf_gyro_z_abs, w)
                feat_dict[f'gyro_z_std_{prefix}'] = gz_s
                feat_dict[f'gyro_z_max_{prefix}'] = gz_mx

        # Filter strictly to requested feature set
        return {k: feat_dict[k] for k in self.feature_names if k in feat_dict}

    def process_sample_to_array(self, raw_sample: Union[Dict[str, float], np.ndarray, pd.Series]) -> np.ndarray:
        """Process 1 sample and return ordered float64 feature array."""
        feat_dict = self.process_sample(raw_sample)
        return np.array([feat_dict[f] for f in self.feature_names], dtype=np.float64)

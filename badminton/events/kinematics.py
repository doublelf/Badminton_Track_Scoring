"""Trajectory kinematics: velocity, acceleration, speed, direction, curvature.

Faithful port of the ``BadmintonTrackNet`` kinematics calculator, which in
turn follows the TrackNetV3 reference implementation.
"""

from __future__ import annotations

import numpy as np


class KinematicsCalculator:
    """Computes per-frame motion features from a shuttlecock trajectory."""

    def __init__(self, fps: float = 30.0, reference_fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.reference_fps = float(reference_fps)
        self.dt = 1.0 / self.fps

    def compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return velocity/acceleration/speed/direction/curvature arrays."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        visibility = np.asarray(visibility, dtype=np.float64)
        n = len(x)
        if len(y) != n or len(visibility) != n:
            raise ValueError("x, y and visibility must have the same length")

        empty_names = ("v_x", "v_y", "a_x", "a_y", "speed", "direction", "curvature")
        if n == 0:
            return {name: np.array([], dtype=np.float64) for name in empty_names}

        visible_indices = np.flatnonzero(visibility > 0)
        if len(visible_indices):
            positions = np.arange(n)
            x_work = np.interp(positions, visible_indices, x[visible_indices])
            y_work = np.interp(positions, visible_indices, y[visible_indices])
        else:
            x_work = np.zeros(n)
            y_work = np.zeros(n)

        scale = self.fps / self.reference_fps
        if n > 1:
            v_x = np.gradient(x_work) * scale
            v_y = np.gradient(y_work) * scale
            a_x = np.gradient(v_x) * scale
            a_y = np.gradient(v_y) * scale
        else:
            v_x = np.zeros(1)
            v_y = np.zeros(1)
            a_x = np.zeros(1)
            a_y = np.zeros(1)

        invisible = visibility == 0
        v_x[invisible] = 0.0
        v_y[invisible] = 0.0
        a_x[invisible] = 0.0
        a_y[invisible] = 0.0

        speed = np.sqrt(v_x**2 + v_y**2)
        direction = np.arctan2(v_y, v_x)
        cross = np.abs(v_x * a_y - v_y * a_x)
        speed_cubed = np.power(speed, 3)
        curvature = np.divide(cross, speed_cubed, out=np.zeros_like(cross), where=speed_cubed > 1e-6)

        return {
            "v_x": v_x,
            "v_y": v_y,
            "a_x": a_x,
            "a_y": a_y,
            "speed": speed,
            "direction": direction,
            "curvature": curvature,
            "visibility": visibility,
        }

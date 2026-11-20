"""Constant-velocity Kalman filter used to stabilise the shuttlecock track."""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """Tracks ``(x, y, vx, vy)`` of a moving object in image space.

    Velocities are expressed in **pixels per frame** (the same unit the
    shuttlecock event rules use), therefore the default time step is one
    frame (``dt = 1.0``).
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 20.0,
        measurement_noise: float = 4.0,
        initial_velocity: float = 0.0,
    ) -> None:
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_velocity = initial_velocity

        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 10.0
        self.initialised = False
        self.frame_index = -1

    # ------------------------------------------------------------------ matrices
    def _transition(self) -> np.ndarray:
        dt = self.dt
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _process_covariance(self) -> np.ndarray:
        dt = self.dt
        q = self.process_noise
        return q * np.array(
            [
                [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
                [0.0, dt**4 / 4, 0.0, dt**3 / 2],
                [dt**3 / 2, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2, 0.0, dt**2],
            ],
            dtype=np.float64,
        )

    # -------------------------------------------------------------------- update
    def initialise(self, x: float, y: float, frame_index: int = -1) -> None:
        self.state = np.array([x, y, self.initial_velocity, self.initial_velocity], dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 10.0
        self.initialised = True
        self.frame_index = frame_index

    def predict(self, dt: float | None = None) -> np.ndarray:
        """Advance the state by one time step."""
        if dt is not None:
            self.dt = dt
        if not self.initialised:
            return self.state.copy()
        transition = self._transition()
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + self._process_covariance()
        return self.state.copy()

    def correct(self, x: float, y: float) -> np.ndarray:
        """Apply a measurement correction to the predicted state."""
        measurement = np.array([x, y], dtype=np.float64)
        if not self.initialised:
            self.initialise(x, y)
            return self.state.copy()

        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        r = np.eye(2, dtype=np.float64) * self.measurement_noise
        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + r
        kalman_gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - kalman_gain @ observation) @ self.covariance
        return self.state.copy()

    def update(self, x: float, y: float, frame_index: int) -> np.ndarray:
        """Predict one step and correct with a new measurement."""
        if not self.initialised:
            self.initialise(x, y, frame_index)
            return self.state.copy()
        self.predict()
        self.correct(x, y)
        self.frame_index = frame_index
        return self.state.copy()

    def reset_velocity(self) -> None:
        """Forget the current velocity (used after a re-acquisition)."""
        self.state[2] = 0.0
        self.state[3] = 0.0

    # ---------------------------------------------------------------- properties
    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.state[2], self.state[3]))

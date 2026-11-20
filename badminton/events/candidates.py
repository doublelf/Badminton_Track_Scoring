"""Hit / landing candidate generation.

Faithful port of the Phase-1 rule engine of the open-source
``BadmintonTrackNet`` project (MIT).  The engine works on a full trajectory
``(x, y, visibility)`` and produces candidate events:

``hit``
    ``vy_reversal``, ``vx_reversal``, ``acceleration_peak``
``landing``
    ``speed_step``, ``low_speed_start``, ``speed_drop``, ``visibility_drop``,
    ``trajectory_end``
``out_of_frame``
    visibility loss near the frame border or while moving upwards
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .kinematics import KinematicsCalculator

RULE_PRIORITY = {
    "vy_reversal": 7,
    "vx_reversal": 6,
    "acceleration_peak": 5,
    "speed_step": 5,
    "low_speed_start": 5,
    "speed_drop": 4,
    "visibility_drop": 3,
    "visibility_drop_edge": 3,
    "visibility_drop_upward": 3,
    "visibility_drop_high_altitude": 3,
    "trajectory_end": 3,
    "y_local_max": 2,
    "speed_local_max": 2,
    "direction_change": 0,
}


class BounceCandidateGenerator:
    """Generates hit / landing candidates from a shuttlecock trajectory."""

    def __init__(
        self,
        speed_drop_ratio: float = 0.3,
        min_speed_before_landing: float = 8.0,
        max_speed_after_landing: float = 5.0,
        high_speed_threshold: float = 8.0,
        low_speed_threshold: float = 2.5,
        low_speed_ratio: float = 0.7,
        pre_window: int = 2,
        post_window: int = 4,
        hold_window: int = 3,
        min_speed_at_hit: float = 3.0,
        vy_reversal_threshold: float = 1.0,
        acc_threshold: float = 3.0,
        min_height_diff: float = 5.0,
        min_peak_speed: float = 10.0,
        min_speed_diff: float = 2.0,
        min_visible_before: int = 3,
        merge_window: int = 3,
        future_check_window: int = 5,
        edge_margin: int = 20,
    ) -> None:
        self.speed_drop_ratio = speed_drop_ratio
        self.min_speed_before_landing = min_speed_before_landing
        self.max_speed_after_landing = max_speed_after_landing
        self.high_speed_threshold = high_speed_threshold
        self.low_speed_threshold = low_speed_threshold
        self.low_speed_ratio = low_speed_ratio
        self.pre_window = pre_window
        self.post_window = post_window
        self.hold_window = hold_window
        self.min_speed_at_hit = min_speed_at_hit
        self.vy_reversal_threshold = vy_reversal_threshold
        self.acc_threshold = acc_threshold
        self.min_height_diff = min_height_diff
        self.min_peak_speed = min_peak_speed
        self.min_speed_diff = min_speed_diff
        self.min_visible_before = min_visible_before
        self.merge_window = merge_window
        self.future_check_window = future_check_window
        self.edge_margin = edge_margin

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _last_visible(visibility: np.ndarray) -> int | None:
        indices = np.flatnonzero(visibility == 1)
        if len(indices) == 0:
            return None
        return int(indices[-1])

    def _find_tail_low_speed_start(self, speed: np.ndarray, visibility: np.ndarray, last_visible: int) -> int | None:
        if last_visible <= 0:
            return None
        index = last_visible
        while index >= 0:
            if visibility[index] != 1:
                index -= 1
                continue
            if speed[index] <= self.low_speed_threshold:
                index -= 1
                continue
            break
        tail_start = index + 1
        if tail_start > last_visible:
            return None
        low_length = 0
        for i in range(tail_start, last_visible + 1):
            if visibility[i] == 1 and speed[i] <= self.low_speed_threshold:
                low_length += 1
        if low_length < self.hold_window:
            return None
        return tail_start

    # ------------------------------------------------------------- hit rules
    def _check_vy_reversal(self, t: int, n: int, visibility, v_y, speed) -> dict[str, Any] | None:
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t - 1] != 1 or visibility[t + 1] != 1:
            return None
        vy_before, vy_after = v_y[t - 1], v_y[t + 1]
        threshold = self.vy_reversal_threshold
        is_reversal = (vy_before > threshold and vy_after < -threshold) or (
            vy_before < -threshold and vy_after > threshold
        )
        if not is_reversal:
            return None
        speed_after = speed[t + 1]
        if speed_after < self.min_speed_at_hit:
            return None
        return {
            "vy_before": float(vy_before),
            "vy_after": float(vy_after),
            "speed_after": float(speed_after),
            "reversal_magnitude": float(abs(vy_before - vy_after)),
        }

    def _check_vx_reversal(self, t: int, n: int, visibility, v_x, speed) -> dict[str, Any] | None:
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t - 1] != 1 or visibility[t + 1] != 1:
            return None
        vx_before, vx_after = v_x[t - 1], v_x[t + 1]
        threshold = self.vy_reversal_threshold
        is_reversal = (vx_before > threshold and vx_after < -threshold) or (
            vx_before < -threshold and vx_after > threshold
        )
        if not is_reversal:
            return None
        speed_after = speed[t + 1]
        if speed_after < self.min_speed_at_hit:
            return None
        return {
            "vx_before": float(vx_before),
            "vx_after": float(vx_after),
            "speed_after": float(speed_after),
            "reversal_magnitude": float(abs(vx_before - vx_after)),
        }

    def _check_acceleration_peak(self, t: int, n: int, visibility, acceleration, speed) -> dict[str, Any] | None:
        if t <= 1 or t + 1 >= n:
            return None
        if visibility[t - 1] != 1 or visibility[t] != 1 or visibility[t + 1] != 1:
            return None
        acc_before, acc_current, acc_after = acceleration[t - 1], acceleration[t], acceleration[t + 1]
        if not (acc_current > acc_before and acc_current > acc_after and acc_current > self.acc_threshold):
            return None
        return {
            "acc_before": float(acc_before),
            "acc_current": float(acc_current),
            "acc_after": float(acc_after),
            "speed": float(speed[t]),
            "acc_ratio": float(acc_current / (max(acc_before, acc_after) + 1e-6)),
        }

    # --------------------------------------------------------- landing rules
    def _check_visibility_drop(self, t, n, x, y, visibility, speed, v_y, img_height, img_width, visible_count):
        if t + 1 >= n:
            return None
        if visibility[t] != 1 or visibility[t + 1] != 0:
            return None
        recent_speed = float(np.mean(speed[max(0, t - 3): t + 1]))
        if recent_speed < self.min_speed_before_landing:
            return None

        is_edge = (
            x[t] < self.edge_margin
            or x[t] > img_width - self.edge_margin
            or y[t] < self.edge_margin
            or y[t] > img_height - self.edge_margin
        )
        edge_distance = float(min(x[t], img_width - x[t], y[t], img_height - y[t]))
        features = {
            "speed_before": recent_speed,
            "v_y": float(v_y[t]),
            "visible_before": int(visible_count),
            "edge_distance": edge_distance,
        }

        if is_edge:
            return {"event_type": "out_of_frame", "rule": "visibility_drop_edge", "confidence": 0.45, "features": features}
        if v_y[t] < -1.0:
            return {
                "event_type": "out_of_frame",
                "rule": "visibility_drop_upward",
                "confidence": 0.50,
                "features": features,
            }
        if y[t] < img_height * 0.15:
            return {
                "event_type": "out_of_frame",
                "rule": "visibility_drop_high_altitude",
                "confidence": 0.50,
                "features": features,
            }
        return {"event_type": "landing", "rule": "visibility_drop", "confidence": 0.55, "features": features}

    def _check_speed_step(self, t: int, n: int, visibility, speed) -> dict[str, Any] | None:
        if t <= 0 or t >= n:
            return None
        if visibility[t] != 1 or visibility[t - 1] != 1:
            return None
        v_prev, v_curr = speed[t - 1], speed[t]
        if v_curr > self.low_speed_threshold:
            return None
        ratio = v_curr / (v_prev + 1e-6)
        if not (v_prev >= self.high_speed_threshold or ratio < self.speed_drop_ratio):
            return None

        post_end = min(n, t + self.post_window + 1)
        post_speeds = speed[t:post_end]
        post_vis = visibility[t:post_end]
        if len(post_speeds) == 0:
            return None
        low_mask = (post_speeds <= self.low_speed_threshold) & (post_vis == 1)
        visible_total = int(np.sum(post_vis))
        low_ratio = float(np.sum(low_mask)) / max(1, visible_total)
        if low_ratio < self.low_speed_ratio:
            return None
        return {"v_prev": float(v_prev), "v_curr": float(v_curr), "ratio": float(ratio), "low_ratio": low_ratio}

    def _check_low_speed_start(self, t: int, n: int, visibility, speed) -> dict[str, Any] | None:
        if t <= 0 or t >= n:
            return None
        if visibility[t] != 1:
            return None
        v_curr = speed[t]
        v_prev = speed[t - 1] if visibility[t - 1] == 1 else v_curr + 1e-6
        if v_curr > self.low_speed_threshold:
            return None
        if v_prev <= self.low_speed_threshold:
            return None
        hold_end = min(n, t + self.hold_window)
        hold_speeds = speed[t:hold_end]
        hold_vis = visibility[t:hold_end]
        if len(hold_speeds) == 0:
            return None
        low_mask = (hold_speeds <= self.low_speed_threshold) & (hold_vis == 1)
        low_ratio = float(np.sum(low_mask)) / max(1, int(np.sum(hold_vis)))
        if low_ratio < self.low_speed_ratio:
            return None
        return {"v_prev": float(v_prev), "v_curr": float(v_curr), "low_ratio": low_ratio}

    def _check_speed_drop(self, t: int, n: int, visibility, speed) -> dict[str, Any] | None:
        if t <= 0 or t + 2 >= n:
            return None
        speed_before = speed[t - 1]
        speed_current = speed[t]
        speed_after = speed[t + 1] if visibility[t + 1] == 1 else 0.0
        if speed_before < self.min_speed_before_landing:
            return None
        speed_ratio = speed_current / (speed_before + 1e-6)
        if not (speed_ratio < self.speed_drop_ratio and speed_after <= self.max_speed_after_landing):
            return None
        check_end = min(n, t + self.future_check_window + 1)
        if check_end > t + 1:
            future_speeds = speed[t + 1: check_end] * visibility[t + 1: check_end]
            future_max_speed = float(np.max(future_speeds)) if len(future_speeds) > 0 else 0.0
            if future_max_speed > 10.0:
                return None
        return {
            "speed_ratio": float(speed_ratio),
            "speed_before": float(speed_before),
            "speed_current": float(speed_current),
            "speed_after": float(speed_after),
        }

    # ------------------------------------------------------- auxiliary rules
    def _check_y_local_max(self, t: int, n: int, y, visibility) -> dict[str, Any] | None:
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t - 1] != 1 or visibility[t] != 1 or visibility[t + 1] != 1:
            return None
        if not (y[t] > y[t - 1] and y[t] > y[t + 1]):
            return None
        height_diff = min(y[t] - y[t - 1], y[t] - y[t + 1])
        if height_diff <= self.min_height_diff:
            return None
        return {
            "y_before": float(y[t - 1]),
            "y_current": float(y[t]),
            "y_after": float(y[t + 1]),
            "height_diff": float(height_diff),
        }

    def _check_speed_local_max(self, t: int, n: int, speed, visibility) -> dict[str, Any] | None:
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t - 1] != 1 or visibility[t] != 1 or visibility[t + 1] != 1:
            return None
        if not (speed[t] > speed[t - 1] and speed[t] > speed[t + 1]):
            return None
        speed_diff = min(speed[t] - speed[t - 1], speed[t] - speed[t + 1])
        if speed[t] <= self.min_peak_speed or speed_diff <= self.min_speed_diff:
            return None
        return {
            "speed_before": float(speed[t - 1]),
            "speed_current": float(speed[t]),
            "speed_after": float(speed[t + 1]),
            "speed_diff": float(speed_diff),
        }

    # ------------------------------------------------------------------ main
    def generate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
        kinematics: dict[str, np.ndarray] | None = None,
        img_height: int = 288,
        img_width: int = 512,
        include_trajectory_end: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate merged event candidates for a whole trajectory."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        visibility = np.asarray(visibility, dtype=np.float64)
        n = len(x)
        if n < self.min_visible_before + 2:
            return []

        if kinematics is None:
            kinematics = KinematicsCalculator().compute(x, y, visibility)

        v_x = np.nan_to_num(kinematics["v_x"], nan=0.0, posinf=0.0, neginf=0.0)
        v_y = np.nan_to_num(kinematics["v_y"], nan=0.0, posinf=0.0, neginf=0.0)
        a_x = np.nan_to_num(kinematics["a_x"], nan=0.0, posinf=0.0, neginf=0.0)
        a_y = np.nan_to_num(kinematics["a_y"], nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.nan_to_num(kinematics["speed"], nan=0.0, posinf=0.0, neginf=0.0)
        acceleration = np.sqrt(a_x**2 + a_y**2)

        candidates: list[dict[str, Any]] = []

        for t in range(self.min_visible_before, n - 1):
            if visibility[t] == 0:
                continue
            visible_count = int(np.sum(visibility[max(0, t - self.min_visible_before): t + 1]))
            if visible_count < self.min_visible_before:
                continue

            primary: dict[str, Any] | None = None
            features: dict[str, Any] = {}

            result = self._check_vy_reversal(t, n, visibility, v_y, speed)
            if result:
                primary = {"event_type": "hit", "rule": "vy_reversal", "confidence": 0.85}
                features = result

            if primary is None:
                result = self._check_vx_reversal(t, n, visibility, v_x, speed)
                if result:
                    primary = {"event_type": "hit", "rule": "vx_reversal", "confidence": 0.80}
                    features = result

            if primary is None:
                result = self._check_acceleration_peak(t, n, visibility, acceleration, speed)
                if result:
                    primary = {"event_type": "hit", "rule": "acceleration_peak", "confidence": 0.75}
                    features = result

            if primary is None:
                result = self._check_speed_step(t, n, visibility, speed)
                if result:
                    primary = {"event_type": "landing", "rule": "speed_step", "confidence": 0.90}
                    features = result

            if primary is None:
                result = self._check_low_speed_start(t, n, visibility, speed)
                if result:
                    primary = {"event_type": "landing", "rule": "low_speed_start", "confidence": 0.85}
                    features = result

            if primary is None:
                result = self._check_speed_drop(t, n, visibility, speed)
                if result:
                    primary = {"event_type": "landing", "rule": "speed_drop", "confidence": 0.80}
                    features = result

            if primary is None:
                result = self._check_visibility_drop(
                    t, n, x, y, visibility, speed, v_y, img_height, img_width, visible_count
                )
                if result:
                    primary = {
                        "event_type": result["event_type"],
                        "rule": result["rule"],
                        "confidence": result["confidence"],
                    }
                    features = result["features"]

            auxiliary_rules: list[str] = []
            auxiliary_features: dict[str, Any] = {}
            confidence_boost = 0.0

            if self._check_y_local_max(t, n, y, visibility):
                auxiliary_rules.append("y_local_max")
                auxiliary_features["y_local_max"] = self._check_y_local_max(t, n, y, visibility)
                if primary:
                    if primary["event_type"] == "landing":
                        confidence_boost += 0.08
                    elif primary["event_type"] == "hit":
                        confidence_boost += 0.05

            if self._check_speed_local_max(t, n, speed, visibility):
                auxiliary_rules.append("speed_local_max")
                auxiliary_features["speed_local_max"] = self._check_speed_local_max(t, n, speed, visibility)
                if primary and primary["event_type"] == "hit":
                    confidence_boost += 0.05

            if primary:
                candidates.append(
                    {
                        "frame": t,
                        "x": float(x[t]),
                        "y": float(y[t]),
                        "event_type": primary["event_type"],
                        "rule": primary["rule"],
                        "confidence": min(0.95, primary["confidence"] + confidence_boost),
                        "all_rules": [primary["rule"]] + auxiliary_rules,
                        "auxiliary_rules": auxiliary_rules,
                        "features": {**features, **auxiliary_features},
                    }
                )
            elif auxiliary_rules:
                candidates.append(
                    {
                        "frame": t,
                        "x": float(x[t]),
                        "y": float(y[t]),
                        "event_type": "hit",
                        "rule": auxiliary_rules[0],
                        "confidence": 0.50,
                        "all_rules": auxiliary_rules,
                        "auxiliary_rules": auxiliary_rules[1:] if len(auxiliary_rules) > 1 else [],
                        "features": auxiliary_features,
                    }
                )

        if include_trajectory_end:
            self._check_trajectory_end(x, y, visibility, speed, candidates)

        merged = self._merge_nearby_candidates(candidates)
        merged.sort(key=lambda candidate: candidate["frame"])
        return merged

    def _check_trajectory_end(self, x, y, visibility, speed, candidates: list[dict[str, Any]]) -> None:
        """A trajectory that simply stops is treated as a landing."""
        last_visible = self._last_visible(visibility)
        if last_visible is None or last_visible <= self.min_visible_before:
            return

        for candidate in candidates:
            if abs(candidate["frame"] - last_visible) <= 3 and candidate.get("confidence", 0) > 0.65:
                return

        tail_start = self._find_tail_low_speed_start(speed, visibility, last_visible)
        if tail_start is not None:
            candidates.append(
                {
                    "frame": tail_start,
                    "x": float(x[tail_start]),
                    "y": float(y[tail_start]),
                    "event_type": "landing",
                    "rule": "trajectory_end",
                    "confidence": 0.88,
                    "all_rules": ["trajectory_end"],
                    "auxiliary_rules": [],
                    "features": {
                        "speed": float(speed[tail_start]),
                        "recent_avg_speed": float(np.mean(speed[max(0, tail_start - 3): tail_start + 1])),
                        "landing_type": "tail_low_speed_start",
                    },
                }
            )
            return

        recent_speeds = speed[max(0, last_visible - 3): last_visible + 1]
        candidates.append(
            {
                "frame": last_visible,
                "x": float(x[last_visible]),
                "y": float(y[last_visible]),
                "event_type": "landing",
                "rule": "trajectory_end",
                "confidence": 0.65,
                "all_rules": ["trajectory_end"],
                "auxiliary_rules": [],
                "features": {
                    "speed": float(speed[last_visible]),
                    "recent_avg_speed": float(np.mean(recent_speeds)),
                    "landing_type": "tail_fallback",
                },
            }
        )

    def _merge_nearby_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge candidates that are close in time, keeping the strongest rule."""
        if len(candidates) <= 1:
            return candidates
        ordered = sorted(candidates, key=lambda candidate: candidate["frame"])
        merged: list[dict[str, Any]] = []
        index = 0
        while index < len(ordered):
            group = [ordered[index]]
            base_frame = ordered[index]["frame"]
            next_index = index + 1
            while next_index < len(ordered) and ordered[next_index]["frame"] - base_frame <= self.merge_window:
                group.append(ordered[next_index])
                next_index += 1
            best = max(group, key=lambda item: (RULE_PRIORITY.get(item["rule"], 0), item["confidence"]))
            merged.append(best)
            index = next_index
        return merged

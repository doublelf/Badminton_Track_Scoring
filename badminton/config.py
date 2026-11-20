"""Configuration loading for the badminton tracking pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {
        "width": 0,
        "height": 0,
        "fps": 0,
    },
    "court": {
        "length": 13.4,
        "width": 6.1,
        "singles_margin": 0.46,
        "service_line_from_net": 1.98,
        "back_service_offset": 0.76,
        "net_height": 1.524,
        "in_out_margin_m": 0.02,
    },
    "player": {
        "model": "weights/yolo11n.pt",
        "imgsz": 640,
        "conf": 0.35,
        "iou": 0.5,
        "every_n_frames": 1,
        "tracker": "bytetrack.yaml",
    },
    "ball": {
        "model": "weights/yolo11s-ball.pt",
        "imgsz": 640,
        "conf": 0.18,
        "iou": 0.45,
        "every_n_frames": 1,
        "max_box_area_ratio": 0.004,
        "max_aspect_ratio": 4.0,
        "max_jump_pixels": 220,
        "prediction_gate_pixels": 260,
        "max_missing_frames": 5,
        "roi_padding_ratio": 0.08,
        # the shuttlecock flies above the court, so the court ROI must not gate it
        "use_roi": False,
    },
    "pose": {
        "enabled": False,
        "model": "weights/yolo11n-pose.pt",
        "imgsz": 640,
        "conf": 0.3,
        "every_n_frames": 3,
    },
    "events": {
        "fps_reference": 30.0,
        "min_visible_before": 3,
        "merge_window": 3,
        "hit": {
            "min_speed_at_hit": 3.0,
            "vy_reversal_threshold": 1.0,
            "vx_reversal_threshold": 1.0,
            "acc_threshold": 3.0,
            "min_height_diff": 5.0,
            "min_peak_speed": 10.0,
            "min_speed_diff": 2.0,
        },
        "landing": {
            "speed_drop_ratio": 0.3,
            "min_speed_before_landing": 8.0,
            "max_speed_after_landing": 5.0,
            "high_speed_threshold": 8.0,
            "low_speed_threshold": 2.5,
            "low_speed_ratio": 0.7,
            "hold_window": 3,
            "pre_window": 2,
            "post_window": 4,
            "future_check_window": 5,
            "edge_margin": 20,
            # landing plausibility checks
            "max_landing_speed": 12.0,
            "max_ascent_velocity": -3.0,
            "max_descent_velocity": 2.0,
            "settle_window": 25,
            "settle_speed": 1.5,
            "min_settle_frames": 6,
        },
        "attribution": {
            "racket_reach_m": 1.6,
            "hit_window_offsets": [0, 3, 5],
            "min_attribution_confidence": 0.25,
        },
    },
    "scoring": {
        "mode": "singles",
        "points_to_win": 21,
        "win_by": 2,
        "point_cap": 30,
        "games_to_win": 2,
        "switch_ends_at": 10,
        "switch_ends_at_third_game": 11,
        "team_a_name": "A",
        "team_b_name": "B",
    },
    "trail": {
        "length": 30,
        "fade": True,
        "color_by_speed": True,
        "min_point_distance": 1.5,
    },
    "render": {
        "hud": True,
        "minimap": True,
        "minimap_size": 300,
        "show_fps": True,
        "show_debug": False,
        "write_video": True,
        "video_fps": 0,
    },
}


class Node:
    """Read-only view over a nested dictionary with attribute and item access."""

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Node(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Node(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        value = self._data[key]
        return Node(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Node({self._data!r})"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> Node:
    """Load configuration, merged on top of :data:`DEFAULT_CONFIG`."""
    data = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config file must contain a mapping: {config_path}")
            data = _deep_merge(data, loaded)
    return Node(data)

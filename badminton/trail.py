"""Shuttlecock trajectory trail rendering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .court.reference import speed_color


@dataclass
class TrailPoint:
    x: float
    y: float
    frame_index: int
    speed: float = 0.0
    visible: bool = True


class TrailBuffer:
    """Keeps the recent shuttlecock positions and draws them as a fading trail."""

    def __init__(
        self,
        length: int = 30,
        min_distance: float = 1.5,
        fade: bool = True,
        color_by_speed: bool = True,
        max_speed: float = 40.0,
    ) -> None:
        self.points: deque[TrailPoint] = deque(maxlen=length)
        self.min_distance = min_distance
        self.fade = fade
        self.color_by_speed = color_by_speed
        self.max_speed = max_speed

    def add(self, x: float, y: float, frame_index: int, speed: float = 0.0, visible: bool = True) -> None:
        if self.points:
            last = self.points[-1]
            if visible and np.hypot(x - last.x, y - last.y) < self.min_distance:
                return
        self.points.append(TrailPoint(x=x, y=y, frame_index=frame_index, speed=speed, visible=visible))

    def clear(self) -> None:
        self.points.clear()

    # ------------------------------------------------------------------ render
    def draw(self, frame: np.ndarray) -> np.ndarray:
        points = [point for point in self.points if point.visible]
        if len(points) < 2:
            return frame

        xs = [point.x for point in points]
        ys = [point.y for point in points]
        margin = 12
        height, width = frame.shape[:2]
        x1 = max(0, int(min(xs)) - margin)
        x2 = min(width - 1, int(max(xs)) + margin)
        y1 = max(0, int(min(ys)) - margin)
        y2 = min(height - 1, int(max(ys)) + margin)
        if x2 <= x1 or y2 <= y1:
            return frame

        region = frame[y1:y2, x1:x2]
        overlay = region.copy()
        count = len(points)
        for index in range(1, count):
            p0, p1 = points[index - 1], points[index]
            alpha = 1.0 if not self.fade else (index / max(count - 1, 1)) ** 1.6
            color = (
                speed_color(p1.speed, self.max_speed)
                if self.color_by_speed
                else (60, 220, 255)
            )
            thickness = 1 + int(2.4 * alpha)
            layer = region.copy()
            cv2.line(
                layer,
                (int(p0.x) - x1, int(p0.y) - y1),
                (int(p1.x) - x1, int(p1.y) - y1),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.addWeighted(layer, alpha, overlay, 1 - alpha, 0, overlay)

        frame[y1:y2, x1:x2] = overlay
        return frame

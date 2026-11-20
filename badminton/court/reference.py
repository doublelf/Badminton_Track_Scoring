"""Badminton court geometry in metric court coordinates.

Court coordinate system used across the project:

* ``x`` runs along the length of the court, ``0`` at the far baseline and
  ``13.4`` at the near baseline (the net sits at ``x = 6.7``).
* ``y`` runs across the width of the court, ``0`` at one sideline and
  ``6.1`` at the other.

Image corner ordering is ``[far-left, far-right, near-right, near-left]``,
which maps onto the court corners listed in :data:`COURT_CORNER_ORDER`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
SINGLES_MARGIN = 0.46
SERVICE_LINE_FROM_NET = 1.98
BACK_SERVICE_OFFSET = 0.76


@dataclass(frozen=True)
class CourtSpec:
    """Metric description of a badminton court."""

    length: float = COURT_LENGTH
    width: float = COURT_WIDTH
    singles_margin: float = SINGLES_MARGIN
    service_line_from_net: float = SERVICE_LINE_FROM_NET
    back_service_offset: float = BACK_SERVICE_OFFSET

    @property
    def net_x(self) -> float:
        return self.length / 2.0

    @property
    def center_y(self) -> float:
        return self.width / 2.0

    @property
    def singles_y_min(self) -> float:
        return self.singles_margin

    @property
    def singles_y_max(self) -> float:
        return self.width - self.singles_margin

    @property
    def corner_order(self) -> list[tuple[float, float]]:
        """Court corners matching image order [far-left, far-right, near-right, near-left]."""
        return [
            (0.0, 0.0),
            (0.0, self.width),
            (self.length, self.width),
            (self.length, 0.0),
        ]

    def lines(self) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
        """Court lines as ``(start, end, weight)`` in court coordinates.

        ``x`` is the length axis (0 at the far baseline, 13.4 at the near
        baseline) and ``y`` the width axis (0 at one sideline, 6.1 at the
        other).
        """
        length, width = self.length, self.width
        net_x = self.net_x
        singles_min = self.singles_y_min
        singles_max = self.singles_y_max
        short_service_top = net_x - self.service_line_from_net
        short_service_bottom = net_x + self.service_line_from_net
        back_service_top = self.back_service_offset
        back_service_bottom = length - self.back_service_offset
        center = self.center_y

        return [
            # baselines (length edges)
            ((0.0, 0.0), (0.0, width), 1.35),
            ((length, 0.0), (length, width), 1.35),
            # doubles sidelines (width edges)
            ((0.0, 0.0), (length, 0.0), 1.35),
            ((0.0, width), (length, width), 1.35),
            # singles sidelines
            ((0.0, singles_min), (length, singles_min), 0.95),
            ((0.0, singles_max), (length, singles_max), 0.95),
            # doubles long service lines
            ((back_service_top, 0.0), (back_service_top, width), 1.0),
            ((back_service_bottom, 0.0), (back_service_bottom, width), 1.0),
            # short service lines
            ((short_service_top, 0.0), (short_service_top, width), 1.15),
            ((short_service_bottom, 0.0), (short_service_bottom, width), 1.15),
            # centre line
            ((0.0, center), (short_service_top, center), 0.8),
            ((short_service_bottom, center), (length, center), 0.8),
            # net
            ((net_x, 0.0), (net_x, width), 0.35),
        ]

    def contains(self, point: tuple[float, float], margin: float = 0.0, singles: bool = True) -> bool:
        """Whether a court-space point lies inside the playing area."""
        x, y = point
        if not (-margin <= x <= self.length + margin):
            return False
        if singles:
            y_min, y_max = self.singles_y_min, self.singles_y_max
        else:
            y_min, y_max = 0.0, self.width
        return -margin <= y - y_min <= (y_max - y_min) + margin

    def half(self, point: tuple[float, float]) -> str:
        """Return ``"A"`` for the far half (x < net) or ``"B"`` for the near half."""
        return "A" if point[0] < self.net_x else "B"

    def distance_to_net(self, point: tuple[float, float]) -> float:
        return abs(point[0] - self.net_x)


def interpolate_color(value: float, low: tuple[int, int, int], high: tuple[int, int, int]) -> tuple[int, int, int]:
    """Linearly interpolate a BGR colour between two anchors."""
    value = max(0.0, min(1.0, value))
    return (
        int(round(low[0] + (high[0] - low[0]) * value)),
        int(round(low[1] + (high[1] - low[1]) * value)),
        int(round(low[2] + (high[2] - low[2]) * value)),
    )


def speed_color(speed: float, max_speed: float = 40.0) -> tuple[int, int, int]:
    """Colour ramp from cool to warm used for trajectory rendering."""
    ratio = 0.0 if max_speed <= 0 else math.sqrt(max(0.0, speed) / max_speed)
    return interpolate_color(ratio, (255, 220, 60), (40, 40, 235))

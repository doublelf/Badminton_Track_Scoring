"""Online event detection with a bounded look-ahead window.

The Phase-1 rules of ``BadmintonTrackNet`` need a few future frames (the
``speed_drop`` rule looks ahead by ``future_check_window`` frames).  This
module keeps a sliding window of the trajectory and evaluates the rules on a
sliding buffer, emitting events with a small delay.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .candidates import BounceCandidateGenerator


class EventDetector:
    """Streaming hit/landing detector."""

    def __init__(
        self,
        generator: BounceCandidateGenerator | None = None,
        img_size: tuple[int, int] = (1280, 720),
        window: int = 48,
    ) -> None:
        self.generator = generator or BounceCandidateGenerator()
        self.img_width, self.img_height = img_size
        self.window = window
        self.x: deque[float] = deque(maxlen=window)
        self.y: deque[float] = deque()
        self.visibility: deque[float] = deque()
        self.frame_indices: deque[int] = deque()
        self.emitted: set[int] = set()

    def _push(self, frame_index: int, x: float, y: float, visible: bool) -> None:
        self.x.append(float(x))
        self.y.append(float(y))
        self.visibility.append(1.0 if visible else 0.0)
        self.frame_indices.append(int(frame_index))

    def update(self, frame_index: int, x: float, y: float, visible: bool) -> list[dict[str, Any]]:
        """Feed one trajectory sample and return any newly detected events."""
        self._push(frame_index, x, y, visible)
        if len(self.frame_indices) < self.window:
            return []

        candidates = self.generator.generate(
            np.array(self.x, dtype=np.float64),
            np.array(self.y, dtype=np.float64),
            np.array(self.visibility, dtype=np.float64),
            img_height=self.img_height,
            img_width=self.img_width,
            include_trajectory_end=False,
        )

        events: list[dict[str, Any]] = []
        for candidate in candidates:
            local_frame = int(candidate["frame"])
            global_frame = self.frame_indices[local_frame]
            if global_frame in self.emitted:
                continue
            # Only report events that are not in the very last frames, so the
            # trailing rules still get their look-ahead window.
            if self.frame_indices[-1] - global_frame < 6:
                continue
            self.emitted.add(global_frame)
            event = dict(candidate)
            event["frame"] = global_frame
            events.append(event)
        return events

    def flush(self) -> list[dict[str, Any]]:
        """Emit the trailing events once the stream ends."""
        if len(self.frame_indices) < 4:
            return []
        candidates = self.generator.generate(
            np.array(self.x, dtype=np.float64),
            np.array(self.y, dtype=np.float64),
            np.array(self.visibility, dtype=np.float64),
            img_height=self.img_height,
            img_width=self.img_width,
            include_trajectory_end=True,
        )
        events: list[dict[str, Any]] = []
        for candidate in candidates:
            local_frame = int(candidate["frame"])
            global_frame = self.frame_indices[local_frame]
            if global_frame in self.emitted:
                continue
            self.emitted.add(global_frame)
            event = dict(candidate)
            event["frame"] = global_frame
            events.append(event)
        return events

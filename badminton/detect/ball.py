"""Shuttlecock detection and tracking.

Follows the open-source ``Good-Badminton`` design: a YOLO detector proposes
candidates, geometric filters remove false positives, and a gated association
step keeps a single stable track.  A constant-velocity Kalman filter smooths
the trajectory and bridges short occlusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..track.kalman import ConstantVelocityKalman


@dataclass
class BallState:
    """State of the shuttlecock for a single frame."""

    frame_index: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    visible: bool = False
    predicted: bool = False
    box: tuple[float, float, float, float] | None = None

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class ShuttleTracker:
    """YOLO based shuttlecock detector with gating and Kalman smoothing."""

    model: object = None
    imgsz: int = 640
    conf: float = 0.18
    iou: float = 0.45
    max_box_area_ratio: float = 0.004
    max_aspect_ratio: float = 4.0
    max_jump_pixels: float = 220.0
    prediction_gate_pixels: float = 260.0
    max_missing_frames: int = 5
    reacquire_after: int = 2
    max_static_frames: int = 12
    static_speed: float = 1.2
    use_roi: bool = True

    history: list[BallState] = field(default_factory=list)
    kalman: ConstantVelocityKalman = field(default_factory=ConstantVelocityKalman)
    last_position: tuple[float, float] | None = None
    last_confidence: float = 0.0
    missing_frames: int = 0
    static_frames: int = 0
    _initialised: bool = False

    # --------------------------------------------------------------- detection
    def _extract_candidates(self, results, frame_shape, roi_polygon, reference_area: int | None = None) -> list[dict]:
        if results is None or len(results) == 0:
            return []
        boxes = results.boxes
        if boxes is None or boxes.xywh is None or len(boxes.xywh) == 0:
            return []

        height, width = frame_shape[:2]
        frame_area = reference_area or max(1, height * width)
        xywh = boxes.xywh.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))

        candidates: list[dict] = []
        for (cx, cy, bw, bh), confidence in zip(xywh, confs):
            bw, bh = float(bw), float(bh)
            if bw <= 0 or bh <= 0:
                continue
            area_ratio = (bw * bh) / frame_area
            aspect = max(bw / bh, bh / bw)
            if area_ratio > self.max_box_area_ratio or aspect > self.max_aspect_ratio:
                continue
            point = (float(cx), float(cy))
            if roi_polygon is not None and not _point_in_polygon(point, roi_polygon):
                continue
            candidates.append(
                {
                    "point": point,
                    "confidence": float(confidence),
                    "box": (float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)),
                }
            )
        return candidates

    def _select_candidate(self, candidates: list[dict]) -> dict | None:
        """Pick the candidate that best matches the current track.

        The gate is measured from the Kalman prediction so that a fast
        shuttlecock is not rejected just because it moved far in one frame.
        When the track has been lost for a couple of frames the strongest
        candidate is accepted instead, which allows re-acquisition.
        """
        if not candidates:
            return None

        best = max(candidates, key=lambda item: item["confidence"])

        if self.last_position is None:
            return best

        if self._initialised:
            predicted = self.kalman.state
            reference = (float(predicted[0]), float(predicted[1]))
            speed = float(np.hypot(predicted[2], predicted[3]))
        else:
            reference = self.last_position
            speed = 0.0

        gate = max(self.max_jump_pixels, 2.5 * speed + 60.0)
        lx, ly = reference
        within = []
        for candidate in candidates:
            distance = float(np.hypot(candidate["point"][0] - lx, candidate["point"][1] - ly))
            if distance <= gate:
                within.append((distance - candidate["confidence"] * 40.0, candidate))
        if within:
            within.sort(key=lambda item: item[0])
            return within[0][1]

        # Nothing close enough: re-acquire when the track was already lost.
        if self.missing_frames >= self.reacquire_after:
            return best
        return None

    # ------------------------------------------------------------------ update
    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        roi_polygon: list[tuple[float, float]] | None = None,
        detect: bool = True,
        reference_area: int | None = None,
        crop_offset: tuple[int, int] = (0, 0),
    ) -> BallState:
        candidates: list[dict] = []
        if detect and self.model is not None:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
            candidates = self._extract_candidates(
                results[0] if results else None, frame.shape, roi_polygon, reference_area
            )
            if crop_offset != (0, 0):
                for candidate in candidates:
                    candidate["point"] = (
                        candidate["point"][0] + crop_offset[0],
                        candidate["point"][1] + crop_offset[1],
                    )
                    bx, by, bw, bh = candidate["box"]
                    candidate["box"] = (bx + crop_offset[0], by + crop_offset[1], bw, bh)

        if self._initialised:
            self.kalman.predict()

        selected = self._select_candidate(candidates)

        if selected is not None:
            x, y = selected["point"]
            reacquired = self.missing_frames >= self.reacquire_after
            if not self._initialised:
                self.kalman.initialise(x, y, frame_index)
            else:
                self.kalman.correct(x, y)
                if reacquired:
                    self.kalman.reset_velocity()
            self.kalman.frame_index = frame_index
            state = self.kalman.state

            # A shuttlecock is always moving during a rally.  A detection that
            # stays put for a long time is a false positive (a light, a
            # spectator, a piece of clothing); drop the track so that the
            # tracker can pick the real shuttlecock up again.
            if float(np.hypot(state[2], state[3])) <= self.static_speed:
                self.static_frames += 1
            else:
                self.static_frames = 0
            if self.static_frames > self.max_static_frames:
                self.static_frames = 0
                self._initialised = False
                self.last_position = None
                self.missing_frames = 0
                self.kalman = ConstantVelocityKalman()

            self.last_position = (float(state[0]), float(state[1]))
            self.last_confidence = selected["confidence"]
            self.missing_frames = 0
            self._initialised = True
            ball = BallState(
                frame_index=frame_index,
                x=float(state[0]),
                y=float(state[1]),
                vx=float(state[2]),
                vy=float(state[3]),
                confidence=selected["confidence"],
                visible=True,
                predicted=False,
                box=selected["box"],
            )
        else:
            self.missing_frames += 1
            if self._initialised and self.missing_frames <= self.max_missing_frames:
                state = self.kalman.predict()
                self.last_position = (float(state[0]), float(state[1]))
                ball = BallState(
                    frame_index=frame_index,
                    x=float(state[0]),
                    y=float(state[1]),
                    vx=float(state[2]),
                    vy=float(state[3]),
                    confidence=self.last_confidence * 0.5,
                    visible=False,
                    predicted=True,
                )
            else:
                if self._initialised:
                    self._initialised = False
                    self.kalman = ConstantVelocityKalman()
                self.last_position = None
                self.last_confidence = 0.0
                ball = BallState(frame_index=frame_index, x=0.0, y=0.0, visible=False, predicted=False)

        self.history.append(ball)
        return ball


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            denom = (y2 - y1)
            if abs(denom) > 1e-9:
                x_intersect = (x2 - x1) * (y - y1) / denom + x1
                if x < x_intersect:
                    inside = not inside
    return inside

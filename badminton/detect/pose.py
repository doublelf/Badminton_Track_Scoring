"""Optional pose estimation used to attribute shuttlecock hits to a player."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEYPOINT_LEFT_WRIST = 15
KEYPOINT_RIGHT_WRIST = 16
KEYPOINT_LEFT_ELBOW = 13
KEYPOINT_RIGHT_ELBOW = 14
KEYPOINT_LEFT_SHOULDER = 11
KEYPOINT_RIGHT_SHOULDER = 12


@dataclass
class PoseObservation:
    """Pose of one player: wrists, elbows and an estimated racket head."""

    track_id: int
    bbox: tuple[float, float, float, float]
    wrists: dict[str, tuple[float, float]]
    elbows: dict[str, tuple[float, float]]
    racket_heads: dict[str, tuple[float, float]]

    def nearest_racket(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """Closest estimated racket head to ``point`` (image pixels)."""
        if not self.racket_heads:
            return None
        best = None
        best_distance = None
        for head in self.racket_heads.values():
            distance = float(np.hypot(head[0] - point[0], head[1] - point[1]))
            if best_distance is None or distance < best_distance:
                best, best_distance = head, distance
        return best


def _match_track(bbox: tuple[float, float, float, float], tracks: list) -> int:
    """Find the tracked player that corresponds to a pose detection box."""
    if not tracks:
        return -1
    px1, py1, px2, py2 = bbox
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    best = -1
    best_area = 0.0
    for track in tracks:
        tx1, ty1, tx2, ty2 = track.bbox
        if tx1 <= pcx <= tx2 and ty1 <= pcy <= ty2:
            area = (tx2 - tx1) * (ty2 - ty1)
            if area > best_area:
                best_area = area
                best = track.track_id
    if best >= 0:
        return best
    # Fall back to the closest box centre.
    best_distance = None
    for track in tracks:
        tx1, ty1, tx2, ty2 = track.bbox
        distance = float(np.hypot((tx1 + tx2) / 2.0 - pcx, (ty1 + ty2) / 2.0 - pcy))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = track.track_id
    return best


class PoseDetector:
    """YOLO pose model wrapper producing racket-head estimates."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.3,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def update(self, frame: np.ndarray, tracks: list | None = None) -> list[PoseObservation]:
        """Detect poses and attach them to the tracked players.

        The pose model is a plain detector, so it does not know the ByteTrack
        ids.  The observations are matched to the players by their bounding
        boxes instead.
        """
        if not tracks:
            return []

        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        keypoints = getattr(results, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return []

        points = keypoints.xy.detach().cpu().numpy()
        boxes = getattr(results, "boxes", None)
        box_values = boxes.xyxy.detach().cpu().numpy() if boxes is not None and boxes.xyxy is not None else None

        observations: list[PoseObservation] = []
        for index, pose in enumerate(points):
            bbox = tuple(float(v) for v in box_values[index]) if box_values is not None else (0.0, 0.0, 0.0, 0.0)
            track_id = _match_track(bbox, tracks)

            wrists: dict[str, tuple[float, float]] = {}
            elbows: dict[str, tuple[float, float]] = {}
            for name, wrist_index, elbow_index in (
                ("left", KEYPOINT_LEFT_WRIST, KEYPOINT_LEFT_ELBOW),
                ("right", KEYPOINT_RIGHT_WRIST, KEYPOINT_RIGHT_ELBOW),
            ):
                wrist = pose[wrist_index]
                elbow = pose[elbow_index]
                if wrist[0] > 0 and wrist[1] > 0:
                    wrists[name] = (float(wrist[0]), float(wrist[1]))
                if elbow[0] > 0 and elbow[1] > 0:
                    elbows[name] = (float(elbow[0]), float(elbow[1]))

            racket_heads = self._estimate_racket_heads(wrists, elbows, bbox)
            observations.append(
                PoseObservation(
                    track_id=track_id,
                    bbox=bbox,  # type: ignore[arg-type]
                    wrists=wrists,
                    elbows=elbows,
                    racket_heads=racket_heads,
                )
            )
        return observations

    @staticmethod
    def _estimate_racket_heads(
        wrists: dict[str, tuple[float, float]],
        elbows: dict[str, tuple[float, float]],
        bbox: tuple[float, float, float, float],
    ) -> dict[str, tuple[float, float]]:
        """Extend the forearm beyond the wrist to approximate the racket head."""
        if not wrists:
            return {}
        height_px = max(1.0, bbox[3] - bbox[1])
        metres_per_pixel = 1.75 / height_px
        racket_length_m = 0.67
        extension_px = racket_length_m / max(metres_per_pixel, 1e-6)

        heads: dict[str, tuple[float, float]] = {}
        for name, wrist in wrists.items():
            elbow = elbows.get(name)
            if elbow is None:
                heads[name] = wrist
                continue
            dx = wrist[0] - elbow[0]
            dy = wrist[1] - elbow[1]
            norm = float(np.hypot(dx, dy))
            if norm < 1e-3:
                heads[name] = wrist
                continue
            heads[name] = (wrist[0] + dx / norm * extension_px, wrist[1] + dy / norm * extension_px)
        return heads

"""Player detection and tracking (Ultralytics YOLO + ByteTrack)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..court.mapper import CourtMapper


@dataclass
class PlayerTrack:
    """One tracked player in a frame."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float = 0.0
    foot_point: tuple[float, float] = (0.0, 0.0)
    court_point: tuple[float, float] | None = None
    half: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def height_px(self) -> float:
        _x1, y1, _x2, y2 = self.bbox
        return max(1.0, y2 - y1)


class PlayerDetector:
    """Wraps ``YOLO(...).track(...)`` and projects players onto the court."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.device = device
        self._warmup_done = False

    def update(self, frame: np.ndarray, mapper: CourtMapper | None = None) -> list[PlayerTrack]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        tracks: list[PlayerTrack] = []
        boxes = getattr(results, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return tracks

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = boxes.id.detach().cpu().numpy() if getattr(boxes, "id", None) is not None else None

        for index, (box, confidence) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = [float(value) for value in box]
            track_id = int(ids[index]) if ids is not None else index
            foot = ((x1 + x2) / 2.0, y2)
            track = PlayerTrack(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence),
                foot_point=foot,
            )
            if mapper is not None:
                court_point = mapper.image_to_court(foot)
                track.court_point = court_point
                track.half = mapper.half_of(court_point)
            tracks.append(track)
        return tracks

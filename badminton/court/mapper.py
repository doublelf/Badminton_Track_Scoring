"""Image <-> court homography mapping and court overlay rendering."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .reference import CourtSpec


class CourtMapper:
    """Maps between pixel coordinates and metric court coordinates."""

    def __init__(
        self,
        image_corners: list[tuple[float, float]],
        court_corners: list[tuple[float, float]] | None = None,
        spec: CourtSpec | None = None,
    ) -> None:
        self.spec = spec or CourtSpec()
        self.image_corners = np.array(image_corners, dtype=np.float32)
        court_corners = court_corners or self.spec.corner_order
        self.court_corners = np.array(court_corners, dtype=np.float32)
        if self.image_corners.shape != (4, 2) or self.court_corners.shape != (4, 2):
            raise ValueError("court mapping requires exactly 4 corners")

        self.matrix = cv2.getPerspectiveTransform(self.image_corners, self.court_corners)
        self.inverse_matrix = cv2.getPerspectiveTransform(self.court_corners, self.image_corners)

    # ------------------------------------------------------------------ mapping
    def image_to_court(self, point: tuple[float, float]) -> tuple[float, float]:
        src = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.matrix).reshape(-1, 2)[0]
        return float(dst[0]), float(dst[1])

    def image_to_court_many(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(src, self.matrix).reshape(-1, 2)

    def court_to_image(self, point: tuple[float, float]) -> tuple[float, float]:
        src = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.inverse_matrix).reshape(-1, 2)[0]
        return float(dst[0]), float(dst[1])

    def court_to_image_many(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(src, self.inverse_matrix).reshape(-1, 2)

    # ------------------------------------------------------------------- query
    def is_in_bounds(self, court_point: tuple[float, float], margin: float = 0.0, singles: bool = True) -> bool:
        return self.spec.contains(court_point, margin=margin, singles=singles)

    def half_of(self, court_point: tuple[float, float]) -> str:
        return self.spec.half(court_point)

    # ------------------------------------------------------------------ render
    def draw_court_overlay(self, image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        overlay = image.copy()
        for start, end, weight in self.spec.lines():
            p1 = self.court_to_image(start)
            p2 = self.court_to_image(end)
            if weight >= 1.3:
                color, thickness = (255, 255, 255), 3
            elif weight <= 0.4:
                color, thickness = (60, 200, 255), 3
            else:
                color, thickness = (220, 220, 220), 2
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)
        cv2.polylines(overlay, [self.image_corners.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # ------------------------------------------------------------------- store
    def to_dict(self) -> dict:
        return {
            "image_corners": [[float(x), float(y)] for x, y in self.image_corners],
            "court_corners": [[float(x), float(y)] for x, y in self.court_corners],
            "spec": {
                "length": self.spec.length,
                "width": self.spec.width,
                "singles_margin": self.spec.singles_margin,
                "service_line_from_net": self.spec.service_line_from_net,
                "back_service_offset": self.spec.back_service_offset,
            },
        }

    def save(self, path: str | Path) -> None:
        payload = self.to_dict()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CourtMapper":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        spec = CourtSpec(**payload.get("spec", {}))
        return cls(
            payload["image_corners"],
            payload.get("court_corners"),
            spec,
        )

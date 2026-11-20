"""Court geometry, homography and auto-detection tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from badminton.court.detector import auto_detect_court_corners
from badminton.court.mapper import CourtMapper
from badminton.court.reference import CourtSpec


def render_synthetic_court(width: int = 1280, height: int = 720) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Render a perspective view of a green badminton court with white lines."""
    image = np.full((height, width, 3), (38, 34, 30), dtype=np.uint8)
    spec = CourtSpec()

    far_left = (int(width * 0.30), int(height * 0.14))
    far_right = (int(width * 0.70), int(height * 0.14))
    near_right = (int(width * 0.93), int(height * 0.95))
    near_left = (int(width * 0.07), int(height * 0.95))
    corners = [far_left, far_right, near_right, near_left]

    court_poly = np.array(corners, dtype=np.int32)
    cv2.fillPoly(image, [court_poly], (70, 150, 70))

    matrix = cv2.getPerspectiveTransform(
        np.array(spec.corner_order, dtype=np.float32),
        np.array(corners, dtype=np.float32),
    )

    def to_image(point):
        src = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(src, matrix).reshape(-1, 2)[0]

    for start, end, _weight in spec.lines():
        p1, p2 = to_image(start), to_image(end)
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (245, 245, 245), 3, cv2.LINE_AA)
    return image, corners


def test_court_reference_lines_and_bounds():
    spec = CourtSpec()
    assert spec.length == pytest.approx(13.4)
    assert spec.width == pytest.approx(6.1)
    assert spec.net_x == pytest.approx(6.7)
    assert spec.singles_y_min == pytest.approx(0.46)
    assert spec.singles_y_max == pytest.approx(5.64)
    assert len(spec.lines()) == 13

    assert spec.contains((6.0, 3.05))
    assert not spec.contains((6.0, 0.2))
    assert not spec.contains((14.0, 3.05))
    assert spec.half((1.0, 3.0)) == "A"
    assert spec.half((12.0, 3.0)) == "B"


def test_mapper_roundtrip():
    _image, corners = render_synthetic_court()
    mapper = CourtMapper(corners)
    for court_point in [(0.0, 0.0), (6.7, 3.05), (13.4, 6.1), (3.0, 1.2)]:
        pixel = mapper.court_to_image(court_point)
        recovered = mapper.image_to_court(pixel)
        assert recovered[0] == pytest.approx(court_point[0], abs=1e-2)
        assert recovered[1] == pytest.approx(court_point[1], abs=1e-2)


def test_mapper_in_bounds_uses_singles_lines():
    _image, corners = render_synthetic_court()
    mapper = CourtMapper(corners)
    inside = mapper.image_to_court(mapper.court_to_image((5.0, 3.05)))
    outside = mapper.image_to_court(mapper.court_to_image((5.0, 0.15)))
    assert mapper.is_in_bounds(inside)
    assert not mapper.is_in_bounds(outside)


def test_auto_detection_recovers_synthetic_court():
    image, corners = render_synthetic_court()
    detected, _mask, _debug = auto_detect_court_corners(image)
    assert detected is not None, "auto detection failed on a clean synthetic court"
    for detected_point, expected_point in zip(detected, corners):
        assert np.hypot(
            detected_point[0] - expected_point[0], detected_point[1] - expected_point[1]
        ) < 40.0

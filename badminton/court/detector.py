"""Automatic badminton court detection.

The approach follows the open-source ``Good-Badminton`` project: white court
lines are extracted with a colour mask, Hough segments are grouped into
length-parallel and width-parallel families, every plausible quad is scored
against the metric court model projected into the image, and the best quad is
finally refined by a small local search.

Coordinate convention (see :mod:`badminton.court.reference`): ``x`` is the
length axis, ``y`` the width axis.  Image corners are ordered
``[far-left, far-right, near-right, near-left]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from .reference import CourtSpec


# --------------------------------------------------------------------- helpers
def _line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return float(angle % 180)


def _line_intersection(line_a, line_b) -> np.ndarray | None:
    x1, y1, x2, y2 = line_a
    x3, y3, x4, y4 = line_b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float32)


def _inside_image(point, width: int, height: int, margin: int = 0) -> bool:
    return margin <= point[0] <= width - 1 - margin and margin <= point[1] <= height - 1 - margin


def _inside_frame_loose(point, width: int, height: int, slack: int = 12) -> bool:
    """Allow corners that touch (or slightly exceed) the frame border."""
    return -slack <= point[0] <= width - 1 + slack and -slack <= point[1] <= height - 1 + slack


def _polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def _is_convex_quad(points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=np.float32)
    return len(cv2.convexHull(pts)) == 4 and bool(cv2.isContourConvex(pts.astype(np.int32)))


def _project(matrix: np.ndarray, points) -> np.ndarray:
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


# ------------------------------------------------------------------ line masks
def _green_court_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = (((h >= 35) & (h <= 95) & (s >= 30) & (v >= 45)).astype(np.uint8)) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(green)
    if count > 1:
        image_area = image.shape[0] * image.shape[1]
        candidates = [idx for idx in range(1, count) if stats[idx, cv2.CC_STAT_AREA] > image_area * 0.08]
        if candidates:
            court_idx = max(candidates, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
            green = ((labels == court_idx).astype(np.uint8)) * 255
    return green


def _blue_court_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blue = (((h >= 95) & (h <= 135) & (s >= 45) & (v >= 45)).astype(np.uint8)) * 255
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(blue)
    if count > 1:
        image_area = image.shape[0] * image.shape[1]
        candidates = [idx for idx in range(1, count) if stats[idx, cv2.CC_STAT_AREA] > image_area * 0.08]
        if candidates:
            court_idx = max(candidates, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
            blue = ((labels == court_idx).astype(np.uint8)) * 255
    return blue


def _court_surface_mask(image: np.ndarray) -> np.ndarray:
    """Largest uniform court surface (green or blue) in the frame."""
    best = None
    for mask in (_green_court_mask(image), _blue_court_mask(image)):
        if cv2.countNonZero(mask) == 0:
            continue
        if best is None or cv2.countNonZero(mask) > cv2.countNonZero(best):
            best = mask
    return best if best is not None else np.zeros(image.shape[:2], dtype=np.uint8)


def order_court_corners(corners) -> list[np.ndarray]:
    """Order four quad corners as ``[far-left, far-right, near-right, near-left]``.

    The two baselines are the longer edges of the quad.  The baseline that
    appears higher in the image becomes the far side (``x = 0``); the choice
    only fixes the arbitrary "A/B" labelling of the two halves.
    """
    points = [np.array(point, dtype=np.float32) for point in corners]
    if len(points) != 4:
        raise ValueError("expected four corners")

    centre = np.mean(points, axis=0)
    angles = [float(np.arctan2(p[1] - centre[1], p[0] - centre[0])) for p in points]
    ordered = [point for _angle, point in sorted(zip(angles, points), key=lambda item: item[0])]

    lengths = [
        float(np.linalg.norm(ordered[(index + 1) % 4] - ordered[index])) for index in range(4)
    ]
    longest = int(np.argmax(lengths))
    baseline_a = ordered[longest]
    baseline_b = ordered[(longest + 1) % 4]
    other = [point for point in ordered if point is not baseline_a and point is not baseline_b]

    baseline_length = max(lengths[longest], 1e-6)
    delta_y = abs(float(baseline_a[1] + baseline_b[1]) / 2.0 - float(other[0][1] + other[1][1]) / 2.0)
    if delta_y > baseline_length * 0.12:
        far_is_a = (float(baseline_a[1]) + float(baseline_b[1])) < (float(other[0][1]) + float(other[1][1]))
    else:
        far_is_a = (float(baseline_a[0]) + float(baseline_b[0])) < (float(other[0][0]) + float(other[1][0]))

    far_pair = sorted([baseline_a, baseline_b] if far_is_a else other, key=lambda p: (p[0], p[1]))
    near_pair = sorted([baseline_a, baseline_b] if not far_is_a else other, key=lambda p: (p[0], p[1]))
    return [far_pair[0], far_pair[1], near_pair[1], near_pair[0]]


def detect_court_quad_from_surface(image: np.ndarray) -> list[np.ndarray] | None:
    """Detect the court quad from the coloured playing surface."""
    mask = _court_surface_mask(image)
    if cv2.countNonZero(mask) == 0:
        return None

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < image.shape[0] * image.shape[1] * 0.05:
        return None

    perimeter = cv2.arcLength(contour, True)
    quad = None
    for epsilon in (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.int32)):
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect).astype(np.float32)
    return order_court_corners(quad)



def build_court_line_mask(image: np.ndarray) -> np.ndarray:
    """Mask of the white lines painted on the court surface."""
    green = _green_court_mask(image)
    court_roi = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)), iterations=1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s, v = cv2.split(hsv)[1], cv2.split(hsv)[2]
    white = (((s <= 95) & (v >= 135)).astype(np.uint8)) * 255
    white = cv2.bitwise_and(white, court_roi)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.bitwise_and(edges, court_roi)

    merged = cv2.bitwise_or(white, edges)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return merged


def build_reference_line_mask(image: np.ndarray) -> np.ndarray:
    """Mask of white/yellow lines used for the distance-transform support score."""
    green = _green_court_mask(image)
    court_roi = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = (((s <= 90) & (v >= 145)).astype(np.uint8)) * 255
    yellow = (((h >= 12) & (h <= 42) & (s >= 45) & (v >= 120)).astype(np.uint8)) * 255
    lines = cv2.bitwise_and(cv2.bitwise_or(white, yellow), court_roi)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return lines


def prepare_line_support(line_mask: np.ndarray | None, image_shape: tuple[int, int]) -> dict | None:
    if line_mask is None:
        return None
    height, width = image_shape[:2]
    support = (line_mask > 0).astype(np.uint8) * 255
    if support.shape[:2] != (height, width):
        support = cv2.resize(support, (width, height), interpolation=cv2.INTER_NEAREST)
    distance_map = cv2.distanceTransform(255 - support, cv2.DIST_L2, 3)
    return {"distance_map": distance_map, "tolerance": max(5.0, min(width, height) * 0.013)}


# ------------------------------------------------------------- segment finding
def _dedupe_lines(lines: list[dict], orientation: str, max_count: int) -> list[dict]:
    selected: list[dict] = []
    for line in sorted(lines, key=lambda item: item["length"], reverse=True):
        duplicate = False
        for existing in selected:
            if orientation == "horizontal":
                same_band = abs(line["mid"][1] - existing["mid"][1]) < 18
                same_angle = abs(line["angle"] - existing["angle"]) < 8
            else:
                same_band = abs(line["mid"][0] - existing["mid"][0]) < 22
                same_angle = abs(line["angle"] - existing["angle"]) < 10
            if same_band and same_angle:
                duplicate = True
                break
        if not duplicate:
            selected.append(line)
        if len(selected) >= max_count:
            break
    return selected


def detect_court_line_segments(image: np.ndarray) -> tuple[list[dict], list[dict], np.ndarray]:
    """Return length-parallel lines, width-parallel lines and the raw line mask."""
    height, width = image.shape[:2]
    mask = build_court_line_mask(image)
    min_line_length = max(46, int(min(width, height) * 0.10))
    max_gap = max(14, int(min(width, height) * 0.045))

    raw_lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )
    if raw_lines is None:
        return [], [], mask

    edge_margin = max(10, int(min(width, height) * 0.02))
    horizontal: list[dict] = []
    side: list[dict] = []
    for raw_line in np.asarray(raw_lines, dtype=np.float32).reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in raw_line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_line_length:
            continue
        if (
            min(x1, x2) <= edge_margin
            or max(x1, x2) >= width - 1 - edge_margin
            or min(y1, y2) <= edge_margin
            or max(y1, y2) >= height - 1 - edge_margin
        ):
            continue

        angle = _line_angle(x1, y1, x2, y2)
        segment = {
            "points": (x1, y1, x2, y2),
            "length": length,
            "mid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            "angle": angle,
        }
        if min(angle, 180 - angle) <= 18:
            horizontal.append(segment)
        elif 40 <= angle <= 140:
            side.append(segment)

    return _dedupe_lines(horizontal, "horizontal", 14), _dedupe_lines(side, "side", 18), mask


# ------------------------------------------------------------------- scoring
def _court_matrix(image_corners, spec: CourtSpec) -> np.ndarray:
    court_points = np.array(spec.corner_order, dtype=np.float32)
    return cv2.getPerspectiveTransform(court_points, np.array(image_corners, dtype=np.float32))


def _alignment_score(image_corners, line_mask: np.ndarray, spec: CourtSpec) -> float:
    """Overlap between the projected metric court model and the detected line mask."""
    height, width = line_mask.shape[:2]
    matrix = _court_matrix(image_corners, spec)
    expected = np.zeros((height, width), dtype=np.uint8)

    for length_pos in (0.0, spec.back_service_offset, spec.net_x - spec.service_line_from_net,
                       spec.net_x, spec.net_x + spec.service_line_from_net,
                       spec.length - spec.back_service_offset, spec.length):
        p1, p2 = _project(matrix, [(length_pos, 0.0), (length_pos, spec.width)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)

    for width_pos in (0.0, spec.singles_margin, spec.center_y, spec.width - spec.singles_margin, spec.width):
        p1, p2 = _project(matrix, [(0.0, width_pos), (spec.length, width_pos)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)

    for center_x in (0.0, spec.net_x - spec.service_line_from_net):
        p1, p2 = _project(matrix, [(center_x, spec.center_y), (spec.length, spec.center_y)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)

    expected_pixels = cv2.countNonZero(expected)
    if expected_pixels == 0:
        return 0.0

    dilated = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    overlap = cv2.bitwise_and(expected, dilated)
    return float(cv2.countNonZero(overlap) / expected_pixels)


def _length_pattern_score(image_corners, horizontal_lines: list[dict], image_shape, spec: CourtSpec) -> float:
    """Weighted match between detected cross-court lines and the expected line positions."""
    if not horizontal_lines:
        return 0.0
    height, _width = image_shape[:2]
    matrix = _court_matrix(image_corners, spec)
    expected_positions = [
        0.0,
        spec.back_service_offset,
        spec.net_x - spec.service_line_from_net,
        spec.net_x,
        spec.net_x + spec.service_line_from_net,
        spec.length - spec.back_service_offset,
        spec.length,
    ]
    weights = [5.0, 4.0, 1.6, 0.8, 1.6, 4.0, 5.0]

    detected = [float(line["mid"][1]) for line in horizontal_lines]

    tolerance = max(12.0, height * 0.022)
    weighted_score = 0.0
    total_weight = 0.0
    for position, weight in zip(expected_positions, weights):
        p1, p2 = _project(matrix, [(position, 0.0), (position, spec.width)])
        expected_y = float((p1[1] + p2[1]) / 2.0)
        nearest = min(abs(y - expected_y) for y in detected)
        weighted_score += max(0.0, 1.0 - nearest / tolerance) * weight
        total_weight += weight
    return weighted_score / total_weight if total_weight else 0.0


def _reference_support_score(image_corners, line_support: dict | None, image_shape, spec: CourtSpec) -> float:
    """Distance-transform support of the reference court lines."""
    if line_support is None:
        return 0.0
    height, width = image_shape[:2]
    matrix = _court_matrix(image_corners, spec)
    distance_map = line_support["distance_map"]
    tolerance = line_support["tolerance"]

    scores = []
    for start, end, weight in spec.lines():
        p1, p2 = _project(matrix, [start, end])
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if length < 4.0:
            continue
        sample_count = max(12, int(length / 5.0))
        xs = np.linspace(float(p1[0]), float(p2[0]), sample_count)
        ys = np.linspace(float(p1[1]), float(p2[1]), sample_count)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.mean(valid) < 0.35:
            continue
        xs = np.clip(np.rint(xs[valid]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.rint(ys[valid]).astype(np.int32), 0, height - 1)
        distances = distance_map[ys, xs]
        line_score = 1.0 - np.clip(distances / tolerance, 0.0, 1.0)
        coverage = np.mean(distances <= tolerance)
        scores.append((0.65 * float(np.mean(line_score)) + 0.35 * float(coverage)) * weight)
    return float(np.mean(scores)) if scores else 0.0


def _geometric_prior_score(corners, image_shape, spec: CourtSpec) -> float:
    """Soft priors describing how a plausible broadcast court framing looks."""
    height, width = image_shape[:2]
    points = np.asarray(corners, dtype=np.float32)
    far_left, far_right, near_right, near_left = points

    far_width = float(np.linalg.norm(far_right - far_left))
    near_width = float(np.linalg.norm(near_right - near_left))
    far_height = float(np.linalg.norm(near_left - far_left))
    near_height = float(np.linalg.norm(near_right - far_right))
    if min(far_width, near_width, far_height, near_height) <= 1:
        return 0.0

    area = _polygon_area(points)
    image_area = width * height
    score = min(area / (image_area * 0.40), 1.0) * 40.0
    score += min((far_height + near_height) / 2.0 / (height * 0.60), 1.0) * 20.0

    ratio = far_width / max(near_width, 1e-6)
    score += (1.0 - min(abs(ratio - 0.85) / 0.45, 1.0)) * 14.0

    center = points.mean(axis=0)
    score += (1.0 - min(abs(center[0] - width * 0.5) / (width * 0.35), 1.0)) * 10.0
    score += (1.0 - min(abs(center[1] - height * 0.60) / (height * 0.30), 1.0)) * 8.0
    return float(score)


def _score_court_quad(corners, image_shape, line_mask, horizontal_lines, spec, line_support) -> tuple[float, dict] | None:
    height, width = image_shape[:2]
    points = np.asarray(corners, dtype=np.float32)
    if not all(_inside_frame_loose(point, width, height) for point in points):
        return None
    if not _is_convex_quad(points):
        return None

    far_left, far_right, near_right, near_left = points
    far_width = float(np.linalg.norm(far_right - far_left))
    near_width = float(np.linalg.norm(near_right - near_left))
    far_height = float(np.linalg.norm(near_left - far_left))
    near_height = float(np.linalg.norm(near_right - far_right))
    avg_height = (far_height + near_height) / 2.0

    if min(far_width, near_width, avg_height) <= 1:
        return None
    area = _polygon_area(points)
    if area < (width * height) * 0.06:
        return None
    if avg_height < height * 0.18:
        return None
    ratio = far_width / max(near_width, 1e-6)
    if not (0.30 <= ratio <= 1.6):
        return None

    far_y = (far_left[1] + far_right[1]) / 2.0
    near_y = (near_left[1] + near_right[1]) / 2.0
    if far_y >= near_y:
        return None
    if near_y < height * 0.45:
        return None

    prior = _geometric_prior_score(points, image_shape, spec)
    alignment = _alignment_score(points, line_mask, spec)
    pattern = _length_pattern_score(points, horizontal_lines, image_shape, spec)
    reference = _reference_support_score(points, line_support, image_shape, spec)

    total = prior + alignment * 120.0 + pattern * 90.0 + reference * 60.0
    details = {
        "alignment": round(alignment, 4),
        "pattern": round(pattern, 4),
        "reference": round(reference, 4),
        "prior": round(prior, 2),
    }
    return float(total), details


# ------------------------------------------------------------------ detection
def auto_detect_court_corners(image: np.ndarray, spec: CourtSpec | None = None) -> tuple[list[tuple[int, int]] | None, np.ndarray, dict]:
    """Detect the court quad in ``image``.

    Two independent detectors feed a common scoring stage: the coloured
    playing surface (fast, robust on broadcast footage) and Hough line
    segments (fallback for unusual floors).  The winning quad is then snapped
    onto the painted lines.

    Returns ``(corners, line_mask, debug)``; ``corners`` is ``None`` when no
    plausible court framing was found.
    """
    spec = spec or CourtSpec()
    horizontal_lines, side_lines, mask = detect_court_line_segments(image)
    reference_mask = build_reference_line_mask(image)
    line_support = prepare_line_support(reference_mask, image.shape)

    debug: dict = {"horizontal": horizontal_lines, "side": side_lines, "score": None, "details": None}
    height, width = image.shape[:2]

    scored_candidates: list[tuple[float, dict, list[np.ndarray]]] = []

    # ---- candidate 1: coloured playing surface -----------------------------
    surface_quad = detect_court_quad_from_surface(image)
    if surface_quad is not None:
        scored = _score_court_quad(surface_quad, image.shape, mask, horizontal_lines, spec, line_support)
        if scored is not None:
            scored_candidates.append((scored[0], scored[1], surface_quad))

    # ---- candidate 2+: quads built from Hough line intersections -----------
    if len(horizontal_lines) >= 2 and len(side_lines) >= 2:
        horizontals = sorted(horizontal_lines, key=lambda item: item["mid"][1])
        sides = sorted(side_lines, key=lambda item: item["mid"][0])
        for top_idx, top_line in enumerate(horizontals):
            for bottom_line in horizontals[top_idx + 1:]:
                if bottom_line["mid"][1] - top_line["mid"][1] < height * 0.12:
                    continue
                for left_idx, left_line in enumerate(sides):
                    for right_line in sides[left_idx + 1:]:
                        if right_line["mid"][0] - left_line["mid"][0] < width * 0.12:
                            continue
                        intersections = [
                            _line_intersection(top_line["points"], left_line["points"]),
                            _line_intersection(top_line["points"], right_line["points"]),
                            _line_intersection(bottom_line["points"], right_line["points"]),
                            _line_intersection(bottom_line["points"], left_line["points"]),
                        ]
                        if any(point is None for point in intersections):
                            continue
                        quad = [np.array(point, dtype=np.float32) for point in intersections]
                        scored = _score_court_quad(quad, image.shape, mask, horizontal_lines, spec, line_support)
                        if scored is None:
                            continue
                        scored_candidates.append((scored[0], scored[1], quad))

    if not scored_candidates:
        return None, mask, debug

    # Re-rank every candidate by how well the projected court model actually
    # sits on the painted lines.  The geometric priors alone are not enough:
    # the inner (singles) rectangle is a perfect rectangle as well, but its
    # singles lines do not exist.
    ranked: list[tuple[float, dict, list[np.ndarray]]] = []
    for _score, details, quad in scored_candidates:
        cost = _quad_cost(quad, line_support, mask, image.shape, spec)
        if np.isfinite(cost):
            ranked.append((cost, details, quad))
    if not ranked:
        return None, mask, debug
    ranked.sort(key=lambda item: item[0])

    best: tuple[float, list[np.ndarray], dict] | None = None
    for _cost, details, quad in ranked[:12]:
        refined = refine_court_corners(image, quad, spec)
        refined_cost = _quad_cost(refined, line_support, mask, image.shape, spec)
        if not np.isfinite(refined_cost) or refined_cost > _cost:
            refined, refined_cost = quad, _cost
        if best is None or refined_cost < best[0]:
            best = (refined_cost, refined, details)

    if best is None:
        return None, mask, debug

    best_cost, corners, best_details = best
    final = _score_court_quad(corners, image.shape, mask, horizontal_lines, spec, line_support)
    if final is not None:
        best_details = final[1]

    debug["score"] = best_cost
    debug["details"] = best_details
    debug["candidates"] = len(scored_candidates)
    return [(int(round(p[0])), int(round(p[1]))) for p in corners], mask, debug


def _line_fit_cost(image_corners, line_support: dict, image_shape, spec: CourtSpec) -> float:
    """Mean per-line distance between the projected court model and the paint.

    Using the mean of the *per line* distances (instead of one global mean)
    prevents a single badly aligned line from being hidden by the others.
    """
    if line_support is None:
        return float("inf")
    height, width = image_shape[:2]
    matrix = _court_matrix(image_corners, spec)
    distance_map = line_support["distance_map"]

    per_line: list[float] = []
    for start, end, _weight in spec.lines():
        p1, p2 = _project(matrix, [start, end])
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if length < 4.0:
            per_line.append(60.0)
            continue
        samples = max(14, int(length / 4.0))
        xs = np.linspace(float(p1[0]), float(p2[0]), samples)
        ys = np.linspace(float(p1[1]), float(p2[1]), samples)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.mean(valid) < 0.3:
            # A model line that falls outside the frame cannot be verified.
            per_line.append(60.0)
            continue
        xs = np.clip(np.rint(xs[valid]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.rint(ys[valid]).astype(np.int32), 0, height - 1)
        distances = distance_map[ys, xs]
        per_line.append(float(np.mean(np.minimum(distances, 60.0))))

    if not per_line:
        return float("inf")
    return float(np.mean(per_line))


def _quad_cost(quad, line_support: dict, line_mask: np.ndarray, image_shape, spec: CourtSpec) -> float:
    """Combined cost of a candidate quad: line fit plus model/mask mismatch."""
    points = [np.array(point, dtype=np.float32) for point in quad]
    fit = _line_fit_cost(points, line_support, image_shape, spec)
    if not np.isfinite(fit):
        return float("inf")
    alignment = _alignment_score(points, line_mask, spec)

    height, width = image_shape[:2]
    area_fraction = _polygon_area(np.asarray(points, dtype=np.float32)) / max(1, width * height)
    # A court that only covers a small part of the frame is a degenerate
    # solution: its model lines can accidentally fall onto other markings.
    penalty = max(0.0, 0.14 - area_fraction) * 900.0
    return fit - 25.0 * alignment + penalty


def refine_court_corners(image: np.ndarray, corners, spec: CourtSpec | None = None, iterations: int = 6) -> list[np.ndarray]:
    """Snap a detected quad onto the painted lines with a local search."""
    spec = spec or CourtSpec()
    line_mask = build_court_line_mask(image)
    reference_mask = build_reference_line_mask(image)
    line_support = prepare_line_support(reference_mask, image.shape)

    current = [np.array(point, dtype=np.float32) for point in corners]

    def objective(points) -> float:
        return _quad_cost(points, line_support, line_mask, image.shape, spec)

    best_score = objective(current)
    offsets = (0.0, -3.0, 3.0, -6.0, 6.0, -10.0, 10.0, -16.0, 16.0, -24.0, 24.0, -34.0, 34.0)

    for _ in range(iterations):
        improved = False
        for index in range(4):
            for offset in offsets:
                if offset == 0.0:
                    continue
                for axis in (0, 1):
                    candidate = [point.copy() for point in current]
                    candidate[index][axis] += offset
                    score = objective(candidate)
                    if score > best_score + 1e-4:
                        best_score = score
                        current = candidate
                        improved = True
        if not improved:
            break
    return current


def render_debug_preview(image: np.ndarray, corners=None, debug: dict | None = None) -> np.ndarray:
    preview = image.copy()
    if corners:
        points = np.array(corners, dtype=np.int32)
        cv2.polylines(preview, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
        for index, point in enumerate(corners, start=1):
            cv2.circle(preview, point, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(preview, str(index), (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    elif debug:
        for line in debug.get("horizontal", []):
            cv2.line(preview, line["points"][:2], line["points"][2:], (0, 220, 255), 2)
        for line in debug.get("side", []):
            cv2.line(preview, line["points"][:2], line["points"][2:], (255, 180, 0), 2)
    return preview

"""Measure how well the calibrated court matches the painted lines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.court.detector import (
    auto_detect_court_corners,
    build_reference_line_mask,
    prepare_line_support,
)
from badminton.court.mapper import CourtMapper
from badminton.court.reference import CourtSpec


def measure(corners, frame, spec: CourtSpec) -> dict:
    mapper = CourtMapper(corners, spec.corner_order, spec)
    mask = build_reference_line_mask(frame)
    support = prepare_line_support(mask, frame.shape)
    distance_map = support["distance_map"]
    tolerance = support["tolerance"]

    distances = []
    per_line = []
    for start, end, _weight in spec.lines():
        p1 = mapper.court_to_image(start)
        p2 = mapper.court_to_image(end)
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        samples = max(16, int(length / 4))
        xs = np.linspace(p1[0], p2[0], samples)
        ys = np.linspace(p1[1], p2[1], samples)
        valid = (xs >= 0) & (xs < frame.shape[1]) & (ys >= 0) & (ys < frame.shape[0])
        if np.mean(valid) < 0.35:
            per_line.append((start, end, None, float("nan")))
            continue
        xs = np.clip(np.rint(xs[valid]).astype(np.int32), 0, frame.shape[1] - 1)
        ys = np.clip(np.rint(ys[valid]).astype(np.int32), 0, frame.shape[0] - 1)
        line_distances = distance_map[ys, xs]
        per_line.append((start, end, float(np.mean(line_distances)), float(np.mean(line_distances <= tolerance))))
        distances.extend(line_distances.tolist())

    distances_arr = np.array(distances)
    return {
        "mean_distance_px": float(distances_arr.mean()) if len(distances_arr) else float("nan"),
        "median_distance_px": float(np.median(distances_arr)) if len(distances_arr) else float("nan"),
        "within_tolerance": float(np.mean(distances_arr <= tolerance)) if len(distances_arr) else 0.0,
        "samples": int(len(distances_arr)),
        "per_line": per_line,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("cannot read frame")
        return 1

    spec = CourtSpec()
    corners, _mask, debug = auto_detect_court_corners(frame, spec)
    print("detected corners:", corners)
    print("score:", round(debug.get("score", 0), 1), debug.get("details"))
    if corners:
        metrics = measure(corners, frame, spec)
        print(
            f"line fit: mean={metrics['mean_distance_px']:.1f}px  "
            f"median={metrics['median_distance_px']:.1f}px  "
            f"within_tol={metrics['within_tolerance']:.1%}  samples={metrics['samples']}"
        )
        print("per line:")
        for start, end, mean_distance, coverage in metrics["per_line"]:
            label = f"{tuple(round(v, 2) for v in start)}->{tuple(round(v, 2) for v in end)}"
            if mean_distance is None:
                print(f"   {label:<28} out of frame")
            else:
                print(f"   {label:<28} dist={mean_distance:6.1f}px  cover={coverage:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

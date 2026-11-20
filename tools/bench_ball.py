"""Benchmark shuttlecock detection: speed vs detection rate for ROI/imgsz combos."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from badminton.court.detector import auto_detect_court_corners


def crop_box(corners, frame_shape, up=0.45, side=0.15, down=0.05):
    height, width = frame_shape[:2]
    points = np.array(corners, dtype=np.float32)
    x1 = max(0, int(points[:, 0].min() - width * side))
    x2 = min(width - 1, int(points[:, 0].max() + width * side))
    top = points[:, 1].min()
    bottom = points[:, 1].max()
    y1 = max(0, int(top - (bottom - top) * up))
    y2 = min(height - 1, int(bottom + (bottom - top) * down))
    return x1, y1, x2, y2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="weights/yolo11s-ball.pt")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.18)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print("no frames")
        return 1

    corners, _mask, _debug = auto_detect_court_corners(frames[0])
    model = YOLO(args.model)

    for label, use_crop, imgsz in [
        ("full  @640", False, 640),
        ("crop  @640", True, 640),
        ("crop  @512", True, 512),
        ("crop  @448", True, 448),
        ("full  @512", False, 512),
    ]:
        hits = 0
        started = time.time()
        for frame in frames:
            if use_crop and corners:
                x1, y1, x2, y2 = crop_box(corners, frame.shape)
                region = frame[y1:y2, x1:x2]
            else:
                region = frame
            results = model.predict(region, imgsz=imgsz, conf=args.conf, verbose=False)[0]
            boxes = results.boxes
            if boxes is not None and len(boxes.xyxy):
                hits += 1
        elapsed = (time.time() - started) / len(frames) * 1000
        print(f"{label}: {elapsed:6.1f} ms/frame   detection rate {hits / len(frames):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

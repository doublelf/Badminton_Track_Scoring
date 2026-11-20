"""Trajectory quality statistics for a badminton video.

Reports the measurable properties of the shuttlecock track: how often the
shuttlecock was found, how fast it moved, how often the tracker had to
re-acquire it and how smooth the resulting trajectory was.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.app import BadmintonApp
from badminton.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    app = BadmintonApp(config=config, source=args.video, allow_interactive=False, progress=False)
    result = app.analyze()
    frames = result.frames
    if not frames:
        print("no frames")
        return 1

    visible = [record.ball for record in frames if record.ball.visible]
    predicted = [record.ball for record in frames if record.ball.predicted]
    missing = len(frames) - len(visible) - len(predicted)

    speeds = np.array([ball.speed for ball in visible]) if visible else np.array([0.0])
    steps = []
    previous = None
    for record in frames:
        if not record.ball.visible:
            previous = None
            continue
        if previous is not None:
            steps.append(float(np.hypot(record.ball.x - previous.x, record.ball.y - previous.y)))
        previous = record.ball

    print("=== shuttlecock track statistics ===")
    print(f"frames                    : {len(frames)}")
    print(f"detected (visible)        : {len(visible)} ({len(visible) / len(frames):.1%})")
    print(f"predicted (coasted)       : {len(predicted)} ({len(predicted) / len(frames):.1%})")
    print(f"lost                      : {missing} ({missing / len(frames):.1%})")
    if len(speeds):
        print(f"speed  mean/median/max    : {speeds.mean():.1f} / {np.median(speeds):.1f} / {speeds.max():.1f} px/frame")
    if steps:
        steps_arr = np.array(steps)
        print(f"step   mean/median/max    : {steps_arr.mean():.1f} / {np.median(steps_arr):.1f} / {steps_arr.max():.1f} px/frame")
        print(f"steps above 80 px/frame   : {np.mean(steps_arr > 80):.1%}")

    print("\n=== players ===")
    counts = [len(record.players) for record in frames]
    print(f"players detected per frame: mean {np.mean(counts):.2f}, min {np.min(counts)}, max {np.max(counts)}")

    print("\n=== events ===")
    by_type: dict[str, int] = {}
    for event in result.events:
        by_type[event["event_type"]] = by_type.get(event["event_type"], 0) + 1
    for key, value in sorted(by_type.items()):
        print(f"  {key:<14}: {value}")
    print(f"  point decisions : {len(result.decisions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build a contact sheet around a landing frame for visual verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.app import BadmintonApp
from badminton.config import load_config
from badminton.events.rally import _ball_at


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--before", type=int, default=6)
    parser.add_argument("--after", type=int, default=6)
    parser.add_argument("--out", default="out/verify")
    args = parser.parse_args()

    app = BadmintonApp(config=load_config(), source=args.video, allow_interactive=False, progress=False)
    result = app.analyze()
    balls = [record.ball for record in result.frames]

    state = _ball_at(balls, args.frame)
    if state is None:
        print("no ball state")
        return 1
    court = app.mapper.image_to_court((state.x, state.y))
    print(f"ball at frame {args.frame}: image=({state.x:.1f},{state.y:.1f}) court=({court[0]:.2f},{court[1]:.2f})")

    cap = cv2.VideoCapture(args.video)
    tiles = []
    for offset in range(-args.before, args.after + 1):
        index = args.frame + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = cap.read()
        if not ok:
            continue
        ball = _ball_at(balls, index)
        if ball is not None and ball.visible:
            cv2.circle(frame, (int(ball.x), int(ball.y)), 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(ball.x), int(ball.y)), 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"f{index}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (426, 240)))
    cap.release()

    if not tiles:
        return 1
    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start: start + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"verify_{Path(args.video).stem}_{args.frame}.png"
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

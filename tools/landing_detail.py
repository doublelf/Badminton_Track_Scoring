"""Show the raw vs settled landing positions for every landing event."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.app import BadmintonApp
from badminton.config import load_config
from badminton.events.rally import _ball_at


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    app = BadmintonApp(config=load_config(), source=args.video, allow_interactive=False, progress=False)
    result = app.analyze()
    analyzer = app.analyzer
    balls = [record.ball for record in result.frames]

    for event in result.events:
        if event.get("event_type") != "landing":
            continue
        frame = int(event["frame"])
        raw = _ball_at(balls, frame)
        settled, _state = analyzer._settled_landing_point(frame, balls)
        print(f"\n--- landing at frame {frame} ({event['rule']}) ---")
        if raw:
            cp = app.mapper.image_to_court(analyzer._ground_contact(raw))
            print(f"  landing frame {frame}: image=({raw.x:7.1f},{raw.y:7.1f}) speed={raw.speed:6.1f} court=({cp[0]:6.2f},{cp[1]:5.2f})")
        if settled:
            cs = app.mapper.image_to_court(settled)
            print(f"  settled position   : image=({settled[0]:7.1f},{settled[1]:7.1f}) court=({cs[0]:6.2f},{cs[1]:5.2f})")
        followers = [
            item
            for item in balls
            if frame < item.frame_index <= frame + 20 and item.visible
        ]
        print("  following frames   : " + ", ".join(
            f"f{item.frame_index}({item.x:.0f},{item.y:.0f},v={item.speed:.0f})" for item in followers[:12]
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

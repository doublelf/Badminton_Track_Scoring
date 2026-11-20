"""Explain why a landing event was accepted or rejected."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.app import BadmintonApp
from badminton.config import load_config
from badminton.events.rally import _ball_at


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--from", dest="start", type=int, default=0)
    parser.add_argument("--to", dest="end", type=int, default=10**9)
    args = parser.parse_args()

    app = BadmintonApp(config=load_config(), source=args.video, allow_interactive=False, progress=False)
    result = app.analyze()
    analyzer = app.analyzer
    balls = [record.ball for record in result.frames]

    for event in result.events:
        frame = int(event["frame"])
        if not (args.start <= frame <= args.end):
            continue
        state = _ball_at(balls, frame)
        plausible = analyzer._is_plausible_landing(event, balls) if event["event_type"] == "landing" else None
        court = analyzer.mapper.image_to_court(analyzer._ground_contact(state)) if state else None
        print(
            f"f{frame:5d} {event['event_type']:<12} {event['rule']:<22} plausible={plausible} "
            f"pos=({state.x:7.1f},{state.y:7.1f}) spd={state.speed:6.1f} vy={state.vy:7.1f} "
            f"vis={state.visible} court={[round(v, 2) for v in court] if court else None}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

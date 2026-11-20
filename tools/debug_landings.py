"""Inspect the trajectory around every landing decision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    for decision in result.decisions:
        frame = decision.frame
        print(
            f"\n=== frame {frame}: {decision.reason} winner={decision.winner} "
            f"court={decision.landing_court} in_bounds={decision.in_bounds} rule={decision.details.get('rule')}"
        )
        for record in frames[max(0, frame - 8): frame + 5]:
            ball = record.ball
            print(
                f"   f{record.index:5d} pos=({ball.x:7.1f},{ball.y:7.1f}) "
                f"v=({ball.vx:7.1f},{ball.vy:7.1f}) speed={ball.speed:7.1f} "
                f"conf={ball.confidence:.2f} vis={ball.visible}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

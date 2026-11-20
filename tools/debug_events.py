"""Debug helper: inspect events, attribution and trajectory quality."""

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
    parser.add_argument("--config", default=None)
    parser.add_argument("--calib", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    app = BadmintonApp(config=config, source=args.video, allow_interactive=False, progress=False)
    result = app.analyze()

    frames = result.frames
    print(f"frames={len(frames)} ball_visible={result.ball_detection_ratio:.2%} events={len(result.events)}")
    print("\n--- events ---")
    for event in result.events:
        frame = event["frame"]
        state = _ball_at([r.ball for r in frames], frame)
        pos = f"({state.x:6.1f},{state.y:6.1f}) vis={state.visible}" if state else "no-state"
        extra = ""
        if event["event_type"] == "hit" and 0 <= frame < len(frames):
            record = frames[frame]
            ball_states = [r.ball for r in frames]
            player_history = {r.index: r.players for r in frames}
            pose_history = {r.index: r.poses for r in frames}
            attribution = app.analyzer.attribute_hit(frame, ball_states, player_history, pose_history)
            extra = (
                f"  attrib: player={attribution.player_id} team={attribution.team} "
                f"dist={attribution.distance_px:.0f} conf={attribution.confidence:.2f} {attribution.method}"
            )
            extra += "  players=" + str([(p.track_id, p.half) for p in record.players])
        print(
            f"  f{frame:4d} {event['event_type']:<13} {event['rule']:<22} conf={event['confidence']:.2f} {pos}{extra}"
        )

    print("\n--- players per frame (first 40 frames with detections) ---")
    for record in frames[:60]:
        if not record.players:
            continue
        print(f"  f{record.index:4d} ball=({record.ball.x:6.1f},{record.ball.y:6.1f}) vis={record.ball.visible} "
              f"players={[(p.track_id, p.half, tuple(round(v) for v in p.bbox)) for p in record.players]}")

    print("\n--- trajectory samples ---")
    for record in frames[::20]:
        print(f"  f{record.index:4d} x={record.ball.x:7.1f} y={record.ball.y:7.1f} "
              f"v=({record.ball.vx:6.2f},{record.ball.vy:6.2f}) conf={record.ball.confidence:.2f} vis={record.ball.visible}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

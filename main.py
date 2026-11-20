"""CLI entry point: analyse a badminton video and render the annotated result.

Examples
--------
    python main.py --video match.mp4
    python main.py --video match.mp4 --calib calib.json --out out/
    python main.py --camera 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from badminton.app import BadmintonApp
from badminton.config import load_config
from badminton.court.mapper import CourtMapper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="badminton-track",
        description="Real-time badminton shuttlecock tracking and scoring",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="path to a badminton video file")
    source.add_argument("--camera", type=int, help="camera index (0 = default webcam)")

    parser.add_argument("--out", default="out", help="output directory (default: out)")
    parser.add_argument("--config", default=None, help="YAML configuration file")
    parser.add_argument(
        "--calib",
        default="auto",
        help="court calibration: 'auto', 'manual' or a path to a calibration JSON file",
    )
    parser.add_argument("--device", default="cpu", help="torch device for inference (default: cpu)")
    parser.add_argument("--no-display", action="store_true", help="do not open a preview window")
    parser.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    parser.add_argument("--limit-frames", type=int, default=0, help="stop after N frames (0 = all)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    source = args.video if args.video else args.camera

    mapper = None
    allow_interactive = True
    if args.calib not in ("auto", "manual"):
        calib_path = Path(args.calib)
        if not calib_path.exists():
            print(f"error: calibration file not found: {calib_path}", file=sys.stderr)
            return 2
        mapper = CourtMapper.load(calib_path)
        allow_interactive = False
    elif args.calib == "manual":
        allow_interactive = True

    app = BadmintonApp(
        config=config,
        source=source,
        mapper=mapper,
        allow_interactive=allow_interactive,
        device=args.device,
        limit_frames=args.limit_frames,
    )

    print(f"[1/2] analysing {source} ...")
    try:
        result = app.analyze()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"      frames={len(result.frames)}  ball visible={result.ball_detection_ratio * 100:.1f}%  "
        f"players visible={result.player_detection_ratio * 100:.1f}%  events={len(result.events)}"
    )

    print(f"[2/2] rendering to {args.out} ...")
    summary = app.render(
        result,
        args.out,
        show=not args.no_display,
        write_video=not args.no_video,
    )

    print("\n--- scoreboard ---")
    print(json.dumps(summary.get("game_history", []), ensure_ascii=False, indent=2))
    print(f"game {summary['game_number']}: {summary['points']['A']} - {summary['points']['B']}"
          f"   (serving: {summary['serving_team']})")
    if summary.get("match_over"):
        print(f"match winner: {summary['match_winner']}")
    print("\n--- points ---")
    for decision in summary.get("decisions", []):
        print(
            f"  frame {decision['frame']:>6}  {decision['winner_name']} +1  "
            f"{decision['reason']:<18} conf={decision['confidence']:.2f}  rule={decision['rule']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

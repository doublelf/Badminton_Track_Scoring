"""Report the measured landing bias of a video."""

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
    args = parser.parse_args()

    app = BadmintonApp(config=load_config(), source=args.video, allow_interactive=False, progress=False)
    app.analyze()
    report = app.landing_bias_report
    print(f"video: {args.video}")
    print(f"samples: {report.get('samples')}")
    print(f"bias   : ({report.get('bias', (0.0, 0.0))[0]:.2f}, {report.get('bias', (0.0, 0.0))[1]:.2f}) metres")
    print(f"spread : {report.get('spread', 0.0):.2f} metres")
    print(f"raw    : {report.get('raw')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

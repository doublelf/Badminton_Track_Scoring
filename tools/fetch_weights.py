"""Download the model weights required by the pipeline.

Sources (all public, no authentication required):

* ``yolo11s-ball.pt`` / ``yolo11n-pose.pt`` - released with the open-source
  ``Good-Badminton`` project (Apache-2.0).
* ``yolo11n.pt`` - official Ultralytics release.
* ``TrackNetV3`` checkpoints - optional, fetched with ``gdown`` when enabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

GOOD_BADMINTON_RELEASE = "https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0"
ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
TRACKNET_GDRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"

DEFAULT_WEIGHTS = {
    "yolo11s-ball.pt": f"{GOOD_BADMINTON_RELEASE}/yolo11s-ball.pt",
    "yolo11n-pose.pt": f"{GOOD_BADMINTON_RELEASE}/yolo11n-pose.pt",
    "yolo11n.pt": f"{ULTRALYTICS_ASSETS}/yolo11n.pt",
}


def download(url: str, destination: Path, force: bool = False) -> bool:
    if destination.exists() and not force and destination.stat().st_size > 0:
        print(f"[skip] {destination.name} already present ({destination.stat().st_size / 1e6:.1f} MB)")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    try:
        request = Request(url, headers={"User-Agent": "badminton-track/1.0"})
        with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            total = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
        print(f"[ok  ] {destination.name} ({total / 1e6:.1f} MB)")
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[fail] {destination.name}: {exc}", file=sys.stderr)
        if destination.exists():
            destination.unlink()
        return False


def download_tracknet(destination: Path, force: bool = False) -> bool:
    """Fetch the TrackNetV3 checkpoints (optional high-accuracy mode)."""
    if destination.exists() and not force:
        print(f"[skip] {destination} already present")
        return True
    try:
        import gdown
    except ImportError:
        print("[fail] gdown is required for TrackNetV3 weights: pip install gdown", file=sys.stderr)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] TrackNetV3 checkpoints -> {destination}")
    try:
        gdown.download(id=TRACKNET_GDRIVE_ID, output=str(destination), quiet=False)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[fail] TrackNetV3 download: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch badminton tracking model weights")
    parser.add_argument("--dir", default="weights", help="target directory (default: weights)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--with-tracknet", action="store_true", help="also fetch TrackNetV3 checkpoints")
    args = parser.parse_args()

    target = Path(args.dir)
    ok = True
    for name, url in DEFAULT_WEIGHTS.items():
        ok = download(url, target / name, force=args.force) and ok

    if args.with_tracknet:
        ok = download_tracknet(target / "TrackNetV3_ckpts.zip", force=args.force) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

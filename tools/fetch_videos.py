"""Download real badminton match footage for testing.

Primary source: public match clips that ship with the open-source projects
this pipeline is built on (GitHub raw, no authentication required).

Optional secondary source: YouTube via ``yt-dlp`` (install with
``pip install yt-dlp``).  YouTube often blocks datacenter IPs, so the GitHub
clips are the reliable default.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

try:  # optional, provides a bundled ffmpeg binary
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None

GITHUB_CLIPS = {
    "japan_open_2024": {
        "url": "https://raw.githubusercontent.com/amramer/Badminton-visionAI/main/data/Input_videos/badminton_Japan_Open_2024.mp4",
        "title": "Japan Open 2024 (1920x1080, public broadcast)",
    },
    "rally_demo": {
        "url": "https://raw.githubusercontent.com/JWM0203/RallyLive/main/videos/demo.mp4",
        "title": "RallyLive demo (1280x720)",
    },
    "rally_demo1": {
        "url": "https://raw.githubusercontent.com/JWM0203/RallyLive/main/videos/demo1.mp4",
        "title": "RallyLive demo 1 (1280x720)",
    },
    "visionai_reference": {
        "url": "https://raw.githubusercontent.com/amramer/Badminton-visionAI/main/demos/badminton-visionAI_01.mp4",
        "title": "Badminton-visionAI reference output (visual comparison)",
    },
}

YOUTUBE_CLIPS = {
    "olympics2024_ms_final": {
        "url": "https://www.youtube.com/watch?v=jsG9swdBk4s",
        "title": "Paris 2024 - Men's Singles Final",
        "section": "600-780",
    },
    "olympics2024_ws_final": {
        "url": "https://www.youtube.com/watch?v=tGTCaUf-LN4",
        "title": "Paris 2024 - Women's Singles Final",
        "section": "600-780",
    },
}


def ffmpeg_path() -> str | None:
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def download_github(name: str, output_dir: Path, force: bool = False) -> Path | None:
    clip = GITHUB_CLIPS[name]
    target = output_dir / f"{name}.mp4"
    if target.exists() and not force and target.stat().st_size > 0:
        print(f"[skip] {target} already present")
        return target
    print(f"[get ] {clip['title']} -> {target}")
    try:
        request = Request(clip["url"], headers={"User-Agent": "badminton-track/1.0"})
        with urlopen(request, timeout=300) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[fail] {name}: {exc}", file=sys.stderr)
        if target.exists():
            target.unlink()
        return None
    print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def download_youtube(name: str, output_dir: Path, height: int = 1080, force: bool = False) -> Path | None:
    clip = YOUTUBE_CLIPS[name]
    target = output_dir / f"{name}.mp4"
    if target.exists() and not force:
        print(f"[skip] {target} already present")
        return target

    ffmpeg = ffmpeg_path()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        f"bestvideo[height<={height}][ext=mp4]/best[height<={height}][ext=mp4]",
        "--no-playlist",
        "-o",
        str(output_dir / f"{name}.raw.%(ext)s"),
    ]
    if clip.get("section"):
        command += ["--download-sections", f"*{clip['section']}"]
    if ffmpeg:
        command += ["--ffmpeg-location", ffmpeg]
    command.append(clip["url"])

    print(f"[get ] {clip['title']} -> {target}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("[fail] yt-dlp returned an error", file=sys.stderr)
        return None

    raw_files = sorted(output_dir.glob(f"{name}.raw.*"))
    if not raw_files:
        print("[fail] no downloaded file found", file=sys.stderr)
        return None
    raw = raw_files[0]
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(raw),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-an",
                str(target),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not target.exists():
        shutil.move(str(raw), str(target))
    for leftover in output_dir.glob(f"{name}.raw.*"):
        leftover.unlink(missing_ok=True)
    print(f"[ok  ] {target}")
    return target if target.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download badminton test footage")
    parser.add_argument(
        "--clip",
        default="all",
        choices=sorted(set(GITHUB_CLIPS) | set(YOUTUBE_CLIPS)) + ["all", "github", "youtube"],
    )
    parser.add_argument("--dir", default="videos", help="target directory (default: videos)")
    parser.add_argument("--height", type=int, default=1080, help="maximum source height (YouTube)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clip == "all":
        github_names = sorted(GITHUB_CLIPS)
        youtube_names: list[str] = []
    elif args.clip == "github":
        github_names, youtube_names = sorted(GITHUB_CLIPS), []
    elif args.clip == "youtube":
        github_names, youtube_names = [], sorted(YOUTUBE_CLIPS)
    elif args.clip in GITHUB_CLIPS:
        github_names, youtube_names = [args.clip], []
    else:
        github_names, youtube_names = [], [args.clip]

    ok = True
    for name in github_names:
        if download_github(name, output_dir, force=args.force) is None:
            ok = False
    for name in youtube_names:
        if download_youtube(name, output_dir, height=args.height, force=args.force) is None:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified video input: camera index, video file, or network stream."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoSource:
    """Thin wrapper around :class:`cv2.VideoCapture` with a stable interface."""

    def __init__(
        self,
        source: str | int,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        self.source = source
        self.is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

        if isinstance(source, str) and not self.is_camera:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"video not found: {source}")

        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"unable to open video source: {source}")

        if self.is_camera and width and height:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if self.is_camera and fps:
            self.capture.set(cv2.CAP_PROP_FPS, fps)

        self._width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_index = -1
        self.start_frame = start_frame
        self.end_frame = end_frame

        if start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        if self._fps and self._fps > 1.0:
            return self._fps
        return 30.0

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    def read(self) -> tuple[bool, "cv2.typing.MatLike | None"]:
        if self.end_frame is not None and self.frame_index >= self.end_frame:
            return False, None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        self.frame_index += 1
        return True, frame

    def seek(self, frame_index: int) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        self.frame_index = int(frame_index) - 1

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

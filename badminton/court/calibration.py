"""Interactive court calibration and calibration persistence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .detector import auto_detect_court_corners, render_debug_preview
from .mapper import CourtMapper
from .reference import CourtSpec

CORNER_PROMPTS = (
    "1/4 far-left baseline corner",
    "2/4 far-right baseline corner",
    "3/4 near-right baseline corner",
    "4/4 near-left baseline corner",
)


def interactive_calibration(
    frame: np.ndarray,
    spec: CourtSpec | None = None,
    window: str = "Calibrate court",
) -> list[tuple[int, int]] | None:
    """Let the user click the four court corners in the given frame.

    Click order is ``far-left, far-right, near-right, near-left``.  Press
    ``s``/``Enter`` to accept, ``r`` to reset and ``Esc`` to cancel.
    """
    spec = spec or CourtSpec()
    state: dict = {"points": [], "window": frame.copy(), "done": False, "cancelled": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 4:
            state["points"].append((int(x), int(y)))
            state["window"] = frame.copy()
            _draw_points(state)

    def _draw_points(state):
        canvas = state["window"]
        for index, point in enumerate(state["points"], start=1):
            cv2.circle(canvas, point, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if len(state["points"]) < 4:
            prompt = CORNER_PROMPTS[len(state["points"])]
        else:
            prompt = "press s/Enter to accept, r to reset, Esc to cancel"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(canvas, prompt, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    _draw_points(state)

    while True:
        cv2.imshow(window, state["window"])
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            state["cancelled"] = True
            break
        if key in (13, 10, ord("s")):
            if len(state["points"]) == 4:
                state["done"] = True
                break
            continue
        if key == ord("r"):
            state["points"] = []
            state["window"] = frame.copy()
            _draw_points(state)

    cv2.destroyWindow(window)
    if state["cancelled"] or not state["done"]:
        return None
    return [tuple(point) for point in state["points"]]


def resolve_court_corners(
    frame: np.ndarray,
    spec: CourtSpec | None = None,
    manual_corners: list[tuple[int, int]] | None = None,
    allow_interactive: bool = True,
) -> tuple[list[tuple[int, int]] | None, dict]:
    """Resolve court corners from manual input, auto detection, or interaction."""
    spec = spec or CourtSpec()
    debug: dict = {}

    if manual_corners and len(manual_corners) == 4:
        return [(int(x), int(y)) for x, y in manual_corners], debug

    corners, _mask, auto_debug = auto_detect_court_corners(frame, spec)
    debug["auto"] = auto_debug
    if corners:
        return corners, debug

    if not allow_interactive:
        return None, debug

    manual = interactive_calibration(frame, spec)
    if manual:
        return manual, debug
    return None, debug


def save_calibration(mapper: CourtMapper, path: str | Path, extra: dict | None = None) -> None:
    payload = mapper.to_dict()
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> CourtMapper:
    return CourtMapper.load(path)


def preview_detection(frame: np.ndarray, spec: CourtSpec | None = None) -> np.ndarray:
    """Render the auto-detection result for debugging purposes."""
    corners, _mask, debug = auto_detect_court_corners(frame, spec)
    return render_debug_preview(frame, corners, debug)

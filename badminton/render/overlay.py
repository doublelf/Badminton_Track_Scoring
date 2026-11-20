"""HUD rendering: scoreboard, event banner, bird's-eye mini court and stats."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..court.mapper import CourtMapper
from ..court.reference import CourtSpec
from ..scoring.rules import MatchRules

TEAM_COLORS = {
    "A": (255, 170, 60),
    "B": (90, 110, 245),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class BannerState:
    text: str = ""
    subtitle: str = ""
    color: tuple[int, int, int] = (255, 255, 255)
    expires_at: float = 0.0


class Overlay:
    """Draws every user-facing element on top of the video frames."""

    def __init__(
        self,
        mapper: CourtMapper | None = None,
        spec: CourtSpec | None = None,
        minimap_size: int = 300,
        show_minimap: bool = True,
        show_fps: bool = True,
    ) -> None:
        self.mapper = mapper
        self.spec = spec or (mapper.spec if mapper else CourtSpec())
        self.minimap_size = minimap_size
        self.show_minimap = show_minimap
        self.show_fps = show_fps
        self.banner = BannerState()

    # ------------------------------------------------------------------ banner
    def show_banner(self, text: str, subtitle: str = "", color: tuple[int, int, int] = (255, 255, 255),
                    duration: float = 2.5) -> None:
        self.banner = BannerState(text=text, subtitle=subtitle, color=color, expires_at=time.time() + duration)

    def clear_banner(self) -> None:
        self.banner = BannerState()

    def draw_banner(self, frame: np.ndarray) -> np.ndarray:
        if not self.banner.text or time.time() > self.banner.expires_at:
            return frame
        height, width = frame.shape[:2]
        text = self.banner.text
        subtitle = self.banner.subtitle
        scale = 1.4
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, scale, 3)
        (sub_w, sub_h), _ = cv2.getTextSize(subtitle, FONT, 0.8, 2)

        box_w = max(text_w, sub_w) + 60
        box_h = text_h + sub_h + 50
        x1 = (width - box_w) // 2
        y1 = int(height * 0.16)
        x2, y2 = x1 + box_w, y1 + box_h

        layer = frame.copy()
        cv2.rectangle(layer, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(layer, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.banner.color, 2, cv2.LINE_AA)

        cv2.putText(frame, text, (x1 + 30, y1 + text_h + 18), FONT, scale, self.banner.color, 3, cv2.LINE_AA)
        if subtitle:
            cv2.putText(frame, subtitle, (x1 + 30, y1 + text_h + sub_h + 40), FONT, 0.8,
                        (235, 235, 235), 2, cv2.LINE_AA)
        return frame

    # --------------------------------------------------------------- scoreboard
    def draw_scoreboard(self, frame: np.ndarray, rules: MatchRules) -> np.ndarray:
        height, width = frame.shape[:2]
        panel_w, panel_h = 430, 132
        # Broadcast footage usually carries its own scoreboard in the top left
        # corner, so ours is drawn in the top right.
        x0 = max(0, width - panel_w)
        layer = frame.copy()
        cv2.rectangle(layer, (x0, 0), (width, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(layer, 0.55, frame, 0.45, 0, frame)
        cv2.line(frame, (x0, panel_h), (width, panel_h), (200, 200, 200), 1, cv2.LINE_AA)

        name_a = rules.name_of("A")
        name_b = rules.name_of("B")
        serving = rules.serving_team

        def team_block(team: str, name: str, y: int) -> None:
            color = TEAM_COLORS[team]
            cv2.rectangle(frame, (x0 + 18, y), (x0 + 30, y + 26), color, -1)
            cv2.putText(frame, name, (x0 + 40, y + 20), FONT, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            if serving == team:
                cv2.circle(frame, (width - 34, y + 12), 7, (80, 235, 120), -1, cv2.LINE_AA)
                cv2.putText(frame, "S", (width - 62, y + 18), FONT, 0.55, (80, 235, 120), 2, cv2.LINE_AA)

        team_block("A", name_a, 18)
        team_block("B", name_b, 58)

        cv2.putText(frame, str(rules.points["A"]), (width - 96, 38), FONT, 1.05,
                    TEAM_COLORS["A"], 3, cv2.LINE_AA)
        cv2.putText(frame, str(rules.points["B"]), (width - 96, 78), FONT, 1.05,
                    TEAM_COLORS["B"], 3, cv2.LINE_AA)

        games = f"{rules.games['A']} - {rules.games['B']}"
        cv2.putText(frame, f"Games {games}   |   Game {rules.game_number}", (x0 + 18, panel_h - 14), FONT, 0.58,
                    (215, 215, 215), 1, cv2.LINE_AA)

        history = " ".join(
            f"{result.score_winner}-{result.score_loser}" for result in rules.game_history
        )
        if history:
            cv2.putText(frame, f"Previous games: {history}", (x0 + 18, panel_h - 40), FONT, 0.5,
                        (170, 170, 170), 1, cv2.LINE_AA)

        if rules.match_over and rules.match_winner:
            cv2.putText(frame, f"{rules.name_of(rules.match_winner)} WINS", (width - 330, 160), FONT, 1.0,
                        (80, 235, 120), 3, cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------------- minimap
    def draw_minimap(
        self,
        frame: np.ndarray,
        players: list,
        ball_states: list | None = None,
        landing_marks: list | None = None,
        trail_points: list | None = None,
    ) -> np.ndarray:
        if not self.show_minimap:
            return frame
        height, width = frame.shape[:2]
        size = self.minimap_size
        map_h = int(size * self.spec.width / self.spec.length)
        margin = 16
        x1 = width - size - margin
        y1 = height - map_h - margin

        layer = frame.copy()
        cv2.rectangle(layer, (x1, y1), (x1 + size, y1 + map_h), (0, 0, 0), -1)
        cv2.addWeighted(layer, 0.55, frame, 0.45, 0, frame)

        def to_map(court_point: tuple[float, float]) -> tuple[int, int]:
            cx, cy = court_point
            px = x1 + int(cx / self.spec.length * size)
            py = y1 + int(cy / self.spec.width * map_h)
            return px, py

        # court lines
        for start, end, weight in self.spec.lines():
            p1, p2 = to_map(start), to_map(end)
            color = (110, 110, 110) if weight <= 0.4 else (215, 215, 215)
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        # players
        for player in players:
            if player.court_point is None:
                continue
            px, py = to_map(player.court_point)
            color = TEAM_COLORS.get(player.half or "A", (200, 200, 200))
            cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, str(player.track_id), (px + 7, py - 5), FONT, 0.38, (235, 235, 235), 1, cv2.LINE_AA)

        # shuttle trail on the court
        for point in trail_points or []:
            px, py = to_map(point)
            cv2.circle(frame, (px, py), 2, (255, 235, 80), -1, cv2.LINE_AA)

        # current shuttle position
        if ball_states:
            latest = ball_states[-1]
            if latest.visible or latest.predicted:
                px, py = to_map(self.mapper.image_to_court((latest.x, latest.y))) if self.mapper else (0, 0)
                cv2.circle(frame, (px, py), 4, (70, 255, 255), -1, cv2.LINE_AA)

        # landing markers
        for mark in landing_marks or []:
            px, py = to_map(mark)
            cv2.drawMarker(frame, (px, py), (60, 60, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

        cv2.putText(frame, "COURT VIEW", (x1 + 6, y1 + 16), FONT, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------- stats
    def draw_stats(self, frame: np.ndarray, fps: float, extra: str = "") -> np.ndarray:
        if not self.show_fps:
            return frame
        height, width = frame.shape[:2]
        text = f"{fps:5.1f} FPS"
        if extra:
            text += f"   {extra}"
        cv2.putText(frame, text, (width - 230, 158), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        return frame

    def draw_players(self, frame: np.ndarray, players: list) -> np.ndarray:
        for player in players:
            x1, y1, x2, y2 = [int(value) for value in player.bbox]
            color = TEAM_COLORS.get(player.half or "A", (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"P{player.track_id}"
            cv2.putText(frame, label, (x1, max(16, y1 - 6)), FONT, 0.55, color, 2, cv2.LINE_AA)
        return frame

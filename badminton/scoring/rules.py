"""BWF scoring rules for singles.

Implements the official rally-point system:

* 21 points to win a game, winning by two points, capped at 30.
* From 20-20 the game continues until a side leads by two points (service then
  changes with every point, which follows automatically from the service rule).
* Best of three games.
* Players change ends at 10 points in games one and two, and at 11 points in
  the third game.
* Service: the service passes whenever the serving side's score is odd after
  a rally, which yields the familiar "change every two points" pattern in
  singles and "change every point" behaviour in the deuce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PointEvent:
    """A single awarded point."""

    team: str
    reason: str
    frame: int = -1
    rally_index: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "reason": self.reason,
            "frame": self.frame,
            "rally_index": self.rally_index,
            "details": self.details,
        }


@dataclass
class GameResult:
    """Outcome of a finished game."""

    winner: str
    score_winner: int
    score_loser: int
    game_number: int


class MatchRules:
    """Stateful BWF singles scoring engine."""

    def __init__(
        self,
        team_a_name: str = "A",
        team_b_name: str = "B",
        points_to_win: int = 21,
        win_by: int = 2,
        point_cap: int = 30,
        games_to_win: int = 2,
        switch_ends_at: int = 10,
        switch_ends_at_third_game: int = 11,
        initial_server: str = "A",
    ) -> None:
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.points_to_win = points_to_win
        self.win_by = win_by
        self.point_cap = point_cap
        self.games_to_win = games_to_win
        self.switch_ends_at = switch_ends_at
        self.switch_ends_at_third_game = switch_ends_at_third_game

        self.points: dict[str, int] = {"A": 0, "B": 0}
        self.games: dict[str, int] = {"A": 0, "B": 0}
        self.game_number = 1
        self.serving_team = initial_server
        self.first_server_of_match = initial_server
        self.game_history: list[GameResult] = []
        self.events: list[PointEvent] = []
        self.ended_ends: list[int] = []
        self.match_over = False
        self.match_winner: str | None = None
        self._ends_switched_this_game = False

    # ------------------------------------------------------------------ helpers
    def name_of(self, team: str) -> str:
        return self.team_a_name if team == "A" else self.team_b_name

    def opponent(self, team: str) -> str:
        return "B" if team == "A" else "A"

    def team_from_name(self, name: str) -> str:
        return "A" if name == self.team_a_name else "B"

    @property
    def deuce(self) -> bool:
        return self.points["A"] >= self.points_to_win - 1 and self.points["B"] >= self.points_to_win - 1

    # ------------------------------------------------------------------ scoring
    def award_point(self, team: str, reason: str, frame: int = -1, rally_index: int = 0,
                    details: dict[str, Any] | None = None) -> PointEvent:
        """Register a point for ``team`` and apply all follow-up rules."""
        if self.match_over:
            raise ValueError("the match is already finished")

        opponent = self.opponent(team)
        self.points[team] += 1

        event = PointEvent(
            team=team,
            reason=reason,
            frame=frame,
            rally_index=rally_index,
            details=details or {},
        )
        self.events.append(event)

        # Service: passes whenever the serving side's score is odd.
        if self.points[self.serving_team] % 2 == 1:
            self.serving_team = opponent

        # Change of ends.
        threshold = self.switch_ends_at_third_game if self.game_number >= 3 else self.switch_ends_at
        if not self._ends_switched_this_game and max(self.points["A"], self.points["B"]) >= threshold:
            self._ends_switched_this_game = True
            self.ended_ends.append(self.game_number)

        # Game / match completion.
        if self._game_won(team):
            self._finish_game(team)
        return event

    def _game_won(self, team: str) -> bool:
        opponent = self.opponent(team)
        if self.points[team] < self.points_to_win:
            return False
        if self.points[team] >= self.point_cap:
            return True
        return self.points[team] - self.points[opponent] >= self.win_by

    def _finish_game(self, team: str) -> None:
        opponent = self.opponent(team)
        self.game_history.append(
            GameResult(
                winner=team,
                score_winner=self.points[team],
                score_loser=self.points[opponent],
                game_number=self.game_number,
            )
        )
        self.games[team] += 1

        if self.games[team] >= self.games_to_win:
            self.match_over = True
            self.match_winner = team
            return

        self.game_number += 1
        self.points = {"A": 0, "B": 0}
        self._ends_switched_this_game = False
        # In the next game the service passes to the receiving side.
        self.serving_team = self.opponent(self.first_server_of_match)
        self.first_server_of_match = self.serving_team

    # -------------------------------------------------------------------- misc
    def undo_last(self) -> PointEvent | None:
        """Undo the most recent point (used by the manual correction keys)."""
        if not self.events:
            return None
        event = self.events.pop()
        self.points[event.team] = max(0, self.points[event.team] - 1)
        return event

    def reset(self) -> None:
        """Reset the whole match."""
        self.points = {"A": 0, "B": 0}
        self.games = {"A": 0, "B": 0}
        self.game_number = 1
        self.serving_team = self.first_server_of_match
        self.game_history.clear()
        self.events.clear()
        self.ended_ends.clear()
        self.match_over = False
        self.match_winner = None
        self._ends_switched_this_game = False

    def score_line(self) -> str:
        """Compact score summary, e.g. ``A 21-15 1-0 | B``."""
        game_scores = " ".join(
            f"{self.name_of(result.winner)} {result.score_winner}-{result.score_loser}"
            for result in self.game_history
        )
        games = f"{self.games['A']}-{self.games['B']}"
        return f"Game {self.game_number} | {game_scores} | games {games} | current {self.points['A']}-{self.points['B']}"

    def summary(self) -> dict[str, Any]:
        return {
            "points": dict(self.points),
            "games": dict(self.games),
            "game_number": self.game_number,
            "serving_team": self.serving_team,
            "match_over": self.match_over,
            "match_winner": self.match_winner,
            "game_history": [
                {
                    "winner": result.winner,
                    "winner_name": self.name_of(result.winner),
                    "score": f"{result.score_winner}-{result.score_loser}",
                    "game": result.game_number,
                }
                for result in self.game_history
            ],
            "events": [event.to_dict() for event in self.events],
        }

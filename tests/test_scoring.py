"""BWF singles scoring rules."""

from __future__ import annotations

import pytest

from badminton.scoring.rules import MatchRules


def make_rules(**kwargs) -> MatchRules:
    defaults = dict(
        team_a_name="A",
        team_b_name="B",
        points_to_win=21,
        win_by=2,
        point_cap=30,
        games_to_win=2,
        switch_ends_at=10,
        switch_ends_at_third_game=11,
        initial_server="A",
    )
    defaults.update(kwargs)
    return MatchRules(**defaults)


def play_to(rules: MatchRules, target_a: int, target_b: int) -> None:
    """Play a balanced game until the requested score is reached.

    Points are given to the trailing side so that neither player wins the
    game before the target score is reached.
    """
    while rules.points["A"] < target_a or rules.points["B"] < target_b:
        team = "A" if rules.points["A"] <= rules.points["B"] else "B"
        rules.award_point(team, "IN")


def win_game(rules: MatchRules, winner: str) -> None:
    """Play a game that ``winner`` takes."""
    game_before = rules.game_number
    while rules.points["A"] < 19 and rules.points["B"] < 19:
        team = "A" if rules.points["A"] <= rules.points["B"] else "B"
        rules.award_point(team, "IN")
    for _ in range(8):
        if rules.match_over:
            return
        rules.award_point(winner, "IN")
        if rules.game_number != game_before:
            return
    raise AssertionError("the game was not finished")


def test_point_increments_and_service_rotation():
    rules = make_rules()
    assert rules.serving_team == "A"

    rules.award_point("A", "IN")
    # A served and won: A's score is 1 (odd) so the service passes.
    assert rules.serving_team == "B"
    assert rules.points == {"A": 1, "B": 0}

    rules.award_point("B", "IN")
    # B won while serving: B's score is 1 (odd) -> service passes back to A.
    assert rules.serving_team == "A"
    assert rules.points == {"A": 1, "B": 1}

    rules.award_point("A", "IN")
    # A's score is 2 (even) -> A keeps the service.
    assert rules.serving_team == "A"
    assert rules.points == {"A": 2, "B": 1}


def test_service_changes_every_two_points_in_singles():
    rules = make_rules()
    # The server keeps the service while their own score is even.
    rules.award_point("A", "IN")
    assert rules.serving_team == "B"  # A's score is odd now
    rules.award_point("B", "IN")
    assert rules.serving_team == "A"  # B's score is odd now
    rules.award_point("A", "IN")
    assert rules.serving_team == "A"  # A's score is 2 (even) -> keeps serving
    rules.award_point("A", "IN")
    assert rules.serving_team == "B"  # A's score is 3 (odd) -> service passes
    assert rules.points == {"A": 3, "B": 1}


def test_game_won_at_21_with_two_point_lead():
    rules = make_rules()
    for _ in range(20):
        rules.award_point("A", "IN")
    for _ in range(19):
        rules.award_point("B", "IN")
    assert rules.game_number == 1
    assert rules.points == {"A": 20, "B": 19}

    rules.award_point("A", "IN")
    assert rules.game_number == 2
    assert rules.games["A"] == 1
    assert rules.points == {"A": 0, "B": 0}


def test_deuce_requires_two_point_lead():
    rules = make_rules()
    for _ in range(20):
        rules.award_point("A", "IN")
    for _ in range(20):
        rules.award_point("B", "IN")
    assert rules.points == {"A": 20, "B": 20}
    assert rules.deuce

    # From 20-20 the service changes with every point.
    server_before = rules.serving_team
    rules.award_point("A", "IN")
    assert rules.serving_team != server_before
    rules.award_point("B", "IN")
    assert rules.serving_team == server_before
    assert rules.points == {"A": 21, "B": 21}
    assert rules.game_number == 1

    rules.award_point("A", "IN")
    rules.award_point("A", "IN")
    assert rules.game_number == 2
    assert rules.games["A"] == 1


def test_point_cap_at_30():
    rules = make_rules()
    play_to(rules, 29, 29)
    assert rules.points == {"A": 29, "B": 29}
    assert rules.game_number == 1

    rules.award_point("A", "IN")
    # 30-29 wins the game despite the one point lead.
    assert rules.game_number == 2
    assert rules.games["A"] == 1


def test_change_of_ends_at_ten_points():
    rules = make_rules()
    for _ in range(10):
        rules.award_point("A", "IN")
    assert 1 in rules.ended_ends
    # Only once per game.
    rules.ended_ends.clear()
    for _ in range(5):
        rules.award_point("A", "IN")
    assert rules.ended_ends == []


def test_change_of_ends_at_eleven_in_third_game():
    rules = make_rules()
    win_game(rules, "A")
    assert rules.game_number == 2
    assert rules.games == {"A": 1, "B": 0}

    win_game(rules, "B")
    assert rules.game_number == 3
    assert rules.games == {"A": 1, "B": 1}

    rules.ended_ends.clear()
    play_to(rules, 11, 11)
    assert 3 in rules.ended_ends


def test_service_passes_to_receiving_side_in_new_game():
    rules = make_rules(initial_server="A")
    play_to(rules, 21, 19)
    rules.award_point("A", "IN")
    assert rules.game_number == 2
    # The side that did not serve first serves the new game.
    assert rules.serving_team == "B"


def test_match_ends_after_two_games():
    rules = make_rules()
    win_game(rules, "A")
    assert not rules.match_over
    win_game(rules, "B")
    assert not rules.match_over
    assert rules.games == {"A": 1, "B": 1}

    win_game(rules, "A")
    assert rules.match_over
    assert rules.match_winner == "A"
    assert rules.games == {"A": 2, "B": 1}


def test_undo_removes_the_last_point():
    rules = make_rules()
    rules.award_point("A", "IN")
    rules.award_point("B", "IN")
    assert rules.points == {"A": 1, "B": 1}
    event = rules.undo_last()
    assert event is not None and event.team == "B"
    assert rules.points == {"A": 1, "B": 0}


def test_reset_clears_the_match():
    rules = make_rules()
    rules.award_point("A", "IN")
    rules.reset()
    assert rules.points == {"A": 0, "B": 0}
    assert rules.events == []
    assert rules.game_number == 1


def test_summary_is_json_serialisable():
    import json

    rules = make_rules()
    rules.award_point("A", "IN", frame=12, details={"in_bounds": True})
    payload = json.dumps(rules.summary())
    assert "points" in payload

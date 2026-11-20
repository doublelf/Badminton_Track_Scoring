"""Rally analysis: landing decisions, hit attribution and point outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..court.mapper import CourtMapper
from ..court.reference import CourtSpec


@dataclass
class HitAttribution:
    """Which player (and side) produced a hit event."""

    player_id: int | None = None
    team: str | None = None
    distance_px: float = float("inf")
    confidence: float = 0.0
    method: str = "none"


@dataclass
class RallyDecision:
    """A resolved point."""

    frame: int
    winner: str
    reason: str
    landing_image: tuple[float, float] | None = None
    landing_court: tuple[float, float] | None = None
    in_bounds: bool | None = None
    last_hitter: HitAttribution | None = None
    confidence: float = 0.0
    rally_index: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class RallyAnalyzer:
    """Turns raw trajectory events into point decisions."""

    def __init__(
        self,
        mapper: CourtMapper,
        spec: CourtSpec | None = None,
        in_out_margin: float = 0.02,
        racket_reach_m: float = 1.6,
        hit_window_offsets: tuple[int, ...] = (0, 3, 5),
        min_attribution_confidence: float = 0.25,
        min_landing_speed: float = 3.0,
        max_landing_speed: float = 12.0,
        max_ascent_velocity: float = -3.0,
        max_descent_velocity: float = 2.0,
        settle_window: int = 25,
        settle_speed: float = 1.5,
        min_settle_frames: int = 6,
        landing_bias: tuple[float, float] = (0.0, 0.0),
        landing_area_margin: float = 1.2,
    ) -> None:
        self.mapper = mapper
        self.spec = spec or mapper.spec
        self.in_out_margin = in_out_margin
        self.racket_reach_m = racket_reach_m
        self.hit_window_offsets = hit_window_offsets
        self.min_attribution_confidence = min_attribution_confidence
        self.min_landing_speed = min_landing_speed
        self.max_landing_speed = max_landing_speed
        self.max_ascent_velocity = max_ascent_velocity
        self.max_descent_velocity = max_descent_velocity
        self.settle_window = settle_window
        self.settle_speed = settle_speed
        self.min_settle_frames = min_settle_frames
        self.landing_bias = landing_bias
        self.landing_area_margin = landing_area_margin

        self.rally_index = 0
        self.rally_active = False
        self.last_hitter: HitAttribution | None = None
        self.last_hit_frame = -1
        self.serve_frame = -1
        self.point_frame = -10_000
        self.min_rally_gap = 20
        self.decisions: list[RallyDecision] = []

    # ------------------------------------------------------------------ helpers
    def _opponent(self, team: str) -> str:
        return "B" if team == "A" else "A"

    def _ground_contact(self, ball_state) -> tuple[float, float]:
        """Image point where the shuttlecock touches the floor."""
        x, y = ball_state.x, ball_state.y
        box = getattr(ball_state, "box", None)
        if box is not None:
            _bx, _by, _bw, bh = box
            y = y + bh / 2.0
        return float(x), float(y)

    def attribute_hit(
        self,
        hit_frame: int,
        ball_states: list,
        player_history: dict[int, list] | list,
        pose_history: dict[int, list] | None = None,
    ) -> HitAttribution:
        """Find the player that most likely produced the hit.

        The shuttlecock is already a few frames away from the racket when the
        kinematic rules fire, so several frame offsets are sampled (the
        "hit window" approach used by the open-source hit-event reference
        implementation).
        """
        if isinstance(player_history, dict):
            def tracks_at(frame: int):
                return player_history.get(frame, [])
        else:
            def tracks_at(frame: int):
                if 0 <= frame < len(player_history):
                    return player_history[frame]
                return []

        best = HitAttribution()
        for offset in self.hit_window_offsets:
            frame = hit_frame + offset
            state = _ball_at(ball_states, frame)
            if state is None:
                continue
            poses = pose_history.get(frame, []) if pose_history else []
            if not poses:
                poses = _nearest_pose(pose_history, frame)
            candidate = self._closest_player(state, tracks_at(frame), poses)
            if candidate.player_id is not None and candidate.confidence > best.confidence:
                best = candidate
        return best

    def _closest_player(self, state, player_tracks: list, pose_observations: list | None) -> HitAttribution:
        if not player_tracks:
            return HitAttribution()

        pose_by_id = {pose.track_id: pose for pose in (pose_observations or [])}
        best = HitAttribution()

        for track in player_tracks:
            reach_points = []
            method = "bbox"
            reach_px = self._reach_pixels(track.height_px)
            pose = pose_by_id.get(track.track_id)
            if pose is not None and pose.racket_heads:
                reach_points = list(pose.racket_heads.values())
                method = "racket"
                # The racket can be held far away from the body.
                reach_px = max(reach_px, self._reach_pixels(track.height_px, self.racket_reach_m))

            if not reach_points:
                centre_x, _centre_y = track.center
                reach_points = [(centre_x, max(0.0, track.bbox[1] - track.height_px * 0.25))]

            best_point = None
            best_distance = None
            for point in reach_points:
                distance = float(np.hypot(point[0] - state.x, point[1] - state.y))
                if distance > reach_px:
                    continue
                if best_distance is None or distance < best_distance:
                    best_point, best_distance = point, distance
            if best_point is None:
                continue
            confidence = max(0.0, 1.0 - best_distance / max(reach_px, 1e-6))
            candidate = HitAttribution(
                player_id=track.track_id,
                team=track.half,
                distance_px=best_distance,
                confidence=confidence,
                method=method,
            )
            if candidate.confidence > best.confidence:
                best = candidate

        return best

    def _reach_pixels(self, player_height_px: float, reach_m: float | None = None) -> float:
        metres_per_pixel = 1.75 / max(player_height_px, 1e-6)
        reach = self.racket_reach_m if reach_m is None else reach_m
        return reach / max(metres_per_pixel, 1e-9)

    # ------------------------------------------------------------------- events
    def process_event(
        self,
        event: dict[str, Any],
        ball_states: list,
        player_history: dict[int, list] | list,
        pose_history: dict[int, list] | None = None,
    ) -> RallyDecision | None:
        """Handle one event; returns a decision when the rally ends."""
        event_type = event.get("event_type")
        frame = int(event.get("frame", -1))

        if event_type == "hit":
            attribution = self.attribute_hit(frame, ball_states, player_history, pose_history)
            if attribution.player_id is not None:
                self.last_hitter = attribution
                self.last_hit_frame = frame
                # A new rally only starts once the previous point has been
                # resolved and the shuttlecock has been served again.  Only
                # hits that can be attributed to a player start a rally, so
                # that kinematic noise in mid-flight does not create points.
                if not self.rally_active and frame - self.point_frame >= self.min_rally_gap:
                    self.rally_active = True
                    self.rally_index += 1
                    self.serve_frame = frame
                    self.last_hitter = attribution
            return None

        if event_type in ("landing", "out_of_frame"):
            # A point can only be decided while a rally is running: the shuttle
            # keeps bouncing and rolling on the floor afterwards.
            if not self.rally_active:
                return None
            if event_type == "landing" and not self._is_plausible_landing(event, ball_states):
                return None
            return self._resolve_point(event, ball_states)

        return None

    @staticmethod
    def _pose_at(pose_history: dict[int, list] | None, frame: int) -> list:
        if not pose_history:
            return []
        return pose_history.get(frame, []) or _nearest_pose(pose_history, frame)

    def _is_plausible_landing(self, event: dict[str, Any], ball_states: list) -> bool:
        """Reject landing candidates that cannot be a bounce on the floor.

        A real landing shows the shuttlecock slowing down to rest (or
        bouncing) on the floor.  Candidates detected while the ball is still
        flying fast, while it is rising (the apex of a lob) or far above the
        court are tracking artefacts.
        """
        frame = int(event.get("frame", -1))
        state = _ball_at(ball_states, frame)
        if state is None or not state.visible:
            return False
        if state.speed > self.max_landing_speed:
            # Still flying fast: this is not a bounce on the floor.
            return False

        window = [
            item
            for item in ball_states
            if frame - 8 <= item.frame_index <= frame and item.visible
        ]
        if not window:
            return False
        recent_vy = float(np.mean([item.vy for item in window[-4:]]))
        if recent_vy < self.max_ascent_velocity:
            # The ball is still on its way up: this is the apex of a lob and
            # not a bounce on the floor.
            return False

        # After a bounce the shuttlecock does not start falling again.  At the
        # apex of a lob it does, which separates the two cases.
        after = [
            item
            for item in ball_states
            if frame < item.frame_index <= frame + 5 and item.visible
        ]
        if len(after) >= 2:
            next_vy = float(np.mean([item.vy for item in after]))
            if next_vy > self.max_descent_velocity:
                return False

        court_point = self.mapper.image_to_court(self._ground_contact(state))
        margin = self.landing_area_margin
        if not (
            -margin <= court_point[0] <= self.spec.length + margin
            and -margin <= court_point[1] <= self.spec.width + margin
        ):
            return False
        return True

    def _settled_landing_point(self, frame: int, ball_states: list) -> tuple[tuple[float, float] | None, object | None]:
        """Best estimate of the point where the shuttlecock touched the floor.

        The ground projection of a ball that is still in the air is displaced
        away from the camera, which is the main source of in/out errors.  Once
        the shuttlecock has come to rest on the floor its ground projection is
        exact, so the resting position right after the landing event is
        preferred over the position of the landing frame itself.

        Only the first few frames of the resting phase are used - a shuttle
       cock keeps bouncing and rolling, and later frames would drift away
        from the impact point.
        """
        state = _ball_at(ball_states, frame)
        if state is None:
            return None, None

        followers = [
            item
            for item in ball_states
            if frame < item.frame_index <= frame + self.settle_window and item.visible
        ]
        settled: list = []
        for item in followers:
            if item.speed > self.settle_speed:
                break
            settled.append(item)
            if len(settled) >= self.min_settle_frames:
                break
        if len(settled) >= self.min_settle_frames:
            xs = [item.x for item in settled]
            ys = [item.y for item in settled]
            position = (float(np.median(xs)), float(np.median(ys)))
            return position, settled[-1]
        return None, state

    def _impact_point(self, frame: int, ball_states: list) -> tuple[float, float] | None:
        """Image position of the bounce itself.

        The kinematic rules react a couple of frames after the shuttlecock
        touched the floor, so the position of the landing frame is still a
        little off.  The bounce is the point where the vertical image motion
        changes direction, i.e. the local extremum of the image ordinate.
        """
        window = [
            item
            for item in ball_states
            if frame - 3 <= item.frame_index <= frame + 4 and item.visible
        ]
        if len(window) < 4:
            return None

        by_index = {item.frame_index: item for item in window}
        before = by_index.get(frame - 2)
        at = by_index.get(frame)
        after = by_index.get(frame + 2)
        if before is None or at is None or after is None:
            return None

        if at.y < before.y and at.y < after.y:
            return (at.x, at.y)  # local minimum
        if at.y > before.y and at.y > after.y:
            return (at.x, at.y)  # local maximum
        return None

    def _resolve_point(self, event: dict[str, Any], ball_states: list) -> RallyDecision | None:
        frame = int(event.get("frame", -1))
        state = _ball_at(ball_states, frame)
        landing_image = None
        landing_court = None
        in_bounds = None
        confidence = float(event.get("confidence", 0.5))

        if state is not None and event.get("event_type") == "landing":
            settled_image, settled_state = self._settled_landing_point(frame, ball_states)
            impact = self._impact_point(frame, ball_states)
            if settled_image is not None:
                landing_image = settled_image
                landing_court = self.mapper.image_to_court(settled_image)
                # The shuttlecock really touched the floor, so the decision is
                # more reliable than the raw landing frame.
                confidence = min(1.0, confidence + 0.05)
            elif impact is not None:
                landing_image = impact
                landing_court = self.mapper.image_to_court(impact)
                confidence = min(1.0, confidence + 0.03)
            else:
                landing_image = self._ground_contact(state)
                landing_court = self.mapper.image_to_court(landing_image)
                # No resting position available: correct the ground
                # projection with the systematic offset measured on the other
                # landings of the same video (the ball was still in the air
                # when the landing fired, which shifts the projection away
                # from the camera).
                landing_court = (
                    landing_court[0] + self.landing_bias[0],
                    landing_court[1] + self.landing_bias[1],
                )
            in_bounds = self.mapper.is_in_bounds(landing_court, margin=self.in_out_margin, singles=True)

        hitter = self.last_hitter
        hitter_team = hitter.team if hitter and hitter.confidence >= self.min_attribution_confidence else None

        if landing_court is not None:
            landing_half = self.spec.half(landing_court)
            if in_bounds:
                if hitter_team is None:
                    winner = self._opponent(landing_half)
                    reason = "IN (hitter unknown)"
                    confidence *= 0.5
                elif landing_half == hitter_team:
                    winner = self._opponent(hitter_team)
                    reason = "NOT OVER NET"
                else:
                    winner = hitter_team
                    reason = "IN"
            else:
                if hitter_team is None:
                    winner = self._opponent(landing_half)
                    reason = "OUT (hitter unknown)"
                    confidence *= 0.5
                else:
                    winner = self._opponent(hitter_team)
                    reason = "OUT"
        else:
            if hitter_team is not None:
                winner = self._opponent(hitter_team)
                reason = "OUT OF FRAME"
                confidence *= 0.8
            else:
                return None

        decision = RallyDecision(
            frame=frame,
            winner=winner,
            reason=reason,
            landing_image=landing_image,
            landing_court=landing_court,
            in_bounds=in_bounds,
            last_hitter=hitter,
            confidence=min(1.0, confidence),
            rally_index=self.rally_index,
            details={"rule": event.get("rule"), "event_type": event.get("event_type")},
        )
        self.decisions.append(decision)
        self.rally_active = False
        self.point_frame = frame
        self.last_hitter = None
        return decision


def _nearest_pose(pose_history: dict[int, list] | None, frame: int, tolerance: int = 4) -> list:
    """Closest available pose observation for ``frame``."""
    if not pose_history:
        return []
    if frame in pose_history and pose_history[frame]:
        return pose_history[frame]
    best = None
    best_distance = None
    for index, poses in pose_history.items():
        if not poses:
            continue
        distance = abs(index - frame)
        if best_distance is None or distance < best_distance:
            best, best_distance = poses, distance
    if best is None or (best_distance is not None and best_distance > tolerance):
        return []
    return best


def _ball_at(ball_states: list, frame: int):
    """Closest ball state at (or just before) ``frame``."""
    if not ball_states:
        return None
    best = None
    best_distance = None
    for state in ball_states:
        distance = abs(state.frame_index - frame)
        if best_distance is None or distance < best_distance:
            best, best_distance = state, distance
    if best is None or best_distance > 3:
        return None
    return best

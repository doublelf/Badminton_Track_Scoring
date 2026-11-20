"""End-to-end pipeline: analysis pass and rendering pass."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Node
from .court.calibration import resolve_court_corners
from .court.mapper import CourtMapper
from .court.reference import CourtSpec
from .detect.ball import BallState, ShuttleTracker
from .detect.player import PlayerDetector, PlayerTrack
from .detect.pose import PoseDetector, PoseObservation
from .events.candidates import BounceCandidateGenerator
from .events.kinematics import KinematicsCalculator
from .events.rally import RallyAnalyzer, RallyDecision, _ball_at
from .io.video_source import VideoSource
from .render.overlay import Overlay
from .scoring.rules import MatchRules
from .trail import TrailBuffer


@dataclass
class FrameRecord:
    """Everything detected in a single frame."""

    index: int
    ball: BallState
    players: list[PlayerTrack] = field(default_factory=list)
    poses: list[PoseObservation] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Output of the analysis pass."""

    frames: list[FrameRecord]
    decisions: list[RallyDecision]
    rules: MatchRules
    events: list[dict[str, Any]]
    fps: float
    resolution: tuple[int, int]
    ball_detection_ratio: float = 0.0
    player_detection_ratio: float = 0.0


class BadmintonApp:
    """Runs detection, event analysis, scoring and rendering."""

    def __init__(
        self,
        config: Node,
        source: str | int,
        mapper: CourtMapper | None = None,
        allow_interactive: bool = True,
        device: str = "cpu",
        progress: bool = True,
        limit_frames: int = 0,
    ) -> None:
        self.config = config
        self.source = source
        self.device = device
        self.progress = progress
        self.limit_frames = max(0, int(limit_frames))
        self.spec = CourtSpec(
            length=config.court.length,
            width=config.court.width,
            singles_margin=config.court.singles_margin,
            service_line_from_net=config.court.service_line_from_net,
            back_service_offset=config.court.back_service_offset,
        )
        self.mapper = mapper
        self.allow_interactive = allow_interactive
        self.video: VideoSource | None = None
        self.overlay: Overlay | None = None
        self.analyzer: RallyAnalyzer | None = None
        self.landing_bias_report: dict = {}

    max_landing_bias_spread: float = 0.8

    # ------------------------------------------------------------------ helpers
    def open(self) -> VideoSource:
        if self.video is None:
            self.video = VideoSource(
                self.source,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
        return self.video

    def _build_mapper(self, frame: np.ndarray) -> CourtMapper:
        if self.mapper is not None:
            return self.mapper
        corners, _debug = resolve_court_corners(
            frame,
            self.spec,
            manual_corners=None,
            allow_interactive=self.allow_interactive,
        )
        if corners is None:
            raise RuntimeError(
                "court calibration failed - provide a calibration file with --calib "
                "or use --calib manual to click the four corners"
            )
        self.mapper = CourtMapper(corners, self.spec.corner_order, self.spec)
        return self.mapper

    def _roi_polygon(self, padding: float) -> list[tuple[float, float]]:
        assert self.mapper is not None
        points = self.mapper.image_corners
        centre = points.mean(axis=0)
        expanded = []
        for point in points:
            vector = point - centre
            expanded.append(tuple((centre + vector * (1.0 + padding)).tolist()))
        return expanded

    # ----------------------------------------------------------------- analysis
    def analyze(self) -> AnalysisResult:
        config = self.config
        _configure_threads()
        video = self.open()
        first_ok, first_frame = video.read()
        if not first_ok or first_frame is None:
            raise RuntimeError("could not read the first frame of the video")

        mapper = self._build_mapper(first_frame)
        height, width = first_frame.shape[:2]
        self.overlay = Overlay(
            mapper=mapper,
            spec=self.spec,
            minimap_size=config.render.minimap_size,
            show_minimap=config.render.minimap,
            show_fps=config.render.show_fps,
        )

        player_model = PlayerDetector(
            model_path=str(config.player.model),
            imgsz=int(config.player.imgsz),
            conf=float(config.player.conf),
            iou=float(config.player.iou),
            tracker=str(config.player.tracker),
            device=self.device,
        )
        ball_model = ShuttleTracker(
            imgsz=int(config.ball.imgsz),
            conf=float(config.ball.conf),
            iou=float(config.ball.iou),
            max_box_area_ratio=float(config.ball.max_box_area_ratio),
            max_aspect_ratio=float(config.ball.max_aspect_ratio),
            max_jump_pixels=float(config.ball.max_jump_pixels),
            prediction_gate_pixels=float(config.ball.prediction_gate_pixels),
            max_missing_frames=int(config.ball.max_missing_frames),
        )
        ball_model.model = self._load_model(str(config.ball.model))

        pose_model = None
        if config.pose.enabled:
            pose_model = PoseDetector(
                model_path=str(config.pose.model),
                imgsz=int(config.pose.imgsz),
                conf=float(config.pose.conf),
                device=self.device,
            )

        roi_polygon = self._roi_polygon(float(config.ball.roi_padding_ratio))
        frames: list[FrameRecord] = []
        player_every = max(1, int(config.player.every_n_frames))
        ball_every = max(1, int(config.ball.every_n_frames))

        frame_index = 0
        ball_visible = 0
        players_seen = 0
        started = time.time()

        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                break
            if self.limit_frames and frame_index >= self.limit_frames:
                break

            players: list[PlayerTrack] = []
            if frame_index % player_every == 0:
                players = player_model.update(frame, mapper)
            else:
                players = _interpolate_players(frames[-1].players if frames else [], mapper)
            players = _filter_players_on_court(players, mapper, margin=1.2)

            poses: list[PoseObservation] = []
            if pose_model is not None and frame_index % max(1, int(config.pose.every_n_frames)) == 0:
                poses = pose_model.update(frame, players)

            ball = ball_model.update(
                frame,
                frame_index,
                roi_polygon=roi_polygon if config.ball.use_roi else None,
                detect=frame_index % ball_every == 0,
                reference_area=height * width,
            )
            if ball.visible:
                ball_visible += 1
            if players:
                players_seen += 1

            frames.append(FrameRecord(index=frame_index, ball=ball, players=players, poses=poses))
            frame_index += 1

            if self.progress and frame_index % 100 == 0:
                elapsed = time.time() - started
                rate = frame_index / max(elapsed, 1e-6)
                print(
                    f"\r  analysed {frame_index} frames ({rate:.1f} fps)",
                    end="",
                    flush=True,
                )

        if self.progress:
            print()

        return self._analyse_events(frames, video.fps, (width, height), ball_visible, players_seen, len(frames))

    def _analyse_events(
        self,
        frames: list[FrameRecord],
        fps: float,
        resolution: tuple[int, int],
        ball_visible: int,
        players_seen: int,
        total_frames: int,
    ) -> AnalysisResult:
        """Run the Phase-1 event engine over the full trajectory."""
        config = self.config
        assert self.mapper is not None

        x = np.array([record.ball.x for record in frames], dtype=np.float64)
        y = np.array([record.ball.y for record in frames], dtype=np.float64)
        visibility = np.array([1.0 if record.ball.visible else 0.0 for record in frames], dtype=np.float64)

        generator = BounceCandidateGenerator(
            speed_drop_ratio=float(config.events.landing.speed_drop_ratio),
            min_speed_before_landing=float(config.events.landing.min_speed_before_landing),
            max_speed_after_landing=float(config.events.landing.max_speed_after_landing),
            high_speed_threshold=float(config.events.landing.high_speed_threshold),
            low_speed_threshold=float(config.events.landing.low_speed_threshold),
            low_speed_ratio=float(config.events.landing.low_speed_ratio),
            pre_window=int(config.events.landing.pre_window),
            post_window=int(config.events.landing.post_window),
            hold_window=int(config.events.landing.hold_window),
            min_speed_at_hit=float(config.events.hit.min_speed_at_hit),
            vy_reversal_threshold=float(config.events.hit.vy_reversal_threshold),
            acc_threshold=float(config.events.hit.acc_threshold),
            min_height_diff=float(config.events.hit.min_height_diff),
            min_peak_speed=float(config.events.hit.min_peak_speed),
            min_speed_diff=float(config.events.hit.min_speed_diff),
            min_visible_before=int(config.events.min_visible_before),
            merge_window=int(config.events.merge_window),
            future_check_window=int(config.events.landing.future_check_window),
            edge_margin=int(config.events.landing.edge_margin),
        )
        kinematics = KinematicsCalculator(fps=fps, reference_fps=float(config.events.fps_reference)).compute(
            x, y, visibility
        )
        events = generator.generate(
            x,
            y,
            visibility,
            kinematics=kinematics,
            img_height=resolution[1],
            img_width=resolution[0],
        )

        analyzer = RallyAnalyzer(
            mapper=self.mapper,
            spec=self.spec,
            in_out_margin=float(config.court.in_out_margin_m),
            racket_reach_m=float(config.events.attribution.racket_reach_m),
            hit_window_offsets=tuple(config.events.attribution.hit_window_offsets),
            min_attribution_confidence=float(config.events.attribution.min_attribution_confidence),
            max_landing_speed=float(config.events.landing.max_landing_speed),
            max_ascent_velocity=float(config.events.landing.max_ascent_velocity),
            max_descent_velocity=float(config.events.landing.max_descent_velocity),
            settle_window=int(config.events.landing.settle_window),
            settle_speed=float(config.events.landing.settle_speed),
            min_settle_frames=int(config.events.landing.min_settle_frames),
        )
        analyzer.landing_bias = self._estimate_landing_bias(analyzer, events, frames)

        rules = MatchRules(
            team_a_name=str(config.scoring.team_a_name),
            team_b_name=str(config.scoring.team_b_name),
            points_to_win=int(config.scoring.points_to_win),
            win_by=int(config.scoring.win_by),
            point_cap=int(config.scoring.point_cap),
            games_to_win=int(config.scoring.games_to_win),
            switch_ends_at=int(config.scoring.switch_ends_at),
            switch_ends_at_third_game=int(config.scoring.switch_ends_at_third_game),
        )

        decisions: list[RallyDecision] = []
        self.analyzer = analyzer
        ball_states = [record.ball for record in frames]
        player_history = {record.index: record.players for record in frames}
        pose_history = {record.index: record.poses for record in frames} if config.pose.enabled else None
        for event in events:
            decision = analyzer.process_event(event, ball_states, player_history, pose_history)
            if decision is not None:
                decisions.append(decision)

        for decision in decisions:
            rules.award_point(
                decision.winner,
                decision.reason,
                frame=decision.frame,
                rally_index=decision.rally_index,
                details={
                    "in_bounds": decision.in_bounds,
                    "landing_court": list(decision.landing_court) if decision.landing_court else None,
                    "confidence": decision.confidence,
                },
            )

        return AnalysisResult(
            frames=frames,
            decisions=decisions,
            rules=rules,
            events=events,
            fps=fps,
            resolution=resolution,
            ball_detection_ratio=ball_visible / max(1, total_frames),
            player_detection_ratio=players_seen / max(1, total_frames),
        )

    def _estimate_landing_bias(
        self, analyzer: RallyAnalyzer, events: list[dict], frames: list[FrameRecord]
    ) -> tuple[float, float]:
        """Measure the systematic offset of the landing ground projections.

        When the landing rule fires the shuttlecock is usually still a little
        above the floor, so its ground projection is displaced away from the
        camera.  Landings where the shuttlecock comes to rest afterwards give
        a reference; the median offset of those is used to correct the
        remaining landings of the same video.
        """
        assert self.mapper is not None
        balls = [record.ball for record in frames]
        offsets: list[tuple[float, float]] = []
        for event in events:
            if event.get("event_type") != "landing":
                continue
            frame = int(event.get("frame", -1))
            settled, _state = analyzer._settled_landing_point(frame, balls)
            if settled is None:
                continue
            raw = _ball_at(balls, frame)
            if raw is None:
                continue
            reference = self.mapper.image_to_court(settled)
            projected = self.mapper.image_to_court(analyzer._ground_contact(raw))
            offsets.append((reference[0] - projected[0], reference[1] - projected[1]))

        if not offsets:
            self.landing_bias_report = {"samples": 0, "bias": (0.0, 0.0), "spread": 0.0, "applied": False}
            return (0.0, 0.0)
        array = np.array(offsets, dtype=np.float64)
        bias = (float(np.median(array[:, 0])), float(np.median(array[:, 1])))
        spread = float(np.median(np.linalg.norm(array - np.array(bias), axis=1)))
        applied = len(array) >= 3 and spread <= self.max_landing_bias_spread
        self.landing_bias_report = {
            "samples": int(len(array)),
            "bias": bias,
            "spread": spread,
            "applied": applied,
            "raw": [(round(float(x), 2), round(float(y), 2)) for x, y in offsets],
        }
        # Only trust the correction when several measurements agree.
        if not applied:
            return (0.0, 0.0)
        return bias

    # ------------------------------------------------------------------ render
    def render(
        self,
        result: AnalysisResult,
        out_dir: str | Path,
        show: bool = True,
        write_video: bool = True,
    ) -> dict[str, Any]:
        """Write the annotated video and the event/score exports."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        assert self.mapper is not None
        config = self.config
        video = self.open()
        video.seek(0)

        overlay = self.overlay or Overlay(
            mapper=self.mapper,
            spec=self.spec,
            minimap_size=int(config.render.minimap_size),
            show_minimap=bool(config.render.minimap),
            show_fps=bool(config.render.show_fps),
        )

        width, height = result.resolution
        out_fps = float(config.render.video_fps) or result.fps or 30.0
        writer = None
        if write_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path / "annotated.mp4"), fourcc, out_fps, (width, height))

        trail = TrailBuffer(
            length=int(config.trail.length),
            min_distance=float(config.trail.min_point_distance),
            fade=bool(config.trail.fade),
            color_by_speed=bool(config.trail.color_by_speed),
        )

        decisions_by_frame: dict[int, list[RallyDecision]] = {}
        for decision in result.decisions:
            decisions_by_frame.setdefault(decision.frame, []).append(decision)

        landing_marks: list[tuple[float, float]] = []
        rules = MatchRules(
            team_a_name=str(config.scoring.team_a_name),
            team_b_name=str(config.scoring.team_b_name),
            points_to_win=int(config.scoring.points_to_win),
            win_by=int(config.scoring.win_by),
            point_cap=int(config.scoring.point_cap),
            games_to_win=int(config.scoring.games_to_win),
            switch_ends_at=int(config.scoring.switch_ends_at),
            switch_ends_at_third_game=int(config.scoring.switch_ends_at_third_game),
        )

        paused = False
        show_trail = True
        show_minimap = bool(config.render.minimap)
        overlay.show_minimap = show_minimap
        frame_index = 0
        started = time.time()
        rendered = 0

        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                break
            if self.limit_frames and frame_index >= self.limit_frames:
                break

            record = result.frames[frame_index] if frame_index < len(result.frames) else None
            players = record.players if record else []
            ball = record.ball if record else None

            canvas = self.mapper.draw_court_overlay(frame)

            if ball is not None:
                trail.add(ball.x, ball.y, ball.frame_index, ball.speed, ball.visible)
                if show_trail:
                    trail.draw(canvas)
                if ball.visible:
                    cv2.circle(canvas, (int(ball.x), int(ball.y)), 5, (60, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (int(ball.x), int(ball.y)), 5, (0, 0, 0), 1, cv2.LINE_AA)

            if players:
                overlay.draw_players(canvas, players)

            for decision in decisions_by_frame.get(frame_index, []):
                rules.award_point(
                    decision.winner,
                    decision.reason,
                    frame=decision.frame,
                    rally_index=decision.rally_index,
                    details={
                        "in_bounds": decision.in_bounds,
                        "landing_court": list(decision.landing_court) if decision.landing_court else None,
                        "confidence": decision.confidence,
                    },
                )
                if decision.landing_court:
                    landing_marks.append(decision.landing_court)
                color = (90, 235, 120) if decision.reason == "IN" else (90, 110, 245)
                subtitle = (
                    f"{rules.name_of(decision.winner)} +1   |   "
                    f"rule: {decision.details.get('rule', '-')}   conf: {decision.confidence:.2f}"
                )
                overlay.show_banner(f"{decision.reason}", subtitle, color)

            trail_points = []
            if ball is not None:
                trail_points = [
                    self.mapper.image_to_court((ball.x, ball.y))
                ]
            overlay.draw_minimap(canvas, players, [record.ball] if record else None, landing_marks, trail_points)
            overlay.draw_scoreboard(canvas, rules)

            elapsed = time.time() - started
            fps = rendered / max(elapsed, 1e-6)
            overlay.draw_stats(canvas, fps, f"ball {result.ball_detection_ratio * 100:.0f}%")

            if writer is not None:
                writer.write(canvas)

            if show:
                cv2.imshow("Badminton Tracking", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("t"):
                    show_trail = not show_trail
                if key == ord("m"):
                    show_minimap = not show_minimap
                    overlay.show_minimap = show_minimap
                if key in (ord("i"), ord("o")):
                    _correct_last(rules, key == ord("i"))
                    overlay.show_banner("MANUAL", "score corrected", (255, 200, 60))

            frame_index += 1
            rendered += 1
            if self.progress and rendered % 100 == 0:
                print(f"\r  rendered {rendered} frames ({fps:.1f} fps)", end="", flush=True)

        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if self.progress:
            print()

        summary = rules.summary()
        summary["resolution"] = list(result.resolution)
        summary["fps"] = result.fps
        summary["ball_detection_ratio"] = result.ball_detection_ratio
        summary["player_detection_ratio"] = result.player_detection_ratio
        summary["decisions"] = [
            {
                "frame": decision.frame,
                "winner": decision.winner,
                "winner_name": rules.name_of(decision.winner),
                "reason": decision.reason,
                "in_bounds": decision.in_bounds,
                "landing_court": list(decision.landing_court) if decision.landing_court else None,
                "confidence": decision.confidence,
                "rule": decision.details.get("rule"),
            }
            for decision in result.decisions
        ]
        (out_path / "scoreboard.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_events_jsonl(out_path / "events.jsonl", result.decisions, rules)
        return summary

    # ------------------------------------------------------------------ models
    @staticmethod
    def _load_model(path: str):
        from ultralytics import YOLO

        return YOLO(path)


def _configure_threads() -> None:
    """Use a sensible number of torch threads on CPU-only machines."""
    import os

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return
    cpus = os.cpu_count() or 4
    threads = max(2, min(8, cpus - 2))
    try:
        torch.set_num_threads(threads)
    except Exception:  # pragma: no cover - defensive
        pass


def _filter_players_on_court(players: list[PlayerTrack], mapper: CourtMapper, margin: float = 2.0) -> list[PlayerTrack]:
    """Keep only the players standing on the playing area.

    Officials, camera operators and spectators are detected as well; they are
    removed by checking whether the foot point projects close to the court.
    """
    kept: list[PlayerTrack] = []
    for track in players:
        if track.court_point is None:
            continue
        x, y = track.court_point
        if -margin <= x <= mapper.spec.length + margin and -margin <= y <= mapper.spec.width + margin:
            kept.append(track)
    return kept


def _interpolate_players(previous: list[PlayerTrack], mapper: CourtMapper | None) -> list[PlayerTrack]:
    """Reuse the last known player boxes when detection is skipped."""
    if not previous:
        return []
    updated = []
    for track in previous:
        copy = PlayerTrack(
            track_id=track.track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            foot_point=track.foot_point,
            court_point=track.court_point,
            half=track.half,
        )
        updated.append(copy)
    return updated


def _correct_last(rules: MatchRules, to_in: bool) -> None:
    """Flip the outcome of the most recent point (manual correction)."""
    if not rules.events:
        return
    last = rules.events[-1]
    rules.undo_last()
    if to_in:
        winner = last.team
        reason = "IN (manual)"
    else:
        winner = "B" if last.team == "A" else "A"
        reason = "OUT (manual)"
    rules.award_point(winner, reason, frame=last.frame, rally_index=last.rally_index)


def _write_events_jsonl(path: Path, decisions: list[RallyDecision], rules: MatchRules) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(
                json.dumps(
                    {
                        "frame": decision.frame,
                        "winner": decision.winner,
                        "winner_name": rules.name_of(decision.winner),
                        "reason": decision.reason,
                        "in_bounds": decision.in_bounds,
                        "landing_court": list(decision.landing_court) if decision.landing_court else None,
                        "confidence": decision.confidence,
                        "rule": decision.details.get("rule"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

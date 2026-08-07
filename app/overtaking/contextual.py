"""State-machine overtaking assessment using ordering and lane transitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import OvertakingConfig, TrafficContextConfig
from app.models import (
    LaneObservation,
    LaneTransition,
    OvertakeState,
    OvertakingAssessment,
    OvertakingStatus,
    SpeedEstimate,
    TrafficFrameContext,
)
from app.motion import MotionHistoryStore


class NoOvertakingPolicy:
    """Compatibility mode that explicitly leaves overtaking unassessed."""

    def update(
        self,
        timestamp_seconds: float,
        observations: list[LaneObservation],
        transitions: list[LaneTransition],
        context: TrafficFrameContext,
        history: MotionHistoryStore,
        speeds: dict[int, SpeedEstimate] | None = None,
    ) -> dict[int, OvertakingAssessment]:
        del speeds
        return {
            observation.vehicle.track_id: OvertakingAssessment(
                track_id=observation.vehicle.track_id,
                status=OvertakingStatus.NOT_ASSESSED,
                state=OvertakeState.NONE,
                confidence=0.0,
                evidence=("contextual_overtaking_disabled",),
            )
            for observation in observations
        }


@dataclass(slots=True)
class _OvertakeAttempt:
    state: OvertakeState
    started_at: float
    from_lane: str
    related_track_id: int | None
    last_target_seen_at: float
    evidence: list[str] = field(default_factory=list)
    completed_at: float | None = None
    last_related_seen_at: float | None = None
    related_temporarily_missing: bool = False


class ContextualOvertakingPolicy:
    def __init__(
        self,
        config: OvertakingConfig,
        context_config: TrafficContextConfig,
        left_lane_id: str,
    ) -> None:
        self._config = config
        self._context_config = context_config
        self._left_lane_id = left_lane_id
        self._attempts: dict[int, _OvertakeAttempt] = {}

    def update(
        self,
        timestamp_seconds: float,
        observations: list[LaneObservation],
        transitions: list[LaneTransition],
        context: TrafficFrameContext,
        history: MotionHistoryStore,
        speeds: dict[int, SpeedEstimate] | None = None,
    ) -> dict[int, OvertakingAssessment]:
        if not self._config.enabled:
            return NoOvertakingPolicy().update(
                timestamp_seconds, observations, transitions, context, history, speeds
            )
        geometry = context.global_context.geometry_integrity
        if geometry is None or not geometry.overtaking_inference_allowed:
            self._attempts.clear()
            return {
                observation.vehicle.track_id: OvertakingAssessment(
                    track_id=observation.vehicle.track_id,
                    status=OvertakingStatus.INSUFFICIENT_EVIDENCE,
                    state=OvertakeState.NONE,
                    confidence=0.0,
                    evidence=(
                        "GEOMETRY_INTEGRITY_LOST",
                        *(geometry.reason_codes if geometry is not None else ()),
                    ),
                )
                for observation in observations
            }

        lane_by_track = {
            observation.vehicle.track_id: observation.lane_id
            for observation in observations
        }
        transition_by_track = {
            transition.track_id: transition for transition in transitions
        }
        for transition in transitions:
            if transition.to_lane == self._left_lane_id:
                self._attempts[transition.track_id] = self._begin_attempt(
                    transition, timestamp_seconds, context, history
                )

        assessments: dict[int, OvertakingAssessment] = {}
        for observation in observations:
            track_id = observation.vehicle.track_id
            attempt = self._attempts.get(track_id)
            if attempt is None:
                assessments[track_id] = self._without_attempt(track_id, history)
                continue
            attempt.last_target_seen_at = timestamp_seconds
            transition = transition_by_track.get(track_id)
            self._advance(
                attempt,
                track_id,
                lane_by_track.get(track_id),
                transition,
                timestamp_seconds,
                context,
                speeds or {},
            )
            assessments[track_id] = self._assessment(attempt, track_id, history)
        seen_ids = {observation.vehicle.track_id for observation in observations}
        for track_id, attempt in list(self._attempts.items()):
            if (
                track_id not in seen_ids
                and timestamp_seconds - attempt.last_target_seen_at
                > self._config.completion_timeout_seconds
            ):
                del self._attempts[track_id]
        return assessments

    def _begin_attempt(
        self,
        transition: LaneTransition,
        timestamp_seconds: float,
        context: TrafficFrameContext,
        history: MotionHistoryStore,
    ) -> _OvertakeAttempt:
        related_track_id = self._entry_target(transition.track_id, context)
        if related_track_id is None:
            related_track_id = self._historical_entry_target(
                transition, timestamp_seconds, history
            )
        evidence = [
            "TARGET_ENTERED_LEFT_LANE",
            f"entered_left_from_{transition.from_lane}_at_{transition.timestamp_seconds:.2f}",
        ]
        if related_track_id is not None:
            evidence.append(f"vehicle_{related_track_id}_ahead_in_previous_lane")
            state = OvertakeState.ENTERED_LEFT
        else:
            evidence.append("no_near_ahead_vehicle_detected_at_entry")
            state = OvertakeState.ABORTED
        return _OvertakeAttempt(
            state=state,
            started_at=timestamp_seconds,
            from_lane=transition.from_lane,
            related_track_id=related_track_id,
            last_target_seen_at=timestamp_seconds,
            evidence=evidence,
            last_related_seen_at=(
                timestamp_seconds if related_track_id is not None else None
            ),
        )

    def _entry_target(self, track_id: int, context: TrafficFrameContext) -> int | None:
        vehicle_context = context.vehicles.get(track_id)
        if vehicle_context is None:
            return None
        ahead = vehicle_context.neighbors.adjacent_right_ahead
        if ahead is not None and ahead.longitudinal_gap <= (
            self._config.entry_target_max_gap_meters
            if ahead.gap_unit == "meters"
            else self._config.entry_target_max_gap_normalized
        ):
            return ahead.track_id
        return None

    def _historical_entry_target(
        self,
        transition: LaneTransition,
        timestamp_seconds: float,
        history: MotionHistoryStore,
    ) -> int | None:
        target = history.latest(transition.track_id)
        if target is None:
            return None
        best: tuple[float, int] | None = None
        cutoff = timestamp_seconds - self._config.observation_window_seconds
        for other_id in history.track_ids():
            if other_id == transition.track_id:
                continue
            other = history.latest(other_id)
            if (
                other is None
                or other.timestamp_seconds < cutoff
                or other.lane_id != transition.from_lane
            ):
                continue
            gap = other.longitudinal_position - target.longitudinal_position
            maximum_gap = (
                self._config.entry_target_max_gap_meters
                if target.coordinate_mode == "calibrated_world"
                and other.coordinate_mode == "calibrated_world"
                else self._config.entry_target_max_gap_normalized
            )
            if 0 < gap <= maximum_gap and (best is None or gap < best[0]):
                best = (gap, other_id)
        return best[1] if best else None

    def _advance(
        self,
        attempt: _OvertakeAttempt,
        track_id: int,
        lane_id: str | None,
        transition: LaneTransition | None,
        timestamp_seconds: float,
        context: TrafficFrameContext,
        speeds: dict[int, SpeedEstimate],
    ) -> None:
        if (
            attempt.state in {OvertakeState.ENTERED_LEFT, OvertakeState.PASSING}
            and timestamp_seconds - attempt.started_at
            > self._config.completion_timeout_seconds
        ):
            attempt.state = OvertakeState.ABORTED
            attempt.evidence.append("overtaking_attempt_timed_out")

        if transition is not None and transition.from_lane == self._left_lane_id:
            if attempt.state in {OvertakeState.PASSED_TARGET, OvertakeState.COMPLETED}:
                attempt.state = OvertakeState.RETURNING_RIGHT
                attempt.evidence.append(
                    f"returned_right_at_{transition.timestamp_seconds:.2f}"
                )
                _append_once(attempt.evidence, "RETURNED_RIGHT")
                attempt.state = OvertakeState.COMPLETED
            elif attempt.state in {OvertakeState.ENTERED_LEFT, OvertakeState.PASSING}:
                attempt.state = OvertakeState.ABORTED
                attempt.evidence.append("returned_before_pass_was_observed")

        related_id = attempt.related_track_id
        target_position = context.positions.get(track_id)
        related_position = (
            context.positions.get(related_id) if related_id is not None else None
        )
        if target_position is not None and related_position is not None:
            attempt.last_related_seen_at = timestamp_seconds
            attempt.related_temporarily_missing = False
            relative = related_position.longitudinal - target_position.longitudinal
            margin = (
                self._config.pass_order_margin_meters
                if target_position.calibrated and related_position.calibrated
                else self._config.pass_order_margin_normalized
            )
            target_speed = speeds.get(track_id)
            related_speed = speeds.get(related_id) if related_id is not None else None
            if (
                target_speed is not None
                and related_speed is not None
                and target_speed.speed_mps is not None
                and related_speed.speed_mps is not None
                and target_speed.speed_mps - related_speed.speed_mps
                >= self._config.minimum_relative_speed_mps
            ):
                _append_once(attempt.evidence, "TARGET_GAINING_ON_VEHICLE")
            if relative > margin and attempt.state in {
                OvertakeState.ENTERED_LEFT,
                OvertakeState.PASSING,
            }:
                attempt.state = OvertakeState.PASSING
            elif relative < -margin and attempt.state in {
                OvertakeState.ENTERED_LEFT,
                OvertakeState.PASSING,
            }:
                attempt.state = OvertakeState.PASSED_TARGET
                attempt.completed_at = timestamp_seconds
                _append_once(attempt.evidence, "RELATIVE_ORDER_CHANGED")
                _append_once(attempt.evidence, "TARGET_PASSED_VEHICLE")
                attempt.evidence.append(
                    f"relative_order_reversed_with_vehicle_{related_id}_at_{timestamp_seconds:.2f}"
                )
        elif (
            related_id is not None
            and attempt.last_related_seen_at is not None
            and timestamp_seconds - attempt.last_related_seen_at
            > self._config.related_track_lost_grace_seconds
        ):
            if not attempt.related_temporarily_missing:
                attempt.evidence.append("related_vehicle_not_currently_observed")
            attempt.related_temporarily_missing = True

        if (
            attempt.state == OvertakeState.PASSED_TARGET
            and attempt.completed_at is not None
            and lane_id == self._left_lane_id
            and timestamp_seconds - attempt.completed_at
            >= self._config.post_overtake_grace_seconds
        ):
            attempt.state = OvertakeState.ABORTED
            attempt.evidence.append("post_overtake_left_lane_grace_elapsed")

    def _assessment(
        self,
        attempt: _OvertakeAttempt,
        track_id: int,
        history: MotionHistoryStore,
    ) -> OvertakingAssessment:
        if attempt.related_temporarily_missing and attempt.state in {
            OvertakeState.ENTERED_LEFT,
            OvertakeState.PASSING,
        }:
            status = OvertakingStatus.INSUFFICIENT_EVIDENCE
            confidence = 0.40
        elif attempt.state in {OvertakeState.ENTERED_LEFT, OvertakeState.PASSING}:
            status = OvertakingStatus.LIKELY_OVERTAKING
            confidence = (
                0.78 if "TARGET_GAINING_ON_VEHICLE" in attempt.evidence else 0.65
            )
        elif attempt.state in {
            OvertakeState.PASSED_TARGET,
            OvertakeState.RETURNING_RIGHT,
            OvertakeState.COMPLETED,
        }:
            confidence = 0.90 if attempt.state != OvertakeState.COMPLETED else 0.98
            if "TARGET_GAINING_ON_VEHICLE" in attempt.evidence:
                confidence = min(1.0, confidence + 0.05)
            status = (
                OvertakingStatus.OVERTAKING_CONFIRMED
                if confidence >= self._config.minimum_confidence
                else OvertakingStatus.LIKELY_OVERTAKING
            )
        elif (
            history.duration_seconds(track_id)
            >= self._context_config.minimum_history_seconds
        ):
            status = OvertakingStatus.NOT_OVERTAKING
            confidence = 0.78
        else:
            status = OvertakingStatus.INSUFFICIENT_EVIDENCE
            confidence = 0.35
        return OvertakingAssessment(
            track_id=track_id,
            status=status,
            state=attempt.state,
            confidence=confidence,
            evidence=tuple(attempt.evidence),
            related_track_ids=(
                (attempt.related_track_id,)
                if attempt.related_track_id is not None
                else ()
            ),
            started_at=attempt.started_at,
            completed_at=attempt.completed_at,
        )

    def _without_attempt(
        self, track_id: int, history: MotionHistoryStore
    ) -> OvertakingAssessment:
        history_duration = history.duration_seconds(track_id)
        if history_duration >= self._context_config.minimum_history_seconds:
            return OvertakingAssessment(
                track_id=track_id,
                status=OvertakingStatus.NOT_OVERTAKING,
                state=OvertakeState.NONE,
                confidence=0.75,
                evidence=("no_active_overtaking_sequence_detected",),
            )
        return OvertakingAssessment(
            track_id=track_id,
            status=OvertakingStatus.INSUFFICIENT_EVIDENCE,
            state=OvertakeState.NONE,
            confidence=0.30,
            evidence=("minimum_motion_history_not_reached",),
        )


def _append_once(evidence: list[str], code: str) -> None:
    if code not in evidence:
        evidence.append(code)

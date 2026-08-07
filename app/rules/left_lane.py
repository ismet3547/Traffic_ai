"""Configurable state machine for left-lane review candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import LeftLaneRuleConfig
from app.models import (
    CandidateTransition,
    LaneObservation,
    RuleEvaluation,
    VehicleRuleStatus,
)


class OvertakingClearancePolicy(Protocol):
    def is_cleared(self, observation: LaneObservation, timestamp_seconds: float) -> bool:
        """Return true only when definite overtaking evidence clears a candidate."""
        ...


class NoOvertakingClearancePolicy:
    """MVP policy: no overtaking inference is attempted."""

    def is_cleared(self, observation: LaneObservation, timestamp_seconds: float) -> bool:
        return False


@dataclass(slots=True)
class _LeftLaneState:
    entered_at: float
    last_seen_at: float
    confidence_sum: float
    observation_count: int
    candidate_active: bool = False
    cleared: bool = False

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.observation_count


class LeftLaneRuleEngine:
    def __init__(
        self,
        config: LeftLaneRuleConfig,
        clearance_policy: OvertakingClearancePolicy | None = None,
    ) -> None:
        self._config = config
        self._clearance_policy = clearance_policy or NoOvertakingClearancePolicy()
        self._states: dict[int, _LeftLaneState] = {}

    def evaluate(
        self, observations: list[LaneObservation], timestamp_seconds: float
    ) -> RuleEvaluation:
        statuses: dict[int, VehicleRuleStatus] = {}
        transitions: list[CandidateTransition] = []
        seen_ids = {observation.vehicle.track_id for observation in observations}

        if not self._config.enabled:
            for observation in observations:
                statuses[observation.vehicle.track_id] = VehicleRuleStatus(
                    track_id=observation.vehicle.track_id,
                    lane_id=observation.lane_id,
                    left_lane_duration_seconds=0.0,
                    is_review_candidate=False,
                )
            self._states.clear()
            return RuleEvaluation(statuses=statuses, transitions=[])

        for observation in observations:
            track_id = observation.vehicle.track_id
            if observation.lane_id != self._config.left_lane_id:
                transition = self._end_state(
                    track_id, timestamp_seconds, "left_lane_exit"
                )
                if transition:
                    transitions.append(transition)
                statuses[track_id] = VehicleRuleStatus(
                    track_id=track_id,
                    lane_id=observation.lane_id,
                    left_lane_duration_seconds=0.0,
                    is_review_candidate=False,
                )
                continue

            state = self._states.get(track_id)
            if state is None:
                state = _LeftLaneState(
                    entered_at=timestamp_seconds,
                    last_seen_at=timestamp_seconds,
                    confidence_sum=observation.vehicle.confidence,
                    observation_count=1,
                )
                self._states[track_id] = state
            else:
                state.last_seen_at = timestamp_seconds
                state.confidence_sum += observation.vehicle.confidence
                state.observation_count += 1

            duration = max(0.0, timestamp_seconds - state.entered_at)
            clears_candidate = self._clearance_policy.is_cleared(
                observation, timestamp_seconds
            )
            if clears_candidate:
                state.cleared = True
            if (
                not state.candidate_active
                and not state.cleared
                and duration >= self._config.occupancy_threshold_seconds
                and state.mean_confidence >= self._config.minimum_mean_confidence
            ):
                state.candidate_active = True
                transitions.append(
                    CandidateTransition(
                        transition="started",
                        track_id=track_id,
                        lane_id=self._config.left_lane_id,
                        start_timestamp_seconds=state.entered_at,
                        timestamp_seconds=timestamp_seconds,
                        duration_seconds=duration,
                        confidence_score=state.mean_confidence,
                    )
                )
            elif state.candidate_active and clears_candidate:
                state.candidate_active = False
                transitions.append(
                    CandidateTransition(
                        transition="ended",
                        track_id=track_id,
                        lane_id=self._config.left_lane_id,
                        start_timestamp_seconds=state.entered_at,
                        timestamp_seconds=timestamp_seconds,
                        duration_seconds=duration,
                        confidence_score=state.mean_confidence,
                        end_reason="definite_overtaking",
                    )
                )

            statuses[track_id] = VehicleRuleStatus(
                track_id=track_id,
                lane_id=observation.lane_id,
                left_lane_duration_seconds=duration,
                is_review_candidate=state.candidate_active,
            )

        for track_id, state in list(self._states.items()):
            if track_id in seen_ids:
                continue
            if timestamp_seconds - state.last_seen_at >= self._config.track_lost_grace_seconds:
                transition = self._end_state(
                    track_id, state.last_seen_at, "track_lost"
                )
                if transition:
                    transitions.append(transition)

        return RuleEvaluation(statuses=statuses, transitions=transitions)

    def finalize(self) -> list[CandidateTransition]:
        transitions: list[CandidateTransition] = []
        for track_id, state in list(self._states.items()):
            transition = self._end_state(track_id, state.last_seen_at, "video_ended")
            if transition:
                transitions.append(transition)
        return transitions

    def _end_state(
        self, track_id: int, timestamp_seconds: float, reason: str
    ) -> CandidateTransition | None:
        state = self._states.pop(track_id, None)
        if state is None or not state.candidate_active:
            return None
        duration = max(0.0, timestamp_seconds - state.entered_at)
        return CandidateTransition(
            transition="ended",
            track_id=track_id,
            lane_id=self._config.left_lane_id,
            start_timestamp_seconds=state.entered_at,
            timestamp_seconds=timestamp_seconds,
            duration_seconds=duration,
            confidence_score=state.mean_confidence,
            end_reason=reason,
        )

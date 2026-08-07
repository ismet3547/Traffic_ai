"""Persistent lane assignment and hysteretic lane-transition events."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import LaneChangeConfig
from app.models import LaneObservation, LaneTransition


@dataclass(frozen=True, slots=True)
class LaneChangeFrame:
    observations: list[LaneObservation]
    transitions: list[LaneTransition]


@dataclass(slots=True)
class _LaneState:
    confirmed_lane: str | None
    last_seen_at: float
    pending_lane: str | None = None
    pending_started_at: float | None = None
    pending_frames: int = 0
    has_pending: bool = False


class LaneTransitionDetector:
    """Debounces polygon crossings before changing a track's lane."""

    def __init__(self, config: LaneChangeConfig) -> None:
        self._config = config
        self._states: dict[int, _LaneState] = {}

    def update(
        self,
        observations: list[LaneObservation],
        timestamp_seconds: float,
    ) -> LaneChangeFrame:
        stabilized: list[LaneObservation] = []
        transitions: list[LaneTransition] = []
        seen_ids: set[int] = set()

        for observation in observations:
            track_id = observation.vehicle.track_id
            seen_ids.add(track_id)
            raw_lane = observation.lane_id
            state = self._states.get(track_id)
            if state is None:
                state = _LaneState(
                    confirmed_lane=raw_lane,
                    last_seen_at=timestamp_seconds,
                )
                self._states[track_id] = state
            else:
                state.last_seen_at = timestamp_seconds
                transition = self._update_state(
                    track_id, state, raw_lane, timestamp_seconds
                )
                if transition is not None:
                    transitions.append(transition)

            stabilized.append(
                LaneObservation(
                    vehicle=observation.vehicle,
                    lane_id=state.confirmed_lane,
                )
            )

        for track_id, state in list(self._states.items()):
            if track_id not in seen_ids and (
                timestamp_seconds - state.last_seen_at > self._config.state_ttl_seconds
            ):
                del self._states[track_id]

        return LaneChangeFrame(observations=stabilized, transitions=transitions)

    def _update_state(
        self,
        track_id: int,
        state: _LaneState,
        raw_lane: str | None,
        timestamp_seconds: float,
    ) -> LaneTransition | None:
        if raw_lane == state.confirmed_lane:
            self._clear_pending(state)
            return None

        if not state.has_pending or raw_lane != state.pending_lane:
            state.pending_lane = raw_lane
            state.pending_started_at = timestamp_seconds
            state.pending_frames = 1
            state.has_pending = True
            return None

        state.pending_frames += 1
        pending_started_at = (
            state.pending_started_at
            if state.pending_started_at is not None
            else timestamp_seconds
        )
        pending_duration = timestamp_seconds - pending_started_at
        if (
            state.pending_frames < self._config.minimum_frames
            or pending_duration < self._config.confirmation_seconds
        ):
            return None

        previous_lane = state.confirmed_lane
        state.confirmed_lane = raw_lane
        self._clear_pending(state)
        if previous_lane is None or raw_lane is None or previous_lane == raw_lane:
            return None
        return LaneTransition(
            track_id=track_id,
            from_lane=previous_lane,
            to_lane=raw_lane,
            timestamp_seconds=timestamp_seconds,
        )

    @staticmethod
    def _clear_pending(state: _LaneState) -> None:
        state.pending_lane = None
        state.pending_started_at = None
        state.pending_frames = 0
        state.has_pending = False

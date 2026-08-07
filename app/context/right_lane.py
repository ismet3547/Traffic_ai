"""Persist realistic opportunities to move into the adjacent right lane."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import RightLaneOpportunityConfig
from app.models import CongestionLevel, TrafficFrameContext


@dataclass(slots=True)
class _OpportunityState:
    available_since: float | None
    last_seen_at: float
    adjacent_lane_id: str | None


class RightLaneOpportunityTracker:
    def __init__(self, config: RightLaneOpportunityConfig) -> None:
        self._config = config
        self._states: dict[int, _OpportunityState] = {}

    def update(
        self, context: TrafficFrameContext, timestamp_seconds: float
    ) -> TrafficFrameContext:
        updated = {}
        seen_ids = set(context.vehicles)
        congested = context.global_context.congestion_level in {
            CongestionLevel.DENSE,
            CongestionLevel.STOP_AND_GO,
        }
        for track_id, vehicle_context in context.vehicles.items():
            state = self._states.setdefault(
                track_id,
                _OpportunityState(
                    available_since=None,
                    last_seen_at=timestamp_seconds,
                    adjacent_lane_id=vehicle_context.adjacent_right_lane_id,
                ),
            )
            state.last_seen_at = timestamp_seconds
            if state.adjacent_lane_id != vehicle_context.adjacent_right_lane_id:
                state.available_since = None
                state.adjacent_lane_id = vehicle_context.adjacent_right_lane_id
            neighbors = vehicle_context.neighbors
            if vehicle_context.adjacent_right_lane_id is None:
                available: bool | None = None
                confidence = 0.0
            else:
                front_clear = (
                    neighbors.adjacent_right_ahead is None
                    or neighbors.adjacent_right_ahead.longitudinal_gap
                    >= self._config.front_gap_normalized
                )
                rear_clear = (
                    neighbors.adjacent_right_behind is None
                    or neighbors.adjacent_right_behind.longitudinal_gap
                    >= self._config.rear_gap_normalized
                )
                available = front_clear and rear_clear and not congested
                observed_bounds = sum(
                    reference is not None
                    for reference in (
                        neighbors.adjacent_right_ahead,
                        neighbors.adjacent_right_behind,
                    )
                )
                confidence = min(
                    1.0,
                    0.70
                    + 0.10 * observed_bounds
                    + 0.10 * context.global_context.confidence,
                )

            if available:
                if state.available_since is None:
                    state.available_since = timestamp_seconds
                duration = max(0.0, timestamp_seconds - state.available_since)
            else:
                state.available_since = None
                duration = 0.0
            updated[track_id] = replace(
                vehicle_context,
                right_lane_available=available,
                right_lane_available_seconds=duration,
                right_lane_confidence=confidence,
            )

        for track_id, state in list(self._states.items()):
            if track_id not in seen_ids and (
                timestamp_seconds - state.last_seen_at > self._config.state_ttl_seconds
            ):
                del self._states[track_id]

        return TrafficFrameContext(
            global_context=context.global_context,
            vehicles=updated,
            positions=context.positions,
        )

"""Persist realistic opportunities to move into the adjacent right lane."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import RightLaneOpportunityConfig
from app.models import (
    CongestionLevel,
    GapEstimate,
    NeighborReference,
    TrafficFrameContext,
)


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
                opportunity_mode = "unavailable"
            else:
                front_reference = neighbors.adjacent_right_ahead
                rear_reference = neighbors.adjacent_right_behind
                gap_unit = next(
                    (
                        reference.gap_unit
                        for reference in (front_reference, rear_reference)
                        if reference is not None
                    ),
                    "meters"
                    if context.positions[track_id].calibrated
                    else "normalized",
                )
                use_meters = gap_unit == "meters" and self._config.mode != "normalized"
                if self._config.mode == "calibrated" and gap_unit != "meters":
                    available = None
                    confidence = 0.0
                    opportunity_mode = "unavailable_uncalibrated"
                    front_clear = rear_clear = False
                else:
                    opportunity_mode = "calibrated" if use_meters else "normalized"
                    front_threshold = (
                        self._config.minimum_front_gap_m
                        if use_meters
                        else self._config.front_gap_normalized
                    )
                    rear_threshold = (
                        self._config.minimum_rear_gap_m
                        if use_meters
                        else self._config.rear_gap_normalized
                    )
                    front_clear = (
                        front_reference is None
                        or front_reference.longitudinal_gap >= front_threshold
                    )
                    rear_clear = (
                        rear_reference is None
                        or rear_reference.longitudinal_gap >= rear_threshold
                    )
                    available = front_clear and rear_clear and not congested
                    observed_bounds = sum(
                        reference is not None
                        for reference in (front_reference, rear_reference)
                    )
                    position_confidence = (
                        context.positions[track_id].calibration_confidence
                        if use_meters
                        else 0.65
                    )
                    confidence = min(
                        1.0,
                        0.55
                        + 0.10 * observed_bounds
                        + 0.15 * context.global_context.confidence
                        + 0.20 * position_confidence,
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
                right_lane_front_gap=_gap_estimate(neighbors.adjacent_right_ahead),
                right_lane_rear_gap=_gap_estimate(neighbors.adjacent_right_behind),
                right_lane_opportunity_mode=opportunity_mode,
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
            speeds=context.speeds,
        )


def _gap_estimate(reference: NeighborReference | None) -> GapEstimate | None:
    if reference is None:
        return None
    return GapEstimate(
        value=reference.longitudinal_gap,
        unit=reference.gap_unit,
        confidence=reference.confidence,
        coordinate_mode=reference.coordinate_mode,
    )

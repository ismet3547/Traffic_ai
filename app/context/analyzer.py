"""Explainable frame-level traffic context in road-coordinate space."""

from __future__ import annotations

from statistics import fmean

from app.config import CongestionConfig, TrafficContextConfig
from app.models import (
    CongestionLevel,
    GlobalTrafficContext,
    LaneObservation,
    NeighborReference,
    NeighborVehicles,
    RoadPosition,
    TrafficFrameContext,
    VehicleTrafficContext,
)
from app.motion import MotionHistoryStore


class TrafficContextAnalyzer:
    def __init__(
        self,
        lane_ids_left_to_right: list[str],
        context_config: TrafficContextConfig,
        congestion_config: CongestionConfig,
    ) -> None:
        self._lane_ids = lane_ids_left_to_right
        self._context_config = context_config
        self._congestion_config = congestion_config

    def analyze(
        self,
        timestamp_seconds: float,
        observations: list[LaneObservation],
        positions: dict[int, RoadPosition],
        history: MotionHistoryStore,
    ) -> TrafficFrameContext:
        lane_by_track = {
            observation.vehicle.track_id: observation.lane_id
            for observation in observations
        }
        lane_counts = {lane_id: 0 for lane_id in self._lane_ids}
        for lane_id in lane_by_track.values():
            if lane_id in lane_counts:
                lane_counts[lane_id] += 1

        global_context = self._global_context(
            timestamp_seconds, positions, lane_counts, history
        )
        vehicles: dict[int, VehicleTrafficContext] = {}
        for track_id, position in positions.items():
            lane_id = lane_by_track.get(track_id)
            adjacent_right = self._adjacent_right(lane_id)
            neighbors = self._neighbors(
                track_id, lane_id, adjacent_right, lane_by_track, positions
            )
            nearby_count = sum(
                1
                for other_id, other_position in positions.items()
                if other_id != track_id
                and abs(other_position.longitudinal - position.longitudinal)
                <= self._context_config.nearby_longitudinal_window_normalized
            )
            vehicles[track_id] = VehicleTrafficContext(
                track_id=track_id,
                neighbors=neighbors,
                nearby_vehicle_count=nearby_count,
                adjacent_right_lane_id=adjacent_right,
                right_lane_available=None,
                right_lane_available_seconds=0.0,
                right_lane_confidence=0.0,
            )
        return TrafficFrameContext(
            global_context=global_context,
            vehicles=vehicles,
            positions=positions,
        )

    def _global_context(
        self,
        timestamp_seconds: float,
        positions: dict[int, RoadPosition],
        lane_counts: dict[str, int],
        history: MotionHistoryStore,
    ) -> GlobalTrafficContext:
        total = sum(lane_counts.values())
        lane_count = max(1, len(self._lane_ids))
        dense_capacity = max(
            1, self._congestion_config.dense_vehicle_count_per_lane * lane_count
        )
        density = min(1.0, total / dense_capacity)
        motions: list[float] = []
        for track_id, position in positions.items():
            previous = history.latest(track_id)
            if previous is None:
                continue
            elapsed = timestamp_seconds - previous.timestamp_seconds
            if elapsed > 0:
                motions.append(
                    abs(position.longitudinal - previous.longitudinal_position)
                    / elapsed
                )
        average_motion = fmean(motions) if motions else None

        level = self._congestion_level(total, density, lane_counts, average_motion)
        if level == CongestionLevel.UNKNOWN:
            confidence = 0.0
        else:
            motion_coverage = len(motions) / max(1, len(positions))
            confidence = min(1.0, 0.65 + 0.35 * motion_coverage)
        coordinate_system = (
            next(iter(positions.values())).coordinate_system
            if positions
            else "normalized_image"
        )
        return GlobalTrafficContext(
            congestion_level=level,
            traffic_density=density,
            active_vehicle_count=total,
            lane_vehicle_counts=lane_counts,
            average_normalized_motion_per_second=average_motion,
            confidence=confidence,
            coordinate_system=coordinate_system,
        )

    def _congestion_level(
        self,
        total: int,
        density: float,
        lane_counts: dict[str, int],
        average_motion: float | None,
    ) -> CongestionLevel:
        config = self._congestion_config
        if not config.enabled or total < config.minimum_observed_vehicles:
            return CongestionLevel.UNKNOWN
        every_lane_dense = all(
            count >= config.dense_vehicle_count_per_lane
            for count in lane_counts.values()
        )
        if every_lane_dense or density >= config.dense_density_ratio:
            if (
                average_motion is not None
                and average_motion
                <= config.stop_and_go_max_motion_per_second_normalized
            ):
                return CongestionLevel.STOP_AND_GO
            return CongestionLevel.DENSE
        if density >= config.moderate_density_ratio:
            if (
                average_motion is not None
                and average_motion <= config.dense_max_motion_per_second_normalized
            ):
                return CongestionLevel.DENSE
            return CongestionLevel.MODERATE
        return CongestionLevel.FREE_FLOW

    def _adjacent_right(self, lane_id: str | None) -> str | None:
        if lane_id not in self._lane_ids:
            return None
        index = self._lane_ids.index(lane_id)
        return self._lane_ids[index + 1] if index + 1 < len(self._lane_ids) else None

    @staticmethod
    def _neighbors(
        track_id: int,
        lane_id: str | None,
        adjacent_right: str | None,
        lane_by_track: dict[int, str | None],
        positions: dict[int, RoadPosition],
    ) -> NeighborVehicles:
        target = positions[track_id]

        def nearest(lane: str | None, ahead: bool) -> NeighborReference | None:
            if lane is None:
                return None
            candidates: list[NeighborReference] = []
            for other_id, other in positions.items():
                if other_id == track_id or lane_by_track.get(other_id) != lane:
                    continue
                signed_gap = other.longitudinal - target.longitudinal
                if (ahead and signed_gap > 0) or (not ahead and signed_gap < 0):
                    candidates.append(
                        NeighborReference(
                            track_id=other_id,
                            longitudinal_gap=abs(signed_gap),
                        )
                    )
            return min(candidates, key=lambda item: item.longitudinal_gap, default=None)

        return NeighborVehicles(
            same_lane_ahead=nearest(lane_id, True),
            same_lane_behind=nearest(lane_id, False),
            adjacent_right_ahead=nearest(adjacent_right, True),
            adjacent_right_behind=nearest(adjacent_right, False),
        )

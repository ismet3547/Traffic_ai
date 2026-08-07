"""Bounded, reusable motion history for persistent vehicle tracks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from app.config import TrafficContextConfig
from app.models import (
    LaneObservation,
    LaneTransition,
    RoadPosition,
    SpeedEstimate,
    VehicleTrafficContext,
)


@dataclass(frozen=True, slots=True)
class TrackMotionSample:
    track_id: int
    timestamp_seconds: float
    frame_index: int
    bounding_box_center: tuple[float, float]
    road_contact_point: tuple[float, float]
    lane_id: str | None
    detector_confidence: float
    longitudinal_position: float
    lateral_position: float
    estimated_longitudinal_progress: float
    estimated_lateral_movement: float
    neighboring_vehicle_track_ids: tuple[int, ...]
    lane_transition: LaneTransition | None = None
    image_position: tuple[float, float] | None = None
    normalized_position: tuple[float, float] | None = None
    world_position: tuple[float, float] | None = None
    coordinate_mode: str = "normalized_image"
    calibration_confidence: float = 0.0
    speed_estimate: SpeedEstimate | None = None


class MotionHistoryStore:
    """Time- and size-bounded histories independent of detector/tracker vendors."""

    def __init__(self, config: TrafficContextConfig) -> None:
        self._config = config
        self._histories: dict[int, deque[TrackMotionSample]] = {}

    def update(
        self,
        frame_index: int,
        timestamp_seconds: float,
        observations: list[LaneObservation],
        positions: dict[int, RoadPosition],
        vehicle_contexts: dict[int, VehicleTrafficContext],
        transitions: list[LaneTransition],
        speed_estimates: dict[int, SpeedEstimate] | None = None,
    ) -> None:
        transition_by_track = {
            transition.track_id: transition for transition in transitions
        }
        for observation in observations:
            track_id = observation.vehicle.track_id
            position = positions.get(track_id)
            if position is None:
                continue
            history = self._histories.setdefault(
                track_id,
                deque(maxlen=self._config.maximum_samples_per_track),
            )
            previous = history[-1] if history else None
            longitudinal_progress = (
                position.longitudinal - previous.longitudinal_position
                if previous is not None
                else 0.0
            )
            lateral_movement = (
                position.lateral - previous.lateral_position
                if previous is not None
                else 0.0
            )
            bbox = observation.vehicle.bbox
            center = ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)
            context = vehicle_contexts.get(track_id)
            neighbor_ids = context.neighbors.track_ids if context is not None else ()
            history.append(
                TrackMotionSample(
                    track_id=track_id,
                    timestamp_seconds=timestamp_seconds,
                    frame_index=frame_index,
                    bounding_box_center=center,
                    road_contact_point=bbox.bottom_center,
                    lane_id=observation.lane_id,
                    detector_confidence=observation.vehicle.confidence,
                    longitudinal_position=position.longitudinal,
                    lateral_position=position.lateral,
                    estimated_longitudinal_progress=longitudinal_progress,
                    estimated_lateral_movement=lateral_movement,
                    neighboring_vehicle_track_ids=neighbor_ids,
                    lane_transition=transition_by_track.get(track_id),
                    image_position=position.image_position,
                    normalized_position=position.normalized_position,
                    world_position=position.world_position,
                    coordinate_mode=position.coordinate_mode,
                    calibration_confidence=position.calibration_confidence,
                    speed_estimate=(speed_estimates or {}).get(track_id),
                )
            )

        cutoff = timestamp_seconds - self._config.history_seconds
        for track_id, history in list(self._histories.items()):
            while history and history[0].timestamp_seconds < cutoff:
                history.popleft()
            if not history:
                del self._histories[track_id]

    def history(self, track_id: int) -> tuple[TrackMotionSample, ...]:
        return tuple(self._histories.get(track_id, ()))

    def latest(self, track_id: int) -> TrackMotionSample | None:
        history = self._histories.get(track_id)
        return history[-1] if history else None

    def duration_seconds(self, track_id: int) -> float:
        history = self._histories.get(track_id)
        if not history or len(history) < 2:
            return 0.0
        return max(0.0, history[-1].timestamp_seconds - history[0].timestamp_seconds)

    def track_ids(self) -> tuple[int, ...]:
        return tuple(self._histories)

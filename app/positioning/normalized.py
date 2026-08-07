"""Uncalibrated normalized image-space road positions."""

from __future__ import annotations

from app.config import RoadPositionConfig
from app.models import LaneObservation, RoadPosition


class NormalizedImageRoadPositionEstimator:
    """Maps road-contact pixels to dimensionless normalized coordinates.

    Longitudinal values increase in the configured direction of travel. They are
    useful for ordering vehicles in one camera view, but are not physical distance.
    """

    coordinate_system = "normalized_image"
    calibrated = False

    def __init__(self, config: RoadPositionConfig) -> None:
        self._config = config

    def estimate(
        self,
        observations: list[LaneObservation],
        frame_width: int,
        frame_height: int,
    ) -> dict[int, RoadPosition]:
        width = max(1, frame_width)
        height = max(1, frame_height)
        positions: dict[int, RoadPosition] = {}
        for observation in observations:
            x, y = observation.vehicle.bbox.bottom_center
            lateral = _clamp(x / width)
            image_y = _clamp(y / height)
            longitudinal = (
                image_y
                if self._config.travel_direction == "toward_bottom"
                else 1.0 - image_y
            )
            positions[observation.vehicle.track_id] = RoadPosition(
                track_id=observation.vehicle.track_id,
                lateral=lateral,
                longitudinal=longitudinal,
                coordinate_system="normalized_image",
                calibrated=False,
            )
        return positions


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

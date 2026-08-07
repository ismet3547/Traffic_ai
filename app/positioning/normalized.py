"""Uncalibrated normalized image-space road positions."""

from __future__ import annotations

from app.config import RoadPositionConfig
from app.models import (
    CalibrationQuality,
    LaneObservation,
    PhysicalMeasurementPermission,
    RoadPosition,
)


class NormalizedImageRoadPositionEstimator:
    """Maps road-contact pixels to dimensionless normalized coordinates.

    Longitudinal values increase in the configured direction of travel. They are
    useful for ordering vehicles in one camera view, but are not physical distance.
    """

    coordinate_system = "normalized_image"
    calibrated = False

    def __init__(
        self,
        config: RoadPositionConfig,
        calibration_quality: CalibrationQuality | None = None,
    ) -> None:
        self._config = config
        self._calibration_quality = calibration_quality or CalibrationQuality(
            mode="normalized",
            matrix_valid=False,
            numerically_stable=False,
            validation_mode="NONE",
            fit_reprojection_error_pixels=None,
            validation_reprojection_error_pixels=None,
            condition_metric=None,
            confidence=0.0,
            confidence_basis="not_calibrated",
            reason_codes=("CALIBRATION_NOT_CONFIGURED",),
            world_units=None,
        )

    @property
    def calibration_quality(self) -> CalibrationQuality:
        return self._calibration_quality

    @property
    def calibration_status(self) -> CalibrationQuality:
        return self._calibration_quality

    def estimate(
        self,
        observations: list[LaneObservation],
        frame_width: int,
        frame_height: int,
        physical_permission: PhysicalMeasurementPermission | None = None,
    ) -> dict[int, RoadPosition]:
        del physical_permission
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
                image_position=(x, y),
                normalized_position=(lateral, longitudinal),
                world_position_m=None,
                world_position_confidence=0.0,
                physical_measurement_status="unavailable",
                physical_measurement_reason_codes=("CALIBRATION_NOT_CONFIGURED",),
            )
        return positions


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

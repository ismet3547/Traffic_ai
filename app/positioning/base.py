"""Road-position estimator contract."""

from __future__ import annotations

from typing import Protocol

from app.models import (
    CalibrationQuality,
    LaneObservation,
    PhysicalMeasurementPermission,
    RoadPosition,
)


class RoadCoordinateTransformer(Protocol):
    @property
    def coordinate_system(self) -> str: ...

    @property
    def calibrated(self) -> bool: ...

    @property
    def calibration_quality(self) -> CalibrationQuality: ...

    @property
    def calibration_status(self) -> CalibrationQuality: ...

    def estimate(
        self,
        observations: list[LaneObservation],
        frame_width: int,
        frame_height: int,
        physical_permission: PhysicalMeasurementPermission | None = None,
    ) -> dict[int, RoadPosition]:
        """Estimate comparable road positions for the current frame."""
        ...


# Phase 2 compatibility name. New code should use RoadCoordinateTransformer.
RoadPositionEstimator = RoadCoordinateTransformer

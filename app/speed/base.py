"""Speed estimator contract independent of detection and tracking vendors."""

from __future__ import annotations

from typing import Protocol

from app.models import PhysicalMeasurementPermission, RoadPosition, SpeedEstimate


class SpeedEstimator(Protocol):
    def update(
        self,
        timestamp_seconds: float,
        positions: dict[int, RoadPosition],
        physical_permission: PhysicalMeasurementPermission | None = None,
    ) -> dict[int, SpeedEstimate]: ...

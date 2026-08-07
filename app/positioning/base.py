"""Road-position estimator contract."""

from __future__ import annotations

from typing import Protocol

from app.models import LaneObservation, RoadPosition


class RoadPositionEstimator(Protocol):
    @property
    def coordinate_system(self) -> str: ...

    @property
    def calibrated(self) -> bool: ...

    def estimate(
        self,
        observations: list[LaneObservation],
        frame_width: int,
        frame_height: int,
    ) -> dict[int, RoadPosition]:
        """Estimate comparable road positions for the current frame."""
        ...

"""Tracker contract."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.models import Detection, TrackedVehicle


class VehicleTracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> list[TrackedVehicle]:
        """Associate current detections with persistent track IDs."""
        ...

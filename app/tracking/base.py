"""Tracker contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.models import Detection, TrackedVehicle


class VehicleTracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> list[TrackedVehicle]:
        """Associate current detections with persistent track IDs."""
        ...

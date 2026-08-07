"""Multi-object tracking interfaces and implementations."""

from .base import VehicleTracker
from .bytetrack_tracker import ByteTrackVehicleTracker

__all__ = ["ByteTrackVehicleTracker", "VehicleTracker"]

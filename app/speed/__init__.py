"""Approximate motion-rate and calibrated physical-speed estimation."""

from .base import SpeedEstimator
from .rolling import RollingSpeedEstimator

__all__ = ["RollingSpeedEstimator", "SpeedEstimator"]

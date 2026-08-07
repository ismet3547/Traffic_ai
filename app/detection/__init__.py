"""Vehicle detection interfaces and implementations."""

from .base import Detector
from .ultralytics_detector import UltralyticsDetector

__all__ = ["Detector", "UltralyticsDetector"]

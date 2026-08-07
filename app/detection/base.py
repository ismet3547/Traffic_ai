"""Detector contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.models import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR frame."""
        ...

"""Camera-motion estimator contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.models import BoundingBox, CameraMotionEstimate


class CameraMotionEstimator(Protocol):
    def update(
        self, frame: np.ndarray, excluded_boxes: list[BoundingBox] | None = None
    ) -> CameraMotionEstimate: ...

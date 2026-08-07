"""Fixed-camera implementation with no estimated motion."""

from __future__ import annotations

import numpy as np

from app.models import BoundingBox, CameraMotionEstimate


class NoCameraMotionEstimator:
    def update(
        self, frame: np.ndarray, excluded_boxes: list[BoundingBox] | None = None
    ) -> CameraMotionEstimate:
        del frame, excluded_boxes
        return CameraMotionEstimate(
            dx=0.0,
            dy=0.0,
            rotation_degrees=0.0,
            confidence=1.0,
            valid=True,
            level="low",
            method="none_fixed_camera",
        )

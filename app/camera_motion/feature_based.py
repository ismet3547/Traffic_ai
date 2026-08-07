"""Experimental background-feature camera-motion diagnostic."""

from __future__ import annotations

import math

import numpy as np

from app.config import CameraMotionConfig
from app.models import BoundingBox, CameraMotionEstimate


class FeatureBasedCameraMotionEstimator:
    """Estimates frame-to-frame affine motion; it does not stabilize frames."""

    def __init__(self, config: CameraMotionConfig) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "feature-based camera motion requires opencv-python"
            ) from exc
        self._cv2 = cv2
        self._config = config
        self._previous_gray: np.ndarray | None = None

    def update(
        self, frame: np.ndarray, excluded_boxes: list[BoundingBox] | None = None
    ) -> CameraMotionEstimate:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            self._previous_gray = gray
            return _invalid("feature_based_initializing")

        previous_gray = self._previous_gray
        mask = np.full(previous_gray.shape, 255, dtype=np.uint8)
        if self._config.mask_vehicle_boxes:
            for box in excluded_boxes or []:
                cv2.rectangle(
                    mask,
                    (max(0, int(box.x1)), max(0, int(box.y1))),
                    (max(0, int(box.x2)), max(0, int(box.y2))),
                    0,
                    thickness=-1,
                )
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self._config.maximum_features,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
        )
        if (
            previous_points is None
            or len(previous_points) < self._config.minimum_tracked_features
        ):
            self._previous_gray = gray
            return _invalid("feature_based_insufficient_features")
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
        )
        self._previous_gray = gray
        if current_points is None or status is None:
            return _invalid("feature_based_optical_flow_failed")
        valid_mask = status.reshape(-1).astype(bool)
        previous_valid = previous_points.reshape(-1, 2)[valid_mask]
        current_valid = current_points.reshape(-1, 2)[valid_mask]
        if len(current_valid) < self._config.minimum_tracked_features:
            return _invalid("feature_based_insufficient_tracked_features")
        affine, inliers = cv2.estimateAffinePartial2D(
            previous_valid,
            current_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if affine is None:
            return _invalid("feature_based_affine_failed")
        dx = float(affine[0, 2])
        dy = float(affine[1, 2])
        rotation = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        inlier_ratio = (
            float(np.mean(inliers)) if inliers is not None and len(inliers) else 0.0
        )
        feature_coverage = min(1.0, len(current_valid) / self._config.maximum_features)
        confidence = min(1.0, 0.75 * inlier_ratio + 0.25 * feature_coverage)
        translation = math.hypot(dx, dy)
        threshold = self._config.excessive_translation_pixels
        if translation >= threshold:
            level = "high"
        elif translation >= threshold * 0.5:
            level = "moderate"
        else:
            level = "low"
        valid = confidence >= self._config.minimum_confidence
        return CameraMotionEstimate(
            dx=dx,
            dy=dy,
            rotation_degrees=rotation,
            confidence=confidence,
            valid=valid,
            level=level if valid else "unknown",
            method="feature_based_lk_affine_experimental",
        )


def _invalid(method: str) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        dx=0.0,
        dy=0.0,
        rotation_degrees=0.0,
        confidence=0.0,
        valid=False,
        level="unknown",
        method=method,
    )

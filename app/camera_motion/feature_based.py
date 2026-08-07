"""Experimental background-feature camera-motion diagnostic."""

from __future__ import annotations

import math

import numpy as np

from app.config import CameraMotionConfig
from app.models import BoundingBox, CameraMotionEstimate, ProjectivePoseEstimate


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
        self._reference_gray: np.ndarray | None = None
        self._reference_points: np.ndarray | None = None
        self._frame_index = -1

    def update(
        self, frame: np.ndarray, excluded_boxes: list[BoundingBox] | None = None
    ) -> CameraMotionEstimate:
        cv2 = self._cv2
        self._frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            self._previous_gray = gray
            reference_mask = self._background_mask(gray.shape, excluded_boxes)
            self._reference_gray = gray
            self._reference_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self._config.maximum_features,
                qualityLevel=0.01,
                minDistance=8,
                mask=reference_mask,
            )
            return _invalid("feature_based_initializing")

        previous_gray = self._previous_gray
        mask = self._background_mask(previous_gray.shape, excluded_boxes)
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
        scale_ratio = _similarity_scale(affine)
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
        projective = self._reference_projective(gray)
        return CameraMotionEstimate(
            dx=dx,
            dy=dy,
            rotation_degrees=rotation,
            confidence=confidence,
            valid=valid,
            level=level if valid else "unknown",
            method="feature_based_lk_affine_experimental",
            stabilization_applied=False,
            scale_ratio=scale_ratio,
            scale_delta=scale_ratio - 1.0,
            scale_confidence=confidence,
            projective=projective,
        )

    def _background_mask(
        self,
        shape: tuple[int, ...],
        excluded_boxes: list[BoundingBox] | None,
    ) -> np.ndarray:
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        if self._config.mask_vehicle_boxes:
            for box in excluded_boxes or []:
                self._cv2.rectangle(
                    mask,
                    (max(0, int(box.x1)), max(0, int(box.y1))),
                    (max(0, int(box.x2)), max(0, int(box.y2))),
                    0,
                    thickness=-1,
                )
        return mask

    def _reference_projective(self, gray: np.ndarray) -> ProjectivePoseEstimate | None:
        """Sample projective drift against the startup reference frame."""

        if self._frame_index % self._config.reference_analysis_interval_frames:
            return None
        cv2 = self._cv2
        if self._reference_gray is None or self._reference_points is None:
            return _invalid_projective("REFERENCE_FEATURES_UNAVAILABLE")
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            self._reference_gray, gray, self._reference_points, None
        )
        if current is None or status is None:
            return _invalid_projective("REFERENCE_OPTICAL_FLOW_FAILED")
        selected = status.reshape(-1).astype(bool)
        source = self._reference_points.reshape(-1, 2)[selected]
        destination = current.reshape(-1, 2)[selected]
        if len(source) < max(4, self._config.minimum_tracked_features):
            return _invalid_projective("REFERENCE_FEATURES_INSUFFICIENT")
        homography, inliers = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            self._config.projective_ransac_threshold_pixels,
        )
        affine, _ = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._config.projective_ransac_threshold_pixels,
        )
        if homography is None or affine is None:
            return _invalid_projective("REFERENCE_PROJECTIVE_FIT_FAILED")
        inlier_mask = (
            inliers.reshape(-1).astype(bool)
            if inliers is not None
            else np.ones(len(source), dtype=bool)
        )
        source_h = source.reshape(-1, 1, 2).astype(np.float32)
        projected = cv2.perspectiveTransform(source_h, homography).reshape(-1, 2)
        errors = np.linalg.norm(
            projected[inlier_mask] - destination[inlier_mask], axis=1
        )
        reprojection_error = float(np.mean(errors)) if len(errors) else math.inf
        height, width = gray.shape[:2]
        drift_score = _projective_corner_drift(homography, affine, width, height, cv2)
        inlier_ratio = float(np.mean(inlier_mask))
        confidence = min(1.0, inlier_ratio * min(1.0, len(source) / 50.0))
        return ProjectivePoseEstimate(
            valid=confidence >= self._config.minimum_confidence,
            drift_score=drift_score,
            reprojection_error_pixels=reprojection_error,
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            reference_frame_index=0,
            sample_frame_index=self._frame_index,
            method="frame_to_reference_lk_homography_diagnostic",
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
        stabilization_applied=False,
    )


def _invalid_projective(reason: str) -> ProjectivePoseEstimate:
    return ProjectivePoseEstimate(
        valid=False,
        drift_score=None,
        reprojection_error_pixels=None,
        inlier_ratio=0.0,
        confidence=0.0,
        method="frame_to_reference_lk_homography_diagnostic",
        reason_codes=(reason,),
    )


def _similarity_scale(affine: np.ndarray) -> float:
    """Mean norm of both linear columns of a partial-affine transform."""

    scale_x = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
    scale_y = math.hypot(float(affine[0, 1]), float(affine[1, 1]))
    return (scale_x + scale_y) / 2.0


def _projective_corner_drift(
    homography: np.ndarray,
    affine: np.ndarray,
    width: int,
    height: int,
    cv2: object,
) -> float:
    """Normalized corner residual after removing the best similarity motion."""

    corners = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(
        -1, 2
    )
    similarity = np.column_stack((corners, np.ones(4))) @ affine.T
    diagonal = max(1.0, math.hypot(width, height))
    return float(np.mean(np.linalg.norm(projected - similarity, axis=1)) / diagonal)

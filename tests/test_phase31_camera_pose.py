from __future__ import annotations

import numpy as np

from app.camera_motion import (
    CameraPoseValidator,
    FeatureBasedCameraMotionEstimator,
    NoCameraMotionEstimator,
)
from app.config import CameraMotionConfig, CameraPoseValidationConfig
from app.models import BoundingBox, CameraMotionEstimate


def _motion(dx: float = 0.0, rotation: float = 0.0) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        dx=dx,
        dy=0.0,
        rotation_degrees=rotation,
        confidence=0.9,
        valid=True,
        level="low",
        method="synthetic_diagnostic",
        stabilization_applied=False,
    )


def _validator() -> CameraPoseValidator:
    return CameraPoseValidator(
        CameraPoseValidationConfig(
            minimum_samples=3,
            translation_warning_px=1.5,
            translation_invalid_px=3.0,
            rotation_warning_deg=0.15,
            rotation_invalid_deg=0.35,
            persistence_seconds=0.4,
        )
    )


def test_stable_fixed_camera_becomes_valid() -> None:
    validator = _validator()
    status = None
    for index in range(4):
        status = validator.update(index * 0.2, _motion())
    assert status is not None and status.status == "stable"


def test_tiny_feature_noise_does_not_invalidate_pose() -> None:
    validator = _validator()
    status = None
    for index in range(8):
        status = validator.update(index * 0.2, _motion(dx=0.1, rotation=0.01))
    assert status is not None and status.status == "stable"


def test_persistent_translation_invalidates_static_pose() -> None:
    validator = _validator()
    status = None
    for index in range(8):
        status = validator.update(index * 0.2, _motion(dx=1.0))
    assert status is not None and status.status == "moved"
    assert "CAMERA_POSE_UNSTABLE" in status.reason_codes


def test_persistent_rotation_invalidates_static_pose() -> None:
    validator = _validator()
    status = None
    for index in range(9):
        status = validator.update(index * 0.2, _motion(rotation=0.1))
    assert status is not None and status.status == "moved"


def test_motion_diagnostic_never_claims_stabilization() -> None:
    estimate = NoCameraMotionEstimator().update(
        np.zeros((8, 8, 3), dtype=np.uint8), [BoundingBox(0, 0, 2, 2)]
    )
    assert not estimate.stabilization_applied
    assert not estimate.valid
    assert estimate.method == "not_configured_no_motion_measurement"
    experimental = FeatureBasedCameraMotionEstimator(
        CameraMotionConfig(mode="feature_based")
    )
    initializing = experimental.update(np.zeros((32, 32, 3), dtype=np.uint8))
    assert not initializing.stabilization_applied

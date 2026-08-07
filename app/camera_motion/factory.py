"""Camera-motion estimator factory."""

from app.config import CameraMotionConfig

from .base import CameraMotionEstimator
from .feature_based import FeatureBasedCameraMotionEstimator
from .none import NoCameraMotionEstimator


def build_camera_motion_estimator(config: CameraMotionConfig) -> CameraMotionEstimator:
    if config.mode == "feature_based":
        return FeatureBasedCameraMotionEstimator(config)
    return NoCameraMotionEstimator()

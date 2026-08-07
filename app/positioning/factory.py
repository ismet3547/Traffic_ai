"""Coordinate-transformer construction and explicit fallback policy."""

from __future__ import annotations

import logging

from app.config import CalibrationConfig, RoadPositionConfig
from app.models import CalibrationQuality

from .base import RoadCoordinateTransformer
from .homography import CalibrationError, HomographyRoadTransformer
from .normalized import NormalizedImageRoadPositionEstimator

LOGGER = logging.getLogger(__name__)


def build_road_coordinate_transformer(
    calibration: CalibrationConfig,
    road_position: RoadPositionConfig,
) -> RoadCoordinateTransformer:
    if calibration.mode == "normalized":
        return NormalizedImageRoadPositionEstimator(road_position)
    try:
        return HomographyRoadTransformer(calibration, road_position)
    except CalibrationError:
        if not calibration.fallback_to_normalized:
            raise
        LOGGER.exception(
            "Homography initialization failed; using explicitly configured normalized fallback"
        )
        return NormalizedImageRoadPositionEstimator(
            road_position,
            CalibrationQuality(
                mode="homography_fallback",
                matrix_valid=False,
                numerically_stable=False,
                validation_mode="NONE",
                fit_reprojection_error_pixels=None,
                validation_reprojection_error_pixels=None,
                condition_metric=None,
                confidence=0.0,
                confidence_basis="homography_initialization_failed",
                reason_codes=("HOMOGRAPHY_INITIALIZATION_FAILED",),
                world_units=None,
            ),
        )

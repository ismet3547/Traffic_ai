"""Coordinate-transformer construction and explicit fallback policy."""

from __future__ import annotations

import logging

from app.config import CalibrationConfig, RoadPositionConfig
from app.models import CalibrationStatus

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
            CalibrationStatus(
                mode="homography_fallback",
                valid=False,
                reprojection_error_pixels=None,
                confidence=0.0,
                reason="configured homography failed validation; normalized fallback active",
                world_units=None,
            ),
        )

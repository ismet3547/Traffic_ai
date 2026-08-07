from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import (
    CalibrationConfig,
    CongestionConfig,
    PhysicalMeasurementsConfig,
    RoadPositionConfig,
    TrafficContextConfig,
)
from app.context import TrafficContextAnalyzer
from app.models import (
    BoundingBox,
    CameraPoseStatus,
    LaneObservation,
    PhysicalMeasurementPermission,
    TrackedVehicle,
)
from app.motion import MotionHistoryStore
from app.physical_measurements import PhysicalMeasurementPolicy
from app.positioning import HomographyRoadTransformer
from tests.helpers import trusted_geometry


def _calibration(**updates: object) -> CalibrationConfig:
    values: dict[str, object] = {
        "mode": "homography",
        "image_points": [(0, 0), (100, 0), (100, 100), (0, 100)],
        "world_points": [(0, 0), (10, 0), (10, 20), (0, 20)],
        "validation_image_points": [(20, 20), (80, 20), (80, 80), (20, 80)],
        "validation_world_points": [(2, 4), (8, 4), (8, 16), (2, 16)],
        "reference_width": 100,
        "reference_height": 100,
        "fallback_to_normalized": False,
    }
    values.update(updates)
    return CalibrationConfig(**values)


def _pose() -> CameraPoseStatus:
    return CameraPoseStatus("stable", 0.0, 0.0, 0.95, 20)


def _observation(track_id: int, x: float, y: float) -> LaneObservation:
    return LaneObservation(
        TrackedVehicle(
            track_id,
            BoundingBox(x - 2, y - 4, x + 2, y),
            0.9,
            2,
            "car",
        ),
        "left",
    )


def _allowed(transformer, config) -> PhysicalMeasurementPermission:
    return PhysicalMeasurementPolicy(PhysicalMeasurementsConfig(), config).evaluate(
        transformer.calibration_quality, _pose()
    )


def test_independent_world_validation_error_low_is_accepted() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    quality = transformer.calibration_quality
    assert quality.validation_world_rmse == pytest.approx(0.0, abs=1e-8)
    assert quality.validation_world_mae == pytest.approx(0.0, abs=1e-8)
    assert quality.validation_world_max_error == pytest.approx(0.0, abs=1e-8)
    assert quality.validation_world_p95_error == pytest.approx(0.0, abs=1e-8)
    assert _allowed(transformer, config).allowed


def test_low_pixel_error_but_unacceptable_world_error_is_rejected() -> None:
    config = _calibration(
        validation_world_points=[
            (2.5, 4),
            (8.5, 4),
            (8.5, 16),
            (2.5, 16),
        ],
        maximum_validation_reprojection_error_pixels=10.0,
        maximum_validation_rmse_world_units=0.2,
        maximum_validation_p95_world_units=0.3,
    )
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    quality = transformer.calibration_quality
    assert quality.validation_reprojection_error_pixels is not None
    assert quality.validation_reprojection_error_pixels < 10.0
    assert "VALIDATION_WORLD_ERROR_HIGH" in quality.reason_codes
    assert not _allowed(transformer, config).allowed


def test_clustered_validation_points_emit_coverage_warning() -> None:
    config = _calibration(
        validation_image_points=[(45, 45), (55, 45), (55, 55), (45, 55)],
        validation_world_points=[(4.5, 9), (5.5, 9), (5.5, 11), (4.5, 11)],
    )
    quality = HomographyRoadTransformer(
        config, RoadPositionConfig()
    ).calibration_quality
    assert quality.validation_coverage is not None
    assert quality.validation_coverage < config.minimum_validation_coverage
    assert "VALIDATION_POINTS_CLUSTERED" in quality.reason_codes
    assert "ROAD_REGION_POORLY_VALIDATED" in quality.reason_codes


def test_well_distributed_points_improve_coverage_score() -> None:
    clustered = HomographyRoadTransformer(
        _calibration(
            validation_image_points=[(45, 45), (55, 45), (55, 55), (45, 55)],
            validation_world_points=[(4.5, 9), (5.5, 9), (5.5, 11), (4.5, 11)],
        ),
        RoadPositionConfig(),
    ).calibration_quality
    distributed = HomographyRoadTransformer(
        _calibration(), RoadPositionConfig()
    ).calibration_quality
    assert distributed.validation_coverage is not None
    assert clustered.validation_coverage is not None
    assert distributed.validation_coverage > clustered.validation_coverage


def test_vehicle_inside_support_region_receives_world_position() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    position = transformer.estimate(
        [_observation(1, 50, 50)], 100, 100, _allowed(transformer, config)
    )[1]
    assert position.inside_calibrated_region is True
    assert position.world_position_m == pytest.approx((5.0, 10.0))


def test_vehicle_outside_support_region_has_no_physical_position() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    position = transformer.estimate(
        [_observation(1, 150, 50)], 100, 100, _allowed(transformer, config)
    )[1]
    assert position.inside_calibrated_region is False
    assert position.world_position_m is None
    assert position.coordinate_mode == "normalized_image"
    assert "OUTSIDE_CALIBRATION_REGION" in position.physical_measurement_reason_codes


def test_outside_track_never_creates_meter_gap() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    permission = _allowed(transformer, config)
    observations = [_observation(1, 150, 50), _observation(2, 50, 70)]
    positions = transformer.estimate(observations, 100, 100, permission)
    analyzer = TrafficContextAnalyzer(
        ["left", "right"], TrafficContextConfig(), CongestionConfig()
    )
    context = analyzer.analyze(
        0.0,
        observations,
        positions,
        MotionHistoryStore(TrafficContextConfig()),
        physical_measurements=permission,
        geometry_integrity=trusted_geometry(physical_allowed=True),
    )
    target = context.vehicles[1]
    references = (
        target.neighbors.same_lane_ahead,
        target.neighbors.same_lane_behind,
    )
    assert all(item is None or item.gap_unit == "normalized" for item in references)


def test_calibration_reference_dimensions_must_be_paired() -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        _calibration(reference_height=None)


def test_uniform_runtime_resize_maps_back_to_reference_pixels() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    position = transformer.estimate(
        [_observation(1, 100, 100)], 200, 200, _allowed(transformer, config)
    )[1]
    assert position.world_position_m == pytest.approx((5.0, 10.0))


def test_calibration_aspect_mismatch_disables_world_position() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    position = transformer.estimate(
        [_observation(1, 50, 60)], 100, 120, _allowed(transformer, config)
    )[1]
    assert position.world_position_m is None
    assert "CALIBRATION_FRAME_GEOMETRY_INCOMPATIBLE" in (
        position.physical_measurement_reason_codes
    )

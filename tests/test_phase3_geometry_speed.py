from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.config import (
    CalibrationConfig,
    CongestionConfig,
    PhysicalMeasurementsConfig,
    RightLaneOpportunityConfig,
    RoadPositionConfig,
    SpeedEstimationConfig,
    TrafficContextConfig,
)
from app.context import RightLaneOpportunityTracker, TrafficContextAnalyzer
from app.models import (
    BoundingBox,
    CameraPoseStatus,
    LaneObservation,
    PhysicalMeasurementPermission,
    RoadPosition,
    TrackedVehicle,
)
from app.motion import MotionHistoryStore
from app.physical_measurements import PhysicalMeasurementPolicy
from app.positioning import (
    CalibrationError,
    HomographyRoadTransformer,
    NormalizedImageRoadPositionEstimator,
    build_road_coordinate_transformer,
)
from app.speed import RollingSpeedEstimator


def _calibration(*, validated: bool = False, **updates: object) -> CalibrationConfig:
    values: dict[str, object] = {
        "mode": "homography",
        "image_points": [(0, 0), (100, 0), (100, 100), (0, 100)],
        "world_points": [(0, 0), (10, 0), (10, 20), (0, 20)],
        "fallback_to_normalized": False,
    }
    if validated:
        values.update(
            validation_image_points=[(25, 75), (75, 25)],
            validation_world_points=[(2.5, 15), (7.5, 5)],
        )
    values.update(updates)
    return CalibrationConfig(**values)


def _pose(status: str = "stable") -> CameraPoseStatus:
    return CameraPoseStatus(status, 0.0, 0.0, 1.0, 10)


def _permission(allowed: bool = True) -> PhysicalMeasurementPermission:
    return PhysicalMeasurementPermission(
        allowed=allowed,
        confidence=0.95 if allowed else 0.0,
        status="available_approximate" if allowed else "unavailable",
        reason_codes=() if allowed else ("CALIBRATION_UNVERIFIED",),
    )


def _observation(track_id: int, lane: str, x: float, y: float) -> LaneObservation:
    return LaneObservation(
        TrackedVehicle(track_id, BoundingBox(x - 2, y - 4, x + 2, y), 0.9, 2, "car"),
        lane,
    )


def _position(track_id: int, longitudinal: float, *, calibrated: bool) -> RoadPosition:
    normalized = (0.5, longitudinal / 100 if calibrated else longitudinal)
    return RoadPosition(
        track_id=track_id,
        lateral=0.0 if calibrated else 0.5,
        longitudinal=longitudinal,
        coordinate_system="calibrated_world" if calibrated else "normalized_image",
        calibrated=calibrated,
        image_position=(50.0, 50.0),
        normalized_position=normalized,
        world_position_m=(0.0, longitudinal) if calibrated else None,
        world_position_confidence=0.95 if calibrated else 0.0,
        physical_measurement_status="available_approximate"
        if calibrated
        else "unavailable",
    )


def test_valid_homography_maps_synthetic_points() -> None:
    transformer = HomographyRoadTransformer(_calibration(), RoadPositionConfig())
    assert transformer.image_to_world((50.0, 50.0)) == pytest.approx((5.0, 10.0))
    assert transformer.world_to_image((2.5, 15.0)) == pytest.approx((25.0, 75.0))
    assert transformer.calibration_quality.matrix_valid


def test_four_fit_points_are_unverified_and_physical_output_is_off_by_default() -> None:
    config = _calibration()
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    quality = transformer.calibration_quality
    assert quality.validation_mode == "FIT_POINTS_ONLY"
    assert quality.confidence < 0.5
    permission = PhysicalMeasurementPolicy(
        PhysicalMeasurementsConfig(), config
    ).evaluate(quality, _pose())
    assert not permission.allowed
    position = transformer.estimate(
        [_observation(1, "left", 50, 50)], 100, 100, permission
    )[1]
    assert position.world_position_m is None
    speed = RollingSpeedEstimator(
        SpeedEstimationConfig(minimum_window_seconds=0.1, minimum_samples=2)
    )
    speed.update(0.0, {1: position}, permission)
    result = speed.update(0.2, {1: position}, permission)[1]
    assert result.speed_kph is None


def test_independent_validation_with_low_error_is_accepted() -> None:
    config = _calibration(validated=True)
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    quality = transformer.calibration_quality
    permission = PhysicalMeasurementPolicy(
        PhysicalMeasurementsConfig(), config
    ).evaluate(quality, _pose())
    assert quality.validation_mode == "INDEPENDENT_VALIDATION_POINTS"
    assert quality.validation_reprojection_error_pixels == pytest.approx(0.0, abs=1e-7)
    assert permission.allowed


def test_high_independent_validation_error_disables_physical_output() -> None:
    config = _calibration(
        validation_image_points=[(25, 75)],
        validation_world_points=[(95, 95)],
    )
    transformer = HomographyRoadTransformer(config, RoadPositionConfig())
    permission = PhysicalMeasurementPolicy(
        PhysicalMeasurementsConfig(), config
    ).evaluate(transformer.calibration_quality, _pose())
    assert "VALIDATION_ERROR_HIGH" in transformer.calibration_quality.reason_codes
    assert not permission.allowed


def test_degenerate_control_points_are_rejected() -> None:
    with pytest.raises(ValidationError, match="degenerate"):
        CalibrationConfig(
            mode="homography",
            image_points=[(0, 0), (1, 0), (2, 0), (3, 0)],
            world_points=[(0, 0), (0, 1), (1, 1), (1, 0)],
        )


def test_badly_conditioned_mapping_is_rejected() -> None:
    with pytest.raises(CalibrationError, match="POORLY_CONDITIONED"):
        HomographyRoadTransformer(
            _calibration(validated=True, maximum_condition_number=2.0),
            RoadPositionConfig(),
        )


def test_normalized_fallback_never_emits_meter_position() -> None:
    transformer = build_road_coordinate_transformer(
        CalibrationConfig(mode="normalized"), RoadPositionConfig()
    )
    position = transformer.estimate([_observation(1, "left", 50, 80)], 100, 100)[1]
    assert isinstance(transformer, NormalizedImageRoadPositionEstimator)
    assert position.world_position_m is None
    assert not position.calibrated


def test_position_estimator_uses_world_coordinates_only_with_permission() -> None:
    transformer = HomographyRoadTransformer(
        _calibration(validated=True), RoadPositionConfig()
    )
    observation = [_observation(7, "left", 50, 50)]
    denied = transformer.estimate(observation, 100, 100, _permission(False))[7]
    allowed = transformer.estimate(observation, 100, 100, _permission(True))[7]
    assert denied.coordinate_mode == "normalized_image"
    assert denied.world_position_m is None
    assert allowed.world_position_m == pytest.approx((5.0, 10.0))
    assert allowed.coordinate_mode == "calibrated_world"


@pytest.mark.parametrize(
    "calibrated, expected_unit", [(True, "meters"), (False, "normalized")]
)
def test_gap_units_follow_explicit_coordinate_capability(
    calibrated: bool, expected_unit: str
) -> None:
    analyzer = TrafficContextAnalyzer(
        ["left", "right"], TrafficContextConfig(), CongestionConfig()
    )
    first = 20.0 if calibrated else 0.40
    second = 50.0 if calibrated else 0.52
    observations = [
        _observation(1, "left", 30, 60),
        _observation(2, "right", 70, 40),
    ]
    positions = {
        1: _position(1, first, calibrated=calibrated),
        2: _position(2, second, calibrated=calibrated),
    }
    context = analyzer.analyze(
        0.0,
        observations,
        positions,
        MotionHistoryStore(TrafficContextConfig()),
        physical_measurements=_permission(calibrated),
    )
    context = RightLaneOpportunityTracker(RightLaneOpportunityConfig()).update(
        context, 0.0
    )
    gap = context.vehicles[1].right_lane_front_gap
    assert gap is not None and gap.unit == expected_unit


def test_world_gap_is_not_exposed_when_central_permission_is_denied() -> None:
    analyzer = TrafficContextAnalyzer(
        ["left", "right"], TrafficContextConfig(), CongestionConfig()
    )
    observations = [
        _observation(1, "left", 30, 60),
        _observation(2, "right", 70, 40),
    ]
    positions = {
        1: _position(1, 20.0, calibrated=True),
        2: _position(2, 50.0, calibrated=True),
    }
    context = analyzer.analyze(
        0.0,
        observations,
        positions,
        MotionHistoryStore(TrafficContextConfig()),
        physical_measurements=_permission(False),
    )
    context = RightLaneOpportunityTracker(RightLaneOpportunityConfig()).update(
        context, 0.0
    )
    gap = context.vehicles[1].right_lane_front_gap
    assert gap is not None and gap.unit == "normalized"


def test_validated_stable_world_motion_produces_approximate_speed() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(
            minimum_window_seconds=0.8, maximum_window_seconds=2.0, minimum_samples=5
        )
    )
    result = None
    for index in range(5):
        timestamp = index * 0.2
        result = estimator.update(
            timestamp, {1: _position(1, 10 * timestamp, calibrated=True)}, _permission()
        )[1]
    assert result is not None
    assert result.speed_mps == pytest.approx(10.0)
    assert result.speed_kph == pytest.approx(36.0)


def test_camera_pose_loss_makes_existing_speed_unavailable() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(minimum_window_seconds=0.1, minimum_samples=2)
    )
    estimator.update(0.0, {1: _position(1, 0.0, calibrated=True)}, _permission())
    assert (
        estimator.update(0.2, {1: _position(1, 1.0, calibrated=True)}, _permission())[
            1
        ].speed_kph
        is not None
    )
    result = estimator.update(
        0.4, {1: _position(1, 2.0, calibrated=True)}, _permission(False)
    )[1]
    assert result.speed_kph is None
    assert result.physical_measurement_status == "unavailable"


def test_tracker_jump_does_not_create_absurd_speed() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(
            minimum_window_seconds=0.1, minimum_samples=2, max_position_jump_meters=20.0
        )
    )
    estimator.update(0.0, {1: _position(1, 0.0, calibrated=True)}, _permission())
    result = estimator.update(
        0.2, {1: _position(1, 100.0, calibrated=True)}, _permission()
    )[1]
    assert result.speed_kph is None
    assert result.speed_mode == "rejected_position_jump"


def test_short_tracker_dropout_is_tolerated() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(
            minimum_window_seconds=0.8,
            maximum_window_seconds=2.0,
            minimum_samples=4,
            tracker_gap_grace_seconds=0.5,
        )
    )
    result = None
    for timestamp in (0.0, 0.2, 0.4, 0.8):
        result = estimator.update(
            timestamp, {1: _position(1, timestamp * 5, calibrated=True)}, _permission()
        )[1]
    assert result is not None and result.speed_mps == pytest.approx(5.0)


def test_normalized_motion_never_produces_physical_speed() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(minimum_window_seconds=0.1, minimum_samples=2)
    )
    estimator.update(0.0, {1: _position(1, 0.1, calibrated=False)}, _permission(False))
    result = estimator.update(
        0.2, {1: _position(1, 0.2, calibrated=False)}, _permission(False)
    )[1]
    assert result.speed_mps is None and result.speed_kph is None
    assert math.isfinite(result.normalized_motion_rate or 0.0)

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.config import (
    CalibrationConfig,
    CongestionConfig,
    RightLaneOpportunityConfig,
    RoadPositionConfig,
    SpeedEstimationConfig,
    TrafficContextConfig,
)
from app.context import RightLaneOpportunityTracker, TrafficContextAnalyzer
from app.models import BoundingBox, LaneObservation, RoadPosition, TrackedVehicle
from app.motion import MotionHistoryStore
from app.positioning import (
    HomographyRoadTransformer,
    NormalizedImageRoadPositionEstimator,
    build_road_coordinate_transformer,
)
from app.speed import RollingSpeedEstimator


def _calibration(**updates: object) -> CalibrationConfig:
    values = {
        "mode": "homography",
        "image_points": [(0, 0), (100, 0), (100, 100), (0, 100)],
        "world_points": [(0, 0), (10, 0), (10, 20), (0, 20)],
        "fallback_to_normalized": False,
    }
    values.update(updates)
    return CalibrationConfig(**values)


def _observation(track_id: int, lane: str, x: float, y: float) -> LaneObservation:
    return LaneObservation(
        vehicle=TrackedVehicle(
            track_id=track_id,
            bbox=BoundingBox(x - 2, y - 4, x + 2, y),
            confidence=0.9,
            class_id=2,
            class_name="car",
        ),
        lane_id=lane,
    )


def _position(
    track_id: int,
    longitudinal: float,
    *,
    calibrated: bool,
    lateral: float = 0.0,
) -> RoadPosition:
    normalized = (0.5, longitudinal / 100 if calibrated else longitudinal)
    world = (lateral, longitudinal) if calibrated else None
    return RoadPosition(
        track_id=track_id,
        lateral=lateral,
        longitudinal=longitudinal,
        coordinate_system="calibrated_world" if calibrated else "normalized_image",
        calibrated=calibrated,
        image_position=(50.0, 50.0),
        normalized_position=normalized,
        world_position=world,
        calibration_confidence=1.0 if calibrated else 0.0,
    )


def test_homography_maps_known_points_correctly() -> None:
    transformer = HomographyRoadTransformer(_calibration(), RoadPositionConfig())
    assert transformer.image_to_world((50.0, 50.0)) == pytest.approx((5.0, 10.0))
    assert transformer.world_to_image((2.5, 15.0)) == pytest.approx((25.0, 75.0))
    assert transformer.calibration_status.reprojection_error_pixels == pytest.approx(
        0.0, abs=1e-8
    )


def test_invalid_calibration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="degenerate"):
        CalibrationConfig(
            mode="homography",
            image_points=[(0, 0), (1, 0), (2, 0), (3, 0)],
            world_points=[(0, 0), (0, 1), (1, 1), (1, 0)],
        )


def test_normalized_mode_is_safe_fallback() -> None:
    transformer = build_road_coordinate_transformer(
        CalibrationConfig(mode="normalized"), RoadPositionConfig()
    )
    position = transformer.estimate([_observation(1, "left", 50, 80)], 100, 100)[1]
    assert isinstance(transformer, NormalizedImageRoadPositionEstimator)
    assert position.normalized_position is not None
    assert position.world_position is None
    assert not position.calibrated


def test_failed_homography_uses_explicit_normalized_fallback() -> None:
    transformer = build_road_coordinate_transformer(
        CalibrationConfig(
            mode="homography",
            image_points=[
                (0, 0),
                (100, 0),
                (100, 100),
                (0, 100),
                (50, 50),
            ],
            world_points=[
                (0, 0),
                (10, 0),
                (10, 20),
                (0, 20),
                (100, 100),
            ],
            maximum_reprojection_error_pixels=0.1,
            fallback_to_normalized=True,
        ),
        RoadPositionConfig(),
    )
    assert isinstance(transformer, NormalizedImageRoadPositionEstimator)
    assert transformer.calibration_status.mode == "homography_fallback"
    assert not transformer.calibration_status.valid


def test_position_estimator_preserves_all_coordinate_representations() -> None:
    transformer = HomographyRoadTransformer(_calibration(), RoadPositionConfig())
    position = transformer.estimate([_observation(7, "left", 50, 50)], 100, 100)[7]
    assert position.image_position == (50.0, 50)
    assert position.normalized_position == pytest.approx((0.5, 0.5))
    assert position.world_position == pytest.approx((5.0, 10.0))
    assert position.coordinate_mode == "calibrated_world"


def test_world_gap_is_meters_only_when_calibrated() -> None:
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
        0.0, observations, positions, MotionHistoryStore(TrafficContextConfig())
    )
    context = RightLaneOpportunityTracker(
        RightLaneOpportunityConfig(mode="auto", minimum_front_gap_m=20.0)
    ).update(context, 0.0)
    gap = context.vehicles[1].right_lane_front_gap
    assert gap is not None
    assert gap.value == pytest.approx(30.0)
    assert gap.unit == "meters"
    assert context.vehicles[1].right_lane_opportunity_mode == "calibrated"


def test_uncalibrated_gap_is_never_labeled_meters() -> None:
    analyzer = TrafficContextAnalyzer(
        ["left", "right"], TrafficContextConfig(), CongestionConfig()
    )
    observations = [
        _observation(1, "left", 30, 60),
        _observation(2, "right", 70, 40),
    ]
    positions = {
        1: _position(1, 0.4, calibrated=False),
        2: _position(2, 0.52, calibrated=False),
    }
    context = analyzer.analyze(
        0.0, observations, positions, MotionHistoryStore(TrafficContextConfig())
    )
    context = RightLaneOpportunityTracker(RightLaneOpportunityConfig()).update(
        context, 0.0
    )
    gap = context.vehicles[1].right_lane_front_gap
    assert gap is not None
    assert gap.unit == "normalized"
    assert context.vehicles[1].right_lane_opportunity_mode == "normalized"


def test_constant_world_motion_produces_approximate_speed() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(
            minimum_window_seconds=0.8,
            maximum_window_seconds=2.0,
            minimum_samples=5,
        )
    )
    result = None
    for index in range(5):
        timestamp = index * 0.2
        result = estimator.update(
            timestamp,
            {1: _position(1, 10.0 * timestamp, calibrated=True)},
        )[1]
    assert result is not None
    assert result.speed_mps == pytest.approx(10.0)
    assert result.speed_kph == pytest.approx(36.0)
    assert result.speed_mode == "approximate_calibrated"


def test_tracker_jump_does_not_create_absurd_speed() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(
            minimum_window_seconds=0.1,
            minimum_samples=2,
            max_position_jump_meters=20.0,
        )
    )
    estimator.update(0.0, {1: _position(1, 0.0, calibrated=True)})
    result = estimator.update(0.2, {1: _position(1, 100.0, calibrated=True)})[1]
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
            timestamp, {1: _position(1, timestamp * 5.0, calibrated=True)}
        )[1]
    assert result is not None
    assert result.speed_mps == pytest.approx(5.0)


def test_uncalibrated_motion_never_produces_physical_speed() -> None:
    estimator = RollingSpeedEstimator(
        SpeedEstimationConfig(minimum_window_seconds=0.1, minimum_samples=2)
    )
    estimator.update(0.0, {1: _position(1, 0.1, calibrated=False)})
    result = estimator.update(0.2, {1: _position(1, 0.2, calibrated=False)})[1]
    assert result.speed_mps is None
    assert result.speed_kph is None
    assert result.speed_mode == "unavailable_uncalibrated"
    assert math.isfinite(result.normalized_motion_rate or 0.0)

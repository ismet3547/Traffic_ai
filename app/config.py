"""YAML-backed, validated application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoConfig(StrictModel):
    input_path: str = "input/highway.mp4"


class DetectorConfig(StrictModel):
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    iou_threshold: float = Field(default=0.70, ge=0, le=1)
    image_size: int = Field(default=960, ge=320)
    device: str | None = None
    vehicle_class_ids: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])


class TrackerConfig(StrictModel):
    track_activation_threshold: float = Field(default=0.25, ge=0, le=1)
    lost_track_buffer: int = Field(default=30, ge=1)
    minimum_matching_threshold: float = Field(default=0.80, ge=0, le=1)
    minimum_consecutive_frames: int = Field(default=2, ge=1)


class LaneConfig(StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    leftmost: bool = False
    polygon: list[tuple[float, float]] = Field(min_length=3)


class LanesConfig(StrictModel):
    coordinate_space: Literal["normalized", "pixels"] = "normalized"
    assignment_anchor: Literal["bottom_center"] = "bottom_center"
    lanes: list[LaneConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lanes(self) -> LanesConfig:
        ids = [lane.id for lane in self.lanes]
        if len(ids) != len(set(ids)):
            raise ValueError("lane IDs must be unique")
        if sum(lane.leftmost for lane in self.lanes) != 1:
            raise ValueError("exactly one lane must have leftmost: true")
        if not self.lanes[0].leftmost:
            raise ValueError("lanes must be ordered left-to-right, with leftmost first")
        if self.coordinate_space == "normalized":
            for lane in self.lanes:
                if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in lane.polygon):
                    raise ValueError(
                        "normalized lane coordinates must be between 0 and 1"
                    )
        return self

    @property
    def leftmost_lane_id(self) -> str:
        return next(lane.id for lane in self.lanes if lane.leftmost)

    @property
    def lane_ids_left_to_right(self) -> list[str]:
        """Lane IDs in configured left-to-right order."""

        return [lane.id for lane in self.lanes]


class TrafficContextConfig(StrictModel):
    history_seconds: float = Field(default=12.0, gt=0)
    minimum_history_seconds: float = Field(default=2.0, ge=0)
    maximum_samples_per_track: int = Field(default=900, ge=2)
    nearby_longitudinal_window_normalized: float = Field(default=0.25, gt=0, le=1)
    nearby_longitudinal_window_meters: float = Field(default=75.0, gt=0)

    @model_validator(mode="after")
    def validate_history_window(self) -> TrafficContextConfig:
        if self.minimum_history_seconds > self.history_seconds:
            raise ValueError("minimum history cannot exceed the history window")
        return self


class LaneChangeConfig(StrictModel):
    confirmation_seconds: float = Field(default=0.4, ge=0)
    minimum_frames: int = Field(default=3, ge=1)
    state_ttl_seconds: float = Field(default=2.0, gt=0)


class RoadPositionConfig(StrictModel):
    mode: Literal["normalized_image"] = "normalized_image"
    travel_direction: Literal["toward_top", "toward_bottom"] = "toward_top"


class CalibrationConfig(StrictModel):
    """Optional mapping from image pixels to a measured road plane."""

    mode: Literal["normalized", "homography"] = "normalized"
    world_units: Literal["meters"] = "meters"
    image_points: list[tuple[float, float]] = Field(default_factory=list)
    world_points: list[tuple[float, float]] = Field(default_factory=list)
    fallback_to_normalized: bool = True
    maximum_reprojection_error_pixels: float = Field(default=5.0, gt=0)
    minimum_confidence_for_physical_measurements: float = Field(
        default=0.55, ge=0, le=1
    )
    suppress_candidates_when_unreliable: bool = False
    world_longitudinal_axis: Literal["x", "y"] = "y"
    world_longitudinal_direction: Literal["positive", "negative"] = "positive"

    @model_validator(mode="after")
    def validate_correspondences(self) -> CalibrationConfig:
        if self.mode == "normalized":
            return self
        if len(self.image_points) != len(self.world_points):
            raise ValueError("image_points and world_points must have equal length")
        if len(self.image_points) < 4:
            raise ValueError(
                "homography calibration requires at least four point pairs"
            )
        for name, points in (
            ("image_points", self.image_points),
            ("world_points", self.world_points),
        ):
            if len(set(points)) != len(points):
                raise ValueError(f"{name} must not contain duplicate points")
            if not _contains_non_collinear_triplet(points):
                raise ValueError(f"{name} are degenerate (all points are collinear)")
        return self


class SpeedEstimationConfig(StrictModel):
    enabled: bool = True
    minimum_window_seconds: float = Field(default=0.8, gt=0)
    maximum_window_seconds: float = Field(default=2.5, gt=0)
    minimum_samples: int = Field(default=5, ge=2)
    smoothing: Literal["median", "linear_regression"] = "median"
    max_reasonable_speed_kph: float = Field(default=220.0, gt=0)
    max_position_jump_meters: float = Field(default=20.0, gt=0)
    tracker_gap_grace_seconds: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def validate_windows(self) -> SpeedEstimationConfig:
        if self.minimum_window_seconds > self.maximum_window_seconds:
            raise ValueError("minimum speed window cannot exceed maximum speed window")
        return self


class CameraMotionConfig(StrictModel):
    mode: Literal["none", "feature_based"] = "none"
    maximum_features: int = Field(default=250, ge=20)
    minimum_tracked_features: int = Field(default=12, ge=4)
    excessive_translation_pixels: float = Field(default=8.0, gt=0)
    minimum_confidence: float = Field(default=0.35, ge=0, le=1)
    mask_vehicle_boxes: bool = True


class CandidateLifecycleConfig(StrictModel):
    invalidation_grace_seconds: float = Field(default=2.0, ge=0)
    suspension_grace_seconds: float = Field(default=3.0, ge=0)
    finalize_after_seconds: float = Field(default=5.0, ge=0)
    restart_cooldown_seconds: float = Field(default=1.5, ge=0)
    maximum_decision_history_entries: int = Field(default=32, ge=4, le=256)


class OvertakingConfig(StrictModel):
    enabled: bool = True
    observation_window_seconds: float = Field(default=10.0, gt=0)
    completion_timeout_seconds: float = Field(default=15.0, gt=0)
    minimum_confidence: float = Field(default=0.65, ge=0, le=1)
    entry_target_max_gap_normalized: float = Field(default=0.20, gt=0, le=1)
    entry_target_max_gap_meters: float = Field(default=60.0, gt=0)
    pass_order_margin_normalized: float = Field(default=0.01, ge=0, le=0.25)
    pass_order_margin_meters: float = Field(default=1.5, ge=0)
    minimum_relative_speed_mps: float = Field(default=0.8, ge=0)
    post_overtake_grace_seconds: float = Field(default=2.0, ge=0)
    related_track_lost_grace_seconds: float = Field(default=1.0, ge=0)


class CongestionConfig(StrictModel):
    enabled: bool = True
    minimum_observed_vehicles: int = Field(default=1, ge=1)
    dense_vehicle_count_per_lane: int = Field(default=3, ge=1)
    moderate_density_ratio: float = Field(default=0.45, ge=0, le=1)
    dense_density_ratio: float = Field(default=0.80, ge=0, le=1)
    stop_and_go_max_motion_per_second_normalized: float = Field(
        default=0.015, ge=0, le=1
    )
    dense_max_motion_per_second_normalized: float = Field(default=0.04, ge=0, le=1)

    @model_validator(mode="after")
    def validate_density_thresholds(self) -> CongestionConfig:
        if self.moderate_density_ratio > self.dense_density_ratio:
            raise ValueError("moderate density ratio cannot exceed dense density ratio")
        if (
            self.stop_and_go_max_motion_per_second_normalized
            > self.dense_max_motion_per_second_normalized
        ):
            raise ValueError(
                "stop-and-go motion threshold cannot exceed dense threshold"
            )
        return self


class RightLaneOpportunityConfig(StrictModel):
    mode: Literal["auto", "normalized", "calibrated"] = "auto"
    minimum_available_seconds: float = Field(default=3.0, ge=0)
    front_gap_normalized: float = Field(default=0.08, gt=0, le=1)
    rear_gap_normalized: float = Field(default=0.06, gt=0, le=1)
    minimum_front_gap_m: float = Field(default=20.0, gt=0)
    minimum_rear_gap_m: float = Field(default=15.0, gt=0)
    minimum_confidence: float = Field(default=0.60, ge=0, le=1)
    state_ttl_seconds: float = Field(default=2.0, gt=0)


class LeftLaneRuleConfig(StrictModel):
    enabled: bool = True
    left_lane_id: str = "left"
    occupancy_threshold_seconds: float = Field(default=8.0, gt=0)
    track_lost_grace_seconds: float = Field(default=1.0, ge=0)
    minimum_mean_confidence: float = Field(default=0.25, ge=0, le=1)
    minimum_evidence_confidence: float = Field(default=0.65, ge=0, le=1)
    overtaking_clearance_mode: Literal["none", "contextual"] = "contextual"
    policy_version: str = Field(default="3.0", min_length=1)


class RulesConfig(StrictModel):
    left_lane: LeftLaneRuleConfig = Field(default_factory=LeftLaneRuleConfig)


class OutputConfig(StrictModel):
    directory: str = "output"
    debug_video_enabled: bool = True
    debug_video_name: str = "annotated.mp4"
    codec: str = "mp4v"
    representative_image_quality: int = Field(default=92, ge=1, le=100)
    clip_pre_event_seconds: float = Field(default=2.0, ge=0)
    clip_max_duration_seconds: float = Field(default=12.0, gt=0)
    show_advanced_debug: bool = True
    show_coordinates: bool = True
    show_speed: bool = True
    show_gaps: bool = True
    show_lifecycle: bool = True

    @model_validator(mode="after")
    def validate_codec(self) -> OutputConfig:
        if len(self.codec) != 4:
            raise ValueError("output codec must be a four-character FourCC code")
        if self.clip_pre_event_seconds >= self.clip_max_duration_seconds:
            raise ValueError(
                "clip_pre_event_seconds must be shorter than the maximum clip"
            )
        return self


class AppConfig(StrictModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    lanes: LanesConfig
    road_position: RoadPositionConfig = Field(default_factory=RoadPositionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    speed_estimation: SpeedEstimationConfig = Field(
        default_factory=SpeedEstimationConfig
    )
    camera_motion: CameraMotionConfig = Field(default_factory=CameraMotionConfig)
    candidate_lifecycle: CandidateLifecycleConfig = Field(
        default_factory=CandidateLifecycleConfig
    )
    traffic_context: TrafficContextConfig = Field(default_factory=TrafficContextConfig)
    lane_change: LaneChangeConfig = Field(default_factory=LaneChangeConfig)
    overtaking: OvertakingConfig = Field(default_factory=OvertakingConfig)
    congestion: CongestionConfig = Field(default_factory=CongestionConfig)
    right_lane_opportunity: RightLaneOpportunityConfig = Field(
        default_factory=RightLaneOpportunityConfig
    )
    rules: RulesConfig = Field(default_factory=RulesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_cross_references(self) -> AppConfig:
        lane_ids = {lane.id for lane in self.lanes.lanes}
        rule_lane = self.rules.left_lane.left_lane_id
        if rule_lane not in lane_ids:
            raise ValueError(f"left-lane rule references unknown lane ID: {rule_lane}")
        if rule_lane != self.lanes.leftmost_lane_id:
            raise ValueError("left-lane rule must reference the lane marked leftmost")
        return self


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return AppConfig.model_validate(raw)


def _contains_non_collinear_triplet(points: list[tuple[float, float]]) -> bool:
    """Return whether at least one triple spans a non-zero area."""

    for first_index in range(len(points) - 2):
        x1, y1 = points[first_index]
        for second_index in range(first_index + 1, len(points) - 1):
            x2, y2 = points[second_index]
            for third_index in range(second_index + 1, len(points)):
                x3, y3 = points[third_index]
                twice_area = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if abs(twice_area) > 1e-9:
                    return True
    return False

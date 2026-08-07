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
    def validate_lanes(self) -> "LanesConfig":
        ids = [lane.id for lane in self.lanes]
        if len(ids) != len(set(ids)):
            raise ValueError("lane IDs must be unique")
        if sum(lane.leftmost for lane in self.lanes) != 1:
            raise ValueError("exactly one lane must have leftmost: true")
        if self.coordinate_space == "normalized":
            for lane in self.lanes:
                if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in lane.polygon):
                    raise ValueError("normalized lane coordinates must be between 0 and 1")
        return self

    @property
    def leftmost_lane_id(self) -> str:
        return next(lane.id for lane in self.lanes if lane.leftmost)


class LeftLaneRuleConfig(StrictModel):
    enabled: bool = True
    left_lane_id: str = "left"
    occupancy_threshold_seconds: float = Field(default=8.0, gt=0)
    track_lost_grace_seconds: float = Field(default=1.0, ge=0)
    minimum_mean_confidence: float = Field(default=0.25, ge=0, le=1)
    overtaking_clearance_mode: Literal["none"] = "none"


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

    @model_validator(mode="after")
    def validate_codec(self) -> "OutputConfig":
        if len(self.codec) != 4:
            raise ValueError("output codec must be a four-character FourCC code")
        if self.clip_pre_event_seconds >= self.clip_max_duration_seconds:
            raise ValueError("clip_pre_event_seconds must be shorter than the maximum clip")
        return self


class AppConfig(StrictModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    lanes: LanesConfig
    rules: RulesConfig = Field(default_factory=RulesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "AppConfig":
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

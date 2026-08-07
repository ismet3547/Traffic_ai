"""Framework-neutral records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned pixel bounding box in ``xyxy`` form."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class TrackedVehicle:
    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class LaneObservation:
    vehicle: TrackedVehicle
    lane_id: str | None


@dataclass(frozen=True, slots=True)
class VehicleRuleStatus:
    track_id: int
    lane_id: str | None
    left_lane_duration_seconds: float
    is_review_candidate: bool


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    transition: Literal["started", "ended"]
    track_id: int
    lane_id: str
    start_timestamp_seconds: float
    timestamp_seconds: float
    duration_seconds: float
    confidence_score: float
    end_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    statuses: dict[int, VehicleRuleStatus]
    transitions: list[CandidateTransition]


@dataclass(frozen=True, slots=True)
class FramePacket:
    index: int
    timestamp_seconds: float
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class EventMetadata(BaseModel):
    """Serializable audit record for one human-review candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    event_id: str
    event_type: Literal["left_lane_review_candidate"] = (
        "left_lane_review_candidate"
    )
    review_status: Literal["pending_human_review"] = "pending_human_review"
    human_review_required: bool = True
    enforcement_action: Literal["none"] = "none"
    track_id: int
    event_start_timestamp_seconds: float = Field(ge=0)
    candidate_created_timestamp_seconds: float = Field(ge=0)
    event_end_timestamp_seconds: float | None = Field(default=None, ge=0)
    duration_seconds: float = Field(ge=0)
    lane_id: str
    confidence_score: float = Field(ge=0, le=1)
    confidence_definition: str = "mean detector confidence while observed in left lane"
    source_video_name: str
    representative_frame: str
    event_video_clip: str
    end_reason: str | None = None
    overtaking_assessment: Literal["not_implemented"] = "not_implemented"

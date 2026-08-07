"""Framework-neutral records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


class CongestionLevel(str, Enum):
    FREE_FLOW = "free_flow"
    MODERATE = "moderate"
    DENSE = "dense"
    STOP_AND_GO = "stop_and_go"
    UNKNOWN = "unknown"


class OvertakeState(str, Enum):
    NONE = "NONE"
    ENTERED_LEFT = "ENTERED_LEFT"
    PASSING = "PASSING"
    PASSED_TARGET = "PASSED_TARGET"
    RETURNING_RIGHT = "RETURNING_RIGHT"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class OvertakingStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LIKELY_OVERTAKING = "likely_overtaking"
    OVERTAKING_CONFIRMED = "overtaking_confirmed"
    NOT_OVERTAKING = "not_overtaking"


class BehaviorClassification(str, Enum):
    OVERTAKING = "overtaking"
    LIKELY_OVERTAKING = "likely_overtaking"
    CONGESTION = "congestion"
    TEMPORARY_LEFT_LANE_USE = "temporary_left_lane_use"
    POSSIBLE_LEFT_LANE_OCCUPATION = "possible_left_lane_occupation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReviewReasonCode(str, Enum):
    LEFT_LANE_DURATION_EXCEEDED = "LEFT_LANE_DURATION_EXCEEDED"
    NO_ACTIVE_OVERTAKE = "NO_ACTIVE_OVERTAKE"
    RIGHT_LANE_AVAILABLE = "RIGHT_LANE_AVAILABLE"
    FREE_FLOW_TRAFFIC = "FREE_FLOW_TRAFFIC"


class SuppressionReason(str, Enum):
    DURATION_BELOW_THRESHOLD = "DURATION_BELOW_THRESHOLD"
    OVERTAKING_CONFIRMED = "OVERTAKING_CONFIRMED"
    ACTIVE_OVERTAKE = "ACTIVE_OVERTAKE"
    CONGESTION = "CONGESTION"
    RIGHT_LANE_UNAVAILABLE = "RIGHT_LANE_UNAVAILABLE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    LOW_EVIDENCE_CONFIDENCE = "LOW_EVIDENCE_CONFIDENCE"


@dataclass(frozen=True, slots=True)
class LaneTransition:
    track_id: int
    from_lane: str
    to_lane: str
    timestamp_seconds: float


@dataclass(frozen=True, slots=True)
class RoadPosition:
    track_id: int
    lateral: float
    longitudinal: float
    coordinate_system: Literal["normalized_image", "calibrated_world"]
    calibrated: bool


@dataclass(frozen=True, slots=True)
class NeighborReference:
    track_id: int
    longitudinal_gap: float


@dataclass(frozen=True, slots=True)
class NeighborVehicles:
    same_lane_ahead: NeighborReference | None = None
    same_lane_behind: NeighborReference | None = None
    adjacent_right_ahead: NeighborReference | None = None
    adjacent_right_behind: NeighborReference | None = None

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(
            reference.track_id
            for reference in (
                self.same_lane_ahead,
                self.same_lane_behind,
                self.adjacent_right_ahead,
                self.adjacent_right_behind,
            )
            if reference is not None
        )


@dataclass(frozen=True, slots=True)
class GlobalTrafficContext:
    congestion_level: CongestionLevel
    traffic_density: float
    active_vehicle_count: int
    lane_vehicle_counts: dict[str, int]
    average_normalized_motion_per_second: float | None
    confidence: float
    coordinate_system: str = "normalized_image"


@dataclass(frozen=True, slots=True)
class VehicleTrafficContext:
    track_id: int
    neighbors: NeighborVehicles
    nearby_vehicle_count: int
    adjacent_right_lane_id: str | None
    right_lane_available: bool | None
    right_lane_available_seconds: float
    right_lane_confidence: float


@dataclass(frozen=True, slots=True)
class TrafficFrameContext:
    global_context: GlobalTrafficContext
    vehicles: dict[int, VehicleTrafficContext]
    positions: dict[int, RoadPosition]


@dataclass(frozen=True, slots=True)
class OvertakingAssessment:
    track_id: int
    status: OvertakingStatus
    state: OvertakeState
    confidence: float
    evidence: tuple[str, ...] = ()
    related_track_ids: tuple[int, ...] = ()
    started_at: float | None = None
    completed_at: float | None = None


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    eligible: bool
    classification: BehaviorClassification
    evidence_confidence: float
    reason_codes: tuple[str, ...] = ()
    suppression_reason: str | None = None


@dataclass(frozen=True, slots=True)
class VehicleRuleStatus:
    track_id: int
    lane_id: str | None
    left_lane_duration_seconds: float
    is_review_candidate: bool
    behavior_classification: str = BehaviorClassification.INSUFFICIENT_EVIDENCE.value
    suppression_reason: str | None = None
    overtake_state: str = OvertakeState.NONE.value
    overtaking_status: str = OvertakingStatus.NOT_ASSESSED.value
    right_lane_available_seconds: float = 0.0
    evidence_confidence: float = 0.0
    related_track_ids: tuple[int, ...] = ()


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
    review_reason_codes: tuple[str, ...] = ()
    policy_version: str = "1.0"
    traffic_context: GlobalTrafficContext | None = None
    vehicle_traffic_context: VehicleTrafficContext | None = None
    overtaking_assessment: OvertakingAssessment | None = None
    behavior_classification: str | None = None
    evidence_confidence_score: float | None = None


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    statuses: dict[int, VehicleRuleStatus]
    transitions: list[CandidateTransition]
    traffic_context: GlobalTrafficContext | None = None


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


class TrafficContextMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    congestion_level: str
    traffic_density: float = Field(ge=0, le=1)
    nearby_vehicle_count: int = Field(ge=0)
    active_vehicle_count: int = Field(ge=0)
    lane_vehicle_counts: dict[str, int]
    average_normalized_motion_per_second: float | None = Field(default=None, ge=0)
    right_lane_available: bool | None = None
    right_lane_available_seconds: float = Field(default=0, ge=0)
    right_lane_confidence: float = Field(default=0, ge=0, le=1)
    coordinate_system: str = "normalized_image"
    calibrated: bool = False
    confidence: float = Field(default=0, ge=0, le=1)


class OvertakingAssessmentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    state: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    related_track_ids: list[int] = Field(default_factory=list)
    started_at: float | None = Field(default=None, ge=0)
    completed_at: float | None = Field(default=None, ge=0)


class EventMetadata(BaseModel):
    """Serializable audit record for one human-review candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0"
    event_id: str
    event_type: Literal["left_lane_review_candidate"] = "left_lane_review_candidate"
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
    behavior_classification: str | None = None
    evidence_confidence_score: float | None = Field(default=None, ge=0, le=1)
    review_reason_codes: list[str] = Field(default_factory=list)
    policy_version: str = "1.0"
    traffic_context: TrafficContextMetadata | None = None
    overtaking_assessment: OvertakingAssessmentMetadata | Literal["not_implemented"] = (
        "not_implemented"
    )

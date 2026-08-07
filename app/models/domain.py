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
    CALIBRATION_UNRELIABLE = "CALIBRATION_UNRELIABLE"
    LOW_POSITION_CONFIDENCE = "LOW_POSITION_CONFIDENCE"
    UNSTABLE_TRACK = "UNSTABLE_TRACK"
    CAMERA_MOTION_HIGH = "CAMERA_MOTION_HIGH"


class CandidateLifecycleState(str, Enum):
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    CANDIDATE_ACTIVE = "candidate_active"
    SUSPENDED = "suspended"
    PENDING_CLOSE = "pending_close"
    CANCELLED = "cancelled"
    FINALIZED = "finalized"


class EvidenceQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


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
    image_position: tuple[float, float] | None = None
    normalized_position: tuple[float, float] | None = None
    world_position_m: tuple[float, float] | None = None
    world_position_confidence: float = 0.0
    physical_measurement_status: str = "unavailable"
    physical_measurement_reason_codes: tuple[str, ...] = ()

    @property
    def coordinate_mode(self) -> str:
        return self.coordinate_system


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    mode: str
    matrix_valid: bool
    numerically_stable: bool
    validation_mode: str
    fit_reprojection_error_pixels: float | None
    validation_reprojection_error_pixels: float | None
    condition_metric: float | None
    confidence: float
    confidence_basis: str
    reason_codes: tuple[str, ...] = ()
    world_units: str | None = None

    @property
    def valid(self) -> bool:
        return self.matrix_valid and self.numerically_stable

    @property
    def reprojection_error_pixels(self) -> float | None:
        return (
            self.validation_reprojection_error_pixels
            if self.validation_reprojection_error_pixels is not None
            else self.fit_reprojection_error_pixels
        )

    @property
    def reason(self) -> str | None:
        return ", ".join(self.reason_codes) if self.reason_codes else None


# Import compatibility only; the Phase 3.1 model semantics are CalibrationQuality.
CalibrationStatus = CalibrationQuality


@dataclass(frozen=True, slots=True)
class CameraMotionEstimate:
    dx: float
    dy: float
    rotation_degrees: float
    confidence: float
    valid: bool
    level: str = "unknown"
    method: str = "none"
    stabilization_applied: bool = False


@dataclass(frozen=True, slots=True)
class CameraPoseStatus:
    status: str
    translation_px: float | None
    rotation_deg: float | None
    confidence: float
    sample_count: int
    stabilization_applied: bool = False
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhysicalMeasurementPermission:
    allowed: bool
    confidence: float
    status: str
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpeedEstimate:
    track_id: int
    speed_mps: float | None
    speed_kph: float | None
    speed_confidence: float
    speed_mode: str
    normalized_motion_rate: float | None = None
    sample_count: int = 0
    physical_measurement_status: str = "unavailable"
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NeighborReference:
    track_id: int
    longitudinal_gap: float
    gap_unit: Literal["meters", "normalized"] = "normalized"
    confidence: float = 0.0
    coordinate_mode: str = "normalized_image"


@dataclass(frozen=True, slots=True)
class GapEstimate:
    value: float
    unit: Literal["meters", "normalized"]
    confidence: float
    coordinate_mode: str


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
    calibration_quality: CalibrationQuality | None = None
    camera_motion: CameraMotionEstimate | None = None
    camera_pose: CameraPoseStatus | None = None
    physical_measurements: PhysicalMeasurementPermission | None = None

    @property
    def calibration_status(self) -> CalibrationQuality | None:
        """Phase 3 compatibility alias."""

        return self.calibration_quality


@dataclass(frozen=True, slots=True)
class VehicleTrafficContext:
    track_id: int
    neighbors: NeighborVehicles
    nearby_vehicle_count: int
    adjacent_right_lane_id: str | None
    right_lane_available: bool | None
    right_lane_available_seconds: float
    right_lane_confidence: float
    right_lane_front_gap: GapEstimate | None = None
    right_lane_rear_gap: GapEstimate | None = None
    right_lane_opportunity_mode: str = "unavailable"


@dataclass(frozen=True, slots=True)
class TrafficFrameContext:
    global_context: GlobalTrafficContext
    vehicles: dict[int, VehicleTrafficContext]
    positions: dict[int, RoadPosition]
    speeds: dict[int, SpeedEstimate] | None = None


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
    candidate_lifecycle_state: str = CandidateLifecycleState.IDLE.value
    speed_kph: float | None = None
    speed_mode: str = "unavailable_uncalibrated"
    coordinate_mode: str = "normalized_image"
    right_lane_gap: GapEstimate | None = None
    physical_measurement_status: str = "unavailable"
    physical_measurement_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateDecisionRecord:
    timestamp_seconds: float
    decision: str
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    transition: Literal[
        "started",
        "suspended",
        "resumed",
        "pending_close",
        "cancelled",
        "finalized",
        "ended",
    ]
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
    lifecycle_state: str = CandidateLifecycleState.CANDIDATE_ACTIVE.value
    candidate_started_at: float | None = None
    suspended_at: float | None = None
    finalized_at: float | None = None
    cancelled_at: float | None = None
    cancellation_reason: str | None = None
    close_requested_at: float | None = None
    close_reason: str | None = None
    decision_history: tuple[CandidateDecisionRecord, ...] = ()
    position: RoadPosition | None = None
    speed_estimate: SpeedEstimate | None = None
    calibration_quality: CalibrationQuality | None = None
    camera_motion: CameraMotionEstimate | None = None
    camera_pose: CameraPoseStatus | None = None
    physical_measurements: PhysicalMeasurementPermission | None = None


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
    right_lane_opportunity_mode: str = "unavailable"
    right_lane_front_gap: GapEstimateMetadata | None = None
    right_lane_rear_gap: GapEstimateMetadata | None = None
    right_lane_front_gap_m: float | None = Field(default=None, ge=0)
    right_lane_rear_gap_m: float | None = Field(default=None, ge=0)
    right_lane_front_gap_normalized: float | None = Field(default=None, ge=0)
    right_lane_rear_gap_normalized: float | None = Field(default=None, ge=0)


class GapEstimateMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0)
    unit: Literal["meters", "normalized"]
    confidence: float = Field(ge=0, le=1)
    coordinate_mode: str


class CalibrationQualityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    matrix_valid: bool
    numerically_stable: bool
    validation_mode: str
    fit_reprojection_error_pixels: float | None = Field(default=None, ge=0)
    validation_reprojection_error_pixels: float | None = Field(default=None, ge=0)
    condition_metric: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    confidence_basis: str
    reason_codes: list[str] = Field(default_factory=list)
    world_units: str | None = None


class CameraMotionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dx: float
    dy: float
    rotation_degrees: float
    confidence: float = Field(ge=0, le=1)
    valid: bool
    level: str
    method: str
    stabilization_applied: bool = False


class CameraPoseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    translation_px: float | None = Field(default=None, ge=0)
    rotation_deg: float | None = None
    confidence: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=0)
    stabilization_applied: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class PhysicalMeasurementMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    confidence: float = Field(ge=0, le=1)
    status: str
    reason_codes: list[str] = Field(default_factory=list)


class SpeedEstimateMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speed_mps: float | None = Field(default=None, ge=0)
    speed_kph: float | None = Field(default=None, ge=0)
    speed_confidence: float = Field(ge=0, le=1)
    speed_mode: str
    normalized_motion_rate: float | None = Field(default=None, ge=0)
    sample_count: int = Field(default=0, ge=0)
    physical_measurement_status: str = "unavailable"
    reason_codes: list[str] = Field(default_factory=list)


class CandidateDecisionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_seconds: float = Field(ge=0)
    decision: str
    reason_codes: list[str] = Field(default_factory=list)


class CandidateLifecycleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    candidate_started_at: float | None = Field(default=None, ge=0)
    suspended_at: float | None = Field(default=None, ge=0)
    finalized_at: float | None = Field(default=None, ge=0)
    cancelled_at: float | None = Field(default=None, ge=0)
    cancellation_reason: str | None = None
    close_requested_at: float | None = Field(default=None, ge=0)
    close_reason: str | None = None


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

    schema_version: str = "3.1"
    event_id: str
    event_type: Literal["left_lane_review_candidate"] = "left_lane_review_candidate"
    review_status: Literal[
        "collecting_evidence", "pending_human_review", "cancelled"
    ] = "collecting_evidence"
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
    candidate_lifecycle: CandidateLifecycleMetadata | None = None
    decision_history: list[CandidateDecisionMetadata] = Field(default_factory=list)
    calibration: CalibrationQualityMetadata | None = None
    camera_motion: CameraMotionMetadata | None = None
    camera_pose: CameraPoseMetadata | None = None
    physical_measurements: PhysicalMeasurementMetadata | None = None
    speed_estimate: SpeedEstimateMetadata | None = None
    image_position: tuple[float, float] | None = None
    normalized_position: tuple[float, float] | None = None
    world_position_m: tuple[float, float] | None = None
    coordinate_mode: str = "normalized_image"

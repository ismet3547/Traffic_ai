"""Strict, prediction-free models for real dataset ground-truth operations."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.benchmark.models import AnnotationConfidence, DatasetSplit

DATASET_ANNOTATION_SCHEMA_VERSION: Final = "2.0"
DATASET_INTAKE_SCHEMA_VERSION: Final = "1.0"
DATASET_RELEASE_SCHEMA_VERSION: Final = "1.1"
ADJUDICATION_SCHEMA_VERSION: Final = "1.1"
ONTOLOGY_VERSION: Final = "pilot-1"
HANDBOOK_VERSION: Final = "1.0"
DATASET_VERSION: Final = "pilot-0.1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetLabel(str, Enum):
    UNNECESSARY_LEFT_LANE_OCCUPATION = "unnecessary_left_lane_occupation"
    LEGITIMATE_OVERTAKING = "legitimate_overtaking"
    CONGESTION_LEFT_LANE_USE = "congestion_left_lane_use"
    TEMPORARY_LEFT_LANE_USE = "temporary_left_lane_use"
    RIGHT_LANE_UNAVAILABLE = "right_lane_unavailable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GEOMETRY_INVALID = "geometry_invalid"
    CAMERA_MOTION_INVALID = "camera_motion_invalid"


class IntegrityReasonCode(str, Enum):
    DATASET_EMPTY = "DATASET_EMPTY"
    ANNOTATION_VIDEO_ID_MISMATCH = "ANNOTATION_VIDEO_ID_MISMATCH"
    ANNOTATION_SOURCE_VIDEO_MISMATCH = "ANNOTATION_SOURCE_VIDEO_MISMATCH"
    ANNOTATION_SOURCE_SIZE_MISMATCH = "ANNOTATION_SOURCE_SIZE_MISMATCH"
    ANNOTATION_PROTOCOL_MISMATCH = "ANNOTATION_PROTOCOL_MISMATCH"
    ANNOTATION_SCHEMA_INVALID = "ANNOTATION_SCHEMA_INVALID"
    ANNOTATION_CONTENT_HASH_MISMATCH = "ANNOTATION_CONTENT_HASH_MISMATCH"
    ANNOTATION_NOT_LOCKED = "ANNOTATION_NOT_LOCKED"
    ANNOTATOR_COUNT_INSUFFICIENT = "ANNOTATOR_COUNT_INSUFFICIENT"
    ANNOTATOR_REVISION_AMBIGUOUS = "ANNOTATOR_REVISION_AMBIGUOUS"
    ADJUDICATION_VIDEO_ID_MISMATCH = "ADJUDICATION_VIDEO_ID_MISMATCH"
    ADJUDICATION_SOURCE_VIDEO_MISMATCH = "ADJUDICATION_SOURCE_VIDEO_MISMATCH"
    ADJUDICATION_SOURCE_SIZE_MISMATCH = "ADJUDICATION_SOURCE_SIZE_MISMATCH"
    ADJUDICATION_ORIGINAL_HASH_MISMATCH = "ADJUDICATION_ORIGINAL_HASH_MISMATCH"
    ADJUDICATION_STALE_SOURCE_ANNOTATION = "ADJUDICATION_STALE_SOURCE_ANNOTATION"
    ADJUDICATION_NOT_APPROVED = "ADJUDICATION_NOT_APPROVED"
    ADJUDICATION_NOT_LOCKED = "ADJUDICATION_NOT_LOCKED"
    ADJUDICATION_SCHEMA_INVALID = "ADJUDICATION_SCHEMA_INVALID"
    ADJUDICATION_CONTENT_HASH_MISMATCH = "ADJUDICATION_CONTENT_HASH_MISMATCH"
    SPLIT_ASSIGNMENT_MISSING = "SPLIT_ASSIGNMENT_MISSING"
    SPLIT_ASSIGNMENT_DUPLICATE = "SPLIT_ASSIGNMENT_DUPLICATE"
    SPLIT_ASSIGNMENT_UNKNOWN_VIDEO = "SPLIT_ASSIGNMENT_UNKNOWN_VIDEO"
    SOURCE_GROUP_ID_MISMATCH = "SOURCE_GROUP_ID_MISMATCH"
    SOURCE_GROUP_SPLIT_LEAKAGE = "SOURCE_GROUP_SPLIT_LEAKAGE"
    DUPLICATE_VIDEO_CROSS_SPLIT_LEAKAGE = "DUPLICATE_VIDEO_CROSS_SPLIT_LEAKAGE"
    SOURCE_VIDEO_IDENTITY_UNVERIFIED = "SOURCE_VIDEO_IDENTITY_UNVERIFIED"
    BENCHMARK_USE_NOT_ALLOWED = "BENCHMARK_USE_NOT_ALLOWED"
    UNKNOWN_ANNOTATION_VIDEO = "UNKNOWN_ANNOTATION_VIDEO"
    UNKNOWN_ADJUDICATION_VIDEO = "UNKNOWN_ADJUDICATION_VIDEO"
    FINAL_GROUND_TRUTH_HASH_MISMATCH = "FINAL_GROUND_TRUTH_HASH_MISMATCH"


class VisibilityQuality(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class VehicleClass(str, Enum):
    PASSENGER_CAR = "passenger_car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    OTHER = "other"
    UNKNOWN = "unknown"


TriState = bool | Literal["unknown"]


class EventEvidence(StrictModel):
    entered_left_from_right: TriState | None = None
    observed_vehicle_being_passed: TriState | None = None
    right_lane_available: TriState | None = None
    congestion_present: TriState | None = None
    returned_right: TriState | None = None
    visibility_quality: VisibilityQuality | None = None
    vehicle_class: VehicleClass | None = None


class RepresentativeBox(StrictModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    coordinate_mode: Literal["pixels", "normalized"] = "pixels"

    @model_validator(mode="after")
    def validate_normalized(self) -> RepresentativeBox:
        if self.coordinate_mode == "normalized" and (
            self.x > 1
            or self.y > 1
            or self.width > 1
            or self.height > 1
            or self.x + self.width > 1 + 1e-9
            or self.y + self.height > 1 + 1e-9
        ):
            raise ValueError("normalized representative box must fit within [0, 1]")
        return self


class DatasetEvent(StrictModel):
    event_id: str = Field(min_length=1, max_length=120)
    vehicle_ref: str = Field(pattern=r"^vehicle_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    label: DatasetLabel
    confidence: AnnotationConfidence
    evidence: EventEvidence = Field(default_factory=EventEvidence)
    reference_timestamp_seconds: float | None = Field(default=None, ge=0)
    representative_box: RepresentativeBox | None = None
    vehicle_description: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_event(self) -> DatasetEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("event end_seconds must be greater than start_seconds")
        if self.reference_timestamp_seconds is not None and not (
            self.start_seconds - 1e-9
            <= self.reference_timestamp_seconds
            <= self.end_seconds + 1e-9
        ):
            raise ValueError("reference timestamp must fall within the event interval")
        return self


class LockOverrideRecord(StrictModel):
    timestamp: datetime
    action: Literal["override_edit", "unlock", "relock"]
    reason: str = Field(min_length=3, max_length=500)


class DatasetAnnotation(StrictModel):
    schema_version: Literal["2.0"] = DATASET_ANNOTATION_SCHEMA_VERSION
    ontology_version: str = ONTOLOGY_VERSION
    handbook_version: str = HANDBOOK_VERSION
    video_id: str = Field(min_length=1)
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_size_bytes: int | None = Field(default=None, gt=0)
    source_file: str = Field(min_length=1)
    fps: float = Field(gt=0)
    video_duration_seconds: float = Field(gt=0)
    annotator_id: str = Field(min_length=1, max_length=80)
    independent_pass: Literal[True] = True
    created_at: datetime
    events: list[DatasetEvent] = Field(default_factory=list)
    locked: bool = False
    locked_at: datetime | None = None
    annotation_hash: str | None = None
    lock_override_history: list[LockOverrideRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_annotation(self) -> DatasetAnnotation:
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("annotation event IDs must be unique")
        outside = [
            event.event_id
            for event in self.events
            if event.end_seconds > self.video_duration_seconds + 1e-9
        ]
        if outside:
            raise ValueError(
                "annotation timestamps exceed video duration: "
                + ", ".join(sorted(outside))
            )
        if self.locked and (self.locked_at is None or self.annotation_hash is None):
            raise ValueError("locked annotation requires locked_at and annotation_hash")
        if self.annotation_hash is not None and (
            len(self.annotation_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.annotation_hash
            )
        ):
            raise ValueError("annotation_hash must be lowercase SHA-256")
        return self


class SourceType(str, Enum):
    USER_PROVIDED = "user_provided"
    OWN_CAPTURE = "own_capture"
    PUBLIC_DATASET = "public_dataset"
    LICENSED_SOURCE = "licensed_source"
    OTHER = "other"


class PermissionStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    DENIED = "denied"
    UNKNOWN = "unknown"


class VideoResolution(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class VideoIntakeRecord(StrictModel):
    video_id: str = Field(min_length=1, max_length=120)
    source_group_id: str = Field(min_length=1, max_length=120)
    source_type: SourceType
    source_reference: str = Field(min_length=1, max_length=1000)
    acquisition_date: date
    license_or_permission_status: PermissionStatus
    redistribution_allowed: bool
    benchmark_use_allowed: bool
    notes: str | None = Field(default=None, max_length=4000)
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_size_bytes: int = Field(gt=0)
    source_identity_verified: bool = True
    duration_seconds: float = Field(gt=0)
    resolution: VideoResolution
    fps: float = Field(gt=0)
    original_filename: str = Field(min_length=1)
    scenario_tags: list[str] = Field(default_factory=list)
    vehicle_classes: list[VehicleClass] = Field(default_factory=list)
    split: DatasetSplit | None = None

    @model_validator(mode="after")
    def normalize_tags(self) -> VideoIntakeRecord:
        tags = [tag.strip().lower() for tag in self.scenario_tags]
        if any(not tag for tag in tags):
            raise ValueError("scenario tags must not be empty")
        if len(tags) != len(set(tags)):
            raise ValueError("scenario tags must be unique")
        self.scenario_tags = sorted(tags)
        self.vehicle_classes = sorted(
            set(self.vehicle_classes), key=lambda item: item.value
        )
        return self


class IntakeRegistry(StrictModel):
    schema_version: Literal["1.0"] = DATASET_INTAKE_SCHEMA_VERSION
    dataset_version: str = DATASET_VERSION
    videos: list[VideoIntakeRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_registry(self) -> IntakeRegistry:
        video_ids = [video.video_id for video in self.videos]
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("intake video IDs must be unique")
        return self


class AgreementConfig(StrictModel):
    minimum_temporal_iou: float = Field(default=0.30, gt=0, le=1)
    boundary_tolerance_seconds: float = Field(default=1.0, ge=0)
    require_vehicle_reference_match: bool = True


class DisagreementType(str, Enum):
    LABEL_DISAGREEMENT = "label_disagreement"
    EVENT_MISSING_A = "event_missing_a"
    EVENT_MISSING_B = "event_missing_b"
    BOUNDARY_DISAGREEMENT = "boundary_disagreement"
    CONFIDENCE_DISAGREEMENT = "confidence_disagreement"
    VEHICLE_REFERENCE_DISAGREEMENT = "vehicle_reference_disagreement"
    AMBIGUOUS_VISIBILITY = "ambiguous_visibility"


class AgreementMatch(StrictModel):
    event_id_a: str
    event_id_b: str
    temporal_iou: float = Field(ge=0, le=1)
    start_difference_seconds: float = Field(ge=0)
    end_difference_seconds: float = Field(ge=0)
    label_agrees: bool
    confidence_agrees: bool
    vehicle_reference_agrees: bool


class AnnotationDisagreement(StrictModel):
    disagreement_id: str
    disagreement_types: list[DisagreementType]
    event_id_a: str | None = None
    event_id_b: str | None = None
    rationale: str


class AgreementReport(StrictModel):
    video_id: str
    annotator_a: str
    annotator_b: str
    config: AgreementConfig
    matched_event_count: int = Field(ge=0)
    event_detection_agreement: float = Field(ge=0, le=1)
    label_agreement: float = Field(ge=0, le=1)
    temporal_boundary_agreement: float = Field(ge=0, le=1)
    confidence_agreement: float = Field(ge=0, le=1)
    mean_temporal_iou: float | None = Field(default=None, ge=0, le=1)
    cohen_kappa_matched_labels: float | None = None
    disagreement_count: int = Field(ge=0)
    matches: list[AgreementMatch] = Field(default_factory=list)
    disagreements: list[AnnotationDisagreement] = Field(default_factory=list)
    caveat: str


class AdjudicationOutcome(str, Enum):
    AGREE = "agree"
    RESOLVED_TO_A = "resolved_to_a"
    RESOLVED_TO_B = "resolved_to_b"
    NEW_CONSENSUS = "new_consensus"
    REMAINS_AMBIGUOUS = "remains_ambiguous"


class AdjudicationDecision(StrictModel):
    decision_id: str
    disagreement_ids: list[str] = Field(default_factory=list)
    event_ids_a: list[str] = Field(default_factory=list)
    event_ids_b: list[str] = Field(default_factory=list)
    outcome: AdjudicationOutcome
    adjudicated_event: DatasetEvent | None = None
    rationale: str = Field(min_length=3, max_length=2000)
    adjudication_confidence: AnnotationConfidence

    @model_validator(mode="after")
    def validate_decision(self) -> AdjudicationDecision:
        if self.outcome == AdjudicationOutcome.REMAINS_AMBIGUOUS:
            if (
                self.adjudicated_event is None
                or self.adjudicated_event.label != DatasetLabel.INSUFFICIENT_EVIDENCE
            ):
                raise ValueError(
                    "ambiguous adjudication requires an insufficient_evidence event"
                )
        elif self.adjudicated_event is None:
            raise ValueError("resolved adjudication requires an adjudicated event")
        return self


class AdjudicationArtifact(StrictModel):
    schema_version: Literal["1.1"] = ADJUDICATION_SCHEMA_VERSION
    video_id: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_size_bytes: int | None = Field(default=None, gt=0)
    ontology_version: str = ONTOLOGY_VERSION
    handbook_version: str = HANDBOOK_VERSION
    annotation_a: DatasetAnnotation
    annotation_b: DatasetAnnotation
    annotation_a_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_b_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agreement_report: AgreementReport
    adjudicator_id: str = Field(min_length=1, max_length=80)
    created_at: datetime
    decisions: list[AdjudicationDecision]
    final_events: list[DatasetEvent]
    approved: bool = False
    locked: bool = False
    locked_at: datetime | None = None
    adjudication_hash: str | None = None

    @model_validator(mode="after")
    def validate_artifact(self) -> AdjudicationArtifact:
        if (
            self.annotation_a.video_id != self.video_id
            or self.annotation_b.video_id != self.video_id
        ):
            raise ValueError("adjudication annotations must match video_id")
        if self.agreement_report.video_id != self.video_id or {
            self.agreement_report.annotator_a,
            self.agreement_report.annotator_b,
        } != {self.annotation_a.annotator_id, self.annotation_b.annotator_id}:
            raise ValueError(
                "adjudication agreement report does not match original annotations"
            )
        if any(
            annotation.ontology_version != self.ontology_version
            or annotation.handbook_version != self.handbook_version
            for annotation in (self.annotation_a, self.annotation_b)
        ):
            raise ValueError(
                "adjudication protocol versions must match original annotations"
            )
        if (
            self.annotation_a.source_video_sha256 != self.source_video_sha256
            or self.annotation_b.source_video_sha256 != self.source_video_sha256
        ):
            raise ValueError(
                "adjudication source identity must match both original annotations"
            )
        annotation_sizes = {
            value
            for value in (
                self.annotation_a.source_video_size_bytes,
                self.annotation_b.source_video_size_bytes,
            )
            if value is not None
        }
        if len(annotation_sizes) > 1 or (
            annotation_sizes and self.source_video_size_bytes not in annotation_sizes
        ):
            raise ValueError("adjudication source size must match original annotations")
        final_ids = [event.event_id for event in self.final_events]
        if len(final_ids) != len(set(final_ids)):
            raise ValueError("adjudicated event IDs must be unique")
        if self.locked and (self.locked_at is None or self.adjudication_hash is None):
            raise ValueError("locked adjudication requires timestamp and hash")
        disagreement_ids = {
            item.disagreement_id for item in self.agreement_report.disagreements
        }
        reviewed_ids = [
            disagreement_id
            for decision in self.decisions
            for disagreement_id in decision.disagreement_ids
        ]
        if set(reviewed_ids) != disagreement_ids or len(reviewed_ids) != len(
            set(reviewed_ids)
        ):
            raise ValueError(
                "adjudication decisions must review every disagreement exactly once"
            )
        return self


class SplitCandidate(StrictModel):
    video_id: str
    source_group_id: str
    duration_seconds: float = Field(gt=0)
    labels: list[DatasetLabel] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)


class SplitAssignment(StrictModel):
    video_id: str
    source_group_id: str
    split: DatasetSplit


class SplitAssignmentDocument(StrictModel):
    seed: int
    target_ratios: dict[DatasetSplit, float]
    assignments: list[SplitAssignment]
    note: str = (
        "Source-group isolation is mandatory; label/scenario balance is approximate."
    )


class CoverageReport(StrictModel):
    total_clips: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    labels: list[str]
    label_counts: dict[str, int]
    confidence_counts: dict[str, int]
    scenario_tag_counts: dict[str, int]
    day_night_distribution: dict[str, int]
    traffic_density_distribution: dict[str, int]
    camera_configuration_distribution: dict[str, int]
    vehicle_class_distribution: dict[str, int]
    split_counts: dict[str, int]
    agreement_statistics: dict[str, float | int | None]


class QualityGateResult(StrictModel):
    gate: str
    passed: bool
    details: str


class IntegrityIssue(StrictModel):
    reason_code: IntegrityReasonCode
    details: str
    video_id: str | None = None
    source_group_id: str | None = None


class ArtifactIntegrityResult(StrictModel):
    valid: bool
    reason_codes: list[IntegrityReasonCode]
    issues: list[IntegrityIssue] = Field(default_factory=list)


class DatasetReleaseIntegrityReport(StrictModel):
    passed: bool
    gates: list[QualityGateResult]
    reason_codes: list[IntegrityReasonCode]
    affected_video_ids: list[str]
    affected_source_group_ids: list[str]
    issues: list[IntegrityIssue] = Field(default_factory=list)


class IntegrityScenarioOutcome(StrictModel):
    scenario: str
    expected: Literal["PASS", "FAIL"]
    actual: Literal["PASS", "FAIL"]
    expectation_met: bool
    reason_codes: list[IntegrityReasonCode] = Field(default_factory=list)
    release_written: bool


class IntegrityScenarioSummary(StrictModel):
    all_expectations_met: bool
    scenarios: list[IntegrityScenarioOutcome]


class ValidatedDoubleAnnotation(StrictModel):
    valid: bool
    annotator_count: int = Field(ge=0)
    source_identity_valid: bool
    protocol_versions_compatible: bool
    locked: bool
    current_revisions: bool
    annotation_hashes: dict[str, str] = Field(default_factory=dict)
    reason_codes: list[IntegrityReasonCode] = Field(default_factory=list)


class AnnotationQualityConfig(StrictModel):
    minimum_label_agreement: float | None = Field(default=None, ge=0, le=1)
    minimum_event_match_rate: float | None = Field(default=None, ge=0, le=1)


class ReleaseVideo(StrictModel):
    video_id: str
    source_group_id: str
    split: DatasetSplit
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_size_bytes: int = Field(gt=0)
    source_identity_verified: bool
    duration_seconds: float = Field(gt=0)
    ontology_version: str
    handbook_version: str
    annotation_hashes: dict[str, str]
    adjudicated_annotation_hash: str | None = None
    benchmark_ground_truth_sha256: str | None = None
    double_annotated: bool
    double_annotation: ValidatedDoubleAnnotation
    adjudication_status: Literal["not_required", "pending", "approved", "ambiguous"]
    test_annotation_locked: bool
    license_or_permission_status: PermissionStatus
    redistribution_allowed: bool
    benchmark_use_allowed: bool


class DatasetRelease(StrictModel):
    schema_version: Literal["1.1"] = DATASET_RELEASE_SCHEMA_VERSION
    dataset_version: str = DATASET_VERSION
    created_at: datetime
    ontology_version: str = ONTOLOGY_VERSION
    handbook_version: str = HANDBOOK_VERSION
    videos: list[ReleaseVideo]
    integrity_report: DatasetReleaseIntegrityReport
    quality_gates: list[QualityGateResult]
    quality_gate_passed: bool
    notes: list[str] = Field(default_factory=list)


JsonObject = dict[str, Any]

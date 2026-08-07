"""Strict, prediction-free models for real dataset ground-truth operations."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.benchmark.fingerprints import canonical_sha256
from app.benchmark.models import AnnotationConfidence, DatasetSplit

DATASET_ANNOTATION_SCHEMA_VERSION: Final = "2.0"
DATASET_INTAKE_SCHEMA_VERSION: Final = "1.0"
DATASET_RELEASE_SCHEMA_VERSION: Final = "1.3"
ADJUDICATION_SCHEMA_VERSION: Final = "1.3"
AGREEMENT_REPORT_SCHEMA_VERSION: Final = "1.1"
AGREEMENT_PROTOCOL_VERSION: Final = "2"
AGREEMENT_CONFIG_VERSION: Final = "1"
AGREEMENT_AGGREGATION_MODE: Final = "macro_per_video"
AGREEMENT_AGGREGATION_VERSION: Final = "1"
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
    AGREEMENT_REPORT_UNKNOWN_VIDEO = "AGREEMENT_REPORT_UNKNOWN_VIDEO"
    DUPLICATE_AGREEMENT_REPORT = "DUPLICATE_AGREEMENT_REPORT"
    STALE_AGREEMENT_REPORT = "STALE_AGREEMENT_REPORT"
    AGREEMENT_SOURCE_VIDEO_MISMATCH = "AGREEMENT_SOURCE_VIDEO_MISMATCH"
    AGREEMENT_SOURCE_SIZE_MISMATCH = "AGREEMENT_SOURCE_SIZE_MISMATCH"
    AGREEMENT_ANNOTATOR_MISMATCH = "AGREEMENT_ANNOTATOR_MISMATCH"
    AGREEMENT_PROTOCOL_UNSUPPORTED = "AGREEMENT_PROTOCOL_UNSUPPORTED"
    AGREEMENT_ONTOLOGY_MISMATCH = "AGREEMENT_ONTOLOGY_MISMATCH"
    AGREEMENT_HANDBOOK_MISMATCH = "AGREEMENT_HANDBOOK_MISMATCH"
    AGREEMENT_INTERNAL_INCOHERENT = "AGREEMENT_INTERNAL_INCOHERENT"
    AGREEMENT_CONTENT_HASH_MISMATCH = "AGREEMENT_CONTENT_HASH_MISMATCH"
    MISSING_CURRENT_AGREEMENT_REPORT = "MISSING_CURRENT_AGREEMENT_REPORT"
    AGREEMENT_ADJUDICATION_REVISION_MISMATCH = (
        "AGREEMENT_ADJUDICATION_REVISION_MISMATCH"
    )
    AGREEMENT_ADJUDICATION_REPORT_MISMATCH = "AGREEMENT_ADJUDICATION_REPORT_MISMATCH"
    AGREEMENT_PROTOCOL_MISMATCH = "AGREEMENT_PROTOCOL_MISMATCH"
    AGREEMENT_CONFIG_MISMATCH = "AGREEMENT_CONFIG_MISMATCH"
    AGREEMENT_MODE_NOT_OFFICIAL = "AGREEMENT_MODE_NOT_OFFICIAL"
    MIXED_AGREEMENT_PROTOCOLS = "MIXED_AGREEMENT_PROTOCOLS"


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
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_temporal_iou: float = Field(default=0.30, gt=0, le=1)
    boundary_tolerance_seconds: float = Field(default=1.0, ge=0)
    require_vehicle_reference_match: bool = True


class AgreementMode(str, Enum):
    OFFICIAL = "official"
    EXPLORATORY = "exploratory"


def agreement_config_fingerprint(config: AgreementConfig) -> str:
    """Stable semantic identity for every agreement matching parameter."""
    return canonical_sha256(config.model_dump(mode="json"))


CANONICAL_AGREEMENT_CONFIG: Final = AgreementConfig(
    minimum_temporal_iou=0.30,
    boundary_tolerance_seconds=1.0,
    require_vehicle_reference_match=True,
)
CANONICAL_AGREEMENT_CONFIG_FINGERPRINT: Final = agreement_config_fingerprint(
    CANONICAL_AGREEMENT_CONFIG
)


class AgreementProtocolIdentity(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str
    config_version: str
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregation_version: str
    ontology_version: str
    handbook_version: str
    agreement_config: AgreementConfig


CANONICAL_AGREEMENT_PROTOCOL: Final = AgreementProtocolIdentity(
    protocol_version=AGREEMENT_PROTOCOL_VERSION,
    config_version=AGREEMENT_CONFIG_VERSION,
    config_fingerprint=CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
    aggregation_version=AGREEMENT_AGGREGATION_VERSION,
    ontology_version=ONTOLOGY_VERSION,
    handbook_version=HANDBOOK_VERSION,
    agreement_config=CANONICAL_AGREEMENT_CONFIG,
)


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
    schema_version: Literal["1.1"] = AGREEMENT_REPORT_SCHEMA_VERSION
    agreement_mode: AgreementMode = AgreementMode.OFFICIAL
    agreement_protocol_version: str = AGREEMENT_PROTOCOL_VERSION
    agreement_config_version: str = AGREEMENT_CONFIG_VERSION
    agreement_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    agreement_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    agreement_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_id: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_size_bytes: int | None = Field(default=None, gt=0)
    annotator_a_id: str
    annotator_b_id: str
    annotation_a_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_b_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_a_ontology_version: str
    annotation_b_ontology_version: str
    annotation_a_handbook_version: str
    annotation_b_handbook_version: str
    annotation_a_event_count: int = Field(ge=0)
    annotation_b_event_count: int = Field(ge=0)
    agreement_config: AgreementConfig
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

    @model_validator(mode="after")
    def validate_internal_coherence(self) -> AgreementReport:
        if self.annotator_a_id == self.annotator_b_id:
            raise ValueError("agreement requires distinct annotator IDs")
        if self.matched_event_count != len(self.matches):
            raise ValueError("matched_event_count does not match matches")
        if self.disagreement_count != len(self.disagreements):
            raise ValueError("disagreement_count does not match disagreements")
        if self.agreement_config_fingerprint != agreement_config_fingerprint(
            self.agreement_config
        ):
            raise ValueError("agreement config fingerprint is internally incoherent")
        denominator = self.annotation_a_event_count + self.annotation_b_event_count
        expected_detection = (
            2 * self.matched_event_count / denominator if denominator else 1.0
        )
        if abs(self.event_detection_agreement - expected_detection) > 1e-9:
            raise ValueError("event_detection_agreement is internally incoherent")
        matched = self.matched_event_count
        expected_label = (
            sum(item.label_agrees for item in self.matches) / matched
            if matched
            else 0.0
        )
        expected_boundary = (
            sum(
                item.start_difference_seconds
                <= self.agreement_config.boundary_tolerance_seconds + 1e-12
                and item.end_difference_seconds
                <= self.agreement_config.boundary_tolerance_seconds + 1e-12
                for item in self.matches
            )
            / matched
            if matched
            else 0.0
        )
        expected_confidence = (
            sum(item.confidence_agrees for item in self.matches) / matched
            if matched
            else 0.0
        )
        expected_iou = (
            sum(item.temporal_iou for item in self.matches) / matched
            if matched
            else None
        )
        if abs(self.label_agreement - expected_label) > 1e-9:
            raise ValueError("label_agreement is internally incoherent")
        if abs(self.temporal_boundary_agreement - expected_boundary) > 1e-9:
            raise ValueError("temporal_boundary_agreement is internally incoherent")
        if abs(self.confidence_agreement - expected_confidence) > 1e-9:
            raise ValueError("confidence_agreement is internally incoherent")
        if (self.mean_temporal_iou is None) != (expected_iou is None) or (
            expected_iou is not None
            and self.mean_temporal_iou is not None
            and abs(self.mean_temporal_iou - expected_iou) > 1e-9
        ):
            raise ValueError("mean_temporal_iou is internally incoherent")
        return self


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
    schema_version: Literal["1.3"] = ADJUDICATION_SCHEMA_VERSION
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
            self.agreement_report.annotator_a_id,
            self.agreement_report.annotator_b_id,
        } != {self.annotation_a.annotator_id, self.annotation_b.annotator_id}:
            raise ValueError(
                "adjudication agreement report does not match original annotations"
            )
        agreement_hashes = {
            self.agreement_report.annotator_a_id: self.agreement_report.annotation_a_content_sha256,
            self.agreement_report.annotator_b_id: self.agreement_report.annotation_b_content_sha256,
        }
        if agreement_hashes != {
            self.annotation_a.annotator_id: self.annotation_a_hash,
            self.annotation_b.annotator_id: self.annotation_b_hash,
        }:
            raise ValueError(
                "adjudication agreement report uses different annotation revisions"
            )
        if self.agreement_report.source_video_sha256 != self.source_video_sha256:
            raise ValueError("adjudication agreement source identity differs")
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
    agreement_statistics: dict[str, float | int | str | None]


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


class AgreementCoverage(StrictModel):
    required_video_count: int = Field(ge=0)
    validated_report_count: int = Field(ge=0)
    missing_report_count: int = Field(ge=0)
    stale_report_count: int = Field(ge=0)
    duplicate_report_count: int = Field(ge=0)
    unknown_report_count: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_coverage(self) -> AgreementCoverage:
        if self.validated_report_count + self.missing_report_count != (
            self.required_video_count
        ):
            raise ValueError("agreement coverage counts do not match required videos")
        expected = (
            self.validated_report_count / self.required_video_count
            if self.required_video_count
            else 1.0
        )
        if abs(self.coverage_ratio - expected) > 1e-9:
            raise ValueError("agreement coverage ratio is internally incoherent")
        return self


class AgreementQualitySummary(StrictModel):
    aggregation_mode: Literal["macro_per_video"] = AGREEMENT_AGGREGATION_MODE
    validated_report_count: int = Field(ge=0)
    total_agreement_videos: int = Field(ge=0)
    zero_event_both_annotators_video_count: int = Field(ge=0)
    positive_event_video_count: int = Field(ge=0)
    label_agreement: float | None = Field(default=None, ge=0, le=1)
    event_detection_agreement: float | None = Field(default=None, ge=0, le=1)
    positive_event_video_event_detection_agreement: float | None = Field(
        default=None, ge=0, le=1
    )
    temporal_boundary_agreement: float | None = Field(default=None, ge=0, le=1)
    confidence_agreement: float | None = Field(default=None, ge=0, le=1)
    thresholds_passed: bool

    @model_validator(mode="after")
    def validate_video_counts(self) -> AgreementQualitySummary:
        if self.validated_report_count != self.total_agreement_videos:
            raise ValueError("agreement quality report counts differ")
        if (
            self.zero_event_both_annotators_video_count
            + self.positive_event_video_count
            != self.total_agreement_videos
        ):
            raise ValueError(
                "zero-event and positive-event video counts are incoherent"
            )
        if (
            self.positive_event_video_count == 0
            and self.positive_event_video_event_detection_agreement is not None
        ):
            raise ValueError("positive-event agreement requires positive-event videos")
        return self


class ValidatedAgreementSet(StrictModel):
    valid: bool
    reports: list[AgreementReport]
    coverage: AgreementCoverage
    issues: list[IntegrityIssue] = Field(default_factory=list)


class ReleaseAgreementProvenance(StrictModel):
    agreement_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    agreement_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    agreement_protocol_version: str
    agreement_config_version: str
    agreement_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    agreement_mode: AgreementMode
    annotation_content_sha256: list[str] = Field(min_length=2, max_length=2)
    aggregation_unit: Literal["video"] = "video"


class IntegrityScenarioOutcome(StrictModel):
    scenario: str
    expected: Literal["PASS", "FAIL"]
    actual: Literal["PASS", "FAIL"]
    expectation_met: bool
    reason_codes: list[IntegrityReasonCode] = Field(default_factory=list)
    release_written: bool
    expected_label_agreement: float | None = Field(default=None, ge=0, le=1)
    actual_label_agreement: float | None = Field(default=None, ge=0, le=1)
    expected_event_detection_agreement: float | None = Field(default=None, ge=0, le=1)
    actual_event_detection_agreement: float | None = Field(default=None, ge=0, le=1)
    supplied_event_detection_agreement: float | None = Field(default=None, ge=0, le=1)
    canonical_recomputed_event_detection_agreement: float | None = Field(
        default=None, ge=0, le=1
    )
    agreement_mode: AgreementMode | None = None
    agreement_protocol_version: str | None = None
    agreement_config_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


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
    agreement_provenance: ReleaseAgreementProvenance | None = None
    adjudication_status: Literal["not_required", "pending", "approved", "ambiguous"]
    test_annotation_locked: bool
    license_or_permission_status: PermissionStatus
    redistribution_allowed: bool
    benchmark_use_allowed: bool


class DatasetRelease(StrictModel):
    schema_version: Literal["1.3"] = DATASET_RELEASE_SCHEMA_VERSION
    dataset_version: str = DATASET_VERSION
    created_at: datetime
    ontology_version: str = ONTOLOGY_VERSION
    handbook_version: str = HANDBOOK_VERSION
    videos: list[ReleaseVideo]
    integrity_report: DatasetReleaseIntegrityReport
    agreement_protocol: AgreementProtocolIdentity
    agreement_coverage: AgreementCoverage
    agreement_quality: AgreementQualitySummary
    quality_gates: list[QualityGateResult]
    quality_gate_passed: bool
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_agreement_provenance(self) -> DatasetRelease:
        if self.agreement_protocol != CANONICAL_AGREEMENT_PROTOCOL:
            raise ValueError("release agreement protocol is not canonical")
        if self.agreement_coverage.validated_report_count != (
            self.agreement_quality.validated_report_count
        ):
            raise ValueError("agreement coverage and quality report counts differ")
        for video in self.videos:
            requires_agreement = video.split in {
                DatasetSplit.VALIDATION,
                DatasetSplit.TEST,
            }
            if requires_agreement and video.agreement_provenance is None:
                raise ValueError(
                    "validation/test release video lacks agreement provenance"
                )
            if video.agreement_provenance is not None and sorted(
                video.annotation_hashes.values()
            ) != sorted(video.agreement_provenance.annotation_content_sha256):
                raise ValueError(
                    "release agreement provenance differs from annotation hashes"
                )
            if video.agreement_provenance is not None and (
                video.agreement_provenance.agreement_mode != AgreementMode.OFFICIAL
                or video.agreement_provenance.agreement_protocol_version
                != self.agreement_protocol.protocol_version
                or video.agreement_provenance.agreement_config_version
                != self.agreement_protocol.config_version
                or video.agreement_provenance.agreement_config_fingerprint
                != self.agreement_protocol.config_fingerprint
            ):
                raise ValueError(
                    "release video agreement provenance is not canonical official"
                )
        return self


JsonObject = dict[str, Any]

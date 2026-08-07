"""Versioned, detector-agnostic models for benchmark inputs and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANNOTATION_SCHEMA_VERSION: Final = "1.0"
BENCHMARK_SCHEMA_VERSION: Final = "1.0"
PREDICTION_SCHEMA_VERSION: Final = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnnotationLabel(str, Enum):
    UNNECESSARY_LEFT_LANE_OCCUPATION = "unnecessary_left_lane_occupation"
    LEGITIMATE_OVERTAKING = "legitimate_overtaking"
    CONGESTION_LEFT_LANE_USE = "congestion_left_lane_use"
    RIGHT_LANE_UNAVAILABLE = "right_lane_unavailable"
    TEMPORARY_LEFT_LANE_USE = "temporary_left_lane_use"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GEOMETRY_INVALID = "geometry_invalid"
    CAMERA_MOTION = "camera_motion"
    LANE_ASSIGNMENT_UNCERTAIN = "lane_assignment_uncertain"


class AnnotationConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnnotationRole(str, Enum):
    POSITIVE = "positive"
    NEGATIVE_CONTROL = "negative_control"
    IGNORE_REGION = "ignore_region"
    DIAGNOSTIC = "diagnostic"


ANNOTATION_ROLE_BY_LABEL: dict[AnnotationLabel, AnnotationRole] = {
    AnnotationLabel.UNNECESSARY_LEFT_LANE_OCCUPATION: AnnotationRole.POSITIVE,
    AnnotationLabel.LEGITIMATE_OVERTAKING: AnnotationRole.NEGATIVE_CONTROL,
    AnnotationLabel.CONGESTION_LEFT_LANE_USE: AnnotationRole.NEGATIVE_CONTROL,
    AnnotationLabel.RIGHT_LANE_UNAVAILABLE: AnnotationRole.NEGATIVE_CONTROL,
    AnnotationLabel.TEMPORARY_LEFT_LANE_USE: AnnotationRole.NEGATIVE_CONTROL,
    AnnotationLabel.INSUFFICIENT_EVIDENCE: AnnotationRole.IGNORE_REGION,
    AnnotationLabel.GEOMETRY_INVALID: AnnotationRole.NEGATIVE_CONTROL,
    AnnotationLabel.CAMERA_MOTION: AnnotationRole.DIAGNOSTIC,
    AnnotationLabel.LANE_ASSIGNMENT_UNCERTAIN: AnnotationRole.DIAGNOSTIC,
}


def annotation_role(label: AnnotationLabel) -> AnnotationRole:
    return ANNOTATION_ROLE_BY_LABEL[label]


class DatasetSplit(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


class GroundTruthEvent(StrictModel):
    event_id: str = Field(min_length=1)
    vehicle_track_hint: int | str | None = None
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    label: AnnotationLabel
    role: AnnotationRole | None = None
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH
    notes: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> GroundTruthEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                "annotation end_seconds must be greater than start_seconds"
            )
        expected_role = annotation_role(self.label)
        if self.role is not None and self.role != expected_role:
            raise ValueError(
                f"annotation role {self.role.value!r} is incompatible with label "
                f"{self.label.value!r}; expected {expected_role.value!r}"
            )
        self.role = expected_role
        return self


class AnnotationDocument(StrictModel):
    schema_version: Literal["1.0"] = ANNOTATION_SCHEMA_VERSION
    video_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    fps: float = Field(gt=0)
    video_duration_seconds: float | None = Field(default=None, gt=0)
    annotator_id: str | None = Field(default=None, min_length=1, max_length=80)
    events: list[GroundTruthEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_events(self) -> AnnotationDocument:
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("annotation event IDs must be unique within a document")
        if self.video_duration_seconds is not None:
            outside = [
                event.event_id
                for event in self.events
                if event.end_seconds > self.video_duration_seconds + 1e-9
            ]
            if outside:
                raise ValueError(
                    "annotation timestamps exceed video_duration_seconds: "
                    + ", ".join(sorted(outside))
                )
        return self


class MatchingConfig(StrictModel):
    minimum_temporal_iou: float = Field(default=0.30, ge=0, le=1)
    start_tolerance_seconds: float | None = Field(default=2.0, ge=0)
    require_track_association_if_available: bool = False


class IgnoredRegionConfig(StrictModel):
    enabled: bool = True
    minimum_prediction_coverage: float = Field(default=0.50, gt=0, le=1)
    minimum_temporal_iou: float = Field(default=0.0, ge=0, le=1)
    allowed_labels: list[AnnotationLabel] = Field(
        default_factory=lambda: [AnnotationLabel.INSUFFICIENT_EVIDENCE]
    )

    @model_validator(mode="after")
    def validate_labels(self) -> IgnoredRegionConfig:
        if len(self.allowed_labels) != len(set(self.allowed_labels)):
            raise ValueError("ignored-region labels must not contain duplicates")
        invalid = [
            label.value
            for label in self.allowed_labels
            if annotation_role(label) != AnnotationRole.IGNORE_REGION
        ]
        if invalid:
            raise ValueError(
                "ignored-region labels must have canonical IGNORE_REGION role: "
                + ", ".join(sorted(invalid))
            )
        return self


class ControlEventConfig(StrictModel):
    minimum_prediction_coverage: float = Field(default=0.50, gt=0, le=1)
    minimum_temporal_iou: float = Field(default=0.20, gt=0, le=1)


class DurationValidationConfig(StrictModel):
    absolute_tolerance_seconds: float = Field(default=1.0, ge=0)
    relative_tolerance: float = Field(default=0.01, ge=0, le=1)
    require_multiple_sources_for_acceptance: bool = True


class AcceptanceCriteria(StrictModel):
    minimum_precision: float | None = Field(default=None, ge=0, le=1)
    minimum_recall: float | None = Field(default=None, ge=0, le=1)
    maximum_false_positives_per_hour: float | None = Field(default=None, ge=0)


class BaselineTolerances(StrictModel):
    precision: float = Field(default=1e-6, ge=0)
    recall: float = Field(default=1e-6, ge=0)
    f1: float = Field(default=1e-6, ge=0)
    false_positives_per_hour: float = Field(default=1e-6, ge=0)
    processing_fps: float = Field(default=1e-3, ge=0)


class BenchmarkSettings(StrictModel):
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    ignored_regions: IgnoredRegionConfig = Field(default_factory=IgnoredRegionConfig)
    control_events: ControlEventConfig = Field(default_factory=ControlEventConfig)
    duration_validation: DurationValidationConfig = Field(
        default_factory=DurationValidationConfig
    )
    minimum_prediction_confidence: float = Field(default=0.0, ge=0, le=1)
    headline_confidences: list[AnnotationConfidence] = Field(
        default_factory=lambda: [AnnotationConfidence.HIGH]
    )
    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    baseline_tolerances: BaselineTolerances = Field(default_factory=BaselineTolerances)
    failure_artifacts: bool = True
    failure_clip_padding_seconds: float = Field(default=2.0, ge=0)

    @model_validator(mode="after")
    def validate_confidences(self) -> BenchmarkSettings:
        if not self.headline_confidences:
            raise ValueError("headline_confidences must contain at least one value")
        if len(self.headline_confidences) != len(set(self.headline_confidences)):
            raise ValueError("headline_confidences must not contain duplicates")
        return self


class ManifestVideo(StrictModel):
    id: str = Field(min_length=1)
    path: str | None = None
    annotation: str | None = None
    additional_annotations: list[str] = Field(default_factory=list)
    config: str | None = None
    tags: list[str] = Field(default_factory=list)
    split: DatasetSplit = DatasetSplit.VALIDATION
    duration_seconds: float | None = Field(default=None, gt=0)
    enabled: bool = True

    @property
    def annotation_paths(self) -> tuple[str, ...]:
        primary = (self.annotation,) if self.annotation else ()
        return (*primary, *self.additional_annotations)

    @model_validator(mode="after")
    def validate_entry(self) -> ManifestVideo:
        if self.enabled and not self.annotation_paths:
            raise ValueError(f"enabled video {self.id!r} requires an annotation file")
        if len(self.annotation_paths) != len(set(self.annotation_paths)):
            raise ValueError(f"video {self.id!r} contains duplicate annotation paths")
        normalized_tags = [tag.strip().lower() for tag in self.tags]
        if any(not tag for tag in normalized_tags):
            raise ValueError("manifest tags must not be empty")
        if len(normalized_tags) != len(set(normalized_tags)):
            raise ValueError(f"video {self.id!r} contains duplicate tags")
        self.tags = normalized_tags
        return self


class BenchmarkManifest(StrictModel):
    schema_version: Literal["1.0"] = BENCHMARK_SCHEMA_VERSION
    name: str = Field(default="traffic_behavior_benchmark", min_length=1)
    synthetic: bool = False
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    videos: list[ManifestVideo]

    @model_validator(mode="after")
    def validate_videos(self) -> BenchmarkManifest:
        identifiers = [video.id for video in self.videos]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("manifest video IDs must be unique")
        return self


class RuntimePerformance(StrictModel):
    total_processing_time_seconds: float = Field(ge=0)
    video_duration_seconds: float = Field(ge=0)
    frames_processed: int = Field(ge=0)
    processing_fps: float = Field(ge=0)
    real_time_factor: float = Field(ge=0)
    average_frame_processing_time_ms: float = Field(ge=0)
    measurement_scope: str = "end_to_end_inference"
    hardware: dict[str, Any] = Field(default_factory=dict)


class VersionMetadata(StrictModel):
    git_commit: str | None = None
    benchmark_schema_version: str = BENCHMARK_SCHEMA_VERSION
    annotation_schema_versions: list[str] = Field(default_factory=list)
    policy_version: str | None = None
    detector_model_identifier: str | None = None
    tracker_identifier: str | None = None


class PredictedEvent(StrictModel):
    event_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    track_id: int | str | None = None
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    event_type: str = "left_lane_review_candidate"
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)
    geometry_status: str = "unavailable"
    overtaking_status: str = "unavailable"
    review_status: str = "pending_human_review"
    congestion_level: str | None = None
    right_lane_available: bool | None = None
    lifecycle_state: str | None = None
    diagnostic_hints: list[str] = Field(default_factory=list)
    source_metadata_path: str | None = None
    representative_frame_path: str | None = None
    event_video_clip_path: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> PredictedEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                "prediction end_seconds must be greater than start_seconds"
            )
        return self


class PredictionDocument(StrictModel):
    schema_version: Literal["1.0"] = PREDICTION_SCHEMA_VERSION
    video_id: str = Field(min_length=1)
    source_file: str | None = None
    predictions: list[PredictedEvent] = Field(default_factory=list)
    cancelled_event_count: int = Field(default=0, ge=0)
    performance: RuntimePerformance | None = None
    versions: VersionMetadata = Field(default_factory=VersionMetadata)

    @model_validator(mode="after")
    def validate_predictions(self) -> PredictionDocument:
        identifiers = [prediction.event_id for prediction in self.predictions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("prediction event IDs must be unique")
        wrong_video = [
            prediction.event_id
            for prediction in self.predictions
            if prediction.video_id != self.video_id
        ]
        if wrong_video:
            raise ValueError(
                "prediction video_id differs from document video_id: "
                + ", ".join(sorted(wrong_video))
            )
        return self


class EventMatch(StrictModel):
    video_id: str
    ground_truth_event_id: str
    predicted_event_id: str
    temporal_iou: float = Field(ge=0, le=1)
    start_time_error_seconds: float
    duration_error_seconds: float


class DurationEvidence(StrictModel):
    source: Literal["manifest", "annotation", "video_metadata", "prediction_cache"]
    seconds: float = Field(gt=0)
    confidence: Literal["high", "medium", "low"]


class DurationValidationResult(StrictModel):
    duration_seconds_used: float = Field(gt=0)
    duration_source: str
    duration_validation_status: Literal[
        "verified_video_metadata",
        "consistent_multiple_sources",
        "single_source_unverified",
        "aggregate",
    ]
    denominator_confidence: Literal["high", "medium", "low"]
    evidence: list[DurationEvidence]


class IgnoredPredictionDiagnostic(StrictModel):
    prediction_id: str
    matched_ignore_annotation_id: str
    prediction_coverage: float = Field(ge=0, le=1)
    temporal_iou: float = Field(ge=0, le=1)
    ignore_reason: str


class FilteredPredictionDiagnostic(StrictModel):
    prediction_id: str
    confidence: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    reason: str = "below_minimum_prediction_confidence"


class EvaluationAccounting(StrictModel):
    total_prediction_records: int = Field(ge=0)
    excluded_non_review_predictions: int = Field(ge=0)
    total_predictions_considered: int = Field(ge=0)
    matched_predictions: int = Field(ge=0)
    false_positive_predictions: int = Field(ge=0)
    ignored_predictions: int = Field(ge=0)
    filtered_low_confidence_predictions: int = Field(ge=0)
    total_positive_gt: int = Field(ge=0)
    matched_positive_gt: int = Field(ge=0)
    false_negative_gt: int = Field(ge=0)
    ignored_ground_truth_events: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_accounting(self) -> EvaluationAccounting:
        if self.total_predictions_considered != (
            self.matched_predictions
            + self.false_positive_predictions
            + self.ignored_predictions
        ):
            raise ValueError("considered prediction accounting does not reconcile")
        if self.total_prediction_records != (
            self.excluded_non_review_predictions
            + self.filtered_low_confidence_predictions
            + self.total_predictions_considered
        ):
            raise ValueError("total prediction accounting does not reconcile")
        if self.total_positive_gt != (
            self.matched_positive_gt
            + self.false_negative_gt
            + self.ignored_ground_truth_events
        ):
            raise ValueError("positive ground-truth accounting does not reconcile")
        return self


class MetricSummary(StrictModel):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    evaluated_video_hours: float = Field(ge=0)
    events_per_hour: float = Field(ge=0)
    false_positives_per_video_hour: float = Field(ge=0)
    false_negatives_per_video_hour: float = Field(ge=0)
    mean_start_time_error_seconds: float | None = None
    median_start_time_error_seconds: float | None = None
    mean_absolute_start_time_error_seconds: float | None = Field(default=None, ge=0)
    mean_duration_error_seconds: float | None = None
    mean_absolute_duration_error_seconds: float | None = Field(default=None, ge=0)
    duration_seconds_used: float = Field(default=0, ge=0)
    duration_source: str = "unavailable"
    duration_validation_status: str = "unavailable"
    denominator_confidence: str = "unavailable"


class FailureRecord(StrictModel):
    failure_id: str
    video_id: str
    kind: Literal["false_positive", "false_negative"]
    suspected_failure_category: str
    diagnostic_rationale: list[str] = Field(default_factory=list)
    timestamp_seconds: float = Field(ge=0)
    ground_truth: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    artifact_directory: str | None = None
    best_candidate_iou: float | None = Field(default=None, ge=0, le=1)
    best_candidate_prediction_coverage: float | None = Field(default=None, ge=0, le=1)
    matching_rejection_reason: str | None = None


class AnnotationAgreement(StrictModel):
    video_id: str
    annotator_a: str
    annotator_b: str
    matched_event_count: int = Field(ge=0)
    event_label_agreement: float = Field(ge=0, le=1)
    temporal_matching_agreement: float = Field(ge=0, le=1)
    mean_temporal_iou: float | None = Field(default=None, ge=0, le=1)
    cohen_kappa_matched_labels: float | None = None
    unmatched_events_a: int = Field(ge=0)
    unmatched_events_b: int = Field(ge=0)
    caveat: str

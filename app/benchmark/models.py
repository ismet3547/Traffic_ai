"""Versioned, detector-agnostic models for benchmark inputs and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANNOTATION_SCHEMA_VERSION = "1.0"
BENCHMARK_SCHEMA_VERSION = "1.0"
PREDICTION_SCHEMA_VERSION = "1.0"


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
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH
    notes: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> GroundTruthEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                "annotation end_seconds must be greater than start_seconds"
            )
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

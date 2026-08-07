"""Dataset release manifest, annotation quality gates, and benchmark export."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from app.benchmark.models import (
    AnnotationDocument,
    AnnotationLabel,
    DatasetSplit,
    GroundTruthEvent,
)
from app.dataset.io import document_sha256
from app.dataset.models import (
    AdjudicationArtifact,
    AdjudicationOutcome,
    AgreementReport,
    AnnotationQualityConfig,
    DatasetAnnotation,
    DatasetLabel,
    DatasetRelease,
    IntakeRegistry,
    QualityGateResult,
    ReleaseVideo,
    SplitAssignmentDocument,
    VideoIntakeRecord,
)

EXPORT_LABELS = {
    DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION: AnnotationLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
    DatasetLabel.LEGITIMATE_OVERTAKING: AnnotationLabel.LEGITIMATE_OVERTAKING,
    DatasetLabel.CONGESTION_LEFT_LANE_USE: AnnotationLabel.CONGESTION_LEFT_LANE_USE,
    DatasetLabel.TEMPORARY_LEFT_LANE_USE: AnnotationLabel.TEMPORARY_LEFT_LANE_USE,
    DatasetLabel.RIGHT_LANE_UNAVAILABLE: AnnotationLabel.RIGHT_LANE_UNAVAILABLE,
    DatasetLabel.INSUFFICIENT_EVIDENCE: AnnotationLabel.INSUFFICIENT_EVIDENCE,
    DatasetLabel.GEOMETRY_INVALID: AnnotationLabel.GEOMETRY_INVALID,
    DatasetLabel.CAMERA_MOTION_INVALID: AnnotationLabel.CAMERA_MOTION,
}


def build_dataset_release(
    registry: IntakeRegistry,
    split_assignments: SplitAssignmentDocument,
    annotations: dict[str, list[DatasetAnnotation]],
    adjudications: dict[str, AdjudicationArtifact],
    *,
    agreements: list[AgreementReport] | None = None,
    quality_config: AnnotationQualityConfig | None = None,
    created_at: datetime | None = None,
) -> DatasetRelease:
    split_by_video = {
        item.video_id: item.split for item in split_assignments.assignments
    }
    videos: list[ReleaseVideo] = []
    for record in sorted(registry.videos, key=lambda item: item.video_id):
        if record.video_id not in split_by_video:
            raise ValueError(f"missing split assignment for {record.video_id}")
        documents = annotations.get(record.video_id, [])
        adjudication = adjudications.get(record.video_id)
        ambiguous = bool(
            adjudication
            and any(
                decision.outcome == AdjudicationOutcome.REMAINS_AMBIGUOUS
                for decision in adjudication.decisions
            )
        )
        if adjudication and adjudication.approved:
            status: Literal["not_required", "pending", "approved", "ambiguous"] = (
                "ambiguous" if ambiguous else "approved"
            )
        elif split_by_video[record.video_id].value == "development":
            status = "not_required"
        else:
            status = "pending"
        videos.append(
            ReleaseVideo(
                video_id=record.video_id,
                source_group_id=record.source_group_id,
                split=split_by_video[record.video_id],
                source_video_sha256=record.source_video_sha256,
                source_video_size_bytes=record.source_video_size_bytes,
                source_identity_verified=record.source_identity_verified,
                duration_seconds=record.duration_seconds,
                annotation_hashes={
                    document.annotator_id: document_sha256(document)
                    for document in sorted(
                        documents, key=lambda item: item.annotator_id
                    )
                },
                adjudicated_annotation_hash=(
                    document_sha256(adjudication) if adjudication else None
                ),
                double_annotated=len({item.annotator_id for item in documents}) >= 2,
                adjudication_status=status,
                test_annotation_locked=bool(adjudication and adjudication.locked),
                license_or_permission_status=record.license_or_permission_status,
                redistribution_allowed=record.redistribution_allowed,
                benchmark_use_allowed=record.benchmark_use_allowed,
            )
        )
    gates = evaluate_quality_gates(
        videos,
        agreements or [],
        quality_config or AnnotationQualityConfig(),
    )
    return DatasetRelease(
        dataset_version=registry.dataset_version,
        created_at=created_at or datetime.now(timezone.utc),
        videos=videos,
        quality_gates=gates,
        quality_gate_passed=all(item.passed for item in gates),
        notes=[
            "No production accuracy requirement is implied by annotation quality gates.",
            "Near-duplicate edited/cropped clips require manual source-group review.",
        ],
    )


def evaluate_quality_gates(
    videos: list[ReleaseVideo],
    agreements: list[AgreementReport],
    config: AnnotationQualityConfig,
) -> list[QualityGateResult]:
    validation_test = [
        item for item in videos if item.split.value in {"validation", "test"}
    ]
    test_videos = [item for item in videos if item.split.value == "test"]
    gates = [
        _gate(
            "artifact_schemas_validated",
            True,
            "release inputs passed strict Pydantic schema and protocol loading",
        ),
        _gate("dataset_has_registered_clips", bool(videos), f"clips={len(videos)}"),
        _gate(
            "dataset_provenance_recorded",
            all(item.benchmark_use_allowed for item in videos),
            "every clip must explicitly allow benchmark use",
        ),
        _gate(
            "source_video_identities_verified",
            bool(videos) and all(item.source_identity_verified for item in videos),
            "all source identities must be verified SHA-256 records",
        ),
        _gate(
            "validation_test_double_annotated",
            all(item.double_annotated for item in validation_test),
            f"validation/test clips={len(validation_test)}",
        ),
        _gate(
            "validation_test_disagreements_adjudicated",
            all(
                item.adjudication_status in {"approved", "ambiguous"}
                for item in validation_test
            ),
            "every validation/test clip requires reviewed adjudication",
        ),
        _gate(
            "test_ground_truth_adjudicated",
            all(
                item.adjudication_status in {"approved", "ambiguous"}
                for item in test_videos
            ),
            f"test clips={len(test_videos)}",
        ),
        _gate(
            "test_annotations_locked",
            all(item.test_annotation_locked for item in test_videos),
            "normal tooling must not edit locked test ground truth",
        ),
    ]
    if config.minimum_label_agreement is not None:
        actual = (
            sum(item.label_agreement for item in agreements) / len(agreements)
            if agreements
            else 0.0
        )
        gates.append(
            _gate(
                "minimum_label_agreement",
                actual + 1e-12 >= config.minimum_label_agreement,
                f"actual={actual:.6f}, threshold={config.minimum_label_agreement:.6f}",
            )
        )
    if config.minimum_event_match_rate is not None:
        actual = (
            sum(item.event_detection_agreement for item in agreements) / len(agreements)
            if agreements
            else 0.0
        )
        gates.append(
            _gate(
                "minimum_event_match_rate",
                actual + 1e-12 >= config.minimum_event_match_rate,
                f"actual={actual:.6f}, threshold={config.minimum_event_match_rate:.6f}",
            )
        )
    return gates


def export_adjudicated_annotation(
    artifact: AdjudicationArtifact,
    intake_record: VideoIntakeRecord,
    *,
    split: DatasetSplit,
) -> AnnotationDocument:
    if not artifact.approved:
        raise ValueError("only approved adjudication may enter benchmark ground truth")
    if split.value == "test" and not artifact.locked:
        raise ValueError("test adjudication must be locked before benchmark export")
    if artifact.video_id != intake_record.video_id:
        raise ValueError("adjudication and intake video IDs differ")
    events = [
        GroundTruthEvent(
            event_id=event.event_id,
            vehicle_track_hint=event.vehicle_ref,
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            label=EXPORT_LABELS[event.label],
            confidence=event.confidence,
            notes=event.notes,
        )
        for event in artifact.final_events
    ]
    return AnnotationDocument(
        video_id=artifact.video_id,
        source_file=intake_record.original_filename,
        fps=intake_record.fps,
        video_duration_seconds=intake_record.duration_seconds,
        annotator_id="adjudicated",
        events=events,
    )


def _gate(name: str, passed: bool, details: str) -> QualityGateResult:
    return QualityGateResult(gate=name, passed=passed, details=details)

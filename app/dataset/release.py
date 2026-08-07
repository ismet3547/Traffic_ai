"""Fail-closed dataset release construction and benchmark export."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from app.benchmark.models import (
    AnnotationDocument,
    AnnotationLabel,
    DatasetSplit,
    GroundTruthEvent,
)
from app.dataset.integrity import (
    DatasetIntegrityError,
    raise_for_release_integrity,
    validate_adjudication_source_identity,
    validate_double_annotation,
    validate_release_integrity,
)
from app.dataset.io import document_sha256
from app.dataset.models import (
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    AdjudicationArtifact,
    AdjudicationOutcome,
    AgreementReport,
    AnnotationQualityConfig,
    DatasetAnnotation,
    DatasetLabel,
    DatasetRelease,
    DatasetReleaseIntegrityReport,
    IntakeRegistry,
    IntegrityIssue,
    IntegrityReasonCode,
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
    """Validate the complete trust boundary before constructing a release."""
    integrity = validate_release_integrity(
        registry, split_assignments, annotations, adjudications
    )
    raise_for_release_integrity(integrity)
    split_by_video = {
        item.video_id: item.split for item in split_assignments.assignments
    }
    videos: list[ReleaseVideo] = []
    for record in sorted(registry.videos, key=lambda item: item.video_id):
        split = split_by_video[record.video_id]
        documents = annotations.get(record.video_id, [])
        double = validate_double_annotation(record, documents)
        adjudication = adjudications.get(record.video_id)
        ambiguous = bool(
            adjudication
            and any(
                decision.outcome == AdjudicationOutcome.REMAINS_AMBIGUOUS
                for decision in adjudication.decisions
            )
        )
        if adjudication is not None:
            status: Literal["not_required", "pending", "approved", "ambiguous"] = (
                "ambiguous" if ambiguous else "approved"
            )
            benchmark_document = export_adjudicated_annotation(
                adjudication,
                record,
                split=split,
                source_annotations=documents,
            )
            ground_truth_hash = document_sha256(benchmark_document)
        else:
            status = "not_required"
            ground_truth_hash = None
        videos.append(
            ReleaseVideo(
                video_id=record.video_id,
                source_group_id=record.source_group_id,
                split=split,
                source_video_sha256=record.source_video_sha256,
                source_video_size_bytes=record.source_video_size_bytes,
                source_identity_verified=record.source_identity_verified,
                duration_seconds=record.duration_seconds,
                ontology_version=ONTOLOGY_VERSION,
                handbook_version=HANDBOOK_VERSION,
                annotation_hashes=double.annotation_hashes,
                adjudicated_annotation_hash=(
                    document_sha256(adjudication) if adjudication is not None else None
                ),
                benchmark_ground_truth_sha256=ground_truth_hash,
                double_annotated=double.valid,
                double_annotation=double,
                adjudication_status=status,
                test_annotation_locked=bool(
                    adjudication is not None and adjudication.locked
                ),
                license_or_permission_status=record.license_or_permission_status,
                redistribution_allowed=record.redistribution_allowed,
                benchmark_use_allowed=record.benchmark_use_allowed,
            )
        )
    threshold_gates = evaluate_agreement_thresholds(
        agreements or [], quality_config or AnnotationQualityConfig()
    )
    gates = [*integrity.gates, *threshold_gates]
    return DatasetRelease(
        dataset_version=registry.dataset_version,
        created_at=created_at or datetime.now(timezone.utc),
        videos=videos,
        integrity_report=integrity,
        quality_gates=gates,
        quality_gate_passed=all(item.passed for item in gates),
        notes=[
            "No production accuracy requirement is implied by annotation quality gates.",
            "Near-duplicate edited/cropped clips require manual source-group review.",
            "Release integrity was revalidated independently of intake, annotation, adjudication, and split tooling.",
        ],
    )


def evaluate_agreement_thresholds(
    agreements: list[AgreementReport],
    config: AnnotationQualityConfig,
) -> list[QualityGateResult]:
    gates: list[QualityGateResult] = []
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
    source_annotations: list[DatasetAnnotation] | None = None,
    expected_ground_truth_sha256: str | None = None,
) -> AnnotationDocument:
    """Export only after revalidating source identity and optional revision/hash ties."""
    validate_adjudication_source_identity(intake_record, artifact, source_annotations)
    if not artifact.approved:
        raise ValueError("only approved adjudication may enter benchmark ground truth")
    if split == DatasetSplit.TEST and not artifact.locked:
        raise ValueError("test adjudication must be locked before benchmark export")
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
    document = AnnotationDocument(
        video_id=artifact.video_id,
        source_file=intake_record.original_filename,
        fps=intake_record.fps,
        video_duration_seconds=intake_record.duration_seconds,
        annotator_id="adjudicated",
        events=events,
    )
    actual_hash = document_sha256(document)
    if (
        expected_ground_truth_sha256 is not None
        and actual_hash != expected_ground_truth_sha256
    ):
        issue = IntegrityIssue(
            reason_code=IntegrityReasonCode.FINAL_GROUND_TRUTH_HASH_MISMATCH,
            details="exported benchmark ground truth differs from release manifest",
            video_id=intake_record.video_id,
            source_group_id=intake_record.source_group_id,
        )
        raise DatasetIntegrityError(
            DatasetReleaseIntegrityReport(
                passed=False,
                gates=[
                    QualityGateResult(
                        gate="final_ground_truth_hash_valid",
                        passed=False,
                        details=issue.reason_code.value,
                    )
                ],
                reason_codes=[issue.reason_code],
                affected_video_ids=[intake_record.video_id],
                affected_source_group_ids=[intake_record.source_group_id],
                issues=[issue],
            )
        )
    return document


def _gate(name: str, passed: bool, details: str) -> QualityGateResult:
    return QualityGateResult(gate=name, passed=passed, details=details)

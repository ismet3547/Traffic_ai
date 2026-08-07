from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.agreement import compare_independent_annotations
from app.dataset.agreement_integrity import (
    AgreementIntegrityError,
    assess_agreement_report,
    validate_agreement_report,
    validate_release_agreements,
)
from app.dataset.integrity import DatasetIntegrityError, StaleAdjudicationError
from app.dataset.io import (
    agreement_report_content_hash,
    document_sha256,
    lock_annotation,
    read_json_model,
    write_json_model,
)
from app.dataset.models import (
    AGREEMENT_PROTOCOL_VERSION,
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    AdjudicationDecision,
    AdjudicationOutcome,
    AgreementReport,
    AnnotationQualityConfig,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    DatasetRelease,
    IntakeRegistry,
    IntegrityReasonCode,
    PermissionStatus,
    SourceType,
    SplitAssignment,
    SplitAssignmentDocument,
    VideoIntakeRecord,
    VideoResolution,
)
from app.dataset.release import build_dataset_release
from app.dataset.reporting import build_coverage_report
from app.tools.build_dataset_release import main as build_release_main

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _record(video_id: str, index: int) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id=video_id,
        source_group_id=f"group_{video_id}",
        source_type=SourceType.OWN_CAPTURE,
        source_reference="synthetic agreement provenance test",
        acquisition_date=date(2026, 3, 1),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
        source_video_sha256=_sha(index),
        source_video_size_bytes=1000 + index,
        source_identity_verified=True,
        duration_seconds=60,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename=f"{video_id}.mp4",
    )


def _events(prefix: str, *, count: int, agreeing: int) -> list[DatasetEvent]:
    return [
        DatasetEvent(
            event_id=f"{prefix}_{number}",
            vehicle_ref=f"vehicle_{number}",
            start_seconds=1.0 + number * 3,
            end_seconds=2.0 + number * 3,
            label=(
                DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION
                if prefix == "a" or number < agreeing
                else DatasetLabel.LEGITIMATE_OVERTAKING
            ),
            confidence=AnnotationConfidence.HIGH,
        )
        for number in range(count)
    ]


def _annotation(
    record: VideoIntakeRecord,
    annotator_id: str,
    *,
    count: int = 1,
    agreeing: int = 1,
) -> DatasetAnnotation:
    prefix = "a" if annotator_id.endswith("a") else "b"
    return lock_annotation(
        DatasetAnnotation(
            video_id=record.video_id,
            source_video_sha256=record.source_video_sha256,
            source_video_size_bytes=record.source_video_size_bytes,
            source_file=record.original_filename,
            fps=record.fps,
            video_duration_seconds=record.duration_seconds,
            annotator_id=annotator_id,
            created_at=NOW,
            events=_events(prefix, count=count, agreeing=agreeing),
        ),
        locked_at=NOW,
    )


def _adjudication(
    first: DatasetAnnotation,
    second: DatasetAnnotation,
):
    report = compare_independent_annotations(first, second)
    first_by_id = {item.event_id: item for item in first.events}
    second_by_id = {item.event_id: item for item in second.events}
    decisions: list[AdjudicationDecision] = []
    for number, disagreement in enumerate(report.disagreements, start=1):
        if disagreement.event_id_a is not None:
            event = first_by_id[disagreement.event_id_a]
            outcome = AdjudicationOutcome.RESOLVED_TO_A
        else:
            assert disagreement.event_id_b is not None
            event = second_by_id[disagreement.event_id_b]
            outcome = AdjudicationOutcome.RESOLVED_TO_B
        decisions.append(
            AdjudicationDecision(
                decision_id=f"decision_{number}",
                disagreement_ids=[disagreement.disagreement_id],
                event_ids_a=(
                    [disagreement.event_id_a]
                    if disagreement.event_id_a is not None
                    else []
                ),
                event_ids_b=(
                    [disagreement.event_id_b]
                    if disagreement.event_id_b is not None
                    else []
                ),
                outcome=outcome,
                adjudicated_event=event,
                rationale="synthetic adjudicator selected the supported event",
                adjudication_confidence=event.confidence,
            )
        )
    return lock_adjudication(
        create_adjudication(
            first,
            second,
            adjudicator_id="adjudicator",
            decisions=decisions,
            created_at=NOW,
        ),
        locked_at=NOW,
    )


def _bundle(
    video_id: str = "clip",
    index: int = 1,
    *,
    count: int = 1,
    agreeing: int = 1,
):
    record = _record(video_id, index)
    first = _annotation(record, "annotator_a", count=count, agreeing=count)
    second = _annotation(record, "annotator_b", count=count, agreeing=agreeing)
    report = compare_independent_annotations(first, second)
    artifact = _adjudication(first, second)
    return record, first, second, report, artifact


def _splits(
    records: list[VideoIntakeRecord], split: DatasetSplit = DatasetSplit.VALIDATION
) -> SplitAssignmentDocument:
    return SplitAssignmentDocument(
        seed=42,
        target_ratios={
            DatasetSplit.DEVELOPMENT: (1 if split == DatasetSplit.DEVELOPMENT else 0),
            DatasetSplit.VALIDATION: 1 if split == DatasetSplit.VALIDATION else 0,
            DatasetSplit.TEST: 1 if split == DatasetSplit.TEST else 0,
        },
        assignments=[
            SplitAssignment(
                video_id=item.video_id,
                source_group_id=item.source_group_id,
                split=split,
            )
            for item in records
        ],
    )


def _rehash(report: AgreementReport, **updates: object) -> AgreementReport:
    candidate = report.model_copy(update=updates)
    return candidate.model_copy(
        update={"agreement_content_sha256": agreement_report_content_hash(candidate)}
    )


def _revise(annotation: DatasetAnnotation) -> DatasetAnnotation:
    revised_event = annotation.events[0].model_copy(
        update={"end_seconds": annotation.events[0].end_seconds + 0.5}
    )
    unlocked = annotation.model_copy(
        update={
            "events": [revised_event, *annotation.events[1:]],
            "locked": False,
            "locked_at": None,
            "annotation_hash": None,
        }
    )
    return lock_annotation(unlocked, locked_at=NOW + timedelta(seconds=1))


def _release_inputs(bundles):
    records = [item[0] for item in bundles]
    annotations = {item[0].video_id: [item[1], item[2]] for item in bundles}
    adjudications = {item[0].video_id: item[4] for item in bundles}
    reports = [item[3] for item in bundles]
    return (
        IntakeRegistry(videos=records),
        _splits(records),
        annotations,
        adjudications,
        reports,
    )


def test_valid_current_agreement_report_is_accepted() -> None:
    record, first, second, report, artifact = _bundle()
    result = validate_agreement_report(
        record, first, second, report, adjudication=artifact
    )
    assert result.valid


def test_agreement_creation_hashes_exact_current_inputs() -> None:
    _, first, second, report, _ = _bundle()
    assert report.annotation_a_content_sha256 == document_sha256(first)
    assert report.annotation_b_content_sha256 == document_sha256(second)
    assert report.source_video_sha256 == first.source_video_sha256


def test_source_sha_mismatch_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(report, source_video_sha256=_sha(999))
    with pytest.raises(AgreementIntegrityError) as raised:
        validate_agreement_report(record, first, second, wrong)
    assert (
        IntegrityReasonCode.AGREEMENT_SOURCE_VIDEO_MISMATCH
        in raised.value.result.reason_codes
    )


def test_source_size_mismatch_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(report, source_video_size_bytes=999999)
    result = assess_agreement_report(record, first, second, wrong)
    assert IntegrityReasonCode.AGREEMENT_SOURCE_SIZE_MISMATCH in result.reason_codes


@pytest.mark.parametrize("revision", ["a", "b"])
def test_stale_annotation_revision_is_rejected(revision: str) -> None:
    record, first, second, report, _ = _bundle()
    if revision == "a":
        first = _revise(first)
    else:
        second = _revise(second)
    result = assess_agreement_report(record, first, second, report)
    assert IntegrityReasonCode.STALE_AGREEMENT_REPORT in result.reason_codes


def test_unknown_video_agreement_is_rejected() -> None:
    known = _bundle("known", 1)
    unknown = _bundle("unknown", 2)
    registry, split_doc, annotations, adjudications, reports = _release_inputs([known])
    validation = validate_release_agreements(
        registry,
        split_doc,
        annotations,
        adjudications,
        [*reports, unknown[3]],
    )
    assert IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO in {
        item.reason_code for item in validation.issues
    }
    assert validation.coverage.unknown_report_count == 1


def test_duplicate_same_logical_report_is_rejected() -> None:
    bundle = _bundle()
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    result = validate_release_agreements(
        registry, split_doc, annotations, adjudications, reports * 2
    )
    assert IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT in {
        item.reason_code for item in result.issues
    }
    assert result.coverage.duplicate_report_count == 1


def test_reversed_pair_is_detected_as_duplicate() -> None:
    bundle = _bundle()
    reversed_report = compare_independent_annotations(bundle[2], bundle[1])
    assert reversed_report.agreement_id == bundle[3].agreement_id
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    result = validate_release_agreements(
        registry,
        split_doc,
        annotations,
        adjudications,
        [*reports, reversed_report],
    )
    assert IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT in {
        item.reason_code for item in result.issues
    }


def test_unsupported_agreement_protocol_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(report, agreement_protocol_version="999")
    result = assess_agreement_report(record, first, second, wrong)
    assert IntegrityReasonCode.AGREEMENT_PROTOCOL_MISMATCH in result.reason_codes
    assert report.agreement_protocol_version == AGREEMENT_PROTOCOL_VERSION


@pytest.mark.parametrize("split", [DatasetSplit.VALIDATION, DatasetSplit.TEST])
def test_missing_required_agreement_fails_release(split: DatasetSplit) -> None:
    bundle = _bundle()
    record, first, second, _, artifact = bundle
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[record]),
            _splits([record], split),
            {record.video_id: [first, second]},
            {record.video_id: artifact},
            agreements=[],
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.MISSING_CURRENT_AGREEMENT_REPORT
        in raised.value.report.reason_codes
    )


def test_development_release_does_not_require_agreement() -> None:
    record = _record("development", 3)
    first = _annotation(record, "annotator_a")
    split_doc = _splits([record], DatasetSplit.DEVELOPMENT)
    release = build_dataset_release(
        IntakeRegistry(videos=[record]),
        split_doc,
        {record.video_id: [first]},
        {},
        agreements=[],
        created_at=NOW,
    )
    assert release.agreement_coverage.required_video_count == 0
    assert release.agreement_coverage.coverage_ratio == 1.0


def test_invalid_development_agreement_is_still_rejected() -> None:
    bundle = _bundle("development", 4)
    record, first, second, report, _ = bundle
    wrong = _rehash(report, source_video_sha256=_sha(999))
    with pytest.raises(DatasetIntegrityError):
        build_dataset_release(
            IntakeRegistry(videos=[record]),
            _splits([record], DatasetSplit.DEVELOPMENT),
            {record.video_id: [first, second]},
            {},
            agreements=[wrong],
            created_at=NOW,
        )


def test_report_ontology_mismatch_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(report, annotation_a_ontology_version="old-ontology")
    result = assess_agreement_report(record, first, second, wrong)
    assert IntegrityReasonCode.AGREEMENT_ONTOLOGY_MISMATCH in result.reason_codes
    assert report.annotation_b_ontology_version == ONTOLOGY_VERSION


def test_report_handbook_mismatch_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(report, annotation_b_handbook_version="old-handbook")
    result = assess_agreement_report(record, first, second, wrong)
    assert IntegrityReasonCode.AGREEMENT_HANDBOOK_MISMATCH in result.reason_codes
    assert report.annotation_a_handbook_version == HANDBOOK_VERSION


def test_agreement_and_adjudication_revision_chain_matches() -> None:
    record, first, second, report, artifact = _bundle()
    result = assess_agreement_report(
        record, first, second, report, adjudication=artifact
    )
    assert result.valid


def test_agreement_and_adjudication_revision_mismatch_is_rejected() -> None:
    record, first, second, _, old_artifact = _bundle()
    revised_second = _revise(second)
    current_report = compare_independent_annotations(first, revised_second)
    result = assess_agreement_report(
        record,
        first,
        revised_second,
        current_report,
        adjudication=old_artifact,
    )
    assert (
        IntegrityReasonCode.AGREEMENT_ADJUDICATION_REVISION_MISMATCH
        in result.reason_codes
    )


def test_internally_forged_score_is_recomputed_and_rejected() -> None:
    record, first, second, report, _ = _bundle(count=1, agreeing=0)
    forged = _rehash(report, label_agreement=1.0)
    result = assess_agreement_report(record, first, second, forged)
    assert IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT in result.reason_codes


def test_unrelated_perfect_reports_cannot_improve_release_score() -> None:
    hard = [_bundle(f"hard_{i:02d}", i, agreeing=0) for i in range(1, 6)]
    easy = [_bundle(f"easy_{i:02d}", i + 20) for i in range(1, 21)]
    registry, split_doc, annotations, adjudications, reports = _release_inputs(hard)
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            registry,
            split_doc,
            annotations,
            adjudications,
            agreements=[*reports, *(item[3] for item in easy)],
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO
        in raised.value.report.reason_codes
    )


def test_duplicated_perfect_report_cannot_overweight_score() -> None:
    bundle = _bundle()
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            registry,
            split_doc,
            annotations,
            adjudications,
            agreements=reports * 20,
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT
        in raised.value.report.reason_codes
    )


def test_macro_quality_uses_exactly_one_validated_report_per_video() -> None:
    bundles = [
        _bundle("hard_01", 31, count=2, agreeing=1),
        _bundle("hard_02", 32, count=5, agreeing=3),
    ]
    registry, split_doc, annotations, adjudications, reports = _release_inputs(bundles)
    release = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        quality_config=AnnotationQualityConfig(minimum_label_agreement=0.55),
        created_at=NOW,
    )
    assert reports[0].label_agreement == 0.5
    assert reports[1].label_agreement == 0.6
    assert release.agreement_quality.aggregation_mode == "macro_per_video"
    assert release.agreement_quality.label_agreement == pytest.approx(0.55)
    assert release.agreement_coverage.validated_report_count == 2
    assert release.quality_gate_passed


def test_hard_release_rejects_three_injected_easy_reports() -> None:
    hard = [
        _bundle("hard_01", 41, count=2, agreeing=1),
        _bundle("hard_02", 42, count=5, agreeing=3),
    ]
    easy = [_bundle(f"easy_{index:02d}", 50 + index) for index in range(1, 4)]
    registry, split_doc, annotations, adjudications, reports = _release_inputs(hard)
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            registry,
            split_doc,
            annotations,
            adjudications,
            agreements=[*reports, *(item[3] for item in easy)],
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO
        in raised.value.report.reason_codes
    )


def test_zero_event_video_semantics_are_deterministic() -> None:
    first = _bundle("zero", 60, count=0, agreeing=0)
    report = first[3]
    assert report.event_detection_agreement == 1.0
    assert report.label_agreement == 0.0
    assert report.temporal_boundary_agreement == 0.0
    assert report.confidence_agreement == 0.0
    assert report.mean_temporal_iou is None
    assert compare_independent_annotations(first[1], first[2]) == report


def test_release_output_contains_complete_agreement_provenance_chain() -> None:
    bundle = _bundle()
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    release = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        created_at=NOW,
    )
    item = release.videos[0]
    assert item.agreement_provenance is not None
    assert item.agreement_provenance.agreement_id == reports[0].agreement_id
    assert item.agreement_provenance.annotation_content_sha256 == sorted(
        [document_sha256(bundle[1]), document_sha256(bundle[2])]
    )
    assert item.adjudicated_annotation_hash == document_sha256(bundle[4])
    assert item.benchmark_ground_truth_sha256 is not None


def test_stale_agreement_and_adjudication_are_both_release_blockers() -> None:
    record, first, second, old_report, old_artifact = _bundle()
    revised_second = _revise(second)
    with pytest.raises(StaleAdjudicationError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[record]),
            _splits([record]),
            {record.video_id: [first, revised_second]},
            {record.video_id: old_artifact},
            agreements=[old_report],
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.STALE_AGREEMENT_REPORT in raised.value.report.reason_codes
    )
    assert (
        IntegrityReasonCode.ADJUDICATION_STALE_SOURCE_ANNOTATION
        in raised.value.report.reason_codes
    )


def test_release_with_bound_agreement_is_deterministic() -> None:
    bundle = _bundle()
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    first = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        created_at=NOW,
    )
    second = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        created_at=NOW,
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_coverage_report_rejects_unvalidated_agreement_injection() -> None:
    known = _bundle("known", 70)
    unknown = _bundle("unknown", 71)
    registry, _, annotations, _, _ = _release_inputs([known])
    with pytest.raises(AgreementIntegrityError):
        build_coverage_report(registry, annotations, [unknown[3]])


def test_release_cli_loads_and_validates_agreement_directory(tmp_path: Path) -> None:
    bundle = _bundle("cli_clip", 72)
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    registry_path = write_json_model(registry, tmp_path / "registry.json")
    splits_path = write_json_model(split_doc, tmp_path / "splits.json")
    annotations_dir = tmp_path / "annotations"
    adjudications_dir = tmp_path / "adjudications"
    agreements_dir = tmp_path / "agreements"
    for number, document in enumerate(annotations["cli_clip"]):
        write_json_model(document, annotations_dir / f"annotation_{number}.json")
    write_json_model(adjudications["cli_clip"], adjudications_dir / "adjudication.json")
    write_json_model(reports[0], agreements_dir / "agreement.json")
    output = tmp_path / "dataset_release.json"
    result = build_release_main(
        [
            "--registry",
            str(registry_path),
            "--splits",
            str(splits_path),
            "--annotations-dir",
            str(annotations_dir),
            "--adjudications-dir",
            str(adjudications_dir),
            "--agreements-dir",
            str(agreements_dir),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    release = read_json_model(output, DatasetRelease)
    assert release.agreement_coverage.validated_report_count == 1

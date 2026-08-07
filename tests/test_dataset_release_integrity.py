from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.integrity import (
    DatasetIntegrityError,
    SourceIdentityMismatchError,
    SplitLeakageError,
    StaleAdjudicationError,
    validate_adjudication_source_identity,
    validate_annotation_source_identity,
    validate_double_annotation,
    validate_release_integrity,
)
from app.dataset.io import (
    document_sha256,
    lock_annotation,
    save_annotation,
    write_json_model,
)
from app.dataset.models import (
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    IntakeRegistry,
    IntegrityReasonCode,
    PermissionStatus,
    SourceType,
    SplitAssignment,
    SplitAssignmentDocument,
    VideoIntakeRecord,
    VideoResolution,
)
from app.dataset.release import build_dataset_release, export_adjudicated_annotation
from app.tools.build_dataset_release import main as build_release_main
from app.tools.export_adjudicated_benchmark import main as export_benchmark_main

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)
AAA = "a" * 64
BBB = "b" * 64
XYZ = "c" * 64


def intake(
    video_id: str = "clip",
    *,
    sha: str = AAA,
    group: str = "session_001",
    verified: bool = True,
) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id=video_id,
        source_group_id=group,
        source_type=SourceType.OWN_CAPTURE,
        source_reference="synthetic integrity test",
        acquisition_date=date(2026, 2, 1),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
        source_video_sha256=sha,
        source_video_size_bytes=100,
        source_identity_verified=verified,
        duration_seconds=60,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename=f"{video_id}.mp4",
    )


def annotation(
    annotator_id: str,
    event_id: str,
    *,
    video_id: str = "clip",
    sha: str = AAA,
    size: int | None = 100,
    ontology: str = ONTOLOGY_VERSION,
    handbook: str = HANDBOOK_VERSION,
    locked: bool = True,
) -> DatasetAnnotation:
    document = DatasetAnnotation(
        video_id=video_id,
        source_video_sha256=sha,
        source_video_size_bytes=size,
        source_file=f"{video_id}.mp4",
        fps=30,
        video_duration_seconds=60,
        annotator_id=annotator_id,
        ontology_version=ontology,
        handbook_version=handbook,
        created_at=NOW,
        events=[
            DatasetEvent(
                event_id=event_id,
                vehicle_ref="vehicle_1",
                start_seconds=10,
                end_seconds=20,
                label=DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
                confidence=AnnotationConfidence.HIGH,
            )
        ],
    )
    return lock_annotation(document, locked_at=NOW) if locked else document


def adjudication(
    first: DatasetAnnotation, second: DatasetAnnotation, *, locked: bool = True
):
    artifact = create_adjudication(
        first,
        second,
        adjudicator_id="adjudicator",
        decisions=[],
        created_at=NOW,
    )
    return lock_adjudication(artifact, locked_at=NOW) if locked else artifact


def splits(*assignments: tuple[str, str, DatasetSplit]) -> SplitAssignmentDocument:
    return SplitAssignmentDocument(
        seed=42,
        target_ratios={
            DatasetSplit.DEVELOPMENT: 0.5,
            DatasetSplit.VALIDATION: 0.25,
            DatasetSplit.TEST: 0.25,
        },
        assignments=[
            SplitAssignment(video_id=video, source_group_id=group, split=split)
            for video, group, split in assignments
        ],
    )


def valid_bundle():
    record = intake()
    first = annotation("annotator_a", "event_a")
    second = annotation("annotator_b", "event_b")
    artifact = adjudication(first, second)
    split_document = splits(("clip", "session_001", DatasetSplit.TEST))
    return record, first, second, artifact, split_document


def test_valid_annotation_matches_registry_sha() -> None:
    result = validate_annotation_source_identity(intake(), annotation("a", "a1"))
    assert result.valid


def test_annotation_sha_mismatch_is_rejected() -> None:
    with pytest.raises(SourceIdentityMismatchError) as raised:
        validate_annotation_source_identity(intake(), annotation("a", "a1", sha=BBB))
    assert (
        IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH
        in raised.value.report.reason_codes
    )


def test_annotation_source_size_mismatch_is_rejected() -> None:
    with pytest.raises(SourceIdentityMismatchError) as raised:
        validate_annotation_source_identity(intake(), annotation("a", "a1", size=999))
    assert (
        IntegrityReasonCode.ANNOTATION_SOURCE_SIZE_MISMATCH
        in raised.value.report.reason_codes
    )


def test_adversarial_perfect_agreement_on_wrong_source_still_fails_release() -> None:
    """Perfect human agreement cannot hide a registry/source-byte mismatch."""
    first = annotation("a", "a1", sha=BBB)
    second = annotation("b", "b1", sha=BBB)
    artifact = adjudication(first, second)
    with pytest.raises(SourceIdentityMismatchError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[intake()]),
            splits(("clip", "session_001", DatasetSplit.TEST)),
            {"clip": [first, second]},
            {"clip": artifact},
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH
        in raised.value.report.reason_codes
    )


def test_adjudication_source_sha_mismatch_is_rejected() -> None:
    record, first, second, artifact, _ = valid_bundle()
    tampered = artifact.model_copy(update={"source_video_sha256": BBB})
    with pytest.raises(SourceIdentityMismatchError) as raised:
        validate_adjudication_source_identity(record, tampered, [first, second])
    assert (
        IntegrityReasonCode.ADJUDICATION_SOURCE_VIDEO_MISMATCH
        in raised.value.report.reason_codes
    )


def test_stale_adjudication_after_audited_annotation_revision_is_rejected(
    tmp_path: Path,
) -> None:
    record, first, second, artifact, split_document = valid_bundle()
    annotation_path = tmp_path / "annotator_b.json"
    save_annotation(second, annotation_path)
    edited = second.model_copy(
        update={"events": [second.events[0].model_copy(update={"end_seconds": 21.0})]}
    )
    revised = save_annotation(
        edited,
        annotation_path,
        override_lock=True,
        override_reason="adjudicator-approved boundary correction",
        timestamp=NOW,
    )
    assert revised.lock_override_history[-1].action == "override_edit"
    with pytest.raises(StaleAdjudicationError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[record]),
            split_document,
            {"clip": [first, revised]},
            {"clip": artifact},
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.ADJUDICATION_STALE_SOURCE_ANNOTATION
        in raised.value.report.reason_codes
    )


def test_export_wrong_source_adjudication_is_rejected() -> None:
    _, first, second, artifact, _ = valid_bundle()
    wrong_registry = intake(sha=BBB)
    with pytest.raises(SourceIdentityMismatchError):
        export_adjudicated_annotation(
            artifact,
            wrong_registry,
            split=DatasetSplit.TEST,
            source_annotations=[first, second],
        )


def test_export_wrong_video_adjudication_is_rejected() -> None:
    _, first, second, artifact, _ = valid_bundle()
    wrong_video = intake(video_id="other_clip", sha=AAA)
    with pytest.raises(SourceIdentityMismatchError) as raised:
        export_adjudicated_annotation(
            artifact,
            wrong_video,
            split=DatasetSplit.TEST,
            source_annotations=[first, second],
        )
    assert (
        IntegrityReasonCode.ADJUDICATION_VIDEO_ID_MISMATCH
        in raised.value.report.reason_codes
    )


def test_export_cli_wrong_source_fails_without_writing_output(
    tmp_path: Path,
) -> None:
    record, first, second, artifact, split_document = valid_bundle()
    release = build_dataset_release(
        IntakeRegistry(videos=[record]),
        split_document,
        {"clip": [first, second]},
        {"clip": artifact},
        agreements=[artifact.agreement_report],
        created_at=NOW,
    )
    wrong_a = annotation("annotator_a", "event_a", sha=BBB)
    wrong_b = annotation("annotator_b", "event_b", sha=BBB)
    wrong_artifact = adjudication(wrong_a, wrong_b)
    registry_path = write_json_model(
        IntakeRegistry(videos=[record]), tmp_path / "registry.json"
    )
    splits_path = write_json_model(split_document, tmp_path / "splits.json")
    release_path = write_json_model(release, tmp_path / "release.json")
    adjudication_path = write_json_model(wrong_artifact, tmp_path / "adjudication.json")
    annotations_directory = tmp_path / "annotations"
    write_json_model(wrong_a, annotations_directory / "a.json")
    write_json_model(wrong_b, annotations_directory / "b.json")
    output = tmp_path / "benchmark.json"
    with pytest.raises(SourceIdentityMismatchError):
        export_benchmark_main(
            [
                "--adjudication",
                str(adjudication_path),
                "--annotations-dir",
                str(annotations_directory),
                "--registry",
                str(registry_path),
                "--splits",
                str(splits_path),
                "--release",
                str(release_path),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_two_valid_locked_annotators_form_valid_double_annotation() -> None:
    result = validate_double_annotation(
        intake(), [annotation("a", "a1"), annotation("b", "b1")]
    )
    assert result.valid
    assert result.annotator_count == 2
    assert result.source_identity_valid
    assert result.protocol_versions_compatible
    assert result.locked


def test_two_annotator_ids_with_one_source_mismatch_are_not_valid_double() -> None:
    result = validate_double_annotation(
        intake(), [annotation("a", "a1"), annotation("b", "b1", sha=BBB)]
    )
    assert not result.valid
    assert not result.source_identity_valid


def test_incompatible_ontology_versions_are_not_valid_double() -> None:
    result = validate_double_annotation(
        intake(),
        [annotation("a", "a1"), annotation("b", "b1", ontology="pilot-old")],
    )
    assert not result.valid
    assert not result.protocol_versions_compatible


def test_same_source_group_in_development_and_test_is_rejected() -> None:
    registry = IntakeRegistry(videos=[intake("clip_a"), intake("clip_b")])
    report = validate_release_integrity(
        registry,
        splits(
            ("clip_a", "session_001", DatasetSplit.DEVELOPMENT),
            ("clip_b", "session_001", DatasetSplit.TEST),
        ),
        {},
        {},
    )
    assert IntegrityReasonCode.SOURCE_GROUP_SPLIT_LEAKAGE in report.reason_codes


def test_manually_tampered_split_document_fails_release() -> None:
    registry = IntakeRegistry(videos=[intake("clip_a"), intake("clip_b")])
    tampered = splits(
        ("clip_a", "session_001", DatasetSplit.DEVELOPMENT),
        ("clip_b", "session_001", DatasetSplit.TEST),
    )
    with pytest.raises(SplitLeakageError) as raised:
        build_dataset_release(registry, tampered, {}, {}, created_at=NOW)
    assert (
        IntegrityReasonCode.SOURCE_GROUP_SPLIT_LEAKAGE
        in raised.value.report.reason_codes
    )


def test_split_source_group_disagreement_with_registry_is_rejected() -> None:
    report = validate_release_integrity(
        IntakeRegistry(videos=[intake()]),
        splits(("clip", "tampered_group", DatasetSplit.DEVELOPMENT)),
        {},
        {},
    )
    assert IntegrityReasonCode.SOURCE_GROUP_ID_MISMATCH in report.reason_codes


def test_duplicate_source_video_sha_across_splits_is_rejected() -> None:
    registry = IntakeRegistry(
        videos=[
            intake("clip_a", sha=XYZ, group="group_a"),
            intake("clip_b", sha=XYZ, group="group_b"),
        ]
    )
    with pytest.raises(SplitLeakageError) as raised:
        build_dataset_release(
            registry,
            splits(
                ("clip_a", "group_a", DatasetSplit.DEVELOPMENT),
                ("clip_b", "group_b", DatasetSplit.TEST),
            ),
            {},
            {},
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.DUPLICATE_VIDEO_CROSS_SPLIT_LEAKAGE
        in raised.value.report.reason_codes
    )


def test_duplicate_source_video_sha_in_same_split_is_allowed() -> None:
    registry = IntakeRegistry(
        videos=[
            intake("clip_a", sha=XYZ, group="group_a"),
            intake("clip_b", sha=XYZ, group="group_b"),
        ]
    )
    release = build_dataset_release(
        registry,
        splits(
            ("clip_a", "group_a", DatasetSplit.DEVELOPMENT),
            ("clip_b", "group_b", DatasetSplit.DEVELOPMENT),
        ),
        {},
        {},
        created_at=NOW,
    )
    assert release.integrity_report.passed


def test_locked_annotation_with_wrong_source_still_fails() -> None:
    wrong = annotation("a", "a1", sha=BBB, locked=True)
    assert wrong.locked
    with pytest.raises(SourceIdentityMismatchError):
        validate_annotation_source_identity(intake(), wrong)


def test_development_single_annotation_is_permitted_and_marked_invalid_double() -> None:
    single = annotation("a", "a1", locked=False)
    release = build_dataset_release(
        IntakeRegistry(videos=[intake()]),
        splits(("clip", "session_001", DatasetSplit.DEVELOPMENT)),
        {"clip": [single]},
        {},
        created_at=NOW,
    )
    assert release.integrity_report.passed
    assert not release.videos[0].double_annotation.valid
    assert release.videos[0].adjudication_status == "not_required"


def test_validation_single_annotation_fails_closed() -> None:
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[intake()]),
            splits(("clip", "session_001", DatasetSplit.VALIDATION)),
            {"clip": [annotation("a", "a1")]},
            {},
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.ANNOTATOR_COUNT_INSUFFICIENT
        in raised.value.report.reason_codes
    )


def test_test_clip_without_adjudication_fails_closed() -> None:
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[intake()]),
            splits(("clip", "session_001", DatasetSplit.TEST)),
            {"clip": [annotation("a", "a1"), annotation("b", "b1")]},
            {},
            created_at=NOW,
        )
    assert (
        IntegrityReasonCode.ADJUDICATION_NOT_APPROVED
        in raised.value.report.reason_codes
    )


def test_all_integrity_gates_valid_produces_release() -> None:
    record, first, second, artifact, split_document = valid_bundle()
    release = build_dataset_release(
        IntakeRegistry(videos=[record]),
        split_document,
        {"clip": [first, second]},
        {"clip": artifact},
        agreements=[artifact.agreement_report],
        created_at=NOW,
    )
    assert release.quality_gate_passed
    assert release.integrity_report.passed
    assert all(gate.passed for gate in release.integrity_report.gates)


def test_integrity_blocker_does_not_overwrite_official_release(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    splits_path = tmp_path / "splits.json"
    annotations_dir = tmp_path / "annotations"
    adjudications_dir = tmp_path / "adjudications"
    output = tmp_path / "release.json"
    first = annotation("a", "a1", sha=BBB)
    second = annotation("b", "b1", sha=BBB)
    artifact = adjudication(first, second)
    write_json_model(IntakeRegistry(videos=[intake()]), registry_path)
    write_json_model(splits(("clip", "session_001", DatasetSplit.TEST)), splits_path)
    write_json_model(first, annotations_dir / "a.json")
    write_json_model(second, annotations_dir / "b.json")
    write_json_model(artifact, adjudications_dir / "adjudication.json")
    output.write_text("existing-valid-release", encoding="utf-8")
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
            "--output",
            str(output),
        ]
    )
    assert result == 2
    assert output.read_text(encoding="utf-8") == "existing-valid-release"


def test_release_manifest_preserves_exact_video_and_ground_truth_hashes() -> None:
    record, first, second, artifact, split_document = valid_bundle()
    release = build_dataset_release(
        IntakeRegistry(videos=[record]),
        split_document,
        {"clip": [first, second]},
        {"clip": artifact},
        agreements=[artifact.agreement_report],
        created_at=NOW,
    )
    entry = release.videos[0]
    exported = export_adjudicated_annotation(
        artifact,
        record,
        split=DatasetSplit.TEST,
        source_annotations=[first, second],
        expected_ground_truth_sha256=entry.benchmark_ground_truth_sha256,
    )
    assert entry.source_video_sha256 == AAA
    assert entry.annotation_hashes == {
        "annotator_a": document_sha256(first),
        "annotator_b": document_sha256(second),
    }
    assert entry.adjudicated_annotation_hash == document_sha256(artifact)
    assert entry.benchmark_ground_truth_sha256 == document_sha256(exported)


def test_release_build_is_deterministic() -> None:
    record, first, second, artifact, split_document = valid_bundle()
    arguments = (
        IntakeRegistry(videos=[record]),
        split_document,
        {"clip": [first, second]},
        {"clip": artifact},
    )
    first_release = build_dataset_release(
        *arguments, agreements=[artifact.agreement_report], created_at=NOW
    )
    second_release = build_dataset_release(
        *arguments, agreements=[artifact.agreement_report], created_at=NOW
    )
    assert first_release.model_dump(mode="json") == second_release.model_dump(
        mode="json"
    )


def test_split_document_missing_duplicate_and_unknown_entries_are_rejected() -> None:
    registry = IntakeRegistry(videos=[intake()])
    report = validate_release_integrity(
        registry,
        splits(
            ("clip", "session_001", DatasetSplit.DEVELOPMENT),
            ("clip", "session_001", DatasetSplit.DEVELOPMENT),
            ("unknown", "other", DatasetSplit.DEVELOPMENT),
        ),
        {},
        {},
    )
    assert IntegrityReasonCode.SPLIT_ASSIGNMENT_DUPLICATE in report.reason_codes
    assert IntegrityReasonCode.SPLIT_ASSIGNMENT_UNKNOWN_VIDEO in report.reason_codes
    missing = validate_release_integrity(registry, splits(), {}, {})
    assert IntegrityReasonCode.SPLIT_ASSIGNMENT_MISSING in missing.reason_codes

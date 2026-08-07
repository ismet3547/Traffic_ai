from __future__ import annotations

from pathlib import Path

import pytest

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.agreement import (
    agreement_pair_id,
    compare_independent_annotations,
)
from app.dataset.agreement_integrity import (
    AgreementIntegrityError,
    assess_agreement_report,
    validate_agreement_report,
)
from app.dataset.integrity import DatasetIntegrityError
from app.dataset.io import load_agreement, lock_annotation, write_json_model
from app.dataset.models import (
    AGREEMENT_CONFIG_VERSION,
    AGREEMENT_PROTOCOL_VERSION,
    CANONICAL_AGREEMENT_CONFIG,
    CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
    CANONICAL_AGREEMENT_PROTOCOL,
    AgreementConfig,
    AgreementMode,
    AnnotationQualityConfig,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    IntakeRegistry,
    IntegrityReasonCode,
    agreement_config_fingerprint,
)
from app.dataset.release import build_dataset_release
from app.tools.compare_annotations import main as compare_main
from tests.test_dataset_agreement_provenance import (
    NOW,
    _adjudication,
    _bundle,
    _record,
    _rehash,
    _release_inputs,
    _splits,
)

PERMISSIVE_CONFIG = AgreementConfig(
    minimum_temporal_iou=0.01,
    boundary_tolerance_seconds=100.0,
    require_vehicle_reference_match=False,
)


def _event(
    event_id: str,
    start: float,
    end: float,
    *,
    vehicle_ref: str = "vehicle_1",
) -> DatasetEvent:
    return DatasetEvent(
        event_id=event_id,
        vehicle_ref=vehicle_ref,
        start_seconds=start,
        end_seconds=end,
        label=DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
        confidence=AnnotationConfidence.HIGH,
    )


def _annotation(record, annotator: str, events: list[DatasetEvent]):
    return lock_annotation(
        DatasetAnnotation(
            video_id=record.video_id,
            source_video_sha256=record.source_video_sha256,
            source_video_size_bytes=record.source_video_size_bytes,
            source_file=record.original_filename,
            fps=record.fps,
            video_duration_seconds=record.duration_seconds,
            annotator_id=annotator,
            created_at=NOW,
            events=events,
        ),
        locked_at=NOW,
    )


def _poor_alignment_pair(video_id: str = "poor", index: int = 80):
    record = _record(video_id, index)
    first = _annotation(record, "annotator_a", [_event("a1", 10, 20)])
    second = _annotation(record, "annotator_b", [_event("b1", 19, 29)])
    return record, first, second


def _release_one(
    record,
    first,
    second,
    report,
    *,
    split: DatasetSplit = DatasetSplit.VALIDATION,
):
    artifact = _adjudication(first, second)
    return build_dataset_release(
        IntakeRegistry(videos=[record]),
        _splits([record], split),
        {record.video_id: [first, second]},
        {record.video_id: artifact},
        agreements=[report],
        created_at=NOW,
    )


def _codes(record, first, second, report) -> set[IntegrityReasonCode]:
    return set(assess_agreement_report(record, first, second, report).reason_codes)


def test_official_report_with_exact_canonical_config_is_accepted() -> None:
    record, first, second, report, artifact = _bundle()
    result = validate_agreement_report(
        record,
        first,
        second,
        report,
        adjudication=artifact,
    )
    assert result.valid
    assert report.agreement_mode == AgreementMode.OFFICIAL
    assert report.agreement_config == CANONICAL_AGREEMENT_CONFIG
    assert report.agreement_config_fingerprint == CANONICAL_AGREEMENT_CONFIG_FINGERPRINT


def test_different_iou_threshold_is_rejected_for_official_use() -> None:
    record, first, second, _, _ = _bundle()
    report = compare_independent_annotations(
        first,
        second,
        AgreementConfig(minimum_temporal_iou=0.2),
    )
    assert IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH in _codes(
        record, first, second, report
    )


def test_different_boundary_tolerance_is_rejected_for_official_use() -> None:
    record, first, second, _, _ = _bundle()
    report = compare_independent_annotations(
        first,
        second,
        AgreementConfig(boundary_tolerance_seconds=5.0),
    )
    assert IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH in _codes(
        record, first, second, report
    )


def test_different_vehicle_reference_policy_is_rejected_for_official_use() -> None:
    record, first, second, _, _ = _bundle()
    report = compare_independent_annotations(
        first,
        second,
        AgreementConfig(require_vehicle_reference_match=False),
    )
    assert IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH in _codes(
        record, first, second, report
    )


@pytest.mark.parametrize("split", [DatasetSplit.VALIDATION, DatasetSplit.TEST])
def test_exploratory_report_is_rejected_for_official_release(
    split: DatasetSplit,
) -> None:
    record, first, second, _, _ = _bundle()
    report = compare_independent_annotations(
        first,
        second,
        PERMISSIVE_CONFIG,
        mode=AgreementMode.EXPLORATORY,
    )
    with pytest.raises(DatasetIntegrityError) as raised:
        _release_one(record, first, second, report, split=split)
    assert (
        IntegrityReasonCode.AGREEMENT_MODE_NOT_OFFICIAL
        in raised.value.report.reason_codes
    )


def test_development_explicitly_allows_valid_exploratory_report() -> None:
    record, first, second, _, _ = _bundle("development", 81)
    report = compare_independent_annotations(
        first,
        second,
        PERMISSIVE_CONFIG,
        mode=AgreementMode.EXPLORATORY,
    )
    release = build_dataset_release(
        IntakeRegistry(videos=[record]),
        _splits([record], DatasetSplit.DEVELOPMENT),
        {record.video_id: [first, second]},
        {},
        agreements=[report],
        created_at=NOW,
    )
    assert release.integrity_report.passed
    assert release.agreement_quality.total_agreement_videos == 0


def test_protocol_version_mismatch_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    old = _rehash(
        report,
        agreement_protocol_version="1",
        agreement_id=agreement_pair_id(
            first,
            second,
            protocol_version="1",
            config_fingerprint=CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
        ),
    )
    assert IntegrityReasonCode.AGREEMENT_PROTOCOL_MISMATCH in _codes(
        record, first, second, old
    )


def test_internally_valid_relaxed_fingerprint_is_not_canonical() -> None:
    record, first, second, _, _ = _bundle()
    relaxed = compare_independent_annotations(first, second, PERMISSIVE_CONFIG)
    claimed_official = _rehash(relaxed, agreement_mode=AgreementMode.OFFICIAL)
    codes = _codes(record, first, second, claimed_official)
    assert IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH in codes
    assert (
        claimed_official.agreement_config_fingerprint
        == agreement_config_fingerprint(PERMISSIVE_CONFIG)
    )


def test_canonical_config_fields_with_wrong_fingerprint_are_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(report, agreement_config_fingerprint="f" * 64)
    assert IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH in _codes(
        record, first, second, wrong
    )


def test_canonical_fingerprint_with_different_config_fields_is_rejected() -> None:
    record, first, second, report, _ = _bundle()
    wrong = _rehash(
        report,
        agreement_config=PERMISSIVE_CONFIG,
        agreement_config_fingerprint=CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
    )
    assert IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH in _codes(
        record, first, second, wrong
    )


def test_release_recomputes_metrics_with_canonical_config() -> None:
    record, first, second = _poor_alignment_pair()
    official = compare_independent_annotations(first, second)
    permissive = compare_independent_annotations(first, second, PERMISSIVE_CONFIG)
    assert official.event_detection_agreement == 0.0
    assert permissive.event_detection_agreement == 1.0
    forged = _rehash(
        permissive,
        agreement_mode=AgreementMode.OFFICIAL,
        agreement_config=CANONICAL_AGREEMENT_CONFIG,
        agreement_config_version=AGREEMENT_CONFIG_VERSION,
        agreement_config_fingerprint=CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
        agreement_id=official.agreement_id,
    )
    result = assess_agreement_report(record, first, second, forged)
    assert IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT in result.reason_codes
    with pytest.raises(DatasetIntegrityError) as raised:
        _release_one(record, first, second, forged)
    assert IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT in (
        raised.value.report.reason_codes
    )


def test_mixed_official_and_exploratory_reports_fail_release() -> None:
    bundles = [_bundle("v1", 82), _bundle("v2", 83)]
    registry, split_doc, annotations, adjudications, reports = _release_inputs(bundles)
    exploratory = compare_independent_annotations(
        bundles[1][1], bundles[1][2], PERMISSIVE_CONFIG
    )
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            registry,
            split_doc,
            annotations,
            adjudications,
            agreements=[reports[0], exploratory],
            created_at=NOW,
        )
    assert IntegrityReasonCode.MIXED_AGREEMENT_PROTOCOLS in (
        raised.value.report.reason_codes
    )


def test_mixed_official_config_fingerprints_fail_release() -> None:
    bundles = [_bundle("v1", 84), _bundle("v2", 85), _bundle("v3", 86)]
    registry, split_doc, annotations, adjudications, reports = _release_inputs(bundles)
    relaxed = compare_independent_annotations(
        bundles[2][1], bundles[2][2], PERMISSIVE_CONFIG
    )
    false_official = _rehash(relaxed, agreement_mode=AgreementMode.OFFICIAL)
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            registry,
            split_doc,
            annotations,
            adjudications,
            agreements=[reports[0], reports[1], false_official],
            created_at=NOW,
        )
    assert IntegrityReasonCode.MIXED_AGREEMENT_PROTOCOLS in (
        raised.value.report.reason_codes
    )


def test_relaxed_config_cannot_inflate_official_quality_score() -> None:
    record, first, second = _poor_alignment_pair("inflation", 87)
    permissive = compare_independent_annotations(first, second, PERMISSIVE_CONFIG)
    assert permissive.event_detection_agreement == 1.0
    with pytest.raises(DatasetIntegrityError) as raised:
        _release_one(record, first, second, permissive)
    assert {
        IntegrityReasonCode.AGREEMENT_MODE_NOT_OFFICIAL,
        IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH,
    } <= set(raised.value.report.reason_codes)


def test_different_vehicle_refs_do_not_match_under_official_protocol() -> None:
    record = _record("vehicles", 88)
    first = _annotation(
        record,
        "annotator_a",
        [_event("a1", 10, 20, vehicle_ref="vehicle_1")],
    )
    second = _annotation(
        record,
        "annotator_b",
        [_event("b1", 10, 20, vehicle_ref="vehicle_2")],
    )
    official = compare_independent_annotations(first, second)
    exploratory = compare_independent_annotations(first, second, PERMISSIVE_CONFIG)
    assert official.matched_event_count == 0
    assert exploratory.matched_event_count == 1


def test_agreement_logical_identity_includes_config_fingerprint() -> None:
    _, first, second, official, _ = _bundle()
    exploratory = compare_independent_annotations(first, second, PERMISSIVE_CONFIG)
    assert official.agreement_config_fingerprint != (
        exploratory.agreement_config_fingerprint
    )
    assert official.agreement_id != exploratory.agreement_id


def test_protocol_change_invalidates_old_report_with_same_annotations() -> None:
    record, first, second, current, _ = _bundle()
    old = _rehash(
        current,
        agreement_protocol_version="1",
        agreement_id=agreement_pair_id(
            first,
            second,
            protocol_version="1",
            config_fingerprint=current.agreement_config_fingerprint,
        ),
    )
    with pytest.raises(AgreementIntegrityError) as raised:
        validate_agreement_report(record, first, second, old)
    assert (
        IntegrityReasonCode.AGREEMENT_PROTOCOL_MISMATCH
        in raised.value.result.reason_codes
    )
    assert compare_independent_annotations(first, second) == current


def test_zero_event_video_count_is_reported() -> None:
    bundle = _bundle("zero", 89, count=0, agreeing=0)
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    release = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        created_at=NOW,
    )
    quality = release.agreement_quality
    assert quality.total_agreement_videos == 1
    assert quality.zero_event_both_annotators_video_count == 1
    assert quality.positive_event_video_count == 0
    assert quality.event_detection_agreement == 1.0
    assert quality.positive_event_video_event_detection_agreement is None


def test_positive_event_subset_agreement_is_reported_separately() -> None:
    zero = _bundle("zero", 90, count=0, agreeing=0)
    record = _record("positive", 91)
    first = _annotation(
        record,
        "annotator_a",
        [_event("a1", 5, 10), _event("a2", 15, 20)],
    )
    second = _annotation(record, "annotator_b", [_event("b1", 5, 10)])
    report = compare_independent_annotations(first, second)
    positive = (record, first, second, report, _adjudication(first, second))
    registry, split_doc, annotations, adjudications, reports = _release_inputs(
        [zero, positive]
    )
    release = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        created_at=NOW,
    )
    expected_positive = 2 / 3
    assert release.agreement_quality.positive_event_video_count == 1
    assert release.agreement_quality.zero_event_both_annotators_video_count == 1
    assert (
        release.agreement_quality.positive_event_video_event_detection_agreement
        == pytest.approx(expected_positive)
    )
    assert release.agreement_quality.event_detection_agreement == pytest.approx(
        (1.0 + expected_positive) / 2
    )


def test_dataset_level_canonical_protocol_metadata_is_preserved() -> None:
    bundle = _bundle("metadata", 92)
    registry, split_doc, annotations, adjudications, reports = _release_inputs([bundle])
    release = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        created_at=NOW,
    )
    assert release.agreement_protocol == CANONICAL_AGREEMENT_PROTOCOL
    assert release.agreement_protocol.protocol_version == AGREEMENT_PROTOCOL_VERSION
    assert release.agreement_protocol.config_version == AGREEMENT_CONFIG_VERSION
    assert (
        release.agreement_protocol.config_fingerprint
        == CANONICAL_AGREEMENT_CONFIG_FINGERPRINT
    )
    assert release.videos[0].agreement_provenance is not None
    assert release.videos[0].agreement_provenance.agreement_mode == (
        AgreementMode.OFFICIAL
    )


def test_existing_thresholds_remain_overall_macro_per_video() -> None:
    zero = _bundle("zero_threshold", 93, count=0, agreeing=0)
    registry, split_doc, annotations, adjudications, reports = _release_inputs([zero])
    release = build_dataset_release(
        registry,
        split_doc,
        annotations,
        adjudications,
        agreements=reports,
        quality_config=AnnotationQualityConfig(minimum_event_match_rate=1.0),
        created_at=NOW,
    )
    gate = next(
        item
        for item in release.quality_gates
        if item.gate == "minimum_event_match_rate"
    )
    assert gate.passed
    assert "scope=overall_macro_per_video" in gate.details


def test_release_under_canonical_protocol_is_deterministic() -> None:
    bundle = _bundle("deterministic", 94)
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


def test_compare_cli_defaults_to_official_and_custom_is_exploratory(
    tmp_path: Path,
) -> None:
    _, first, second, _, _ = _bundle("cli_protocol", 95)
    first_path = write_json_model(first, tmp_path / "a.json")
    second_path = write_json_model(second, tmp_path / "b.json")
    official_path = tmp_path / "official.json"
    exploratory_path = tmp_path / "exploratory.json"
    assert (
        compare_main(
            [str(first_path), str(second_path), "--output", str(official_path)]
        )
        == 0
    )
    assert (
        compare_main(
            [
                str(first_path),
                str(second_path),
                "--output",
                str(exploratory_path),
                "--minimum-temporal-iou",
                "0.01",
            ]
        )
        == 0
    )
    assert load_agreement(official_path).agreement_mode == AgreementMode.OFFICIAL
    assert load_agreement(exploratory_path).agreement_mode == AgreementMode.EXPLORATORY


def test_compare_cli_official_mode_forbids_overrides(tmp_path: Path) -> None:
    _, first, second, _, _ = _bundle("cli_guard", 96)
    first_path = write_json_model(first, tmp_path / "a.json")
    second_path = write_json_model(second, tmp_path / "b.json")
    with pytest.raises(SystemExit):
        compare_main(
            [
                str(first_path),
                str(second_path),
                "--output",
                str(tmp_path / "invalid.json"),
                "--official",
                "--minimum-temporal-iou",
                "0.01",
            ]
        )

"""Execute Phase 4.2.3 canonical agreement protocol scenarios."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.agreement import agreement_pair_id, compare_independent_annotations
from app.dataset.integrity import DatasetIntegrityError
from app.dataset.io import (
    agreement_report_content_hash,
    lock_annotation,
    write_json_model,
)
from app.dataset.models import (
    AGREEMENT_PROTOCOL_VERSION,
    CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
    AdjudicationArtifact,
    AdjudicationDecision,
    AdjudicationOutcome,
    AgreementConfig,
    AgreementReport,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    IntakeRegistry,
    IntegrityScenarioOutcome,
    IntegrityScenarioSummary,
    PermissionStatus,
    SourceType,
    SplitAssignment,
    SplitAssignmentDocument,
    VideoIntakeRecord,
    VideoResolution,
)
from app.dataset.release import build_dataset_release

NOW = datetime(2026, 3, 2, tzinfo=timezone.utc)
PERMISSIVE_CONFIG = AgreementConfig(
    minimum_temporal_iou=0.01,
    boundary_tolerance_seconds=100.0,
    require_vehicle_reference_match=False,
)


def _record(video_id: str, index: int) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id=video_id,
        source_group_id=f"group_{video_id}",
        source_type=SourceType.OTHER,
        source_reference="generated:phase-4.2.3-protocol-scenario",
        acquisition_date=date(2026, 3, 2),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
        source_video_sha256=f"{index:064x}",
        source_video_size_bytes=2000 + index,
        duration_seconds=40,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename=f"{video_id}.mp4",
    )


def _annotation(
    record: VideoIntakeRecord,
    annotator: str,
    *,
    start: float,
    end: float,
) -> DatasetAnnotation:
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
            events=[
                DatasetEvent(
                    event_id=f"event_{annotator}",
                    vehicle_ref="vehicle_1",
                    start_seconds=start,
                    end_seconds=end,
                    label=DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
                    confidence=AnnotationConfidence.HIGH,
                )
            ],
        ),
        locked_at=NOW,
    )


def _artifact(
    first: DatasetAnnotation, second: DatasetAnnotation
) -> AdjudicationArtifact:
    report = compare_independent_annotations(first, second)
    first_events = {item.event_id: item for item in first.events}
    second_events = {item.event_id: item for item in second.events}
    decisions: list[AdjudicationDecision] = []
    for index, disagreement in enumerate(report.disagreements, start=1):
        if disagreement.event_id_a is not None:
            event = first_events[disagreement.event_id_a]
            outcome = AdjudicationOutcome.RESOLVED_TO_A
        else:
            assert disagreement.event_id_b is not None
            event = second_events[disagreement.event_id_b]
            outcome = AdjudicationOutcome.RESOLVED_TO_B
        decisions.append(
            AdjudicationDecision(
                decision_id=f"decision_{index}",
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
                rationale="synthetic canonical-protocol scenario decision",
                adjudication_confidence=event.confidence,
            )
        )
    return lock_adjudication(
        create_adjudication(
            first,
            second,
            adjudicator_id="synthetic_adjudicator",
            decisions=decisions,
            created_at=NOW,
        ),
        locked_at=NOW,
    )


def _splits(records: list[VideoIntakeRecord]) -> SplitAssignmentDocument:
    return SplitAssignmentDocument(
        seed=42,
        target_ratios={
            DatasetSplit.DEVELOPMENT: 0,
            DatasetSplit.VALIDATION: 1,
            DatasetSplit.TEST: 0,
        },
        assignments=[
            SplitAssignment(
                video_id=item.video_id,
                source_group_id=item.source_group_id,
                split=DatasetSplit.VALIDATION,
            )
            for item in records
        ],
    )


def _run(
    name: str,
    expected: Literal["PASS", "FAIL"],
    output: Path,
    records: list[VideoIntakeRecord],
    annotations: dict[str, list[DatasetAnnotation]],
    adjudications: dict[str, AdjudicationArtifact],
    reports: list[AgreementReport],
    *,
    canonical_recomputed_metric: float,
) -> IntegrityScenarioOutcome:
    destination = output / name / "dataset_release.json"
    reason_codes = []
    actual_label: float | None = None
    actual_event: float | None = None
    try:
        release = build_dataset_release(
            IntakeRegistry(videos=records),
            _splits(records),
            annotations,
            adjudications,
            agreements=reports,
            created_at=NOW,
        )
        write_json_model(release, destination)
        actual: Literal["PASS", "FAIL"] = "PASS"
        actual_label = release.agreement_quality.label_agreement
        actual_event = release.agreement_quality.event_detection_agreement
    except DatasetIntegrityError as exc:
        actual = "FAIL"
        reason_codes = exc.report.reason_codes
    representative = reports[-1]
    return IntegrityScenarioOutcome(
        scenario=name,
        expected=expected,
        actual=actual,
        expectation_met=(
            actual == expected and destination.is_file() == (expected == "PASS")
        ),
        reason_codes=reason_codes,
        release_written=destination.is_file(),
        expected_label_agreement=(actual_label if expected == "PASS" else None),
        actual_label_agreement=actual_label,
        expected_event_detection_agreement=(
            canonical_recomputed_metric if expected == "PASS" else None
        ),
        actual_event_detection_agreement=actual_event,
        supplied_event_detection_agreement=(representative.event_detection_agreement),
        canonical_recomputed_event_detection_agreement=(canonical_recomputed_metric),
        agreement_mode=representative.agreement_mode,
        agreement_protocol_version=representative.agreement_protocol_version,
        agreement_config_fingerprint=(representative.agreement_config_fingerprint),
    )


def _pair(
    video_id: str,
    index: int,
    *,
    aligned: bool,
) -> tuple[VideoIntakeRecord, DatasetAnnotation, DatasetAnnotation]:
    record = _record(video_id, index)
    first = _annotation(record, "annotator_a", start=10, end=20)
    second = _annotation(
        record,
        "annotator_b",
        start=10 if aligned else 19,
        end=20 if aligned else 29,
    )
    return record, first, second


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)

    aligned = _pair("canonical_clip", 1, aligned=True)
    poor = _pair("poor_alignment", 2, aligned=False)
    aligned_report = compare_independent_annotations(aligned[1], aligned[2])
    poor_official = compare_independent_annotations(poor[1], poor[2])
    poor_exploratory = compare_independent_annotations(
        poor[1], poor[2], PERMISSIVE_CONFIG
    )
    aligned_artifact = _artifact(aligned[1], aligned[2])
    poor_artifact = _artifact(poor[1], poor[2])

    old_protocol = aligned_report.model_copy(
        update={
            "agreement_protocol_version": "1",
            "agreement_id": agreement_pair_id(
                aligned[1],
                aligned[2],
                protocol_version="1",
                config_fingerprint=CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
            ),
        }
    )
    old_protocol = old_protocol.model_copy(
        update={"agreement_content_sha256": agreement_report_content_hash(old_protocol)}
    )

    mixed_pairs = [
        _pair("mixed_v1", 10, aligned=True),
        _pair("mixed_v2", 11, aligned=True),
        _pair("mixed_v3", 12, aligned=False),
    ]
    mixed_reports = [
        compare_independent_annotations(item[1], item[2]) for item in mixed_pairs[:2]
    ] + [
        compare_independent_annotations(
            mixed_pairs[2][1], mixed_pairs[2][2], PERMISSIVE_CONFIG
        )
    ]

    scenarios = [
        _run(
            "official_canonical_reports",
            "PASS",
            output,
            [aligned[0]],
            {aligned[0].video_id: [aligned[1], aligned[2]]},
            {aligned[0].video_id: aligned_artifact},
            [aligned_report],
            canonical_recomputed_metric=1.0,
        ),
        _run(
            "permissive_exploratory_report",
            "FAIL",
            output,
            [poor[0]],
            {poor[0].video_id: [poor[1], poor[2]]},
            {poor[0].video_id: poor_artifact},
            [poor_exploratory],
            canonical_recomputed_metric=poor_official.event_detection_agreement,
        ),
        _run(
            "mixed_configs",
            "FAIL",
            output,
            [item[0] for item in mixed_pairs],
            {item[0].video_id: [item[1], item[2]] for item in mixed_pairs},
            {item[0].video_id: _artifact(item[1], item[2]) for item in mixed_pairs},
            mixed_reports,
            canonical_recomputed_metric=(1.0 + 1.0 + 0.0) / 3,
        ),
        _run(
            "old_protocol_report",
            "FAIL",
            output,
            [aligned[0]],
            {aligned[0].video_id: [aligned[1], aligned[2]]},
            {aligned[0].video_id: aligned_artifact},
            [old_protocol],
            canonical_recomputed_metric=1.0,
        ),
        _run(
            "regenerated_current_protocol",
            "PASS",
            output,
            [aligned[0]],
            {aligned[0].video_id: [aligned[1], aligned[2]]},
            {aligned[0].video_id: aligned_artifact},
            [compare_independent_annotations(aligned[1], aligned[2])],
            canonical_recomputed_metric=1.0,
        ),
    ]
    summary = IntegrityScenarioSummary(
        all_expectations_met=all(item.expectation_met for item in scenarios),
        scenarios=scenarios,
    )
    write_json_model(summary, output / "scenario_results.json")
    print(
        f"canonical_protocol={AGREEMENT_PROTOCOL_VERSION} "
        f"canonical_config_sha={CANONICAL_AGREEMENT_CONFIG_FINGERPRINT}"
    )
    for item in scenarios:
        print(
            f"{item.scenario}: expected={item.expected} actual={item.actual} "
            f"mode={item.agreement_mode.value if item.agreement_mode else None} "
            f"config_sha={item.agreement_config_fingerprint} "
            f"supplied_event={item.supplied_event_detection_agreement} "
            f"canonical_event={item.canonical_recomputed_event_detection_agreement} "
            f"release_event={item.actual_event_detection_agreement} "
            f"reason_codes={[code.value for code in item.reason_codes]}"
        )
    return 0 if summary.all_expectations_met else 1


if __name__ == "__main__":
    raise SystemExit(main())

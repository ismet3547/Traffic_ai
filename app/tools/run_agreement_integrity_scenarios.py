"""Execute Phase 4.2.2 agreement-provenance release scenarios."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.agreement import compare_independent_annotations
from app.dataset.integrity import DatasetIntegrityError
from app.dataset.io import lock_annotation, write_json_model
from app.dataset.models import (
    AdjudicationArtifact,
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

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _record(video_id: str, index: int) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id=video_id,
        source_group_id=f"group_{video_id}",
        source_type=SourceType.OTHER,
        source_reference="generated:phase-4.2.2-agreement-scenario",
        acquisition_date=date(2026, 3, 1),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
        source_video_sha256=f"{index:064x}",
        source_video_size_bytes=1000 + index,
        duration_seconds=30,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename=f"{video_id}.mp4",
    )


def _annotation(record: VideoIntakeRecord, annotator: str) -> DatasetAnnotation:
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
                    start_seconds=5,
                    end_seconds=15,
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
    return lock_adjudication(
        create_adjudication(
            first,
            second,
            adjudicator_id="synthetic_adjudicator",
            decisions=[],
            created_at=NOW,
        ),
        locked_at=NOW,
    )


def _splits(record: VideoIntakeRecord) -> SplitAssignmentDocument:
    return SplitAssignmentDocument(
        seed=42,
        target_ratios={
            DatasetSplit.DEVELOPMENT: 0,
            DatasetSplit.VALIDATION: 1,
            DatasetSplit.TEST: 0,
        },
        assignments=[
            SplitAssignment(
                video_id=record.video_id,
                source_group_id=record.source_group_id,
                split=DatasetSplit.VALIDATION,
            )
        ],
    )


def _run(
    name: str,
    expected: Literal["PASS", "FAIL"],
    output: Path,
    record: VideoIntakeRecord,
    first: DatasetAnnotation,
    second: DatasetAnnotation,
    artifact: AdjudicationArtifact,
    reports: list[AgreementReport],
) -> IntegrityScenarioOutcome:
    destination = output / name / "dataset_release.json"
    actual_label: float | None = None
    actual_event: float | None = None
    reason_codes = []
    try:
        release = build_dataset_release(
            IntakeRegistry(videos=[record]),
            _splits(record),
            {record.video_id: [first, second]},
            {record.video_id: artifact},
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
    return IntegrityScenarioOutcome(
        scenario=name,
        expected=expected,
        actual=actual,
        expectation_met=(
            actual == expected and destination.is_file() == (expected == "PASS")
        ),
        reason_codes=reason_codes,
        release_written=destination.is_file(),
        expected_label_agreement=1.0 if expected == "PASS" else None,
        actual_label_agreement=actual_label,
        expected_event_detection_agreement=1.0 if expected == "PASS" else None,
        actual_event_detection_agreement=actual_event,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)

    record = _record("current_clip", 1)
    first = _annotation(record, "annotator_a")
    second = _annotation(record, "annotator_b")
    report = compare_independent_annotations(first, second)
    artifact = _artifact(first, second)

    unrelated_record = _record("unrelated_easy_clip", 2)
    unrelated_report = compare_independent_annotations(
        _annotation(unrelated_record, "annotator_a"),
        _annotation(unrelated_record, "annotator_b"),
    )

    revised_second = lock_annotation(
        second.model_copy(
            update={
                "notes": "audited semantic clarification",
                "locked": False,
                "locked_at": None,
                "annotation_hash": None,
            }
        ),
        locked_at=NOW + timedelta(seconds=1),
    )
    revised_report = compare_independent_annotations(first, revised_second)
    revised_artifact = _artifact(first, revised_second)

    scenarios = [
        _run(
            "valid_current_reports",
            "PASS",
            output,
            record,
            first,
            second,
            artifact,
            [report],
        ),
        _run(
            "unrelated_perfect_report_injected",
            "FAIL",
            output,
            record,
            first,
            second,
            artifact,
            [report, unrelated_report],
        ),
        _run(
            "duplicate_report_injected",
            "FAIL",
            output,
            record,
            first,
            second,
            artifact,
            [report, report],
        ),
        _run(
            "stale_report_after_annotation_edit",
            "FAIL",
            output,
            record,
            first,
            revised_second,
            revised_artifact,
            [report],
        ),
        _run(
            "agreement_adjudication_revision_mismatch",
            "FAIL",
            output,
            record,
            first,
            revised_second,
            artifact,
            [revised_report],
        ),
    ]
    summary = IntegrityScenarioSummary(
        all_expectations_met=all(item.expectation_met for item in scenarios),
        scenarios=scenarios,
    )
    write_json_model(summary, output / "scenario_results.json")
    for item in scenarios:
        print(
            f"{item.scenario}: expected={item.expected} actual={item.actual} "
            f"reason_codes={[code.value for code in item.reason_codes]} "
            f"expected_label={item.expected_label_agreement} "
            f"actual_label={item.actual_label_agreement} "
            f"expected_event={item.expected_event_detection_agreement} "
            f"actual_event={item.actual_event_detection_agreement}"
        )
    return 0 if summary.all_expectations_met else 1


if __name__ == "__main__":
    raise SystemExit(main())

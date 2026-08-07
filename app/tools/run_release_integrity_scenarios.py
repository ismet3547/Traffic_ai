"""Execute the five Phase 4.2.1 synthetic release-integrity scenarios."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.integrity import DatasetIntegrityError
from app.dataset.io import lock_annotation, write_json_model
from app.dataset.models import (
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

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)
AAA = "a" * 64
BBB = "b" * 64
XYZ = "c" * 64


def _record(
    video_id: str = "clip", *, sha: str = AAA, group: str = "session"
) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id=video_id,
        source_group_id=group,
        source_type=SourceType.OTHER,
        source_reference="generated:phase-4.2.1-integrity-scenario",
        acquisition_date=date(2026, 2, 1),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
        source_video_sha256=sha,
        source_video_size_bytes=100,
        duration_seconds=60,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename=f"{video_id}.mp4",
    )


def _annotation(annotator: str, event_id: str, *, sha: str = AAA) -> DatasetAnnotation:
    return lock_annotation(
        DatasetAnnotation(
            video_id="clip",
            source_video_sha256=sha,
            source_video_size_bytes=100,
            source_file="clip.mp4",
            fps=30,
            video_duration_seconds=60,
            annotator_id=annotator,
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
        ),
        locked_at=NOW,
    )


def _adjudication(first: DatasetAnnotation, second: DatasetAnnotation):
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


def _splits(*items: tuple[str, str, DatasetSplit]) -> SplitAssignmentDocument:
    return SplitAssignmentDocument(
        seed=42,
        target_ratios={
            DatasetSplit.DEVELOPMENT: 0.5,
            DatasetSplit.VALIDATION: 0.25,
            DatasetSplit.TEST: 0.25,
        },
        assignments=[
            SplitAssignment(video_id=video, source_group_id=group, split=split)
            for video, group, split in items
        ],
    )


def _run(
    name: str,
    expected: Literal["PASS", "FAIL"],
    output: Path,
    registry: IntakeRegistry,
    split_document: SplitAssignmentDocument,
    annotations: dict[str, list[DatasetAnnotation]],
    adjudications: dict,
    agreements: list[AgreementReport] | None = None,
) -> IntegrityScenarioOutcome:
    destination = output / name / "dataset_release.json"
    reason_codes = []
    try:
        release = build_dataset_release(
            registry,
            split_document,
            annotations,
            adjudications,
            agreements=agreements or [],
            created_at=NOW,
        )
        write_json_model(release, destination)
        actual: Literal["PASS", "FAIL"] = "PASS"
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    first, second = _annotation("a", "a1"), _annotation("b", "b1")
    artifact = _adjudication(first, second)
    wrong_a, wrong_b = _annotation("a", "a1", sha=BBB), _annotation("b", "b1", sha=BBB)
    wrong_artifact = _adjudication(wrong_a, wrong_b)
    revised_b = lock_annotation(
        second.model_copy(
            update={
                "events": [second.events[0].model_copy(update={"end_seconds": 21.0})],
                "locked": False,
                "locked_at": None,
                "annotation_hash": None,
            }
        ),
        locked_at=NOW,
    )
    scenarios = [
        _run(
            "valid_release",
            "PASS",
            output,
            IntakeRegistry(videos=[_record()]),
            _splits(("clip", "session", DatasetSplit.TEST)),
            {"clip": [first, second]},
            {"clip": artifact},
            [artifact.agreement_report],
        ),
        _run(
            "wrong_source_sha_perfect_agreement",
            "FAIL",
            output,
            IntakeRegistry(videos=[_record()]),
            _splits(("clip", "session", DatasetSplit.TEST)),
            {"clip": [wrong_a, wrong_b]},
            {"clip": wrong_artifact},
            [wrong_artifact.agreement_report],
        ),
        _run(
            "manual_source_group_leakage",
            "FAIL",
            output,
            IntakeRegistry(videos=[_record("clip_a"), _record("clip_b", sha=BBB)]),
            _splits(
                ("clip_a", "session", DatasetSplit.DEVELOPMENT),
                ("clip_b", "session", DatasetSplit.TEST),
            ),
            {},
            {},
        ),
        _run(
            "duplicate_bytes_cross_split",
            "FAIL",
            output,
            IntakeRegistry(
                videos=[
                    _record("clip_a", sha=XYZ, group="group_a"),
                    _record("clip_b", sha=XYZ, group="group_b"),
                ]
            ),
            _splits(
                ("clip_a", "group_a", DatasetSplit.DEVELOPMENT),
                ("clip_b", "group_b", DatasetSplit.TEST),
            ),
            {},
            {},
        ),
        _run(
            "stale_adjudication",
            "FAIL",
            output,
            IntakeRegistry(videos=[_record()]),
            _splits(("clip", "session", DatasetSplit.TEST)),
            {"clip": [first, revised_b]},
            {"clip": artifact},
            [artifact.agreement_report],
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
            f"release_written={item.release_written} "
            f"reason_codes={[code.value for code in item.reason_codes]}"
        )
    return 0 if summary.all_expectations_met else 1


if __name__ == "__main__":
    raise SystemExit(main())

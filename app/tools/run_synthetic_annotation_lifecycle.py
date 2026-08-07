"""Run the full Phase 4.2 lifecycle using explicitly synthetic metadata."""

from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.agreement import compare_independent_annotations
from app.dataset.io import lock_annotation, write_json_model
from app.dataset.models import (
    AdjudicationDecision,
    AdjudicationOutcome,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    EventEvidence,
    IntakeRegistry,
    PermissionStatus,
    SourceType,
    SplitCandidate,
    VehicleClass,
    VideoIntakeRecord,
    VideoResolution,
    VisibilityQuality,
)
from app.dataset.release import build_dataset_release, export_adjudicated_annotation
from app.dataset.reporting import build_coverage_report, write_coverage_report
from app.dataset.splitting import assign_group_aware_splits


def _event(
    event_id: str,
    label: DatasetLabel,
    start: float,
    end: float,
    confidence: AnnotationConfidence,
) -> DatasetEvent:
    return DatasetEvent(
        event_id=event_id,
        vehicle_ref="vehicle_001",
        start_seconds=start,
        end_seconds=end,
        label=label,
        confidence=confidence,
        evidence=EventEvidence(
            right_lane_available=True,
            congestion_present=False,
            visibility_quality=VisibilityQuality.GOOD,
            vehicle_class=VehicleClass.PASSENGER_CAR,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    digest = hashlib.sha256(
        b"phase-4.2 synthetic fixture; not real footage"
    ).hexdigest()
    record = VideoIntakeRecord(
        video_id="synthetic_phase42",
        source_group_id="synthetic_source_group",
        source_type=SourceType.OTHER,
        source_reference="generated:test-only:no-video",
        acquisition_date=date(2026, 1, 1),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
        notes="Synthetic CI fixture. This is not a real-video claim.",
        source_video_sha256=digest,
        source_video_size_bytes=44,
        duration_seconds=60,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename="synthetic_phase42.mp4",
        scenario_tags=["daylight", "fixed_camera", "free_flow", "synthetic"],
        vehicle_classes=[VehicleClass.PASSENGER_CAR],
    )
    registry = IntakeRegistry(videos=[record])
    annotation_a = lock_annotation(
        DatasetAnnotation(
            video_id=record.video_id,
            source_video_sha256=digest,
            source_video_size_bytes=record.source_video_size_bytes,
            source_file=record.original_filename,
            fps=record.fps,
            video_duration_seconds=record.duration_seconds,
            created_at=now,
            annotator_id="annotator_a",
            events=[
                _event(
                    "a_001",
                    DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
                    10,
                    20,
                    AnnotationConfidence.HIGH,
                ),
                _event(
                    "a_002",
                    DatasetLabel.LEGITIMATE_OVERTAKING,
                    30,
                    40,
                    AnnotationConfidence.HIGH,
                ),
            ],
        ),
        locked_at=now,
    )
    annotation_b = lock_annotation(
        DatasetAnnotation(
            video_id=record.video_id,
            source_video_sha256=digest,
            source_video_size_bytes=record.source_video_size_bytes,
            source_file=record.original_filename,
            fps=record.fps,
            video_duration_seconds=record.duration_seconds,
            created_at=now,
            annotator_id="annotator_b",
            events=[
                _event(
                    "b_001",
                    DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
                    10.2,
                    20.2,
                    AnnotationConfidence.HIGH,
                ),
                _event(
                    "b_002",
                    DatasetLabel.TEMPORARY_LEFT_LANE_USE,
                    30,
                    40,
                    AnnotationConfidence.MEDIUM,
                ),
            ],
        ),
        locked_at=now,
    )
    agreement = compare_independent_annotations(annotation_a, annotation_b)
    disputed = next(
        item for item in agreement.disagreements if item.event_id_a == "a_002"
    )
    decision = AdjudicationDecision(
        decision_id="decision_001",
        disagreement_ids=[disputed.disagreement_id],
        event_ids_a=["a_002"],
        event_ids_b=["b_002"],
        outcome=AdjudicationOutcome.RESOLVED_TO_A,
        adjudicated_event=annotation_a.events[1],
        rationale="Synthetic adjudicator selected A after reviewing the contrived fixture.",
        adjudication_confidence=AnnotationConfidence.HIGH,
    )
    adjudication = lock_adjudication(
        create_adjudication(
            annotation_a,
            annotation_b,
            adjudicator_id="adjudicator_01",
            decisions=[decision],
            created_at=now,
        ),
        locked_at=now,
    )
    splits = assign_group_aware_splits(
        [
            SplitCandidate(
                video_id=record.video_id,
                source_group_id=record.source_group_id,
                duration_seconds=record.duration_seconds,
                labels=[event.label for event in adjudication.final_events],
                scenario_tags=record.scenario_tags,
            )
        ],
        target_ratios={
            DatasetSplit.DEVELOPMENT: 0,
            DatasetSplit.VALIDATION: 0,
            DatasetSplit.TEST: 1,
        },
        seed=42,
    )
    annotations = {record.video_id: [annotation_a, annotation_b]}
    coverage = build_coverage_report(registry, annotations, [agreement])
    release = build_dataset_release(
        registry,
        splits,
        annotations,
        {record.video_id: adjudication},
        agreements=[agreement],
        created_at=now,
    )
    release_entry = release.videos[0]
    exported = export_adjudicated_annotation(
        adjudication,
        record,
        split=DatasetSplit.TEST,
        source_annotations=annotations[record.video_id],
        expected_ground_truth_sha256=release_entry.benchmark_ground_truth_sha256,
    )
    write_json_model(registry, output / "intake_registry.json")
    write_json_model(annotation_a, output / "annotations" / "annotator_a.json")
    write_json_model(annotation_b, output / "annotations" / "annotator_b.json")
    write_json_model(agreement, output / "agreement_report.json")
    write_json_model(adjudication, output / "adjudication.json")
    write_json_model(splits, output / "split_assignments.json")
    write_coverage_report(coverage, output)
    write_json_model(release, output / "dataset_release.json")
    write_json_model(exported, output / "benchmark_annotation.json")
    print(
        f"Synthetic lifecycle complete: {output}; quality_gate_passed={release.quality_gate_passed}"
    )
    return 0 if release.quality_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

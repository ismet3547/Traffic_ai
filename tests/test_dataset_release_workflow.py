from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from app.benchmark.models import AnnotationConfidence, DatasetSplit
from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.intake import DuplicateVideoError, register_video
from app.dataset.integrity import DatasetIntegrityError
from app.dataset.io import lock_annotation
from app.dataset.models import (
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    IntakeRegistry,
    PermissionStatus,
    SourceType,
    SplitCandidate,
    VideoIntakeRecord,
    VideoResolution,
)
from app.dataset.release import build_dataset_release, export_adjudicated_annotation
from app.dataset.reporting import build_coverage_report, coverage_markdown
from app.dataset.splitting import assign_group_aware_splits
from app.tools.run_synthetic_annotation_lifecycle import main as lifecycle_main

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SHA = "b" * 64


def record(*, verified: bool = True) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id="clip",
        source_group_id="session",
        source_type=SourceType.OWN_CAPTURE,
        source_reference="test fixture",
        acquisition_date=date(2026, 1, 1),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=False,
        benchmark_use_allowed=True,
        source_video_sha256=SHA,
        source_video_size_bytes=123,
        source_identity_verified=verified,
        duration_seconds=60,
        resolution=VideoResolution(width=1280, height=720),
        fps=30,
        original_filename="clip.mp4",
        scenario_tags=["daylight", "free_flow", "fixed_camera"],
    )


def annotation(annotator: str, event_id: str) -> DatasetAnnotation:
    return lock_annotation(
        DatasetAnnotation(
            video_id="clip",
            source_video_sha256=SHA,
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


def split_document(candidates: list[SplitCandidate] | None = None):
    return assign_group_aware_splits(
        candidates
        or [
            SplitCandidate(
                video_id="clip", source_group_id="session", duration_seconds=60
            )
        ],
        target_ratios={
            DatasetSplit.DEVELOPMENT: 0,
            DatasetSplit.VALIDATION: 0,
            DatasetSplit.TEST: 1,
        },
        seed=9,
    )


def test_source_group_never_crosses_splits_and_assignment_is_deterministic() -> None:
    candidates = [
        SplitCandidate(
            video_id="a",
            source_group_id="same",
            duration_seconds=10,
            labels=[DatasetLabel.LEGITIMATE_OVERTAKING],
            scenario_tags=["night"],
        ),
        SplitCandidate(
            video_id="b",
            source_group_id="same",
            duration_seconds=10,
            labels=[DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION],
            scenario_tags=["night"],
        ),
        SplitCandidate(
            video_id="c",
            source_group_id="other",
            duration_seconds=20,
            labels=[DatasetLabel.CONGESTION_LEFT_LANE_USE],
            scenario_tags=["dense_traffic"],
        ),
    ]
    first = assign_group_aware_splits(candidates, seed=22)
    second = assign_group_aware_splits(list(reversed(candidates)), seed=22)
    assert first == second
    same_splits = {
        item.split for item in first.assignments if item.source_group_id == "same"
    }
    assert len(same_splits) == 1


def test_split_balancing_never_breaks_group_isolation() -> None:
    candidates = [
        SplitCandidate(
            video_id=f"clip_{i}",
            source_group_id="one_recording",
            duration_seconds=10,
            labels=[DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION],
        )
        for i in range(6)
    ]
    result = assign_group_aware_splits(candidates)
    assert len({item.split for item in result.assignments}) == 1


def test_coverage_counts_actual_events_tags_duration_and_markdown() -> None:
    report = build_coverage_report(
        IntakeRegistry(videos=[record()]),
        {"clip": [annotation("a", "e1"), annotation("b", "e2")]},
    )
    assert report.total_clips == 1
    assert report.total_duration_seconds == 60
    assert report.label_counts[DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION.value] == 2
    assert report.scenario_tag_counts["daylight"] == 1
    assert "Total clips: 1" in coverage_markdown(report)


def test_release_contains_source_and_annotation_hashes() -> None:
    first, second = annotation("a", "a1"), annotation("b", "b1")
    adjudicated = lock_adjudication(
        create_adjudication(
            first, second, adjudicator_id="reviewer", decisions=[], created_at=NOW
        ),
        locked_at=NOW,
    )
    release = build_dataset_release(
        IntakeRegistry(videos=[record()]),
        split_document(),
        {"clip": [first, second]},
        {"clip": adjudicated},
        agreements=[adjudicated.agreement_report],
        created_at=NOW,
    )
    item = release.videos[0]
    assert item.source_video_sha256 == SHA
    assert set(item.annotation_hashes) == {"a", "b"}
    assert item.adjudicated_annotation_hash is not None
    assert release.quality_gate_passed


def test_unverified_source_identity_fails_quality_gate() -> None:
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[record(verified=False)]),
            split_document(),
            {"clip": []},
            {},
            created_at=NOW,
        )
    assert "SOURCE_VIDEO_IDENTITY_UNVERIFIED" in str(raised.value)


def test_nonadjudicated_test_ground_truth_fails_quality_gate() -> None:
    first, second = annotation("a", "a1"), annotation("b", "b1")
    with pytest.raises(DatasetIntegrityError) as raised:
        build_dataset_release(
            IntakeRegistry(videos=[record()]),
            split_document(),
            {"clip": [first, second]},
            {},
            created_at=NOW,
        )
    assert "ADJUDICATION_NOT_APPROVED" in str(raised.value)


def test_adjudicated_benchmark_export_is_deterministic() -> None:
    first, second = annotation("a", "a1"), annotation("b", "b1")
    artifact = lock_adjudication(
        create_adjudication(
            first, second, adjudicator_id="reviewer", decisions=[], created_at=NOW
        ),
        locked_at=NOW,
    )
    left = export_adjudicated_annotation(artifact, record(), split=DatasetSplit.TEST)
    right = export_adjudicated_annotation(artifact, record(), split=DatasetSplit.TEST)
    assert left.model_dump(mode="json") == right.model_dump(mode="json")
    assert left.events[0].vehicle_track_hint == "vehicle_1"


def test_unlocked_test_adjudication_cannot_export() -> None:
    first, second = annotation("a", "a1"), annotation("b", "b1")
    artifact = create_adjudication(
        first, second, adjudicator_id="reviewer", decisions=[], created_at=NOW
    )
    with pytest.raises(ValueError, match="locked"):
        export_adjudicated_annotation(artifact, record(), split=DatasetSplit.TEST)


def test_exact_duplicate_intake_requires_explicit_override(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"synthetic video bytes")
    monkeypatch.setattr(
        "app.dataset.intake.inspect_video",
        lambda _: (10.0, VideoResolution(width=10, height=10), 30.0),
    )
    registry, _, _ = register_video(
        IntakeRegistry(),
        source,
        video_id="a",
        source_group_id="group",
        source_type=SourceType.OTHER,
        source_reference="synthetic",
        acquisition_date=date(2026, 1, 1),
        permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=True,
        benchmark_use_allowed=True,
    )
    with pytest.raises(DuplicateVideoError, match="DUPLICATE_VIDEO_CONTENT"):
        register_video(
            registry,
            source,
            video_id="b",
            source_group_id="group",
            source_type=SourceType.OTHER,
            source_reference="synthetic",
            acquisition_date=date(2026, 1, 1),
            permission_status=PermissionStatus.VERIFIED,
            redistribution_allowed=True,
            benchmark_use_allowed=True,
        )


def test_full_synthetic_lifecycle_writes_passing_release(tmp_path) -> None:
    assert lifecycle_main(["--output", str(tmp_path)]) == 0
    release = json.loads(
        (tmp_path / "dataset_release.json").read_text(encoding="utf-8")
    )
    assert release["quality_gate_passed"] is True
    assert (tmp_path / "benchmark_annotation.json").is_file()
    assert (tmp_path / "dataset_coverage.md").is_file()

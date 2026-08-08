from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.benchmark.fingerprints import streaming_file_sha256
from app.benchmark.models import PredictionDocument, VersionMetadata
from app.benchmark.protocol import current_evaluation_protocol
from app.dataset.io import write_json_model
from app.dataset.models import CANONICAL_AGREEMENT_PROTOCOL, IntakeRegistry
from app.dataset.pilot import (
    MINI_PILOT_ACCURACY_WARNING,
    NO_REAL_VIDEO_STATUS,
    PilotArtifactLayout,
    PilotBaselineExistsError,
    PilotClipSelection,
    PilotManifest,
    PilotReviewBlockedError,
    authorize_posthoc_model_review,
    build_pilot_status,
    freeze_pilot_baseline,
    render_pilot_status,
)
from app.dataset.release import build_dataset_release, export_adjudicated_annotation
from tests.test_dataset_agreement_provenance import NOW, _bundle, _splits

FROZEN_AT = datetime(2026, 8, 8, tzinfo=timezone.utc)


def _manifest(root: Path, *, with_clip: bool) -> tuple[PilotManifest, Path]:
    manifest = PilotManifest(
        pilot_id="mini-pilot-test",
        agreement_protocol=CANONICAL_AGREEMENT_PROTOCOL,
        frozen_at=FROZEN_AT,
        clips=(
            [
                PilotClipSelection(
                    video_id="real_clip",
                    real_world_source_confirmed=True,
                    local_video_path="real_clip.mp4",
                    production_config_path="production.yaml",
                    annotation_duration_minutes={
                        "annotator_a": 10,
                        "annotator_b": 12,
                    },
                )
            ]
            if with_clip
            else []
        ),
        artifacts=PilotArtifactLayout(
            registry="registry.json",
            annotations_directory="annotations",
            agreements_directory="agreements",
            adjudications_directory="adjudications",
            ground_truth_directory="ground_truth",
            dataset_release="dataset_release.json",
            benchmark_manifest="benchmark_manifest.yaml",
            benchmark_run_directory="current_run",
            baseline_directory="pilot_baseline_0",
        ),
    )
    path = write_json_model(manifest, root / "pilot_manifest.json")
    return manifest, path


def _completed_workspace(root: Path):
    manifest, manifest_path = _manifest(root, with_clip=True)
    record, first, second, agreement, adjudication = _bundle("real_clip", 301)
    write_json_model(IntakeRegistry(videos=[record]), root / "registry.json")
    (root / "real_clip.mp4").write_bytes(b"synthetic test placeholder")
    (root / "production.yaml").write_text("detector: {}\n", encoding="utf-8")
    write_json_model(first, root / "annotations" / "annotator_a.json")
    write_json_model(second, root / "annotations" / "annotator_b.json")
    write_json_model(agreement, root / "agreements" / "real_clip.json")
    write_json_model(adjudication, root / "adjudications" / "real_clip.json")
    release = build_dataset_release(
        IntakeRegistry(videos=[record]),
        _splits([record]),
        {record.video_id: [first, second]},
        {record.video_id: adjudication},
        agreements=[agreement],
        created_at=NOW,
    )
    write_json_model(release, root / "dataset_release.json")
    ground_truth = export_adjudicated_annotation(
        adjudication,
        record,
        split=_splits([record]).assignments[0].split,
        source_annotations=[first, second],
    )
    write_json_model(ground_truth, root / "ground_truth" / "real_clip.json")
    prediction = PredictionDocument(
        video_id="real_clip",
        source_file=record.original_filename,
        source_video_sha256=record.source_video_sha256,
        source_video_size_bytes=record.source_video_size_bytes,
        versions=VersionMetadata(
            git_commit="a" * 40,
            detector_model_identifier="yolo-test.pt",
            tracker_identifier="ByteTrack",
        ),
    )
    prediction_path = write_json_model(
        prediction, root / "current_run" / "predictions" / "real_clip.json"
    )
    protocol = current_evaluation_protocol()
    report = {
        "benchmark_schema_version": "1.0",
        "synthetic": False,
        "per_video_metrics": {"real_clip": {"metrics": {}}},
        "overall_metrics": {
            "true_positives": 1,
            "false_positives": 2,
            "false_negatives": 3,
            "precision": 1 / 3,
            "recall": 0.25,
            "f1": 2 / 7,
            "false_positives_per_video_hour": 2.0,
        },
        "failures": [
            *[
                {
                    "failure_id": f"fp_{index:04d}",
                    "video_id": "real_clip",
                    "kind": "false_positive",
                    "timestamp_seconds": float(index),
                    "ground_truth": None,
                    "prediction": {
                        "event_id": f"prediction_{index}",
                        "start_seconds": float(index),
                        "end_seconds": float(index) + 1.0,
                    },
                }
                for index in range(1, 3)
            ],
            *[
                {
                    "failure_id": f"fn_{index:04d}",
                    "video_id": "real_clip",
                    "kind": "false_negative",
                    "timestamp_seconds": float(index + 2),
                    "ground_truth": {
                        "event_id": f"truth_{index}",
                        "start_seconds": float(index + 2),
                        "end_seconds": float(index + 3),
                    },
                    "prediction": None,
                }
                for index in range(1, 4)
            ],
        ],
        "reproducibility": {
            "git_commit": "a" * 40,
            "git_worktree_dirty": False,
            "resolved_config_hash_sha256": "b" * 64,
            "production_config_hash_sha256": "c" * 64,
            "dataset_fingerprint": "d" * 64,
            "dataset_identity_status": "verified",
            "evaluation_protocol": protocol.model_dump(mode="json"),
            "production_identifiers": {
                "real_clip": {
                    "detector_model_identifier": "yolo-test.pt",
                    "tracker_identifier": "ByteTrack",
                }
            },
            "prediction_cache_hashes_sha256": {
                "real_clip": streaming_file_sha256(prediction_path)
            },
        },
    }
    report_path = root / "current_run" / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest, manifest_path, release, report


def test_empty_pilot_reports_zero_counts_without_fake_completion(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _manifest(tmp_path, with_clip=False)
    write_json_model(IntakeRegistry(), tmp_path / "registry.json")

    status = build_pilot_status(manifest, manifest_path)

    assert status.real_pilot_status == NO_REAL_VIDEO_STATUS
    assert not status.pilot_executed
    assert not status.pilot_baseline_frozen
    assert not status.posthoc_model_review_allowed
    assert set(status.counts.model_dump().values()) == {0}
    assert [item.code for item in status.blockers] == [
        "NO_REAL_SOURCE_VIDEO_REGISTERED"
    ]
    assert "TP, FP, FN" in render_pilot_status(status)


def test_pilot_status_counts_only_valid_completed_artifacts(tmp_path: Path) -> None:
    manifest, manifest_path, _, _ = _completed_workspace(tmp_path)

    status = build_pilot_status(manifest, manifest_path)

    assert status.counts.model_dump() == {
        "selected_clips": 1,
        "registered_clips": 1,
        "real_world_confirmed_clips": 1,
        "total_duration_seconds": 60.0,
        "double_annotated_clips": 1,
        "agreement_ready_clips": 1,
        "adjudicated_clips": 1,
        "benchmark_exported_clips": 1,
        "inference_complete_clips": 1,
        "benchmark_complete_clips": 1,
    }
    assert not status.pilot_executed
    assert not status.posthoc_model_review_allowed
    assert {item.code for item in status.blockers} == {"PILOT_BASELINE_NOT_FROZEN"}
    assert status.annotation_effort_total_minutes == 22
    assert status.annotation_effort_mean_minutes_per_pass == 11


def test_locked_ground_truth_is_required_for_posthoc_model_review(
    tmp_path: Path,
) -> None:
    _, _, release, report = _completed_workspace(tmp_path)
    unlocked_video = release.videos[0].model_copy(
        update={"test_annotation_locked": False}
    )
    unlocked_release = release.model_copy(update={"videos": [unlocked_video]})

    with pytest.raises(PilotReviewBlockedError, match="not finalized and locked"):
        authorize_posthoc_model_review(
            unlocked_release, report, {unlocked_video.video_id}
        )


def test_pilot_baseline_is_frozen_once_and_metadata_is_preserved(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, _, _ = _completed_workspace(tmp_path)

    destination, metadata = freeze_pilot_baseline(
        manifest, manifest_path, frozen_at=FROZEN_AT
    )
    original = (destination / "baseline_metadata.json").read_bytes()

    assert metadata.baseline_id == "pilot_baseline_0"
    assert metadata.system_git_commit == "a" * 40
    assert metadata.detector_model_identifiers == ["yolo-test.pt"]
    assert metadata.tracker_identifiers == ["ByteTrack"]
    assert metadata.true_positives == 1
    assert metadata.false_positives == 2
    assert metadata.false_negatives == 3
    assert metadata.accuracy_warning == MINI_PILOT_ACCURACY_WARNING
    assert (destination / "provenance" / "dataset_release.json").is_file()
    status = build_pilot_status(manifest, manifest_path)
    assert status.pilot_baseline_frozen
    assert status.posthoc_model_review_allowed
    assert not status.pilot_executed
    assert "FAILURE_REVIEW_INCOMPLETE" in {item.code for item in status.blockers}
    rendered = render_pilot_status(status)
    assert "required=5, reviewed=0, missing=5" in rendered
    assert "Every FP and FN" in rendered
    with pytest.raises(PilotBaselineExistsError):
        freeze_pilot_baseline(manifest, manifest_path, frozen_at=FROZEN_AT)
    assert (destination / "baseline_metadata.json").read_bytes() == original


def test_pilot_status_and_rendering_are_deterministic(tmp_path: Path) -> None:
    manifest, manifest_path, _, _ = _completed_workspace(tmp_path)

    first = build_pilot_status(manifest, manifest_path)
    second = build_pilot_status(manifest, manifest_path)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert render_pilot_status(first) == render_pilot_status(second)


def test_legacy_completion_fields_cannot_bypass_failure_review(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, _, _ = _completed_workspace(tmp_path)
    freeze_pilot_baseline(manifest, manifest_path, frozen_at=FROZEN_AT)
    forged = manifest.model_copy(
        update={
            "failure_review_completed": True,
            "first_agreement_review_video_ids": ["real_clip"],
            "scale_up_recommendation": "GO",
        }
    )

    status = build_pilot_status(forged, manifest_path)

    assert not status.pilot_executed
    assert status.pilot_state.value == "FAILURE_REVIEW_REQUIRED"
    assert status.failure_review.required_count == 5
    assert status.failure_review.missing_count == 5
    assert {
        "FAILURE_REVIEW_INCOMPLETE",
        "LEGACY_PILOT_COMPLETION_FIELDS_IGNORED",
    }.issubset({item.code for item in status.blockers})

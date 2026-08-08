from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.dataset.pilot as pilot_module
from app.benchmark.fingerprints import streaming_file_sha256
from app.dataset.io import write_json_model
from app.dataset.pilot import (
    PilotIssueSeverity,
    PilotState,
    assess_local_source_identity,
    build_pilot_status,
)
from tests.test_pilot_terminal_state_integrity import _complete_pilot


def _different_bytes_same_size(original: bytes) -> bytes:
    replacement = bytearray(original)
    replacement[len(replacement) // 2] ^= 0xFF
    return bytes(replacement)


def _blocker_codes(status: object) -> set[str]:
    return {item.code for item in status.blockers}  # type: ignore[attr-defined]


def test_matching_local_source_is_verified_against_registry(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    record = workspace.registry.videos[0]
    source_path = tmp_path / f"{record.video_id}.mp4"

    assessment = assess_local_source_identity(source_path, record)

    assert assessment.present
    assert assessment.identity_verified
    assert assessment.reason_code is None
    assert assessment.actual_size_bytes == record.source_video_size_bytes
    assert assessment.actual_sha256 == record.source_video_sha256


def test_size_mismatch_fails_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _complete_pilot(tmp_path)
    record = workspace.registry.videos[0]
    source_path = tmp_path / f"{record.video_id}.mp4"
    source_path.write_bytes(workspace.source_bytes[record.video_id] + b"changed-size")

    def unexpected_hash(_: object) -> str:
        raise AssertionError("size mismatch must short-circuit SHA-256")

    monkeypatch.setattr(pilot_module, "streaming_file_sha256", unexpected_hash)
    assessment = assess_local_source_identity(source_path, record)

    assert assessment.reason_code == "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH"
    assert assessment.actual_sha256 is None
    assert "not_computed_size_mismatch" in assessment.details


def test_same_path_same_size_replacement_revokes_and_restoration_recovers_terminal(
    tmp_path: Path,
) -> None:
    workspace = _complete_pilot(tmp_path)
    valid_status = build_pilot_status(workspace.manifest, workspace.manifest_path)
    record = workspace.registry.videos[0]
    source_path = tmp_path / f"{record.video_id}.mp4"
    original = workspace.source_bytes[record.video_id]
    replacement = _different_bytes_same_size(original)
    registry_path = tmp_path / "registry.json"
    historical_paths = [
        tmp_path / "pilot_baseline_0" / "baseline_metadata.json",
        tmp_path / "pilot_baseline_0" / "benchmark_report.json",
        tmp_path / "failure_review.json",
        tmp_path / "first_agreement_review.json",
        tmp_path / "scale_up_decision.json",
    ]
    historical_hashes = {path: streaming_file_sha256(path) for path in historical_paths}
    registry_hash = streaming_file_sha256(registry_path)

    assert valid_status.pilot_state == PilotState.COMPLETE_GO
    assert valid_status.pilot_executed
    assert len(replacement) == len(original)
    assert source_path.name == f"{record.video_id}.mp4"
    source_path.write_bytes(replacement)

    corrupted_status = build_pilot_status(workspace.manifest, workspace.manifest_path)
    assessment = assess_local_source_identity(source_path, record)

    assert assessment.actual_size_bytes == record.source_video_size_bytes
    assert assessment.actual_sha256 != record.source_video_sha256
    assert assessment.reason_code == "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH"
    assert corrupted_status.pilot_state not in {
        PilotState.COMPLETE_GO,
        PilotState.COMPLETE_CONDITIONAL_GO,
        PilotState.COMPLETE_NO_GO,
    }
    assert not corrupted_status.pilot_executed
    assert "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH" in _blocker_codes(corrupted_status)
    assert corrupted_status.counts.inference_complete_clips == 3
    assert corrupted_status.scale_up_decision.valid
    assert all(
        streaming_file_sha256(path) == digest
        for path, digest in historical_hashes.items()
    )
    assert streaming_file_sha256(registry_path) == registry_hash

    source_path.write_bytes(original)
    restored_status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert restored_status.pilot_state == PilotState.COMPLETE_GO
    assert restored_status.pilot_executed
    assert "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH" not in _blocker_codes(restored_status)


def test_different_size_replacement_revokes_terminal(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    record = workspace.registry.videos[0]
    source_path = tmp_path / f"{record.video_id}.mp4"
    source_path.write_bytes(workspace.source_bytes[record.video_id][:-1])

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert status.pilot_state == PilotState.BASELINE_FROZEN
    assert not status.pilot_executed
    issue = next(
        item
        for item in status.blockers
        if item.code == "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH"
    )
    assert issue.severity == PilotIssueSeverity.BLOCKER
    assert "expected_size=" in issue.details
    assert "actual_size=" in issue.details


def test_missing_source_is_distinct_from_identity_mismatch(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    record = workspace.registry.videos[0]
    (tmp_path / f"{record.video_id}.mp4").unlink()

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    codes = _blocker_codes(status)
    assert "LOCAL_VIDEO_MISSING" in codes
    assert "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH" not in codes
    assert not status.pilot_executed


@pytest.mark.parametrize(
    ("identity_update", "hash_expected"),
    [
        ({"source_video_size_bytes": 999_999}, False),
        ({"source_video_sha256": "f" * 64}, True),
    ],
)
def test_wrong_registered_identity_fails_closed_without_registry_repair(
    tmp_path: Path,
    identity_update: dict[str, object],
    hash_expected: bool,
) -> None:
    workspace = _complete_pilot(tmp_path)
    original_record = workspace.registry.videos[0]
    wrong_record = original_record.model_copy(update=identity_update)
    registry = workspace.registry.model_copy(
        update={
            "videos": [
                wrong_record if item.video_id == wrong_record.video_id else item
                for item in workspace.registry.videos
            ]
        }
    )
    registry_path = write_json_model(registry, tmp_path / "registry.json")
    registry_hash_before_status = streaming_file_sha256(registry_path)
    source_path = tmp_path / f"{wrong_record.video_id}.mp4"

    assessment = assess_local_source_identity(source_path, wrong_record)
    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert assessment.reason_code == "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH"
    assert (assessment.actual_sha256 is not None) is hash_expected
    assert "LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH" in _blocker_codes(status)
    assert not status.pilot_executed
    assert streaming_file_sha256(registry_path) == registry_hash_before_status


def test_loaded_unverified_registry_identity_has_specific_blocker(
    tmp_path: Path,
) -> None:
    workspace = _complete_pilot(tmp_path)
    record = workspace.registry.videos[0]
    unverified = record.model_copy(update={"source_identity_verified": False})
    registry = workspace.registry.model_copy(
        update={
            "videos": [
                unverified if item.video_id == unverified.video_id else item
                for item in workspace.registry.videos
            ]
        }
    )
    write_json_model(registry, tmp_path / "registry.json")

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert "REGISTERED_SOURCE_IDENTITY_INVALID" in _blocker_codes(status)
    assert not status.pilot_executed


def test_malformed_registry_identity_fails_closed_at_registry_load(
    tmp_path: Path,
) -> None:
    workspace = _complete_pilot(tmp_path)
    registry_path = tmp_path / "registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    del payload["videos"][0]["source_video_sha256"]
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert "REGISTRY_INVALID" in _blocker_codes(status)
    assert not status.pilot_executed

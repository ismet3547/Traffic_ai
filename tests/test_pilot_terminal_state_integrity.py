from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.benchmark.fingerprints import canonical_sha256, streaming_file_sha256
from app.benchmark.models import PredictionDocument, VersionMetadata
from app.benchmark.protocol import current_evaluation_protocol
from app.dataset.io import write_json_model
from app.dataset.models import (
    CANONICAL_AGREEMENT_PROTOCOL,
    IntakeRegistry,
    PermissionStatus,
)
from app.dataset.pilot import (
    MINI_PILOT_ACCURACY_WARNING,
    TERMINAL_PILOT_STATES,
    PilotArtifactLayout,
    PilotClipSelection,
    PilotIssue,
    PilotIssueSeverity,
    PilotManifest,
    PilotState,
    PilotStatus,
    build_pilot_status,
    current_required_agreement_reports,
    freeze_pilot_baseline,
    render_pilot_status,
)
from app.dataset.pilot_review import (
    AgreementReviewAction,
    FailureCategory,
    FailureReviewDocument,
    FailureReviewEntry,
    FirstAgreementReviewDocument,
    FirstAgreementReviewSummary,
    ReviewSeverity,
    ScaleUpDecision,
    ScaleUpDecisionDocument,
    SystematicRisk,
    assess_failure_review,
    assess_first_agreement_review,
    build_failure_review_document,
    build_first_agreement_review_document,
    build_scale_up_decision_document,
    derive_required_failures,
    failure_review_content_hash,
    first_agreement_review_content_hash,
    load_baseline_review_identity,
    save_failure_review,
    save_first_agreement_review,
    save_scale_up_decision,
    scale_up_decision_content_hash,
)
from app.dataset.release import build_dataset_release, export_adjudicated_annotation
from tests.test_dataset_agreement_provenance import NOW, _bundle, _splits

FROZEN_AT = datetime(2026, 8, 8, tzinfo=timezone.utc)


@dataclass
class CompletePilotWorkspace:
    root: Path
    manifest: PilotManifest
    manifest_path: Path
    registry: IntakeRegistry
    failure_review: FailureReviewDocument
    agreement_review: FirstAgreementReviewDocument
    decision: ScaleUpDecisionDocument


def _complete_pilot(
    root: Path,
    decision: ScaleUpDecision = ScaleUpDecision.GO,
) -> CompletePilotWorkspace:
    bundles = [_bundle(f"clip_{index}", 500 + index) for index in range(3)]
    records = [item[0] for item in bundles]
    annotations = {item[0].video_id: [item[1], item[2]] for item in bundles}
    agreements = [item[3] for item in bundles]
    adjudications = {item[0].video_id: item[4] for item in bundles}
    registry = IntakeRegistry(videos=records)
    write_json_model(registry, root / "registry.json")
    production_configs: dict[str, object] = {}
    clips: list[PilotClipSelection] = []
    for record in records:
        video_path = root / f"{record.video_id}.mp4"
        video_path.write_bytes(b"synthetic test placeholder")
        config_path = root / f"{record.video_id}.yaml"
        config_path.write_text("detector: {}\n", encoding="utf-8")
        production_configs[record.video_id] = {"detector": {}}
        clips.append(
            PilotClipSelection(
                video_id=record.video_id,
                real_world_source_confirmed=True,
                local_video_path=video_path.name,
                production_config_path=config_path.name,
                annotation_duration_minutes={"annotator_a": 5, "annotator_b": 6},
            )
        )
    manifest = PilotManifest(
        pilot_id="terminal-state-test",
        agreement_protocol=CANONICAL_AGREEMENT_PROTOCOL,
        frozen_at=FROZEN_AT,
        clips=clips,
        first_agreement_review_count=3,
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
            failure_review="failure_review.json",
            first_agreement_review="first_agreement_review.json",
            scale_up_decision="scale_up_decision.json",
        ),
    )
    manifest_path = write_json_model(manifest, root / "pilot_manifest.json")
    for bundle in bundles:
        video_id = bundle[0].video_id
        write_json_model(bundle[1], root / "annotations" / f"{video_id}_a.json")
        write_json_model(bundle[2], root / "annotations" / f"{video_id}_b.json")
        write_json_model(bundle[3], root / "agreements" / f"{video_id}.json")
        write_json_model(bundle[4], root / "adjudications" / f"{video_id}.json")
    splits = _splits(records)
    release = build_dataset_release(
        registry,
        splits,
        annotations,
        adjudications,
        agreements=agreements,
        created_at=NOW,
    )
    release_path = write_json_model(release, root / "dataset_release.json")
    prediction_hashes: dict[str, str] = {}
    for bundle in bundles:
        record = bundle[0]
        ground_truth = export_adjudicated_annotation(
            bundle[4],
            record,
            split=splits.assignments[0].split,
            source_annotations=[bundle[1], bundle[2]],
        )
        write_json_model(
            ground_truth, root / "ground_truth" / f"{record.video_id}.json"
        )
        prediction = PredictionDocument(
            video_id=record.video_id,
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
            prediction,
            root / "current_run" / "predictions" / f"{record.video_id}.json",
        )
        prediction_hashes[record.video_id] = streaming_file_sha256(prediction_path)
    protocol = current_evaluation_protocol()
    failure = {
        "failure_id": "fp_0001",
        "video_id": records[0].video_id,
        "kind": "false_positive",
        "timestamp_seconds": 1.0,
        "ground_truth": None,
        "prediction": {
            "event_id": "prediction_failure_1",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
        },
    }
    report = {
        "benchmark_schema_version": "1.0",
        "synthetic": False,
        "per_video_metrics": {item.video_id: {"metrics": {}} for item in records},
        "overall_metrics": {
            "true_positives": 1,
            "false_positives": 1,
            "false_negatives": 0,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2 / 3,
            "false_positives_per_video_hour": 1.0,
        },
        "failures": [failure],
        "reproducibility": {
            "git_commit": "a" * 40,
            "git_worktree_dirty": False,
            "resolved_config_hash_sha256": "b" * 64,
            "production_config_hash_sha256": canonical_sha256(production_configs),
            "dataset_fingerprint": "d" * 64,
            "dataset_identity_status": "verified",
            "evaluation_fingerprint": "e" * 64,
            "evaluation_protocol": protocol.model_dump(mode="json"),
            "production_identifiers": {
                item.video_id: {
                    "detector_model_identifier": "yolo-test.pt",
                    "tracker_identifier": "ByteTrack",
                }
                for item in records
            },
            "prediction_cache_hashes_sha256": prediction_hashes,
        },
    }
    report_path = root / "current_run" / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_pilot_baseline(manifest, manifest_path, frozen_at=FROZEN_AT)
    identity, frozen_report = load_baseline_review_identity(root / "pilot_baseline_0")
    required_failures = derive_required_failures(identity, frozen_report)
    required = required_failures[0]
    failure_entry = FailureReviewEntry(
        failure_id=required.failure_id,
        failure_type=required.failure_type,
        video_id=required.video_id,
        prediction_event_id=required.prediction_event_id,
        ground_truth_event_id=required.ground_truth_event_id,
        timestamp_start=required.timestamp_start,
        timestamp_end=required.timestamp_end,
        suspected_failure_category=FailureCategory.UNKNOWN,
        reviewer_note="Reviewed exact synthetic integrity evidence.",
        severity=ReviewSeverity.MEDIUM,
        systematic_risk=SystematicRisk.UNKNOWN,
        proposed_action="Retain for regression testing.",
    )
    failure_review = build_failure_review_document(
        identity, [failure_entry], created_at=FROZEN_AT
    )
    save_failure_review(failure_review, root / "failure_review.json")
    failure_status = assess_failure_review(identity, required_failures, failure_review)
    required_agreements = current_required_agreement_reports(manifest, manifest_path)
    agreement_review = build_first_agreement_review_document(
        manifest.pilot_id,
        required_agreements,
        FirstAgreementReviewSummary(
            recurring_disagreement_categories=[],
            handbook_issue_ids=[],
            action_required=AgreementReviewAction.NONE,
            notes="All three required reports were reviewed.",
        ),
        created_at=FROZEN_AT,
    )
    save_first_agreement_review(agreement_review, root / "first_agreement_review.json")
    agreement_status = assess_first_agreement_review(
        manifest.pilot_id, required_agreements, agreement_review
    )
    scale_decision = build_scale_up_decision_document(
        identity,
        failure_review,
        agreement_review,
        failure_status,
        agreement_status,
        decision=decision,
        rationale="Evidence supports this synthetic terminal-state integrity decision.",
        conditions=(
            ["Collect more diverse real clips."]
            if decision == ScaleUpDecision.CONDITIONAL_GO
            else []
        ),
        known_blockers=(
            ["Resolve the recorded pilot blocker."]
            if decision == ScaleUpDecision.NO_GO
            else []
        ),
        known_limitations=[MINI_PILOT_ACCURACY_WARNING],
        created_at=FROZEN_AT,
    )
    save_scale_up_decision(scale_decision, root / "scale_up_decision.json")
    assert streaming_file_sha256(release_path) == identity.dataset_release_sha256
    return CompletePilotWorkspace(
        root=root,
        manifest=manifest,
        manifest_path=manifest_path,
        registry=registry,
        failure_review=failure_review,
        agreement_review=agreement_review,
        decision=scale_decision,
    )


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (ScaleUpDecision.GO, PilotState.COMPLETE_GO),
        (ScaleUpDecision.CONDITIONAL_GO, PilotState.COMPLETE_CONDITIONAL_GO),
        (ScaleUpDecision.NO_GO, PilotState.COMPLETE_NO_GO),
    ],
)
def test_fully_valid_decisions_reach_terminal_state(
    tmp_path: Path, decision: ScaleUpDecision, expected: PilotState
) -> None:
    workspace = _complete_pilot(tmp_path, decision)

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert status.pilot_state == expected
    assert status.pilot_executed
    assert not status.blockers


def test_real_world_source_unconfirmed_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    clips = [
        workspace.manifest.clips[0].model_copy(
            update={"real_world_source_confirmed": False}
        ),
        *workspace.manifest.clips[1:],
    ]
    current = workspace.manifest.model_copy(update={"clips": clips})

    status = build_pilot_status(current, workspace.manifest_path)

    _assert_revoked(status, "REAL_WORLD_SOURCE_NOT_CONFIRMED")
    assert status.scale_up_decision.valid
    assert status.scale_up_decision.decision == ScaleUpDecision.GO


def test_benchmark_use_disallowed_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    changed = workspace.registry.videos[0].model_copy(
        update={"benchmark_use_allowed": False}
    )
    write_json_model(
        workspace.registry.model_copy(
            update={"videos": [changed, *workspace.registry.videos[1:]]}
        ),
        workspace.root / "registry.json",
    )

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "BENCHMARK_USE_NOT_ALLOWED")


def test_permission_revocation_and_restoration_is_fail_closed(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    original = workspace.registry.videos[0]
    revoked = original.model_copy(
        update={"license_or_permission_status": PermissionStatus.UNKNOWN}
    )
    write_json_model(
        workspace.registry.model_copy(
            update={"videos": [revoked, *workspace.registry.videos[1:]]}
        ),
        workspace.root / "registry.json",
    )

    revoked_status = build_pilot_status(workspace.manifest, workspace.manifest_path)
    _assert_revoked(revoked_status, "PERMISSION_NOT_VERIFIED")
    assert revoked_status.scale_up_decision.valid
    write_json_model(workspace.registry, workspace.root / "registry.json")
    restored_status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert restored_status.pilot_state == PilotState.COMPLETE_GO
    assert restored_status.pilot_executed


def test_protocol_mismatch_and_restoration_is_fail_closed(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    incompatible = workspace.manifest.model_copy(update={"handbook_version": "2.0"})

    mismatched = build_pilot_status(incompatible, workspace.manifest_path)
    _assert_revoked(mismatched, "PILOT_PROTOCOL_FREEZE_MISMATCH")
    assert mismatched.scale_up_decision.valid
    restored = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert restored.pilot_state == PilotState.COMPLETE_GO
    assert restored.pilot_executed


def test_missing_required_local_video_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    (workspace.root / "clip_0.mp4").unlink()

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "LOCAL_VIDEO_MISSING")


def test_stale_current_production_config_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    (workspace.root / "clip_0.yaml").write_text(
        "detector:\n  confidence: 0.99\n", encoding="utf-8"
    )

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "PILOT_BASELINE_STALE")


def test_invalid_frozen_baseline_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    report_path = workspace.root / "pilot_baseline_0" / "benchmark_report.json"
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "PILOT_BASELINE_INVALID")


def test_stale_failure_review_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    review = workspace.failure_review
    stale = review.model_copy(
        update={"benchmark_report_sha256": "9" * 64, "content_sha256": "0" * 64}
    )
    stale = stale.model_copy(
        update={"content_sha256": failure_review_content_hash(stale)}
    )
    save_failure_review(stale, workspace.root / "failure_review.json")

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "STALE_FAILURE_REVIEW")


def test_stale_agreement_review_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    review = workspace.agreement_review
    revised_ref = review.agreement_reports[0].model_copy(
        update={"agreement_content_sha256": "8" * 64}
    )
    stale = review.model_copy(
        update={
            "agreement_reports": [revised_ref, *review.agreement_reports[1:]],
            "content_sha256": "0" * 64,
        }
    )
    stale = stale.model_copy(
        update={"content_sha256": first_agreement_review_content_hash(stale)}
    )
    save_first_agreement_review(stale, workspace.root / "first_agreement_review.json")

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "STALE_AGREEMENT_REVIEW")


def test_stale_scale_up_decision_revokes_completion(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    decision = workspace.decision
    stale = decision.model_copy(
        update={"failure_review_hash": "7" * 64, "content_sha256": "0" * 64}
    )
    stale = stale.model_copy(
        update={"content_sha256": scale_up_decision_content_hash(stale)}
    )
    save_scale_up_decision(stale, workspace.root / "scale_up_decision.json")

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    _assert_revoked(status, "STALE_SCALE_UP_DECISION")


def test_warning_only_does_not_revoke_terminal_state(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)

    status = build_pilot_status(workspace.manifest, workspace.manifest_path)

    assert status.pilot_state == PilotState.COMPLETE_GO
    assert status.pilot_executed
    assert not status.blockers
    assert [item.code for item in status.warnings] == ["MINI_PILOT_ACCURACY_WARNING"]
    assert status.warnings[0].severity == PilotIssueSeverity.WARNING


def test_unknown_issue_codes_fail_closed() -> None:
    issue = PilotIssue(code="FUTURE_INTEGRITY_FAILURE", details="New integrity gate.")
    forged_warning = PilotIssue(
        code="PERMISSION_NOT_VERIFIED",
        details="Caller tried to downgrade this issue.",
        severity=PilotIssueSeverity.WARNING,
    )

    assert issue.severity == PilotIssueSeverity.BLOCKER
    assert forged_warning.severity == PilotIssueSeverity.BLOCKER
    with pytest.raises(ValidationError, match="frozen"):
        forged_warning.severity = PilotIssueSeverity.INFO  # type: ignore[misc]


def test_terminal_and_executed_invariants_are_enforced(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    status = build_pilot_status(workspace.manifest, workspace.manifest_path)
    payload = status.model_dump(mode="json")
    payload["blockers"] = [
        PilotIssue(code="PERMISSION_NOT_VERIFIED", details="Revoked.").model_dump(
            mode="json"
        )
    ]

    with pytest.raises(ValueError, match="terminal pilot state"):
        PilotStatus.model_validate(payload)

    payload["pilot_state"] = PilotState.BASELINE_FROZEN.value
    payload["pilot_executed"] = True
    with pytest.raises(ValueError, match="pilot_executed"):
        PilotStatus.model_validate(payload)


def test_legacy_completion_fields_cannot_override_current_blocker(
    tmp_path: Path,
) -> None:
    workspace = _complete_pilot(tmp_path)
    clips = [
        workspace.manifest.clips[0].model_copy(
            update={"real_world_source_confirmed": False}
        ),
        *workspace.manifest.clips[1:],
    ]
    forged = workspace.manifest.model_copy(
        update={
            "clips": clips,
            "failure_review_completed": True,
            "first_agreement_review_video_ids": [item.video_id for item in clips],
            "scale_up_recommendation": "GO",
        }
    )

    status = build_pilot_status(forged, workspace.manifest_path)

    _assert_revoked(status, "REAL_WORLD_SOURCE_NOT_CONFIRMED")
    assert "LEGACY_PILOT_COMPLETION_FIELDS_IGNORED" in {
        item.code for item in status.information
    }


def test_status_output_separates_blockers_warnings_and_information(
    tmp_path: Path,
) -> None:
    workspace = _complete_pilot(tmp_path)
    clips = [
        workspace.manifest.clips[0].model_copy(
            update={"real_world_source_confirmed": False}
        ),
        *workspace.manifest.clips[1:],
    ]
    current = workspace.manifest.model_copy(
        update={"clips": clips, "failure_review_completed": True}
    )
    status = build_pilot_status(current, workspace.manifest_path)

    rendered = render_pilot_status(status)

    assert "## Blockers" in rendered
    assert "`REAL_WORLD_SOURCE_NOT_CONFIRMED`" in rendered
    assert "## Warnings" in rendered
    assert "`MINI_PILOT_ACCURACY_WARNING`" in rendered
    assert "## Information" in rendered
    assert "`LEGACY_PILOT_COMPLETION_FIELDS_IGNORED`" in rendered


def test_every_built_status_obeys_terminal_invariant(tmp_path: Path) -> None:
    workspace = _complete_pilot(tmp_path)
    complete = build_pilot_status(workspace.manifest, workspace.manifest_path)
    blocked_manifest = workspace.manifest.model_copy(
        update={
            "clips": [
                workspace.manifest.clips[0].model_copy(
                    update={"real_world_source_confirmed": False}
                ),
                *workspace.manifest.clips[1:],
            ]
        }
    )
    blocked = build_pilot_status(blocked_manifest, workspace.manifest_path)

    for status in (complete, blocked):
        if status.pilot_state in TERMINAL_PILOT_STATES:
            assert not status.blockers
        if status.pilot_executed:
            assert status.pilot_state in TERMINAL_PILOT_STATES
            assert not status.blockers


def _assert_revoked(status: PilotStatus, code: str) -> None:
    assert status.pilot_state not in TERMINAL_PILOT_STATES
    assert not status.pilot_executed
    assert code in {item.code for item in status.blockers}

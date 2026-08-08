"""Run deterministic in-memory Phase 4.3.1 evidence-integrity scenarios."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import cast

from app.benchmark.fingerprints import canonical_sha256
from app.dataset.models import DisagreementType
from app.dataset.pilot import PilotIssue, PilotStageCounts, derive_pilot_state
from app.dataset.pilot_review import (
    AgreementReportReference,
    AgreementReviewAction,
    BaselineReviewIdentity,
    FailureCategory,
    FailureReviewEntry,
    FirstAgreementReviewSummary,
    RequiredFailure,
    ReviewSeverity,
    ScaleUpDecision,
    SystematicRisk,
    assess_failure_review,
    assess_first_agreement_review,
    assess_scale_up_decision,
    build_failure_review_document,
    build_first_agreement_review_document,
    build_scale_up_decision_document,
    derive_required_failures,
)

NOW = datetime(2026, 8, 8, tzinfo=timezone.utc)


def main() -> int:
    report = _report()
    identity = _identity(canonical_sha256(report))
    required = derive_required_failures(identity, report)

    scenario_a = assess_failure_review(identity, required, None)

    entries = [_entry(item) for item in required]
    failure_review = build_failure_review_document(identity, entries, created_at=NOW)
    scenario_b = assess_failure_review(identity, required, failure_review)

    required_agreements = _agreement_references()
    agreement_review = build_first_agreement_review_document(
        identity.pilot_id,
        required_agreements,
        FirstAgreementReviewSummary(
            recurring_disagreement_categories=[DisagreementType.BOUNDARY_DISAGREEMENT],
            handbook_issue_ids=[],
            action_required=AgreementReviewAction.NONE,
            notes="Synthetic integrity scenario: no handbook revision required.",
        ),
        created_at=NOW,
    )
    scenario_c = assess_first_agreement_review(
        identity.pilot_id, required_agreements, agreement_review
    )

    complete_counts = PilotStageCounts(
        selected_clips=3,
        registered_clips=3,
        real_world_confirmed_clips=3,
        total_duration_seconds=180.0,
        double_annotated_clips=3,
        agreement_ready_clips=3,
        adjudicated_clips=3,
        benchmark_exported_clips=3,
        inference_complete_clips=3,
        benchmark_complete_clips=3,
    )
    no_decision = assess_scale_up_decision(
        identity,
        scenario_a,
        scenario_c,
        None,
        agreement_review,
        None,
        current_dataset_release_sha256=identity.dataset_release_sha256,
    )
    state_a = derive_pilot_state(
        complete_counts,
        scenario_c,
        True,
        scenario_a,
        no_decision,
        has_completion_blocker=False,
    )

    decision = build_scale_up_decision_document(
        identity,
        failure_review,
        agreement_review,
        scenario_b,
        scenario_c,
        decision=ScaleUpDecision.GO,
        rationale="Synthetic scenario verifies evidence binding; it is not pilot evidence.",
        known_limitations=[
            "Synthetic integrity fixture; no real-world performance claim."
        ],
        created_at=NOW,
    )
    scenario_d = assess_scale_up_decision(
        identity,
        scenario_b,
        scenario_c,
        failure_review,
        agreement_review,
        decision,
        current_dataset_release_sha256=identity.dataset_release_sha256,
    )
    state_d = derive_pilot_state(
        complete_counts,
        scenario_c,
        True,
        scenario_b,
        scenario_d,
        has_completion_blocker=False,
    )

    changed_identity = _identity("9" * 64)
    changed_failure = assess_failure_review(changed_identity, required, failure_review)
    scenario_e = assess_scale_up_decision(
        changed_identity,
        changed_failure,
        scenario_c,
        failure_review,
        agreement_review,
        decision,
        current_dataset_release_sha256=changed_identity.dataset_release_sha256,
    )
    state_e = derive_pilot_state(
        complete_counts,
        scenario_c,
        True,
        changed_failure,
        scenario_e,
        has_completion_blocker=False,
    )
    permission_issue = PilotIssue(
        code="PERMISSION_NOT_VERIFIED",
        details="Synthetic current-eligibility revocation.",
    )
    protocol_issue = PilotIssue(
        code="PILOT_PROTOCOL_FREEZE_MISMATCH",
        details="Synthetic current-protocol mismatch.",
    )
    permission_revoked_state = derive_pilot_state(
        complete_counts,
        scenario_c,
        True,
        scenario_b,
        scenario_d,
        has_completion_blocker=True,
    )
    permission_restored_state = derive_pilot_state(
        complete_counts,
        scenario_c,
        True,
        scenario_b,
        scenario_d,
        has_completion_blocker=False,
    )
    protocol_mismatch_state = derive_pilot_state(
        complete_counts,
        scenario_c,
        True,
        scenario_b,
        scenario_d,
        has_completion_blocker=True,
    )
    output = {
        "synthetic": True,
        "warning": "These are integrity fixtures, not real pilot evidence.",
        "artifact_hashes": {
            "benchmark_report_sha256": identity.benchmark_report_sha256,
            "failure_review_sha256": failure_review.content_sha256,
            "first_agreement_review_sha256": agreement_review.content_sha256,
            "scale_up_decision_sha256": decision.content_sha256,
        },
        "scenarios": {
            "A": {
                "expected_state": "FAILURE_REVIEW_REQUIRED",
                "actual_state": state_a.value,
                "required": scenario_a.required_count,
                "missing": scenario_a.missing_count,
            },
            "B": {
                "expected_failure_review_complete": True,
                "actual_failure_review_complete": scenario_b.complete,
                "reviewed": scenario_b.reviewed_count,
            },
            "C": {
                "expected_agreement_review_complete": True,
                "actual_agreement_review_complete": scenario_c.complete,
                "required_reports": scenario_c.required_report_count,
            },
            "D": {
                "expected_state": "COMPLETE_GO",
                "actual_state": state_d.value,
            },
            "E": {
                "expected_state": "FAILURE_REVIEW_REQUIRED",
                "actual_state": state_e.value,
                "old_go_stale": scenario_e.stale,
                "failure_reasons": changed_failure.reason_codes,
                "decision_reasons": scenario_e.reason_codes,
            },
        },
        "terminal_state_lifecycle": {
            "A": {
                "expected_state": "COMPLETE_GO",
                "actual_state": state_d.value,
                "blockers": [],
            },
            "B": {
                "expected_state": "BASELINE_FROZEN",
                "actual_state": permission_revoked_state.value,
                "blockers": [permission_issue.code],
            },
            "C": {
                "expected_state": "COMPLETE_GO",
                "actual_state": permission_restored_state.value,
                "blockers": [],
            },
            "D": {
                "expected_state": "BASELINE_FROZEN",
                "actual_state": protocol_mismatch_state.value,
                "blockers": [protocol_issue.code],
            },
            "E": {
                "expected_state": "COMPLETE_GO",
                "actual_state": permission_restored_state.value,
                "blockers": [],
            },
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if _passed(output) else 1


def _identity(report_hash: str) -> BaselineReviewIdentity:
    return BaselineReviewIdentity(
        pilot_id="synthetic-pilot-review",
        baseline_id="pilot_baseline_0",
        benchmark_report_sha256=report_hash,
        benchmark_report_file_sha256="1" * 64,
        dataset_fingerprint="2" * 64,
        evaluation_fingerprint="3" * 64,
        system_git_commit="4" * 40,
        dataset_release_sha256="5" * 64,
    )


def _report() -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index in range(3):
        failures.append(
            {
                "failure_id": f"fp_{index:04d}",
                "video_id": f"clip_{index}",
                "kind": "false_positive",
                "timestamp_seconds": float(index),
                "prediction": {
                    "event_id": f"prediction_{index}",
                    "start_seconds": float(index),
                    "end_seconds": float(index + 1),
                },
                "ground_truth": None,
            }
        )
    for index in range(2):
        failures.append(
            {
                "failure_id": f"fn_{index:04d}",
                "video_id": f"clip_{index}",
                "kind": "false_negative",
                "timestamp_seconds": float(index + 3),
                "prediction": None,
                "ground_truth": {
                    "event_id": f"truth_{index}",
                    "start_seconds": float(index + 3),
                    "end_seconds": float(index + 4),
                },
            }
        )
    return {
        "overall_metrics": {"false_positives": 3, "false_negatives": 2},
        "failures": failures,
    }


def _entry(item: RequiredFailure) -> FailureReviewEntry:
    return FailureReviewEntry(
        failure_id=item.failure_id,
        failure_type=item.failure_type,
        video_id=item.video_id,
        prediction_event_id=item.prediction_event_id,
        ground_truth_event_id=item.ground_truth_event_id,
        timestamp_start=item.timestamp_start,
        timestamp_end=item.timestamp_end,
        suspected_failure_category=FailureCategory.UNKNOWN,
        reviewer_note="Synthetic scenario records one exact human-review-shaped item.",
        severity=ReviewSeverity.MEDIUM,
        systematic_risk=SystematicRisk.UNKNOWN,
        proposed_action="Use only to validate evidence lifecycle behavior.",
    )


def _agreement_references() -> list[AgreementReportReference]:
    return [
        AgreementReportReference(
            video_id=f"clip_{index}",
            agreement_id=f"{index + 10:064x}",
            agreement_content_sha256=f"{index + 20:064x}",
        )
        for index in range(3)
    ]


def _passed(output: dict[str, object]) -> bool:
    scenarios = cast(dict[str, dict[str, object]], output["scenarios"])
    scenario_a = scenarios["A"]
    scenario_b = scenarios["B"]
    scenario_c = scenarios["C"]
    scenario_d = scenarios["D"]
    scenario_e = scenarios["E"]
    review_scenarios_passed = bool(
        scenario_a["expected_state"] == scenario_a["actual_state"]
        and scenario_b["actual_failure_review_complete"] is True
        and scenario_c["actual_agreement_review_complete"] is True
        and scenario_d["expected_state"] == scenario_d["actual_state"]
        and scenario_e["expected_state"] == scenario_e["actual_state"]
        and scenario_e["old_go_stale"] is True
    )
    lifecycle = cast(dict[str, dict[str, object]], output["terminal_state_lifecycle"])
    lifecycle_passed = all(
        item["expected_state"] == item["actual_state"] for item in lifecycle.values()
    )
    return review_scenarios_passed and lifecycle_passed


if __name__ == "__main__":
    raise SystemExit(main())

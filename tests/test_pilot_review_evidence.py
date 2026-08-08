from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.dataset.pilot import PilotStageCounts, PilotState, derive_pilot_state
from app.dataset.pilot_review import (
    AgreementReportReference,
    AgreementReviewAction,
    BaselineReviewIdentity,
    FailureCategory,
    FailureReviewDocument,
    FailureReviewEntry,
    FirstAgreementReviewDocument,
    FirstAgreementReviewSummary,
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
    failure_review_content_hash,
    first_agreement_review_content_hash,
    required_agreement_reports,
    save_failure_review,
    scale_up_decision_content_hash,
)
from tests.test_dataset_agreement_provenance import _bundle

NOW = datetime(2026, 8, 8, tzinfo=timezone.utc)


def _identity(*, report_hash: str = "a" * 64) -> BaselineReviewIdentity:
    return BaselineReviewIdentity(
        pilot_id="mini-pilot-test",
        baseline_id="pilot_baseline_0",
        benchmark_report_sha256=report_hash,
        benchmark_report_file_sha256="b" * 64,
        dataset_fingerprint="c" * 64,
        evaluation_fingerprint="d" * 64,
        system_git_commit="e" * 40,
        dataset_release_sha256="f" * 64,
    )


def _report(fp: int, fn: int) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index in range(fp):
        failures.append(
            {
                "failure_id": f"fp_{index:04d}",
                "video_id": f"clip_{index % 2}",
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
    for index in range(fn):
        failures.append(
            {
                "failure_id": f"fn_{index:04d}",
                "video_id": f"clip_{index % 2}",
                "kind": "false_negative",
                "timestamp_seconds": float(index + fp),
                "prediction": None,
                "ground_truth": {
                    "event_id": f"truth_{index}",
                    "start_seconds": float(index + fp),
                    "end_seconds": float(index + fp + 1),
                },
            }
        )
    return {
        "overall_metrics": {"false_positives": fp, "false_negatives": fn},
        "failures": failures,
    }


def _entries(
    identity: BaselineReviewIdentity, fp: int, fn: int
) -> list[FailureReviewEntry]:
    required = derive_required_failures(identity, _report(fp, fn))
    return [
        FailureReviewEntry(
            failure_id=item.failure_id,
            failure_type=item.failure_type,
            video_id=item.video_id,
            prediction_event_id=item.prediction_event_id,
            ground_truth_event_id=item.ground_truth_event_id,
            timestamp_start=item.timestamp_start,
            timestamp_end=item.timestamp_end,
            suspected_failure_category=FailureCategory.UNKNOWN,
            reviewer_note="Human reviewed the representative evidence.",
            severity=ReviewSeverity.MEDIUM,
            systematic_risk=SystematicRisk.UNKNOWN,
            proposed_action="Compare with similar development-set failures.",
        )
        for item in required
    ]


def _agreement_refs(count: int = 3) -> list[AgreementReportReference]:
    return [
        AgreementReportReference(
            video_id=f"clip_{index}",
            agreement_id=f"{index + 1:064x}",
            agreement_content_sha256=f"{index + 101:064x}",
        )
        for index in range(count)
    ]


def _agreement_document(
    refs: list[AgreementReportReference] | None = None,
) -> FirstAgreementReviewDocument:
    return build_first_agreement_review_document(
        "mini-pilot-test",
        refs or _agreement_refs(),
        FirstAgreementReviewSummary(
            recurring_disagreement_categories=["boundary_disagreement"],
            handbook_issue_ids=[],
            action_required=AgreementReviewAction.NONE,
            notes="No recurring ambiguity requires a handbook change.",
        ),
        created_at=NOW,
    )


def _complete_failure_review(
    identity: BaselineReviewIdentity, fp: int = 3, fn: int = 2
) -> tuple[list[object], FailureReviewDocument]:
    required = derive_required_failures(identity, _report(fp, fn))
    document = build_failure_review_document(
        identity, _entries(identity, fp, fn), created_at=NOW
    )
    return required, document


def test_zero_failure_baseline_auto_completes_without_fake_review() -> None:
    identity = _identity()
    required = derive_required_failures(identity, _report(0, 0))

    coverage = assess_failure_review(identity, required, None)

    assert coverage.complete
    assert coverage.required_count == 0
    assert coverage.coverage_ratio == 1
    assert coverage.message == "No FP/FN review required for this baseline."


def test_complete_exact_failure_review_is_complete() -> None:
    identity = _identity()
    required, document = _complete_failure_review(identity)

    coverage = assess_failure_review(identity, required, document)

    assert coverage.complete
    assert coverage.reviewed_count == coverage.required_count == 5


def test_missing_one_failure_is_incomplete() -> None:
    identity = _identity()
    required = derive_required_failures(identity, _report(3, 2))
    document = build_failure_review_document(
        identity, _entries(identity, 3, 2)[:-1], created_at=NOW
    )

    coverage = assess_failure_review(identity, required, document)

    assert not coverage.complete
    assert coverage.missing_count == 1
    assert coverage.reviewed_count == 4


def test_duplicate_review_does_not_inflate_coverage() -> None:
    identity = _identity()
    required = derive_required_failures(identity, _report(5, 5))
    reviews = _entries(identity, 5, 5)[:-1]
    reviews.append(reviews[0])
    document = build_failure_review_document(identity, reviews, created_at=NOW)

    coverage = assess_failure_review(identity, required, document)

    assert not coverage.complete
    assert coverage.reviewed_count == 9
    assert coverage.missing_count == 1
    assert coverage.duplicate_count == 1
    assert "DUPLICATE_FAILURE_REVIEW" in coverage.reason_codes


def test_unknown_failure_review_is_invalid() -> None:
    identity = _identity()
    required, document = _complete_failure_review(identity)
    forged = document.reviews[0].model_copy(update={"failure_id": "9" * 64})
    candidate = build_failure_review_document(
        identity, [*document.reviews, forged], created_at=NOW
    )

    coverage = assess_failure_review(identity, required, candidate)

    assert not coverage.complete
    assert coverage.unknown_count == 1
    assert "UNKNOWN_FAILURE_REVIEW" in coverage.reason_codes


def test_stale_benchmark_hash_invalidates_failure_review() -> None:
    original = _identity(report_hash="1" * 64)
    required, document = _complete_failure_review(original)
    current = _identity(report_hash="2" * 64)

    coverage = assess_failure_review(current, required, document)

    assert not coverage.complete
    assert coverage.stale_count == 5
    assert "STALE_FAILURE_REVIEW" in coverage.reason_codes


def test_failure_ids_are_deterministic_and_not_source_list_ids() -> None:
    identity = _identity()
    first = derive_required_failures(identity, _report(2, 1))
    changed_source_ids = _report(2, 1)
    for item in changed_source_ids["failures"]:  # type: ignore[index]
        item["failure_id"] = "renumbered"  # type: ignore[index]
    second = derive_required_failures(identity, changed_source_ids)

    assert [item.failure_id for item in first] == [item.failure_id for item in second]
    assert all(len(item.failure_id) == 64 for item in first)


def test_exact_coverage_counts_separate_unknown_duplicate_and_missing() -> None:
    identity = _identity()
    required = derive_required_failures(identity, _report(3, 2))
    reviews = _entries(identity, 3, 2)[:-1]
    reviews.extend([reviews[0], reviews[0].model_copy(update={"failure_id": "8" * 64})])
    document = build_failure_review_document(identity, reviews, created_at=NOW)

    coverage = assess_failure_review(identity, required, document)

    assert coverage.model_dump()["required_count"] == 5
    assert coverage.reviewed_count == 4
    assert coverage.missing_count == 1
    assert coverage.duplicate_count == 1
    assert coverage.unknown_count == 1


def test_first_agreement_review_exact_required_reports_is_complete() -> None:
    required = _agreement_refs(3)
    document = _agreement_document(required)

    status = assess_first_agreement_review("mini-pilot-test", required, document)

    assert status.complete
    assert status.reviewed_report_count == status.required_report_count == 3


def test_required_agreement_reports_are_first_n_by_video_id() -> None:
    reports = [
        _bundle(video_id, 100 + index)[3]
        for index, video_id in enumerate(
            ["clip_e", "clip_b", "clip_d", "clip_a", "clip_c"]
        )
    ]

    required = required_agreement_reports(list(reversed(reports)), 3)

    assert [item.video_id for item in required] == ["clip_a", "clip_b", "clip_c"]


def test_missing_required_agreement_report_is_incomplete() -> None:
    required = _agreement_refs(5)
    complete = _agreement_document(required)
    partial = complete.model_copy(
        update={
            "agreement_reports": complete.agreement_reports[:3],
            "content_sha256": "0" * 64,
        }
    )
    partial = partial.model_copy(
        update={"content_sha256": first_agreement_review_content_hash(partial)}
    )

    status = assess_first_agreement_review("mini-pilot-test", required, partial)

    assert not status.complete
    assert status.missing_count == 2


def test_agreement_report_revision_makes_review_stale() -> None:
    required = _agreement_refs()
    document = _agreement_document(required)
    revised = [
        required[0].model_copy(update={"agreement_content_sha256": "7" * 64}),
        *required[1:],
    ]

    status = assess_first_agreement_review("mini-pilot-test", revised, document)

    assert not status.complete
    assert status.stale
    assert "STALE_AGREEMENT_REVIEW" in status.reason_codes


def test_handbook_revision_requires_issue_linkage() -> None:
    with pytest.raises(ValueError, match="handbook_issue_id"):
        FirstAgreementReviewSummary(
            recurring_disagreement_categories=["label_disagreement"],
            handbook_issue_ids=[],
            action_required=AgreementReviewAction.HANDBOOK_REVISION,
            notes="The ontology wording needs a revision.",
        )


def test_valid_evidence_bound_scale_up_decision_is_accepted() -> None:
    identity = _identity()
    required, failure_document = _complete_failure_review(identity)
    failure_status = assess_failure_review(identity, required, failure_document)
    agreement_document = _agreement_document()
    agreement_status = assess_first_agreement_review(
        identity.pilot_id, _agreement_refs(), agreement_document
    )
    decision = build_scale_up_decision_document(
        identity,
        failure_document,
        agreement_document,
        failure_status,
        agreement_status,
        decision=ScaleUpDecision.GO,
        rationale="The evidence supports a controlled larger data collection.",
        known_limitations=["Tiny sample; no production accuracy claim."],
        created_at=NOW,
    )

    result = assess_scale_up_decision(
        identity,
        failure_status,
        agreement_status,
        failure_document,
        agreement_document,
        decision,
        current_dataset_release_sha256=identity.dataset_release_sha256,
    )

    assert result.valid
    assert result.decision == ScaleUpDecision.GO


def test_go_with_incomplete_failure_review_is_rejected() -> None:
    identity = _identity()
    required, failure_document = _complete_failure_review(identity)
    agreement_document = _agreement_document()
    agreement_status = assess_first_agreement_review(
        identity.pilot_id, _agreement_refs(), agreement_document
    )
    complete = assess_failure_review(identity, required, failure_document)
    decision = build_scale_up_decision_document(
        identity,
        failure_document,
        agreement_document,
        complete,
        agreement_status,
        decision=ScaleUpDecision.GO,
        rationale="A human rationale is present but evidence will be made incomplete.",
        created_at=NOW,
    )
    partial_document = build_failure_review_document(
        identity, failure_document.reviews[:-1], created_at=NOW
    )
    partial = assess_failure_review(identity, required, partial_document)

    result = assess_scale_up_decision(
        identity,
        partial,
        agreement_status,
        partial_document,
        agreement_document,
        decision,
        current_dataset_release_sha256=identity.dataset_release_sha256,
    )

    assert not result.valid
    assert "SCALE_UP_DECISION_FAILURE_REVIEW_INCOMPLETE" in result.reason_codes


def test_decision_becomes_stale_after_failure_review_changes() -> None:
    identity = _identity()
    required, failure_document = _complete_failure_review(identity)
    failure_status = assess_failure_review(identity, required, failure_document)
    agreement_document = _agreement_document()
    agreement_status = assess_first_agreement_review(
        identity.pilot_id, _agreement_refs(), agreement_document
    )
    decision = build_scale_up_decision_document(
        identity,
        failure_document,
        agreement_document,
        failure_status,
        agreement_status,
        decision=ScaleUpDecision.GO,
        rationale="Current evidence supports a controlled scale-up.",
        created_at=NOW,
    )
    changed_entry = failure_document.reviews[0].model_copy(
        update={
            "reviewer_note": "A revised evidence interpretation changed this review."
        }
    )
    changed_review = build_failure_review_document(
        identity, [changed_entry, *failure_document.reviews[1:]], created_at=NOW
    )
    changed_status = assess_failure_review(identity, required, changed_review)

    result = assess_scale_up_decision(
        identity,
        changed_status,
        agreement_status,
        changed_review,
        agreement_document,
        decision,
        current_dataset_release_sha256=identity.dataset_release_sha256,
    )

    assert not result.valid
    assert result.stale
    assert "STALE_SCALE_UP_DECISION" in result.reason_codes


def test_conditional_go_and_no_go_require_structured_reasons() -> None:
    identity = _identity()
    required, failure_document = _complete_failure_review(identity)
    failure_status = assess_failure_review(identity, required, failure_document)
    agreement_document = _agreement_document()
    agreement_status = assess_first_agreement_review(
        identity.pilot_id, _agreement_refs(), agreement_document
    )
    with pytest.raises(ValueError, match="conditions"):
        build_scale_up_decision_document(
            identity,
            failure_document,
            agreement_document,
            failure_status,
            agreement_status,
            decision=ScaleUpDecision.CONDITIONAL_GO,
            rationale="Only a conditional decision is supported.",
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="known_blockers"):
        build_scale_up_decision_document(
            identity,
            failure_document,
            agreement_document,
            failure_status,
            agreement_status,
            decision=ScaleUpDecision.NO_GO,
            rationale="The evidence does not support scaling.",
            created_at=NOW,
        )


def test_artifact_hashes_are_deterministic() -> None:
    identity = _identity()
    entries = _entries(identity, 3, 2)
    first = build_failure_review_document(identity, entries, created_at=NOW)
    second = build_failure_review_document(
        identity, list(reversed(entries)), created_at=NOW
    )
    agreement_first = _agreement_document()
    agreement_second = _agreement_document(list(reversed(_agreement_refs())))

    assert first.content_sha256 == second.content_sha256
    assert first.content_sha256 == failure_review_content_hash(first)
    assert agreement_first.content_sha256 == agreement_second.content_sha256
    assert agreement_first.content_sha256 == first_agreement_review_content_hash(
        agreement_first
    )


def test_review_output_is_atomically_written(tmp_path: Path) -> None:
    identity = _identity()
    _, document = _complete_failure_review(identity)
    output = tmp_path / "failure_review.json"

    save_failure_review(document, output)

    assert output.is_file()
    assert not list(tmp_path.glob(".failure_review.json.*.tmp"))
    assert document.content_sha256 in output.read_text(encoding="utf-8")


def test_content_hash_tampering_is_detected() -> None:
    identity = _identity()
    required, document = _complete_failure_review(identity)
    tampered = document.model_copy(
        update={
            "reviews": [
                document.reviews[0].model_copy(
                    update={"reviewer_note": "Changed without recomputing the hash."}
                ),
                *document.reviews[1:],
            ]
        }
    )

    coverage = assess_failure_review(identity, required, tampered)

    assert not coverage.complete
    assert "FAILURE_REVIEW_CONTENT_HASH_MISMATCH" in coverage.reason_codes


def test_scale_decision_content_hash_is_canonical() -> None:
    identity = _identity()
    required, failure_document = _complete_failure_review(identity)
    failure_status = assess_failure_review(identity, required, failure_document)
    agreement_document = _agreement_document()
    agreement_status = assess_first_agreement_review(
        identity.pilot_id, _agreement_refs(), agreement_document
    )
    decision = build_scale_up_decision_document(
        identity,
        failure_document,
        agreement_document,
        failure_status,
        agreement_status,
        decision=ScaleUpDecision.GO,
        rationale="Evidence-bound human judgment supports controlled scale-up.",
        created_at=NOW,
    )

    assert decision.content_sha256 == scale_up_decision_content_hash(decision)


def test_pilot_state_is_derived_only_from_current_valid_artifacts() -> None:
    identity = _identity()
    required, failure_document = _complete_failure_review(identity)
    failure_status = assess_failure_review(identity, required, failure_document)
    agreement_document = _agreement_document()
    agreement_status = assess_first_agreement_review(
        identity.pilot_id, _agreement_refs(), agreement_document
    )
    decision = build_scale_up_decision_document(
        identity,
        failure_document,
        agreement_document,
        failure_status,
        agreement_status,
        decision=ScaleUpDecision.GO,
        rationale="Only the current validated evidence permits this state.",
        created_at=NOW,
    )
    decision_status = assess_scale_up_decision(
        identity,
        failure_status,
        agreement_status,
        failure_document,
        agreement_document,
        decision,
        current_dataset_release_sha256=identity.dataset_release_sha256,
    )
    counts = PilotStageCounts(
        selected_clips=3,
        registered_clips=3,
        real_world_confirmed_clips=3,
        total_duration_seconds=180,
        double_annotated_clips=3,
        agreement_ready_clips=3,
        adjudicated_clips=3,
        benchmark_exported_clips=3,
        inference_complete_clips=3,
        benchmark_complete_clips=3,
    )

    assert (
        derive_pilot_state(
            counts,
            agreement_status,
            True,
            failure_status,
            decision_status,
            has_completion_blocker=False,
        )
        == PilotState.COMPLETE_GO
    )
    stale_failure = failure_status.model_copy(update={"complete": False})
    assert (
        derive_pilot_state(
            counts,
            agreement_status,
            True,
            stale_failure,
            decision_status,
            has_completion_blocker=False,
        )
        == PilotState.FAILURE_REVIEW_REQUIRED
    )

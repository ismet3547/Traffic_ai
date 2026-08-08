"""Provenance-bound human-review evidence for the real mini pilot.

The frozen benchmark report defines the failure set.  Official completion is
derived from canonical, content-addressed documents; manifest flags never enter
these decisions.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field, model_validator

from app.benchmark.fingerprints import canonical_sha256, streaming_file_sha256
from app.dataset.io import read_json_model, write_json_model
from app.dataset.models import AgreementReport, DisagreementType, StrictModel

FAILURE_REVIEW_SCHEMA_VERSION: Final = "1.0"
FAILURE_REVIEW_PROTOCOL_VERSION: Final = "1"
AGREEMENT_REVIEW_SCHEMA_VERSION: Final = "1.0"
AGREEMENT_REVIEW_PROTOCOL_VERSION: Final = "1"
SCALE_UP_DECISION_SCHEMA_VERSION: Final = "1.0"
SCALE_UP_DECISION_PROTOCOL_VERSION: Final = "1"
DEFAULT_FIRST_AGREEMENT_REVIEW_COUNT: Final = 3
SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"


class FailureCategory(str, Enum):
    DETECTION_FAILURE = "DETECTION_FAILURE"
    TRACKING_ID_SWITCH = "TRACKING_ID_SWITCH"
    LANE_ASSIGNMENT_ERROR = "LANE_ASSIGNMENT_ERROR"
    GEOMETRY_INTEGRITY_ERROR = "GEOMETRY_INTEGRITY_ERROR"
    OVERTAKING_LOGIC_ERROR = "OVERTAKING_LOGIC_ERROR"
    CONGESTION_LOGIC_ERROR = "CONGESTION_LOGIC_ERROR"
    RIGHT_LANE_OPPORTUNITY_ERROR = "RIGHT_LANE_OPPORTUNITY_ERROR"
    CANDIDATE_LIFECYCLE_ERROR = "CANDIDATE_LIFECYCLE_ERROR"
    EVENT_MATCHING_ERROR = "EVENT_MATCHING_ERROR"
    GROUND_TRUTH_AMBIGUITY = "GROUND_TRUTH_AMBIGUITY"
    UNKNOWN = "UNKNOWN"


class ReviewSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SystematicRisk(str, Enum):
    ISOLATED = "ISOLATED"
    POSSIBLY_SYSTEMATIC = "POSSIBLY_SYSTEMATIC"
    SYSTEMATIC = "SYSTEMATIC"
    UNKNOWN = "UNKNOWN"


class AgreementReviewAction(str, Enum):
    NONE = "NONE"
    CLARIFICATION = "CLARIFICATION"
    HANDBOOK_REVISION = "HANDBOOK_REVISION"
    REANNOTATION = "REANNOTATION"
    PAUSE = "PAUSE"


class ScaleUpDecision(str, Enum):
    GO = "GO"
    CONDITIONAL_GO = "CONDITIONAL_GO"
    NO_GO = "NO_GO"


class BaselineReviewIdentity(StrictModel):
    """The exact frozen inputs to which review evidence is bound."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pilot_id: str
    baseline_id: str
    benchmark_report_sha256: str = Field(pattern=SHA256_PATTERN)
    benchmark_report_file_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    evaluation_fingerprint: str = Field(pattern=SHA256_PATTERN)
    system_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_release_sha256: str = Field(pattern=SHA256_PATTERN)


class RequiredFailure(StrictModel):
    failure_id: str = Field(pattern=SHA256_PATTERN)
    source_failure_id: str | None = None
    failure_type: Literal["false_positive", "false_negative"]
    video_id: str
    prediction_event_id: str | None = None
    ground_truth_event_id: str | None = None
    timestamp_start: float = Field(ge=0)
    timestamp_end: float = Field(ge=0)
    artifact_directory: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> RequiredFailure:
        if self.timestamp_end < self.timestamp_start:
            raise ValueError("failure timestamp_end precedes timestamp_start")
        if self.failure_type == "false_positive" and not self.prediction_event_id:
            raise ValueError("false-positive identity requires prediction_event_id")
        if self.failure_type == "false_negative" and not self.ground_truth_event_id:
            raise ValueError("false-negative identity requires ground_truth_event_id")
        return self


class FailureReviewEntry(StrictModel):
    failure_id: str = Field(pattern=SHA256_PATTERN)
    failure_type: Literal["false_positive", "false_negative"]
    video_id: str
    prediction_event_id: str | None = None
    ground_truth_event_id: str | None = None
    timestamp_start: float = Field(ge=0)
    timestamp_end: float = Field(ge=0)
    suspected_failure_category: FailureCategory
    reviewer_note: str = Field(min_length=3, max_length=4000)
    severity: ReviewSeverity
    systematic_risk: SystematicRisk
    proposed_action: str = Field(min_length=3, max_length=2000)
    reviewed: Literal[True] = True


class FailureReviewDocument(StrictModel):
    schema_version: Literal["1.0"] = FAILURE_REVIEW_SCHEMA_VERSION
    pilot_id: str
    baseline_id: str
    benchmark_report_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    evaluation_fingerprint: str = Field(pattern=SHA256_PATTERN)
    system_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    reviews: list[FailureReviewEntry] = Field(default_factory=list)
    created_at: datetime
    review_protocol_version: Literal["1"] = FAILURE_REVIEW_PROTOCOL_VERSION
    content_sha256: str = Field(pattern=SHA256_PATTERN)


class FailureReviewCoverage(StrictModel):
    required_count: int = Field(ge=0)
    reviewed_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    stale_count: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    complete: bool
    artifact_present: bool
    artifact_content_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reason_codes: list[str] = Field(default_factory=list)
    message: str


class AgreementReportReference(StrictModel):
    video_id: str
    agreement_id: str = Field(pattern=SHA256_PATTERN)
    agreement_content_sha256: str = Field(pattern=SHA256_PATTERN)


class FirstAgreementReviewSummary(StrictModel):
    recurring_disagreement_categories: list[DisagreementType] = Field(
        default_factory=list
    )
    handbook_issue_ids: list[str] = Field(default_factory=list)
    action_required: AgreementReviewAction
    notes: str = Field(min_length=3, max_length=6000)

    @model_validator(mode="after")
    def require_handbook_links(self) -> FirstAgreementReviewSummary:
        if (
            self.action_required == AgreementReviewAction.HANDBOOK_REVISION
            and not self.handbook_issue_ids
        ):
            raise ValueError(
                "handbook revision requires at least one handbook_issue_id"
            )
        if len(self.handbook_issue_ids) != len(set(self.handbook_issue_ids)):
            raise ValueError("handbook_issue_ids must be unique")
        return self


class FirstAgreementReviewDocument(StrictModel):
    schema_version: Literal["1.0"] = AGREEMENT_REVIEW_SCHEMA_VERSION
    pilot_id: str
    required_report_count: int = Field(ge=3, le=5)
    agreement_reports: list[AgreementReportReference]
    summary: FirstAgreementReviewSummary
    protocol_version: Literal["1"] = AGREEMENT_REVIEW_PROTOCOL_VERSION
    created_at: datetime
    content_sha256: str = Field(pattern=SHA256_PATTERN)


class AgreementReviewStatus(StrictModel):
    required: bool
    complete: bool
    stale: bool
    required_report_count: int = Field(ge=0)
    reviewed_report_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    artifact_present: bool
    artifact_content_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reason_codes: list[str] = Field(default_factory=list)


class ScaleUpDecisionDocument(StrictModel):
    schema_version: Literal["1.0"] = SCALE_UP_DECISION_SCHEMA_VERSION
    pilot_id: str
    baseline_id: str
    failure_review_hash: str = Field(pattern=SHA256_PATTERN)
    first_agreement_review_hash: str = Field(pattern=SHA256_PATTERN)
    dataset_release_hash: str = Field(pattern=SHA256_PATTERN)
    benchmark_report_sha256: str = Field(pattern=SHA256_PATTERN)
    failure_review_coverage: FailureReviewCoverage
    first_agreement_review_status: AgreementReviewStatus
    decision: ScaleUpDecision
    rationale: str = Field(min_length=3, max_length=8000)
    conditions: list[str] = Field(default_factory=list)
    known_blockers: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    created_at: datetime
    decision_protocol_version: Literal["1"] = SCALE_UP_DECISION_PROTOCOL_VERSION
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_decision_support(self) -> ScaleUpDecisionDocument:
        if self.decision == ScaleUpDecision.CONDITIONAL_GO and not self.conditions:
            raise ValueError("CONDITIONAL_GO requires non-empty conditions")
        if self.decision == ScaleUpDecision.NO_GO and not self.known_blockers:
            raise ValueError("NO_GO requires non-empty known_blockers")
        return self


class ScaleUpDecisionStatus(StrictModel):
    present: bool
    valid: bool
    stale: bool
    decision: ScaleUpDecision | None = None
    artifact_content_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reason_codes: list[str] = Field(default_factory=list)


def canonical_report_sha256(report: dict[str, Any]) -> str:
    return canonical_sha256(report)


def failure_review_content_hash(document: FailureReviewDocument) -> str:
    return canonical_sha256(
        document.model_dump(mode="json", exclude={"content_sha256"})
    )


def first_agreement_review_content_hash(
    document: FirstAgreementReviewDocument,
) -> str:
    return canonical_sha256(
        document.model_dump(mode="json", exclude={"content_sha256"})
    )


def scale_up_decision_content_hash(document: ScaleUpDecisionDocument) -> str:
    return canonical_sha256(
        document.model_dump(mode="json", exclude={"content_sha256"})
    )


def failure_review_evidence_hash(
    identity: BaselineReviewIdentity,
    coverage: FailureReviewCoverage,
    document: FailureReviewDocument | None,
) -> str:
    """Hash a review artifact or the explicit zero-required-failure conclusion."""

    if document is not None:
        return document.content_sha256
    if coverage.complete and coverage.required_count == 0:
        return canonical_sha256(
            {
                "evidence_type": "zero_failure_review_not_required",
                "baseline_identity": identity.model_dump(mode="json"),
                "coverage": coverage.model_dump(mode="json"),
            }
        )
    raise ValueError("complete failure review evidence is unavailable")


def load_baseline_review_identity(
    baseline_directory: str | Path,
) -> tuple[BaselineReviewIdentity, dict[str, Any]]:
    """Load and verify the minimum immutable baseline identity needed by reviews."""

    root = Path(baseline_directory)
    metadata = _read_object(root / "baseline_metadata.json")
    report_path = root / "benchmark_report.json"
    report = _read_object(report_path)
    expected_file_hash = _required_sha(metadata, "benchmark_report_sha256")
    actual_file_hash = streaming_file_sha256(report_path)
    if actual_file_hash != expected_file_hash:
        raise ValueError("PILOT_BASELINE_INVALID: frozen benchmark report hash changed")
    if metadata.get("posthoc_model_review_allowed") is not True:
        raise ValueError(
            "POSTHOC_REVIEW_BLOCKED: baseline does not authorize model review"
        )
    if metadata.get("git_worktree_dirty") is not False:
        raise ValueError("PILOT_BASELINE_INVALID: baseline run was not clean")
    release_hash = _required_sha(metadata, "dataset_release_sha256")
    if (
        streaming_file_sha256(root / "provenance" / "dataset_release.json")
        != release_hash
    ):
        raise ValueError("PILOT_BASELINE_INVALID: frozen dataset release hash changed")
    manifest_hash = _required_sha(metadata, "pilot_manifest_sha256")
    if (
        streaming_file_sha256(root / "provenance" / "pilot_manifest.json")
        != manifest_hash
    ):
        raise ValueError("PILOT_BASELINE_INVALID: frozen pilot manifest hash changed")
    prediction_hashes = metadata.get("prediction_cache_hashes_sha256")
    if not isinstance(prediction_hashes, dict):
        raise TypeError("PILOT_BASELINE_INVALID: prediction identities unavailable")
    for video_id, expected in prediction_hashes.items():
        if (
            not _is_sha256(expected)
            or streaming_file_sha256(root / "predictions" / f"{video_id}.json")
            != expected
        ):
            raise ValueError(
                f"PILOT_BASELINE_INVALID: frozen prediction hash changed for {video_id}"
            )
    reproducibility = report.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise TypeError("PILOT_BASELINE_INVALID: report lacks reproducibility")
    evaluation = reproducibility.get("evaluation_fingerprint")
    if not _is_sha256(evaluation):
        evaluation = metadata.get("benchmark_protocol_fingerprint")
    if not _is_sha256(evaluation):
        raise ValueError("PILOT_BASELINE_INVALID: evaluation fingerprint unavailable")
    identity = BaselineReviewIdentity(
        pilot_id=str(metadata["pilot_id"]),
        baseline_id=str(metadata["baseline_id"]),
        benchmark_report_sha256=canonical_report_sha256(report),
        benchmark_report_file_sha256=actual_file_hash,
        dataset_fingerprint=_required_sha(metadata, "dataset_fingerprint"),
        evaluation_fingerprint=str(evaluation),
        system_git_commit=str(metadata["system_git_commit"]),
        dataset_release_sha256=release_hash,
    )
    return identity, report


def derive_required_failures(
    identity: BaselineReviewIdentity,
    report: dict[str, Any],
) -> list[RequiredFailure]:
    raw = report.get("failures", [])
    if not isinstance(raw, list):
        raise TypeError("benchmark report failures must be a list")
    metrics = report.get("overall_metrics")
    if not isinstance(metrics, dict):
        raise TypeError("benchmark report lacks overall_metrics")
    expected = int(metrics.get("false_positives", 0)) + int(
        metrics.get("false_negatives", 0)
    )
    if len(raw) != expected:
        raise ValueError(
            "FAILURE_SET_COUNT_MISMATCH: benchmark failure records do not match raw FP/FN counts"
        )
    required = [_required_failure(identity, item) for item in raw]
    ids = [item.failure_id for item in required]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_BASELINE_FAILURE_IDENTITY")
    return sorted(required, key=lambda item: item.failure_id)


def build_failure_review_document(
    identity: BaselineReviewIdentity,
    reviews: list[FailureReviewEntry],
    *,
    created_at: datetime | None = None,
) -> FailureReviewDocument:
    candidate = FailureReviewDocument(
        pilot_id=identity.pilot_id,
        baseline_id=identity.baseline_id,
        benchmark_report_sha256=identity.benchmark_report_sha256,
        dataset_fingerprint=identity.dataset_fingerprint,
        evaluation_fingerprint=identity.evaluation_fingerprint,
        system_git_commit=identity.system_git_commit,
        reviews=sorted(
            reviews,
            key=lambda item: canonical_sha256(item.model_dump(mode="json")),
        ),
        created_at=created_at or datetime.now(timezone.utc),
        content_sha256="0" * 64,
    )
    return candidate.model_copy(
        update={"content_sha256": failure_review_content_hash(candidate)}
    )


def assess_failure_review(
    identity: BaselineReviewIdentity,
    required: list[RequiredFailure],
    document: FailureReviewDocument | None,
) -> FailureReviewCoverage:
    required_ids = {item.failure_id for item in required}
    if document is None:
        complete = not required_ids
        return FailureReviewCoverage(
            required_count=len(required_ids),
            reviewed_count=0,
            missing_count=len(required_ids),
            duplicate_count=0,
            unknown_count=0,
            stale_count=0,
            coverage_ratio=1.0 if complete else 0.0,
            complete=complete,
            artifact_present=False,
            reason_codes=[] if complete else ["FAILURE_REVIEW_INCOMPLETE"],
            message=(
                "No FP/FN review required for this baseline."
                if complete
                else "Every FP and FN in the frozen baseline must be reviewed exactly once."
            ),
        )
    counts = Counter(item.failure_id for item in document.reviews)
    reviewed_ids = set(counts) & required_ids
    unknown_entries = sum(
        count for item, count in counts.items() if item not in required_ids
    )
    duplicate_count = sum(max(0, count - 1) for count in counts.values())
    missing_count = len(required_ids - reviewed_ids)
    required_by_id = {item.failure_id: item for item in required}
    incoherent = any(
        item.failure_id in required_by_id
        and not _review_matches_required(item, required_by_id[item.failure_id])
        for item in document.reviews
    )
    identity_stale = any(
        (
            document.pilot_id != identity.pilot_id,
            document.baseline_id != identity.baseline_id,
            document.benchmark_report_sha256 != identity.benchmark_report_sha256,
            document.dataset_fingerprint != identity.dataset_fingerprint,
            document.evaluation_fingerprint != identity.evaluation_fingerprint,
            document.system_git_commit != identity.system_git_commit,
        )
    )
    hash_valid = document.content_sha256 == failure_review_content_hash(document)
    reason_codes: list[str] = []
    if missing_count:
        reason_codes.append("MISSING_FAILURE_REVIEW")
    if duplicate_count:
        reason_codes.append("DUPLICATE_FAILURE_REVIEW")
    if unknown_entries:
        reason_codes.append("UNKNOWN_FAILURE_REVIEW")
    if identity_stale:
        reason_codes.append("STALE_FAILURE_REVIEW")
    if incoherent:
        reason_codes.append("FAILURE_REVIEW_IDENTITY_MISMATCH")
    if not hash_valid:
        reason_codes.append("FAILURE_REVIEW_CONTENT_HASH_MISMATCH")
    complete = not reason_codes and len(reviewed_ids) == len(required_ids)
    return FailureReviewCoverage(
        required_count=len(required_ids),
        reviewed_count=len(reviewed_ids),
        missing_count=missing_count,
        duplicate_count=duplicate_count,
        unknown_count=unknown_entries,
        stale_count=len(document.reviews) if identity_stale else 0,
        coverage_ratio=(len(reviewed_ids) / len(required_ids) if required_ids else 1.0),
        complete=complete,
        artifact_present=True,
        artifact_content_sha256=document.content_sha256,
        reason_codes=reason_codes,
        message=(
            "No FP/FN review required for this baseline."
            if not required_ids and complete
            else (
                "Every required FP/FN is reviewed exactly once and bound to the baseline."
                if complete
                else "Failure review evidence is incomplete, duplicated, unknown, stale, or tampered."
            )
        ),
    )


def required_agreement_reports(
    valid_reports: list[AgreementReport],
    required_count: int,
) -> list[AgreementReportReference]:
    if not 3 <= required_count <= 5:
        raise ValueError("first agreement review count must be between 3 and 5")
    by_video: dict[str, AgreementReport] = {}
    for report in valid_reports:
        if report.video_id in by_video:
            raise ValueError(f"multiple valid agreement reports for {report.video_id}")
        by_video[report.video_id] = report
    if len(by_video) < required_count:
        return []
    return [
        AgreementReportReference(
            video_id=report.video_id,
            agreement_id=report.agreement_id,
            agreement_content_sha256=report.agreement_content_sha256,
        )
        for report in (
            by_video[video_id] for video_id in sorted(by_video)[:required_count]
        )
    ]


def build_first_agreement_review_document(
    pilot_id: str,
    required: list[AgreementReportReference],
    summary: FirstAgreementReviewSummary,
    *,
    created_at: datetime | None = None,
) -> FirstAgreementReviewDocument:
    if not 3 <= len(required) <= 5:
        raise ValueError(
            "required first-agreement report set must contain 3 to 5 reports"
        )
    candidate = FirstAgreementReviewDocument(
        pilot_id=pilot_id,
        required_report_count=len(required),
        agreement_reports=sorted(required, key=lambda item: item.video_id),
        summary=summary.model_copy(
            update={
                "recurring_disagreement_categories": sorted(
                    set(summary.recurring_disagreement_categories)
                ),
                "handbook_issue_ids": sorted(summary.handbook_issue_ids),
            }
        ),
        created_at=created_at or datetime.now(timezone.utc),
        content_sha256="0" * 64,
    )
    return candidate.model_copy(
        update={"content_sha256": first_agreement_review_content_hash(candidate)}
    )


def assess_first_agreement_review(
    pilot_id: str,
    required: list[AgreementReportReference],
    document: FirstAgreementReviewDocument | None,
) -> AgreementReviewStatus:
    if not required:
        return AgreementReviewStatus(
            required=False,
            complete=False,
            stale=False,
            required_report_count=0,
            reviewed_report_count=0,
            missing_count=0,
            unknown_count=0,
            artifact_present=document is not None,
            artifact_content_sha256=document.content_sha256 if document else None,
            reason_codes=[],
        )
    expected = {item.video_id: item for item in required}
    if document is None:
        return AgreementReviewStatus(
            required=True,
            complete=False,
            stale=False,
            required_report_count=len(required),
            reviewed_report_count=0,
            missing_count=len(required),
            unknown_count=0,
            artifact_present=False,
            reason_codes=["FIRST_AGREEMENT_REVIEW_REQUIRED"],
        )
    supplied = {item.video_id: item for item in document.agreement_reports}
    duplicates = len(document.agreement_reports) != len(supplied)
    missing = set(expected) - set(supplied)
    unknown = set(supplied) - set(expected)
    stale = document.pilot_id != pilot_id or document.required_report_count != len(
        required
    )
    stale = stale or any(
        supplied[video_id] != expected[video_id]
        for video_id in set(expected) & set(supplied)
    )
    hash_valid = document.content_sha256 == first_agreement_review_content_hash(
        document
    )
    reasons: list[str] = []
    if missing:
        reasons.append("MISSING_AGREEMENT_REVIEW")
    if unknown or duplicates:
        reasons.append("UNKNOWN_AGREEMENT_REVIEW")
    if stale:
        reasons.append("STALE_AGREEMENT_REVIEW")
    if not hash_valid:
        reasons.append("AGREEMENT_REVIEW_CONTENT_HASH_MISMATCH")
    return AgreementReviewStatus(
        required=True,
        complete=not reasons,
        stale=stale,
        required_report_count=len(required),
        reviewed_report_count=len(set(expected) & set(supplied)),
        missing_count=len(missing),
        unknown_count=len(unknown) + int(duplicates),
        artifact_present=True,
        artifact_content_sha256=document.content_sha256,
        reason_codes=reasons,
    )


def build_scale_up_decision_document(
    identity: BaselineReviewIdentity,
    failure_review: FailureReviewDocument | None,
    agreement_review: FirstAgreementReviewDocument,
    failure_coverage: FailureReviewCoverage,
    agreement_status: AgreementReviewStatus,
    *,
    decision: ScaleUpDecision,
    rationale: str,
    conditions: list[str] | None = None,
    known_blockers: list[str] | None = None,
    known_limitations: list[str] | None = None,
    created_at: datetime | None = None,
) -> ScaleUpDecisionDocument:
    candidate = ScaleUpDecisionDocument(
        pilot_id=identity.pilot_id,
        baseline_id=identity.baseline_id,
        failure_review_hash=failure_review_evidence_hash(
            identity, failure_coverage, failure_review
        ),
        first_agreement_review_hash=agreement_review.content_sha256,
        dataset_release_hash=identity.dataset_release_sha256,
        benchmark_report_sha256=identity.benchmark_report_sha256,
        failure_review_coverage=failure_coverage,
        first_agreement_review_status=agreement_status,
        decision=decision,
        rationale=rationale,
        conditions=conditions or [],
        known_blockers=known_blockers or [],
        known_limitations=known_limitations or [],
        created_at=created_at or datetime.now(timezone.utc),
        content_sha256="0" * 64,
    )
    return candidate.model_copy(
        update={"content_sha256": scale_up_decision_content_hash(candidate)}
    )


def assess_scale_up_decision(
    identity: BaselineReviewIdentity | None,
    failure_coverage: FailureReviewCoverage,
    agreement_status: AgreementReviewStatus,
    failure_review: FailureReviewDocument | None,
    agreement_review: FirstAgreementReviewDocument | None,
    document: ScaleUpDecisionDocument | None,
    *,
    current_dataset_release_sha256: str | None,
) -> ScaleUpDecisionStatus:
    if document is None:
        return ScaleUpDecisionStatus(
            present=False,
            valid=False,
            stale=False,
            reason_codes=["SCALE_UP_DECISION_REQUIRED"],
        )
    reasons: list[str] = []
    if identity is None:
        reasons.append("SCALE_UP_DECISION_BASELINE_MISSING")
    if not failure_coverage.complete:
        reasons.append("SCALE_UP_DECISION_FAILURE_REVIEW_INCOMPLETE")
    if not agreement_status.required or not agreement_status.complete:
        reasons.append("SCALE_UP_DECISION_AGREEMENT_REVIEW_INCOMPLETE")
    current_failure_hash = None
    if identity is not None:
        try:
            current_failure_hash = failure_review_evidence_hash(
                identity, failure_coverage, failure_review
            )
        except ValueError:
            pass
    if document.failure_review_hash != current_failure_hash:
        reasons.append("STALE_SCALE_UP_DECISION")
    if agreement_review is None or (
        document.first_agreement_review_hash != agreement_review.content_sha256
    ):
        reasons.append("STALE_SCALE_UP_DECISION")
    if (
        document.failure_review_coverage != failure_coverage
        or document.first_agreement_review_status != agreement_status
    ):
        reasons.append("STALE_SCALE_UP_DECISION")
    if identity is not None and any(
        (
            document.pilot_id != identity.pilot_id,
            document.baseline_id != identity.baseline_id,
            document.dataset_release_hash != identity.dataset_release_sha256,
            document.benchmark_report_sha256 != identity.benchmark_report_sha256,
            current_dataset_release_sha256 != identity.dataset_release_sha256,
        )
    ):
        reasons.append("STALE_SCALE_UP_DECISION")
    if document.content_sha256 != scale_up_decision_content_hash(document):
        reasons.append("SCALE_UP_DECISION_CONTENT_HASH_MISMATCH")
    reasons = list(dict.fromkeys(reasons))
    stale = any(code.startswith("STALE_") for code in reasons)
    return ScaleUpDecisionStatus(
        present=True,
        valid=not reasons,
        stale=stale,
        decision=document.decision,
        artifact_content_sha256=document.content_sha256,
        reason_codes=reasons,
    )


def save_failure_review(document: FailureReviewDocument, path: str | Path) -> Path:
    if document.content_sha256 != failure_review_content_hash(document):
        raise ValueError("FAILURE_REVIEW_CONTENT_HASH_MISMATCH")
    return write_json_model(document, path)


def save_first_agreement_review(
    document: FirstAgreementReviewDocument, path: str | Path
) -> Path:
    if document.content_sha256 != first_agreement_review_content_hash(document):
        raise ValueError("AGREEMENT_REVIEW_CONTENT_HASH_MISMATCH")
    return write_json_model(document, path)


def save_scale_up_decision(document: ScaleUpDecisionDocument, path: str | Path) -> Path:
    if document.content_sha256 != scale_up_decision_content_hash(document):
        raise ValueError("SCALE_UP_DECISION_CONTENT_HASH_MISMATCH")
    return write_json_model(document, path)


def load_optional_failure_review(path: str | Path) -> FailureReviewDocument | None:
    source = Path(path)
    return read_json_model(source, FailureReviewDocument) if source.is_file() else None


def load_optional_first_agreement_review(
    path: str | Path,
) -> FirstAgreementReviewDocument | None:
    source = Path(path)
    return (
        read_json_model(source, FirstAgreementReviewDocument)
        if source.is_file()
        else None
    )


def load_optional_scale_up_decision(path: str | Path) -> ScaleUpDecisionDocument | None:
    source = Path(path)
    return (
        read_json_model(source, ScaleUpDecisionDocument) if source.is_file() else None
    )


def render_failure_review_summary(
    identity: BaselineReviewIdentity,
    coverage: FailureReviewCoverage,
    document: FailureReviewDocument | None,
) -> str:
    lines = [
        "# Mini Pilot Failure Review",
        "",
        f"Pilot: `{identity.pilot_id}`",
        f"Baseline: `{identity.baseline_id}`",
        f"Benchmark report SHA-256 (canonical): `{identity.benchmark_report_sha256}`",
        "",
        "## Machine-verifiable coverage",
        "",
        f"- Required: {coverage.required_count}",
        f"- Reviewed unique required failures: {coverage.reviewed_count}",
        f"- Missing: {coverage.missing_count}",
        f"- Duplicate: {coverage.duplicate_count}",
        f"- Unknown: {coverage.unknown_count}",
        f"- Stale: {coverage.stale_count}",
        f"- Complete: {'yes' if coverage.complete else 'no'}",
        f"- Status: {coverage.message}",
        "",
    ]
    if document is None or not document.reviews:
        lines.append("No human failure review entries are present.")
    else:
        lines.extend(["## Reviewed failures", ""])
        for review in sorted(document.reviews, key=lambda item: item.failure_id):
            lines.append(
                f"- `{review.failure_id}` {review.failure_type} / `{review.video_id}`: "
                f"{review.suspected_failure_category.value}, {review.severity.value}, "
                f"{review.systematic_risk.value} — {review.reviewer_note}"
            )
    lines.extend(
        [
            "",
            "> Mini-pilot sample size is too small for production accuracy claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _required_failure(identity: BaselineReviewIdentity, raw: Any) -> RequiredFailure:
    if not isinstance(raw, dict):
        raise TypeError("benchmark failure entry must be an object")
    kind = raw.get("kind")
    if kind not in {"false_positive", "false_negative"}:
        raise ValueError(
            "benchmark failure kind must be false_positive or false_negative"
        )
    raw_prediction = raw.get("prediction")
    prediction: dict[str, Any] = (
        raw_prediction if isinstance(raw_prediction, dict) else {}
    )
    raw_ground_truth = raw.get("ground_truth")
    ground_truth: dict[str, Any] = (
        raw_ground_truth if isinstance(raw_ground_truth, dict) else {}
    )
    prediction_id = _optional_text(prediction.get("event_id"))
    truth_id = _optional_text(ground_truth.get("event_id"))
    start = _first_number(
        prediction.get("start_seconds"),
        ground_truth.get("start_seconds"),
        raw.get("timestamp_seconds"),
    )
    end = _first_number(
        prediction.get("end_seconds"), ground_truth.get("end_seconds"), start
    )
    payload = {
        "baseline_id": identity.baseline_id,
        "video_id": str(raw.get("video_id", "")),
        "failure_type": kind,
        "prediction_event_id": prediction_id,
        "ground_truth_event_id": truth_id,
        "timestamp_start": start,
        "timestamp_end": end,
    }
    return RequiredFailure(
        failure_id=canonical_sha256(payload),
        source_failure_id=_optional_text(raw.get("failure_id")),
        failure_type=kind,
        video_id=payload["video_id"],
        prediction_event_id=prediction_id,
        ground_truth_event_id=truth_id,
        timestamp_start=start,
        timestamp_end=end,
        artifact_directory=_optional_text(raw.get("artifact_directory")),
    )


def _review_matches_required(
    review: FailureReviewEntry, required: RequiredFailure
) -> bool:
    return all(
        (
            review.failure_type == required.failure_type,
            review.video_id == required.video_id,
            review.prediction_event_id == required.prediction_event_id,
            review.ground_truth_event_id == required.ground_truth_event_id,
            abs(review.timestamp_start - required.timestamp_start) <= 1e-9,
            abs(review.timestamp_end - required.timestamp_end) <= 1e-9,
        )
    )


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _required_sha(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not _is_sha256(result):
        raise ValueError(f"baseline metadata lacks valid {field}")
    return str(result)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None and str(value) else None


def _first_number(*values: object) -> float:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise ValueError("failure record lacks a numeric timestamp")

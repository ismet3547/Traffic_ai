"""Operational status and immutable-baseline controls for the real mini pilot."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

from app.benchmark.annotations import load_annotation as load_benchmark_annotation
from app.benchmark.fingerprints import canonical_sha256, streaming_file_sha256
from app.benchmark.models import EvaluationProtocolIdentity, PredictionDocument
from app.dataset.agreement_integrity import assess_agreement_report
from app.dataset.integrity import (
    assess_adjudication_source_identity,
    validate_double_annotation,
)
from app.dataset.io import (
    document_sha256,
    load_adjudication,
    load_agreement,
    load_annotation,
    read_json_model,
    write_json_model,
)
from app.dataset.models import (
    AGREEMENT_PROTOCOL_VERSION,
    CANONICAL_AGREEMENT_PROTOCOL,
    DATASET_VERSION,
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    AdjudicationArtifact,
    AgreementProtocolIdentity,
    AgreementReport,
    DatasetAnnotation,
    DatasetRelease,
    IntakeRegistry,
    PermissionStatus,
    StrictModel,
    VideoIntakeRecord,
)
from app.dataset.pilot_review import (
    AgreementReportReference,
    AgreementReviewStatus,
    FailureReviewCoverage,
    ScaleUpDecision,
    ScaleUpDecisionStatus,
    assess_failure_review,
    assess_first_agreement_review,
    assess_scale_up_decision,
    derive_required_failures,
    load_baseline_review_identity,
    load_optional_failure_review,
    load_optional_first_agreement_review,
    load_optional_scale_up_decision,
    required_agreement_reports,
)

PILOT_MANIFEST_SCHEMA_VERSION: Final = "1.0"
PILOT_STATUS_SCHEMA_VERSION: Final = "1.2"
PILOT_BASELINE_SCHEMA_VERSION: Final = "1.0"
MINI_PILOT_ACCURACY_WARNING: Final = (
    "Mini-pilot sample size is too small for production accuracy claims."
)
NO_REAL_VIDEO_STATUS: Final = (
    "REAL PILOT STATUS: BLOCKED — NO REAL SOURCE VIDEO REGISTERED"
)


class PilotArtifactLayout(StrictModel):
    """Artifact locations resolved relative to the pilot manifest."""

    registry: str = "../intake_registry.json"
    annotations_directory: str = "annotations"
    agreements_directory: str = "agreements"
    adjudications_directory: str = "adjudications"
    ground_truth_directory: str = "ground_truth"
    dataset_release: str = "dataset_release.json"
    benchmark_manifest: str = "benchmark_manifest.yaml"
    benchmark_run_directory: str = (
        "../../../benchmark_output/mini-pilot-001/current_run"
    )
    baseline_directory: str = (
        "../../../benchmark_output/mini-pilot-001/pilot_baseline_0"
    )
    failure_review: str = "failure_review.json"
    first_agreement_review: str = "first_agreement_review.json"
    scale_up_decision: str = "scale_up_decision.json"


class PilotClipSelection(StrictModel):
    video_id: str = Field(min_length=1, max_length=120)
    real_world_source_confirmed: bool = False
    local_video_path: str | None = None
    production_config_path: str | None = None
    selection_notes: str | None = Field(default=None, max_length=1000)
    annotation_duration_minutes: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_effort(self) -> PilotClipSelection:
        if any(not annotator.strip() for annotator in self.annotation_duration_minutes):
            raise ValueError("annotation effort requires anonymous annotator IDs")
        if any(value <= 0 for value in self.annotation_duration_minutes.values()):
            raise ValueError("annotation effort minutes must be greater than zero")
        return self


class PilotManifest(StrictModel):
    schema_version: Literal["1.0"] = PILOT_MANIFEST_SCHEMA_VERSION
    pilot_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,79}$")
    dataset_version: str = DATASET_VERSION
    ontology_version: str = ONTOLOGY_VERSION
    handbook_version: str = HANDBOOK_VERSION
    agreement_protocol: AgreementProtocolIdentity
    frozen_at: datetime
    clips: list[PilotClipSelection] = Field(default_factory=list)
    first_agreement_review_count: int = Field(default=3, ge=3, le=5)
    # Legacy fields parse for migration only. They are excluded from writes and
    # never participate in any status or completion decision.
    first_agreement_review_video_ids: list[str] | None = Field(
        default=None, exclude=True
    )
    failure_review_completed: bool | None = Field(default=None, exclude=True)
    scale_up_recommendation: str | None = Field(default=None, exclude=True)
    artifacts: PilotArtifactLayout = Field(default_factory=PilotArtifactLayout)

    @model_validator(mode="after")
    def validate_identity(self) -> PilotManifest:
        video_ids = [item.video_id for item in self.clips]
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("pilot clip video IDs must be unique")
        return self


class PilotStageCounts(StrictModel):
    selected_clips: int = Field(ge=0)
    registered_clips: int = Field(ge=0)
    real_world_confirmed_clips: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    double_annotated_clips: int = Field(ge=0)
    agreement_ready_clips: int = Field(ge=0)
    adjudicated_clips: int = Field(ge=0)
    benchmark_exported_clips: int = Field(ge=0)
    inference_complete_clips: int = Field(ge=0)
    benchmark_complete_clips: int = Field(ge=0)


class PilotIssueSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    BLOCKER = "BLOCKER"


NON_BLOCKING_PILOT_ISSUE_SEVERITIES: Final = {
    "MINI_PILOT_ACCURACY_WARNING": PilotIssueSeverity.WARNING,
    "LIMITED_SCENARIO_DIVERSITY": PilotIssueSeverity.WARNING,
    "LOW_TOTAL_DURATION": PilotIssueSeverity.WARNING,
    "ANNOTATION_EFFORT_NOT_RECORDED": PilotIssueSeverity.WARNING,
    "LEGACY_PILOT_COMPLETION_FIELDS_IGNORED": PilotIssueSeverity.INFO,
}


def pilot_issue_severity(code: str) -> PilotIssueSeverity:
    """Classify one issue; unknown codes fail closed as completion blockers."""

    configured = NON_BLOCKING_PILOT_ISSUE_SEVERITIES.get(code)
    return configured if configured is not None else PilotIssueSeverity.BLOCKER


class PilotIssue(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[A-Z0-9_]+$")
    details: str
    video_id: str | None = None
    severity: PilotIssueSeverity = PilotIssueSeverity.BLOCKER

    @model_validator(mode="before")
    @classmethod
    def apply_central_severity(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("code"), str):
            normalized = dict(value)
            normalized["severity"] = pilot_issue_severity(value["code"])
            return normalized
        return value


# Compatibility alias for callers that imported the former model name. Severity
# is still assigned exclusively by the centralized classifier above.
PilotBlocker = PilotIssue


class PilotState(str, Enum):
    PREPARED = "PREPARED"
    REGISTERING_DATA = "REGISTERING_DATA"
    ANNOTATING = "ANNOTATING"
    AGREEMENT_REVIEW_REQUIRED = "AGREEMENT_REVIEW_REQUIRED"
    ADJUDICATING = "ADJUDICATING"
    BENCHMARK_READY = "BENCHMARK_READY"
    BASELINE_FROZEN = "BASELINE_FROZEN"
    FAILURE_REVIEW_REQUIRED = "FAILURE_REVIEW_REQUIRED"
    SCALE_UP_DECISION_REQUIRED = "SCALE_UP_DECISION_REQUIRED"
    COMPLETE_GO = "COMPLETE_GO"
    COMPLETE_CONDITIONAL_GO = "COMPLETE_CONDITIONAL_GO"
    COMPLETE_NO_GO = "COMPLETE_NO_GO"


TERMINAL_PILOT_STATES: Final = frozenset(
    {
        PilotState.COMPLETE_GO,
        PilotState.COMPLETE_CONDITIONAL_GO,
        PilotState.COMPLETE_NO_GO,
    }
)


class PilotStatus(StrictModel):
    schema_version: Literal["1.2"] = PILOT_STATUS_SCHEMA_VERSION
    pilot_id: str
    pilot_state: PilotState
    real_pilot_status: str
    pilot_executed: bool
    counts: PilotStageCounts
    scenario_tag_counts: dict[str, int]
    annotation_effort_total_minutes: float | None = Field(default=None, ge=0)
    annotation_effort_mean_minutes_per_pass: float | None = Field(default=None, ge=0)
    first_agreement_review_required: bool
    first_agreement_review_completed: bool
    first_agreement_review: AgreementReviewStatus
    pilot_baseline_frozen: bool
    posthoc_model_review_allowed: bool
    failure_review: FailureReviewCoverage
    scale_up_decision: ScaleUpDecisionStatus
    scale_up_recommendation: Literal[
        "NOT_ASSESSED", "GO", "CONDITIONAL_GO", "NO_GO"
    ] = "NOT_ASSESSED"
    blockers: list[PilotIssue]
    warnings: list[PilotIssue] = Field(default_factory=list)
    information: list[PilotIssue] = Field(default_factory=list)
    accuracy_warning: str = MINI_PILOT_ACCURACY_WARNING

    @model_validator(mode="after")
    def enforce_terminal_invariants(self) -> PilotStatus:
        all_issues = [*self.blockers, *self.warnings, *self.information]
        if any(item.severity != pilot_issue_severity(item.code) for item in all_issues):
            raise ValueError("pilot issue severity differs from central classification")
        if any(item.severity != PilotIssueSeverity.BLOCKER for item in self.blockers):
            raise ValueError("pilot blockers must contain only BLOCKER issues")
        if any(item.severity != PilotIssueSeverity.WARNING for item in self.warnings):
            raise ValueError("pilot warnings must contain only WARNING issues")
        if any(item.severity != PilotIssueSeverity.INFO for item in self.information):
            raise ValueError("pilot information must contain only INFO issues")
        terminal = self.pilot_state in TERMINAL_PILOT_STATES
        if terminal and self.blockers:
            raise ValueError("terminal pilot state cannot coexist with active blockers")
        if self.pilot_executed != (terminal and not self.blockers):
            raise ValueError(
                "pilot_executed must exactly reflect a blocker-free terminal state"
            )
        return self


class PilotBaselineMetadata(StrictModel):
    """Small immutable identity record stored with the copied baseline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = PILOT_BASELINE_SCHEMA_VERSION
    pilot_id: str
    baseline_id: Literal["pilot_baseline_0"] = "pilot_baseline_0"
    frozen_at: datetime
    system_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_worktree_dirty: Literal[False] = False
    resolved_config_hash_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_config_hash_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_model_identifiers: list[str]
    tracker_identifiers: list[str]
    benchmark_protocol: EvaluationProtocolIdentity
    benchmark_protocol_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_cache_hashes_sha256: dict[str, str]
    video_ids: list[str]
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_positives_per_video_hour: float = Field(ge=0)
    posthoc_model_review_allowed: Literal[True] = True
    accuracy_warning: str = MINI_PILOT_ACCURACY_WARNING

    @model_validator(mode="after")
    def validate_identity(self) -> PilotBaselineMetadata:
        expected = canonical_sha256(self.benchmark_protocol.model_dump(mode="json"))
        if self.benchmark_protocol_fingerprint != expected:
            raise ValueError("benchmark protocol fingerprint is incoherent")
        if sorted(self.prediction_cache_hashes_sha256) != sorted(self.video_ids):
            raise ValueError("baseline prediction hashes must cover every video")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.prediction_cache_hashes_sha256.values()
        ):
            raise ValueError("baseline prediction hashes must be lowercase SHA-256")
        return self


class PilotReviewBlockedError(ValueError):
    pass


class PilotBaselineExistsError(FileExistsError):
    pass


def load_pilot_manifest(path: str | Path) -> PilotManifest:
    return read_json_model(path, PilotManifest)


def current_required_agreement_reports(
    manifest: PilotManifest,
    manifest_path: str | Path,
) -> list[AgreementReportReference]:
    """Return the deterministic first-N current canonical agreement identities."""

    base = Path(manifest_path).resolve().parent
    blockers: list[PilotBlocker] = []
    registry = _load_registry_status(
        _resolve(base, manifest.artifacts.registry), blockers
    )
    records = {item.video_id: item for item in registry.videos}
    annotations = _load_annotations(
        _resolve(base, manifest.artifacts.annotations_directory), blockers
    )
    agreements = _load_agreements(
        _resolve(base, manifest.artifacts.agreements_directory), blockers
    )
    valid: list[AgreementReport] = []
    for video_id in sorted(item.video_id for item in manifest.clips):
        record = records.get(video_id)
        documents = annotations.get(video_id, [])
        reports = agreements.get(video_id, [])
        if record is None or not validate_double_annotation(record, documents).valid:
            continue
        if len(reports) != 1:
            continue
        by_id = {item.annotator_id: item for item in documents}
        report = reports[0]
        first = by_id.get(report.annotator_a_id)
        second = by_id.get(report.annotator_b_id)
        if first is None or second is None:
            continue
        if assess_agreement_report(record, first, second, report).valid:
            valid.append(report)
    return required_agreement_reports(valid, manifest.first_agreement_review_count)


def build_pilot_status(
    manifest: PilotManifest,
    manifest_path: str | Path,
) -> PilotStatus:
    """Calculate current progress from real artifacts; never infer completion."""

    base = Path(manifest_path).resolve().parent
    layout = manifest.artifacts
    blockers: list[PilotBlocker] = []
    selected = {item.video_id: item for item in manifest.clips}
    registry_path = _resolve(base, layout.registry)
    registry = _load_registry_status(registry_path, blockers)
    records = {item.video_id: item for item in registry.videos}
    registered = {
        video_id: records[video_id] for video_id in selected if video_id in records
    }

    if not selected:
        if registry.videos:
            blockers.append(
                PilotBlocker(
                    code="NO_PILOT_CLIPS_SELECTED",
                    details=(
                        "The registry has source records, but no real clip is selected "
                        "in this pilot manifest."
                    ),
                )
            )
        else:
            blockers.append(
                PilotBlocker(
                    code="NO_REAL_SOURCE_VIDEO_REGISTERED",
                    details=(
                        "No real clip is selected in the pilot manifest; register and "
                        "explicitly add the first legally usable source video."
                    ),
                )
            )
    _check_protocol_freeze(manifest, blockers)
    for video_id, selection in sorted(selected.items()):
        record = records.get(video_id)
        if record is None:
            blockers.append(
                PilotBlocker(
                    code="PILOT_CLIP_NOT_REGISTERED",
                    details="Pilot selection has no matching intake registry record.",
                    video_id=video_id,
                )
            )
            continue
        _check_registered_clip(base, selection, record, blockers)

    annotations = _load_annotations(
        _resolve(base, layout.annotations_directory), blockers
    )
    double_ready: set[str] = set()
    for video_id, record in registered.items():
        if validate_double_annotation(record, annotations.get(video_id, [])).valid:
            double_ready.add(video_id)

    agreements = _load_agreements(_resolve(base, layout.agreements_directory), blockers)
    agreement_ready: set[str] = set()
    valid_agreement_reports: list[AgreementReport] = []
    for video_id in double_ready:
        reports = agreements.get(video_id, [])
        documents = annotations[video_id]
        if len(reports) == 1:
            by_id = {item.annotator_id: item for item in documents}
            agreement_report = reports[0]
            first = by_id.get(agreement_report.annotator_a_id)
            second = by_id.get(agreement_report.annotator_b_id)
            if first is not None and second is not None:
                result = assess_agreement_report(
                    registered[video_id], first, second, agreement_report
                )
                if result.valid:
                    agreement_ready.add(video_id)
                    valid_agreement_reports.append(agreement_report)

    adjudications = _load_adjudications(
        _resolve(base, layout.adjudications_directory), blockers
    )
    adjudicated: set[str] = set()
    for video_id in agreement_ready:
        artifacts = adjudications.get(video_id, [])
        if len(artifacts) == 1:
            artifact = artifacts[0]
            result = assess_adjudication_source_identity(
                registered[video_id], artifact, annotations[video_id]
            )
            if result.valid and artifact.approved and artifact.locked:
                adjudicated.add(video_id)

    release = _load_release_status(_resolve(base, layout.dataset_release), blockers)
    benchmark_exported = _benchmark_exported_ids(
        selected,
        adjudicated,
        release,
        _resolve(base, layout.ground_truth_directory),
        blockers,
    )
    run_directory = _active_run_directory(base, layout)
    inference_complete = _inference_complete_ids(
        selected,
        records,
        run_directory / "predictions",
        blockers,
    )
    benchmark_report = _load_benchmark_report(
        run_directory / "benchmark_report.json", blockers
    )
    benchmark_complete = {
        video_id
        for video_id in selected
        if video_id in (benchmark_report or {}).get("per_video_metrics", {})
    }
    locked_review_eligible = False
    if benchmark_report is not None and release is not None and selected:
        try:
            authorize_posthoc_model_review(release, benchmark_report, set(selected))
            locked_review_eligible = True
        except PilotReviewBlockedError as exc:
            blockers.append(
                PilotBlocker(code="POSTHOC_REVIEW_BLOCKED", details=str(exc))
            )

    for video_id in sorted(selected):
        if video_id not in registered:
            continue
        if video_id not in double_ready:
            code = "DOUBLE_ANNOTATION_INCOMPLETE"
            details = "Two independent annotation passes are not valid and locked."
        elif video_id not in agreement_ready:
            code = "AGREEMENT_NOT_READY"
            details = "One current canonical official agreement report is required."
        elif video_id not in adjudicated:
            code = "ADJUDICATION_NOT_LOCKED"
            details = "Adjudication is not approved, current, and locked."
        elif video_id not in benchmark_exported:
            code = "BENCHMARK_GROUND_TRUTH_NOT_EXPORTED"
            details = "Released adjudicated benchmark ground truth is missing."
        elif video_id not in inference_complete:
            code = "INFERENCE_NOT_COMPLETE"
            details = "A source-bound prediction cache is missing."
        elif video_id not in benchmark_complete:
            code = "BENCHMARK_NOT_COMPLETE"
            details = "The current benchmark report does not contain this clip."
        else:
            continue
        blockers.append(PilotBlocker(code=code, details=details, video_id=video_id))

    required_agreements = required_agreement_reports(
        valid_agreement_reports, manifest.first_agreement_review_count
    )
    agreement_review_document = _load_review_artifact(
        _resolve(base, layout.first_agreement_review),
        load_optional_first_agreement_review,
        "FIRST_AGREEMENT_REVIEW_INVALID",
        blockers,
    )
    agreement_review_status = assess_first_agreement_review(
        manifest.pilot_id,
        required_agreements,
        agreement_review_document,
    )
    if agreement_review_status.required and not agreement_review_status.complete:
        blockers.append(
            PilotBlocker(
                code="FIRST_AGREEMENT_REVIEW_REQUIRED",
                details=(
                    "Pause new annotation and review the deterministic first "
                    f"{manifest.first_agreement_review_count} current agreement reports."
                ),
            )
        )
        for reason in agreement_review_status.reason_codes:
            if reason != "FIRST_AGREEMENT_REVIEW_REQUIRED":
                blockers.append(
                    PilotBlocker(
                        code=reason,
                        details="Initial agreement review evidence is not current and exact.",
                    )
                )

    effort = [
        minutes
        for clip in manifest.clips
        for minutes in clip.annotation_duration_minutes.values()
    ]
    tag_counts = Counter(
        tag for record in registered.values() for tag in record.scenario_tags
    )
    baseline_path = _resolve(base, layout.baseline_directory) / "baseline_metadata.json"
    baseline_frozen = _baseline_metadata_valid(
        baseline_path, manifest.pilot_id, blockers
    )
    if baseline_frozen:
        _check_baseline_current_config(manifest, base, baseline_path, blockers)
    if locked_review_eligible and not baseline_frozen:
        blockers.append(
            PilotBlocker(
                code="PILOT_BASELINE_NOT_FROZEN",
                details=(
                    "The complete locked-GT run must be frozen as pilot_baseline_0 "
                    "before post-hoc model review."
                ),
            )
        )
    review_allowed = locked_review_eligible and baseline_frozen
    counts = PilotStageCounts(
        selected_clips=len(selected),
        registered_clips=len(registered),
        real_world_confirmed_clips=sum(
            selected[video_id].real_world_source_confirmed for video_id in registered
        ),
        total_duration_seconds=sum(
            item.duration_seconds for item in registered.values()
        ),
        double_annotated_clips=len(double_ready),
        agreement_ready_clips=len(agreement_ready),
        adjudicated_clips=len(adjudicated),
        benchmark_exported_clips=len(benchmark_exported),
        inference_complete_clips=len(inference_complete),
        benchmark_complete_clips=len(benchmark_complete),
    )
    baseline_identity = None
    failure_review_document = None
    failure_coverage = FailureReviewCoverage(
        required_count=0,
        reviewed_count=0,
        missing_count=0,
        duplicate_count=0,
        unknown_count=0,
        stale_count=0,
        coverage_ratio=0.0,
        complete=False,
        artifact_present=False,
        reason_codes=["PILOT_BASELINE_NOT_FROZEN"],
        message="Failure review cannot begin until pilot_baseline_0 is frozen.",
    )
    if baseline_frozen:
        try:
            baseline_identity, frozen_report = load_baseline_review_identity(
                baseline_path.parent
            )
            required_failures = derive_required_failures(
                baseline_identity, frozen_report
            )
            failure_review_document = _load_review_artifact(
                _resolve(base, layout.failure_review),
                load_optional_failure_review,
                "FAILURE_REVIEW_INVALID",
                blockers,
            )
            failure_coverage = assess_failure_review(
                baseline_identity, required_failures, failure_review_document
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                PilotBlocker(code="FAILURE_REVIEW_SOURCE_INVALID", details=str(exc))
            )
    if baseline_frozen and not failure_coverage.complete:
        blockers.append(
            PilotBlocker(
                code="FAILURE_REVIEW_INCOMPLETE",
                details=failure_coverage.message,
            )
        )
        for reason in failure_coverage.reason_codes:
            if reason != "FAILURE_REVIEW_INCOMPLETE":
                blockers.append(
                    PilotBlocker(code=reason, details=failure_coverage.message)
                )
    if (
        baseline_frozen
        and failure_coverage.complete
        and not agreement_review_status.required
    ):
        blockers.append(
            PilotBlocker(
                code="FIRST_AGREEMENT_REVIEW_NOT_TRIGGERED",
                details=(
                    "A scale-up decision requires the configured first-N current "
                    "agreement-report set; the trigger has not been reached."
                ),
            )
        )

    scale_decision_document = _load_review_artifact(
        _resolve(base, layout.scale_up_decision),
        load_optional_scale_up_decision,
        "SCALE_UP_DECISION_INVALID",
        blockers,
    )
    release_path = _resolve(base, layout.dataset_release)
    current_release_hash = (
        streaming_file_sha256(release_path) if release_path.is_file() else None
    )
    scale_decision_status = assess_scale_up_decision(
        baseline_identity,
        failure_coverage,
        agreement_review_status,
        failure_review_document,
        agreement_review_document,
        scale_decision_document,
        current_dataset_release_sha256=current_release_hash,
    )
    if (
        baseline_frozen
        and failure_coverage.complete
        and agreement_review_status.complete
        and not scale_decision_status.valid
    ):
        blockers.append(
            PilotBlocker(
                code="SCALE_UP_DECISION_REQUIRED",
                details=(
                    "Record a reasoned scale-up decision bound to the current "
                    "failure review, agreement review, release, and baseline."
                ),
            )
        )
        for reason in scale_decision_status.reason_codes:
            if reason != "SCALE_UP_DECISION_REQUIRED":
                blockers.append(
                    PilotBlocker(
                        code=reason,
                        details="Scale-up decision evidence is incomplete or stale.",
                    )
                )

    if any(
        value is not None
        for value in (
            manifest.failure_review_completed,
            manifest.first_agreement_review_video_ids,
            manifest.scale_up_recommendation,
        )
    ):
        blockers.append(
            PilotBlocker(
                code="LEGACY_PILOT_COMPLETION_FIELDS_IGNORED",
                details=(
                    "Legacy completion fields were loaded for migration but have no "
                    "authority; current review artifacts determine pilot status."
                ),
            )
        )

    blockers.append(
        PilotIssue(
            code="MINI_PILOT_ACCURACY_WARNING",
            details=MINI_PILOT_ACCURACY_WARNING,
        )
    )
    issues = _deduplicate_issues(blockers)
    completion_blockers = [
        item for item in issues if item.severity == PilotIssueSeverity.BLOCKER
    ]
    warnings = [item for item in issues if item.severity == PilotIssueSeverity.WARNING]
    information = [item for item in issues if item.severity == PilotIssueSeverity.INFO]
    pilot_state = derive_pilot_state(
        counts,
        agreement_review_status,
        baseline_frozen,
        failure_coverage,
        scale_decision_status,
        has_completion_blocker=bool(completion_blockers),
    )
    executed = pilot_state in TERMINAL_PILOT_STATES and not completion_blockers
    recommendation: Literal["NOT_ASSESSED", "GO", "CONDITIONAL_GO", "NO_GO"]
    if scale_decision_status.valid and scale_decision_status.decision is not None:
        recommendation = scale_decision_status.decision.value
    else:
        recommendation = "NOT_ASSESSED"
    if not selected and not registry.videos:
        real_status = NO_REAL_VIDEO_STATUS
    elif not selected:
        real_status = "REAL PILOT STATUS: BLOCKED — NO PILOT CLIP SELECTED"
    elif not registered:
        real_status = "REAL PILOT STATUS: BLOCKED — SELECTED CLIPS NOT REGISTERED"
    elif executed:
        real_status = f"REAL PILOT STATUS: {pilot_state.value}"
    elif completion_blockers:
        real_status = "REAL PILOT STATUS: IN PROGRESS — BLOCKERS PRESENT"
    elif baseline_frozen and review_allowed:
        real_status = "REAL PILOT STATUS: PILOT BASELINE 0 FROZEN"
    else:
        real_status = "REAL PILOT STATUS: READY FOR BASELINE FREEZE"
    return PilotStatus(
        pilot_id=manifest.pilot_id,
        pilot_state=pilot_state,
        real_pilot_status=real_status,
        pilot_executed=executed,
        counts=counts,
        scenario_tag_counts=dict(sorted(tag_counts.items())),
        annotation_effort_total_minutes=sum(effort) if effort else None,
        annotation_effort_mean_minutes_per_pass=(
            sum(effort) / len(effort) if effort else None
        ),
        first_agreement_review_required=agreement_review_status.required,
        first_agreement_review_completed=agreement_review_status.complete,
        first_agreement_review=agreement_review_status,
        pilot_baseline_frozen=baseline_frozen,
        posthoc_model_review_allowed=review_allowed,
        failure_review=failure_coverage,
        scale_up_decision=scale_decision_status,
        scale_up_recommendation=recommendation,
        blockers=completion_blockers,
        warnings=warnings,
        information=information,
    )


def render_pilot_status(status: PilotStatus) -> str:
    counts = status.counts
    lines = [
        f"# Mini Pilot Status — {status.pilot_id}",
        "",
        f"**{status.real_pilot_status}**",
        "",
        f"Pilot state: `{status.pilot_state.value}`",
        "",
        f"> {status.accuracy_warning}",
        "",
        "## Progress",
        "",
        f"- Selected clips: {counts.selected_clips}",
        f"- Registered clips: {counts.registered_clips}",
        f"- Confirmed real-world clips: {counts.real_world_confirmed_clips}",
        f"- Total duration: {counts.total_duration_seconds:.3f} seconds",
        f"- Double annotated: {counts.double_annotated_clips}",
        f"- Agreement ready: {counts.agreement_ready_clips}",
        f"- Adjudicated and locked: {counts.adjudicated_clips}",
        f"- Benchmark GT exported: {counts.benchmark_exported_clips}",
        f"- Inference complete: {counts.inference_complete_clips}",
        f"- Benchmark complete: {counts.benchmark_complete_clips}",
        f"- Pilot baseline 0 frozen: {'yes' if status.pilot_baseline_frozen else 'no'}",
        (
            "- Post-hoc model review allowed: "
            + ("yes" if status.posthoc_model_review_allowed else "no")
        ),
        "",
        "## Evidence-backed review",
        "",
        (
            "- First agreement review: "
            f"required={status.first_agreement_review.required}, "
            f"complete={status.first_agreement_review.complete}, "
            f"stale={status.first_agreement_review.stale}, "
            f"required reports={status.first_agreement_review.required_report_count}"
        ),
        (
            "- Failure review: "
            f"required={status.failure_review.required_count}, "
            f"reviewed={status.failure_review.reviewed_count}, "
            f"missing={status.failure_review.missing_count}, "
            f"duplicate={status.failure_review.duplicate_count}, "
            f"unknown={status.failure_review.unknown_count}, "
            f"stale={status.failure_review.stale_count}, "
            f"complete={status.failure_review.complete}"
        ),
        f"- Failure review note: {status.failure_review.message}",
        (
            "- Scale-up decision: "
            f"present={status.scale_up_decision.present}, "
            f"valid={status.scale_up_decision.valid}, "
            f"stale={status.scale_up_decision.stale}, "
            f"decision={status.scale_up_recommendation}"
        ),
        "",
        "## Actual scenario coverage",
        "",
    ]
    if status.scenario_tag_counts:
        lines.extend(
            f"- `{tag}`: {count} clip(s)"
            for tag, count in status.scenario_tag_counts.items()
        )
    else:
        lines.append("No real scenario coverage is registered.")
    lines.extend(["", "## Blockers", ""])
    if status.blockers:
        for blocker in status.blockers:
            scope = f" ({blocker.video_id})" if blocker.video_id else ""
            lines.append(f"- `{blocker.code}`{scope}: {blocker.details}")
    else:
        lines.append("No current workflow blockers.")
    lines.extend(["", "## Warnings", ""])
    if status.warnings:
        for warning in status.warnings:
            scope = f" ({warning.video_id})" if warning.video_id else ""
            lines.append(f"- `{warning.code}`{scope}: {warning.details}")
    else:
        lines.append("No current warnings.")
    lines.extend(["", "## Information", ""])
    if status.information:
        for notice in status.information:
            scope = f" ({notice.video_id})" if notice.video_id else ""
            lines.append(f"- `{notice.code}`{scope}: {notice.details}")
    else:
        lines.append("No informational notices.")
    recommendation = status.scale_up_recommendation
    recommendation_detail = (
        "a real end-to-end mini pilot has not yet supplied evidence for a "
        "GO/NO-GO decision."
        if recommendation == "NOT_ASSESSED"
        else "see the pilot summaries for the evidence and reasons."
    )
    lines.extend(
        [
            "",
            "## Scale-up recommendation",
            "",
            f"`{recommendation}` — {recommendation_detail}",
            "",
            "Pilot completion is derived from current review artifacts; legacy manifest completion fields are ignored.",
            "",
            "No TP, FP, FN, precision, recall, F1, or FP/hour values are reported until a locked real pilot baseline exists.",
            "",
        ]
    )
    return "\n".join(lines)


def authorize_posthoc_model_review(
    release: DatasetRelease,
    benchmark_report: dict[str, Any],
    expected_video_ids: set[str],
) -> None:
    """Require locked, released GT before humans may inspect model errors."""

    if not expected_video_ids:
        raise PilotReviewBlockedError("no real pilot videos are available for review")
    if benchmark_report.get("synthetic") is not False:
        raise PilotReviewBlockedError("synthetic reports are not real pilot evidence")
    report_ids = set((benchmark_report.get("per_video_metrics") or {}).keys())
    if report_ids != expected_video_ids:
        raise PilotReviewBlockedError(
            "benchmark report does not cover exactly the selected pilot videos"
        )
    if not release.integrity_report.passed or not release.quality_gate_passed:
        raise PilotReviewBlockedError("dataset release integrity is not passing")
    release_by_video = {item.video_id: item for item in release.videos}
    for video_id in sorted(expected_video_ids):
        item = release_by_video.get(video_id)
        if item is None:
            raise PilotReviewBlockedError(
                f"{video_id}: no dataset release entry exists"
            )
        if (
            not item.source_identity_verified
            or not item.benchmark_use_allowed
            or item.license_or_permission_status != PermissionStatus.VERIFIED
        ):
            raise PilotReviewBlockedError(
                f"{video_id}: source identity or benchmark permission is not verified"
            )
        if (
            item.adjudication_status not in {"approved", "ambiguous"}
            or not item.test_annotation_locked
            or item.adjudicated_annotation_hash is None
            or item.benchmark_ground_truth_sha256 is None
        ):
            raise PilotReviewBlockedError(
                f"{video_id}: adjudicated ground truth is not finalized and locked"
            )


def freeze_pilot_baseline(
    manifest: PilotManifest,
    manifest_path: str | Path,
    *,
    frozen_at: datetime | None = None,
) -> tuple[Path, PilotBaselineMetadata]:
    """Copy a completed run into an immutable, never-overwritten baseline directory."""

    manifest_source = Path(manifest_path).resolve()
    base = manifest_source.parent
    run_directory = _resolve(base, manifest.artifacts.benchmark_run_directory)
    destination = _resolve(base, manifest.artifacts.baseline_directory)
    release_path = _resolve(base, manifest.artifacts.dataset_release)
    report_path = run_directory / "benchmark_report.json"
    if not _protocol_freeze_is_current(manifest):
        raise PilotReviewBlockedError(
            "pilot dataset/ontology/handbook/agreement protocol freeze is not current"
        )
    if not manifest.clips or any(
        not item.real_world_source_confirmed for item in manifest.clips
    ):
        raise PilotReviewBlockedError(
            "baseline requires explicitly confirmed real-world pilot clips"
        )
    if destination.exists():
        raise PilotBaselineExistsError(
            f"pilot baseline already exists and will not be overwritten: {destination}"
        )
    if not run_directory.is_dir() or not report_path.is_file():
        raise FileNotFoundError(f"completed benchmark run not found: {run_directory}")
    if not release_path.is_file():
        raise FileNotFoundError(f"dataset release not found: {release_path}")
    if destination.is_relative_to(run_directory) or run_directory.is_relative_to(
        destination
    ):
        raise ValueError("baseline and current-run directories must not overlap")
    report = _read_json(report_path)
    release = read_json_model(release_path, DatasetRelease)
    video_ids = {item.video_id for item in manifest.clips}
    authorize_posthoc_model_review(release, report, video_ids)
    metadata = _baseline_metadata(
        manifest,
        manifest_source,
        release_path,
        report_path,
        report,
        frozen_at=frozen_at,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for source in run_directory.iterdir():
            target = staging / source.name
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        provenance = staging / "provenance"
        provenance.mkdir()
        shutil.copy2(manifest_source, provenance / "pilot_manifest.json")
        shutil.copy2(release_path, provenance / "dataset_release.json")
        write_json_model(metadata, staging / "baseline_metadata.json")
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination, metadata


def _baseline_metadata(
    manifest: PilotManifest,
    manifest_path: Path,
    release_path: Path,
    report_path: Path,
    report: dict[str, Any],
    *,
    frozen_at: datetime | None,
) -> PilotBaselineMetadata:
    reproducibility = report.get("reproducibility") or {}
    commit = reproducibility.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise PilotReviewBlockedError("benchmark report lacks an exact Git commit")
    if reproducibility.get("git_worktree_dirty") is not False:
        raise PilotReviewBlockedError("baseline requires a clean Git worktree run")
    if reproducibility.get("dataset_identity_status") != "verified":
        raise PilotReviewBlockedError("baseline dataset identity is not verified")
    protocol = EvaluationProtocolIdentity.model_validate(
        reproducibility.get("evaluation_protocol")
    )
    identifiers = reproducibility.get("production_identifiers") or {}
    if set(identifiers) != {item.video_id for item in manifest.clips} or any(
        not item.get("detector_model_identifier") or not item.get("tracker_identifier")
        for item in identifiers.values()
    ):
        raise PilotReviewBlockedError(
            "baseline lacks per-video detector-model or tracker identifiers"
        )
    detector_models = sorted(
        {
            str(item["detector_model_identifier"])
            for item in identifiers.values()
            if item.get("detector_model_identifier")
        }
    )
    trackers = sorted(
        {
            str(item["tracker_identifier"])
            for item in identifiers.values()
            if item.get("tracker_identifier")
        }
    )
    prediction_hashes = reproducibility.get("prediction_cache_hashes_sha256") or {}
    metrics = report.get("overall_metrics") or {}
    return PilotBaselineMetadata(
        pilot_id=manifest.pilot_id,
        frozen_at=frozen_at or datetime.now(timezone.utc),
        system_git_commit=commit,
        resolved_config_hash_sha256=reproducibility["resolved_config_hash_sha256"],
        production_config_hash_sha256=reproducibility["production_config_hash_sha256"],
        detector_model_identifiers=detector_models,
        tracker_identifiers=trackers,
        benchmark_protocol=protocol,
        benchmark_protocol_fingerprint=canonical_sha256(
            protocol.model_dump(mode="json")
        ),
        dataset_fingerprint=reproducibility["dataset_fingerprint"],
        dataset_release_sha256=streaming_file_sha256(release_path),
        pilot_manifest_sha256=streaming_file_sha256(manifest_path),
        benchmark_report_sha256=streaming_file_sha256(report_path),
        prediction_cache_hashes_sha256=prediction_hashes,
        video_ids=sorted(item.video_id for item in manifest.clips),
        true_positives=metrics["true_positives"],
        false_positives=metrics["false_positives"],
        false_negatives=metrics["false_negatives"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        false_positives_per_video_hour=metrics["false_positives_per_video_hour"],
    )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_registry_status(path: Path, blockers: list[PilotBlocker]) -> IntakeRegistry:
    try:
        return read_json_model(path, IntakeRegistry)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(PilotBlocker(code="REGISTRY_INVALID", details=str(exc)))
        return IntakeRegistry()


def _check_protocol_freeze(
    manifest: PilotManifest, blockers: list[PilotBlocker]
) -> None:
    if not _protocol_freeze_is_current(manifest):
        blockers.append(
            PilotBlocker(
                code="PILOT_PROTOCOL_FREEZE_MISMATCH",
                details=(
                    "Pilot identity differs from the current dataset, ontology, "
                    "handbook, or canonical agreement protocol; document migration "
                    "and re-review before mixing annotations."
                ),
            )
        )


def _protocol_freeze_is_current(manifest: PilotManifest) -> bool:
    return (
        manifest.dataset_version == DATASET_VERSION
        and manifest.ontology_version == ONTOLOGY_VERSION
        and manifest.handbook_version == HANDBOOK_VERSION
        and manifest.agreement_protocol == CANONICAL_AGREEMENT_PROTOCOL
        and manifest.agreement_protocol.protocol_version == AGREEMENT_PROTOCOL_VERSION
    )


def _check_registered_clip(
    base: Path,
    selection: PilotClipSelection,
    record: VideoIntakeRecord,
    blockers: list[PilotBlocker],
) -> None:
    if not selection.real_world_source_confirmed:
        blockers.append(
            PilotBlocker(
                code="REAL_WORLD_SOURCE_NOT_CONFIRMED",
                details="Pilot operator has not explicitly confirmed real-world footage.",
                video_id=record.video_id,
            )
        )
    if not record.benchmark_use_allowed:
        blockers.append(
            PilotBlocker(
                code="BENCHMARK_USE_NOT_ALLOWED",
                details="Registry does not permit benchmark use for this source.",
                video_id=record.video_id,
            )
        )
    if record.license_or_permission_status != PermissionStatus.VERIFIED:
        blockers.append(
            PilotBlocker(
                code="PERMISSION_NOT_VERIFIED",
                details="Permission must be verified for official pilot evidence.",
                video_id=record.video_id,
            )
        )
    if (
        not selection.local_video_path
        or not _resolve(base, selection.local_video_path).is_file()
    ):
        blockers.append(
            PilotBlocker(
                code="LOCAL_VIDEO_MISSING",
                details="A readable local source path is required for inference.",
                video_id=record.video_id,
            )
        )
    if (
        not selection.production_config_path
        or not _resolve(base, selection.production_config_path).is_file()
    ):
        blockers.append(
            PilotBlocker(
                code="PRODUCTION_CONFIG_MISSING",
                details="The fixed, pre-review production config path is missing.",
                video_id=record.video_id,
            )
        )


def _load_annotations(
    directory: Path, blockers: list[PilotBlocker]
) -> dict[str, list[DatasetAnnotation]]:
    values: dict[str, list[DatasetAnnotation]] = defaultdict(list)
    if not directory.is_dir():
        return values
    for path in sorted(directory.rglob("*.json")):
        try:
            document = load_annotation(path)
            values[document.video_id].append(document)
        except (ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                PilotBlocker(
                    code="ANNOTATION_ARTIFACT_INVALID",
                    details=f"{path}: {exc}",
                )
            )
    return values


def _load_agreements(
    directory: Path, blockers: list[PilotBlocker]
) -> dict[str, list[AgreementReport]]:
    values: dict[str, list[AgreementReport]] = defaultdict(list)
    if not directory.is_dir():
        return values
    for path in sorted(directory.rglob("*.json")):
        try:
            report = load_agreement(path)
            values[report.video_id].append(report)
        except (ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                PilotBlocker(
                    code="AGREEMENT_ARTIFACT_INVALID", details=f"{path}: {exc}"
                )
            )
    return values


def _load_adjudications(
    directory: Path, blockers: list[PilotBlocker]
) -> dict[str, list[AdjudicationArtifact]]:
    values: dict[str, list[AdjudicationArtifact]] = defaultdict(list)
    if not directory.is_dir():
        return values
    for path in sorted(directory.rglob("*.json")):
        try:
            artifact = load_adjudication(path)
            values[artifact.video_id].append(artifact)
        except (ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                PilotBlocker(
                    code="ADJUDICATION_ARTIFACT_INVALID", details=f"{path}: {exc}"
                )
            )
    return values


def _load_release_status(
    path: Path, blockers: list[PilotBlocker]
) -> DatasetRelease | None:
    if not path.is_file():
        return None
    try:
        return read_json_model(path, DatasetRelease)
    except (ValueError, json.JSONDecodeError) as exc:
        blockers.append(
            PilotBlocker(code="DATASET_RELEASE_INVALID", details=f"{path}: {exc}")
        )
        return None


def _benchmark_exported_ids(
    selected: dict[str, PilotClipSelection],
    adjudicated: set[str],
    release: DatasetRelease | None,
    directory: Path,
    blockers: list[PilotBlocker],
) -> set[str]:
    if release is None:
        return set()
    release_by_id = {item.video_id: item for item in release.videos}
    exported: set[str] = set()
    for video_id in sorted(set(selected) & adjudicated):
        path = directory / f"{video_id}.json"
        if not path.is_file():
            continue
        try:
            document = load_benchmark_annotation(path)
            item = release_by_id.get(video_id)
            if (
                item is not None
                and document.video_id == video_id
                and item.benchmark_ground_truth_sha256 == document_sha256(document)
            ):
                exported.add(video_id)
        except (ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                PilotBlocker(
                    code="BENCHMARK_GROUND_TRUTH_INVALID",
                    details=f"{path}: {exc}",
                    video_id=video_id,
                )
            )
    return exported


def _active_run_directory(base: Path, layout: PilotArtifactLayout) -> Path:
    baseline = _resolve(base, layout.baseline_directory)
    return (
        baseline
        if (baseline / "baseline_metadata.json").is_file()
        else _resolve(base, layout.benchmark_run_directory)
    )


def _inference_complete_ids(
    selected: dict[str, PilotClipSelection],
    records: dict[str, VideoIntakeRecord],
    directory: Path,
    blockers: list[PilotBlocker],
) -> set[str]:
    complete: set[str] = set()
    for video_id in sorted(selected):
        path = directory / f"{video_id}.json"
        if not path.is_file() or video_id not in records:
            continue
        try:
            prediction = read_json_model(path, PredictionDocument)
            record = records[video_id]
            if (
                prediction.video_id == video_id
                and prediction.source_video_sha256 == record.source_video_sha256
                and prediction.source_video_size_bytes == record.source_video_size_bytes
            ):
                complete.add(video_id)
        except (ValueError, json.JSONDecodeError) as exc:
            blockers.append(
                PilotBlocker(
                    code="INFERENCE_ARTIFACT_INVALID",
                    details=f"{path}: {exc}",
                    video_id=video_id,
                )
            )
    return complete


def _load_benchmark_report(
    path: Path, blockers: list[PilotBlocker]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        report = _read_json(path)
        if not isinstance(report.get("per_video_metrics"), dict):
            raise TypeError("benchmark report lacks per_video_metrics")
        return report
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(
            PilotBlocker(code="BENCHMARK_REPORT_INVALID", details=f"{path}: {exc}")
        )
        return None


def _baseline_metadata_valid(
    path: Path, pilot_id: str, blockers: list[PilotBlocker]
) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = read_json_model(path, PilotBaselineMetadata)
        if metadata.pilot_id != pilot_id:
            raise ValueError("baseline pilot_id differs from manifest")
        baseline = path.parent
        if streaming_file_sha256(baseline / "benchmark_report.json") != (
            metadata.benchmark_report_sha256
        ):
            raise ValueError("frozen benchmark report hash changed")
        if streaming_file_sha256(baseline / "provenance" / "dataset_release.json") != (
            metadata.dataset_release_sha256
        ):
            raise ValueError("frozen dataset release hash changed")
        if streaming_file_sha256(baseline / "provenance" / "pilot_manifest.json") != (
            metadata.pilot_manifest_sha256
        ):
            raise ValueError("frozen pilot manifest hash changed")
        for video_id, expected in metadata.prediction_cache_hashes_sha256.items():
            actual = streaming_file_sha256(
                baseline / "predictions" / f"{video_id}.json"
            )
            if actual != expected:
                raise ValueError(f"frozen prediction hash changed for {video_id}")
        return True
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(
            PilotBlocker(code="PILOT_BASELINE_INVALID", details=f"{path}: {exc}")
        )
        return False


def _check_baseline_current_config(
    manifest: PilotManifest,
    base: Path,
    baseline_metadata_path: Path,
    issues: list[PilotIssue],
) -> None:
    """Fail completion when current production configs no longer match baseline 0."""

    production_configs: dict[str, Any] = {}
    for clip in sorted(manifest.clips, key=lambda item: item.video_id):
        if not clip.production_config_path:
            return
        path = _resolve(base, clip.production_config_path)
        if not path.is_file():
            return
        try:
            with path.open("r", encoding="utf-8") as stream:
                production_configs[clip.video_id] = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            issues.append(
                PilotIssue(
                    code="PRODUCTION_CONFIG_INVALID",
                    details=f"{path}: {exc}",
                    video_id=clip.video_id,
                )
            )
            return
    try:
        metadata = read_json_model(baseline_metadata_path, PilotBaselineMetadata)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return
    current_hash = canonical_sha256(dict(sorted(production_configs.items())))
    if current_hash != metadata.production_config_hash_sha256:
        issues.append(
            PilotIssue(
                code="PILOT_BASELINE_STALE",
                details=(
                    "Current production configuration content differs from the "
                    "configuration identity frozen in pilot_baseline_0."
                ),
            )
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _load_review_artifact(
    path: Path,
    loader: Any,
    invalid_code: str,
    blockers: list[PilotBlocker],
) -> Any:
    try:
        return loader(path)
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(PilotBlocker(code=invalid_code, details=f"{path}: {exc}"))
        return None


def derive_pilot_state(
    counts: PilotStageCounts,
    agreement_review: AgreementReviewStatus,
    baseline_frozen: bool,
    failure_review: FailureReviewCoverage,
    decision: ScaleUpDecisionStatus,
    *,
    has_completion_blocker: bool,
) -> PilotState:
    if counts.selected_clips == 0:
        return PilotState.PREPARED
    if counts.registered_clips < counts.selected_clips:
        return PilotState.REGISTERING_DATA
    if counts.double_annotated_clips < counts.selected_clips:
        return PilotState.ANNOTATING
    if agreement_review.required and not agreement_review.complete:
        return PilotState.AGREEMENT_REVIEW_REQUIRED
    if counts.adjudicated_clips < counts.selected_clips:
        return PilotState.ADJUDICATING
    if (
        counts.benchmark_exported_clips < counts.selected_clips
        or counts.inference_complete_clips < counts.selected_clips
        or counts.benchmark_complete_clips < counts.selected_clips
        or not baseline_frozen
    ):
        return PilotState.BENCHMARK_READY
    if not failure_review.complete:
        return PilotState.FAILURE_REVIEW_REQUIRED
    if not agreement_review.required:
        return PilotState.BASELINE_FROZEN
    if not agreement_review.complete:
        return PilotState.AGREEMENT_REVIEW_REQUIRED
    if not decision.valid or decision.decision is None:
        return PilotState.SCALE_UP_DECISION_REQUIRED
    if has_completion_blocker:
        return PilotState.BASELINE_FROZEN
    return {
        ScaleUpDecision.GO: PilotState.COMPLETE_GO,
        ScaleUpDecision.CONDITIONAL_GO: PilotState.COMPLETE_CONDITIONAL_GO,
        ScaleUpDecision.NO_GO: PilotState.COMPLETE_NO_GO,
    }[decision.decision]


def _deduplicate_issues(issues: list[PilotIssue]) -> list[PilotIssue]:
    values = {(item.code, item.video_id, item.details): item for item in issues}
    return [values[key] for key in sorted(values)]

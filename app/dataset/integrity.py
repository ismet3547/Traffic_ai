"""Fail-closed provenance and split integrity checks for dataset releases."""

from __future__ import annotations

from collections import Counter, defaultdict

from app.benchmark.models import DatasetSplit
from app.dataset.agreement_integrity import validate_release_agreements
from app.dataset.io import (
    adjudication_content_hash,
    annotation_content_hash,
    document_sha256,
)
from app.dataset.models import (
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    AdjudicationArtifact,
    AgreementReport,
    ArtifactIntegrityResult,
    DatasetAnnotation,
    DatasetReleaseIntegrityReport,
    IntakeRegistry,
    IntegrityIssue,
    IntegrityReasonCode,
    QualityGateResult,
    SplitAssignmentDocument,
    ValidatedDoubleAnnotation,
    VideoIntakeRecord,
)


class DatasetIntegrityError(ValueError):
    """Release blocker carrying a structured integrity report."""

    def __init__(self, report: DatasetReleaseIntegrityReport) -> None:
        self.report = report
        codes = ", ".join(item.value for item in report.reason_codes)
        super().__init__(f"DATASET_RELEASE_INTEGRITY_FAILED: {codes}")


class SourceIdentityMismatchError(DatasetIntegrityError):
    """An artifact is bound to different source-video bytes."""


class SplitLeakageError(DatasetIntegrityError):
    """Related or byte-identical video content crosses release splits."""


class StaleAdjudicationError(DatasetIntegrityError):
    """An adjudication no longer references current annotation revisions."""


def assess_annotation_source_identity(
    intake_record: VideoIntakeRecord,
    annotation: DatasetAnnotation,
) -> ArtifactIntegrityResult:
    issues: list[IntegrityIssue] = []
    try:
        DatasetAnnotation.model_validate(annotation.model_dump(mode="python"))
    except ValueError as exc:
        issues.append(
            _issue(
                IntegrityReasonCode.ANNOTATION_SCHEMA_INVALID,
                f"annotation failed strict schema revalidation: {exc}",
                intake_record,
            )
        )
    if annotation.video_id != intake_record.video_id:
        issues.append(
            _issue(
                IntegrityReasonCode.ANNOTATION_VIDEO_ID_MISMATCH,
                f"annotation video_id={annotation.video_id!r} does not match registry",
                intake_record,
            )
        )
    if annotation.source_video_sha256 != intake_record.source_video_sha256:
        issues.append(
            _issue(
                IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH,
                "annotation source SHA-256 does not match registry source bytes",
                intake_record,
            )
        )
    if (
        annotation.source_video_size_bytes is not None
        and annotation.source_video_size_bytes != intake_record.source_video_size_bytes
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.ANNOTATION_SOURCE_SIZE_MISMATCH,
                "annotation source byte size does not match registry",
                intake_record,
            )
        )
    if (
        annotation.ontology_version != ONTOLOGY_VERSION
        or annotation.handbook_version != HANDBOOK_VERSION
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.ANNOTATION_PROTOCOL_MISMATCH,
                "annotation ontology/handbook version is not current",
                intake_record,
            )
        )
    if annotation.locked and annotation.annotation_hash != annotation_content_hash(
        annotation
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.ANNOTATION_CONTENT_HASH_MISMATCH,
                "locked annotation content hash is invalid",
                intake_record,
            )
        )
    return _result(issues)


def validate_annotation_source_identity(
    intake_record: VideoIntakeRecord,
    annotation: DatasetAnnotation,
) -> ArtifactIntegrityResult:
    result = assess_annotation_source_identity(intake_record, annotation)
    if not result.valid:
        raise SourceIdentityMismatchError(_single_artifact_report(result))
    return result


def assess_adjudication_source_identity(
    intake_record: VideoIntakeRecord,
    artifact: AdjudicationArtifact,
    current_annotations: list[DatasetAnnotation] | None = None,
) -> ArtifactIntegrityResult:
    issues: list[IntegrityIssue] = []
    try:
        AdjudicationArtifact.model_validate(artifact.model_dump(mode="python"))
    except ValueError as exc:
        issues.append(
            _issue(
                IntegrityReasonCode.ADJUDICATION_SCHEMA_INVALID,
                f"adjudication failed strict schema revalidation: {exc}",
                intake_record,
            )
        )
    if artifact.video_id != intake_record.video_id:
        issues.append(
            _issue(
                IntegrityReasonCode.ADJUDICATION_VIDEO_ID_MISMATCH,
                f"adjudication video_id={artifact.video_id!r} does not match registry",
                intake_record,
            )
        )
    if artifact.source_video_sha256 != intake_record.source_video_sha256:
        issues.append(
            _issue(
                IntegrityReasonCode.ADJUDICATION_SOURCE_VIDEO_MISMATCH,
                "adjudication source SHA-256 does not match registry source bytes",
                intake_record,
            )
        )
    if (
        artifact.source_video_size_bytes is not None
        and artifact.source_video_size_bytes != intake_record.source_video_size_bytes
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.ADJUDICATION_SOURCE_SIZE_MISMATCH,
                "adjudication source byte size does not match registry",
                intake_record,
            )
        )
    if artifact.locked and artifact.adjudication_hash != adjudication_content_hash(
        artifact
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.ADJUDICATION_CONTENT_HASH_MISMATCH,
                "locked adjudication content hash is invalid",
                intake_record,
            )
        )
    for annotation in (artifact.annotation_a, artifact.annotation_b):
        issues.extend(
            assess_annotation_source_identity(intake_record, annotation).issues
        )
    embedded_hashes = {
        artifact.annotation_a.annotator_id: document_sha256(artifact.annotation_a),
        artifact.annotation_b.annotator_id: document_sha256(artifact.annotation_b),
    }
    if (
        embedded_hashes[artifact.annotation_a.annotator_id]
        != artifact.annotation_a_hash
        or embedded_hashes[artifact.annotation_b.annotator_id]
        != artifact.annotation_b_hash
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.ADJUDICATION_ORIGINAL_HASH_MISMATCH,
                "adjudication embedded-original hashes are invalid",
                intake_record,
            )
        )
    if current_annotations is not None:
        current_by_annotator: dict[str, list[DatasetAnnotation]] = defaultdict(list)
        for annotation in current_annotations:
            current_by_annotator[annotation.annotator_id].append(annotation)
        expected = {
            artifact.annotation_a.annotator_id: artifact.annotation_a_hash,
            artifact.annotation_b.annotator_id: artifact.annotation_b_hash,
        }
        stale = set(current_by_annotator) != set(expected) or any(
            len(current_by_annotator[annotator_id]) != 1
            or document_sha256(current_by_annotator[annotator_id][0]) != digest
            for annotator_id, digest in expected.items()
        )
        if stale:
            issues.append(
                _issue(
                    IntegrityReasonCode.ADJUDICATION_STALE_SOURCE_ANNOTATION,
                    "adjudication does not reference the current released annotation revisions",
                    intake_record,
                )
            )
    return _result(issues)


def validate_adjudication_source_identity(
    intake_record: VideoIntakeRecord,
    artifact: AdjudicationArtifact,
    current_annotations: list[DatasetAnnotation] | None = None,
) -> ArtifactIntegrityResult:
    result = assess_adjudication_source_identity(
        intake_record, artifact, current_annotations
    )
    if not result.valid:
        report = _single_artifact_report(result)
        if (
            IntegrityReasonCode.ADJUDICATION_STALE_SOURCE_ANNOTATION
            in result.reason_codes
        ):
            raise StaleAdjudicationError(report)
        raise SourceIdentityMismatchError(report)
    return result


def validate_double_annotation(
    intake_record: VideoIntakeRecord,
    annotations: list[DatasetAnnotation],
) -> ValidatedDoubleAnnotation:
    annotator_counts = Counter(item.annotator_id for item in annotations)
    source_results = [
        assess_annotation_source_identity(intake_record, item) for item in annotations
    ]
    source_identity_valid = all(
        not {
            IntegrityReasonCode.ANNOTATION_VIDEO_ID_MISMATCH,
            IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH,
            IntegrityReasonCode.ANNOTATION_SOURCE_SIZE_MISMATCH,
        }.intersection(result.reason_codes)
        for result in source_results
    )
    protocol_compatible = bool(annotations) and all(
        item.ontology_version == ONTOLOGY_VERSION
        and item.handbook_version == HANDBOOK_VERSION
        for item in annotations
    )
    locked = bool(annotations) and all(
        item.locked and item.annotation_hash == annotation_content_hash(item)
        for item in annotations
    )
    current_revisions = all(count == 1 for count in annotator_counts.values())
    reason_codes = {code for result in source_results for code in result.reason_codes}
    if len(annotator_counts) < 2:
        reason_codes.add(IntegrityReasonCode.ANNOTATOR_COUNT_INSUFFICIENT)
    if annotations and not protocol_compatible:
        reason_codes.add(IntegrityReasonCode.ANNOTATION_PROTOCOL_MISMATCH)
    if annotations and not locked:
        reason_codes.add(IntegrityReasonCode.ANNOTATION_NOT_LOCKED)
    if not current_revisions:
        reason_codes.add(IntegrityReasonCode.ANNOTATOR_REVISION_AMBIGUOUS)
    valid = (
        len(annotator_counts) >= 2
        and source_identity_valid
        and protocol_compatible
        and locked
        and current_revisions
        and not reason_codes
    )
    return ValidatedDoubleAnnotation(
        valid=valid,
        annotator_count=len(annotator_counts),
        source_identity_valid=source_identity_valid,
        protocol_versions_compatible=protocol_compatible,
        locked=locked,
        current_revisions=current_revisions,
        annotation_hashes={
            item.annotator_id: document_sha256(item)
            for item in sorted(annotations, key=lambda value: value.annotator_id)
        },
        reason_codes=sorted(reason_codes, key=lambda item: item.value),
    )


def validate_release_integrity(
    registry: IntakeRegistry,
    split_assignments: SplitAssignmentDocument,
    annotations: dict[str, list[DatasetAnnotation]],
    adjudications: dict[str, AdjudicationArtifact],
    agreements: list[AgreementReport] | None = None,
) -> DatasetReleaseIntegrityReport:
    issues: list[IntegrityIssue] = []
    records = {item.video_id: item for item in registry.videos}
    if not records:
        issues.append(
            IntegrityIssue(
                reason_code=IntegrityReasonCode.DATASET_EMPTY,
                details="official release contains no registered clips",
            )
        )
    for video_id in sorted(set(annotations) - set(records)):
        issues.append(
            IntegrityIssue(
                reason_code=IntegrityReasonCode.UNKNOWN_ANNOTATION_VIDEO,
                details="annotation mapping contains a video absent from registry",
                video_id=video_id,
            )
        )
    for video_id in sorted(set(adjudications) - set(records)):
        issues.append(
            IntegrityIssue(
                reason_code=IntegrityReasonCode.UNKNOWN_ADJUDICATION_VIDEO,
                details="adjudication mapping contains a video absent from registry",
                video_id=video_id,
            )
        )

    assignment_counts = Counter(item.video_id for item in split_assignments.assignments)
    for video_id in sorted(records):
        count = assignment_counts[video_id]
        if count == 0:
            issues.append(
                _issue(
                    IntegrityReasonCode.SPLIT_ASSIGNMENT_MISSING,
                    "registered video has no split assignment",
                    records[video_id],
                )
            )
        elif count > 1:
            issues.append(
                _issue(
                    IntegrityReasonCode.SPLIT_ASSIGNMENT_DUPLICATE,
                    "registered video has multiple split assignments",
                    records[video_id],
                )
            )
    for assignment in split_assignments.assignments:
        record = records.get(assignment.video_id)
        if record is None:
            issues.append(
                IntegrityIssue(
                    reason_code=IntegrityReasonCode.SPLIT_ASSIGNMENT_UNKNOWN_VIDEO,
                    details="split assignment references a video absent from registry",
                    video_id=assignment.video_id,
                    source_group_id=assignment.source_group_id,
                )
            )
        elif assignment.source_group_id != record.source_group_id:
            issues.append(
                _issue(
                    IntegrityReasonCode.SOURCE_GROUP_ID_MISMATCH,
                    "split source_group_id disagrees with registry source of truth",
                    record,
                )
            )

    group_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    hash_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    for assignment in split_assignments.assignments:
        record = records.get(assignment.video_id)
        if record is None:
            continue
        group_splits[record.source_group_id].add(assignment.split)
        hash_splits[record.source_video_sha256].add(assignment.split)
    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            issues.append(
                IntegrityIssue(
                    reason_code=IntegrityReasonCode.SOURCE_GROUP_SPLIT_LEAKAGE,
                    details="registry source group appears in multiple splits: "
                    + ", ".join(sorted(item.value for item in splits)),
                    source_group_id=group_id,
                )
            )
    for digest, splits in sorted(hash_splits.items()):
        if len(splits) > 1:
            video_ids = sorted(
                item.video_id
                for item in registry.videos
                if item.source_video_sha256 == digest
            )
            issues.append(
                IntegrityIssue(
                    reason_code=IntegrityReasonCode.DUPLICATE_VIDEO_CROSS_SPLIT_LEAKAGE,
                    details="byte-identical source video appears across splits: "
                    + ", ".join(video_ids),
                    video_id=video_ids[0] if video_ids else None,
                )
            )

    assignment_by_video = {
        item.video_id: item
        for item in split_assignments.assignments
        if assignment_counts[item.video_id] == 1 and item.video_id in records
    }
    for record in registry.videos:
        if not record.source_identity_verified:
            issues.append(
                _issue(
                    IntegrityReasonCode.SOURCE_VIDEO_IDENTITY_UNVERIFIED,
                    "registry source-video identity is not cryptographically verified",
                    record,
                )
            )
        if not record.benchmark_use_allowed:
            issues.append(
                _issue(
                    IntegrityReasonCode.BENCHMARK_USE_NOT_ALLOWED,
                    "registry does not explicitly permit benchmark use",
                    record,
                )
            )
        documents = annotations.get(record.video_id, [])
        for document in documents:
            issues.extend(assess_annotation_source_identity(record, document).issues)
        duplicate_annotators = {
            annotator
            for annotator, count in Counter(
                item.annotator_id for item in documents
            ).items()
            if count > 1
        }
        if duplicate_annotators:
            issues.append(
                _issue(
                    IntegrityReasonCode.ANNOTATOR_REVISION_AMBIGUOUS,
                    "multiple current revisions exist for annotator(s): "
                    + ", ".join(sorted(duplicate_annotators)),
                    record,
                )
            )
        current_assignment = assignment_by_video.get(record.video_id)
        requires_adjudication = bool(
            current_assignment
            and current_assignment.split in {DatasetSplit.VALIDATION, DatasetSplit.TEST}
        )
        if requires_adjudication:
            double = validate_double_annotation(record, documents)
            for code in double.reason_codes:
                issues.append(
                    _issue(code, "validation/test double annotation is invalid", record)
                )
        artifact = adjudications.get(record.video_id)
        if requires_adjudication and artifact is None:
            issues.append(
                _issue(
                    IntegrityReasonCode.ADJUDICATION_NOT_APPROVED,
                    "validation/test clip has no approved adjudication",
                    record,
                )
            )
        if artifact is not None:
            issues.extend(
                assess_adjudication_source_identity(record, artifact, documents).issues
            )
            if not artifact.approved:
                issues.append(
                    _issue(
                        IntegrityReasonCode.ADJUDICATION_NOT_APPROVED,
                        "adjudication is not approved",
                        record,
                    )
                )
            if (
                current_assignment is not None
                and current_assignment.split == DatasetSplit.TEST
                and not artifact.locked
            ):
                issues.append(
                    _issue(
                        IntegrityReasonCode.ADJUDICATION_NOT_LOCKED,
                        "test adjudication is not locked",
                        record,
                    )
                )

    agreement_validation = validate_release_agreements(
        registry,
        split_assignments,
        annotations,
        adjudications,
        agreements or [],
    )
    issues.extend(agreement_validation.issues)

    issues = _deduplicate_issues(issues)
    gates = _integrity_gates(issues, bool(records))
    reason_codes = sorted(
        {item.reason_code for item in issues}, key=lambda item: item.value
    )
    return DatasetReleaseIntegrityReport(
        passed=not issues,
        gates=gates,
        reason_codes=reason_codes,
        affected_video_ids=sorted(
            {item.video_id for item in issues if item.video_id is not None}
        ),
        affected_source_group_ids=sorted(
            {
                item.source_group_id
                for item in issues
                if item.source_group_id is not None
            }
        ),
        issues=issues,
    )


def raise_for_release_integrity(report: DatasetReleaseIntegrityReport) -> None:
    if report.passed:
        return
    codes = set(report.reason_codes)
    if IntegrityReasonCode.ADJUDICATION_STALE_SOURCE_ANNOTATION in codes:
        raise StaleAdjudicationError(report)
    if codes.intersection(
        {
            IntegrityReasonCode.SOURCE_GROUP_SPLIT_LEAKAGE,
            IntegrityReasonCode.DUPLICATE_VIDEO_CROSS_SPLIT_LEAKAGE,
        }
    ):
        raise SplitLeakageError(report)
    if codes.intersection(
        {
            IntegrityReasonCode.ANNOTATION_VIDEO_ID_MISMATCH,
            IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH,
            IntegrityReasonCode.ANNOTATION_SOURCE_SIZE_MISMATCH,
            IntegrityReasonCode.ADJUDICATION_VIDEO_ID_MISMATCH,
            IntegrityReasonCode.ADJUDICATION_SOURCE_VIDEO_MISMATCH,
            IntegrityReasonCode.ADJUDICATION_SOURCE_SIZE_MISMATCH,
            IntegrityReasonCode.AGREEMENT_SOURCE_VIDEO_MISMATCH,
            IntegrityReasonCode.AGREEMENT_SOURCE_SIZE_MISMATCH,
        }
    ):
        raise SourceIdentityMismatchError(report)
    raise DatasetIntegrityError(report)


def _integrity_gates(
    issues: list[IntegrityIssue], has_records: bool
) -> list[QualityGateResult]:
    codes = {item.reason_code for item in issues}

    def gate(
        name: str, blockers: set[IntegrityReasonCode], details: str
    ) -> QualityGateResult:
        active = sorted(codes.intersection(blockers), key=lambda item: item.value)
        return QualityGateResult(
            gate=name,
            passed=not active,
            details=(
                details
                if not active
                else "blocked by " + ", ".join(item.value for item in active)
            ),
        )

    source_codes = {
        IntegrityReasonCode.ANNOTATION_VIDEO_ID_MISMATCH,
        IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH,
        IntegrityReasonCode.ANNOTATION_SOURCE_SIZE_MISMATCH,
        IntegrityReasonCode.ADJUDICATION_VIDEO_ID_MISMATCH,
        IntegrityReasonCode.ADJUDICATION_SOURCE_VIDEO_MISMATCH,
        IntegrityReasonCode.ADJUDICATION_SOURCE_SIZE_MISMATCH,
        IntegrityReasonCode.ADJUDICATION_ORIGINAL_HASH_MISMATCH,
    }
    split_document_codes = {
        IntegrityReasonCode.SPLIT_ASSIGNMENT_MISSING,
        IntegrityReasonCode.SPLIT_ASSIGNMENT_DUPLICATE,
        IntegrityReasonCode.SPLIT_ASSIGNMENT_UNKNOWN_VIDEO,
        IntegrityReasonCode.SOURCE_GROUP_ID_MISMATCH,
    }
    double_codes = {
        IntegrityReasonCode.ANNOTATOR_COUNT_INSUFFICIENT,
        IntegrityReasonCode.ANNOTATOR_REVISION_AMBIGUOUS,
        IntegrityReasonCode.ANNOTATION_NOT_LOCKED,
        IntegrityReasonCode.ANNOTATION_SCHEMA_INVALID,
        IntegrityReasonCode.ANNOTATION_CONTENT_HASH_MISMATCH,
        IntegrityReasonCode.ANNOTATION_VIDEO_ID_MISMATCH,
        IntegrityReasonCode.ANNOTATION_SOURCE_VIDEO_MISMATCH,
        IntegrityReasonCode.ANNOTATION_SOURCE_SIZE_MISMATCH,
    }
    adjudication_codes = {
        IntegrityReasonCode.ADJUDICATION_STALE_SOURCE_ANNOTATION,
        IntegrityReasonCode.ADJUDICATION_NOT_APPROVED,
        IntegrityReasonCode.ADJUDICATION_ORIGINAL_HASH_MISMATCH,
        IntegrityReasonCode.ADJUDICATION_SCHEMA_INVALID,
        IntegrityReasonCode.ADJUDICATION_CONTENT_HASH_MISMATCH,
    }
    agreement_codes = {
        IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO,
        IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT,
        IntegrityReasonCode.STALE_AGREEMENT_REPORT,
        IntegrityReasonCode.AGREEMENT_SOURCE_VIDEO_MISMATCH,
        IntegrityReasonCode.AGREEMENT_SOURCE_SIZE_MISMATCH,
        IntegrityReasonCode.AGREEMENT_ANNOTATOR_MISMATCH,
        IntegrityReasonCode.AGREEMENT_PROTOCOL_UNSUPPORTED,
        IntegrityReasonCode.AGREEMENT_ONTOLOGY_MISMATCH,
        IntegrityReasonCode.AGREEMENT_HANDBOOK_MISMATCH,
        IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT,
        IntegrityReasonCode.AGREEMENT_CONTENT_HASH_MISMATCH,
        IntegrityReasonCode.AGREEMENT_ADJUDICATION_REVISION_MISMATCH,
    }
    return [
        QualityGateResult(
            gate="dataset_has_registered_clips",
            passed=has_records,
            details=(
                "registered clips present" if has_records else "no clips registered"
            ),
        ),
        gate(
            "source_identity_chain_valid",
            source_codes,
            "registry, annotation, and adjudication source identities agree",
        ),
        gate(
            "split_document_complete",
            split_document_codes,
            "each registered video has exactly one registry-aligned split",
        ),
        gate(
            "split_group_isolation_valid",
            {
                IntegrityReasonCode.SOURCE_GROUP_SPLIT_LEAKAGE,
                IntegrityReasonCode.DUPLICATE_VIDEO_CROSS_SPLIT_LEAKAGE,
            },
            "source groups and byte-identical videos do not cross splits",
        ),
        gate(
            "double_annotation_valid",
            double_codes,
            "validation/test double annotations are locked and identity-valid",
        ),
        gate(
            "adjudication_current",
            adjudication_codes,
            "adjudications are approved and reference current annotation revisions",
        ),
        gate(
            "agreement_integrity",
            agreement_codes,
            "agreement reports match exact source, annotation, and adjudication revisions",
        ),
        gate(
            "agreement_coverage_complete",
            {IntegrityReasonCode.MISSING_CURRENT_AGREEMENT_REPORT},
            "every validation/test clip has exactly one current agreement report",
        ),
        gate(
            "test_annotations_locked",
            {IntegrityReasonCode.ADJUDICATION_NOT_LOCKED},
            "test adjudications are locked",
        ),
        gate(
            "source_video_identity_verified",
            {IntegrityReasonCode.SOURCE_VIDEO_IDENTITY_UNVERIFIED},
            "all source video identities are verified",
        ),
        gate(
            "ontology_compatible",
            {IntegrityReasonCode.ANNOTATION_PROTOCOL_MISMATCH},
            "all released annotations use the current ontology and handbook",
        ),
        gate(
            "benchmark_use_allowed",
            {IntegrityReasonCode.BENCHMARK_USE_NOT_ALLOWED},
            "all registry entries explicitly permit benchmark use",
        ),
        QualityGateResult(
            gate="release_artifacts_validated",
            passed=not issues,
            details=(
                "all release inputs passed explicit integrity validation"
                if not issues
                else f"{len(issues)} integrity blocker(s) detected"
            ),
        ),
    ]


def _result(issues: list[IntegrityIssue]) -> ArtifactIntegrityResult:
    return ArtifactIntegrityResult(
        valid=not issues,
        reason_codes=sorted(
            {item.reason_code for item in issues}, key=lambda item: item.value
        ),
        issues=_deduplicate_issues(issues),
    )


def _single_artifact_report(
    result: ArtifactIntegrityResult,
) -> DatasetReleaseIntegrityReport:
    return DatasetReleaseIntegrityReport(
        passed=False,
        gates=[
            QualityGateResult(
                gate="source_identity_chain_valid",
                passed=False,
                details="blocked by "
                + ", ".join(item.value for item in result.reason_codes),
            )
        ],
        reason_codes=result.reason_codes,
        affected_video_ids=sorted(
            {item.video_id for item in result.issues if item.video_id is not None}
        ),
        affected_source_group_ids=sorted(
            {
                item.source_group_id
                for item in result.issues
                if item.source_group_id is not None
            }
        ),
        issues=result.issues,
    )


def _issue(
    reason_code: IntegrityReasonCode,
    details: str,
    record: VideoIntakeRecord,
) -> IntegrityIssue:
    return IntegrityIssue(
        reason_code=reason_code,
        details=details,
        video_id=record.video_id,
        source_group_id=record.source_group_id,
    )


def _deduplicate_issues(issues: list[IntegrityIssue]) -> list[IntegrityIssue]:
    unique = {
        (
            item.reason_code.value,
            item.video_id,
            item.source_group_id,
            item.details,
        ): item
        for item in issues
    }
    return [unique[key] for key in sorted(unique)]

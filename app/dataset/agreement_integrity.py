"""Provenance validation for agreement reports used by official releases."""

from __future__ import annotations

from collections import Counter, defaultdict

from app.benchmark.models import DatasetSplit
from app.dataset.agreement import agreement_pair_id, compare_independent_annotations
from app.dataset.io import agreement_report_content_hash, document_sha256
from app.dataset.models import (
    AGREEMENT_CONFIG_VERSION,
    AGREEMENT_PROTOCOL_VERSION,
    CANONICAL_AGREEMENT_CONFIG,
    CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    AdjudicationArtifact,
    AgreementCoverage,
    AgreementMode,
    AgreementReport,
    ArtifactIntegrityResult,
    DatasetAnnotation,
    IntakeRegistry,
    IntegrityIssue,
    IntegrityReasonCode,
    SplitAssignmentDocument,
    ValidatedAgreementSet,
    VideoIntakeRecord,
    agreement_config_fingerprint,
)


class AgreementIntegrityError(ValueError):
    def __init__(self, result: ArtifactIntegrityResult) -> None:
        self.result = result
        codes = ", ".join(item.value for item in result.reason_codes)
        super().__init__(f"AGREEMENT_REPORT_INTEGRITY_FAILED: {codes}")


def assess_agreement_report(
    intake_record: VideoIntakeRecord,
    annotation_a: DatasetAnnotation,
    annotation_b: DatasetAnnotation,
    report: AgreementReport,
    *,
    adjudication: AdjudicationArtifact | None = None,
    official_required: bool = True,
) -> ArtifactIntegrityResult:
    issues: list[IntegrityIssue] = []
    try:
        AgreementReport.model_validate(report.model_dump(mode="python"))
    except ValueError as exc:
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT,
                f"agreement report failed strict revalidation: {exc}",
                intake_record,
            )
        )
    if report.video_id != intake_record.video_id:
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO,
                "agreement video_id does not match registry record",
                intake_record,
            )
        )
    if report.source_video_sha256 != intake_record.source_video_sha256:
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_SOURCE_VIDEO_MISMATCH,
                "agreement source SHA-256 does not match registry source bytes",
                intake_record,
            )
        )
    if (
        report.source_video_size_bytes is not None
        and report.source_video_size_bytes != intake_record.source_video_size_bytes
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_SOURCE_SIZE_MISMATCH,
                "agreement source byte size does not match registry",
                intake_record,
            )
        )
    if report.agreement_protocol_version != AGREEMENT_PROTOCOL_VERSION:
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_PROTOCOL_MISMATCH,
                "agreement protocol version is not the current canonical version",
                intake_record,
            )
        )
    actual_config_fingerprint = agreement_config_fingerprint(report.agreement_config)
    if report.agreement_config_fingerprint != actual_config_fingerprint:
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH,
                "agreement config fingerprint does not match its config fields",
                intake_record,
            )
        )
    if official_required:
        if report.agreement_mode != AgreementMode.OFFICIAL:
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_MODE_NOT_OFFICIAL,
                    "validation/test agreement report is not official",
                    intake_record,
                )
            )
        if (
            report.agreement_config_version != AGREEMENT_CONFIG_VERSION
            or report.agreement_config_fingerprint
            != CANONICAL_AGREEMENT_CONFIG_FINGERPRINT
            or report.agreement_config != CANONICAL_AGREEMENT_CONFIG
        ):
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_CONFIG_MISMATCH,
                    "agreement report does not use the canonical release config",
                    intake_record,
                )
            )
    actual_by_annotator = {
        annotation_a.annotator_id: annotation_a,
        annotation_b.annotator_id: annotation_b,
    }
    report_annotators = {report.annotator_a_id, report.annotator_b_id}
    if len(actual_by_annotator) != 2 or report_annotators != set(actual_by_annotator):
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_ANNOTATOR_MISMATCH,
                "agreement annotator pair does not match current annotations",
                intake_record,
            )
        )
    report_hashes = {
        report.annotator_a_id: report.annotation_a_content_sha256,
        report.annotator_b_id: report.annotation_b_content_sha256,
    }
    current_hashes = {
        annotator_id: document_sha256(annotation)
        for annotator_id, annotation in actual_by_annotator.items()
    }
    if report_hashes != current_hashes:
        issues.append(
            _issue(
                IntegrityReasonCode.STALE_AGREEMENT_REPORT,
                "agreement report does not identify current annotation revisions",
                intake_record,
            )
        )
    report_ontology = {
        report.annotator_a_id: report.annotation_a_ontology_version,
        report.annotator_b_id: report.annotation_b_ontology_version,
    }
    current_ontology = {
        item.annotator_id: item.ontology_version
        for item in (annotation_a, annotation_b)
    }
    if report_ontology != current_ontology or any(
        value != ONTOLOGY_VERSION for value in report_ontology.values()
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_ONTOLOGY_MISMATCH,
                "agreement ontology provenance differs from current annotations",
                intake_record,
            )
        )
    report_handbook = {
        report.annotator_a_id: report.annotation_a_handbook_version,
        report.annotator_b_id: report.annotation_b_handbook_version,
    }
    current_handbook = {
        item.annotator_id: item.handbook_version
        for item in (annotation_a, annotation_b)
    }
    if report_handbook != current_handbook or any(
        value != HANDBOOK_VERSION for value in report_handbook.values()
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_HANDBOOK_MISMATCH,
                "agreement handbook provenance differs from current annotations",
                intake_record,
            )
        )
    expected_config = (
        CANONICAL_AGREEMENT_CONFIG if official_required else report.agreement_config
    )
    expected_config_fingerprint = agreement_config_fingerprint(expected_config)
    if report.agreement_id != agreement_pair_id(
        annotation_a,
        annotation_b,
        protocol_version=AGREEMENT_PROTOCOL_VERSION,
        config_fingerprint=expected_config_fingerprint,
    ):
        issues.append(
            _issue(
                IntegrityReasonCode.STALE_AGREEMENT_REPORT,
                "agreement pair identity differs from current annotation pair",
                intake_record,
            )
        )
    if report.agreement_content_sha256 != agreement_report_content_hash(report):
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_CONTENT_HASH_MISMATCH,
                "agreement content hash is invalid",
                intake_record,
            )
        )
    try:
        expected_report = compare_independent_annotations(
            annotation_a,
            annotation_b,
            expected_config,
            mode=(
                AgreementMode.OFFICIAL if official_required else report.agreement_mode
            ),
        )
        metric_fields = {
            "annotation_a_event_count",
            "annotation_b_event_count",
            "agreement_config",
            "matched_event_count",
            "event_detection_agreement",
            "label_agreement",
            "temporal_boundary_agreement",
            "confidence_agreement",
            "mean_temporal_iou",
            "cohen_kappa_matched_labels",
            "disagreement_count",
            "matches",
            "disagreements",
        }
        if report.model_dump(include=metric_fields, mode="json") != (
            expected_report.model_dump(include=metric_fields, mode="json")
        ):
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT,
                    "agreement metrics do not reproduce from the bound annotations",
                    intake_record,
                )
            )
    except ValueError as exc:
        issues.append(
            _issue(
                IntegrityReasonCode.AGREEMENT_INTERNAL_INCOHERENT,
                f"agreement cannot be reproduced from current annotations: {exc}",
                intake_record,
            )
        )
    if adjudication is not None:
        adjudication_hashes = {
            adjudication.annotation_a.annotator_id: adjudication.annotation_a_hash,
            adjudication.annotation_b.annotator_id: adjudication.annotation_b_hash,
        }
        if report_hashes != adjudication_hashes:
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_ADJUDICATION_REVISION_MISMATCH,
                    "agreement and adjudication use different annotation revisions",
                    intake_record,
                )
            )
        if (
            report.agreement_id != adjudication.agreement_report.agreement_id
            or report.agreement_content_sha256
            != adjudication.agreement_report.agreement_content_sha256
        ):
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_ADJUDICATION_REPORT_MISMATCH,
                    "release agreement differs from the report used for adjudication",
                    intake_record,
                )
            )
    return _result(issues)


def validate_agreement_report(
    intake_record: VideoIntakeRecord,
    annotation_a: DatasetAnnotation,
    annotation_b: DatasetAnnotation,
    report: AgreementReport,
    *,
    adjudication: AdjudicationArtifact | None = None,
    official_required: bool = True,
) -> ArtifactIntegrityResult:
    result = assess_agreement_report(
        intake_record,
        annotation_a,
        annotation_b,
        report,
        adjudication=adjudication,
        official_required=official_required,
    )
    if not result.valid:
        raise AgreementIntegrityError(result)
    return result


def validate_supplied_agreements(
    registry: IntakeRegistry,
    annotations: dict[str, list[DatasetAnnotation]],
    reports: list[AgreementReport],
) -> list[AgreementReport]:
    """Validate every supplied report without imposing split coverage policy."""
    records = {item.video_id: item for item in registry.videos}
    report_counts = Counter(item.agreement_id for item in reports)
    duplicate_ids = {
        agreement_id for agreement_id, count in report_counts.items() if count > 1
    }
    issues: list[IntegrityIssue] = []
    _append_mixed_protocol_issue(issues, reports)
    for agreement_id in sorted(duplicate_ids):
        matching = [item for item in reports if item.agreement_id == agreement_id]
        issues.append(
            IntegrityIssue(
                reason_code=IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT,
                details=f"logical agreement report occurs {len(matching)} times",
                video_id=matching[0].video_id,
            )
        )
    validated: list[AgreementReport] = []
    for report in reports:
        record = records.get(report.video_id)
        if record is None:
            issues.append(
                IntegrityIssue(
                    reason_code=IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO,
                    details="agreement report references a video outside the registry",
                    video_id=report.video_id,
                )
            )
            continue
        by_annotator: dict[str, list[DatasetAnnotation]] = defaultdict(list)
        for document in annotations.get(record.video_id, []):
            by_annotator[document.annotator_id].append(document)
        selected = [
            by_annotator[annotator_id][0]
            for annotator_id in (report.annotator_a_id, report.annotator_b_id)
            if len(by_annotator.get(annotator_id, [])) == 1
        ]
        if len(selected) != 2:
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_ANNOTATOR_MISMATCH,
                    "agreement annotator pair cannot be resolved uniquely",
                    record,
                )
            )
            continue
        result = assess_agreement_report(
            record,
            selected[0],
            selected[1],
            report,
            official_required=(report.agreement_mode == AgreementMode.OFFICIAL),
        )
        issues.extend(result.issues)
        if result.valid and report.agreement_id not in duplicate_ids:
            validated.append(report)
    result = _result(issues)
    if not result.valid:
        raise AgreementIntegrityError(result)
    return sorted(validated, key=lambda item: (item.video_id, item.agreement_id))


def validate_release_agreements(
    registry: IntakeRegistry,
    split_assignments: SplitAssignmentDocument,
    annotations: dict[str, list[DatasetAnnotation]],
    adjudications: dict[str, AdjudicationArtifact],
    reports: list[AgreementReport],
) -> ValidatedAgreementSet:
    records = {item.video_id: item for item in registry.videos}
    split_by_video = {
        item.video_id: item.split for item in split_assignments.assignments
    }
    required_ids = {
        video_id
        for video_id, split in split_by_video.items()
        if video_id in records and split in {DatasetSplit.VALIDATION, DatasetSplit.TEST}
    }
    issues: list[IntegrityIssue] = []
    required_reports = [item for item in reports if item.video_id in required_ids]
    _append_mixed_protocol_issue(issues, required_reports)
    report_counts = Counter(item.agreement_id for item in reports)
    duplicate_ids = {
        agreement_id for agreement_id, count in report_counts.items() if count > 1
    }
    for agreement_id in sorted(duplicate_ids):
        matching = [item for item in reports if item.agreement_id == agreement_id]
        issues.append(
            IntegrityIssue(
                reason_code=IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT,
                details=f"logical agreement report occurs {len(matching)} times",
                video_id=matching[0].video_id,
            )
        )
    valid_by_video: dict[str, list[AgreementReport]] = defaultdict(list)
    stale_report_count = 0
    unknown_report_count = 0
    for report in reports:
        record = records.get(report.video_id)
        if record is None:
            unknown_report_count += 1
            issues.append(
                IntegrityIssue(
                    reason_code=IntegrityReasonCode.AGREEMENT_REPORT_UNKNOWN_VIDEO,
                    details="agreement report references a video outside the release",
                    video_id=report.video_id,
                )
            )
            continue
        documents = annotations.get(record.video_id, [])
        by_annotator: dict[str, list[DatasetAnnotation]] = defaultdict(list)
        for document in documents:
            by_annotator[document.annotator_id].append(document)
        selected: list[DatasetAnnotation] = []
        for annotator_id in (report.annotator_a_id, report.annotator_b_id):
            candidates = by_annotator.get(annotator_id, [])
            if len(candidates) == 1:
                selected.append(candidates[0])
        if len(selected) != 2:
            issues.append(
                _issue(
                    IntegrityReasonCode.AGREEMENT_ANNOTATOR_MISMATCH,
                    "agreement annotator pair cannot be resolved uniquely",
                    record,
                )
            )
            continue
        result = assess_agreement_report(
            record,
            selected[0],
            selected[1],
            report,
            adjudication=adjudications.get(record.video_id),
            official_required=(record.video_id in required_ids),
        )
        issues.extend(result.issues)
        if IntegrityReasonCode.STALE_AGREEMENT_REPORT in result.reason_codes:
            stale_report_count += 1
        if result.valid and report.agreement_id not in duplicate_ids:
            valid_by_video[record.video_id].append(report)
    missing_ids = {
        video_id for video_id in required_ids if len(valid_by_video[video_id]) == 0
    }
    for video_id in sorted(missing_ids):
        record = records[video_id]
        issues.append(
            _issue(
                IntegrityReasonCode.MISSING_CURRENT_AGREEMENT_REPORT,
                "validation/test clip lacks one validated current agreement report",
                record,
            )
        )
    for video_id in sorted(required_ids):
        if len(valid_by_video[video_id]) > 1:
            issues.append(
                _issue(
                    IntegrityReasonCode.DUPLICATE_AGREEMENT_REPORT,
                    "validation/test clip has multiple validated agreement reports",
                    records[video_id],
                )
            )
    issues = _deduplicate(issues)
    validated = [
        valid_by_video[video_id][0]
        for video_id in sorted(required_ids)
        if len(valid_by_video[video_id]) == 1
    ]
    required_count = len(required_ids)
    validated_count = len(validated)
    return ValidatedAgreementSet(
        valid=not issues,
        reports=validated,
        coverage=AgreementCoverage(
            required_video_count=required_count,
            validated_report_count=validated_count,
            missing_report_count=len(missing_ids),
            stale_report_count=stale_report_count,
            duplicate_report_count=sum(
                count - 1 for count in report_counts.values() if count > 1
            ),
            unknown_report_count=unknown_report_count,
            coverage_ratio=(
                validated_count / required_count if required_count else 1.0
            ),
        ),
        issues=issues,
    )


def _result(issues: list[IntegrityIssue]) -> ArtifactIntegrityResult:
    unique = _deduplicate(issues)
    return ArtifactIntegrityResult(
        valid=not unique,
        reason_codes=sorted(
            {item.reason_code for item in unique}, key=lambda item: item.value
        ),
        issues=unique,
    )


def _append_mixed_protocol_issue(
    issues: list[IntegrityIssue], reports: list[AgreementReport]
) -> None:
    identities = {
        (
            item.agreement_mode.value,
            item.agreement_protocol_version,
            item.agreement_config_version,
            item.agreement_config_fingerprint,
            agreement_config_fingerprint(item.agreement_config),
        )
        for item in reports
    }
    if len(identities) > 1:
        issues.append(
            IntegrityIssue(
                reason_code=IntegrityReasonCode.MIXED_AGREEMENT_PROTOCOLS,
                details=(
                    "agreement collection contains multiple mode/protocol/config "
                    "identities"
                ),
            )
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


def _deduplicate(issues: list[IntegrityIssue]) -> list[IntegrityIssue]:
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

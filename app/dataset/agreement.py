"""Agreement semantics for independent human event annotations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.benchmark.fingerprints import canonical_sha256
from app.benchmark.matcher import match_events, temporal_iou
from app.benchmark.models import MatchingConfig
from app.dataset.io import agreement_report_content_hash, document_sha256
from app.dataset.models import (
    AGREEMENT_CONFIG_VERSION,
    AGREEMENT_PROTOCOL_VERSION,
    CANONICAL_AGREEMENT_CONFIG,
    CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
    AgreementConfig,
    AgreementMatch,
    AgreementMode,
    AgreementReport,
    AnnotationDisagreement,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    DisagreementType,
    VisibilityQuality,
    agreement_config_fingerprint,
)


@dataclass(slots=True)
class _AgreementTemporalEvent:
    event_id: str
    start_seconds: float
    end_seconds: float
    vehicle_track_hint: str | None
    track_id: str | None


def compare_independent_annotations(
    first: DatasetAnnotation,
    second: DatasetAnnotation,
    config: AgreementConfig | None = None,
    *,
    mode: AgreementMode | None = None,
) -> AgreementReport:
    active_mode = mode or (
        AgreementMode.OFFICIAL
        if config is None or config == CANONICAL_AGREEMENT_CONFIG
        else AgreementMode.EXPLORATORY
    )
    if active_mode == AgreementMode.OFFICIAL:
        if config is not None and config != CANONICAL_AGREEMENT_CONFIG:
            raise ValueError(
                "official agreement requires the canonical agreement config"
            )
        active = CANONICAL_AGREEMENT_CONFIG
    else:
        active = config or CANONICAL_AGREEMENT_CONFIG
    config_fingerprint = agreement_config_fingerprint(active)
    _validate_pair(first, second)
    first_events = [_temporal(event) for event in first.events]
    second_events = [_temporal(event) for event in second.events]
    result = match_events(
        first.video_id,
        first_events,
        second_events,
        MatchingConfig(
            minimum_temporal_iou=active.minimum_temporal_iou,
            start_tolerance_seconds=None,
            require_track_association_if_available=(
                active.require_vehicle_reference_match
            ),
        ),
    )
    first_by_id = {event.event_id: event for event in first.events}
    second_by_id = {event.event_id: event for event in second.events}
    matches: list[AgreementMatch] = []
    disagreements: list[AnnotationDisagreement] = []
    paired_labels: list[tuple[str, str]] = []
    for match in result.matches:
        event_a = first_by_id[match.ground_truth_event_id]
        event_b = second_by_id[match.predicted_event_id]
        start_difference = abs(event_a.start_seconds - event_b.start_seconds)
        end_difference = abs(event_a.end_seconds - event_b.end_seconds)
        label_agrees = event_a.label == event_b.label
        confidence_agrees = event_a.confidence == event_b.confidence
        vehicle_agrees = event_a.vehicle_ref == event_b.vehicle_ref
        matches.append(
            AgreementMatch(
                event_id_a=event_a.event_id,
                event_id_b=event_b.event_id,
                temporal_iou=match.temporal_iou,
                start_difference_seconds=start_difference,
                end_difference_seconds=end_difference,
                label_agrees=label_agrees,
                confidence_agrees=confidence_agrees,
                vehicle_reference_agrees=vehicle_agrees,
            )
        )
        types: list[DisagreementType] = []
        if not label_agrees:
            types.append(DisagreementType.LABEL_DISAGREEMENT)
        if (
            start_difference > active.boundary_tolerance_seconds + 1e-12
            or end_difference > active.boundary_tolerance_seconds + 1e-12
        ):
            types.append(DisagreementType.BOUNDARY_DISAGREEMENT)
        if not confidence_agrees:
            types.append(DisagreementType.CONFIDENCE_DISAGREEMENT)
        if not vehicle_agrees:
            types.append(DisagreementType.VEHICLE_REFERENCE_DISAGREEMENT)
        if _visibility_ambiguous(event_a) or _visibility_ambiguous(event_b):
            types.append(DisagreementType.AMBIGUOUS_VISIBILITY)
        if types:
            disagreements.append(
                _disagreement(
                    disagreements,
                    types,
                    event_a.event_id,
                    event_b.event_id,
                    "matched events differ in "
                    + ", ".join(item.value for item in types),
                )
            )
        paired_labels.append((event_a.label.value, event_b.label.value))

    for event_id in result.unmatched_ground_truth_ids:
        disagreements.append(
            _disagreement(
                disagreements,
                [DisagreementType.EVENT_MISSING_B],
                event_id,
                None,
                "event from annotator A has no eligible event in annotator B",
            )
        )
    for event_id in result.unmatched_prediction_ids:
        disagreements.append(
            _disagreement(
                disagreements,
                [DisagreementType.EVENT_MISSING_A],
                None,
                event_id,
                "event from annotator B has no eligible event in annotator A",
            )
        )
    _add_vehicle_reference_diagnostics(
        disagreements,
        result.unmatched_ground_truth_ids,
        result.unmatched_prediction_ids,
        first_by_id,
        second_by_id,
        active,
    )

    matched = len(matches)
    detection_denominator = len(first.events) + len(second.events)
    event_detection_agreement = (
        2 * matched / detection_denominator if detection_denominator else 1.0
    )
    hash_a = document_sha256(first)
    hash_b = document_sha256(second)
    agreement_id = agreement_pair_id(
        first,
        second,
        config_fingerprint=config_fingerprint,
    )
    report = AgreementReport(
        agreement_mode=active_mode,
        agreement_protocol_version=AGREEMENT_PROTOCOL_VERSION,
        agreement_config_version=AGREEMENT_CONFIG_VERSION,
        agreement_config_fingerprint=config_fingerprint,
        agreement_id=agreement_id,
        agreement_content_sha256="0" * 64,
        video_id=first.video_id,
        source_video_sha256=first.source_video_sha256,
        source_video_size_bytes=(
            first.source_video_size_bytes
            if first.source_video_size_bytes is not None
            else second.source_video_size_bytes
        ),
        annotator_a_id=first.annotator_id,
        annotator_b_id=second.annotator_id,
        annotation_a_content_sha256=hash_a,
        annotation_b_content_sha256=hash_b,
        annotation_a_ontology_version=first.ontology_version,
        annotation_b_ontology_version=second.ontology_version,
        annotation_a_handbook_version=first.handbook_version,
        annotation_b_handbook_version=second.handbook_version,
        annotation_a_event_count=len(first.events),
        annotation_b_event_count=len(second.events),
        agreement_config=active,
        matched_event_count=matched,
        event_detection_agreement=event_detection_agreement,
        label_agreement=(
            sum(item.label_agrees for item in matches) / matched if matched else 0.0
        ),
        temporal_boundary_agreement=(
            sum(
                item.start_difference_seconds
                <= active.boundary_tolerance_seconds + 1e-12
                and item.end_difference_seconds
                <= active.boundary_tolerance_seconds + 1e-12
                for item in matches
            )
            / matched
            if matched
            else 0.0
        ),
        confidence_agreement=(
            sum(item.confidence_agrees for item in matches) / matched
            if matched
            else 0.0
        ),
        mean_temporal_iou=(
            sum(item.temporal_iou for item in matches) / matched if matched else None
        ),
        cohen_kappa_matched_labels=_cohen_kappa(paired_labels),
        disagreement_count=len(disagreements),
        matches=matches,
        disagreements=disagreements,
        caveat=(
            "Cohen's kappa uses only one-to-one temporally matched event pairs. "
            "Unmatched events and boundary disagreements are reported separately; "
            "no arbitrary frame-level negatives are invented. When both annotators "
            "record zero events, event detection agreement is explicitly 1.0 while "
            "matched-event label, boundary, and confidence agreement are 0.0. "
            f"This is an {active_mode.value} report under agreement protocol "
            f"{AGREEMENT_PROTOCOL_VERSION}, config version {AGREEMENT_CONFIG_VERSION}."
        ),
    )
    return report.model_copy(
        update={"agreement_content_sha256": agreement_report_content_hash(report)}
    )


def agreement_pair_id(
    first: DatasetAnnotation,
    second: DatasetAnnotation,
    *,
    protocol_version: str = AGREEMENT_PROTOCOL_VERSION,
    config_fingerprint: str = CANONICAL_AGREEMENT_CONFIG_FINGERPRINT,
) -> str:
    pair_identity = sorted(
        (
            {
                "annotator_id": first.annotator_id,
                "annotation_sha256": document_sha256(first),
            },
            {
                "annotator_id": second.annotator_id,
                "annotation_sha256": document_sha256(second),
            },
        ),
        key=lambda item: (item["annotator_id"], item["annotation_sha256"]),
    )
    return canonical_sha256(
        {
            "video_id": first.video_id,
            "source_video_sha256": first.source_video_sha256,
            "annotation_pair": pair_identity,
            "agreement_protocol_version": protocol_version,
            "agreement_config_fingerprint": config_fingerprint,
        }
    )


def _validate_pair(first: DatasetAnnotation, second: DatasetAnnotation) -> None:
    if first.video_id != second.video_id:
        raise ValueError("agreement requires the same video_id")
    if first.annotator_id == second.annotator_id:
        raise ValueError("agreement requires two distinct anonymous annotator IDs")
    if first.ontology_version != second.ontology_version:
        raise ValueError("ONTOLOGY_VERSION_MISMATCH between annotators")
    if first.handbook_version != second.handbook_version:
        raise ValueError("HANDBOOK_VERSION_MISMATCH between annotators")
    if first.source_video_sha256 != second.source_video_sha256:
        raise ValueError("annotators reference different source video identities")
    if (
        first.source_video_size_bytes is not None
        and second.source_video_size_bytes is not None
        and first.source_video_size_bytes != second.source_video_size_bytes
    ):
        raise ValueError("annotators reference different source video sizes")


def _temporal(event: DatasetEvent) -> _AgreementTemporalEvent:
    return _AgreementTemporalEvent(
        event_id=event.event_id,
        start_seconds=event.start_seconds,
        end_seconds=event.end_seconds,
        vehicle_track_hint=event.vehicle_ref,
        track_id=event.vehicle_ref,
    )


def _visibility_ambiguous(event: DatasetEvent) -> bool:
    return event.label == DatasetLabel.INSUFFICIENT_EVIDENCE or (
        event.evidence.visibility_quality
        in {VisibilityQuality.POOR, VisibilityQuality.UNKNOWN}
    )


def _disagreement(
    existing: list[AnnotationDisagreement],
    types: list[DisagreementType],
    event_id_a: str | None,
    event_id_b: str | None,
    rationale: str,
) -> AnnotationDisagreement:
    return AnnotationDisagreement(
        disagreement_id=f"disagreement_{len(existing) + 1:03d}",
        disagreement_types=list(dict.fromkeys(types)),
        event_id_a=event_id_a,
        event_id_b=event_id_b,
        rationale=rationale,
    )


def _add_vehicle_reference_diagnostics(
    disagreements: list[AnnotationDisagreement],
    unmatched_a: tuple[str, ...],
    unmatched_b: tuple[str, ...],
    first_by_id: dict[str, DatasetEvent],
    second_by_id: dict[str, DatasetEvent],
    config: AgreementConfig,
) -> None:
    for event_id_a in unmatched_a:
        event_a = first_by_id[event_id_a]
        candidates = [
            second_by_id[event_id_b]
            for event_id_b in unmatched_b
            if temporal_iou(_temporal(event_a), _temporal(second_by_id[event_id_b]))
            + 1e-12
            >= config.minimum_temporal_iou
            and event_a.vehicle_ref != second_by_id[event_id_b].vehicle_ref
        ]
        if candidates:
            selected = min(candidates, key=lambda item: item.event_id)
            disagreements.append(
                _disagreement(
                    disagreements,
                    [DisagreementType.VEHICLE_REFERENCE_DISAGREEMENT],
                    event_a.event_id,
                    selected.event_id,
                    "temporally compatible events use different vehicle references",
                )
            )


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs))
        for label in labels
    )
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else None
    return (observed - expected) / (1.0 - expected)

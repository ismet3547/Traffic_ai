"""Basic pairwise inter-annotator agreement for temporal event labels."""

from __future__ import annotations

from collections import Counter

from app.benchmark.matcher import match_events
from app.benchmark.models import AnnotationAgreement, AnnotationDocument, MatchingConfig


def compare_annotations(
    first: AnnotationDocument,
    second: AnnotationDocument,
    matching: MatchingConfig,
) -> AnnotationAgreement:
    if first.video_id != second.video_id:
        raise ValueError("annotation agreement requires the same video_id")
    result = match_events(first.video_id, first.events, second.events, matching)
    first_by_id = {event.event_id: event for event in first.events}
    second_by_id = {event.event_id: event for event in second.events}
    paired_labels = [
        (
            first_by_id[item.ground_truth_event_id].label.value,
            second_by_id[item.predicted_event_id].label.value,
        )
        for item in result.matches
    ]
    matched_count = len(paired_labels)
    label_agreement = (
        sum(left == right for left, right in paired_labels) / matched_count
        if matched_count
        else 0.0
    )
    denominator = max(len(first.events), len(second.events))
    temporal_agreement = matched_count / denominator if denominator else 1.0
    mean_iou = (
        sum(item.temporal_iou for item in result.matches) / matched_count
        if matched_count
        else None
    )
    return AnnotationAgreement(
        video_id=first.video_id,
        annotator_a=first.annotator_id or "annotator_a",
        annotator_b=second.annotator_id or "annotator_b",
        matched_event_count=matched_count,
        event_label_agreement=label_agreement,
        temporal_matching_agreement=temporal_agreement,
        mean_temporal_iou=mean_iou,
        cohen_kappa_matched_labels=_cohen_kappa(paired_labels),
        unmatched_events_a=len(result.unmatched_ground_truth_ids),
        unmatched_events_b=len(result.unmatched_prediction_ids),
        caveat=(
            "Kappa is calculated only on temporally matched events; temporal boundary "
            "disagreement and unmatched events are reported separately."
        ),
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

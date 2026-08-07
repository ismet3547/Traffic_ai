"""Exact event-level metrics and policy-specific suppression measurements."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from app.benchmark.matcher import MatchResult, match_events, temporal_iou
from app.benchmark.models import (
    AnnotationLabel,
    EventMatch,
    GroundTruthEvent,
    MatchingConfig,
    MetricSummary,
    PredictedEvent,
)

POSITIVE_LABEL = AnnotationLabel.UNNECESSARY_LEFT_LANE_OCCUPATION


@dataclass(frozen=True, slots=True)
class EventEvaluation:
    metrics: MetricSummary
    matches: tuple[EventMatch, ...]
    false_positives: tuple[PredictedEvent, ...]
    false_negatives: tuple[GroundTruthEvent, ...]
    ignored_prediction_ids: tuple[str, ...] = ()


def evaluate_events(
    video_id: str,
    ground_truth: list[GroundTruthEvent],
    predictions: list[PredictedEvent],
    matching: MatchingConfig,
    video_duration_seconds: float,
    *,
    ignored_annotations: list[GroundTruthEvent] | None = None,
) -> EventEvaluation:
    positives = [event for event in ground_truth if event.label == POSITIVE_LABEL]
    review_predictions = [
        event
        for event in predictions
        if event.review_status == "pending_human_review"
        and event.event_type == "left_lane_review_candidate"
    ]
    result = match_events(video_id, positives, review_predictions, matching)
    predictions_by_id = {event.event_id: event for event in review_predictions}
    truth_by_id = {event.event_id: event for event in positives}

    ignored_ids: set[str] = set()
    for prediction_id in result.unmatched_prediction_ids:
        prediction = predictions_by_id[prediction_id]
        if any(
            temporal_iou(annotation, prediction) > 0
            for annotation in (ignored_annotations or [])
        ):
            ignored_ids.add(prediction_id)

    false_positives = tuple(
        predictions_by_id[event_id]
        for event_id in result.unmatched_prediction_ids
        if event_id not in ignored_ids
    )
    false_negatives = tuple(
        truth_by_id[event_id] for event_id in result.unmatched_ground_truth_ids
    )
    metrics = summarize_metrics(
        true_positives=len(result.matches),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        prediction_count=len(result.matches) + len(false_positives),
        video_duration_seconds=video_duration_seconds,
        matches=list(result.matches),
    )
    return EventEvaluation(
        metrics=metrics,
        matches=result.matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        ignored_prediction_ids=tuple(sorted(ignored_ids)),
    )


def summarize_metrics(
    *,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    prediction_count: int,
    video_duration_seconds: float,
    matches: list[EventMatch],
) -> MetricSummary:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    video_hours = video_duration_seconds / 3600.0
    start_errors = [match.start_time_error_seconds for match in matches]
    duration_errors = [match.duration_error_seconds for match in matches]
    return MetricSummary(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        evaluated_video_hours=video_hours,
        events_per_hour=(prediction_count / video_hours if video_hours else 0.0),
        false_positives_per_video_hour=(
            false_positives / video_hours if video_hours else 0.0
        ),
        false_negatives_per_video_hour=(
            false_negatives / video_hours if video_hours else 0.0
        ),
        mean_start_time_error_seconds=(mean(start_errors) if start_errors else None),
        median_start_time_error_seconds=(
            median(start_errors) if start_errors else None
        ),
        mean_absolute_start_time_error_seconds=(
            mean(abs(value) for value in start_errors) if start_errors else None
        ),
        mean_duration_error_seconds=(
            mean(duration_errors) if duration_errors else None
        ),
        mean_absolute_duration_error_seconds=(
            mean(abs(value) for value in duration_errors) if duration_errors else None
        ),
    )


def combine_evaluations(evaluations: list[EventEvaluation]) -> MetricSummary:
    matches = [match for evaluation in evaluations for match in evaluation.matches]
    return summarize_metrics(
        true_positives=sum(item.metrics.true_positives for item in evaluations),
        false_positives=sum(item.metrics.false_positives for item in evaluations),
        false_negatives=sum(item.metrics.false_negatives for item in evaluations),
        prediction_count=sum(
            item.metrics.true_positives + item.metrics.false_positives
            for item in evaluations
        ),
        video_duration_seconds=sum(
            item.metrics.evaluated_video_hours * 3600.0 for item in evaluations
        ),
        matches=matches,
    )


def policy_specific_metrics(
    annotations: list[GroundTruthEvent],
    predictions: list[PredictedEvent],
    matching: MatchingConfig,
) -> dict[str, float | int]:
    categories = {
        "overtake": AnnotationLabel.LEGITIMATE_OVERTAKING,
        "congestion": AnnotationLabel.CONGESTION_LEFT_LANE_USE,
        "right_lane_unavailable": AnnotationLabel.RIGHT_LANE_UNAVAILABLE,
        "geometry": AnnotationLabel.GEOMETRY_INVALID,
        "temporary_left_lane": AnnotationLabel.TEMPORARY_LEFT_LANE_USE,
        "insufficient_evidence": AnnotationLabel.INSUFFICIENT_EVIDENCE,
    }
    output: dict[str, float | int] = {}
    for name, label in categories.items():
        controls = [event for event in annotations if event.label == label]
        result: MatchResult = match_events("policy", controls, predictions, matching)
        false_positive_count = len(result.matches)
        control_count = len(controls)
        rate = false_positive_count / control_count if control_count else 0.0
        output[f"{name}_control_events"] = control_count
        output[f"{name}_false_positive_count"] = false_positive_count
        output[f"{name}_false_positive_rate"] = rate
        output[f"{name}_suppression_success_rate"] = (
            (control_count - false_positive_count) / control_count
            if control_count
            else 0.0
        )
    output["geometry_fail_closed_success_rate"] = output[
        "geometry_suppression_success_rate"
    ]
    return output

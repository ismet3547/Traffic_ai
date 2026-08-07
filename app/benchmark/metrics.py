"""Conservative event metrics with explicit, reconciled accounting."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Literal, cast

from app.benchmark.matcher import (
    match_control_events,
    match_events,
    matching_rejection_reason,
    prediction_coverage,
    temporal_iou,
)
from app.benchmark.models import (
    AnnotationLabel,
    AnnotationRole,
    ControlEventConfig,
    DurationValidationResult,
    EvaluationAccounting,
    EventMatch,
    FilteredPredictionDiagnostic,
    GroundTruthEvent,
    IgnoredPredictionDiagnostic,
    IgnoredRegionConfig,
    MatchingConfig,
    MetricSummary,
    PredictedEvent,
    annotation_role,
)

POSITIVE_LABEL = AnnotationLabel.UNNECESSARY_LEFT_LANE_OCCUPATION


@dataclass(frozen=True, slots=True)
class EventEvaluation:
    metrics: MetricSummary
    matches: tuple[EventMatch, ...]
    false_positives: tuple[PredictedEvent, ...]
    false_negatives: tuple[GroundTruthEvent, ...]
    ignored_predictions: tuple[IgnoredPredictionDiagnostic, ...]
    filtered_predictions: tuple[FilteredPredictionDiagnostic, ...]
    considered_predictions: tuple[PredictedEvent, ...]
    ignored_ground_truth: tuple[GroundTruthEvent, ...]
    accounting: EvaluationAccounting
    matching_diagnostics: dict[str, dict[str, float | str | None]]

    @property
    def ignored_prediction_ids(self) -> tuple[str, ...]:
        return tuple(item.prediction_id for item in self.ignored_predictions)


def evaluate_events(
    video_id: str,
    ground_truth: list[GroundTruthEvent],
    predictions: list[PredictedEvent],
    matching: MatchingConfig,
    duration: float | DurationValidationResult,
    *,
    ignored_annotations: list[GroundTruthEvent] | None = None,
    ignored_ground_truth: list[GroundTruthEvent] | None = None,
    ignored_region_config: IgnoredRegionConfig | None = None,
    minimum_prediction_confidence: float = 0.0,
) -> EventEvaluation:
    duration_result = _duration_result(duration)
    positives = [
        event
        for event in ground_truth
        if annotation_role(event.label) == AnnotationRole.POSITIVE
    ]
    review_predictions = [
        event
        for event in predictions
        if event.review_status == "pending_human_review"
        and event.event_type == "left_lane_review_candidate"
    ]
    filtered = tuple(
        FilteredPredictionDiagnostic(
            prediction_id=event.event_id,
            confidence=event.confidence,
            threshold=minimum_prediction_confidence,
        )
        for event in sorted(review_predictions, key=lambda item: item.event_id)
        if event.confidence + 1e-12 < minimum_prediction_confidence
    )
    filtered_ids = {item.prediction_id for item in filtered}
    considered = tuple(
        event
        for event in sorted(review_predictions, key=lambda item: item.event_id)
        if event.event_id not in filtered_ids
    )
    result = match_events(video_id, positives, list(considered), matching)
    predictions_by_id = {event.event_id: event for event in considered}
    truth_by_id = {event.event_id: event for event in positives}

    ignore_config = ignored_region_config or IgnoredRegionConfig()
    ignored_records: list[IgnoredPredictionDiagnostic] = []
    ignored_ids: set[str] = set()
    for prediction_id in result.unmatched_prediction_ids:
        diagnostic = _ignore_diagnostic(
            predictions_by_id[prediction_id],
            ignored_annotations or [],
            ignore_config,
        )
        if diagnostic is not None:
            ignored_records.append(diagnostic)
            ignored_ids.add(prediction_id)

    false_positives = tuple(
        predictions_by_id[event_id]
        for event_id in result.unmatched_prediction_ids
        if event_id not in ignored_ids
    )
    false_negatives = tuple(
        truth_by_id[event_id] for event_id in result.unmatched_ground_truth_ids
    )
    ignored_gt = tuple(
        sorted(ignored_ground_truth or [], key=lambda event: event.event_id)
    )
    accounting = EvaluationAccounting(
        total_prediction_records=len(predictions),
        excluded_non_review_predictions=len(predictions) - len(review_predictions),
        total_predictions_considered=len(considered),
        matched_predictions=len(result.matches),
        false_positive_predictions=len(false_positives),
        ignored_predictions=len(ignored_records),
        filtered_low_confidence_predictions=len(filtered),
        total_positive_gt=len(positives) + len(ignored_gt),
        matched_positive_gt=len(result.matches),
        false_negative_gt=len(false_negatives),
        ignored_ground_truth_events=len(ignored_gt),
    )
    metrics = summarize_metrics(
        true_positives=len(result.matches),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        prediction_count=len(considered),
        duration=duration_result,
        matches=list(result.matches),
    )
    diagnostics = _matching_diagnostics(
        false_positives,
        false_negatives,
        positives,
        list(considered),
        matching,
    )
    evaluation = EventEvaluation(
        metrics=metrics,
        matches=result.matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        ignored_predictions=tuple(
            sorted(ignored_records, key=lambda item: item.prediction_id)
        ),
        filtered_predictions=filtered,
        considered_predictions=considered,
        ignored_ground_truth=ignored_gt,
        accounting=accounting,
        matching_diagnostics=diagnostics,
    )
    _validate_invariants(evaluation)
    return evaluation


def summarize_metrics(
    *,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    prediction_count: int,
    duration: DurationValidationResult,
    matches: list[EventMatch],
) -> MetricSummary:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    video_hours = duration.duration_seconds_used / 3600.0
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
        duration_seconds_used=duration.duration_seconds_used,
        duration_source=duration.duration_source,
        duration_validation_status=duration.duration_validation_status,
        denominator_confidence=duration.denominator_confidence,
    )


def combine_evaluations(evaluations: list[EventEvaluation]) -> MetricSummary:
    matches = [match for evaluation in evaluations for match in evaluation.matches]
    confidences = [
        cast(
            Literal["high", "medium", "low"],
            evaluation.metrics.denominator_confidence,
        )
        for evaluation in evaluations
    ]
    confidence_rank = {"high": 2, "medium": 1, "low": 0, "unavailable": -1}
    if not confidences:
        raise ValueError("cannot combine an empty evaluation collection")
    denominator_confidence = cast(
        Literal["high", "medium", "low"],
        min(confidences, key=lambda value: confidence_rank[value]),
    )
    duration = DurationValidationResult(
        duration_seconds_used=sum(
            item.metrics.duration_seconds_used for item in evaluations
        ),
        duration_source="aggregate_video_durations",
        duration_validation_status="aggregate",
        denominator_confidence=denominator_confidence,
        evidence=[],
    )
    return summarize_metrics(
        true_positives=sum(item.metrics.true_positives for item in evaluations),
        false_positives=sum(item.metrics.false_positives for item in evaluations),
        false_negatives=sum(item.metrics.false_negatives for item in evaluations),
        prediction_count=sum(
            item.accounting.total_predictions_considered for item in evaluations
        ),
        duration=duration,
        matches=matches,
    )


def combine_accounting(evaluations: list[EventEvaluation]) -> EvaluationAccounting:
    fields = EvaluationAccounting.model_fields
    return EvaluationAccounting(
        **{
            name: sum(getattr(item.accounting, name) for item in evaluations)
            for name in fields
        }
    )


def policy_specific_metrics(
    annotations: list[GroundTruthEvent],
    predictions: list[PredictedEvent],
    config: ControlEventConfig,
) -> dict[str, float | int]:
    categories = {
        "overtake": AnnotationLabel.LEGITIMATE_OVERTAKING,
        "congestion": AnnotationLabel.CONGESTION_LEFT_LANE_USE,
        "right_lane_unavailable": AnnotationLabel.RIGHT_LANE_UNAVAILABLE,
        "geometry": AnnotationLabel.GEOMETRY_INVALID,
        "temporary_left_lane": AnnotationLabel.TEMPORARY_LEFT_LANE_USE,
    }
    output: dict[str, float | int] = {}
    for name, label in categories.items():
        controls = [
            event
            for event in annotations
            if event.label == label
            and annotation_role(event.label) == AnnotationRole.NEGATIVE_CONTROL
        ]
        result = match_control_events("policy", controls, predictions, config)
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


def _ignore_diagnostic(
    prediction: PredictedEvent,
    annotations: list[GroundTruthEvent],
    config: IgnoredRegionConfig,
) -> IgnoredPredictionDiagnostic | None:
    if not config.enabled:
        return None
    candidates = []
    for annotation in annotations:
        if (
            annotation.label not in config.allowed_labels
            or annotation_role(annotation.label) != AnnotationRole.IGNORE_REGION
        ):
            continue
        coverage = prediction_coverage(prediction, annotation)
        overlap = temporal_iou(prediction, annotation)
        if (
            coverage + 1e-12 >= config.minimum_prediction_coverage
            and overlap + 1e-12 >= config.minimum_temporal_iou
        ):
            candidates.append((-coverage, -overlap, annotation.event_id, annotation))
    if not candidates:
        return None
    _, _, _, selected = min(candidates, key=lambda item: item[:3])
    return IgnoredPredictionDiagnostic(
        prediction_id=prediction.event_id,
        matched_ignore_annotation_id=selected.event_id,
        prediction_coverage=prediction_coverage(prediction, selected),
        temporal_iou=temporal_iou(prediction, selected),
        ignore_reason=(
            f"label={selected.label.value}; prediction_coverage and temporal_iou "
            "met configured ignore-region thresholds"
        ),
    )


def _matching_diagnostics(
    false_positives: tuple[PredictedEvent, ...],
    false_negatives: tuple[GroundTruthEvent, ...],
    positives: list[GroundTruthEvent],
    predictions: list[PredictedEvent],
    matching: MatchingConfig,
) -> dict[str, dict[str, float | str | None]]:
    output: dict[str, dict[str, float | str | None]] = {}
    for prediction in false_positives:
        if not positives:
            output[prediction.event_id] = {
                "best_candidate_iou": None,
                "best_candidate_prediction_coverage": None,
                "matching_rejection_reason": "NO_POSITIVE_GROUND_TRUTH",
            }
            continue
        truth = max(
            positives,
            key=lambda item: (temporal_iou(item, prediction), item.event_id),
        )
        output[prediction.event_id] = {
            "best_candidate_iou": temporal_iou(truth, prediction),
            "best_candidate_prediction_coverage": prediction_coverage(
                prediction, truth
            ),
            "matching_rejection_reason": (
                matching_rejection_reason(truth, prediction, matching)
                or "VALID_EDGE_NOT_SELECTED_BY_OPTIMAL_ASSIGNMENT"
            ),
        }
    for truth in false_negatives:
        key = f"ground_truth:{truth.event_id}"
        if not predictions:
            output[key] = {
                "best_candidate_iou": None,
                "best_candidate_prediction_coverage": None,
                "matching_rejection_reason": "NO_PREDICTIONS_CONSIDERED",
            }
            continue
        prediction = max(
            predictions,
            key=lambda item: (temporal_iou(truth, item), item.event_id),
        )
        output[key] = {
            "best_candidate_iou": temporal_iou(truth, prediction),
            "best_candidate_prediction_coverage": prediction_coverage(
                prediction, truth
            ),
            "matching_rejection_reason": (
                matching_rejection_reason(truth, prediction, matching)
                or "VALID_EDGE_NOT_SELECTED_BY_OPTIMAL_ASSIGNMENT"
            ),
        }
    return dict(sorted(output.items()))


def _duration_result(
    duration: float | DurationValidationResult,
) -> DurationValidationResult:
    if isinstance(duration, DurationValidationResult):
        return duration
    return DurationValidationResult(
        duration_seconds_used=duration,
        duration_source="explicit_argument",
        duration_validation_status="single_source_unverified",
        denominator_confidence="low",
        evidence=[],
    )


def _validate_invariants(evaluation: EventEvaluation) -> None:
    matched_prediction_ids = [item.predicted_event_id for item in evaluation.matches]
    matched_truth_ids = [item.ground_truth_event_id for item in evaluation.matches]
    fp_ids = [item.event_id for item in evaluation.false_positives]
    ignored_ids = [item.prediction_id for item in evaluation.ignored_predictions]
    considered_ids = [item.event_id for item in evaluation.considered_predictions]
    fn_ids = [item.event_id for item in evaluation.false_negatives]
    if len(matched_prediction_ids) != len(set(matched_prediction_ids)):
        raise RuntimeError(
            "benchmark invariant failed: matched prediction IDs not unique"
        )
    if len(matched_truth_ids) != len(set(matched_truth_ids)):
        raise RuntimeError("benchmark invariant failed: matched GT IDs not unique")
    prediction_sets = [set(matched_prediction_ids), set(fp_ids), set(ignored_ids)]
    if any(
        first & second
        for index, first in enumerate(prediction_sets)
        for second in prediction_sets[index + 1 :]
    ):
        raise RuntimeError("benchmark invariant failed: prediction categories overlap")
    if set(considered_ids) != set().union(*prediction_sets):
        raise RuntimeError(
            "benchmark invariant failed: considered predictions vanished"
        )
    if set(matched_truth_ids) & set(fn_ids):
        raise RuntimeError("benchmark invariant failed: GT is both TP and FN")
    if evaluation.metrics.true_positives != len(evaluation.matches):
        raise RuntimeError("benchmark invariant failed: TP does not equal matches")

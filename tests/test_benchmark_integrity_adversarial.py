from __future__ import annotations

import pytest

from app.benchmark.matcher import match_events, prediction_coverage
from app.benchmark.metrics import evaluate_events, policy_specific_metrics
from app.benchmark.models import (
    ControlEventConfig,
    GroundTruthEvent,
    IgnoredRegionConfig,
    MatchingConfig,
    PredictedEvent,
)


def _event(
    event_id: str,
    start: float,
    end: float,
    label: str = "unnecessary_left_lane_occupation",
    confidence: str = "high",
) -> GroundTruthEvent:
    return GroundTruthEvent(
        event_id=event_id,
        start_seconds=start,
        end_seconds=end,
        label=label,
        confidence=confidence,
    )


def _prediction(
    event_id: str,
    start: float,
    end: float,
    confidence: float = 0.9,
) -> PredictedEvent:
    return PredictedEvent(
        event_id=event_id,
        video_id="video_a",
        start_seconds=start,
        end_seconds=end,
        confidence=confidence,
    )


def _evaluate(
    predictions: list[PredictedEvent],
    *,
    positives: list[GroundTruthEvent] | None = None,
    ignored: list[GroundTruthEvent] | None = None,
    threshold: float = 0.0,
):
    return evaluate_events(
        "video_a",
        positives or [],
        predictions,
        MatchingConfig(minimum_temporal_iou=0.3, start_tolerance_seconds=2.0),
        200.0,
        ignored_annotations=ignored or [],
        ignored_region_config=IgnoredRegionConfig(
            minimum_prediction_coverage=0.5,
            minimum_temporal_iou=0.0,
        ),
        minimum_prediction_confidence=threshold,
    )


def test_tiny_overlap_with_long_ignore_region_remains_false_positive() -> None:
    ignore = _event("ignore", 10.0, 40.0, "insufficient_evidence")
    prediction = _prediction("tiny", 39.9, 45.0)
    assert prediction_coverage(prediction, ignore) == pytest.approx(0.1 / 5.1)
    result = _evaluate([prediction], ignored=[ignore])
    assert result.metrics.false_positives == 1
    assert result.accounting.ignored_predictions == 0


def test_prediction_mostly_contained_in_true_ignore_region_is_audited_as_ignored() -> (
    None
):
    ignore = _event("ignore", 10.0, 40.0, "insufficient_evidence")
    prediction = _prediction("mostly_inside", 12.0, 20.0)
    result = _evaluate([prediction], ignored=[ignore])
    assert result.metrics.false_positives == 0
    assert result.accounting.ignored_predictions == 1
    diagnostic = result.ignored_predictions[0]
    assert diagnostic.matched_ignore_annotation_id == "ignore"
    assert diagnostic.prediction_coverage == 1.0


def test_negative_control_label_cannot_act_as_ignore_region() -> None:
    control = _event("overtake", 10.0, 20.0, "legitimate_overtaking")
    result = _evaluate([_prediction("candidate", 10.0, 20.0)], ignored=[control])
    assert result.metrics.false_positives == 1
    assert result.ignored_predictions == ()


def test_optimal_matching_repairs_documented_greedy_failure_graph() -> None:
    # Valid edges at IoU >= 0.2:
    # g1-p1=.667, g1-p2=.5, g2-p1=.25. Greedy takes g1-p1 and gets
    # one match. Maximum-cardinality matching reassigns to g1-p2 + g2-p1.
    truths = [_event("g1", 0, 10), _event("g2", 8, 18)]
    predictions = [_prediction("p1", 2, 12), _prediction("p2", 0, 5)]
    result = match_events(
        "video_a",
        truths,
        predictions,
        MatchingConfig(minimum_temporal_iou=0.2, start_tolerance_seconds=None),
    )
    assert [
        (item.ground_truth_event_id, item.predicted_event_id) for item in result.matches
    ] == [("g1", "p2"), ("g2", "p1")]
    assert result.unmatched_ground_truth_ids == ()
    assert result.unmatched_prediction_ids == ()


def test_equal_quality_optimal_assignment_is_deterministic_by_ids() -> None:
    truths = [_event("g2", 0, 10), _event("g1", 0, 10)]
    predictions = [_prediction("p2", 0, 10), _prediction("p1", 0, 10)]
    first = match_events("video_a", truths, predictions, MatchingConfig())
    second = match_events(
        "video_a", list(reversed(truths)), list(reversed(predictions)), MatchingConfig()
    )
    assert first.matches == second.matches


def test_matched_prediction_cannot_also_be_ignored() -> None:
    positive = _event("positive", 10.0, 20.0)
    ignore = _event("ignore", 10.0, 20.0, "insufficient_evidence")
    result = _evaluate(
        [_prediction("candidate", 10.0, 20.0)],
        positives=[positive],
        ignored=[ignore],
    )
    assert result.accounting.matched_predictions == 1
    assert result.accounting.ignored_predictions == 0


def test_ignored_prediction_cannot_also_be_false_positive() -> None:
    ignore = _event("ignore", 10.0, 20.0, "insufficient_evidence")
    result = _evaluate([_prediction("candidate", 10.0, 20.0)], ignored=[ignore])
    assert result.accounting.ignored_predictions == 1
    assert result.accounting.false_positive_predictions == 0


def test_one_frame_control_overlap_is_not_a_suppression_failure() -> None:
    control = _event("overtake", 10.0, 20.0, "legitimate_overtaking")
    prediction = _prediction("candidate", 19.99, 30.0)
    metrics = policy_specific_metrics(
        [control],
        [prediction],
        ControlEventConfig(
            minimum_prediction_coverage=0.5,
            minimum_temporal_iou=0.2,
        ),
    )
    assert metrics["overtake_false_positive_count"] == 0


def test_low_confidence_prediction_is_filtered_and_reconciled() -> None:
    result = _evaluate([_prediction("low", 10.0, 20.0, confidence=0.2)], threshold=0.5)
    assert result.accounting.total_prediction_records == 1
    assert result.accounting.filtered_low_confidence_predictions == 1
    assert result.accounting.total_predictions_considered == 0
    assert result.metrics.false_positives == 0

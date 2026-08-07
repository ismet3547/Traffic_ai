from __future__ import annotations

import pytest

from app.benchmark.matcher import match_events, temporal_iou
from app.benchmark.metrics import evaluate_events
from app.benchmark.models import (
    GroundTruthEvent,
    MatchingConfig,
    PredictedEvent,
)


def _truth(
    event_id: str = "gt_1", start: float = 10.0, end: float = 20.0
) -> GroundTruthEvent:
    return GroundTruthEvent(
        event_id=event_id,
        start_seconds=start,
        end_seconds=end,
        label="unnecessary_left_lane_occupation",
        confidence="high",
    )


def _prediction(
    event_id: str = "pred_1", start: float = 10.0, end: float = 20.0
) -> PredictedEvent:
    return PredictedEvent(
        event_id=event_id,
        video_id="video_a",
        start_seconds=start,
        end_seconds=end,
        confidence=0.9,
    )


def _evaluate(
    truth: list[GroundTruthEvent],
    predictions: list[PredictedEvent],
    duration: float = 3600.0,
):
    return evaluate_events(
        "video_a",
        truth,
        predictions,
        MatchingConfig(minimum_temporal_iou=0.3, start_tolerance_seconds=2.0),
        duration,
    )


def test_perfect_prediction_has_exact_unit_metrics() -> None:
    metrics = _evaluate([_truth()], [_prediction()]).metrics
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_no_predictions_with_ground_truth_has_zero_recall() -> None:
    metrics = _evaluate([_truth()], []).metrics
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0


def test_predictions_without_ground_truth_are_false_positives() -> None:
    metrics = _evaluate([], [_prediction()]).metrics
    assert metrics.false_positives == 1
    assert metrics.true_positives == 0
    assert metrics.precision == 0.0


def test_duplicate_predictions_cannot_match_the_same_ground_truth() -> None:
    result = _evaluate(
        [_truth()],
        [_prediction("pred_a"), _prediction("pred_b", 10.1, 19.9)],
    )
    assert result.metrics.true_positives == 1
    assert result.metrics.false_positives == 1
    assert len(result.matches) == 1


def test_temporal_iou_threshold_is_inclusive_at_boundary() -> None:
    truth = _truth(start=0.0, end=10.0)
    prediction = _prediction(start=5.0, end=15.0)
    assert temporal_iou(truth, prediction) == pytest.approx(1 / 3)
    result = match_events(
        "video_a",
        [truth],
        [prediction],
        MatchingConfig(
            minimum_temporal_iou=1 / 3,
            start_tolerance_seconds=None,
        ),
    )
    assert len(result.matches) == 1


def test_start_tolerance_can_reject_an_iou_eligible_pair() -> None:
    result = match_events(
        "video_a",
        [_truth(start=0.0, end=20.0)],
        [_prediction(start=5.0, end=20.0)],
        MatchingConfig(minimum_temporal_iou=0.3, start_tolerance_seconds=2.0),
    )
    assert result.matches == ()


def test_track_association_is_optional_and_deterministic() -> None:
    truth = _truth().model_copy(update={"vehicle_track_hint": 7})
    prediction = _prediction().model_copy(update={"track_id": 8})
    strict = match_events(
        "video_a",
        [truth],
        [prediction],
        MatchingConfig(require_track_association_if_available=True),
    )
    permissive = match_events("video_a", [truth], [prediction], MatchingConfig())
    assert strict.matches == ()
    assert len(permissive.matches) == 1


def test_fp_and_fn_per_hour_use_video_duration_denominator() -> None:
    metrics = _evaluate(
        [_truth("gt_unmatched", 100.0, 110.0)],
        [_prediction("pred_unmatched", 10.0, 20.0)],
        duration=1800.0,
    ).metrics
    assert metrics.false_positives_per_video_hour == 2.0
    assert metrics.false_negatives_per_video_hour == 2.0
    assert metrics.events_per_hour == 2.0


def test_start_and_duration_errors_are_reported() -> None:
    metrics = _evaluate(
        [_truth(start=10.0, end=20.0)],
        [_prediction(start=11.0, end=23.0)],
    ).metrics
    assert metrics.mean_start_time_error_seconds == 1.0
    assert metrics.median_start_time_error_seconds == 1.0
    assert metrics.mean_absolute_start_time_error_seconds == 1.0
    assert metrics.mean_duration_error_seconds == 2.0
    assert metrics.mean_absolute_duration_error_seconds == 2.0

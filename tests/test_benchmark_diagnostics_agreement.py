from __future__ import annotations

import pytest

from app.benchmark.agreement import compare_annotations
from app.benchmark.diagnostics import diagnose_false_positive
from app.benchmark.models import (
    AnnotationDocument,
    GroundTruthEvent,
    MatchingConfig,
    PredictedEvent,
)


def _event(event_id: str, label: str, start: float = 1.0, end: float = 3.0):
    return GroundTruthEvent(
        event_id=event_id,
        start_seconds=start,
        end_seconds=end,
        label=label,
        confidence="high",
    )


def _document(annotator: str, events: list[GroundTruthEvent]) -> AnnotationDocument:
    return AnnotationDocument(
        video_id="video_a",
        source_file="video_a.mp4",
        fps=30,
        video_duration_seconds=10,
        annotator_id=annotator,
        events=events,
    )


def test_overtaking_false_positive_receives_transparent_hint() -> None:
    prediction = PredictedEvent(
        event_id="pred",
        video_id="video_a",
        start_seconds=1,
        end_seconds=3,
        confidence=0.8,
        overtaking_status="none",
    )
    failure = diagnose_false_positive(
        prediction, [_event("gt", "legitimate_overtaking")], 1
    )
    assert failure.suspected_failure_category == "OVERTAKING_LOGIC_ERROR"
    assert "runtime overtake status was none" in failure.diagnostic_rationale[0]


def test_degraded_geometry_takes_precedence_in_diagnostic_hint() -> None:
    prediction = PredictedEvent(
        event_id="pred",
        video_id="video_a",
        start_seconds=1,
        end_seconds=3,
        confidence=0.8,
        geometry_status="degraded",
    )
    failure = diagnose_false_positive(prediction, [], 1)
    assert failure.suspected_failure_category == "GEOMETRY_INTEGRITY_ERROR"


def test_two_annotator_label_and_temporal_agreement() -> None:
    first = _document(
        "a",
        [
            _event("a1", "unnecessary_left_lane_occupation"),
            _event("a2", "legitimate_overtaking", 5, 7),
        ],
    )
    second = _document(
        "b",
        [
            _event("b1", "unnecessary_left_lane_occupation", 1.1, 3.1),
            _event("b2", "temporary_left_lane_use", 5, 7),
        ],
    )
    result = compare_annotations(first, second, MatchingConfig())
    assert result.matched_event_count == 2
    assert result.temporal_matching_agreement == 1.0
    assert result.event_label_agreement == 0.5
    assert result.cohen_kappa_matched_labels == pytest.approx(1 / 3)

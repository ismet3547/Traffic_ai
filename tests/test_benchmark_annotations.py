from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.benchmark.annotations import load_annotation
from app.benchmark.metrics import evaluate_events
from app.benchmark.models import (
    AnnotationDocument,
    GroundTruthEvent,
    MatchingConfig,
    PredictedEvent,
)


def _event(**updates):
    values = {
        "event_id": "gt_1",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "label": "unnecessary_left_lane_occupation",
        "confidence": "high",
    }
    values.update(updates)
    return values


def _document(events: list[dict]) -> dict:
    return {
        "schema_version": "1.0",
        "video_id": "video_a",
        "source_file": "video_a.mp4",
        "fps": 30,
        "video_duration_seconds": 10.0,
        "annotator_id": "annotator_a",
        "events": events,
    }


def test_invalid_annotation_interval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="end_seconds"):
        AnnotationDocument.model_validate(
            _document([_event(start_seconds=2.0, end_seconds=2.0)])
        )


def test_unsupported_annotation_label_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported_label"):
        AnnotationDocument.model_validate(
            _document([_event(label="unsupported_label")])
        )


def test_duplicate_annotation_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        AnnotationDocument.model_validate(_document([_event(), _event()]))


def test_annotation_beyond_declared_video_duration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exceed"):
        AnnotationDocument.model_validate(
            _document([_event(start_seconds=9.0, end_seconds=11.0)])
        )


def test_json_annotation_loader_accepts_versioned_document(tmp_path) -> None:
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(_document([_event()])), encoding="utf-8")
    document = load_annotation(path)
    assert document.schema_version == "1.0"
    assert document.annotator_id == "annotator_a"


def test_low_confidence_positive_does_not_become_generic_ignore_region() -> None:
    high = GroundTruthEvent.model_validate(
        _event(event_id="high", start_seconds=1, end_seconds=2)
    )
    low = GroundTruthEvent.model_validate(
        _event(event_id="low", start_seconds=5, end_seconds=7, confidence="low")
    )
    prediction = PredictedEvent(
        event_id="pred_low",
        video_id="video_a",
        start_seconds=5,
        end_seconds=7,
        confidence=0.8,
    )
    result = evaluate_events(
        "video_a",
        [high],
        [prediction],
        MatchingConfig(),
        10.0,
        ignored_annotations=[low],
        ignored_ground_truth=[low],
    )
    assert result.metrics.false_positives == 1
    assert result.ignored_prediction_ids == ()
    assert result.accounting.ignored_ground_truth_events == 1


def test_explicit_annotation_role_must_match_label() -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        GroundTruthEvent.model_validate(
            _event(label="legitimate_overtaking", role="ignore_region")
        )
